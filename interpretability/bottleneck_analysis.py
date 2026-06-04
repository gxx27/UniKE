#!/usr/bin/env python3
"""
Analyzes the conditioning-pathway bottleneck between the edited LLM and the image generator.
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    MODEL_CONFIGS, get_pkl_path, load_fresh_model, load_edited_model,
    load_dataset, get_case_info, get_conditioning, get_llm, get_module,
    save_results, print_table,
)


def get_projection_matrix_ovis(model) -> torch.Tensor:
    """Extract the frozen Linear(4096, 1536) from Ovis's Yak SingleTokenRefiner.

    Path: visual_generator.backbone.txt_in.input_embedder
    This is the primary bottleneck: projects 4096-dim conditioning into 1536-dim DiT space.
    """
    vg = model.visual_generator
    backbone = vg.backbone
    txt_in = backbone.txt_in  # SingleTokenRefiner
    W = txt_in.input_embedder.weight.float()  # (1536, 4096)
    return W


def compute_svd_projection_fractions(W: torch.Tensor, delta: torch.Tensor, k_values: list) -> dict:
    """
    Compute what fraction of delta projects into top-k singular vectors of W.

    Args:
        W: projection matrix (out_dim, in_dim), e.g. (1536, 4096)
        delta: perturbation vector(s) (seq, in_dim) or (in_dim,)
        k_values: list of k values to test

    Returns:
        dict with fraction of delta captured at each k
    """
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # Vh: (min(out,in), in_dim) - right singular vectors

    if delta.dim() > 1:
        delta_vec = delta.mean(dim=0)
    else:
        delta_vec = delta

    delta_norm = torch.norm(delta_vec).item()
    if delta_norm < 1e-10:
        return {k: 0.0 for k in k_values}

    fractions = {}
    for k in k_values:
        k_actual = min(k, Vh.shape[0])
        Vh_topk = Vh[:k_actual]  # (k, in_dim)
        proj = Vh_topk @ delta_vec  # (k,)
        proj_norm = torch.norm(proj).item()
        fractions[k] = (proj_norm / delta_norm) ** 2  # fraction of variance
    return fractions


@torch.inference_mode()
def analyze_case_ovis(fresh_model, edited_model, tok, prompt: str, model_name: str,
                      W_proj: torch.Tensor, k_values: list, device: str = "cuda"):
    """Analyze a single case for Ovis (has projection bottleneck)."""
    cond_fresh = get_conditioning(fresh_model, tok, prompt, model_name, device).float().cpu()
    cond_edited = get_conditioning(edited_model, tok, prompt, model_name, device).float().cpu()

    delta = cond_edited - cond_fresh  # (seq, 4096)
    delta_norm = torch.norm(delta).item()

    # SVD projection analysis: how much of the perturbation survives the frozen Linear?
    fractions = compute_svd_projection_fractions(W_proj.cpu(), delta, k_values)

    # What the generator actually receives after the Linear projection
    delta_mean = delta.mean(dim=0)  # (4096,)
    projected = W_proj.cpu() @ delta_mean  # (1536,)
    projected_norm = torch.norm(projected).item()
    input_norm = torch.norm(delta_mean).item()

    return {
        "delta_norm": delta_norm,
        "delta_mean_norm": input_norm,
        "projected_norm": projected_norm,
        "bottleneck_ratio": projected_norm / (input_norm + 1e-10),
        "svd_fractions": fractions,
    }


@torch.inference_mode()
def analyze_case_direct(fresh_model, edited_model, tok, prompt: str, model_name: str, device: str = "cuda"):
    """Analyze a single case for BLIP3o/OmniGen2 (no projection bottleneck).

    These models feed LLM hidden states directly to the DiT with no dimensionality reduction.
    The edit perturbation passes through to the conditioning without loss.
    """
    cond_fresh = get_conditioning(fresh_model, tok, prompt, model_name, device).float().cpu()
    cond_edited = get_conditioning(edited_model, tok, prompt, model_name, device).float().cpu()

    delta = cond_edited - cond_fresh
    delta_norm = torch.norm(delta).item()

    # Per-token perturbation magnitude
    per_token_norms = torch.norm(delta, dim=-1)  # (n_tokens,)
    max_token_norm = per_token_norms.max().item()
    mean_token_norm = per_token_norms.mean().item()

    # Relative perturbation (normalized by fresh conditioning magnitude)
    fresh_norm = torch.norm(cond_fresh).item()
    relative_perturbation = delta_norm / (fresh_norm + 1e-10)

    # Cosine similarity between fresh and edited conditioning (averaged over tokens)
    cos_sim = torch.nn.functional.cosine_similarity(
        cond_fresh, cond_edited, dim=-1
    ).mean().item()

    return {
        "delta_norm": delta_norm,
        "relative_perturbation": relative_perturbation,
        "cosine_similarity": cos_sim,
        "mean_token_perturbation": mean_token_norm,
        "max_token_perturbation": max_token_norm,
        "n_conditioning_tokens": delta.shape[0],
        "bottleneck_ratio": 1.0,  # no bottleneck — all signal passes through
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--method", type=str, required=True, choices=["AlphaEdit", "MEMIT", "PMET"])
    parser.add_argument("--num_cases", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_name]
    pkl_path = get_pkl_path(args.model_name, args.method)
    assert pkl_path.exists(), f"PKL not found: {pkl_path}"

    dataset = load_dataset(limit=args.num_cases)

    print(f"Loading fresh model: {args.model_name}")
    fresh_model, tok = load_fresh_model(args.model_name, device=args.device)
    fresh_model.eval()

    print(f"Loading edited model: {pkl_path}")
    edited_model, tok_e, _ = load_edited_model(str(pkl_path), device=args.device)
    edited_model.eval()

    is_ovis = (cfg["model_type"] == "ovis")

    if is_ovis:
        # Ovis: analyze the frozen projection bottleneck via SVD
        k_values = [1, 5, 10, 50, 100, 200, 500, 1000, 1536]

        print("Extracting projection matrix (frozen Linear 4096->1536)...")
        W_proj = get_projection_matrix_ovis(fresh_model)
        print(f"  Projection matrix shape: {W_proj.shape}")

        print("  Computing SVD...")
        U, S, Vh = torch.linalg.svd(W_proj.float(), full_matrices=False)
        print(f"  Top-5 singular values: {S[:5].tolist()}")
        print(f"  Condition number (S[0]/S[-1]): {S[0].item() / (S[-1].item() + 1e-10):.2f}")

        all_results = []
        for i, case in enumerate(tqdm(dataset, desc=f"Exp2 {args.model_name} {args.method}")):
            info = get_case_info(case)
            result = analyze_case_ovis(
                fresh_model, edited_model, tok, info["image_prompt"],
                args.model_name, W_proj, k_values, args.device
            )
            result["case_idx"] = i
            result["category"] = info["category"]
            all_results.append(result)

        # Aggregate for Ovis
        avg_fractions = {k: float(np.mean([r["svd_fractions"].get(k, 0) for r in all_results])) for k in k_values}
        avg_delta_norm = float(np.mean([r["delta_norm"] for r in all_results]))
        avg_projected_norm = float(np.mean([r["projected_norm"] for r in all_results]))
        avg_bottleneck_ratio = float(np.mean([r["bottleneck_ratio"] for r in all_results]))

        fraction_at_full_rank = avg_fractions[k_values[-1]]
        summary = {
            "model_name": args.model_name,
            "method": args.method,
            "model_type": "ovis",
            "has_bottleneck": True,
            "bottleneck_description": "Frozen Linear(4096->1536) in SingleTokenRefiner",
            "num_cases": len(all_results),
            "projection_matrix_shape": list(W_proj.shape),
            "condition_number": S[0].item() / (S[-1].item() + 1e-10),
            "top_5_singular_values": S[:5].tolist(),
            "avg_delta_norm": avg_delta_norm,
            "avg_projected_norm": avg_projected_norm,
            "signal_retained_fraction": fraction_at_full_rank,
            "signal_lost_fraction": 1.0 - fraction_at_full_rank,
            "avg_svd_fractions": avg_fractions,
        }

        # Print table
        headers = ["Top-k", "Fraction Captured"]
        rows = [[str(k), avg_fractions[k]] for k in k_values]
        print_table(headers, rows, f"SVD Projection Bottleneck: {args.model_name} / {args.method}")
        print(f"Avg perturbation norm (full 4096-dim): {avg_delta_norm:.4f}")
        print(f"Avg projected norm (after Linear): {avg_projected_norm:.4f}")
        fraction_at_full_rank = avg_fractions[k_values[-1]]
        print(f"Fraction of perturbation variance in projection column space (k={k_values[-1]}): {fraction_at_full_rank:.4f}")
        print(f"  -> {(1 - fraction_at_full_rank)*100:.1f}% of edit signal is ORTHOGONAL to projection and LOST")

    else:
        # BLIP3o and OmniGen2: no projection bottleneck, measure direct perturbation
        pathway_desc = {
            "blip3o": "Latent queries pass through same edited MLP layers -> no projection",
            "omnigen2": "LLM hidden states fed directly to DiT -> no projection",
        }[cfg["model_type"]]

        print(f"Model has NO projection bottleneck: {pathway_desc}")

        all_results = []
        for i, case in enumerate(tqdm(dataset, desc=f"Exp2 {args.model_name} {args.method}")):
            info = get_case_info(case)
            result = analyze_case_direct(
                fresh_model, edited_model, tok, info["image_prompt"],
                args.model_name, args.device
            )
            result["case_idx"] = i
            result["category"] = info["category"]
            all_results.append(result)

        avg_delta_norm = float(np.mean([r["delta_norm"] for r in all_results]))
        avg_relative = float(np.mean([r["relative_perturbation"] for r in all_results]))
        avg_cos = float(np.mean([r["cosine_similarity"] for r in all_results]))
        avg_mean_token = float(np.mean([r["mean_token_perturbation"] for r in all_results]))
        avg_n_tokens = float(np.mean([r["n_conditioning_tokens"] for r in all_results]))

        summary = {
            "model_name": args.model_name,
            "method": args.method,
            "model_type": cfg["model_type"],
            "has_bottleneck": False,
            "bottleneck_description": pathway_desc,
            "num_cases": len(all_results),
            "avg_delta_norm": avg_delta_norm,
            "avg_relative_perturbation": avg_relative,
            "avg_cosine_similarity": avg_cos,
            "avg_mean_token_perturbation": avg_mean_token,
            "avg_n_conditioning_tokens": avg_n_tokens,
            "avg_bottleneck_ratio": 1.0,
        }

        # Print table
        headers = ["Metric", "Value"]
        rows = [
            ["Total perturbation L2", avg_delta_norm],
            ["Relative perturbation", avg_relative],
            ["Cosine similarity", avg_cos],
            ["Mean per-token perturbation", avg_mean_token],
            ["N conditioning tokens", avg_n_tokens],
            ["Bottleneck ratio", 1.0],
        ]
        print_table(headers, rows, f"Direct Pathway (No Bottleneck): {args.model_name} / {args.method}")
        print(f"  -> 100% of edit signal reaches the image generator (no projection loss)")

    save_results({"summary": summary, "per_case": all_results}, "exp2_conditioning_projection", args.model_name, args.method)

    del fresh_model, edited_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
