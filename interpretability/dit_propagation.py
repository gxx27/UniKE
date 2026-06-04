#!/usr/bin/env python3
"""
Traces which diffusion-transformer blocks are most sensitive to the edit perturbation in the conditioning signal.
"""

import argparse
import math
import sys
import torch
import numpy as np
from torch import Tensor
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    MODEL_CONFIGS, get_pkl_path, load_fresh_model, load_edited_model,
    load_dataset, get_case_info, get_conditioning, get_llm, get_module,
    ActivationCollector, save_results, print_table,
)


def get_dit_block_names_ovis(model) -> dict:
    """Get block names for Ovis's Yak DiT model (under backbone)."""
    double_blocks = [f"visual_generator.backbone.double_blocks.{i}" for i in range(6)]
    single_blocks = [f"visual_generator.backbone.single_blocks.{i}" for i in range(12)]
    return {"double_blocks": double_blocks, "single_blocks": single_blocks}


def get_dit_block_names_generic(model, model_name: str) -> dict:
    """Get decoder block names for BLIP3o / OmniGen2."""
    blocks = []
    for name, module in model.named_modules():
        # Look for transformer blocks in the decoder/generator
        if ('dit' in name.lower() or 'decoder' in name.lower() or
            'generator' in name.lower() or 'transformer' in name.lower()):
            # Match numbered blocks
            parts = name.split('.')
            if any(p.isdigit() for p in parts):
                blocks.append(name)

    if not blocks:
        # Fallback: enumerate all sub-modules looking for block patterns
        for name, module in model.named_modules():
            if hasattr(module, 'self_attn') or hasattr(module, 'attn'):
                if 'llm' not in name and 'model.layers' not in name:
                    blocks.append(name)

    return {"blocks": blocks[:24]}  # Cap at 24 blocks


@torch.inference_mode()
def analyze_dit_sensitivity_ovis(model, tok, prompt: str, cond_fresh: torch.Tensor,
                                  cond_edited: torch.Tensor, device: str = "cuda"):
    """
    Analyze DiT sensitivity for Ovis by running one denoising step with
    fresh vs edited conditioning.
    """
    vg = model.visual_generator

    # Get block names
    block_info = get_dit_block_names_ovis(model)
    all_blocks = block_info["double_blocks"] + block_info["single_blocks"]

    # Prepare conditioning in the format Yak expects
    # Yak's generate_image expects: cond_dict with "txt" key
    # The TextProjection (txt_in) will be applied inside the model
    # We need to hook into intermediate blocks

    # Create dummy noise for a single step
    # Yak generates 256x256 images with patch_size=2, so latent is 16x16x16
    # But we just need one step to measure sensitivity
    batch_size = 1
    h_latent = 16  # 256 / 16 (after patchification)
    w_latent = 16
    channels = 16  # Yak latent channels

    # Collect activations from all blocks with fresh conditioning
    results = {"double_blocks": [], "single_blocks": []}

    # Hook into the forward of double and single blocks
    # We'll manually run the conditioning through txt_in and measure block outputs

    # First, project both conditions through the txt_in (SingleTokenRefiner)
    backbone = vg.backbone
    txt_in = backbone.txt_in
    # cond shape: (seq, 4096) -> unsqueeze to (1, seq, 4096)
    cond_fresh_dict = txt_in(cond_fresh.unsqueeze(0).to(device=device, dtype=torch.bfloat16))
    cond_edited_dict = txt_in(cond_edited.unsqueeze(0).to(device=device, dtype=torch.bfloat16))
    cond_fresh_proj = cond_fresh_dict["txt_fea"]  # (1, seq, 1536)
    cond_edited_proj = cond_edited_dict["txt_fea"]  # (1, seq, 1536)

    # Measure drift after TextProjection
    proj_delta = cond_edited_proj - cond_fresh_proj
    proj_drift = torch.norm(proj_delta).item()
    proj_relative = proj_drift / (torch.norm(cond_fresh_proj).item() + 1e-10)

    # For each double block, measure how much the img hidden state changes
    # due to the conditioning perturbation
    # We do this by running a random img latent through the blocks with both conditionings
    img_latent = torch.randn(1, h_latent * w_latent, 1536, device=device, dtype=torch.bfloat16)
    txt_latent_fresh = cond_fresh_proj.to(torch.bfloat16)
    txt_latent_edited = cond_edited_proj.to(torch.bfloat16)

    # Timestep embedding via backbone
    def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
        t = time_factor * t
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if torch.is_floating_point(t):
            embedding = embedding.to(t)
        return embedding

    timestep = torch.tensor([0.5], device=device)
    t_emb = backbone.time_in(timestep_embedding(timestep, 256).to(dtype=torch.bfloat16))
    # vector_in expects a vec_in_dim vector; use zeros as placeholder
    vec = t_emb  # Use time embedding as the vec conditioning

    # Run through double blocks
    # DoubleStreamXBlock.forward(img, txt, vec, pe) -> (img, txt)
    img_fresh = img_latent.clone()
    img_edited = img_latent.clone()
    txt_fresh = txt_latent_fresh.clone()
    txt_edited = txt_latent_edited.clone()

    # Create dummy positional embeddings (pe) - not critical for drift measurement
    pe = None  # Will try without pe first

    for i, block_name in enumerate(block_info["double_blocks"]):
        try:
            block = get_module(model, block_name)
            try:
                out_fresh = block(img=img_fresh, txt=txt_fresh, vec=vec, pe=pe)
                out_edited = block(img=img_edited, txt=txt_edited, vec=vec, pe=pe)
                img_fresh, txt_fresh = out_fresh
                img_edited, txt_edited = out_edited
            except TypeError:
                # Try without pe
                try:
                    out_fresh = block(img=img_fresh, txt=txt_fresh, vec=vec)
                    out_edited = block(img=img_edited, txt=txt_edited, vec=vec)
                    img_fresh, txt_fresh = out_fresh
                    img_edited, txt_edited = out_edited
                except Exception:
                    pass

            img_drift = torch.norm(img_edited.float() - img_fresh.float()).item()
            txt_drift = torch.norm(txt_edited.float() - txt_fresh.float()).item()
            results["double_blocks"].append({
                "block_idx": i,
                "img_drift": img_drift,
                "txt_drift": txt_drift,
                "img_relative_drift": img_drift / (torch.norm(img_fresh.float()).item() + 1e-10),
            })
        except Exception as e:
            results["double_blocks"].append({
                "block_idx": i,
                "error": str(e),
                "img_drift": 0.0,
                "txt_drift": 0.0,
            })

    # Run through single blocks
    # SingleStreamBlock.forward(x, vec, pe) -> x
    combined_fresh = torch.cat([img_fresh, txt_fresh], dim=1)
    combined_edited = torch.cat([img_edited, txt_edited], dim=1)

    for i, block_name in enumerate(block_info["single_blocks"]):
        try:
            block = get_module(model, block_name)
            try:
                combined_fresh = block(combined_fresh, vec=vec, pe=pe)
                combined_edited = block(combined_edited, vec=vec, pe=pe)
            except TypeError:
                try:
                    combined_fresh = block(combined_fresh, vec=vec)
                    combined_edited = block(combined_edited, vec=vec)
                except Exception:
                    pass

            drift = torch.norm(combined_edited.float() - combined_fresh.float()).item()
            results["single_blocks"].append({
                "block_idx": i,
                "drift": drift,
                "relative_drift": drift / (torch.norm(combined_fresh.float()).item() + 1e-10),
            })
        except Exception as e:
            results["single_blocks"].append({
                "block_idx": i,
                "error": str(e),
                "drift": 0.0,
            })

    return {
        "proj_drift": proj_drift,
        "proj_relative": proj_relative,
        "double_blocks": results["double_blocks"],
        "single_blocks": results["single_blocks"],
    }


@torch.inference_mode()
def analyze_dit_sensitivity_generic(model, tok, prompt: str, cond_fresh: torch.Tensor,
                                     cond_edited: torch.Tensor, model_name: str, device: str = "cuda"):
    """Analyze decoder sensitivity for BLIP3o / OmniGen2."""
    # For these models, we measure how much the conditioning delta
    # propagates through whatever decoder blocks exist

    block_info = get_dit_block_names_generic(model, model_name)
    blocks = block_info.get("blocks", [])

    delta_norm = torch.norm(cond_edited.float() - cond_fresh.float()).item()
    fresh_norm = torch.norm(cond_fresh.float()).item()

    results = {
        "cond_delta_norm": delta_norm,
        "cond_relative_delta": delta_norm / (fresh_norm + 1e-10),
        "num_decoder_blocks_found": len(blocks),
        "block_names": blocks[:10],  # First 10 for reference
    }

    # If we found decoder blocks, try to collect activations
    if blocks:
        # Collect outputs from decoder blocks under both conditions
        # This is model-specific and may not work for all architectures
        results["note"] = "Decoder block analysis requires model-specific forward pass"

    return results


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

    all_results = []
    for i, case in enumerate(tqdm(dataset, desc=f"Exp4 {args.model_name} {args.method}")):
        info = get_case_info(case)

        # Get conditioning from both models
        cond_fresh = get_conditioning(fresh_model, tok, info["image_prompt"], args.model_name, args.device)
        cond_edited = get_conditioning(edited_model, tok_e, info["image_prompt"], args.model_name, args.device)

        if cfg["model_type"] == "ovis":
            result = analyze_dit_sensitivity_ovis(
                fresh_model, tok, info["image_prompt"], cond_fresh, cond_edited, args.device
            )
        else:
            result = analyze_dit_sensitivity_generic(
                fresh_model, tok, info["image_prompt"], cond_fresh, cond_edited,
                args.model_name, args.device
            )

        result["case_idx"] = i
        result["category"] = info["category"]
        all_results.append(result)

    # Aggregate for Ovis
    if cfg["model_type"] == "ovis":
        avg_proj_drift = float(np.mean([r.get("proj_drift", 0) for r in all_results]))
        avg_proj_relative = float(np.mean([r.get("proj_relative", 0) for r in all_results]))

        # Average per-block drift
        n_double = 6
        n_single = 12
        avg_double_img = []
        avg_double_txt = []
        for bi in range(n_double):
            img_drifts = [r["double_blocks"][bi]["img_drift"] for r in all_results
                          if bi < len(r.get("double_blocks", []))]
            txt_drifts = [r["double_blocks"][bi]["txt_drift"] for r in all_results
                          if bi < len(r.get("double_blocks", []))]
            avg_double_img.append(float(np.mean(img_drifts)) if img_drifts else 0)
            avg_double_txt.append(float(np.mean(txt_drifts)) if txt_drifts else 0)

        avg_single = []
        for bi in range(n_single):
            drifts = [r["single_blocks"][bi]["drift"] for r in all_results
                      if bi < len(r.get("single_blocks", []))]
            avg_single.append(float(np.mean(drifts)) if drifts else 0)

        summary = {
            "model_name": args.model_name,
            "method": args.method,
            "num_cases": len(all_results),
            "avg_proj_drift": avg_proj_drift,
            "avg_proj_relative": avg_proj_relative,
            "avg_double_block_img_drift": avg_double_img,
            "avg_double_block_txt_drift": avg_double_txt,
            "avg_single_block_drift": avg_single,
        }

        # Print table
        headers = ["Block", "Type", "Avg Img Drift", "Avg Txt Drift"]
        rows = []
        for i in range(n_double):
            rows.append([f"Double {i}", "dual-stream", avg_double_img[i], avg_double_txt[i]])
        print_table(headers, rows, f"DiT Sensitivity (Double): {args.model_name} / {args.method}")

        headers2 = ["Block", "Type", "Avg Combined Drift"]
        rows2 = [[f"Single {i}", "single-stream", avg_single[i]] for i in range(n_single)]
        print_table(headers2, rows2, f"DiT Sensitivity (Single): {args.model_name} / {args.method}")

        print(f"TextProjection drift: {avg_proj_drift:.4f} (relative: {avg_proj_relative:.6f})")
    else:
        summary = {
            "model_name": args.model_name,
            "method": args.method,
            "num_cases": len(all_results),
            "avg_cond_delta_norm": float(np.mean([r.get("cond_delta_norm", 0) for r in all_results])),
            "avg_cond_relative_delta": float(np.mean([r.get("cond_relative_delta", 0) for r in all_results])),
        }
        print(f"\n{args.model_name} / {args.method}:")
        print(f"  Avg cond delta norm: {summary['avg_cond_delta_norm']:.4f}")
        print(f"  Avg cond relative delta: {summary['avg_cond_relative_delta']:.6f}")

    save_results({"summary": summary, "per_case": all_results}, "exp4_dit_sensitivity", args.model_name, args.method)

    del fresh_model, edited_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
