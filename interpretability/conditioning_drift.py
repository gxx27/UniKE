#!/usr/bin/env python3
"""
Compares conditioning-perturbation characteristics across all three editing methods and models.
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    MODEL_CONFIGS, METHODS, get_pkl_path, load_fresh_model, load_edited_model,
    load_dataset, get_case_info, get_conditioning, get_all_hidden_states,
    get_llm, save_results, print_table,
)


@torch.inference_mode()
def compute_drift_metrics(fresh_model, edited_model, tok, prompt: str,
                          model_name: str, device: str = "cuda"):
    """Compute comprehensive drift metrics for a single case."""
    cfg = MODEL_CONFIGS[model_name]

    # Get conditioning
    cond_fresh = get_conditioning(fresh_model, tok, prompt, model_name, device).float().cpu()
    cond_edited = get_conditioning(edited_model, tok, prompt, model_name, device).float().cpu()

    delta = cond_edited - cond_fresh  # (seq, dim)
    seq_len = delta.shape[0]

    # Total metrics
    l2_total = torch.norm(delta).item()
    l2_mean_per_token = torch.norm(delta, dim=-1).mean().item()
    l2_last_token = torch.norm(delta[-1]).item()

    # Cosine distance
    cond_fresh_flat = cond_fresh.reshape(-1)
    cond_edited_flat = cond_edited.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        cond_fresh_flat.unsqueeze(0), cond_edited_flat.unsqueeze(0)
    ).item()
    cos_dist = 1.0 - cos_sim

    # Relative drift
    fresh_norm = torch.norm(cond_fresh).item()
    relative_drift = l2_total / (fresh_norm + 1e-10)

    # Per-layer hidden state analysis
    fresh_hidden = get_all_hidden_states(fresh_model, tok, prompt, model_name, device)
    edited_hidden = get_all_hidden_states(edited_model, tok, prompt, model_name, device)

    # Drift at edited vs non-edited layers
    edited_layer_drift = 0.0
    non_edited_layer_drift = 0.0
    per_layer_drift = []

    for layer_idx in range(cfg["total_layers"]):
        h_idx = layer_idx + 1  # hidden_states[0] is embedding
        if h_idx < len(fresh_hidden) and h_idx < len(edited_hidden):
            drift = torch.norm(edited_hidden[h_idx] - fresh_hidden[h_idx]).item()
            per_layer_drift.append(drift)
            if layer_idx in cfg["edited_layers"]:
                edited_layer_drift += drift
            else:
                non_edited_layer_drift += drift

    # Drift concentration ratio
    total_layer_drift = edited_layer_drift + non_edited_layer_drift
    drift_concentration = edited_layer_drift / (total_layer_drift + 1e-10)

    return {
        "l2_total": l2_total,
        "l2_mean_per_token": l2_mean_per_token,
        "l2_last_token": l2_last_token,
        "cosine_distance": cos_dist,
        "relative_drift": relative_drift,
        "drift_concentration": drift_concentration,
        "edited_layer_drift": edited_layer_drift,
        "non_edited_layer_drift": non_edited_layer_drift,
        "per_layer_drift": per_layer_drift,
        "seq_len": seq_len,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--method", type=str, required=True, choices=METHODS)
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
    for i, case in enumerate(tqdm(dataset, desc=f"Exp3 {args.model_name} {args.method}")):
        info = get_case_info(case)
        result = compute_drift_metrics(
            fresh_model, edited_model, tok, info["image_prompt"],
            args.model_name, args.device
        )
        result["case_idx"] = i
        result["category"] = info["category"]
        all_results.append(result)

    # Aggregate
    metric_keys = ["l2_total", "l2_mean_per_token", "l2_last_token",
                   "cosine_distance", "relative_drift", "drift_concentration"]
    summary = {
        "model_name": args.model_name,
        "method": args.method,
        "num_cases": len(all_results),
    }
    for key in metric_keys:
        vals = [r[key] for r in all_results]
        summary[f"avg_{key}"] = float(np.mean(vals))
        summary[f"std_{key}"] = float(np.std(vals))

    # Per-category breakdown
    categories = set(r["category"] for r in all_results if r["category"])
    category_breakdown = {}
    for cat in categories:
        cat_results = [r for r in all_results if r["category"] == cat]
        if cat_results:
            category_breakdown[cat] = {
                "count": len(cat_results),
                "avg_l2_total": float(np.mean([r["l2_total"] for r in cat_results])),
                "avg_relative_drift": float(np.mean([r["relative_drift"] for r in cat_results])),
            }
    summary["category_breakdown"] = category_breakdown

    save_results({"summary": summary, "per_case": all_results}, "exp3_multi_method", args.model_name, args.method)

    # Print summary table
    headers = ["Metric", "Mean", "Std"]
    rows = [[key, summary[f"avg_{key}"], summary[f"std_{key}"]] for key in metric_keys]
    print_table(headers, rows, f"Drift Metrics: {args.model_name} / {args.method}")

    del fresh_model, edited_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
