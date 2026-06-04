#!/usr/bin/env python3
"""
Measures and compares the conditioning signals under the Direct vs Reasoning-augmented protocols.
"""

import argparse
import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import (
    MODEL_CONFIGS, get_pkl_path, get_reasoning_path, load_fresh_model,
    load_edited_model, load_dataset, get_case_info, get_conditioning,
    save_results, print_table,
)


def load_reasoning_outputs(model_name: str, method: str) -> dict:
    """Load reasoning outputs, returns dict mapping case_idx -> reasoning text.

    File format: list of dicts with structure:
    {case_id, subject, category, source, stages: {stage_1: {reasoning_output, ...}}, ...}
    """
    reasoning_path = get_reasoning_path(model_name, method)
    if not reasoning_path.exists():
        print(f"Warning: reasoning file not found: {reasoning_path}")
        return {}
    with open(reasoning_path) as f:
        data = json.load(f)

    result = {}
    if isinstance(data, list):
        for item in data:
            case_id = item.get("case_id", None)
            if case_id is None:
                continue
            # Extract reasoning_output from stages
            stages = item.get("stages", {})
            for stage_key in ["stage_1", "stage_2", "stage_3", "stage_4"]:
                if stage_key in stages and stages[stage_key].get("reasoning_output"):
                    result[case_id] = stages[stage_key]["reasoning_output"]
                    break
    return result


def build_reasoning_prompt(image_prompt: str, reasoning_text: str) -> str:
    """Build a reasoning-augmented prompt from the base prompt and reasoning output."""
    if not reasoning_text:
        return image_prompt
    # The reasoning-augmented prompt includes the reasoning as context
    return f"{reasoning_text}\n\nBased on the above, generate an image: {image_prompt}"


@torch.inference_mode()
def analyze_reasoning_conditioning(
    fresh_model, edited_model, tok, prompt_direct: str, prompt_reasoning: str,
    model_name: str, device: str = "cuda"
):
    """
    Compute conditioning vectors under 4 conditions and decompose.
    """
    # 4 conditions
    c_fresh_direct = get_conditioning(fresh_model, tok, prompt_direct, model_name, device).float().cpu()
    c_edit_direct = get_conditioning(edited_model, tok, prompt_direct, model_name, device).float().cpu()
    c_fresh_reason = get_conditioning(fresh_model, tok, prompt_reasoning, model_name, device).float().cpu()
    c_edit_reason = get_conditioning(edited_model, tok, prompt_reasoning, model_name, device).float().cpu()

    # Use mean over sequence (since reasoning prompt is longer)
    c_fd = c_fresh_direct.mean(dim=0)
    c_ed = c_edit_direct.mean(dim=0)
    c_fr = c_fresh_reason.mean(dim=0)
    c_er = c_edit_reason.mean(dim=0)

    # Decomposition
    delta_edit = c_ed - c_fd  # edit-induced perturbation (direct)
    delta_reason = c_fr - c_fd  # reasoning-induced shift (on fresh model)
    delta_edit_on_reason = c_er - c_fr  # edit perturbation on reasoning prompt

    # Combined effects
    delta_total_direct = c_ed - c_fd  # total change in direct mode
    delta_total_reason = c_er - c_fd  # total change in reasoning mode

    # Norms
    norm_edit = torch.norm(delta_edit).item()
    norm_reason = torch.norm(delta_reason).item()
    norm_edit_on_reason = torch.norm(delta_edit_on_reason).item()
    norm_total_direct = torch.norm(delta_total_direct).item()
    norm_total_reason = torch.norm(delta_total_reason).item()
    norm_fresh_direct = torch.norm(c_fd).item()

    # Ratios
    edit_to_reason_ratio = norm_edit / (norm_reason + 1e-10)
    edit_amplification = norm_edit_on_reason / (norm_edit + 1e-10)  # >1 means reasoning amplifies edit

    # Cosine alignment between edit direction and reasoning direction
    cos_edit_reason = torch.nn.functional.cosine_similarity(
        delta_edit.unsqueeze(0), delta_reason.unsqueeze(0)
    ).item()

    # Cosine alignment between edit_direct and edit_on_reasoning
    cos_edit_consistency = torch.nn.functional.cosine_similarity(
        delta_edit.unsqueeze(0), delta_edit_on_reason.unsqueeze(0)
    ).item()

    # Signal-to-noise ratio: how much of reasoning signal is in the edit direction?
    # Project reasoning shift onto edit direction
    if norm_edit > 1e-10:
        edit_dir = delta_edit / (norm_edit + 1e-10)
        reason_proj_on_edit = torch.dot(delta_reason, edit_dir).item()
    else:
        reason_proj_on_edit = 0.0

    return {
        "norm_edit_direct": norm_edit,
        "norm_reasoning_shift": norm_reason,
        "norm_edit_on_reasoning": norm_edit_on_reason,
        "norm_total_direct": norm_total_direct,
        "norm_total_reasoning": norm_total_reason,
        "norm_fresh_direct": norm_fresh_direct,
        "edit_to_reason_ratio": edit_to_reason_ratio,
        "edit_amplification": edit_amplification,
        "cos_edit_reason": cos_edit_reason,
        "cos_edit_consistency": cos_edit_consistency,
        "reason_proj_on_edit_dir": reason_proj_on_edit,
        "relative_edit_direct": norm_edit / (norm_fresh_direct + 1e-10),
        "relative_edit_reasoning": norm_edit_on_reason / (norm_fresh_direct + 1e-10),
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

    # Load reasoning outputs
    reasoning_outputs = load_reasoning_outputs(args.model_name, args.method)
    print(f"Loaded {len(reasoning_outputs)} reasoning outputs")

    print(f"Loading fresh model: {args.model_name}")
    fresh_model, tok = load_fresh_model(args.model_name, device=args.device)
    fresh_model.eval()

    print(f"Loading edited model: {pkl_path}")
    edited_model, tok_e, _ = load_edited_model(str(pkl_path), device=args.device)
    edited_model.eval()

    all_results = []
    skipped = 0
    for i, case in enumerate(tqdm(dataset, desc=f"Exp6 {args.model_name} {args.method}")):
        info = get_case_info(case)
        prompt_direct = info["image_prompt"]

        # Get reasoning text for this case
        reasoning_text = reasoning_outputs.get(i, reasoning_outputs.get(str(i), ""))
        if not reasoning_text:
            skipped += 1
            continue

        prompt_reasoning = build_reasoning_prompt(prompt_direct, reasoning_text)

        result = analyze_reasoning_conditioning(
            fresh_model, edited_model, tok, prompt_direct, prompt_reasoning,
            args.model_name, args.device
        )
        result["case_idx"] = i
        result["category"] = info["category"]
        result["prompt_direct_len"] = len(tok.encode(prompt_direct))
        result["prompt_reasoning_len"] = len(tok.encode(prompt_reasoning))
        all_results.append(result)

    print(f"Analyzed {len(all_results)} cases, skipped {skipped} (no reasoning)")

    if not all_results:
        print("No results to aggregate. Check reasoning file format.")
        return

    # Aggregate
    metric_keys = [
        "norm_edit_direct", "norm_reasoning_shift", "norm_edit_on_reasoning",
        "edit_to_reason_ratio", "edit_amplification",
        "cos_edit_reason", "cos_edit_consistency",
        "relative_edit_direct", "relative_edit_reasoning",
    ]

    summary = {
        "model_name": args.model_name,
        "method": args.method,
        "num_cases_analyzed": len(all_results),
        "num_cases_skipped": skipped,
    }
    for key in metric_keys:
        vals = [r[key] for r in all_results]
        summary[f"avg_{key}"] = float(np.mean(vals))
        summary[f"std_{key}"] = float(np.std(vals))
        summary[f"median_{key}"] = float(np.median(vals))

    # Per-category breakdown
    categories = set(r["category"] for r in all_results if r["category"])
    category_breakdown = {}
    for cat in categories:
        cat_results = [r for r in all_results if r["category"] == cat]
        if cat_results:
            category_breakdown[cat] = {
                "count": len(cat_results),
                "avg_edit_to_reason_ratio": float(np.mean([r["edit_to_reason_ratio"] for r in cat_results])),
                "avg_edit_amplification": float(np.mean([r["edit_amplification"] for r in cat_results])),
                "avg_cos_edit_reason": float(np.mean([r["cos_edit_reason"] for r in cat_results])),
            }
    summary["category_breakdown"] = category_breakdown

    save_results({"summary": summary, "per_case": all_results}, "exp6_reasoning_conditioning", args.model_name, args.method)

    # Print table
    headers = ["Metric", "Mean", "Std", "Median"]
    rows = [[key, summary[f"avg_{key}"], summary[f"std_{key}"], summary[f"median_{key}"]]
            for key in metric_keys]
    print_table(headers, rows, f"Reasoning Conditioning: {args.model_name} / {args.method}")

    # Key interpretation
    ratio = summary["avg_edit_to_reason_ratio"]
    amp = summary["avg_edit_amplification"]
    print(f"\nInterpretation:")
    print(f"  Edit/Reasoning ratio: {ratio:.3f} {'(edit dominates)' if ratio > 1 else '(reasoning dominates)'}")
    print(f"  Edit amplification: {amp:.3f} {'(reasoning amplifies edit)' if amp > 1 else '(reasoning attenuates edit)'}")
    print(f"  Cos(edit, reason): {summary['avg_cos_edit_reason']:.3f}")

    del fresh_model, edited_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
