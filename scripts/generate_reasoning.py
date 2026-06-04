"""
Reasoning generation for edited models via the vLLM offline LLM engine.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from util.reasoning_prompt import (
    color_reasoning_prompt,
    material_reasoning_prompt,
    shape_reasoning_prompt,
    pattern_reasoning_prompt,
    size_reasoning_prompt,
    occupation_reasoning_prompt,
    location_reasoning_prompt,
    creator_reasoning_prompt,
    affiliation_reasoning_prompt,
)
from util.size_synonyms import reasoning_contains_size_target, is_size_adjective


ATTRIBUTE_REASONING_PROMPTS = {
    "color": color_reasoning_prompt,
    "material": material_reasoning_prompt,
    "shape": shape_reasoning_prompt,
    "pattern": pattern_reasoning_prompt,
    "size": size_reasoning_prompt,
}
RELATION_REASONING_PROMPTS = {
    "occupation": occupation_reasoning_prompt,
    "location": location_reasoning_prompt,
    "creator": creator_reasoning_prompt,
    "affiliation": affiliation_reasoning_prompt,
}


def _target_in_reasoning(category: str, gt_target: str, reasoning_output: str) -> bool:
    if not gt_target or not reasoning_output:
        return False
    if (category or "").strip().lower() == "size" and is_size_adjective(gt_target):
        return reasoning_contains_size_target(reasoning_output, gt_target)
    return gt_target.lower() in reasoning_output.lower()


def get_reasoning_prompt(source: str, category: str, subject: str, image_prompt: str) -> str:
    if source == "attribute":
        template = ATTRIBUTE_REASONING_PROMPTS.get((category or "").lower())
    else:
        template = RELATION_REASONING_PROMPTS.get((category or "").lower())
    if template is None:
        return f"Describe the {category} of {subject} in detail, then generate a visual description for: {image_prompt}"
    return template.format(subject=subject, attribute=category, image_prompt=image_prompt)


# ---------------- per-model chat formatting ---------------------------------

def _format_prompt(model_type: str, tokenizer, raw_prompt: str) -> str:
    """Apply the chat template that matches what the original HF pipeline does."""
    if model_type == "ovis":
        # Qwen3ConversationFormatter with enable_thinking=False
        return (
            "<|im_start|>user\n" + raw_prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    if model_type == "blip3o":
        # BLIP3o conv_qwen: ChatML with default helpful-assistant system prompt
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n" + raw_prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    # omnigen2 / generic -> use the tokenizer chat template (Qwen2.5-VL style)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": raw_prompt}],
        tokenize=False, add_generation_prompt=True,
    )


# ---------------- task expansion --------------------------------------------

def build_tasks(dataset_slice, start_idx: int) -> Tuple[List[Tuple], Dict[int, dict]]:
    """Expand the dataset into (case_id, source, key, raw_prompt, meta) tuples.

    Mirrors the per-stage expansion in the loader so the downstream VQA judge
    keys (case_<id>_<stage>.png) line up.
    """
    tasks = []
    items_by_id = {}
    for i, item in enumerate(dataset_slice):
        case_id = start_idx + i
        subject = item.get("subject", "")
        category = item.get("category", "")
        source = item.get("source", "")
        items_by_id[case_id] = {
            "case_id": case_id, "subject": subject, "category": category,
            "source": source, "stages": {},
            "success": True, "error": None,
        }

        if source == "attribute":
            stage_keys = ["stage_1", "stage_2", "stage_3", "stage_4"]
        else:
            stage_keys = ["stage_1"]
        for stage_key in stage_keys:
            if stage_key not in item:
                continue
            stage = item[stage_key]
            image_prompt = stage.get("image_prompt", "")
            if not image_prompt:
                continue
            reasoning_prompt = get_reasoning_prompt(source, category, subject, image_prompt)
            tasks.append((case_id, source, stage_key, reasoning_prompt, {
                "image_prompt": image_prompt,
                "gt": stage.get("gt", ""),
                "gt_target": stage.get("gt_target", ""),
                "edit_prompt": stage.get("prompt", ""),
                "category": category,
            }))
    return tasks, items_by_id


# ---------------- main -------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Reasoning generation via vLLM offline engine")
    p.add_argument("--vllm_model_dir", required=True,
                   help="HF directory produced by extract_edited_llm.py")
    p.add_argument("--model_type", required=True,
                   choices=["ovis", "blip3o", "omnigen2"],
                   help="Which unified model the LLM came from (selects chat template)")
    p.add_argument("--data_path", default="data/UniKE.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--start_idx", type=int, default=None)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--max_num_seqs", type=int, default=128)
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPU ids; sets CUDA_VISIBLE_DEVICES "
                        "(must match tensor_parallel_size)")
    p.add_argument("--enforce_eager", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        n_gpus = len([g for g in args.gpus.split(",") if g.strip()])
        if n_gpus != args.tensor_parallel_size:
            raise SystemExit(
                f"--gpus has {n_gpus} entries but --tensor_parallel_size={args.tensor_parallel_size}; "
                "they must match."
            )

    # Defer vLLM import until after env vars are set.
    from vllm import LLM, SamplingParams

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[vllm-reason] data: {args.data_path}")
    print(f"[vllm-reason] model dir: {args.vllm_model_dir} ({args.model_type})")
    print(f"[vllm-reason] tp={args.tensor_parallel_size}  max_num_seqs={args.max_num_seqs}  "
          f"max_model_len={args.max_model_len}")

    with open(args.data_path) as f:
        dataset = json.load(f)
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    dataset_slice = dataset[start_idx:end_idx]
    print(f"[vllm-reason] processing items [{start_idx}, {end_idx}) -> {len(dataset_slice)} cases")

    tasks, items_by_id = build_tasks(dataset_slice, start_idx)
    print(f"[vllm-reason] expanded into {len(tasks)} reasoning prompts")

    # Init engine.
    engine_kwargs = dict(
        model=args.vllm_model_dir,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    # Qwen2.5-VL family in vLLM enables multimodal by default; we never feed
    # images here so cap mm slots to zero to free KV space.
    if args.model_type in ("blip3o", "omnigen2"):
        engine_kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

    t_init = time.time()
    llm = LLM(**engine_kwargs)
    tokenizer = llm.get_tokenizer()
    print(f"[vllm-reason] engine ready in {time.time() - t_init:.1f}s")

    sp = SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1,
        max_tokens=args.max_new_tokens,
        repetition_penalty=1.0, seed=args.seed,
    )

    # Format every prompt up-front so the engine can do continuous batching.
    formatted_prompts = [_format_prompt(args.model_type, tokenizer, raw)
                         for (_, _, _, raw, _) in tasks]
    raw_prompts = [raw for (_, _, _, raw, _) in tasks]

    t_gen = time.time()
    outputs = llm.generate(formatted_prompts, sp)
    gen_secs = time.time() - t_gen
    print(f"[vllm-reason] generated {len(outputs)} completions in {gen_secs:.1f}s "
          f"({len(outputs)/max(gen_secs,1e-6):.2f} prompts/s)")

    # Stitch results back per case.
    success = 0
    attr_target_count = attr_total = 0
    rel_target_count = rel_total = 0

    for (case_id, source, key, _raw_prompt, meta), out in zip(tasks, outputs):
        text = out.outputs[0].text if out.outputs else ""
        contains_target = _target_in_reasoning(meta["category"], meta["gt_target"], text)

        record = {
            "image_prompt": meta["image_prompt"],
            "gt": meta["gt"],
            "gt_target": meta["gt_target"],
            "edit_prompt": meta["edit_prompt"],
            "reasoning_prompt": _raw_prompt,
            "reasoning_output": text,
            "contains_target": contains_target,
        }
        items_by_id[case_id]["stages"][key] = record

    for case_id, item in items_by_id.items():
        if item["source"] == "attribute":
            attr_total += 1
            if item["stages"].get("stage_1", {}).get("contains_target", False):
                attr_target_count += 1
            success += 1
        elif item["source"] == "relation":
            rel_total += 1
            if item["stages"].get("stage_1", {}).get("contains_target", False):
                rel_target_count += 1
            success += 1

    results = sorted(items_by_id.values(), key=lambda r: r["case_id"])
    out_file = output_dir / f"reasoning_{start_idx}_{end_idx}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "model_dir": args.vllm_model_dir,
        "model_type": args.model_type,
        "engine": "vllm",
        "vllm_kwargs": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "num_prompts": len(outputs),
        "num_cases": len(results),
        "success_count": success,
        "generation_seconds": gen_secs,
        "prompts_per_sec": len(outputs) / max(gen_secs, 1e-6),
    }
    with open(output_dir / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[vllm-reason] wrote {out_file}")
    print(f"[vllm-reason]   total cases  : {len(results)}")
    print(f"[vllm-reason]   success      : {success}/{len(results)}")
    if attr_total:
        print(f"[vllm-reason]   attr stage_1 contains_target: "
              f"{attr_target_count}/{attr_total} ({100*attr_target_count/attr_total:.1f}%)")
    if rel_total:
        print(f"[vllm-reason]   relation stage_1 contains_target: "
              f"{rel_target_count}/{rel_total} ({100*rel_target_count/rel_total:.1f}%)")
    print(f"[vllm-reason]   gen wall-time: {gen_secs:.1f}s "
          f"({len(outputs)/max(gen_secs,1e-6):.2f} prompts/s)")


if __name__ == "__main__":
    main()
