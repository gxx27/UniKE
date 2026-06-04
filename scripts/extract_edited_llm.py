"""
Export the LLM backbone of an edited unified model as a stand-alone HuggingFace directory for vLLM.
"""

import argparse
import io
import json
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from util.model_utils import get_model_type  # noqa: E402


class _CpuUnpickler(pickle.Unpickler):
    """Unpickler that forces all torch tensors to CPU."""
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _load_pickle_cpu(path: str) -> dict:
    with open(path, "rb") as f:
        return _CpuUnpickler(f).load()


def _resolve_local_snapshot(model_name: str) -> Path:
    """Return the on-disk HF snapshot dir for a HF model id."""
    hub_dirs = [
        Path(os.environ.get("HF_HOME", "")) / "hub",
        Path(os.environ.get("HF_HOME", "")),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    repo_dirname = "models--" + model_name.replace("/", "--")
    for d in hub_dirs:
        candidate = d / repo_dirname / "snapshots"
        if candidate.is_dir():
            snaps = sorted([s for s in candidate.iterdir() if s.is_dir()])
            if snaps:
                return snaps[-1]
    raise FileNotFoundError(f"Could not locate local snapshot for '{model_name}' under HF_HOME")


def _split_state_dict(state_dict: Dict[str, torch.Tensor], target_bytes: int = 4 * 1024 ** 3):
    """Split a state dict into shards each <= target_bytes (default 4 GB)."""
    shards = [{}]
    sizes = [0]
    for key, tensor in state_dict.items():
        nbytes = tensor.numel() * tensor.element_size()
        if sizes[-1] + nbytes > target_bytes and sizes[-1] > 0:
            shards.append({})
            sizes.append(0)
        shards[-1][key] = tensor
        sizes[-1] += nbytes
    return shards


def _save_sharded_safetensors(state_dict: Dict[str, torch.Tensor], output_dir: Path):
    """Save a state dict as one or more sharded safetensors files + index."""
    # Convert any float32 tensors to bfloat16 to reduce checkpoint size to match
    # the original on-disk format and keep parity with the runtime dtype used
    # by both the HF and vLLM paths.
    converted = {}
    for k, v in state_dict.items():
        if v.dtype == torch.float32:
            v = v.to(torch.bfloat16)
        converted[k] = v.contiguous()
    state_dict = converted

    shards = _split_state_dict(state_dict)
    n = len(shards)
    if n == 1:
        save_file(shards[0], str(output_dir / "model.safetensors"), metadata={"format": "pt"})
        return None
    weight_map = {}
    total_size = 0
    for idx, shard in enumerate(shards, start=1):
        fname = f"model-{idx:05d}-of-{n:05d}.safetensors"
        save_file(shard, str(output_dir / fname), metadata={"format": "pt"})
        for k, v in shard.items():
            weight_map[k] = fname
            total_size += v.numel() * v.element_size()
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    return index


# ---------------------------- per-model exporters ----------------------------

def _export_ovis(state_dict, source_snapshot: Path, output_dir: Path):
    """Strip `llm.` prefix and save as a Qwen3ForCausalLM checkpoint."""
    llm_state = {}
    for k, v in state_dict.items():
        if k.startswith("llm."):
            llm_state[k[len("llm."):]] = v
    if not llm_state:
        raise RuntimeError("No `llm.*` keys found in Ovis edited state dict")

    with open(source_snapshot / "config.json") as f:
        ovis_cfg = json.load(f)

    llm_cfg = dict(ovis_cfg["llm_config"])
    llm_cfg["architectures"] = ["Qwen3ForCausalLM"]
    llm_cfg["model_type"] = "qwen3"
    llm_cfg["torch_dtype"] = "bfloat16"
    llm_cfg.pop("_name_or_path", None)
    llm_cfg.pop("_attn_implementation_autoset", None)
    for noise_key in ("id2label", "label2id", "begin_suppress_tokens", "bad_words_ids", "task_specific_params"):
        llm_cfg.pop(noise_key, None)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(llm_cfg, f, indent=2)

    _save_sharded_safetensors(llm_state, output_dir)

    for fname in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "added_tokens.json", "generation_config.json"):
        src = source_snapshot / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)

    return llm_cfg


def _export_qwen25vl(state_dict, source_cfg_path: Path, source_tokenizer_dir: Path, output_dir: Path):
    """Save a Qwen2.5-VL ForConditionalGeneration checkpoint.

    `state_dict` must already contain the relevant `model.*`, `lm_head.*` and
    `visual.*` keys. Any extra keys (BLIP3o's `model.dit.*`, `model.latent_queries`,
    `model.gen_*`) must be filtered out by the caller.
    """
    with open(source_cfg_path) as f:
        cfg = json.load(f)

    cfg["architectures"] = ["Qwen2_5_VLForConditionalGeneration"]
    cfg["model_type"] = "qwen2_5_vl"
    cfg["torch_dtype"] = "bfloat16"
    cfg.pop("_name_or_path", None)
    cfg.pop("auto_map", None)

    # Strip BLIP3o-only fields that confuse Qwen2.5-VL config parsing.
    for k in ("freeze_mm_mlp_adapter", "image_aspect_ratio", "mm_patch_merge_type",
               "mm_projector_lr", "mm_projector_type", "mm_use_im_patch_token",
               "mm_use_im_start_end", "mm_vision_select_feature", "mm_vision_select_layer",
               "n_query", "tune_mm_mlp_adapter", "use_mm_proj",
               "tokenizer_model_max_length", "tokenizer_padding_side",
               "vision_tower_pretrained", "gen_hidden_size", "gen_pooling",
               "gen_vision_tower"):
        cfg.pop(k, None)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    _save_sharded_safetensors(state_dict, output_dir)

    for fname in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "added_tokens.json", "preprocessor_config.json",
                   "chat_template.json", "generation_config.json"):
        src = source_tokenizer_dir / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)

    return cfg


def _export_blip3o(state_dict, source_snapshot: Path, output_dir: Path):
    """BLIP3o has Qwen2.5-VL backbone + extra DiT / gen heads. Drop the extras."""
    drop_prefixes = ("model.dit.", "model.latent_queries", "model.gen_", "gen_")
    filtered = {k: v for k, v in state_dict.items()
                if not any(k.startswith(p) for p in drop_prefixes)}
    return _export_qwen25vl(
        filtered,
        source_cfg_path=source_snapshot / "config.json",
        source_tokenizer_dir=source_snapshot,
        output_dir=output_dir,
    )


def _export_omnigen2(state_dict, source_snapshot: Path, output_dir: Path):
    """OmniGen2 stores the LLM under mllm/ ; the saved state-dict is already the
    plain Qwen2.5-VL state-dict for that submodule."""
    return _export_qwen25vl(
        state_dict,
        source_cfg_path=source_snapshot / "mllm" / "config.json",
        source_tokenizer_dir=source_snapshot / "mllm_processor",
        output_dir=output_dir,
    )


# ---------------------------------- main -------------------------------------

def main():
    p = argparse.ArgumentParser(description="Extract LLM portion of an edited unified model into a vLLM-loadable HF directory")
    p.add_argument("--edited_model_path", required=True,
                   help="Path to results/.../edited_model.pkl")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for the HF checkpoint")
    p.add_argument("--model_name", default=None,
                   help="Override model name (otherwise read from pickle)")
    args = p.parse_args()

    print(f"[extract] loading {args.edited_model_path}")
    save_data = _load_pickle_cpu(args.edited_model_path)
    state_dict = save_data["model_state_dict"]
    model_name = args.model_name or save_data["model_name"]
    model_type = get_model_type(model_name)

    print(f"[extract] model_name={model_name}  model_type={model_type}  num_keys={len(state_dict)}")

    source_snapshot = _resolve_local_snapshot(model_name)
    print(f"[extract] source snapshot: {source_snapshot}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "ovis":
        cfg = _export_ovis(state_dict, source_snapshot, output_dir)
    elif model_type == "blip3o":
        cfg = _export_blip3o(state_dict, source_snapshot, output_dir)
    elif model_type == "omnigen2":
        cfg = _export_omnigen2(state_dict, source_snapshot, output_dir)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    meta = {
        "source_pickle": str(args.edited_model_path),
        "source_model_name": model_name,
        "source_model_type": model_type,
        "exported_architecture": cfg["architectures"][0],
        "num_edits": save_data.get("num_edits"),
    }
    with open(output_dir / "extract_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[extract] saved {cfg['architectures'][0]} checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
