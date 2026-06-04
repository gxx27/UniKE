"""
Shared utilities for the interpretability experiments.
"""

import os
import sys
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from contextlib import contextmanager

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_CONFIGS = {
    "AIDC-AI/Ovis-U1-3B": {
        "edited_layers": [4, 5, 6, 7, 8],
        "total_layers": 28,
        "hidden_size": 2048,
        "layer_template": "llm.model.layers.{}",
        "mlp_template": "llm.model.layers.{}.mlp",
        "down_proj_template": "llm.model.layers.{}.mlp.down_proj",
        "model_type": "ovis",
    },
    "BLIP3o/BLIP3o-Model-4B": {
        "edited_layers": [6, 7, 8, 9, 10],
        "total_layers": 28,
        "hidden_size": 2048,
        "layer_template": "model.layers.{}",
        "mlp_template": "model.layers.{}.mlp",
        "down_proj_template": "model.layers.{}.mlp.down_proj",
        "model_type": "blip3o",
    },
    "OmniGen2/OmniGen2": {
        "edited_layers": [6, 7, 8, 9, 10],
        "total_layers": 28,
        "hidden_size": 2048,
        "layer_template": "model.layers.{}",
        "mlp_template": "model.layers.{}.mlp",
        "down_proj_template": "model.layers.{}.mlp.down_proj",
        "model_type": "omnigen2",
    },
}

METHODS = ["AlphaEdit", "MEMIT", "PMET"]

RESULTS_BASE = PROJECT_ROOT / "results"

def get_pkl_path(model_name: str, method: str) -> Path:
    slug = model_name.replace("/", "_")
    return RESULTS_BASE / slug / method / "run_000" / "edited_model.pkl"


def get_reasoning_path(model_name: str, method: str) -> Path:
    slug = model_name.replace("/", "_")
    return RESULTS_BASE / slug / method / "run_000" / "reasoning" / "reasoning_0_2971.json"


def get_module(model: nn.Module, name: str) -> nn.Module:
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(f"Module '{name}' not found")


def get_llm(model):
    if hasattr(model, "llm"):
        return model.llm
    if hasattr(model, "language_model"):
        return model.language_model
    return model


def load_fresh_model(model_name: str, device: str = "cuda"):
    from util.model_utils import load_model_and_tok
    return load_model_and_tok(model_name, device=device, dtype=torch.bfloat16)


def load_edited_model(edited_model_path: str, device: str = "cuda"):
    with open(edited_model_path, "rb") as f:
        save_data = pickle.load(f)

    model_name = save_data["model_name"]
    from util.model_utils import load_model_and_tok
    model, tok = load_model_and_tok(model_name, device=device, dtype=torch.bfloat16)

    state_dict = save_data["model_state_dict"]
    converted = 0
    for key in state_dict:
        if state_dict[key].dtype == torch.float32:
            state_dict[key] = state_dict[key].to(torch.bfloat16)
            converted += 1

    model.load_state_dict(state_dict)
    print(f"Loaded edited: {model_name} ({save_data.get('num_edits', '?')} edits, {converted} fp32->bf16)")
    return model, tok, model_name


def load_dataset(limit: int = 100) -> List[Dict]:
    data_path = PROJECT_ROOT / "data" / "UniKE.json"
    with open(data_path) as f:
        data = json.load(f)
    return data[:limit]


def get_case_info(case: Dict) -> Dict[str, str]:
    for key in ["stage_1", "stage_2", "stage_3", "stage_4"]:
        if key in case and isinstance(case[key], dict) and "image_prompt" in case[key]:
            stage = case[key]
            return {
                "image_prompt": stage["image_prompt"],
                "visual_target": stage.get("visual_target", ""),
                "gt_target": stage.get("gt_target", ""),
                "subject": case.get("subject", ""),
                "category": case.get("category", ""),
                "source": case.get("source", ""),
            }
    return {
        "image_prompt": f"A photo of {case.get('subject', 'the subject')}.",
        "visual_target": "",
        "gt_target": "",
        "subject": case.get("subject", ""),
        "category": case.get("category", ""),
        "source": case.get("source", ""),
    }


@torch.inference_mode()
def get_conditioning(model, tok, prompt: str, model_name: str, device: str = "cuda") -> torch.Tensor:
    """
    Get the conditioning signal that would be fed to the image generator.
    Matches the actual image generation pathway for each model:
    - Ovis: cat(hidden[-1], hidden[-2]) for all tokens -> (seq, 4096)
    - BLIP3o: append latent_queries to text embeddings, process through same LLM
              (including edited layers), extract last N_QUERY hidden states -> (N_QUERY, 2048)
    - OmniGen2: run text through LLM, return hidden_states[-1] for text tokens -> (seq, 2048)
    """
    cfg = MODEL_CONFIGS[model_name]

    if cfg["model_type"] == "ovis":
        llm = get_llm(model)
        inputs = tok(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        outputs = llm(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        h_last = outputs.hidden_states[-1]    # (1, seq, 2048)
        h_penult = outputs.hidden_states[-2]  # (1, seq, 2048)
        cond = torch.cat([h_last, h_penult], dim=-1)  # (1, seq, 4096)
        return cond[0]  # (seq, 4096)

    elif cfg["model_type"] == "blip3o":
        # Actual BLIP3o conditioning pathway: embed text, append latent_queries,
        # run through the SAME LLM (with edited MLP layers), extract last N_QUERY positions
        inputs = tok(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        # Append the special end token (151665) as in the actual generate_image()
        end_token = torch.tensor([[151665]], device=device)
        input_ids = torch.cat([input_ids, end_token], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=device, dtype=attn_mask.dtype)], dim=1)

        # Get text embeddings
        text_embeds = model.get_model().embed_tokens(input_ids)

        # Append latent queries (these pass through the same edited LLM layers)
        latent_queries = model.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)
        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones(1, latent_queries.shape[1], device=device, dtype=attn_mask.dtype)], dim=1)

        # Forward through the inner model (same LLM with edited layers 6-10)
        outputs = model.model(
            inputs_embeds=text_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Extract last N_QUERY positions — this IS the conditioning signal
        n_query = model.get_n_query()
        hidden_states = outputs.hidden_states[-1][:, -n_query:, :]  # (1, N_QUERY, 2048)
        return hidden_states[0]  # (N_QUERY, 2048)

    else:
        # OmniGen2: run text through LLM, return hidden_states[-1] for all text tokens
        # The actual pipeline passes these directly to the DiT (no projection bottleneck)
        inputs = tok(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        h_last = outputs.hidden_states[-1]
        return h_last[0]  # (seq, hidden_size)


@torch.inference_mode()
def get_all_hidden_states(model, tok, prompt: str, model_name: str, device: str = "cuda") -> List[torch.Tensor]:
    """Get hidden states at all layers for the actual conditioning pathway.
    Returns list of (seq, hidden_size) tensors. For BLIP3o, returns states at the
    N_QUERY positions (the actual conditioning tokens)."""
    cfg = MODEL_CONFIGS[model_name]

    if cfg["model_type"] == "ovis":
        llm = get_llm(model)
        inputs = tok(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        outputs = llm(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        return [h[0].float().cpu() for h in outputs.hidden_states]

    elif cfg["model_type"] == "blip3o":
        inputs = tok(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        end_token = torch.tensor([[151665]], device=device)
        input_ids = torch.cat([input_ids, end_token], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones(1, 1, device=device, dtype=attn_mask.dtype)], dim=1)
        text_embeds = model.get_model().embed_tokens(input_ids)
        latent_queries = model.get_model().latent_queries.repeat(text_embeds.shape[0], 1, 1)
        text_embeds = torch.cat([text_embeds, latent_queries], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones(1, latent_queries.shape[1], device=device, dtype=attn_mask.dtype)], dim=1)
        outputs = model.model(
            inputs_embeds=text_embeds, attention_mask=attn_mask,
            output_hidden_states=True, return_dict=True,
        )
        n_query = model.get_n_query()
        return [h[0, -n_query:, :].float().cpu() for h in outputs.hidden_states]

    else:
        # OmniGen2
        inputs = tok(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        return [h[0].float().cpu() for h in outputs.hidden_states]


class ActivationCollector:
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.activations = {}
        self.hooks = []

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook

    def __enter__(self):
        self.activations = {}
        for name in self.layer_names:
            try:
                module = get_module(self.model, name)
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
            except LookupError as e:
                print(f"Warning: {e}")
        return self

    def __exit__(self, *args):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        return False


class ActivationPatcher:
    def __init__(self, model: nn.Module, patches: Dict[str, torch.Tensor]):
        """patches: dict mapping layer_name -> replacement activation tensor"""
        self.model = model
        self.patches = patches
        self.hooks = []

    def _make_patch_hook(self, replacement: torch.Tensor):
        def hook(module, input, output):
            if isinstance(output, tuple):
                return (replacement.to(output[0].device),) + output[1:]
            return replacement.to(output.device)
        return hook

    def __enter__(self):
        for name, replacement in self.patches.items():
            try:
                module = get_module(self.model, name)
                hook = module.register_forward_hook(self._make_patch_hook(replacement))
                self.hooks.append(hook)
            except LookupError as e:
                print(f"Warning: {e}")
        return self

    def __exit__(self, *args):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        return False


def save_results(results: dict, exp_name: str, model_name: str, method: str):
    """Save results JSON to interpretability/results/<exp_name>/"""
    out_dir = SCRIPT_DIR / "results" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = model_name.replace("/", "_")
    out_path = out_dir / f"{slug}_{method}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    print(f"Saved: {out_path}")
    return out_path


def print_table(headers: List[str], rows: List[List], title: str = ""):
    """Print a formatted ASCII table."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]))
    print()
