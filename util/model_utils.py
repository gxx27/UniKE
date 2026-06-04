"""
Model-specific utilities (loading, LLM-caller resolution, type detection) for multi-model knowledge editing.
"""

import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent


def get_model_type(model_name: str) -> str:
    """Determine the model type from the model name string."""
    name_lower = model_name.lower()
    if "ovis" in name_lower:
        return "ovis"
    elif "blip3o" in name_lower:
        return "blip3o"
    elif "omnigen2" in name_lower:
        return "omnigen2"
    else:
        return "generic"


def get_llm_caller(model):
    """
    Return the LLM sub-module used for forward passes (logits computation).

    For Ovis: model.llm
    For BLIP3o / OmniGen2 / generic: model itself
    """
    if hasattr(model, "llm"):
        return model.llm
    if hasattr(model, "language_model"):
        return model.language_model
    return model


def is_shared_visual_model(model) -> bool:
    """
    Detect models whose text backbone also owns an integrated visual tower.

    These shared VL backbones (e.g. Qwen2.5-VL variants) need more conservative
    AlphaEdit updates because the edited transformer layers are reused across
    text and vision-conditioned pathways.
    """
    caller = get_llm_caller(model)
    if caller is not model:
        return False

    inner_model = getattr(model, "model", None)
    return hasattr(model, "visual") or hasattr(inner_model, "visual")


def _ensure_blip3o_path():
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _ensure_omnigen2_path():
    repo_root = str(REPO_ROOT)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def load_model_and_tok(model_name: str, device: str = "cuda", dtype=None):
    """
    Load model and tokenizer for any supported model type.

    Returns:
        (model, tokenizer)
    """
    model_type = get_model_type(model_name)

    if model_type == "ovis":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)
        tok = model.get_text_tokenizer()
        model.to(dtype=torch.bfloat16)

    elif model_type == "blip3o":
        _ensure_blip3o_path()
        from transformers import AutoConfig
        from BLIP3o.model import blip3oQwenForInferenceLM
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if dtype is None:
            dtype = torch.bfloat16
        model = blip3oQwenForInferenceLM.from_pretrained(
            model_name, config=config, torch_dtype=dtype,
            low_cpu_mem_usage=True, trust_remote_code=True
        ).to(device)
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    elif model_type == "omnigen2":
        from transformers import Qwen2_5_VLForConditionalGeneration
        if dtype is None:
            dtype = torch.bfloat16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, subfolder="mllm",
            torch_dtype=dtype, trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
        tok = AutoTokenizer.from_pretrained(
            model_name, subfolder="mllm_processor", trust_remote_code=True
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tok = AutoTokenizer.from_pretrained(model_name)

    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return model, tok


def apply_chat_template(model_name: str, tok, prompt: str) -> str:
    """
    Apply a chat template to a prompt for models that need it for reasoning.
    Returns the formatted prompt string ready for tokenization.
    """
    model_type = get_model_type(model_name)

    if model_type == "blip3o":
        _ensure_blip3o_path()
        from BLIP3o.conversation import conv_templates
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    elif model_type == "omnigen2":
        messages = [{"role": "user", "content": prompt}]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    else:
        return prompt


def generate_text(model, tok, prompt: str, model_name: str = "",
                  max_new_tokens: int = 512, use_chat_template: bool = True) -> str:
    """
    Generate text from a model using the appropriate method.
    Works for all supported model types.
    """
    model_type = get_model_type(model_name or model.config._name_or_path)

    if use_chat_template:
        formatted = apply_chat_template(
            model_name or model.config._name_or_path, tok, prompt
        )
    else:
        formatted = prompt

    inputs = tok(formatted, return_tensors="pt", padding=True).to(
        next(model.parameters()).device
    )

    caller = get_llm_caller(model)

    with torch.inference_mode():
        output_ids = caller.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)
