from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook
from util.model_utils import get_llm_caller

from .pmet_hparams import PMETHyperParams


def compute_zs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: PMETHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vectors for both attention and MLP for the rank-1 update.
    Runs a simple optimization procedure.
    """
    
    caller = get_llm_caller(model)

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_module(model, f"{hparams.lm_head_module}").weight.T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(caller.config.vocab_size)

    print("Computing right vectors (v) for ATTN and MLP")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Cast to float32 for optimization precision (bfloat16 models lose small deltas)
    original_dtype = next(caller.parameters()).dtype
    needs_float32 = original_dtype != torch.float32
    if needs_float32:
        print(f"  Upcasting LLM from {original_dtype} to float32 for compute_zs")
        caller.float()
        lm_w, ln_f = (
            nethook.get_module(model, f"{hparams.lm_head_module}").weight.T,
            nethook.get_module(model, hparams.ln_f_module),
        )

    # Set up an optimization over latent vectors for both attn and mlp
    if hasattr(model.config, 'n_embd'):
        hidden_size = model.config.n_embd
    elif hasattr(model.config, 'hidden_size'):
        hidden_size = model.config.hidden_size
    elif hasattr(caller.config, 'hidden_size'):
        hidden_size = caller.config.hidden_size
    else:
        raise NotImplementedError("Cannot determine hidden size")
    
    delta_attn = torch.zeros((hidden_size,), requires_grad=True, device="cuda")
    delta_mlp = torch.zeros((hidden_size,), requires_grad=True, device="cuda")
    target_init_attn, target_init_mlp, kl_distr_init = None, None, None

    _DELTA_SCALE_REF = 100.0
    _MAX_DELTA_SCALE = 5.0
    delta_scale_mlp = 1.0
    delta_scale_attn = 1.0

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init_attn, target_init_mlp, delta_scale_mlp, delta_scale_attn

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            if target_init_mlp is None:
                print("Recording initial value of v* in mlp")
                target_init_mlp = cur_out[0, lookup_idxs[0]].detach().clone()
                init_norm = target_init_mlp.norm().item()
                delta_scale_mlp = min(max(1.0, init_norm / _DELTA_SCALE_REF), _MAX_DELTA_SCALE)
                if delta_scale_mlp > 1.0:
                    print(f"  Large MLP init_norm={init_norm:.1f}, delta_scale={delta_scale_mlp:.1f}")

            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta_mlp * delta_scale_mlp
                    
        if cur_layer == hparams.attn_module_tmp.format(layer):
            if target_init_attn is None:
                print("Recording initial value of v* in attn")
                target_init_attn = cur_out[0, lookup_idxs[0]].detach().clone()
                init_norm = target_init_attn.norm().item()
                delta_scale_attn = min(max(1.0, init_norm / _DELTA_SCALE_REF), _MAX_DELTA_SCALE)
                if delta_scale_attn > 1.0:
                    print(f"  Large ATTN init_norm={init_norm:.1f}, delta_scale={delta_scale_attn:.1f}")

            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta_attn * delta_scale_attn
        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta_mlp, delta_attn], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    nll_loss_factor = hparams.nll_loss_factor
    kl_factor = hparams.kl_factor

    # Execute optimization (use while loop to allow adaptive step count)
    import math
    max_steps = hparams.v_num_grad_steps
    it = 0
    while it < max_steps:
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
                hparams.attn_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            caller = get_llm_caller(model)
            logits = caller(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # After first forward pass, adapt optimizer for large init_norm
        if it == 0 and max(delta_scale_mlp, delta_scale_attn) > 1.0:
            ds = max(delta_scale_mlp, delta_scale_attn)
            lr_peak = hparams.v_lr * min(50.0, ds ** 0.35)
            for g in opt.param_groups:
                g['lr'] = lr_peak
            max_steps = min(100, max(hparams.v_num_grad_steps, int(math.sqrt(ds) * 3)))
            print(f"  Adaptive opt: lr_peak={lr_peak:.4f}, max_steps={max_steps}")
        elif max(delta_scale_mlp, delta_scale_attn) > 1.0 and it > 0:
            frac = it / max_steps
            lr_cur = lr_peak * (0.5 * (1 + math.cos(math.pi * frac)) * 0.9 + 0.1)
            for g in opt.param_groups:
                g['lr'] = lr_cur

        # Compute loss on rewriting targets
        output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        if output.shape[1] != rewriting_targets.shape[1]:
            output = torch.transpose(output, 0, 1)
        full_repr = output[:len(rewriting_prompts)]
        
        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        max_probs = torch.max(log_probs, dim=2)[0]
        max_prob = torch.exp((max_probs * mask.to(max_probs.device)).sum(1) / target_ids.size(0)).mean().item()
        
        # Aggregate total losses
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_factor * nll_loss_each.mean()
        kl_loss = kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            (torch.norm(delta_mlp * delta_scale_mlp) / torch.norm(target_init_mlp)) ** 2
            + (torch.norm(delta_attn * delta_scale_attn) / torch.norm(target_init_attn)) ** 2
        )
        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        prob = torch.exp(-nll_loss_each).mean().item()
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{prob}"
        )
        if loss < 5e-2:
            break
        if max_prob == prob:
            nll_loss_factor = 0.1 * hparams.nll_loss_factor
            if kl_loss <= 0.01:
                break
        else:
            nll_loss_factor = hparams.nll_loss_factor
            
        if it == max_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball (clamp the effective delta)
        capped_ref_mlp = min(target_init_mlp.norm().item(), _DELTA_SCALE_REF * _MAX_DELTA_SCALE)
        max_norm = hparams.clamp_norm_factor * capped_ref_mlp
        eff_mlp_norm = (delta_mlp * delta_scale_mlp).norm()
        if eff_mlp_norm > max_norm:
            with torch.no_grad():
                delta_mlp[...] = delta_mlp * max_norm / eff_mlp_norm

        capped_ref_attn = min(target_init_attn.norm().item(), _DELTA_SCALE_REF * _MAX_DELTA_SCALE)
        max_norm = hparams.clamp_norm_factor * capped_ref_attn
        eff_attn_norm = (delta_attn * delta_scale_attn).norm()
        if eff_attn_norm > max_norm:
            with torch.no_grad():
                delta_attn[...] = delta_attn * max_norm / eff_attn_norm

        it += 1

    target_attn = target_init_attn + delta_attn * delta_scale_attn
    target_mlp = target_init_mlp + delta_mlp * delta_scale_mlp
    print(
        f"[ATTN]: Init norm {target_init_attn.norm()} | Delta norm {(delta_attn * delta_scale_attn).norm()} | Target norm {target_attn.norm()}",
        f"[MLP]: Init norm {target_init_mlp.norm()} | Delta norm {(delta_mlp * delta_scale_mlp).norm()} | Target norm {target_mlp.norm()}",
    )

    if needs_float32:
        caller.to(original_dtype)

    return target_attn, target_mlp


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
