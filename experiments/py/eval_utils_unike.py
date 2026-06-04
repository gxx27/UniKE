"""
Efficacy evaluation utilities for the UniKE dataset.
"""

import typing

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.model_utils import get_llm_caller


def compute_rewrite_quality_unike(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips=None,  # Unused, kept for interface compatibility
    vec=None,    # Unused, kept for interface compatibility
) -> typing.Dict:
    """
    Computes Efficacy metric for UniKE cross-modal knowledge editing.

    Handles both attribute and relation data (both keyed by stages).

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: UniKE dataset record
    :return: Dictionary containing efficacy metrics
    """
    source = record.get("source", "")
    
    if source == "attribute":
        return _compute_attribute_efficacy(model, tok, record)
    elif source == "relation":
        return _compute_relation_efficacy(model, tok, record)
    else:
        return {"error": f"Unknown source: {source}"}


def _compute_attribute_efficacy(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
) -> typing.Dict:
    """
    Computes Efficacy for attribute-based records.
    
    Evaluates each stage (stage_1, stage_2, stage_3, stage_4) and computes:
    - Individual stage efficacy
    - stage_n: 1 if any stage has efficacy=1
    """
    subject = record["requested_rewrite"]["subject"]
    target_new = record["requested_rewrite"]["target_new"]["str"]
    stages = record.get("stages", {})
    category = record.get("category", "")
    
    # Results dictionary
    results = {
        "source": "attribute",
        "category": category,
        "stage_efficacy": {},
    }
    
    # Evaluate each stage
    any_correct = False
    for stage_key in ["stage_1", "stage_2", "stage_3", "stage_4"]:
        if stage_key in stages:
            stage = stages[stage_key]
            prompt = stage.get("prompt", "")
            
            if prompt:
                # Stage's gt_target should be the same as the main target_new for stage_1
                # For other stages, they test generalization to the same target
                stage_target = stage.get("gt_target", target_new)
                efficacy_correct = test_prediction_acc(model, tok, prompt, stage_target)
                results["stage_efficacy"][stage_key] = efficacy_correct
                if efficacy_correct:
                    any_correct = True
            else:
                results["stage_efficacy"][stage_key] = False
    
    # stage_n: 1 if any stage is correct
    results["stage_n_efficacy"] = any_correct
    
    # Primary efficacy is stage_1
    results["efficacy_correct"] = results["stage_efficacy"].get("stage_1", False)
    
    return results


def _compute_relation_efficacy(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
) -> typing.Dict:
    """Computes Efficacy for relation-based records (single stage_1 edit)."""
    target_new = record["requested_rewrite"]["target_new"]["str"]
    stages = record.get("stages", {})
    category = record.get("category", "")

    results = {
        "source": "relation",
        "category": category,
        "stage_efficacy": {},
    }

    stage_1 = stages.get("stage_1", {})
    stage_1_prompt = stage_1.get("prompt", "")
    stage_1_target = stage_1.get("gt_target", target_new)

    if stage_1_prompt:
        stage_1_correct = test_prediction_acc(model, tok, stage_1_prompt, stage_1_target)
    else:
        stage_1_correct = False
    results["stage_efficacy"]["stage_1"] = stage_1_correct
    results["efficacy_correct"] = stage_1_correct

    return results


def test_prediction_acc(model, tok, prompt: str, target: str) -> bool:
    """
    Test if model's next token prediction matches target.
    
    :param model: Model to test
    :param tok: Tokenizer
    :param prompt: Input prompt
    :param target: Expected target string
    :return: True if prediction matches target
    """
    prompt_tok = tok(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        caller = get_llm_caller(model)
        logits = caller(**prompt_tok).logits
        # Get logits for last token
        last_logits = logits[0, -1, :]
        predicted_id = torch.argmax(last_logits).item()
        
        # Get target token id (skip BOS if the tokenizer prepends one)
        target_tok = tok(" " + target)["input_ids"]
        if tok.bos_token_id is not None and target_tok[0] == tok.bos_token_id and len(target_tok) > 1:
            target_id = target_tok[1]
        else:
            target_id = target_tok[0]
        
        return predicted_id == target_id


def compute_efficacy_summary(results_list: typing.List[typing.Dict]) -> typing.Dict:
    """
    Compute summary statistics from a list of efficacy results.
    
    Returns breakdown by:
    - Overall efficacy
    - Per-source (attribute vs relation)
    - Per-category
    - Per-stage (stage_1..4 for attribute, stage_1 for relation)
    """
    summary = {
        "total": len(results_list),
        "overall_efficacy": 0,
        "by_source": {
            "attribute": {"correct": 0, "total": 0},
            "relation": {"correct": 0, "total": 0},
        },
        "by_category": {},
        "attribute_stages": {
            "stage_1": {"correct": 0, "total": 0},
            "stage_2": {"correct": 0, "total": 0},
            "stage_3": {"correct": 0, "total": 0},
            "stage_4": {"correct": 0, "total": 0},
            "stage_n": {"correct": 0, "total": 0},
        },
        "relation_stages": {
            "stage_1": {"correct": 0, "total": 0},
        },
    }
    
    correct_count = 0
    
    for result in results_list:
        source = result.get("source", "")
        category = result.get("category", "")
        efficacy_correct = result.get("efficacy_correct", False)
        
        # Overall
        if efficacy_correct:
            correct_count += 1
        
        # By source
        if source in summary["by_source"]:
            summary["by_source"][source]["total"] += 1
            if efficacy_correct:
                summary["by_source"][source]["correct"] += 1
        
        # By category
        if category not in summary["by_category"]:
            summary["by_category"][category] = {"correct": 0, "total": 0}
        summary["by_category"][category]["total"] += 1
        if efficacy_correct:
            summary["by_category"][category]["correct"] += 1
        
        # Attribute stage breakdown
        if source == "attribute":
            stage_efficacy = result.get("stage_efficacy", {})
            for stage_key in ["stage_1", "stage_2", "stage_3", "stage_4"]:
                if stage_key in stage_efficacy:
                    summary["attribute_stages"][stage_key]["total"] += 1
                    if stage_efficacy[stage_key]:
                        summary["attribute_stages"][stage_key]["correct"] += 1
            
            # stage_n
            summary["attribute_stages"]["stage_n"]["total"] += 1
            if result.get("stage_n_efficacy", False):
                summary["attribute_stages"]["stage_n"]["correct"] += 1
        
        # Relation stage breakdown
        elif source == "relation":
            stage_efficacy = result.get("stage_efficacy", {})
            for stage_key in ["stage_1"]:
                if stage_key in stage_efficacy:
                    summary["relation_stages"][stage_key]["total"] += 1
                    if stage_efficacy[stage_key]:
                        summary["relation_stages"][stage_key]["correct"] += 1
    
    # Calculate rates
    summary["overall_efficacy"] = correct_count / len(results_list) if results_list else 0
    
    # Calculate rates for each breakdown
    for key in summary["by_source"]:
        data = summary["by_source"][key]
        data["rate"] = data["correct"] / data["total"] if data["total"] > 0 else 0
    
    for key in summary["by_category"]:
        data = summary["by_category"][key]
        data["rate"] = data["correct"] / data["total"] if data["total"] > 0 else 0
    
    for key in summary["attribute_stages"]:
        data = summary["attribute_stages"][key]
        data["rate"] = data["correct"] / data["total"] if data["total"] > 0 else 0
    
    for key in summary["relation_stages"]:
        data = summary["relation_stages"][key]
        data["rate"] = data["correct"] / data["total"] if data["total"] > 0 else 0
    
    return summary
