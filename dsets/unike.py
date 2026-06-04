import json
import re
import typing
from pathlib import Path

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from util.globals import *


def create_prompt_template(prompt: str, subject: str) -> str:
    """
    Create a prompt template by replacing the subject with {} placeholder.
    Uses case-insensitive matching to handle variations like:
    - Subject: "Swimming Pool" 
    - In prompt: "swimming pool"
    
    Args:
        prompt: The complete prompt (e.g., "The color of the swimming pool is")
        subject: The subject to replace (e.g., "Swimming Pool")
    
    Returns:
        Prompt template with {} placeholder (e.g., "The color of the {} is")
    """
    # Escape special regex characters in subject
    escaped_subject = re.escape(subject)

    # Case-insensitive replacement
    # Use a function to preserve a single {} placeholder
    pattern = re.compile(escaped_subject, re.IGNORECASE)

    # Check if subject appears in prompt
    if pattern.search(prompt):
        # Replace first occurrence only (in case subject appears multiple times)
        # Any pre-existing literal braces in the remainder must be escaped so
        # the resulting template can be passed to str.format safely.
        head, sep, tail = prompt.partition(pattern.search(prompt).group(0))
        safe_head = head.replace("{", "{{").replace("}", "}}")
        safe_tail = tail.replace("{", "{{").replace("}", "}}")
        return safe_head + "{}" + safe_tail

    # Subject not found literally — but the prompt may already be
    # pre-templated (e.g. "The capital of {} is" with subject
    # "United States of America"). If it contains exactly one "{}" token
    # and no other single-brace sequences, treat it as the template itself.
    if prompt.count("{}") == 1 and prompt.count("{") == 1 and prompt.count("}") == 1:
        return prompt

    # True edge case: subject missing and prompt is not pre-templated.
    # Escape any stray braces to avoid str.format() errors downstream,
    # then append a single {} placeholder.
    print(f"  Warning: Subject '{subject}' not found in prompt '{prompt}'. Appending {{}} at end.")
    safe_prompt = prompt.replace("{", "{{").replace("}", "}}")
    return safe_prompt + " {}"


class UniKEDataset(Dataset):
    """
    Dataset for UniKE cross-modal knowledge editing.

    Attribute data (source="attribute"):
        - Categories: color, material, pattern, shape, size
        - Stages: stage_1, stage_2, stage_3, stage_4

    Relation data (source="relation"):
        - Categories: affiliation, creator, location, occupation
        - A single stage_1 edit per record.
    """

    ATTRIBUTE_CATEGORIES = ["color", "material", "pattern", "shape", "size"]
    RELATION_CATEGORIES = ["affiliation", "creator", "location", "occupation"]

    DATA_FILENAME = "UniKE.json"

    def __init__(
        self,
        data_dir: str,
        tok: AutoTokenizer = None,
        size: typing.Optional[int] = None,
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)
        data_loc = data_dir / self.DATA_FILENAME

        with open(data_loc, "r") as f:
            raw = json.load(f)

        data = []
        attribute_count = 0
        relation_count = 0
        dropped = 0

        for i, record in enumerate(raw):
            subject = record["subject"]
            category = record.get("category", "")
            source = record.get("source", "")

            if source == "attribute":
                processed = self._process_attribute_record(i, record, subject, category)
                if processed:
                    data.append(processed)
                    attribute_count += 1
                else:
                    dropped += 1
            elif source == "relation":
                processed = self._process_relation_record(i, record, subject, category)
                if processed:
                    data.append(processed)
                    relation_count += 1
                else:
                    dropped += 1
            else:
                dropped += 1

        self._data = data[:size] if size is not None else data
        print(f"Loaded UniKEDataset with {len(self)} elements "
              f"(attribute: {min(attribute_count, size) if size else attribute_count}, "
              f"relation: {min(relation_count, size) if size else relation_count}; "
              f"dropped {dropped} malformed records)")

    def _process_attribute_record(self, idx: int, record: dict, subject: str, category: str) -> dict:
        """Process attribute-based record with stages."""
        # Extract all stages
        stages = {}
        for stage_key in ["stage_1", "stage_2", "stage_3", "stage_4"]:
            if stage_key in record:
                stage = record[stage_key]
                stages[stage_key] = {
                    "question": stage.get("question", ""),
                    "prompt": stage.get("prompt", ""),
                    "gt": stage.get("gt", ""),
                    "gt_target": stage.get("gt_target", ""),
                    "image_prompt": stage.get("image_prompt", ""),
                    "visual_target": stage.get("visual_target", ""),
                    "vqa_question": stage.get("vqa_question", ""),
                }
        
        # Primary rewrite from stage_1
        stage_1 = stages.get("stage_1", {})
        prompt = stage_1.get("prompt", "")
        target_new = stage_1.get("gt_target", "")
        target_true = stage_1.get("gt", "")
        
        if not prompt or not target_new:
            return None

        # Create prompt template by replacing subject with {} placeholder
        prompt_template = create_prompt_template(prompt, subject)
        
        # Collect paraphrase prompts from other stages
        paraphrase_prompts = []
        for stage_key in ["stage_2", "stage_3", "stage_4"]:
            if stage_key in stages:
                stage_prompt = stages[stage_key].get("prompt", "")
                if stage_prompt:
                    paraphrase_prompts.append(stage_prompt)
        
        return {
            "case_id": idx,
            "source": "attribute",
            "category": category,
            "subject": subject,
            "requested_rewrite": {
                "prompt": prompt_template,
                "subject": subject,
                "target_new": {"str": target_new},
                "target_true": {"str": target_true},
            },
            "paraphrase_prompts": paraphrase_prompts,
            "neighborhood_prompts": [],
            "attribute_prompts": [],
            "generation_prompts": [],
            # Full stage data for downstream evaluation
            "stages": stages,
        }

    def _process_relation_record(self, idx: int, record: dict, subject: str, category: str) -> dict:
        """Process relation-based record (single stage_1 edit)."""
        stages = {}
        for stage_key in ["stage_1"]:
            if stage_key in record:
                stage = record[stage_key]
                stages[stage_key] = {
                    "question": stage.get("question", ""),
                    "prompt": stage.get("prompt", ""),
                    "gt": stage.get("gt", ""),
                    "gt_target": stage.get("gt_target", ""),
                    "image_prompt": stage.get("image_prompt", ""),
                    "visual_target": stage.get("visual_target", ""),
                    "vqa_question": stage.get("vqa_question", ""),
                }

        stage_1 = stages.get("stage_1", {})
        prompt = stage_1.get("prompt", "")
        target_new = stage_1.get("gt_target", "")
        target_true = stage_1.get("gt", "")

        if not prompt or not target_new:
            return None

        prompt_template = create_prompt_template(prompt, subject)

        return {
            "case_id": idx,
            "source": "relation",
            "category": category,
            "subject": subject,
            "requested_rewrite": {
                "prompt": prompt_template,
                "subject": subject,
                "target_new": {"str": target_new},
                "target_true": {"str": target_true},
            },
            "paraphrase_prompts": [],
            "neighborhood_prompts": [],
            "attribute_prompts": [],
            "generation_prompts": [],
            # Full stage data for downstream evaluation
            "stages": stages,
        }

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]
    
    def get_attribute_records(self) -> list:
        """Return only attribute-based records."""
        return [r for r in self._data if r["source"] == "attribute"]
    
    def get_relation_records(self) -> list:
        """Return only relation-based records."""
        return [r for r in self._data if r["source"] == "relation"]
    
    def get_records_by_category(self, category: str) -> list:
        """Return records filtered by category."""
        return [r for r in self._data if r["category"] == category]
