#!/usr/bin/env python3
"""
VQA evaluation of generated images via a remote VLM through the OpenRouter API.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import random
import re
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx
except ImportError as e:  # pragma: no cover - exercised when env missing dep
    sys.stderr.write(
        "[fatal] httpx is required. Install with `pip install httpx`.\n"
    )
    raise

try:
    from PIL import Image
except ImportError as e:  # pragma: no cover
    sys.stderr.write(
        "[fatal] pillow is required. Install with `pip install pillow`.\n"
    )
    raise


# ---------------------------------------------------------------------------
# Prompt + parser -- kept byte-for-byte identical to the offline evaluator so
# the resulting vqa_results.json schema is interchangeable.
# ---------------------------------------------------------------------------
def create_vqa_prompt(vqa_question: str, visual_target: str) -> str:
    return f"""Look at this image carefully and answer the following question.

Question: {vqa_question}

Target Criterion (The image MUST satisfy this statement): {visual_target}

Your task:
1. Describe what you observe in the image related to the question.
2. Determine if the image STRICTLY satisfies the Target Criterion.
   - If the Target Criterion says "must be X", and the image shows Y, then it does NOT match.
   - Be rigorous. The image must clearly demonstrate the target property.
3. Respond ONLY with a JSON object in this exact format:
{{
    "observation": "describe what you see in the image",
    "matches_target": true or false,
    "confidence": "high", "medium", or "low",
    "explanation": "why you think it matches or doesn't match the expected target"
}}

Respond ONLY with the JSON object, no other text."""


def parse_vlm_response(response_text: str) -> Dict[str, Any]:
    try:
        text = (response_text or "").strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        result = json.loads(text)
        return {
            "vqa_correct": bool(result.get("matches_target", False)),
            "confidence": result.get("confidence", "unknown"),
            "observation": result.get("observation", ""),
            "explanation": result.get("explanation", ""),
            "raw_response": response_text,
            "parse_success": True,
        }
    except (json.JSONDecodeError, ValueError):
        lower = (response_text or "").lower()
        guessed = "true" in lower and "matches" in lower
        return {
            "vqa_correct": guessed,
            "confidence": "low",
            "observation": "",
            "explanation": "",
            "raw_response": response_text,
            "parse_success": False,
        }


# ---------------------------------------------------------------------------
# Reasoning / summary helpers
# ---------------------------------------------------------------------------
def load_reasoning_results(reasoning_dir: str) -> Dict[int, Dict]:
    reasoning_dir = Path(reasoning_dir)
    all_results: Dict[int, Dict] = {}
    for json_file in reasoning_dir.glob("reasoning_*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[warn] failed to read {json_file}: {e}", file=sys.stderr)
            continue
        for item in data:
            case_id = item.get("case_id")
            if case_id is not None:
                all_results[case_id] = item
    return all_results


def build_evaluation_results(
    dataset: List[Dict],
    vqa_results: Dict[str, Dict],
    reasoning_results: Dict[int, Dict],
    start_idx: int,
    end_idx: int,
) -> List[Dict]:
    eval_results: List[Dict] = []
    dataset_slice = dataset[start_idx:end_idx]
    for i, item in enumerate(dataset_slice):
        global_idx = item.get("case_id", start_idx + i)
        source = item.get("source", "")
        subject = item.get("subject", "")
        category = item.get("category", "")
        reasoning_data = reasoning_results.get(global_idx, {})
        if source == "attribute":
            item_eval = {
                "case_id": global_idx, "subject": subject,
                "category": category, "source": "attribute", "stages": {},
            }
            stages_reasoning = reasoning_data.get("stages", {})
            for stage_key in ["stage_1", "stage_2", "stage_3", "stage_4"]:
                if stage_key not in item:
                    continue
                gt_target = item[stage_key].get("gt_target", "")
                key = f"case_{global_idx}_{stage_key}"
                stage_eval = {
                    "gt_target": gt_target,
                    "vqa": vqa_results.get(
                        key, {"vqa_correct": False, "error": "No VQA result found"}
                    ),
                }
                stage_reasoning = stages_reasoning.get(stage_key, {})
                stage_eval["reasoning"] = {
                    "contains_target": stage_reasoning.get("contains_target", False),
                    "reasoning_snippet": (stage_reasoning.get("reasoning_output") or "")[:300],
                }
                item_eval["stages"][stage_key] = stage_eval
            eval_results.append(item_eval)
        elif source == "relation":
            item_eval = {
                "case_id": global_idx, "subject": subject,
                "category": category, "source": "relation", "stages": {},
            }
            stages_reasoning = reasoning_data.get("stages", {})
            for stage_key in ["stage_1"]:
                if stage_key not in item:
                    continue
                gt_target = item[stage_key].get("gt_target", "")
                key = f"case_{global_idx}_{stage_key}"
                stage_eval = {
                    "gt_target": gt_target,
                    "vqa": vqa_results.get(
                        key, {"vqa_correct": False, "error": "No VQA result found"}
                    ),
                }
                stage_reasoning = stages_reasoning.get(stage_key, {})
                stage_eval["reasoning"] = {
                    "contains_target": stage_reasoning.get("contains_target", False),
                    "reasoning_snippet": (stage_reasoning.get("reasoning_output") or "")[:300],
                }
                item_eval["stages"][stage_key] = stage_eval
            eval_results.append(item_eval)
    return eval_results


def compute_summary(eval_results: List[Dict]) -> Dict:
    summary = {
        "total_items": len(eval_results),
        "attribute": {
            "total": 0,
            "stages": {sk: {"vqa_correct": 0, "vqa_total": 0,
                            "reasoning_correct": 0, "reasoning_total": 0}
                       for sk in ["stage_1", "stage_2", "stage_3", "stage_4", "stage_n"]},
            "categories": defaultdict(lambda: {"vqa_correct": 0, "vqa_total": 0,
                                               "reasoning_correct": 0, "reasoning_total": 0}),
        },
        "relation": {
            "total": 0,
            "stages": {sk: {"vqa_correct": 0, "vqa_total": 0,
                            "reasoning_correct": 0, "reasoning_total": 0}
                       for sk in ["stage_1"]},
            "categories": defaultdict(lambda: {"vqa_correct": 0, "vqa_total": 0,
                                               "reasoning_correct": 0, "reasoning_total": 0}),
        },
    }
    for result in eval_results:
        source = result.get("source", "")
        category = result.get("category", "")
        if source == "attribute":
            summary["attribute"]["total"] += 1
            stages = result.get("stages", {})
            any_vqa = False
            any_rea = False
            for sk in ["stage_1", "stage_2", "stage_3", "stage_4"]:
                if sk not in stages:
                    continue
                stage = stages[sk]
                vqa = stage.get("vqa", {})
                if "error" not in vqa and "skipped_reason" not in vqa:
                    summary["attribute"]["stages"][sk]["vqa_total"] += 1
                    if vqa.get("vqa_correct", False):
                        summary["attribute"]["stages"][sk]["vqa_correct"] += 1
                        any_vqa = True
                rea = stage.get("reasoning", {})
                summary["attribute"]["stages"][sk]["reasoning_total"] += 1
                if rea.get("contains_target", False):
                    summary["attribute"]["stages"][sk]["reasoning_correct"] += 1
                    any_rea = True
            summary["attribute"]["stages"]["stage_n"]["vqa_total"] += 1
            summary["attribute"]["stages"]["stage_n"]["reasoning_total"] += 1
            if any_vqa:
                summary["attribute"]["stages"]["stage_n"]["vqa_correct"] += 1
            if any_rea:
                summary["attribute"]["stages"]["stage_n"]["reasoning_correct"] += 1
            if "stage_1" in stages:
                s1 = stages["stage_1"]
                vqa = s1.get("vqa", {}); rea = s1.get("reasoning", {})
                if "error" not in vqa and "skipped_reason" not in vqa:
                    summary["attribute"]["categories"][category]["vqa_total"] += 1
                    if vqa.get("vqa_correct", False):
                        summary["attribute"]["categories"][category]["vqa_correct"] += 1
                summary["attribute"]["categories"][category]["reasoning_total"] += 1
                if rea.get("contains_target", False):
                    summary["attribute"]["categories"][category]["reasoning_correct"] += 1
        elif source == "relation":
            summary["relation"]["total"] += 1
            stages = result.get("stages", {})
            for sk in ["stage_1"]:
                if sk not in stages:
                    continue
                stage = stages[sk]
                vqa = stage.get("vqa", {})
                if "error" not in vqa and "skipped_reason" not in vqa:
                    summary["relation"]["stages"][sk]["vqa_total"] += 1
                    if vqa.get("vqa_correct", False):
                        summary["relation"]["stages"][sk]["vqa_correct"] += 1
                rea = stage.get("reasoning", {})
                if "skipped_reason" not in rea:
                    summary["relation"]["stages"][sk]["reasoning_total"] += 1
                    if rea.get("contains_target", False):
                        summary["relation"]["stages"][sk]["reasoning_correct"] += 1
            if "stage_1" in stages:
                s1 = stages["stage_1"]
                vqa = s1.get("vqa", {}); rea = s1.get("reasoning", {})
                if "error" not in vqa and "skipped_reason" not in vqa:
                    summary["relation"]["categories"][category]["vqa_total"] += 1
                    if vqa.get("vqa_correct", False):
                        summary["relation"]["categories"][category]["vqa_correct"] += 1
                summary["relation"]["categories"][category]["reasoning_total"] += 1
                if rea.get("contains_target", False):
                    summary["relation"]["categories"][category]["reasoning_correct"] += 1
    for sk in summary["attribute"]["stages"]:
        d = summary["attribute"]["stages"][sk]
        d["vqa_rate"] = d["vqa_correct"] / d["vqa_total"] if d["vqa_total"] > 0 else 0
        d["reasoning_rate"] = d["reasoning_correct"] / d["reasoning_total"] if d["reasoning_total"] > 0 else 0
    for sk in summary["relation"]["stages"]:
        d = summary["relation"]["stages"][sk]
        d["vqa_rate"] = d["vqa_correct"] / d["vqa_total"] if d["vqa_total"] > 0 else 0
        d["reasoning_rate"] = d["reasoning_correct"] / d["reasoning_total"] if d["reasoning_total"] > 0 else 0
    summary["attribute"]["categories"] = dict(summary["attribute"]["categories"])
    summary["relation"]["categories"] = dict(summary["relation"]["categories"])
    for cd in summary["attribute"]["categories"].values():
        cd["vqa_rate"] = cd["vqa_correct"] / cd["vqa_total"] if cd["vqa_total"] > 0 else 0
        cd["reasoning_rate"] = cd["reasoning_correct"] / cd["reasoning_total"] if cd["reasoning_total"] > 0 else 0
    for cd in summary["relation"]["categories"].values():
        cd["vqa_rate"] = cd["vqa_correct"] / cd["vqa_total"] if cd["vqa_total"] > 0 else 0
        cd["reasoning_rate"] = cd["reasoning_correct"] / cd["reasoning_total"] if cd["reasoning_total"] > 0 else 0
    return summary


# ---------------------------------------------------------------------------
# OpenRouter call
# ---------------------------------------------------------------------------
INSUFFICIENT_CREDIT_PATTERN = re.compile(
    r"(insufficient[_ -]?credit|no[_ -]?credit|credit[s]?[ _]?(left|exhausted)"
    r"|need[s]?[ ]*more[ ]*credit|payment[ _]required|requires[ _]more[ _]credits)",
    re.IGNORECASE,
)


class CreditExhausted(RuntimeError):
    """Raised when OpenRouter signals insufficient credits."""


def _encode_image_to_data_url(
    path: Path, max_dim: int, quality: int = 90
) -> Tuple[str, str]:
    """Open ``path``, optionally shrink to ``max_dim``, return (mime, data-url).

    Re-encodes everything as JPEG to drop the PNG overhead and to give Pillow
    the freedom to do a small quality/size trade-off. Returns the *full*
    ``data:image/jpeg;base64,...`` URL ready to drop into the
    ``image_url.url`` field of the OpenRouter messages payload.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max_dim > 0 and max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            im = im.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return "image/jpeg", f"data:image/jpeg;base64,{b64}"


def _is_credit_error(status_code: int, body: Any) -> bool:
    """Heuristic check for OpenRouter's 'insufficient credits' signal.

    OpenRouter usually returns a 402 with a body like
        {"error": {"message": "...insufficient credits...", "code": 402}}
    but providers may forward 403 / 400 with a textual hint, so we also
    look at the message itself.
    """
    if status_code == 402:
        return True
    try:
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                if err.get("code") in (402, "402", "insufficient_quota"):
                    return True
                msg = err.get("message", "")
                if msg and INSUFFICIENT_CREDIT_PATTERN.search(str(msg)):
                    return True
            msg = body.get("message", "")
            if msg and INSUFFICIENT_CREDIT_PATTERN.search(str(msg)):
                return True
        if isinstance(body, str) and INSUFFICIENT_CREDIT_PATTERN.search(body):
            return True
    except Exception:
        pass
    return False


async def call_openrouter(
    client: httpx.AsyncClient,
    api_key: str,
    model_id: str,
    messages: List[Dict],
    max_tokens: int,
    extra_headers: Optional[Dict[str, str]],
    max_retries: int,
    base_backoff: float,
    timeout: float,
) -> Dict[str, Any]:
    """POST /chat/completions with retries.

    Returns ``{"ok": True, "text": str, "usage": {...}}`` on success,
    ``{"ok": False, "error": str, "status": int}`` on a non-credit terminal
    failure, and **raises** ``CreditExhausted`` if OpenRouter signals an
    insufficient-credits condition.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    last_err = "unknown"
    last_status = 0
    for attempt in range(max_retries + 1):
        try:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except (httpx.TimeoutException, httpx.TransportError, httpx.RemoteProtocolError) as e:
            last_err = f"transport: {type(e).__name__}: {e}"
            last_status = -1
        else:
            last_status = r.status_code
            ct = r.headers.get("content-type", "")
            try:
                body = r.json() if "json" in ct else r.text
            except Exception:
                body = r.text
            if _is_credit_error(r.status_code, body):
                raise CreditExhausted(
                    f"openrouter signalled insufficient credits "
                    f"(status={r.status_code}, body={str(body)[:300]})"
                )
            if r.status_code == 200:
                if not isinstance(body, dict):
                    last_err = f"non-json 200: {str(body)[:200]}"
                else:
                    choices = body.get("choices") or []
                    if not choices:
                        last_err = f"missing choices in 200 body: {str(body)[:200]}"
                    else:
                        msg = choices[0].get("message") or {}
                        text = msg.get("content") or ""
                        return {
                            "ok": True,
                            "text": text,
                            "usage": body.get("usage", {}),
                            "model": body.get("model", model_id),
                        }
            elif r.status_code in (429, 500, 502, 503, 504):
                last_err = f"transient {r.status_code}: {str(body)[:300]}"
            else:
                return {
                    "ok": False,
                    "error": f"http {r.status_code}: {str(body)[:600]}",
                    "status": r.status_code,
                }
        if attempt < max_retries:
            # capped exponential backoff with jitter
            wait = min(60.0, base_backoff * (2 ** attempt))
            wait *= 0.5 + random.random()
            await asyncio.sleep(wait)
    return {
        "ok": False,
        "error": f"exhausted retries: {last_err}",
        "status": last_status,
    }


# ---------------------------------------------------------------------------
# Task building / driver
# ---------------------------------------------------------------------------
def build_task_list(
    dataset: List[Dict], image_dir: Path, start_idx: int, end_idx: int
) -> List[Dict]:
    tasks: List[Dict] = []
    dataset_slice = dataset[start_idx:end_idx]
    for i, item in enumerate(dataset_slice):
        gid = item.get("case_id", start_idx + i)
        source = item.get("source", "")
        if source == "attribute":
            stage_keys = ["stage_1", "stage_2", "stage_3", "stage_4"]
        elif source == "relation":
            stage_keys = ["stage_1"]
        else:
            continue
        for sk in stage_keys:
            if sk not in item:
                continue
            stage = item[sk]
            vqa_q = stage.get("vqa_question") or ""
            if not vqa_q:
                continue
            tasks.append({
                "case_id": gid,
                "key": sk,
                "result_key": f"case_{gid}_{sk}",
                "image_path": str(image_dir / f"case_{gid}_{sk}.png"),
                "vqa_question": vqa_q,
                "visual_target": stage.get("visual_target", ""),
                "gt_target": stage.get("gt_target", ""),
            })
    return tasks


def load_existing_vqa(output_file: Path) -> Tuple[Dict[str, Dict], Optional[Dict]]:
    """Returns (existing_results_dict, full_blob_or_None).

    A 'result' is considered done iff it does not contain a transient error
    (so the same task will be retried automatically next run).
    """
    if not output_file.exists():
        return {}, None
    try:
        blob = json.loads(output_file.read_text())
    except Exception as e:
        print(f"[warn] could not read existing {output_file}: {e}; starting fresh")
        return {}, None
    out: Dict[str, Dict] = {}
    for item in blob.get("results", []):
        # When build_evaluation_results() flattens results into stages,
        # the per-VQA blob is nested. But the canonical vqa_results.json
        # written by this script (and by the offline evaluator) keeps a
        # flat ``vqa_index`` keyed by result_key.  We support both shapes.
        pass
    vqa_index = blob.get("vqa_index")
    if isinstance(vqa_index, dict):
        for k, v in vqa_index.items():
            if not isinstance(v, dict):
                continue
            # Only treat as 'done' if it's not a known transient/errored
            # state (those should be retried).
            if v.get("transient_error") or v.get("error") in {
                "rate_limited", "timeout", "transient",
            }:
                continue
            out[k] = v
    return out, blob


def write_results(
    output_file: Path,
    args,
    dataset: List[Dict],
    vqa_index: Dict[str, Dict],
    reasoning_results: Dict[int, Dict],
    start_idx: int,
    end_idx: int,
    extra_meta: Optional[Dict] = None,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    eval_results = build_evaluation_results(
        dataset, vqa_index, reasoning_results, start_idx, end_idx
    )
    summary = compute_summary(eval_results)
    blob = {
        "summary": summary,
        "results": eval_results,
        # ``vqa_index`` is what *this* script reads back on resume. We keep
        # it alongside the human-friendly ``results`` so downstream tools
        # like aggregate_results.py see the same schema as the vLLM
        # evaluator emits.
        "vqa_index": vqa_index,
        "eval_model": args.model_id,
        "judge_provider": "openrouter",
        "judge_endpoint": "https://openrouter.ai/api/v1/chat/completions",
    }
    if extra_meta:
        blob.update(extra_meta)
    tmp = output_file.with_suffix(output_file.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=2))
    os.replace(tmp, output_file)


async def process_task(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    task: Dict,
    api_key: str,
    model_id: str,
    extra_headers: Optional[Dict[str, str]],
    image_max_dim: int,
    jpeg_quality: int,
    max_tokens: int,
    max_retries: int,
    base_backoff: float,
    timeout: float,
) -> Tuple[str, Dict[str, Any]]:
    """Returns (result_key, vqa_record_or_error)."""
    rk = task["result_key"]
    img_path = Path(task["image_path"])
    if not img_path.exists():
        return rk, {
            "case_id": task["case_id"], "key": task["key"],
            "gt_target": task["gt_target"],
            "vqa_correct": False, "error": "image_not_found",
            "image_path": task["image_path"],
        }
    # Image encoding is CPU-bound, so run it in the default thread executor
    # to avoid blocking the event loop while we are otherwise IO-bound.
    loop = asyncio.get_running_loop()
    try:
        _, data_url = await loop.run_in_executor(
            None,
            _encode_image_to_data_url,
            img_path,
            image_max_dim,
            jpeg_quality,
        )
    except Exception as e:
        return rk, {
            "case_id": task["case_id"], "key": task["key"],
            "gt_target": task["gt_target"],
            "vqa_correct": False, "error": f"image_encode_error: {e}",
        }

    prompt_text = create_vqa_prompt(task["vqa_question"], task["visual_target"])
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt_text},
        ],
    }]

    async with semaphore:
        resp = await call_openrouter(
            client=client,
            api_key=api_key,
            model_id=model_id,
            messages=messages,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            max_retries=max_retries,
            base_backoff=base_backoff,
            timeout=timeout,
        )
    if not resp.get("ok"):
        return rk, {
            "case_id": task["case_id"], "key": task["key"],
            "gt_target": task["gt_target"],
            "vqa_correct": False,
            "error": "api_error",
            "transient_error": True,
            "error_message": resp.get("error", ""),
            "http_status": resp.get("status", 0),
        }
    parsed = parse_vlm_response(resp["text"])
    parsed["case_id"] = task["case_id"]
    parsed["key"] = task["key"]
    parsed["gt_target"] = task["gt_target"]
    parsed["judge_model"] = resp.get("model", model_id)
    usage = resp.get("usage") or {}
    if usage:
        parsed["usage"] = usage
    return rk, parsed


async def run_async(args, dataset, tasks, reasoning_results, output_file, log_prefix):
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OpenRouter API key missing. Pass --api_key or set OPENROUTER_API_KEY."
        )
    extra_headers: Dict[str, str] = {}
    if args.referer:
        extra_headers["HTTP-Referer"] = args.referer
    if args.app_title:
        extra_headers["X-Title"] = args.app_title

    existing_vqa, _existing_blob = load_existing_vqa(output_file)
    remaining_tasks = [t for t in tasks if t["result_key"] not in existing_vqa]
    print(
        f"{log_prefix} tasks total={len(tasks)} done={len(existing_vqa)} "
        f"remaining={len(remaining_tasks)}"
    )
    if not remaining_tasks:
        # Even when nothing to do, refresh the output file so downstream
        # tools see a consistent schema/summary.
        write_results(output_file, args, dataset, existing_vqa, reasoning_results,
                      args._start_idx, args._end_idx,
                      extra_meta={"completed": True})
        return {"completed": len(existing_vqa), "remaining": 0,
                "credit_exhausted": False, "errored": 0}

    semaphore = asyncio.Semaphore(args.concurrency)
    limits = httpx.Limits(
        max_connections=max(args.concurrency * 2, 16),
        max_keepalive_connections=max(args.concurrency, 8),
    )
    timeouts = httpx.Timeout(args.timeout, connect=min(args.timeout, 30.0))

    state = {
        "stop": False,
        "credit_exhausted": False,
        "done_since_flush": 0,
        "completed": 0,
        "errored": 0,
    }

    def _on_signal(signum, _frame):
        print(f"{log_prefix} received signal {signum}; flushing and stopping...",
              flush=True)
        state["stop"] = True

    old_sigint = signal.getsignal(signal.SIGINT)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)
    except (ValueError, OSError):
        # Signal handlers are only installable from the main thread of the
        # main interpreter; harmless to skip otherwise.
        pass

    t0 = time.time()
    try:
        async with httpx.AsyncClient(limits=limits, timeout=timeouts, http2=False) as client:
            pending: set = set()
            task_iter = iter(remaining_tasks)

            def submit_next(n: int) -> int:
                added = 0
                for _ in range(n):
                    try:
                        t = next(task_iter)
                    except StopIteration:
                        return added
                    coro = process_task(
                        semaphore=semaphore,
                        client=client,
                        task=t,
                        api_key=api_key,
                        model_id=args.model_id,
                        extra_headers=extra_headers or None,
                        image_max_dim=args.image_max_dim,
                        jpeg_quality=args.jpeg_quality,
                        max_tokens=args.max_tokens,
                        max_retries=args.max_retries,
                        base_backoff=args.base_backoff,
                        timeout=args.timeout,
                    )
                    pending.add(asyncio.create_task(coro))
                    added += 1
                return added

            in_flight_cap = max(args.concurrency * 2, args.concurrency + 4)
            submit_next(in_flight_cap)

            while pending and not state["stop"] and not state["credit_exhausted"]:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for fut in done:
                    try:
                        result_key, vqa = fut.result()
                    except CreditExhausted as e:
                        print(f"{log_prefix} CREDIT EXHAUSTED: {e}", flush=True)
                        state["credit_exhausted"] = True
                        continue
                    except Exception as e:
                        print(f"{log_prefix} unexpected exception: {e}", flush=True)
                        state["errored"] += 1
                        continue
                    existing_vqa[result_key] = vqa
                    state["completed"] += 1
                    state["done_since_flush"] += 1
                    if "error" in vqa:
                        state["errored"] += 1
                # Checkpoint periodically.
                if state["done_since_flush"] >= args.checkpoint_every:
                    write_results(output_file, args, dataset, existing_vqa,
                                  reasoning_results, args._start_idx, args._end_idx)
                    elapsed = time.time() - t0
                    rate = state["completed"] / elapsed if elapsed > 0 else 0
                    eta = (len(remaining_tasks) - state["completed"]) / rate if rate > 0 else float("inf")
                    print(
                        f"{log_prefix} checkpoint: completed={state['completed']}/"
                        f"{len(remaining_tasks)} errored={state['errored']} "
                        f"elapsed={elapsed:.0f}s rate={rate:.2f}/s "
                        f"eta={eta:.0f}s",
                        flush=True,
                    )
                    state["done_since_flush"] = 0
                if not state["credit_exhausted"] and not state["stop"]:
                    # Top up in-flight tasks.
                    submit_next(in_flight_cap - len(pending))

            if state["credit_exhausted"] or state["stop"]:
                # Cancel remaining workers; let already-completed flush.
                for fut in pending:
                    fut.cancel()
                # Drain so transport closes cleanly.
                if pending:
                    try:
                        await asyncio.gather(*pending, return_exceptions=True)
                    except Exception:
                        pass
    finally:
        try:
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)
        except Exception:
            pass

    # Final flush.
    extra_meta = {
        "completed_count": state["completed"],
        "errored_count": state["errored"],
        "credit_exhausted": state["credit_exhausted"],
        "interrupted": state["stop"],
    }
    write_results(output_file, args, dataset, existing_vqa, reasoning_results,
                  args._start_idx, args._end_idx, extra_meta=extra_meta)

    # Save the list of still-pending tasks (anything that did not produce a
    # *successful* parsed VQA record).  This lets the orchestrator surface
    # the work that remains after a credit recharge.
    pending_remaining = []
    errored_now = []
    for t in tasks:
        rk = t["result_key"]
        rec = existing_vqa.get(rk)
        if rec is None:
            pending_remaining.append(t)
            continue
        if "error" in rec:
            errored_now.append({**t, "_last_error": rec.get("error"),
                                "_error_message": rec.get("error_message", "")})
            if rec.get("transient_error"):
                # Transient errors are eligible for automatic retry.
                pending_remaining.append(t)

    failed_file = output_file.parent / "failed_tasks.json"
    errored_file = output_file.parent / "errored_tasks.json"
    credit_file = output_file.parent / "credit_status.json"

    if pending_remaining or state["credit_exhausted"]:
        failed_file.write_text(json.dumps({
            "credit_exhausted": state["credit_exhausted"],
            "interrupted": state["stop"],
            "image_dir": args.image_dir,
            "output_file": str(output_file),
            "model_id": args.model_id,
            "pending_count": len(pending_remaining),
            "pending_tasks": pending_remaining,
        }, indent=2))
        print(f"{log_prefix} wrote pending list -> {failed_file} "
              f"({len(pending_remaining)} tasks)")
    elif failed_file.exists():
        # Everything resolved; clean up the stale failure list.
        failed_file.unlink()

    if errored_now:
        errored_file.write_text(json.dumps({
            "errored_count": len(errored_now),
            "errored_tasks": errored_now,
        }, indent=2))
    elif errored_file.exists():
        errored_file.unlink()

    if state["credit_exhausted"]:
        credit_file.write_text(json.dumps({
            "credit_exhausted": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "completed": state["completed"],
            "remaining": len(pending_remaining),
        }, indent=2))
    elif credit_file.exists():
        credit_file.unlink()

    elapsed = time.time() - t0
    print(
        f"{log_prefix} DONE completed={state['completed']} "
        f"errored={state['errored']} "
        f"pending={len(pending_remaining)} "
        f"credit_exhausted={state['credit_exhausted']} "
        f"interrupted={state['stop']} elapsed={elapsed:.0f}s",
        flush=True,
    )
    return {
        "completed": state["completed"],
        "errored": state["errored"],
        "remaining": len(pending_remaining),
        "credit_exhausted": state["credit_exhausted"],
        "interrupted": state["stop"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image_dir", required=True)
    p.add_argument("--reasoning_dir", default=None)
    p.add_argument("--data_path", default="data/UniKE.json")
    p.add_argument("--output_file", required=True)
    p.add_argument("--model_id", default="qwen/qwen3-vl-235b-a22b-instruct",
                   help="OpenRouter model slug")
    p.add_argument("--api_key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    p.add_argument("--referer", default=os.environ.get("OPENROUTER_REFERER", ""),
                   help="Optional HTTP-Referer header (for OpenRouter rankings)")
    p.add_argument("--app_title", default=os.environ.get("OPENROUTER_APP_TITLE", ""),
                   help="Optional X-Title header for OpenRouter")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--checkpoint_every", type=int, default=25)
    p.add_argument("--max_retries", type=int, default=4)
    p.add_argument("--base_backoff", type=float, default=2.0)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--max_tokens", type=int, default=512,
                   help="Max completion tokens per VQA request")
    p.add_argument("--image_max_dim", type=int, default=512,
                   help="Resize longest edge to this many pixels (0 to disable)")
    p.add_argument("--jpeg_quality", type=int, default=88)
    p.add_argument("--start_idx", type=int, default=None)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--log_prefix", default="[judge]")
    p.add_argument("--dry_run", action="store_true",
                   help="Build the task list and print stats, then exit without API calls")
    args = p.parse_args()

    if args.concurrency < 1:
        args.concurrency = 1

    output_file = Path(args.output_file)

    dataset = json.loads(Path(args.data_path).read_text())
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(dataset)
    args._start_idx = start_idx
    args._end_idx = end_idx
    print(f"{args.log_prefix} dataset={args.data_path} items={len(dataset)} "
          f"slice=[{start_idx}:{end_idx}]")

    tasks = build_task_list(dataset, Path(args.image_dir), start_idx, end_idx)
    print(f"{args.log_prefix} built {len(tasks)} VQA tasks "
          f"(image_dir={args.image_dir})")

    reasoning_results: Dict[int, Dict] = {}
    if args.reasoning_dir and Path(args.reasoning_dir).exists():
        reasoning_results = load_reasoning_results(args.reasoning_dir)
        print(f"{args.log_prefix} loaded reasoning for {len(reasoning_results)} cases")

    if args.dry_run:
        existing_vqa, _ = load_existing_vqa(output_file)
        remaining = [t for t in tasks if t["result_key"] not in existing_vqa]
        print(f"{args.log_prefix} DRY-RUN done={len(existing_vqa)} remaining={len(remaining)}")
        return 0

    try:
        status = asyncio.run(run_async(
            args, dataset, tasks, reasoning_results, output_file, args.log_prefix
        ))
    except CreditExhausted as e:
        print(f"{args.log_prefix} CREDIT EXHAUSTED (top-level): {e}", flush=True)
        return 42

    if status.get("credit_exhausted"):
        return 42
    if status.get("interrupted"):
        return 130
    if status.get("remaining", 0) > 0 or status.get("errored", 0) > 0:
        # Non-zero so the orchestrator can flag "incomplete" but distinct
        # from credit-exhausted.
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
