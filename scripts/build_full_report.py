"""
Build the full Markdown results report from the per-run final_results.json files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MODEL_SLUGS = {
    "ovis":     ("Ovis-U1",  "AIDC-AI_Ovis-U1-3B"),
    "blip3o":   ("BLIP3o",   "BLIP3o_BLIP3o-Model-4B"),
    "omnigen2": ("OmniGen2", "OmniGen2_OmniGen2"),
}

ATTR_CATS = ["color", "material", "pattern", "shape", "size"]
REL_CATS  = ["affiliation", "creator", "location", "occupation"]
ATTR_STAGES = ["stage_1", "stage_2", "stage_3", "stage_4", "stage_n"]
REL_STAGES  = ["stage_1"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def latest_run_dir(slug: str, alg: str, results_root: Path) -> Optional[Path]:
    base = results_root / slug / alg
    if not base.is_dir():
        return None
    runs = sorted(p for p in base.iterdir()
                  if p.is_dir() and p.name.startswith("run_"))
    return runs[-1] if runs else None


def load_final(run_dir: Path, results_subdir: str) -> Optional[dict]:
    f = run_dir / results_subdir / "final_results.json"
    if not f.is_file():
        return None
    try:
        with open(f) as fp:
            return json.load(fp)
    except Exception:
        return None


def load_pending(run_dir: Path, results_subdir: str) -> Optional[dict]:
    f = run_dir / results_subdir / "failed_tasks.json"
    if not f.is_file():
        return None
    try:
        with open(f) as fp:
            return json.load(fp)
    except Exception:
        return None


def fmt_pct(correct: int, total: int) -> str:
    if total <= 0:
        return "  -  "
    return f"{100 * correct / total:5.2f}"


def fmt_pct_fromdict(d: Dict) -> str:
    """Accept a {'correct': c, 'total': t} dict (or empty) and return formatted %."""
    if not d:
        return "  -  "
    return fmt_pct(int(d.get("correct", 0)), int(d.get("total", 0)))


# ---------------------------------------------------------------------------
# Cell extraction
# ---------------------------------------------------------------------------
def summarize(final: dict) -> dict:
    """Pull the bits we need out of aggregate_results.py's final_results.json."""
    s = final.get("summary", final)
    attr = s.get("attribute", {})
    rel  = s.get("relation",  {})
    return {
        "n_attr": int(attr.get("total", 0)),
        "n_rel":  int(rel.get("total",  0)),
        "attr_stages":  attr.get("stages", {}),
        "rel_stages":   rel.get("stages",  {}),
        "attr_cats":    attr.get("categories", {}),
        "rel_cats":     rel.get("categories",  {}),
    }


def headline_cells(s: dict, reasoning_col: bool) -> List[str]:
    """One row's cells for the headline tables, mirroring smoke_test_report.md."""
    a_s1 = s["attr_stages"].get("stage_1", {})
    r_h1 = s["rel_stages"].get("stage_1", {})

    a_eff = a_s1.get("efficacy",  {})
    a_rea = a_s1.get("reasoning", {})
    a_vqa = a_s1.get("vqa",       {})
    r_eff = r_h1.get("efficacy",  {})
    r_rea = r_h1.get("reasoning", {})
    r_vqa = r_h1.get("vqa",       {})

    def overall(a: Dict, r: Dict) -> str:
        c = int(a.get("correct", 0)) + int(r.get("correct", 0))
        t = int(a.get("total",   0)) + int(r.get("total",   0))
        return fmt_pct(c, t)

    cells = [str(s["n_attr"]), str(s["n_rel"]),
             fmt_pct_fromdict(a_eff)]
    if reasoning_col:
        cells.append(fmt_pct_fromdict(a_rea))
    cells.append(fmt_pct_fromdict(a_vqa))
    cells.append(fmt_pct_fromdict(r_eff))
    if reasoning_col:
        cells.append(fmt_pct_fromdict(r_rea))
    cells.append(fmt_pct_fromdict(r_vqa))
    cells.append(overall(a_eff, r_eff))
    if reasoning_col:
        cells.append(overall(a_rea, r_rea))
    cells.append(overall(a_vqa, r_vqa))
    return cells


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------
def md_row(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def md_sep(n: int) -> str:
    return "| " + " | ".join("-----" for _ in range(n)) + " |"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def render_headline_table(
    out: List[str], title: str,
    rows: List[Tuple[str, str, dict]], reasoning_col: bool,
) -> None:
    out.append(f"## {title}\n")
    if not rows:
        out.append("_No data._\n")
        return
    header = ["Model", "Method", "n_attr", "n_rel", "Attr Eff."]
    if reasoning_col:
        header.append("Attr Reason.")
    header.append("Attr VQA")
    header.append("Rel Eff.")
    if reasoning_col:
        header.append("Rel Reason.")
    header.append("Rel VQA")
    header.append("Overall Eff.")
    if reasoning_col:
        header.append("Overall Reason.")
    header.append("Overall VQA")
    out.append(md_row(header))
    out.append(md_sep(len(header)))
    for (model, alg, s) in rows:
        cells = [model, alg] + headline_cells(s, reasoning_col)
        out.append(md_row(cells))
    out.append("")


def render_per_stage_table(
    out: List[str], title: str,
    rows: List[Tuple[str, str, dict]],
    section: str,        # "attr" | "rel"
    metric: str,          # "efficacy" | "reasoning" | "vqa"
) -> None:
    """Per-stage breakdown for a single metric across all combos."""
    out.append(f"### {title}\n")
    if section == "attr":
        keys = ATTR_STAGES
        key_field = "attr_stages"
    else:
        keys = REL_STAGES
        key_field = "rel_stages"
    if not rows:
        out.append("_No data._\n")
        return
    header = ["Model", "Method"] + keys
    out.append(md_row(header))
    out.append(md_sep(len(header)))
    for (model, alg, s) in rows:
        stage_d = s.get(key_field, {})
        cells = [model, alg]
        for k in keys:
            d = stage_d.get(k, {}).get(metric, {})
            cells.append(fmt_pct_fromdict(d))
        out.append(md_row(cells))
    out.append("")


def render_category_table(
    out: List[str], title: str,
    rows: List[Tuple[str, str, dict]],
    metric: str,
) -> None:
    out.append(f"### {title}\n")
    if not rows:
        out.append("_No data._\n")
        return
    cats = ATTR_CATS + REL_CATS
    header = ["Model", "Method"] + cats
    out.append(md_row(header))
    out.append(md_sep(len(header)))
    for (model, alg, s) in rows:
        cells = [model, alg]
        for c in ATTR_CATS:
            d = s["attr_cats"].get(c, {}).get(metric, {})
            cells.append(fmt_pct_fromdict(d))
        for c in REL_CATS:
            d = s["rel_cats"].get(c, {}).get(metric, {})
            cells.append(fmt_pct_fromdict(d))
        out.append(md_row(cells))
    out.append("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_root", default="results_summary",
                   help="Where to read final_results.json from. Mirrors the "
                        "layout of results/ but we never read images here.")
    p.add_argument("--algs", default="AlphaEdit MEMIT PMET",
                   help="Space- or comma-separated algorithm names.")
    p.add_argument("--models", default="ovis blip3o omnigen2",
                   help="Space-separated model tags (must be keys in MODEL_SLUGS).")
    p.add_argument("--model_filter", default="",
                   help="If set, restrict the report to this subset of model tags.")
    p.add_argument("--modes", default="reasoning direct",
                   help="Which modes to include. Order is preserved.")
    p.add_argument("--results_reasoning_subdir", default="results")
    p.add_argument("--results_direct_subdir",    default="results_no_reasoning")
    p.add_argument("--data_path", default="data/UniKE.json")
    p.add_argument("--out_path",  default="results_summary/full_report.md")
    args = p.parse_args()

    algs   = [a for a in args.algs.replace(",", " ").split() if a]
    models = [m for m in args.models.replace(",", " ").split() if m]
    if args.model_filter:
        keep = set(args.model_filter.replace(",", " ").split())
        models = [m for m in models if m in keep]
    modes  = [m for m in args.modes.replace(",", " ").split() if m]

    root = Path(args.results_root)

    # rows per mode -> [(display_name, alg, summary_dict), ...]
    rows: Dict[str, List[Tuple[str, str, dict]]] = {m: [] for m in modes}
    pending_rows: List[Tuple[str, str, str, str]] = []  # (display, alg, mode, str)

    for tag in models:
        if tag not in MODEL_SLUGS:
            print(f"[report] WARN: unknown model tag '{tag}'")
            continue
        display, slug = MODEL_SLUGS[tag]
        for alg in algs:
            run_dir = latest_run_dir(slug, alg, root)
            if run_dir is None:
                print(f"[report] no run_dir under {root / slug / alg}")
                continue
            for mode in modes:
                if mode == "reasoning":
                    subdir = args.results_reasoning_subdir
                elif mode == "direct":
                    subdir = args.results_direct_subdir
                else:
                    print(f"[report] WARN: unknown mode '{mode}'")
                    continue
                final = load_final(run_dir, subdir)
                if final is None:
                    print(f"[report] missing final_results.json in {run_dir}/{subdir}")
                    continue
                s = summarize(final)
                rows[mode].append((display, alg, s))

                pf = load_pending(run_dir, subdir)
                if pf and pf.get("pending_count", 0) > 0:
                    pending_rows.append((
                        display, alg, mode,
                        f"{pf.get('pending_count', 0)} tasks "
                        f"(credit_exhausted={pf.get('credit_exhausted', False)})"
                    ))

    # ----------------------------------------------------------------------
    # Compose the markdown
    # ----------------------------------------------------------------------
    out: List[str] = []
    out.append("# Cross-Modality Knowledge Editing -- Full Judge Report\n")
    out.append(f"Dataset: `{args.data_path}`\n")
    out.append(f"Reading results from: `{args.results_root}/`\n")
    out.append("Judge: OpenRouter (`qwen/qwen3-vl-235b-a22b-instruct`)\n")
    out.append("Numbers are accuracies in percent (`%`). "
               "`Attr` = stage_1 metric on attribute items, "
               "`Rel` = stage_1 metric on relation items, "
               "`Overall` = count-weighted average.\n")

    # Surface pending work up front so the reader notices it.
    if pending_rows:
        out.append("> **Note: some combos still have pending tasks** "
                   "(typically due to insufficient credits). "
                   "Re-run `bash run_judge_openrouter_retry.sh` after topping up.\n")
        out.append("| Model | Method | Mode | Pending |")
        out.append("| ----- | ------ | ---- | ------- |")
        for r in pending_rows:
            out.append(md_row(list(r)))
        out.append("")

    # Headline tables -- one per mode.
    for mode in modes:
        if mode == "reasoning":
            render_headline_table(
                out, "Reasoning-Augmented -- Headline Numbers",
                rows[mode], reasoning_col=True)
        elif mode == "direct":
            render_headline_table(
                out, "Direct (no reasoning) -- Headline Numbers",
                rows[mode], reasoning_col=False)

    # Per-stage tables (Efficacy / Reasoning / VQA) for each mode.
    out.append("## Per-Stage Accuracy (Attribute)\n")
    for mode in modes:
        mode_label = ("Reasoning-Augmented" if mode == "reasoning"
                      else "Direct (no reasoning)")
        for metric, m_label in [("efficacy",  "Efficacy"),
                                ("reasoning", "Reasoning"),
                                ("vqa",       "VQA")]:
            if mode == "direct" and metric == "reasoning":
                continue  # reasoning is not produced in direct mode
            render_per_stage_table(
                out,
                f"{m_label} -- {mode_label}",
                rows[mode], section="attr", metric=metric,
            )

    out.append("## Per-Hop Accuracy (Relation)\n")
    for mode in modes:
        mode_label = ("Reasoning-Augmented" if mode == "reasoning"
                      else "Direct (no reasoning)")
        for metric, m_label in [("efficacy",  "Efficacy"),
                                ("reasoning", "Reasoning"),
                                ("vqa",       "VQA")]:
            if mode == "direct" and metric == "reasoning":
                continue
            render_per_stage_table(
                out,
                f"{m_label} -- {mode_label}",
                rows[mode], section="rel", metric=metric,
            )

    # Per-category VQA (preserves the layout of smoke_test_report.md) and
    # Reasoning for the reasoning mode.
    out.append("## Per-Category Accuracy\n")
    for mode in modes:
        mode_label = ("Reasoning-Augmented" if mode == "reasoning"
                      else "Direct (no reasoning)")
        for metric, m_label in [("efficacy",  "Efficacy"),
                                ("reasoning", "Reasoning"),
                                ("vqa",       "VQA")]:
            if mode == "direct" and metric == "reasoning":
                continue
            render_category_table(
                out,
                f"{m_label} -- {mode_label}",
                rows[mode], metric=metric,
            )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(out).rstrip() + "\n")
    print(f"[report] wrote {out_path}")


if __name__ == "__main__":
    main()
