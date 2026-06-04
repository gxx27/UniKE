"""
Aggregate per-case efficacy, reasoning, and VQA metrics into a unified summary for the UniKE dataset.
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional


def load_efficacy_results(run_dir: Path) -> Dict[int, Dict]:
    """Load efficacy results from individual case JSON files.
    
    Looks in run_dir/edit_case/ first (new location), falls back to run_dir/ (old location).
    """
    # Try new location first (edit_case folder)
    edit_case_dir = run_dir / "edit_case"
    if edit_case_dir.exists():
        case_files = sorted(edit_case_dir.glob("*_edits-case_*.json"))
    else:
        # Fall back to old location
        case_files = sorted(run_dir.glob("*_edits-case_*.json"))
    
    results = {}
    for f in case_files:
        with open(f) as fp:
            data = json.load(fp)
        case_id = data.get("case_id", -1)
        post = data.get("post", {})
        results[case_id] = {
            "efficacy_correct": post.get("efficacy_correct", False),
            "source": post.get("source", ""),
            "category": post.get("category", ""),
            "stage_efficacy": post.get("stage_efficacy", {}),
            "stage_n_efficacy": post.get("stage_n_efficacy", False),
        }
    
    return results


def load_reasoning_results(reasoning_dir: Path) -> Dict[int, Dict]:
    """Load reasoning results from JSON files."""
    all_results = {}
    
    for json_file in reasoning_dir.glob("reasoning_*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            for item in data:
                case_id = item.get("case_id")
                all_results[case_id] = item
    
    return all_results


def load_vqa_results(vqa_file: Path) -> Dict[int, Dict]:
    """Load VQA results from evaluation JSON file."""
    if not vqa_file.exists():
        return {}
    
    with open(vqa_file, 'r') as f:
        data = json.load(f)
    
    results = {}
    for item in data.get("results", []):
        case_id = item.get("case_id")
        results[case_id] = item
    
    return results


def load_dataset(data_path: str) -> List[Dict]:
    """Load the UniKE dataset."""
    with open(data_path, 'r') as f:
        return json.load(f)


def compute_comprehensive_summary(
    dataset: List[Dict],
    efficacy_results: Dict[int, Dict],
    reasoning_results: Dict[int, Dict],
    vqa_results: Dict[int, Dict],
) -> Dict:
    """Compute comprehensive summary combining all metrics."""
    
    summary = {
        "total_items": len(dataset),
        "attribute": {
            "total": 0,
            "stages": {},
            "categories": {},
        },
        "relation": {
            "total": 0,
            "stages": {},
            "categories": {},
        },
    }

    # Initialize stage structures
    for stage_key in ["stage_1", "stage_2", "stage_3", "stage_4", "stage_n"]:
        summary["attribute"]["stages"][stage_key] = {
            "efficacy": {"correct": 0, "total": 0},
            "reasoning": {"correct": 0, "total": 0},
            "vqa": {"correct": 0, "total": 0},
        }

    for stage_key in ["stage_1"]:
        summary["relation"]["stages"][stage_key] = {
            "efficacy": {"correct": 0, "total": 0},
            "reasoning": {"correct": 0, "total": 0},
            "vqa": {"correct": 0, "total": 0},
        }
    
    # Process each item
    for i, item in enumerate(dataset):
        source = item.get("source", "")
        category = item.get("category", "")
        case_id = item.get("case_id", i)
        
        efficacy = efficacy_results.get(case_id, {})
        reasoning = reasoning_results.get(case_id, {})
        vqa = vqa_results.get(case_id, {})
        
        if source == "attribute":
            summary["attribute"]["total"] += 1
            
            # Initialize category if needed
            if category not in summary["attribute"]["categories"]:
                summary["attribute"]["categories"][category] = {
                    "efficacy": {"correct": 0, "total": 0},
                    "reasoning": {"correct": 0, "total": 0},
                    "vqa": {"correct": 0, "total": 0},
                }
            
            # Track if any stage passes for stage_n
            any_efficacy = False
            any_reasoning = False
            any_vqa = False
            
            stages_reasoning = reasoning.get("stages", {})
            stages_vqa = vqa.get("stages", {})
            stage_efficacy = efficacy.get("stage_efficacy", {})
            
            for stage_key in ["stage_1", "stage_2", "stage_3", "stage_4"]:
                if stage_key not in item:
                    continue
                
                # Efficacy
                summary["attribute"]["stages"][stage_key]["efficacy"]["total"] += 1
                if stage_efficacy.get(stage_key, False):
                    summary["attribute"]["stages"][stage_key]["efficacy"]["correct"] += 1
                    any_efficacy = True
                
                # Reasoning
                stage_reasoning = stages_reasoning.get(stage_key, {})
                summary["attribute"]["stages"][stage_key]["reasoning"]["total"] += 1
                if stage_reasoning.get("contains_target", False):
                    summary["attribute"]["stages"][stage_key]["reasoning"]["correct"] += 1
                    any_reasoning = True
                
                # VQA
                stage_vqa = stages_vqa.get(stage_key, {})
                vqa_data = stage_vqa.get("vqa", {})
                if "error" not in vqa_data and "skipped_reason" not in vqa_data:
                    summary["attribute"]["stages"][stage_key]["vqa"]["total"] += 1
                    if vqa_data.get("vqa_correct", False):
                        summary["attribute"]["stages"][stage_key]["vqa"]["correct"] += 1
                        any_vqa = True
            
            # stage_n
            summary["attribute"]["stages"]["stage_n"]["efficacy"]["total"] += 1
            summary["attribute"]["stages"]["stage_n"]["reasoning"]["total"] += 1
            summary["attribute"]["stages"]["stage_n"]["vqa"]["total"] += 1
            
            if any_efficacy or efficacy.get("stage_n_efficacy", False):
                summary["attribute"]["stages"]["stage_n"]["efficacy"]["correct"] += 1
            if any_reasoning:
                summary["attribute"]["stages"]["stage_n"]["reasoning"]["correct"] += 1
            if any_vqa:
                summary["attribute"]["stages"]["stage_n"]["vqa"]["correct"] += 1
            
            # Category (using stage_1 metrics)
            cat_data = summary["attribute"]["categories"][category]
            cat_data["efficacy"]["total"] += 1
            cat_data["reasoning"]["total"] += 1
            
            if stage_efficacy.get("stage_1", False):
                cat_data["efficacy"]["correct"] += 1
            
            stage_1_reasoning = stages_reasoning.get("stage_1", {})
            if stage_1_reasoning.get("contains_target", False):
                cat_data["reasoning"]["correct"] += 1
            
            stage_1_vqa = stages_vqa.get("stage_1", {})
            vqa_data = stage_1_vqa.get("vqa", {})
            if "error" not in vqa_data and "skipped_reason" not in vqa_data:
                cat_data["vqa"]["total"] += 1
                if vqa_data.get("vqa_correct", False):
                    cat_data["vqa"]["correct"] += 1
                    
        elif source == "relation":
            summary["relation"]["total"] += 1
            
            # Initialize category if needed
            if category not in summary["relation"]["categories"]:
                summary["relation"]["categories"][category] = {
                    "efficacy": {"correct": 0, "total": 0},
                    "reasoning": {"correct": 0, "total": 0},
                    "vqa": {"correct": 0, "total": 0},
                }
            
            stages_reasoning = reasoning.get("stages", {})
            stages_vqa = vqa.get("stages", {})
            stage_efficacy = efficacy.get("stage_efficacy", {})

            for stage_key in ["stage_1"]:
                if stage_key not in item:
                    continue

                # Efficacy
                summary["relation"]["stages"][stage_key]["efficacy"]["total"] += 1
                if stage_efficacy.get(stage_key, False):
                    summary["relation"]["stages"][stage_key]["efficacy"]["correct"] += 1

                # Reasoning
                stage_reasoning = stages_reasoning.get(stage_key, {})
                if "skipped_reason" not in stage_reasoning:
                    summary["relation"]["stages"][stage_key]["reasoning"]["total"] += 1
                    if stage_reasoning.get("contains_target", False):
                        summary["relation"]["stages"][stage_key]["reasoning"]["correct"] += 1

                # VQA
                stage_vqa = stages_vqa.get(stage_key, {})
                vqa_data = stage_vqa.get("vqa", {})
                if "error" not in vqa_data and "skipped_reason" not in vqa_data:
                    summary["relation"]["stages"][stage_key]["vqa"]["total"] += 1
                    if vqa_data.get("vqa_correct", False):
                        summary["relation"]["stages"][stage_key]["vqa"]["correct"] += 1

            # Category (using stage_1 metrics)
            cat_data = summary["relation"]["categories"][category]
            cat_data["efficacy"]["total"] += 1
            cat_data["reasoning"]["total"] += 1

            if stage_efficacy.get("stage_1", False):
                cat_data["efficacy"]["correct"] += 1

            stage_1_reasoning = stages_reasoning.get("stage_1", {})
            if stage_1_reasoning.get("contains_target", False):
                cat_data["reasoning"]["correct"] += 1

            stage_1_vqa = stages_vqa.get("stage_1", {})
            vqa_data = stage_1_vqa.get("vqa", {})
            if "error" not in vqa_data and "skipped_reason" not in vqa_data:
                cat_data["vqa"]["total"] += 1
                if vqa_data.get("vqa_correct", False):
                    cat_data["vqa"]["correct"] += 1
    
    # Calculate rates
    def calc_rate(data):
        for metric in ["efficacy", "reasoning", "vqa"]:
            if metric in data:
                total = data[metric]["total"]
                correct = data[metric]["correct"]
                data[metric]["rate"] = correct / total if total > 0 else 0
    
    for stage_key in summary["attribute"]["stages"]:
        calc_rate(summary["attribute"]["stages"][stage_key])
    
    for stage_key in summary["relation"]["stages"]:
        calc_rate(summary["relation"]["stages"][stage_key])
    
    for cat_data in summary["attribute"]["categories"].values():
        calc_rate(cat_data)
    
    for cat_data in summary["relation"]["categories"].values():
        calc_rate(cat_data)
    
    return summary


def print_summary(summary: Dict):
    """Print a formatted summary report."""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\nTotal Items: {summary['total_items']}")
    
    # Attribute Results
    print("\n" + "-" * 40)
    print("ATTRIBUTE RESULTS")
    print("-" * 40)
    print(f"Total: {summary['attribute']['total']}")
    
    print("\nBy Stage:")
    print(f"{'Stage':<12} {'Efficacy':<20} {'Reasoning':<20} {'VQA':<20}")
    print("-" * 72)
    for stage_key in ["stage_1", "stage_2", "stage_3", "stage_4", "stage_n"]:
        data = summary["attribute"]["stages"][stage_key]
        eff = data["efficacy"]
        rea = data["reasoning"]
        vqa = data["vqa"]
        
        eff_str = f"{eff['rate']:.2%} ({eff['correct']}/{eff['total']})" if eff['total'] > 0 else "N/A"
        rea_str = f"{rea['rate']:.2%} ({rea['correct']}/{rea['total']})" if rea['total'] > 0 else "N/A"
        vqa_str = f"{vqa['rate']:.2%} ({vqa['correct']}/{vqa['total']})" if vqa['total'] > 0 else "N/A"
        
        print(f"{stage_key:<12} {eff_str:<20} {rea_str:<20} {vqa_str:<20}")
    
    print("\nBy Category (stage_1 metrics):")
    print(f"{'Category':<12} {'Efficacy':<20} {'Reasoning':<20} {'VQA':<20}")
    print("-" * 72)
    for cat, data in sorted(summary["attribute"]["categories"].items()):
        eff = data["efficacy"]
        rea = data["reasoning"]
        vqa = data["vqa"]
        
        eff_str = f"{eff['rate']:.2%} ({eff['correct']}/{eff['total']})" if eff['total'] > 0 else "N/A"
        rea_str = f"{rea['rate']:.2%} ({rea['correct']}/{rea['total']})" if rea['total'] > 0 else "N/A"
        vqa_str = f"{vqa['rate']:.2%} ({vqa['correct']}/{vqa['total']})" if vqa['total'] > 0 else "N/A"
        
        print(f"{cat:<12} {eff_str:<20} {rea_str:<20} {vqa_str:<20}")
    
    # Relation Results
    print("\n" + "-" * 40)
    print("RELATION RESULTS")
    print("-" * 40)
    print(f"Total: {summary['relation']['total']}")
    
    print("\nBy Stage:")
    print(f"{'Stage':<12} {'Efficacy':<20} {'Reasoning':<20} {'VQA':<20}")
    print("-" * 72)
    for stage_key in ["stage_1"]:
        data = summary["relation"]["stages"][stage_key]
        eff = data["efficacy"]
        rea = data["reasoning"]
        vqa = data["vqa"]

        eff_str = f"{eff['rate']:.2%} ({eff['correct']}/{eff['total']})" if eff['total'] > 0 else "N/A"
        rea_str = f"{rea['rate']:.2%} ({rea['correct']}/{rea['total']})" if rea['total'] > 0 else "N/A"
        vqa_str = f"{vqa['rate']:.2%} ({vqa['correct']}/{vqa['total']})" if vqa['total'] > 0 else "N/A"

        print(f"{stage_key:<12} {eff_str:<20} {rea_str:<20} {vqa_str:<20}")

    print("\nBy Category (stage_1 metrics):")
    print(f"{'Category':<12} {'Efficacy':<20} {'Reasoning':<20} {'VQA':<20}")
    print("-" * 72)
    for cat, data in sorted(summary["relation"]["categories"].items()):
        eff = data["efficacy"]
        rea = data["reasoning"]
        vqa = data["vqa"]
        
        eff_str = f"{eff['rate']:.2%} ({eff['correct']}/{eff['total']})" if eff['total'] > 0 else "N/A"
        rea_str = f"{rea['rate']:.2%} ({rea['correct']}/{rea['total']})" if rea['total'] > 0 else "N/A"
        vqa_str = f"{vqa['rate']:.2%} ({vqa['correct']}/{vqa['total']})" if vqa['total'] > 0 else "N/A"
        
        print(f"{cat:<12} {eff_str:<20} {rea_str:<20} {vqa_str:<20}")
    
    print("\n" + "=" * 80)


def write_summary_txt(summary: Dict, output_file: str, no_reasoning: bool = False):
    """Write summary to a text file in the same format as print_summary."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("COMPREHENSIVE EVALUATION RESULTS")
    lines.append("=" * 80)
    
    lines.append(f"\nTotal Items: {summary['total_items']}")
    if no_reasoning:
        lines.append("Mode: No Reasoning (direct image generation)")
    
    # Attribute Results
    lines.append("\n" + "-" * 40)
    lines.append("ATTRIBUTE RESULTS")
    lines.append("-" * 40)
    lines.append(f"Total: {summary['attribute']['total']}")
    
    lines.append("\nBy Stage:")
    lines.append(f"{'Stage':<12} {'Efficacy':<20} {'Reasoning':<20} {'VQA':<20}")
    lines.append("-" * 72)
    for stage_key in ["stage_1", "stage_2", "stage_3", "stage_4", "stage_n"]:
        data = summary["attribute"]["stages"][stage_key]
        eff = data["efficacy"]
        rea = data["reasoning"]
        vqa = data["vqa"]
        
        eff_str = f"{eff['rate']:.2%} ({eff['correct']}/{eff['total']})" if eff['total'] > 0 else "N/A"
        if no_reasoning:
            rea_str = "-"
        else:
            rea_str = f"{rea['rate']:.2%} ({rea['correct']}/{rea['total']})" if rea['total'] > 0 else "N/A"
        vqa_str = f"{vqa['rate']:.2%} ({vqa['correct']}/{vqa['total']})" if vqa['total'] > 0 else "N/A"
        
        lines.append(f"{stage_key:<12} {eff_str:<20} {rea_str:<20} {vqa_str:<20}")
    
    lines.append("\nBy Category (stage_1 metrics):")
    lines.append(f"{'Category':<12} {'Efficacy':<20} {'Reasoning':<20} {'VQA':<20}")
    lines.append("-" * 72)
    for cat, data in sorted(summary["attribute"]["categories"].items()):
        eff = data["efficacy"]
        rea = data["reasoning"]
        vqa = data["vqa"]
        
        eff_str = f"{eff['rate']:.2%} ({eff['correct']}/{eff['total']})" if eff['total'] > 0 else "N/A"
        if no_reasoning:
            rea_str = "-"
        else:
            rea_str = f"{rea['rate']:.2%} ({rea['correct']}/{rea['total']})" if rea['total'] > 0 else "N/A"
        vqa_str = f"{vqa['rate']:.2%} ({vqa['correct']}/{vqa['total']})" if vqa['total'] > 0 else "N/A"
        
        lines.append(f"{cat:<12} {eff_str:<20} {rea_str:<20} {vqa_str:<20}")
    
    # Relation Results
    lines.append("\n" + "-" * 40)
    lines.append("RELATION RESULTS")
    lines.append("-" * 40)
    lines.append(f"Total: {summary['relation']['total']}")
    
    lines.append("\nBy Stage:")
    lines.append(f"{'Stage':<12} {'Efficacy':<20} {'Reasoning':<20} {'VQA':<20}")
    lines.append("-" * 72)
    for stage_key in ["stage_1"]:
        data = summary["relation"]["stages"][stage_key]
        eff = data["efficacy"]
        rea = data["reasoning"]
        vqa = data["vqa"]

        eff_str = f"{eff['rate']:.2%} ({eff['correct']}/{eff['total']})" if eff['total'] > 0 else "N/A"
        if no_reasoning:
            rea_str = "-"
        else:
            rea_str = f"{rea['rate']:.2%} ({rea['correct']}/{rea['total']})" if rea['total'] > 0 else "N/A"
        vqa_str = f"{vqa['rate']:.2%} ({vqa['correct']}/{vqa['total']})" if vqa['total'] > 0 else "N/A"

        lines.append(f"{stage_key:<12} {eff_str:<20} {rea_str:<20} {vqa_str:<20}")

    lines.append("\nBy Category (stage_1 metrics):")
    lines.append(f"{'Category':<12} {'Efficacy':<20} {'Reasoning':<20} {'VQA':<20}")
    lines.append("-" * 72)
    for cat, data in sorted(summary["relation"]["categories"].items()):
        eff = data["efficacy"]
        rea = data["reasoning"]
        vqa = data["vqa"]
        
        eff_str = f"{eff['rate']:.2%} ({eff['correct']}/{eff['total']})" if eff['total'] > 0 else "N/A"
        if no_reasoning:
            rea_str = "-"
        else:
            rea_str = f"{rea['rate']:.2%} ({rea['correct']}/{rea['total']})" if rea['total'] > 0 else "N/A"
        vqa_str = f"{vqa['rate']:.2%} ({vqa['correct']}/{vqa['total']})" if vqa['total'] > 0 else "N/A"
        
        lines.append(f"{cat:<12} {eff_str:<20} {rea_str:<20} {vqa_str:<20}")
    
    lines.append("\n" + "=" * 80)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description="Aggregate all evaluation results")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to the run directory")
    parser.add_argument("--data_path", type=str, default="data/UniKE.json",
                        help="Path to the UniKE.json dataset")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file path (default: run_dir/results/final_results.txt)")
    parser.add_argument("--no_reasoning", action="store_true",
                        help="No reasoning mode - set reasoning accuracy to '-'")
    parser.add_argument("--vqa_results_file", type=str, default=None,
                        help="Custom path to VQA results file (default: auto-detect based on output_file)")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    results_dir = run_dir / "results"
    output_file = args.output_file or str(results_dir / "final_results.txt")
    
    print(f"Loading data from: {run_dir}")
    
    # Load dataset
    dataset = load_dataset(args.data_path)
    print(f"Loaded dataset with {len(dataset)} items")
    
    # Load efficacy results
    efficacy_results = load_efficacy_results(run_dir)
    print(f"Loaded efficacy results for {len(efficacy_results)} items")
    
    # Load reasoning results (skip if no_reasoning mode)
    reasoning_dir = run_dir / "reasoning"
    reasoning_results = {}
    if not args.no_reasoning and reasoning_dir.exists():
        reasoning_results = load_reasoning_results(reasoning_dir)
        print(f"Loaded reasoning results for {len(reasoning_results)} items")
    elif args.no_reasoning:
        print("No reasoning mode - skipping reasoning results")
    
    # Load VQA results
    # Try to find VQA results in the same directory as output_file, or use custom path
    if args.vqa_results_file:
        vqa_file = Path(args.vqa_results_file)
    else:
        # Auto-detect: look in the same directory as output_file
        output_dir = Path(output_file).parent
        vqa_file = output_dir / "vqa_results.json"
        if not vqa_file.exists():
            # Fall back to standard results directory
            vqa_file = results_dir / "vqa_results.json"
    
    vqa_results = {}
    if vqa_file.exists():
        vqa_results = load_vqa_results(vqa_file)
        print(f"Loaded VQA results for {len(vqa_results)} items from {vqa_file}")
    else:
        print(f"Warning: VQA results file not found at {vqa_file}")
    
    # Compute comprehensive summary
    summary = compute_comprehensive_summary(
        dataset, efficacy_results, reasoning_results, vqa_results
    )
    
    # Print summary
    print_summary(summary)
    
    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Write text summary file (human readable)
    write_summary_txt(summary, output_file, no_reasoning=args.no_reasoning)
    print(f"\nResults saved to: {output_file}")
    
    # Also save JSON for programmatic access
    json_output_file = output_file.replace('.txt', '.json')
    if json_output_file == output_file:
        json_output_file = output_file + '.json'
    
    output = {
        "summary": summary,
        "metadata": {
            "run_dir": str(run_dir),
            "data_path": args.data_path,
            "efficacy_count": len(efficacy_results),
            "reasoning_count": len(reasoning_results),
            "vqa_count": len(vqa_results),
            "no_reasoning_mode": args.no_reasoning,
        }
    }
    
    with open(json_output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"JSON results also saved to: {json_output_file}")


if __name__ == "__main__":
    main()
