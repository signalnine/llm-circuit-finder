#!/usr/bin/env python3
"""
Compare lm_eval results across multiple runs.

Reads the results JSON files from lm_eval output directories and
prints a side-by-side comparison table.

Usage:
    python compare_eval.py ./eval_base ./eval_rys_balanced ./eval_rys_triple
    python compare_eval.py ./eval_*
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(folder: str) -> dict:
    """Load lm_eval results from an output folder."""
    folder = Path(folder)

    # lm_eval saves results in a JSON file inside the folder
    # The filename varies, so find it
    candidates = list(folder.glob("**/*results*.json"))
    if not candidates:
        # Try the folder itself as a JSON file
        if folder.suffix == '.json' and folder.is_file():
            with open(folder) as f:
                return json.load(f)
        print(f"WARNING: No results JSON found in {folder}", file=sys.stderr)
        return {}

    # Use the most recent one
    results_file = max(candidates, key=lambda p: p.stat().st_mtime)

    with open(results_file) as f:
        data = json.load(f)

    return data


def extract_metrics(data: dict) -> dict:
    """Extract task metrics from lm_eval results format."""
    metrics = {}

    results = data.get("results", {})
    for task_name, task_data in results.items():
        for key, value in task_data.items():
            if key.endswith(",none") or key.endswith(",flexible-extract") or key.endswith(",strict-match") or key.endswith(",get-answer"):
                # Parse "metric_name,filter_name" format
                parts = key.rsplit(",", 1)
                metric = parts[0]
                filter_name = parts[1] if len(parts) > 1 else ""

                if isinstance(value, (int, float)):
                    display_name = f"{task_name}"
                    if filter_name and filter_name != "none":
                        display_name += f" ({filter_name})"
                    if metric not in ("alias",):
                        label = f"{task_name}|{metric}|{filter_name}"
                        metrics[label] = {
                            "task": task_name,
                            "metric": metric,
                            "filter": filter_name,
                            "value": value,
                        }

    return metrics


def get_display_name(label: str, metric_info: dict) -> str:
    """Create a readable display name from metric info."""
    task = metric_info["task"]
    metric = metric_info["metric"]
    filt = metric_info["filter"]

    # Shorten common task names
    task = task.replace("bbh_cot_fewshot_", "bbh/")

    if filt and filt not in ("none", "get-answer"):
        return f"{task} [{filt}]"
    return f"{task} [{metric}]"


def compare(folders: list[str], names: list[str] = None):
    """Compare results across folders."""
    if names is None:
        names = [Path(f).name for f in folders]

    # Pad names to equal length
    max_name_len = max(len(n) for n in names)

    # Load all results
    all_metrics = {}
    for i, folder in enumerate(folders):
        data = load_results(folder)
        metrics = extract_metrics(data)
        all_metrics[names[i]] = metrics

    # Collect all unique metric labels
    all_labels = set()
    for metrics in all_metrics.values():
        all_labels.update(metrics.keys())

    # Sort labels by task name
    sorted_labels = sorted(all_labels)

    # Print header
    col_width = 10
    name_col = max(45, max_name_len)

    header = f"{'Metric':<{name_col}}"
    for name in names:
        header += f" {name:>{col_width}}"
    if len(names) > 1:
        header += f" {'Δ(last-first)':>{col_width+2}}"

    print()
    print("=" * len(header))
    print("lm_eval Results Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    prev_task = None
    for label in sorted_labels:
        # Get display info from first run that has this metric
        metric_info = None
        for metrics in all_metrics.values():
            if label in metrics:
                metric_info = metrics[label]
                break
        if metric_info is None:
            continue

        display = get_display_name(label, metric_info)

        # Skip stderr and alias entries
        if "stderr" in label.lower() or "alias" in label.lower():
            continue

        # Add separator between tasks
        current_task = metric_info["task"]
        if prev_task and current_task != prev_task:
            print()
        prev_task = current_task

        row = f"{display:<{name_col}}"

        values = []
        for name in names:
            metrics = all_metrics[name]
            if label in metrics:
                val = metrics[label]["value"]
                values.append(val)
                if isinstance(val, float):
                    row += f" {val:>{col_width}.4f}"
                else:
                    row += f" {val:>{col_width}}"
            else:
                values.append(None)
                row += f" {'---':>{col_width}}"

        # Delta column (last - first)
        if len(names) > 1 and values[0] is not None and values[-1] is not None:
            delta = values[-1] - values[0]
            if delta > 0:
                row += f"  {delta:>+{col_width}.4f}"
            elif delta < 0:
                row += f"  {delta:>+{col_width}.4f}"
            else:
                row += f"  {'0':>{col_width+1}}"
        elif len(names) > 1:
            row += f"  {'---':>{col_width+1}}"

        print(row)

    print("=" * len(header))

    # Summary: average across all metrics per run
    print(f"\n{'Average (all metrics)':<{name_col}}", end="")
    avgs = []
    for name in names:
        metrics = all_metrics[name]
        vals = [m["value"] for m in metrics.values()
                if isinstance(m["value"], (int, float)) and "stderr" not in m["metric"]]
        if vals:
            avg = sum(vals) / len(vals)
            avgs.append(avg)
            print(f" {avg:>{col_width}.4f}", end="")
        else:
            avgs.append(None)
            print(f" {'---':>{col_width}}", end="")

    if len(avgs) > 1 and avgs[0] is not None and avgs[-1] is not None:
        delta = avgs[-1] - avgs[0]
        print(f"  {delta:>+{col_width}.4f}", end="")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare lm_eval results across runs",
        epilog="Example: python compare_eval.py ./eval_base ./eval_rys"
    )
    parser.add_argument("folders", nargs="+", help="lm_eval output directories or JSON files")
    parser.add_argument("--names", nargs="+", help="Display names for each run (default: folder names)")
    args = parser.parse_args()

    names = args.names if args.names else None
    compare(args.folders, names)


if __name__ == "__main__":
    main()
