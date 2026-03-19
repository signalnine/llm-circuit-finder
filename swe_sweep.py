#!/usr/bin/env python3
"""
SWE-focused circuit sweep for Qwen3-Coder using Thunderdome tasks as the fitness function.
Creates modified GGUFs with duplicated layers, imports into ollama, runs Thunderdome tasks, scores.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd, timeout=1800, capture=True):
    """Run a shell command and return stdout."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture, text=True, timeout=timeout,
            errors="replace"
        )
        if capture:
            return result.stdout.strip(), result.returncode
        return "", result.returncode
    except subprocess.TimeoutExpired:
        print(f"  WARNING: command timed out after {timeout}s")
        return "", 1


def create_modified_gguf(source_gguf, output_gguf, dup_start, dup_end, layer_path_script):
    """Create a GGUF with duplicated layers using layer_path.py."""
    total_layers = 48  # qwen3-coder
    # Build layer path string: 0..dup_end, dup_start..dup_end, (dup_end+1)..47
    parts = []
    if dup_start > 0:
        parts.append(f"0..{dup_end}")
    parts.append(f"{dup_start}..{dup_end}")
    if dup_end < total_layers - 1:
        parts.append(f"{dup_end + 1}..{total_layers - 1}")
    layer_path = ",".join(parts)

    cmd = f"python3 {layer_path_script} {source_gguf} {output_gguf} -p \"{layer_path}\" -v"
    print(f"  Creating GGUF: layers ({dup_start},{dup_end}) path={layer_path}")
    stdout, rc = run_cmd(cmd, timeout=120)
    if rc != 0:
        print(f"  ERROR: layer_path.py failed (rc={rc})")
        print(stdout)
        return False
    return True


def import_to_ollama(gguf_path, model_name):
    """Import a GGUF into ollama."""
    modelfile = f"FROM {gguf_path}\n"
    modelfile_path = "/tmp/swe_sweep_modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile)

    # Remove old model if exists
    run_cmd(f"ollama rm {model_name} 2>/dev/null")
    stdout, rc = run_cmd(f"ollama create {model_name} -f {modelfile_path}", timeout=120)
    if rc != 0:
        print(f"  ERROR: ollama create failed: {stdout}")
        return False

    # Pre-load the model
    run_cmd(
        f'curl -s http://localhost:11434/api/generate -d \'{{"model":"{model_name}","prompt":"hi","stream":false,"options":{{"num_predict":1}}}}\' > /dev/null',
        timeout=120,
    )
    return True


def run_thunderdome_tasks(orchestrator, tasks, thunderdome_dir):
    """Run Thunderdome tasks and return per-task scores."""
    scores = {}
    for task_cat in tasks:
        cmd = (
            f"cd {thunderdome_dir} && "
            f"export PATH=$PATH:/usr/local/go/bin && "
            f"./thunderdome run --orchestrator {orchestrator} --category \"{task_cat}\" --trials 1 2>&1"
        )
        stdout, rc = run_cmd(cmd, timeout=1800, capture=True)

    # Collect scores from the latest run directories
    runs_dir = Path(thunderdome_dir) / "results" / "runs"
    # Find runs from the last 30 minutes
    cutoff = time.time() - 1800
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if run_dir.stat().st_mtime < cutoff:
            break
        for meta in run_dir.rglob("meta.json"):
            if orchestrator not in str(meta):
                continue
            task_name = meta.parent.parent.name
            with open(meta) as f:
                data = json.load(f)
            score = data.get("composite_score", 0)
            if task_name not in scores:
                scores[task_name] = score

    return scores


def main():
    parser = argparse.ArgumentParser(description="SWE-focused circuit sweep")
    parser.add_argument("--model", required=True, help="Path to source GGUF")
    parser.add_argument("--thunderdome-dir", required=True, help="Path to thunderdome repo")
    parser.add_argument("--tmpdir", default="/dev/shm/swe-sweep", help="Temp dir for GGUFs")
    parser.add_argument("--results", default="swe_sweep_results.jsonl", help="Output JSONL")
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--start-min", type=int, default=8)
    parser.add_argument("--start-max", type=int, default=36)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["greenfield/complex", "bugfix/medium"],
        help="Thunderdome task categories to use as fitness function",
    )
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    layer_path_script = str(Path(__file__).parent / "layer_path.py")
    os.makedirs(args.tmpdir, exist_ok=True)

    # Adapter and model names
    baseline_orch = "aider-local-qwen3-coder"
    circuit_orch = "aider-local-qwen3-coder-circuit"
    circuit_model = "qwen3-coder-circuit"

    # Build configs
    configs = []
    for bs in args.block_sizes:
        for start in range(args.start_min, args.start_max + 1, args.stride):
            end = start + bs
            if end > 48:
                break
            configs.append((start, end))

    print(f"Source model: {args.model}")
    print(f"Tasks: {args.tasks}")
    print(f"Configs to test: {len(configs)}")
    print(f"Range: ({configs[0][0]},{configs[0][1]}) to ({configs[-1][0]},{configs[-1][1]})")
    print()

    results = []

    # Run baseline
    if not args.skip_baseline:
        print(">>> Running BASELINE...")
        baseline_scores = run_thunderdome_tasks(
            baseline_orch, args.tasks, args.thunderdome_dir
        )
        baseline_avg = sum(baseline_scores.values()) / max(len(baseline_scores), 1)
        print(f"  Baseline scores: {baseline_scores}")
        print(f"  Baseline average: {baseline_avg:.3f}")

        entry = {
            "is_baseline": True,
            "dup_start": -1,
            "dup_end": -1,
            "task_scores": baseline_scores,
            "avg_score": baseline_avg,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        results.append(entry)
        with open(args.results, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print()

    # Sweep
    for i, (start, end) in enumerate(configs):
        bs = end - start
        label = f"({start},{end})"
        print(f">>> [{i+1}/{len(configs)}] Testing config {label} (+{bs} layers)...")

        # Create modified GGUF
        output_gguf = os.path.join(args.tmpdir, f"swe_{start}_{end}.gguf")
        if not create_modified_gguf(args.model, output_gguf, start, end, layer_path_script):
            continue

        # Import to ollama
        print(f"  Importing to ollama as {circuit_model}...")
        if not import_to_ollama(output_gguf, circuit_model):
            continue

        # Run tasks
        print(f"  Running Thunderdome tasks...")
        task_scores = run_thunderdome_tasks(
            circuit_orch, args.tasks, args.thunderdome_dir
        )
        avg_score = sum(task_scores.values()) / max(len(task_scores), 1)

        print(f"  Scores: {task_scores}")
        print(f"  Average: {avg_score:.3f}")

        entry = {
            "is_baseline": False,
            "dup_start": start,
            "dup_end": end,
            "task_scores": task_scores,
            "avg_score": avg_score,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        results.append(entry)
        with open(args.results, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Cleanup GGUF
        try:
            os.remove(output_gguf)
        except OSError:
            pass

        print()

    # Print summary
    print("\n" + "=" * 80)
    baseline_entry = next((r for r in results if r.get("is_baseline")), None)
    b_avg = baseline_entry["avg_score"] if baseline_entry else 0

    print(f"{'Config':>12s}  {'Layers':>6s}  {'Avg Score':>9s}  {'Delta':>8s}  Task Breakdown")
    print("-" * 80)
    if baseline_entry:
        print(
            f"{'BASELINE':>12s}  {'0':>6s}  {b_avg:>9.3f}  {'---':>8s}  {baseline_entry['task_scores']}"
        )
        print("-" * 80)

    sorted_results = sorted(
        [r for r in results if not r.get("is_baseline")],
        key=lambda r: r["avg_score"],
        reverse=True,
    )
    for r in sorted_results:
        label = f"({r['dup_start']},{r['dup_end']})"
        bs = r["dup_end"] - r["dup_start"]
        delta = r["avg_score"] - b_avg
        marker = " <<<" if delta > 0.05 else ""
        print(
            f"{label:>12s}  {f'+{bs}':>6s}  {r['avg_score']:>9.3f}  {delta:>+8.3f}{marker}  {r['task_scores']}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
