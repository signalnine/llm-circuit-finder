#!/usr/bin/env python3
"""
SWE Benchmark Sweep — uses Thunderdome + Aider as the fitness function.

Creates modified GGUFs (pruned or duplicated layers), imports into ollama
with the correct template, runs Thunderdome tasks via aider, and scores.

Usage:
    python swe_sweep.py \
        --model /path/to/model.gguf \
        --model-name qwen3-coder \
        --thunderdome-dir /path/to/thunderdome \
        --tasks "greenfield/simple" "bugfix/medium" "features/medium" \
        --prune-blocks 2 --dup-blocks 3 \
        --stride 4 --start-min 8 --start-max 36
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd, timeout=1800):
    """Run a shell command, return (stdout, returncode)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, errors="replace"
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        print(f"  WARNING: timed out after {timeout}s", flush=True)
        return "", 1


def get_layer_count(gguf_path):
    """Read layer count from GGUF metadata."""
    from gguf import GGUFReader
    reader = GGUFReader(gguf_path, 'r')
    arch = reader.get_field('general.architecture').contents()
    n_layers = reader.get_field(f'{arch}.block_count').contents()
    return n_layers, arch


def build_layer_path(mode, start, end, n_layers):
    """Build layer_path.py path string for prune or dup."""
    if mode == "prune":
        parts = []
        if start > 0:
            parts.append(f"0..{start - 1}")
        if end < n_layers:
            parts.append(f"{end}..{n_layers - 1}")
        return ",".join(parts)
    else:  # dup
        parts = []
        if start > 0:
            parts.append(f"0..{end}")
        parts.append(f"{start}..{end}")
        if end < n_layers - 1:
            parts.append(f"{end + 1}..{n_layers - 1}")
        return ",".join(parts)


def create_modified_gguf(source, output, mode, start, end, n_layers):
    """Create pruned or duplicated GGUF."""
    layer_path_script = str(Path(__file__).parent / "layer_path.py")
    path_str = build_layer_path(mode, start, end, n_layers)
    cmd = f'python3 {layer_path_script} "{source}" "{output}" -p "{path_str}"'
    _, rc = run_cmd(cmd, timeout=180)
    return rc == 0


def import_to_ollama(gguf_path, model_name, template_model):
    """Import GGUF into ollama, copying template from an existing model."""
    # Get template from the original model
    stdout, rc = run_cmd(f"ollama show {template_model} --modelfile 2>/dev/null")
    if rc != 0:
        print(f"  WARNING: can't get template from {template_model}", flush=True)
        modelfile = f"FROM {gguf_path}\n"
    else:
        # Replace the FROM line with our GGUF
        lines = stdout.split("\n")
        new_lines = []
        for line in lines:
            if line.startswith("FROM ") and not line.startswith("# FROM"):
                new_lines.append(f"FROM {gguf_path}")
            elif not line.startswith("#"):
                new_lines.append(line)
        modelfile = "\n".join(new_lines)

    mf_path = "/tmp/swe_sweep_modelfile"
    with open(mf_path, "w") as f:
        f.write(modelfile)

    run_cmd(f"ollama rm {model_name} 2>/dev/null", timeout=30)
    _, rc = run_cmd(f"ollama create {model_name} -f {mf_path}", timeout=180)
    if rc != 0:
        return False

    # Pre-load
    run_cmd(
        f'curl -s http://localhost:11434/api/generate -d \'{{"model":"{model_name}","prompt":"hi","stream":false,"options":{{"num_predict":1}}}}\' > /dev/null',
        timeout=120,
    )
    return True


def run_thunderdome(orchestrator, task_categories, thunderdome_dir, run_id):
    """Run Thunderdome tasks and collect scores."""
    scores = {}
    for cat in task_categories:
        cmd = (
            f"cd {thunderdome_dir} && "
            f"export PATH=$PATH:/usr/local/go/bin && "
            f"./thunderdome run --orchestrator {orchestrator} --category \"{cat}\" --trials 1 2>&1"
        )
        run_cmd(cmd, timeout=1800)

    # Collect scores from most recent runs
    runs_dir = Path(thunderdome_dir) / "results" / "runs"
    cutoff = time.time() - 3600  # last hour
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        try:
            if run_dir.stat().st_mtime < cutoff:
                break
        except OSError:
            continue
        for meta in run_dir.rglob("meta.json"):
            if orchestrator not in str(meta):
                continue
            task_name = meta.parent.parent.name
            with open(meta) as f:
                data = json.load(f)
            score = data.get("composite_score", 0)
            tokens = data.get("input_tokens", 0) + data.get("output_tokens", 0)
            if task_name not in scores:
                scores[task_name] = score

    return scores


def main():
    parser = argparse.ArgumentParser(description="SWE Benchmark Sweep")
    parser.add_argument("--model", required=True, help="Path to source GGUF")
    parser.add_argument("--model-name", required=True,
                        help="Ollama model name for template (e.g., qwen3-coder)")
    parser.add_argument("--thunderdome-dir", required=True)
    parser.add_argument("--tmpdir", default="/mnt/ai/tmp/swe-sweep")
    parser.add_argument("--results", default="swe_sweep_results.jsonl")
    parser.add_argument("--tasks", nargs="+",
                        default=["greenfield/simple", "bugfix/medium", "features/medium"])
    parser.add_argument("--prune-blocks", type=int, nargs="*", default=[2],
                        help="Block sizes for pruning (empty to skip)")
    parser.add_argument("--dup-blocks", type=int, nargs="*", default=[3],
                        help="Block sizes for duplication (empty to skip)")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--start-min", type=int, default=8)
    parser.add_argument("--start-max", type=int, default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.tmpdir, exist_ok=True)
    n_layers, arch = get_layer_count(args.model)

    if args.start_max is None:
        args.start_max = n_layers - max(
            max(args.prune_blocks or [0]), max(args.dup_blocks or [0])
        ) - 4

    # Orchestrator names
    baseline_orch = f"aider-local-{args.model_name}"
    circuit_orch = f"aider-local-{args.model_name}-circuit"
    circuit_ollama = f"{args.model_name}-sweep-circuit"

    print(f"Model: {args.model}", flush=True)
    print(f"Architecture: {arch}, Layers: {n_layers}", flush=True)
    print(f"Template model: {args.model_name}", flush=True)
    print(f"Tasks: {args.tasks}", flush=True)
    print(f"Baseline orchestrator: {baseline_orch}", flush=True)
    print(f"Circuit orchestrator: {circuit_orch}", flush=True)

    # Ensure circuit orchestrator adapter exists
    td = Path(args.thunderdome_dir)
    circuit_adapter_dir = td / "adapters" / circuit_orch
    if not circuit_adapter_dir.exists():
        # Create it from the baseline adapter
        baseline_adapter_dir = td / "adapters" / baseline_orch
        if baseline_adapter_dir.exists():
            circuit_adapter_dir.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(baseline_adapter_dir / "adapter.sh", circuit_adapter_dir / "adapter.sh")
            # Replace model name in adapter
            adapter_text = (circuit_adapter_dir / "adapter.sh").read_text()
            adapter_text = adapter_text.replace(
                f"openai/{args.model_name}",
                f"openai/{circuit_ollama}"
            )
            (circuit_adapter_dir / "adapter.sh").write_text(adapter_text)
            print(f"Created circuit adapter: {circuit_adapter_dir}", flush=True)

    results_path = Path(args.results)
    results = []
    baseline = None

    # Load existing
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("is_baseline"):
                    baseline = entry
                else:
                    results.append(entry)

    # Baseline
    if not args.skip_baseline and baseline is None:
        print(f"\n>>> BASELINE ({baseline_orch})...", flush=True)
        scores = run_thunderdome(baseline_orch, args.tasks, args.thunderdome_dir, "baseline")
        avg = sum(scores.values()) / max(len(scores), 1)
        baseline = {
            "is_baseline": True, "task_scores": scores, "avg_score": avg,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(results_path, "a") as f:
            f.write(json.dumps(baseline) + "\n")
        print(f"  Scores: {scores}", flush=True)
        print(f"  Average: {avg:.3f}", flush=True)

    b_avg = baseline["avg_score"] if baseline else 0

    # Build configs
    configs = []
    done = {(r.get("mode"), r.get("start"), r.get("end")) for r in results}
    for bs in (args.prune_blocks or []):
        for start in range(args.start_min, args.start_max + 1, args.stride):
            end = start + bs
            if end <= n_layers - 2 and ("prune", start, end) not in done:
                configs.append(("prune", start, end))
    for bs in (args.dup_blocks or []):
        for start in range(args.start_min, args.start_max + 1, args.stride):
            end = start + bs
            if end <= n_layers and ("dup", start, end) not in done:
                configs.append(("dup", start, end))

    print(f"\nConfigs to test: {len(configs)}", flush=True)

    for i, (mode, start, end) in enumerate(configs):
        label = f"{'del' if mode == 'prune' else 'dup'}({start},{end})"
        print(f"\n>>> [{i+1}/{len(configs)}] {label} ({mode})...", flush=True)

        # Create modified GGUF
        output = os.path.join(args.tmpdir, f"sweep_{mode}_{start}_{end}.gguf")
        if not create_modified_gguf(args.model, output, mode, start, end, n_layers):
            print("  GGUF failed", flush=True)
            continue

        # Import to ollama with template
        print(f"  Importing to ollama as {circuit_ollama}...", flush=True)
        if not import_to_ollama(output, circuit_ollama, args.model_name):
            print("  Ollama import failed", flush=True)
            try: os.remove(output)
            except: pass
            continue

        # Run Thunderdome
        print("  Running Thunderdome tasks...", flush=True)
        scores = run_thunderdome(circuit_orch, args.tasks, args.thunderdome_dir, label)
        avg = sum(scores.values()) / max(len(scores), 1)
        delta = avg - b_avg

        print(f"  Scores: {scores}", flush=True)
        print(f"  Average: {avg:.3f} ({delta:+.3f})", flush=True)

        entry = {
            "is_baseline": False, "mode": mode, "start": start, "end": end,
            "label": label, "task_scores": scores, "avg_score": avg,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        results.append(entry)
        with open(results_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Cleanup
        try: os.remove(output)
        except: pass

        # Unload from ollama
        run_cmd(f'curl -s http://localhost:11434/api/generate -d \'{{"model":"{circuit_ollama}","keep_alive":0}}\' > /dev/null', timeout=30)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print(f"{'Config':>14s}  {'Avg':>7s}  {'Delta':>7s}  Tasks", flush=True)
    print("-" * 70, flush=True)
    if baseline:
        print(f"{'BASELINE':>14s}  {b_avg:>7.3f}  {'---':>7s}  {baseline['task_scores']}", flush=True)
    print("-" * 70, flush=True)
    for r in sorted(results, key=lambda x: x["avg_score"], reverse=True):
        d = r["avg_score"] - b_avg
        marker = " <<<" if d > 0.03 else ""
        print(f"{r['label']:>14s}  {r['avg_score']:>7.3f}  {d:>+7.3f}{marker}  {r['task_scores']}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
