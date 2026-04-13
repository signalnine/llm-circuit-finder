#!/usr/bin/env python3
"""
Layer Pruning Sweep for Circuit Discovery

Instead of duplicating layers, this sweep REMOVES blocks of layers to find
which ones are least important for a target metric. Models get smaller
(faster inference) and potentially better at specific tasks.

Uses the same probe infrastructure as sweep.py.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from sweep_lib import (
    DEFAULT_PORT, wait_for_server, start_server, stop_server,
    dump_server_log, run_evaluation, load_jsonl, append_jsonl,
)


def create_pruned_gguf(source_gguf, output_gguf, prune_start, prune_end, n_layers, layer_path_script):
    """Create a GGUF with layers removed using layer_path.py."""
    # Build path that skips prune_start..prune_end-1
    parts = []
    if prune_start > 0:
        parts.append(f"0..{prune_start - 1}")
    if prune_end < n_layers:
        parts.append(f"{prune_end}..{n_layers - 1}")

    if not parts:
        print(f"  ERROR: would remove all layers!")
        return False

    layer_path = ",".join(parts)
    removed = prune_end - prune_start
    remaining = n_layers - removed

    cmd = f'python3 {layer_path_script} {source_gguf} {output_gguf} -p "{layer_path}" -v'
    print(f"  Pruning layers {prune_start}-{prune_end-1} ({removed} removed, {remaining} remaining)")
    print(f"  Layer path: {layer_path}")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  ERROR: layer_path.py failed")
        print(result.stderr[-500:] if result.stderr else "")
        return False
    return True


def print_results_table(results, baseline=None):
    """Print results table for pruning sweep."""
    print("\n" + "=" * 130)
    print(f"{'Config':>14} {'Removed':>8} {'Remain':>8} {'Math':>8} {'EQ':>8} {'Reason':>8} {'Code':>8} "
          f"{'Math Δ':>8} {'Reas Δ':>8} {'Code Δ':>8} {'Combined Δ':>11}")
    print("-" * 130)

    if baseline:
        brs = baseline.get('reasoning_score', 0)
        bcs = baseline.get('code_score', 0)
        n = baseline.get('n_layers', '?')
        print(f"{'BASELINE':>14} {'0':>8} {str(n):>8} "
              f"{baseline['math_score']:>8.4f} {baseline['eq_score']:>8.2f} {brs:>8.2%} {bcs:>8.2%} "
              f"{'---':>8} {'---':>8} {'---':>8} {'---':>11}")
        print("-" * 130)

    for r in results:
        config = f"del({r['prune_start']},{r['prune_end']})"
        removed = r['prune_end'] - r['prune_start']
        remaining = r.get('remaining_layers', '?')
        rs = r.get('reasoning_score', 0)
        cs = r.get('code_score', 0)

        if baseline:
            math_delta = r['math_score'] - baseline['math_score']
            reas_delta = rs - baseline.get('reasoning_score', 0)
            code_delta = cs - baseline.get('code_score', 0)
            combined = (reas_delta * 100) + (code_delta * 100)
            math_d = f"{math_delta:>+8.4f}"
            reas_d = f"{reas_delta:>+8.2%}"
            code_d = f"{code_delta:>+8.2%}"
            comb_d = f"{combined:>+11.2f}"
        else:
            math_d = reas_d = code_d = comb_d = "---"

        print(f"{config:>14} {removed:>8} {remaining:>8} "
              f"{r['math_score']:>8.4f} {r['eq_score']:>8.2f} {rs:>8.2%} {cs:>8.2%} "
              f"{math_d} {reas_d} {code_d} {comb_d}")

    print("=" * 130)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Layer Pruning Sweep")
    parser.add_argument("--model", required=True, help="Path to input GGUF model")
    parser.add_argument("--llama-server", required=True, help="Path to llama-server binary")
    parser.add_argument("--tmpdir", default="/dev/shm/prune",
                        help="Temp directory for modified GGUFs")
    parser.add_argument("--results", default="prune_results.jsonl",
                        help="Output results file (JSONL)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[2, 3],
                        help="Block sizes to try removing (default: 2 3)")
    parser.add_argument("--stride", type=int, default=4,
                        help="Stride between start positions")
    parser.add_argument("--start-min", type=int, default=4,
                        help="Earliest layer to start pruning (avoid early layers)")
    parser.add_argument("--start-max", type=int, default=None,
                        help="Latest layer to start pruning (avoid final layers)")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--server-args", nargs=argparse.REMAINDER, default=[],
                        help="Extra args to pass to llama-server (must be last)")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    layer_path_script = str(Path(__file__).parent / "layer_path.py")

    results_path = Path(args.results)
    baseline, results = load_jsonl(str(results_path))
    if results or baseline:
        print(f"Loaded {len(results)} existing results + baseline={baseline is not None}")

    # Get model layer count
    from gguf import GGUFReader
    reader = GGUFReader(str(model_path), 'r')
    arch_field = reader.get_field('general.architecture')
    arch = arch_field.contents()
    block_count_field = reader.get_field(f'{arch}.block_count')
    n_layers = block_count_field.contents()

    print(f"Model: {model_path.name}")
    print(f"Architecture: {arch}, Layers: {n_layers}")

    # Run baseline
    if not args.skip_baseline and baseline is None:
        print("\n>>> Running BASELINE evaluation...")
        proc = start_server(args.llama_server, str(model_path), args.port, args.server_args)
        try:
            if not wait_for_server(args.port):
                print("ERROR: Server failed to start for baseline", file=sys.stderr)
                dump_server_log(proc)
                stop_server(proc)
                sys.exit(1)

            print("  Server ready. Running probes...")
            eval_result = run_evaluation(args.port)
            baseline = {
                "is_baseline": True,
                "prune_start": -1,
                "prune_end": -1,
                "n_layers": n_layers,
                "math_score": eval_result["math_score"],
                "eq_score": eval_result["eq_score"],
                "reasoning_score": eval_result["reasoning_score"],
                "reasoning_cats": eval_result.get("reasoning_cats", {}),
                "code_score": eval_result.get("code_score", 0),
                "code_tasks": eval_result.get("code_tasks", {}),
                "timestamp": datetime.now().isoformat(),
            }

            append_jsonl(str(results_path), baseline)

            brs = baseline['reasoning_score']
            bcs = baseline.get('code_score', 0)
            print(f"  Baseline: math={baseline['math_score']:.4f} eq={baseline['eq_score']:.2f} "
                  f"reasoning={brs:.2%} code={bcs:.2%}")
        finally:
            stop_server(proc)

    # Generate pruning configs
    if args.start_max is None:
        args.start_max = n_layers - max(args.block_sizes) - 2  # avoid last 2 layers

    configs = []
    for bs in args.block_sizes:
        for start in range(args.start_min, args.start_max + 1, args.stride):
            end = start + bs
            if end <= n_layers - 2:  # always keep the last 2 layers
                configs.append((start, end))

    # Filter already-completed
    done = {(r["prune_start"], r["prune_end"]) for r in results}
    configs = [(s, e) for s, e in configs if (s, e) not in done]

    print(f"\nConfigs to test: {len(configs)}")
    if configs:
        print(f"  Range: del({configs[0][0]},{configs[0][1]}) to del({configs[-1][0]},{configs[-1][1]})")

    print_results_table(results, baseline)

    for idx, (prune_start, prune_end) in enumerate(configs):
        removed = prune_end - prune_start
        remaining = n_layers - removed
        config_str = f"del({prune_start},{prune_end})"
        print(f"\n>>> [{idx+1}/{len(configs)}] Testing {config_str} "
              f"(-{removed} layers, {remaining} remaining)...")

        modified_path = tmpdir / f"pruned_{prune_start}_{prune_end}.gguf"
        if not create_pruned_gguf(str(model_path), str(modified_path),
                                   prune_start, prune_end, n_layers, layer_path_script):
            continue

        print(f"  Starting server...")
        proc = start_server(
            args.llama_server, str(modified_path), args.port, args.server_args
        )

        try:
            if not wait_for_server(args.port):
                print(f"  ERROR: Server failed to start for {config_str}", file=sys.stderr)
                dump_server_log(proc)
                continue

            print(f"  Server ready. Running probes...")
            eval_result = run_evaluation(args.port)

            entry = {
                "prune_start": prune_start,
                "prune_end": prune_end,
                "removed_layers": removed,
                "remaining_layers": remaining,
                "math_score": eval_result["math_score"],
                "eq_score": eval_result["eq_score"],
                "reasoning_score": eval_result["reasoning_score"],
                "reasoning_cats": eval_result.get("reasoning_cats", {}),
                "code_score": eval_result.get("code_score", 0),
                "code_tasks": eval_result.get("code_tasks", {}),
                "timestamp": datetime.now().isoformat(),
            }

            results.append(entry)
            append_jsonl(str(results_path), entry)

            print_results_table(results, baseline)

        finally:
            stop_server(proc)

            if modified_path.exists():
                modified_path.unlink()
                print(f"  Cleaned up {modified_path.name}")

    print("\n\nPruning sweep complete!")
    print_results_table(results, baseline)


if __name__ == "__main__":
    main()
