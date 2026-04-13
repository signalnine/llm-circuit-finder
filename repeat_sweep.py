#!/usr/bin/env python3
"""
Duplication Count Sensitivity Sweep

Tests N-fold repetition (1x through max_repeats) of known good circuits
to find the saturation point where additional repetitions stop helping.

For each circuit, creates layer paths like:
    1x (baseline dup): 0..11,8,9,10,11,12..31
    2x:                 0..11,8,9,10,11,8,9,10,11,12..31
    3x:                 0..11,8,9,10,11,8,9,10,11,8,9,10,11,12..31

Usage:
    python repeat_sweep.py \
        --model /path/to/model.gguf \
        --llama-server /path/to/llama-server \
        --circuit 8,11 \
        --max-repeats 5 \
        --results repeat_results.jsonl
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from sweep_lib import (
    DEFAULT_PORT, get_layer_count, create_modified_gguf,
    wait_for_server, start_server, stop_server, dump_server_log,
    run_evaluation, load_jsonl, append_jsonl,
)


def build_repeat_path(n_layers: int, dup_start: int, dup_end: int,
                      n_repeats: int) -> str:
    """Build a layer path with n_repeats total executions of the dup block.

    Args:
        n_layers: total layers in original model
        dup_start: first layer of block (inclusive)
        dup_end: last layer of block (inclusive)
        n_repeats: total times the block executes (1 = original, 2 = one extra copy, etc.)
    """
    parts = []
    # Original layers up through dup_end
    parts.append(f"0..{dup_end}")
    # Extra copies
    for _ in range(n_repeats - 1):
        parts.append(f"{dup_start}..{dup_end}")
    # Remaining layers
    if dup_end < n_layers - 1:
        parts.append(f"{dup_end + 1}..{n_layers - 1}")
    return ",".join(parts)


def print_results_table(results, baseline=None):
    """Print results table for repeat sweep."""
    print("\n" + "=" * 120)
    print(f"{'Circuit':>12} {'Repeats':>8} {'Layers':>7} {'Math':>8} {'EQ':>8} {'Reason':>8} {'Code':>8} {'SWE':>8} "
          f"{'Code Δ':>8} {'SWE Δ':>8}")
    print("-" * 120)

    if baseline:
        bcs = baseline.get('code_score', 0)
        bss = baseline.get('swe_score', 0)
        print(f"{'BASELINE':>12} {'0':>8} {'---':>7} "
              f"{baseline.get('math_score',0):>8.4f} {baseline.get('eq_score',0):>8.2f} "
              f"{baseline.get('reasoning_score',0):>8.2%} {bcs:>8.2%} {bss:>8.2%} "
              f"{'---':>8} {'---':>8}")
        print("-" * 120)

    for r in sorted(results, key=lambda x: (x.get('dup_start', 0), x.get('n_repeats', 0))):
        circuit = f"({r['dup_start']},{r['dup_end']})"
        n = r.get('total_layers', '?')
        cs = r.get('code_score', 0)
        ss = r.get('swe_score', 0)

        if baseline:
            code_d = f"{cs - baseline.get('code_score', 0):>+8.2%}"
            swe_d = f"{ss - baseline.get('swe_score', 0):>+8.2%}"
        else:
            code_d = swe_d = "---"

        print(f"{circuit:>12} {r['n_repeats']:>8} {n:>7} "
              f"{r.get('math_score',0):>8.4f} {r.get('eq_score',0):>8.2f} "
              f"{r.get('reasoning_score',0):>8.2%} {cs:>8.2%} {ss:>8.2%} "
              f"{code_d} {swe_d}")

    print("=" * 120)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Duplication Count Sensitivity Sweep")
    parser.add_argument("--model", required=True, help="Path to input GGUF model")
    parser.add_argument("--llama-server", required=True, help="Path to llama-server binary")
    parser.add_argument("--tmpdir", default="/dev/shm/repeat",
                        help="Temp directory for modified GGUFs")
    parser.add_argument("--results", default="repeat_results.jsonl")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--circuits", nargs="+", required=True,
                        help="Circuits as 'start,end' (inclusive), e.g., 8,11 24,27")
    parser.add_argument("--max-repeats", type=int, default=5,
                        help="Maximum repetition count to test (default: 5)")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--server-args", nargs=argparse.REMAINDER, default=[],
                        help="Extra args to pass to llama-server (must be last)")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    n_layers, arch = get_layer_count(str(model_path))
    print(f"Model: {model_path.name}")
    print(f"Architecture: {arch}, Layers: {n_layers}")

    circuits = []
    for c in args.circuits:
        parts = c.split(",")
        circuits.append((int(parts[0]), int(parts[1])))
    print(f"Circuits: {circuits}")
    print(f"Repeat range: 1x to {args.max_repeats}x")

    results_path = Path(args.results)
    baseline, results = load_jsonl(str(results_path))
    if results or baseline:
        print(f"Loaded {len(results)} existing results + baseline={baseline is not None}")

    # Baseline
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
                **eval_result,
                "timestamp": datetime.now().isoformat(),
            }
            append_jsonl(str(results_path), baseline)
            print(f"  Baseline: code={eval_result['code_score']:.2%} swe={eval_result['swe_score']:.2%}")
        finally:
            stop_server(proc)

    # Generate configs: each circuit × each repeat count
    configs = []
    done = {(r.get("dup_start"), r.get("dup_end"), r.get("n_repeats"))
            for r in results}

    for dup_start, dup_end in circuits:
        for n_rep in range(1, args.max_repeats + 1):
            if (dup_start, dup_end, n_rep) in done:
                continue
            block_size = dup_end - dup_start + 1
            extra_layers = block_size * (n_rep - 1)  # n_rep=1 means just the original pass + 1 dup
            total = n_layers + extra_layers
            path = build_repeat_path(n_layers, dup_start, dup_end, n_rep + 1)
            # n_rep+1 because n_rep=1 means "1 extra copy" = 2 total passes
            configs.append({
                "dup_start": dup_start,
                "dup_end": dup_end,
                "n_repeats": n_rep,
                "total_layers": n_layers + block_size * n_rep,
                "layer_path": path,
            })

    print(f"\nConfigs to test: {len(configs)}")
    print_results_table(results, baseline)

    for idx, config in enumerate(configs):
        label = f"({config['dup_start']},{config['dup_end']})x{config['n_repeats']}"
        print(f"\n>>> [{idx+1}/{len(configs)}] Testing {label} "
              f"({config['total_layers']} total layers)...")
        print(f"  Layer path: {config['layer_path']}")

        modified_path = tmpdir / f"repeat_{config['dup_start']}_{config['dup_end']}_x{config['n_repeats']}.gguf"
        if not create_modified_gguf(str(model_path), str(modified_path), config["layer_path"]):
            continue

        proc = start_server(args.llama_server, str(modified_path), args.port, args.server_args)
        try:
            if not wait_for_server(args.port):
                print(f"  ERROR: Server failed to start for {label}", file=sys.stderr)
                dump_server_log(proc)
                continue

            print(f"  Server ready. Running probes...")
            eval_result = run_evaluation(args.port)

            entry = {
                "dup_start": config["dup_start"],
                "dup_end": config["dup_end"],
                "n_repeats": config["n_repeats"],
                "total_layers": config["total_layers"],
                **eval_result,
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

    print("\n\nRepeat sweep complete!")
    print_results_table(results, baseline)


if __name__ == "__main__":
    main()
