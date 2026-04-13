#!/usr/bin/env python3
"""
Combined Prune + Duplicate Sweep

Tests simultaneous pruning AND duplication on a single model. For example,
prune interfering layers (28-30) while duplicating beneficial ones (12-15)
in Qwen3-Coder to see if the effects compose.

Uses layer_path.py to build combined paths like:
    "0..14,12,13,14,15..27,31..47"
    (duplicates 12-14, prunes 28-30)

Usage:
    python combined_sweep.py \
        --model /path/to/model.gguf \
        --llama-server /path/to/llama-server \
        --prune-ranges 28,30 32,34 \
        --dup-ranges 8,11 12,15 \
        --results combined_results.jsonl
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from sweep_lib import (
    DEFAULT_PORT, get_layer_count, create_modified_gguf,
    wait_for_server, start_server, stop_server, dump_server_log,
    run_evaluation, load_jsonl, append_jsonl,
)


def parse_range(s: str) -> tuple[int, int]:
    """Parse 'start,end' into (start, end) tuple."""
    parts = s.split(",")
    return int(parts[0]), int(parts[1])


def build_combined_path(n_layers: int, prune_start: int, prune_end: int,
                        dup_start: int, dup_end: int) -> str:
    """Build a layer path that prunes one region and duplicates another.

    Works by constructing the full layer sequence explicitly:
    1. Start with all original layers
    2. Remove the prune region
    3. Insert the dup region copy at the right position

    Args:
        n_layers: total layers in original model
        prune_start: first layer to remove (inclusive)
        prune_end: last layer to remove (exclusive)
        dup_start: first layer to duplicate (inclusive)
        dup_end: last layer to duplicate (exclusive)

    Returns:
        layer_path.py compatible path string
    """
    # Build the sequence: all layers except pruned, with dup block inserted after dup_end
    layers = []
    for i in range(n_layers):
        if prune_start <= i < prune_end:
            continue  # skip pruned layers
        layers.append(i)
        # After the last layer of the dup block, insert the duplicate
        if i == dup_end - 1:
            for j in range(dup_start, dup_end):
                if prune_start <= j < prune_end:
                    continue  # don't duplicate pruned layers
                layers.append(j)

    # Compress into range notation
    return compress_layer_list(layers)


def compress_layer_list(layers: list[int]) -> str:
    """Compress a layer list into range notation (e.g., [0,1,2,5,6] -> '0..2,5..6')."""
    if not layers:
        return ""
    parts = []
    run_start = layers[0]
    run_end = layers[0]
    for i in range(1, len(layers)):
        if layers[i] == run_end + 1:
            run_end = layers[i]
        else:
            if run_start == run_end:
                parts.append(str(run_start))
            else:
                parts.append(f"{run_start}..{run_end}")
            run_start = layers[i]
            run_end = layers[i]
    if run_start == run_end:
        parts.append(str(run_start))
    else:
        parts.append(f"{run_start}..{run_end}")
    return ",".join(parts)


def generate_combined_configs(prune_ranges, dup_ranges, n_layers):
    """Generate all non-overlapping (prune, dup) combinations."""
    configs = []
    for ps, pe in prune_ranges:
        for ds, de in dup_ranges:
            # Check for overlap
            if ds < pe and ps < de:
                continue  # ranges overlap, skip
            path = build_combined_path(n_layers, ps, pe, ds, de)
            configs.append({
                "prune_start": ps, "prune_end": pe,
                "dup_start": ds, "dup_end": de,
                "layer_path": path,
            })
    return configs


def print_results_table(results, baseline=None):
    """Print results table for combined sweep."""
    print("\n" + "=" * 140)
    print(f"{'Config':>24} {'Layers':>7} {'Math':>8} {'EQ':>8} {'Reason':>8} {'Code':>8} {'SWE':>8} "
          f"{'Code Δ':>8} {'SWE Δ':>8}")
    print("-" * 140)

    if baseline:
        bcs = baseline.get('code_score', 0)
        bss = baseline.get('swe_score', 0)
        print(f"{'BASELINE':>24} {'---':>7} "
              f"{baseline.get('math_score',0):>8.4f} {baseline.get('eq_score',0):>8.2f} "
              f"{baseline.get('reasoning_score',0):>8.2%} {bcs:>8.2%} {bss:>8.2%} "
              f"{'---':>8} {'---':>8}")
        print("-" * 140)

    for r in results:
        config = f"del({r['prune_start']},{r['prune_end']})+dup({r['dup_start']},{r['dup_end']})"
        n = r.get('n_layers', '?')
        cs = r.get('code_score', 0)
        ss = r.get('swe_score', 0)

        if baseline:
            code_d = f"{cs - baseline.get('code_score', 0):>+8.2%}"
            swe_d = f"{ss - baseline.get('swe_score', 0):>+8.2%}"
        else:
            code_d = swe_d = "---"

        print(f"{config:>24} {n:>7} "
              f"{r.get('math_score',0):>8.4f} {r.get('eq_score',0):>8.2f} "
              f"{r.get('reasoning_score',0):>8.2%} {cs:>8.2%} {ss:>8.2%} "
              f"{code_d} {swe_d}")

    print("=" * 140)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Combined Prune + Duplicate Sweep")
    parser.add_argument("--model", required=True, help="Path to input GGUF model")
    parser.add_argument("--llama-server", required=True, help="Path to llama-server binary")
    parser.add_argument("--tmpdir", default="/dev/shm/combined",
                        help="Temp directory for modified GGUFs")
    parser.add_argument("--results", default="combined_results.jsonl")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--prune-ranges", nargs="+", required=True,
                        help="Prune ranges as 'start,end' (exclusive end), e.g., 28,31 32,35")
    parser.add_argument("--dup-ranges", nargs="+", required=True,
                        help="Dup ranges as 'start,end' (exclusive end), e.g., 8,12 12,16")
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

    prune_ranges = [parse_range(r) for r in args.prune_ranges]
    dup_ranges = [parse_range(r) for r in args.dup_ranges]

    print(f"Prune regions: {prune_ranges}")
    print(f"Dup regions: {dup_ranges}")

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

    # Generate configs
    configs = generate_combined_configs(prune_ranges, dup_ranges, n_layers)

    # Filter already-completed
    done = {(r["prune_start"], r["prune_end"], r["dup_start"], r["dup_end"]) for r in results}
    configs = [c for c in configs
               if (c["prune_start"], c["prune_end"], c["dup_start"], c["dup_end"]) not in done]

    print(f"\nConfigs to test: {len(configs)}")
    print_results_table(results, baseline)

    for idx, config in enumerate(configs):
        label = f"del({config['prune_start']},{config['prune_end']})+dup({config['dup_start']},{config['dup_end']})"
        print(f"\n>>> [{idx+1}/{len(configs)}] Testing {label}...")
        print(f"  Layer path: {config['layer_path']}")

        modified_path = tmpdir / f"combined_{config['prune_start']}_{config['prune_end']}_{config['dup_start']}_{config['dup_end']}.gguf"
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
                **config,
                **eval_result,
                "n_layers": len(config["layer_path"].replace("..", ",").split(",")),  # approximate
                "timestamp": datetime.now().isoformat(),
            }
            # Recount layers properly
            path_layers = []
            for part in config["layer_path"].split(","):
                if ".." in part:
                    a, b = part.split("..")
                    path_layers.extend(range(int(a), int(b) + 1))
                else:
                    path_layers.append(int(part))
            entry["n_layers"] = len(path_layers)

            results.append(entry)
            append_jsonl(str(results_path), entry)
            print_results_table(results, baseline)

        finally:
            stop_server(proc)
            if modified_path.exists():
                modified_path.unlink()
                print(f"  Cleaned up {modified_path.name}")

    print("\n\nCombined sweep complete!")
    print_results_table(results, baseline)


if __name__ == "__main__":
    main()
