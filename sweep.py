#!/usr/bin/env python3
"""
RYS Layer Sweep

Supports four modes:
    add            - duplicate a block of layers (classic RYS)
    remove         - skip (prune) a block of layers
    replace-next   - replace layer N with layer N+1 (swap)
    replace-same   - replace with next same-type layer (hybrid-aware,
                     skips attention layers as replacement source)

Usage:
    python sweep.py --model M.gguf --llama-server llama-server \\
        --mode add --block-sizes 8 --stride 4 \\
        --results rys.jsonl --probe-dir probes/
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from sweep_lib import (
    DEFAULT_PORT, wait_for_server, start_server, stop_server,
    dump_server_log, run_evaluation, load_jsonl, append_jsonl,
    create_modified_gguf, get_model_info, is_attention_layer,
)


def build_path_for_mode(mode: str, n_layers: int, a: int, b: int) -> str:
    """Build a layer_path string for the given mode.

    add/remove: a=start, b=end (exclusive).
    replace-*:  a=target layer, b=replacement layer.
    """
    if mode == "add":
        layers = list(range(b)) + list(range(a, b)) + list(range(b, n_layers))
    elif mode == "remove":
        layers = list(range(a)) + list(range(b, n_layers))
    else:  # replace-next / replace-same
        layers = list(range(n_layers))
        layers[a] = b
    return ",".join(str(x) for x in layers)


def print_results_table(results, baseline=None):
    print("\n" + "=" * 128)
    print(f"{'Mode':>12} {'Config':>10} {'Math':>8} {'EQ':>8} {'Reason':>8} "
          f"{'Code':>8} {'Gen':>8} {'Math Δ':>8} {'Reas Δ':>8} {'Code Δ':>8} {'Gen Δ':>8}")
    print("-" * 128)
    if baseline:
        brs = baseline.get('reasoning_score', 0)
        bcs = baseline.get('code_score', 0)
        bgs = baseline.get('general_score', 0)
        print(f"{'BASELINE':>12} {'---':>10} "
              f"{baseline['math_score']:>8.4f} {baseline['eq_score']:>8.2f} "
              f"{brs:>8.2%} {bcs:>8.2%} {bgs:>8.2%} "
              f"{'---':>8} {'---':>8} {'---':>8} {'---':>8}")
        print("-" * 128)
    for r in results:
        mode = r.get('mode', 'add')
        config = f"({r['dup_start']},{r['dup_end']})"
        rs = r.get('reasoning_score', 0)
        cs = r.get('code_score', 0)
        gs = r.get('general_score', 0)
        if baseline:
            md = f"{r['math_score'] - baseline['math_score']:>+8.4f}"
            rd = f"{rs - baseline.get('reasoning_score', 0):>+8.2%}"
            cd = f"{cs - baseline.get('code_score', 0):>+8.2%}"
            gd = f"{gs - baseline.get('general_score', 0):>+8.2%}"
        else:
            md = rd = cd = gd = "---"
        print(f"{mode:>12} {config:>10} "
              f"{r['math_score']:>8.4f} {r['eq_score']:>8.2f} "
              f"{rs:>8.2%} {cs:>8.2%} {gs:>8.2%} "
              f"{md} {rd} {cd} {gd}")
    print("=" * 128)
    sys.stdout.flush()


def generate_sweep_configs(n_layers, block_sizes, start_min=4,
                           start_max=None, stride=4):
    """For add/remove modes: (start, end) where end is exclusive."""
    if start_max is None:
        start_max = n_layers - max(block_sizes) - 4
    configs = []
    for bs in block_sizes:
        for start in range(start_min, start_max + 1, stride):
            end = start + bs
            if end <= n_layers:
                configs.append((start, end))
    return configs


def generate_replace_configs(n_layers, full_attention_interval, mode,
                             start_min=1, start_max=None):
    """For replace-*: (target, replacement)."""
    if start_max is None:
        start_max = n_layers - 2
    configs = []
    for target in range(start_min, start_max + 1):
        replacement = target + 1
        if mode == "replace-same":
            while (replacement < n_layers
                   and is_attention_layer(replacement, full_attention_interval)):
                replacement += 1
        if replacement < n_layers:
            configs.append((target, replacement))
    return configs


def main():
    p = argparse.ArgumentParser(description="RYS Layer Sweep")
    p.add_argument("--model", required=True)
    p.add_argument("--llama-server", required=True)
    p.add_argument("--mode", choices=["add", "remove", "replace-next", "replace-same"],
                   default="add")
    p.add_argument("--tmpdir", default="/dev/shm/rys")
    p.add_argument("--results", default="rys_results.jsonl")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--block-sizes", type=int, nargs="+", default=[8])
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--start-min", type=int, default=4)
    p.add_argument("--start-max", type=int, default=None)
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--probe-dir", default=None,
                   help="Optional dir of general_*.json probe categories")
    p.add_argument("--server-args", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    model_path = Path(args.model).resolve()
    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results)

    info = get_model_info(str(model_path))
    arch = info["architecture"]
    n_layers = info["block_count"]
    fai = info["full_attention_interval"]
    print(f"Model: {model_path.name}")
    print(f"Architecture: {arch}, Layers: {n_layers}")
    if fai:
        print(f"Hybrid model: full_attention_interval={fai}")

    baseline, results = load_jsonl(str(results_path))
    if results or baseline:
        print(f"Loaded {len(results)} existing results + baseline={baseline is not None}")

    if not args.skip_baseline and baseline is None:
        print("\n>>> Running BASELINE evaluation...")
        proc = start_server(args.llama_server, str(model_path), args.port, args.server_args)
        try:
            if not wait_for_server(args.port):
                print("ERROR: baseline server start failed", file=sys.stderr)
                dump_server_log(proc)
                stop_server(proc)
                sys.exit(1)
            eval_result = run_evaluation(args.port, args.probe_dir)
            baseline = {
                "is_baseline": True,
                "dup_start": -1, "dup_end": -1,
                **eval_result,
                "timestamp": datetime.now().isoformat(),
            }
            append_jsonl(str(results_path), baseline)
            print(f"  Baseline: math={baseline['math_score']:.4f} "
                  f"eq={baseline['eq_score']:.2f} "
                  f"reasoning={baseline['reasoning_score']:.2%} "
                  f"code={baseline.get('code_score',0):.2%}")
        finally:
            stop_server(proc)

    if args.mode in ("replace-next", "replace-same"):
        configs = generate_replace_configs(n_layers, fai, args.mode,
                                           args.start_min, args.start_max)
    else:
        configs = generate_sweep_configs(n_layers, args.block_sizes,
                                         args.start_min, args.start_max, args.stride)

    done = {(r.get("mode", "add"), r["dup_start"], r["dup_end"]) for r in results}
    configs = [(a, b) for a, b in configs if (args.mode, a, b) not in done]

    print(f"\nMode: {args.mode}")
    print(f"Configs to test: {len(configs)}")
    print_results_table(results, baseline)

    for idx, (a, b) in enumerate(configs):
        if args.mode == "add":
            desc = f"+{b - a} layers"
        elif args.mode == "remove":
            desc = f"-{b - a} layers"
        else:
            desc = f"layer {a} -> {b}"
        print(f"\n>>> [{idx+1}/{len(configs)}] {args.mode} ({a},{b}) — {desc}")

        modified = tmpdir / f"rys_{args.mode}_{a}_{b}.gguf"
        path_str = build_path_for_mode(args.mode, n_layers, a, b)
        if not create_modified_gguf(str(model_path), str(modified), path_str):
            continue

        proc = start_server(args.llama_server, str(modified), args.port, args.server_args)
        try:
            if not wait_for_server(args.port):
                print(f"  ERROR: server start failed", file=sys.stderr)
                dump_server_log(proc)
                continue
            eval_result = run_evaluation(args.port, args.probe_dir)
            entry = {
                "mode": args.mode,
                "dup_start": a, "dup_end": b,
                **eval_result,
                "timestamp": datetime.now().isoformat(),
            }
            results.append(entry)
            append_jsonl(str(results_path), entry)
            print_results_table(results, baseline)
        finally:
            stop_server(proc)
            if modified.exists():
                modified.unlink()

    print("\nSweep complete!")
    print_results_table(results, baseline)


if __name__ == "__main__":
    main()
