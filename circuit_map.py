#!/usr/bin/env python3
"""
Cross-Task Circuit Mapping

Fine-grained sweep (stride=1, small block sizes) with ALL probes on every config.
Produces per-probe deltas for each layer range, enabling identification of
task-specific circuits.

Output: JSONL with all probe scores per config, suitable for heatmap visualization
via `visualize.py --heatmap`.

Usage:
    python circuit_map.py \
        --model /path/to/model.gguf \
        --llama-server /path/to/llama-server \
        --mode dup \
        --block-sizes 1 2 3 \
        --results circuit_map_results.jsonl

    # Visualize the results
    python visualize.py --heatmap circuit_map_results.jsonl
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from sweep_lib import (
    DEFAULT_PORT, get_layer_count, create_modified_gguf, build_layer_path,
    wait_for_server, start_server, stop_server, dump_server_log,
    run_evaluation, load_jsonl, append_jsonl,
)


def generate_configs(n_layers: int, mode: str, block_sizes: list[int],
                     start_min: int, start_max: int, stride: int) -> list[dict]:
    """Generate fine-grained sweep configs."""
    configs = []
    for bs in block_sizes:
        for start in range(start_min, start_max + 1, stride):
            if mode == "prune":
                end = start + bs
                if end > n_layers - 2:
                    continue
                path = build_layer_path("prune", start, end, n_layers)
            else:  # dup
                end = start + bs - 1  # inclusive end for dup
                if end >= n_layers:
                    continue
                path = build_layer_path("dup", start, end, n_layers)

            configs.append({
                "mode": mode,
                "start": start,
                "end": start + bs,  # exclusive end for consistency
                "block_size": bs,
                "layer_path": path,
            })
    return configs


def print_results_table(results, baseline=None):
    """Print compact results table."""
    print("\n" + "=" * 130)
    print(f"{'Config':>16} {'BS':>3} {'Math':>8} {'EQ':>8} {'Reason':>8} {'Code':>8} {'SWE':>8} "
          f"{'Math Δ':>8} {'Reas Δ':>8} {'Code Δ':>8} {'SWE Δ':>8}")
    print("-" * 130)

    if baseline:
        print(f"{'BASELINE':>16} {'':>3} "
              f"{baseline.get('math_score',0):>8.4f} {baseline.get('eq_score',0):>8.2f} "
              f"{baseline.get('reasoning_score',0):>8.2%} {baseline.get('code_score',0):>8.2%} "
              f"{baseline.get('swe_score',0):>8.2%} "
              f"{'---':>8} {'---':>8} {'---':>8} {'---':>8}")
        print("-" * 130)

    for r in sorted(results, key=lambda x: (x.get('block_size', 0), x.get('start', 0))):
        prefix = "del" if r.get('mode') == 'prune' else "dup"
        config = f"{prefix}({r['start']},{r['end']})"
        bs = r.get('block_size', '?')

        if baseline:
            md = f"{r.get('math_score',0) - baseline.get('math_score',0):>+8.4f}"
            rd = f"{r.get('reasoning_score',0) - baseline.get('reasoning_score',0):>+8.2%}"
            cd = f"{r.get('code_score',0) - baseline.get('code_score',0):>+8.2%}"
            sd = f"{r.get('swe_score',0) - baseline.get('swe_score',0):>+8.2%}"
        else:
            md = rd = cd = sd = "---"

        print(f"{config:>16} {bs:>3} "
              f"{r.get('math_score',0):>8.4f} {r.get('eq_score',0):>8.2f} "
              f"{r.get('reasoning_score',0):>8.2%} {r.get('code_score',0):>8.2%} "
              f"{r.get('swe_score',0):>8.2%} "
              f"{md} {rd} {cd} {sd}")

    print("=" * 130)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Cross-Task Circuit Mapping")
    parser.add_argument("--model", required=True, help="Path to input GGUF model")
    parser.add_argument("--llama-server", required=True, help="Path to llama-server binary")
    parser.add_argument("--tmpdir", default="/dev/shm/circuit_map",
                        help="Temp directory for modified GGUFs")
    parser.add_argument("--results", default="circuit_map_results.jsonl")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--mode", choices=["dup", "prune"], default="dup",
                        help="Surgery mode (default: dup)")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1, 2, 3],
                        help="Block sizes to sweep (default: 1 2 3)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride between start positions (default: 1)")
    parser.add_argument("--start-min", type=int, default=4)
    parser.add_argument("--start-max", type=int, default=None)
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
    print(f"Mode: {args.mode}, Block sizes: {args.block_sizes}, Stride: {args.stride}")

    if args.start_max is None:
        args.start_max = n_layers - max(args.block_sizes) - 2

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
            print(f"  Baseline: math={eval_result['math_score']:.4f} "
                  f"code={eval_result['code_score']:.2%} swe={eval_result['swe_score']:.2%}")
        finally:
            stop_server(proc)

    # Generate configs
    configs = generate_configs(n_layers, args.mode, args.block_sizes,
                               args.start_min, args.start_max, args.stride)

    # Filter already-completed
    done = {(r.get("mode"), r.get("start"), r.get("end")) for r in results}
    configs = [c for c in configs if (c["mode"], c["start"], c["end"]) not in done]

    print(f"\nConfigs to test: {len(configs)}")
    est_minutes = len(configs) * 2  # ~2 min per config
    print(f"Estimated time: ~{est_minutes} minutes ({est_minutes / 60:.1f} hours)")
    print_results_table(results, baseline)

    for idx, config in enumerate(configs):
        prefix = "del" if config['mode'] == 'prune' else "dup"
        label = f"{prefix}({config['start']},{config['end']})"
        print(f"\n>>> [{idx+1}/{len(configs)}] Testing {label} (bs={config['block_size']})...")

        modified_path = tmpdir / f"cmap_{config['mode']}_{config['start']}_{config['end']}.gguf"
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
                "timestamp": datetime.now().isoformat(),
            }
            # Remove layer_path from stored results (it's derivable)
            entry.pop("layer_path", None)

            results.append(entry)
            append_jsonl(str(results_path), entry)

            # Print progress summary
            if baseline:
                cd = eval_result['code_score'] - baseline.get('code_score', 0)
                sd = eval_result['swe_score'] - baseline.get('swe_score', 0)
                print(f"  Result: code={eval_result['code_score']:.2%} ({cd:+.2%}) "
                      f"swe={eval_result['swe_score']:.2%} ({sd:+.2%})")

        finally:
            stop_server(proc)
            if modified_path.exists():
                modified_path.unlink()

    print("\n\nCircuit mapping complete!")
    print_results_table(results, baseline)
    print(f"\nVisualize with: python visualize.py --heatmap {results_path}")


if __name__ == "__main__":
    main()
