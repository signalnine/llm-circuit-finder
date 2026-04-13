#!/usr/bin/env python3
"""Test compositions of Qwen3.5 hot circuits via multi-layer swap."""
import argparse
import sys
from datetime import datetime
from pathlib import Path

from sweep_lib import (
    DEFAULT_PORT, wait_for_server, start_server, stop_server,
    dump_server_log, run_evaluation, append_jsonl, create_modified_gguf,
    get_model_info,
)


# Each combo: list of (target, replacement) swaps to apply together.
COMBOS = {
    "code14+swe18":       [(14, 16), (18, 20)],
    "code14+swe24":       [(14, 16), (24, 25)],
    "swe18+swe24":        [(18, 20), (24, 25)],
    "code14+swe18+swe24": [(14, 16), (18, 20), (24, 25)],
    "code14+reas21":      [(14, 16), (21, 22)],
    "code14+gen22":       [(14, 16), (22, 24)],
    "code14+swe18+reas21": [(14, 16), (18, 20), (21, 22)],
}


def build_multiswap_path(n_layers: int, swaps: list[tuple[int, int]]) -> str:
    layers = list(range(n_layers))
    for tgt, rep in swaps:
        layers[tgt] = rep
    return ",".join(str(x) for x in layers)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--llama-server", required=True)
    p.add_argument("--tmpdir", default="/dev/shm/rys")
    p.add_argument("--results", default="qwen35_combo.jsonl")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--probe-dir", default="probes/")
    p.add_argument("--server-args", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    info = get_model_info(args.model)
    n_layers = info["block_count"]
    print(f"Model: {info['architecture']}, {n_layers} layers, fai={info['full_attention_interval']}")

    tmpdir = Path(args.tmpdir); tmpdir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results)

    for name, swaps in COMBOS.items():
        print(f"\n>>> {name}: swaps={swaps}")
        path_str = build_multiswap_path(n_layers, swaps)
        modified = tmpdir / f"combo_{name}.gguf"
        if not create_modified_gguf(args.model, str(modified), path_str):
            continue

        proc = start_server(args.llama_server, str(modified), args.port, args.server_args)
        try:
            if not wait_for_server(args.port):
                print(f"  ERROR: server failed for {name}", file=sys.stderr)
                dump_server_log(proc)
                continue
            eval_result = run_evaluation(args.port, args.probe_dir)
            entry = {
                "combo": name,
                "swaps": swaps,
                **eval_result,
                "timestamp": datetime.now().isoformat(),
            }
            append_jsonl(str(results_path), entry)
            print(f"  {name}: math={entry['math_score']:.3f} "
                  f"reas={entry['reasoning_score']:.2%} "
                  f"code={entry['code_score']:.2%} "
                  f"swe={entry['swe_score']:.2%} "
                  f"gen={entry.get('general_score',0):.2%}")
        finally:
            stop_server(proc)
            if modified.exists():
                modified.unlink()


if __name__ == "__main__":
    main()
