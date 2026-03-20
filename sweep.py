#!/usr/bin/env python3
"""
RYS Layer Duplication Sweep

Orchestrates the search for optimal layer duplication configuration:
1. Generate modified GGUF with duplicated layers
2. Start llama-server with the modified model
3. Run math + EQ probes
4. Score and record results
5. Print live results table
6. Kill server, repeat

Usage:
    python sweep.py \
        --model /path/to/model.gguf \
        --llama-server /path/to/llama-server \
        --tmpdir /dev/shm/rys \
        --results results.jsonl

The sweep strategy:
    Pass 1: 8-layer blocks at stride 4 across the middle
    Pass 2: Refine within the hot zone with smaller blocks
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

from gguf_surgery import duplicate_layers
from math_probe import MATH_QUESTIONS, score_math_response
from eq_probe import EQ_SCENARIOS, build_eq_prompt, parse_eq_response, score_eq_response
from reasoning_probe import REASONING_QUESTIONS, score_reasoning_response
from code_probe import CODE_TASKS, score_code_response
from swe_probe import SWE_TASKS, score_swe_response


# Server config
DEFAULT_PORT = 8099
SERVER_STARTUP_TIMEOUT = 120  # seconds
REQUEST_TIMEOUT = 60  # seconds per completion


def wait_for_server(port: int, timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    """Wait for llama-server to be ready."""
    url = f"http://127.0.0.1:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "ok":
                    return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)
    return False


def start_server(llama_server_path: str, model_path: str, port: int,
                 extra_args: list[str] = None) -> subprocess.Popen:
    """Start llama-server and return the process handle."""
    cmd = [
        llama_server_path,
        "-m", model_path,
        "--port", str(port),
        "-c", "4096",           # small context for probe eval
        "-ngl", "99",           # offload all layers to GPU
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--no-warmup",
        "-np", "1",             # single slot
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  [CMD] {' '.join(cmd)}", flush=True)

    # Let server output go to a log file so we can debug without pipe deadlocks
    log_path = Path(f"/tmp/rys_server_{port}.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    proc._log_file = log_file  # keep reference so it doesn't get GC'd
    proc._log_path = log_path
    print(f"  [PID] Server started as PID {proc.pid}, log: {log_path}", flush=True)
    return proc


def stop_server(proc: subprocess.Popen):
    """Gracefully stop the server."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    # Close the log file
    if hasattr(proc, '_log_file'):
        proc._log_file.close()


def dump_server_log(proc: subprocess.Popen, tail_lines: int = 30):
    """Print the last N lines of the server log for debugging."""
    if hasattr(proc, '_log_path') and proc._log_path.exists():
        lines = proc._log_path.read_text().splitlines()
        print(f"  --- Server log (last {tail_lines} lines) ---", file=sys.stderr)
        for line in lines[-tail_lines:]:
            print(f"  | {line}", file=sys.stderr)
        print(f"  --- End server log ---", file=sys.stderr)


def query_model(prompt: str, port: int, max_tokens: int = 64) -> str | None:
    """Send a completion request to llama-server."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"

    payload = {
        "model": "test",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    try:
        r = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"]
        else:
            print(f"  [WARN] Server returned {r.status_code}", file=sys.stderr)
            return None
    except (requests.ConnectionError, requests.Timeout) as e:
        print(f"  [WARN] Request failed: {e}", file=sys.stderr)
        return None


def run_math_probe(port: int) -> float:
    """Run all math questions and return average score (0-1)."""
    scores = []
    for question, answer in MATH_QUESTIONS:
        response = query_model(question, port, max_tokens=48)
        if response is not None:
            score = score_math_response(answer, response)
            scores.append(score)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def run_eq_probe(port: int) -> float:
    """Run all EQ scenarios and return average score (0-100)."""
    scores = []
    for scenario in EQ_SCENARIOS:
        prompt = build_eq_prompt(scenario)
        response = query_model(prompt, port, max_tokens=48)
        if response is not None:
            predicted = parse_eq_response(response, len(scenario["emotions"]))
            score = score_eq_response(scenario["reference"], predicted)
            scores.append(score)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def run_reasoning_probe(port: int) -> dict:
    """Run all reasoning questions, return scores by category and overall."""
    by_category = {}
    for q in REASONING_QUESTIONS:
        cat = q["type"]
        if cat not in by_category:
            by_category[cat] = []
        response = query_model(q["prompt"], port, max_tokens=512)
        score = score_reasoning_response(q, response)
        by_category[cat].append(score)

    # Per-category averages
    cat_scores = {}
    for cat, scores in by_category.items():
        cat_scores[cat] = sum(scores) / len(scores) if scores else 0.0

    # Overall reasoning score (0-1)
    all_scores = [s for scores in by_category.values() for s in scores]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {"categories": cat_scores, "overall": overall}


def run_code_probe(port: int) -> dict:
    """Run all coding tasks and return scores by task and overall."""
    task_scores = {}
    for task in CODE_TASKS:
        response = query_model(task["prompt"], port, max_tokens=1024)
        score = score_code_response(task, response)
        task_scores[task["id"]] = score

    overall = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    return {"tasks": task_scores, "overall": overall}


def run_swe_probe(port: int) -> dict:
    """Run all SWE agentic tasks and return scores."""
    task_scores = {}
    for task in SWE_TASKS:
        response = query_model(task["prompt"], port, max_tokens=1024)
        score = score_swe_response(task, response)
        task_scores[task["id"]] = score

    overall = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    return {"tasks": task_scores, "overall": overall}


def run_evaluation(port: int) -> dict:
    """Run all probes and return results."""
    math_score = run_math_probe(port)
    eq_score = run_eq_probe(port)
    reasoning = run_reasoning_probe(port)
    code = run_code_probe(port)
    swe = run_swe_probe(port)
    return {
        "math_score": math_score,
        "eq_score": eq_score,
        "reasoning_score": reasoning["overall"],
        "reasoning_cats": reasoning["categories"],
        "code_score": code["overall"],
        "code_tasks": code["tasks"],
        "swe_score": swe["overall"],
        "swe_tasks": swe["tasks"],
    }


def print_results_table(results: list[dict], baseline: dict | None = None):
    """Print a live-updating results table."""
    print("\n" + "=" * 120)
    print(f"{'Config':>12} {'Layers':>8} {'Math':>8} {'EQ':>8} {'Reason':>8} {'Code':>8} "
          f"{'Math Δ':>8} {'EQ Δ':>8} {'Reas Δ':>8} {'Code Δ':>8} {'Combined Δ':>11}")
    print("-" * 120)

    if baseline:
        brs = baseline.get('reasoning_score', 0)
        bcs = baseline.get('code_score', 0)
        print(f"{'BASELINE':>12} {'0':>8} "
              f"{baseline['math_score']:>8.4f} {baseline['eq_score']:>8.2f} {brs:>8.2%} {bcs:>8.2%} "
              f"{'---':>8} {'---':>8} {'---':>8} {'---':>8} {'---':>11}")
        print("-" * 120)

    for r in results:
        config = f"({r['dup_start']},{r['dup_end']})"
        n_dup = r['dup_end'] - r['dup_start']
        rs = r.get('reasoning_score', 0)
        cs = r.get('code_score', 0)

        if baseline:
            math_delta = r['math_score'] - baseline['math_score']
            eq_delta = r['eq_score'] - baseline['eq_score']
            reas_delta = rs - baseline.get('reasoning_score', 0)
            code_delta = cs - baseline.get('code_score', 0)
            combined = eq_delta + (reas_delta * 100) + (code_delta * 100)
            math_d = f"{math_delta:>+8.4f}"
            eq_d = f"{eq_delta:>+8.2f}"
            reas_d = f"{reas_delta:>+8.2%}"
            code_d = f"{code_delta:>+8.2%}"
            comb_d = f"{combined:>+11.2f}"
        else:
            math_d = eq_d = reas_d = code_d = comb_d = "---"

        print(f"{config:>12} {n_dup:>8} "
              f"{r['math_score']:>8.4f} {r['eq_score']:>8.2f} {rs:>8.2%} {cs:>8.2%} "
              f"{math_d} {eq_d} {reas_d} {code_d} {comb_d}")

    print("=" * 120)
    sys.stdout.flush()


def generate_sweep_configs(n_layers: int, block_sizes: list[int],
                           start_min: int = 4, start_max: int = None,
                           stride: int = 4) -> list[tuple[int, int]]:
    """
    Generate (dup_start, dup_end) configs for the sweep.

    Args:
        n_layers: Total layers in the model
        block_sizes: List of block sizes to try (e.g., [8])
        start_min: Earliest layer to start duplication
        start_max: Latest layer to start (default: n_layers - max(block_sizes) - 4)
        stride: Step between start positions
    """
    if start_max is None:
        start_max = n_layers - max(block_sizes) - 4

    configs = []
    for bs in block_sizes:
        for start in range(start_min, start_max + 1, stride):
            end = start + bs
            if end <= n_layers:
                configs.append((start, end))

    return configs


def main():
    parser = argparse.ArgumentParser(description="RYS Layer Duplication Sweep")
    parser.add_argument("--model", required=True, help="Path to input GGUF model")
    parser.add_argument("--llama-server", required=True, help="Path to llama-server binary")
    parser.add_argument("--tmpdir", default="/dev/shm/rys",
                        help="Temp directory for modified GGUFs (use tmpfs/RAM)")
    parser.add_argument("--results", default="rys_results.jsonl",
                        help="Output results file (JSONL)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[8],
                        help="Block sizes to sweep (default: 8)")
    parser.add_argument("--stride", type=int, default=4,
                        help="Stride between start positions (default: 4)")
    parser.add_argument("--start-min", type=int, default=4,
                        help="Earliest layer to start duplication")
    parser.add_argument("--start-max", type=int, default=None,
                        help="Latest layer to start duplication")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline run (use if already in results)")
    parser.add_argument("--server-args", nargs=argparse.REMAINDER, default=[],
                        help="Extra args to pass to llama-server (must be last)")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    results_path = Path(args.results)
    results = []
    baseline = None

    # Load existing results if resuming
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    if entry.get("is_baseline"):
                        baseline = entry
                    else:
                        results.append(entry)
        print(f"Loaded {len(results)} existing results + baseline={baseline is not None}")

    # Run baseline (unmodified model)
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
                "dup_start": -1,
                "dup_end": -1,
                "math_score": eval_result["math_score"],
                "eq_score": eval_result["eq_score"],
                "reasoning_score": eval_result["reasoning_score"],
                "reasoning_cats": eval_result.get("reasoning_cats", {}),
                "code_score": eval_result.get("code_score", 0),
                "code_tasks": eval_result.get("code_tasks", {}),
                "timestamp": datetime.now().isoformat(),
            }

            with open(results_path, "a") as f:
                f.write(json.dumps(baseline) + "\n")

            brs = baseline['reasoning_score']
            bcs = baseline.get('code_score', 0)
            print(f"  Baseline: math={baseline['math_score']:.4f} eq={baseline['eq_score']:.2f} reasoning={brs:.2%} code={bcs:.2%}")
        finally:
            stop_server(proc)

    # Get model layer count from the GGUF metadata
    from gguf import GGUFReader
    reader = GGUFReader(str(model_path), 'r')
    arch_field = reader.get_field('general.architecture')
    arch = arch_field.contents()
    block_count_field = reader.get_field(f'{arch}.block_count')
    n_layers = block_count_field.contents()
    print(f"\nModel: {model_path.name}")
    print(f"Architecture: {arch}, Layers: {n_layers}")

    # Generate sweep configurations
    configs = generate_sweep_configs(
        n_layers=n_layers,
        block_sizes=args.block_sizes,
        start_min=args.start_min,
        start_max=args.start_max,
        stride=args.stride,
    )

    # Filter out already-completed configs
    done = {(r["dup_start"], r["dup_end"]) for r in results}
    configs = [(s, e) for s, e in configs if (s, e) not in done]

    print(f"Configs to test: {len(configs)}")
    if configs:
        print(f"  Range: ({configs[0][0]},{configs[0][1]}) to ({configs[-1][0]},{configs[-1][1]})")

    print_results_table(results, baseline)

    for idx, (dup_start, dup_end) in enumerate(configs):
        n_dup = dup_end - dup_start
        config_str = f"({dup_start},{dup_end})"
        print(f"\n>>> [{idx+1}/{len(configs)}] Testing config {config_str} "
              f"(+{n_dup} layers)...")

        # Generate modified GGUF
        modified_path = tmpdir / f"rys_{dup_start}_{dup_end}.gguf"
        print(f"  Generating modified GGUF...")
        try:
            duplicate_layers(
                str(model_path), str(modified_path),
                dup_start, dup_end, verbose=False
            )
        except Exception as e:
            print(f"  ERROR generating GGUF: {e}", file=sys.stderr)
            continue

        # Start server with modified model
        print(f"  Starting server...")
        proc = start_server(
            args.llama_server, str(modified_path), args.port, args.server_args
        )

        try:
            if not wait_for_server(args.port):
                print(f"  ERROR: Server failed to start for {config_str}", file=sys.stderr)
                dump_server_log(proc)
                print(f"  Check server log above for details.", file=sys.stderr)
                continue

            print(f"  Server ready. Running probes...")
            eval_result = run_evaluation(args.port)

            entry = {
                "dup_start": dup_start,
                "dup_end": dup_end,
                "n_dup_layers": n_dup,
                "math_score": eval_result["math_score"],
                "eq_score": eval_result["eq_score"],
                "reasoning_score": eval_result["reasoning_score"],
                "reasoning_cats": eval_result.get("reasoning_cats", {}),
                "code_score": eval_result.get("code_score", 0),
                "code_tasks": eval_result.get("code_tasks", {}),
                "timestamp": datetime.now().isoformat(),
            }

            results.append(entry)

            # Append to results file
            with open(results_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            print_results_table(results, baseline)

        finally:
            stop_server(proc)

            # Clean up modified GGUF to free tmpfs space
            if modified_path.exists():
                modified_path.unlink()
                print(f"  Cleaned up {modified_path.name}")

    print("\n\nSweep complete!")
    print_results_table(results, baseline)


if __name__ == "__main__":
    main()
