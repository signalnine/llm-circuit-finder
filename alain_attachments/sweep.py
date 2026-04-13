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

from layer_path import get_model_info, build_gguf_from_path
from math_probe import MATH_QUESTIONS, score_math_response, parse_number_from_response
from eq_probe import EQ_SCENARIOS, build_eq_prompt, parse_eq_response, score_eq_response
from reasoning_probe import REASONING_QUESTIONS, score_reasoning_response, extract_final_answer
from general_probe import load_probe_files, score_general_response, extract_answer


# Server config
DEFAULT_PORT = 8099
SERVER_STARTUP_TIMEOUT = 120  # seconds
REQUEST_TIMEOUT = 180  # seconds per completion


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


def run_math_probe(port: int) -> tuple[float, list[dict]]:
    """Run all math questions and return (average score, answer details)."""
    scores = []
    answers = []
    for question, correct in MATH_QUESTIONS:
        response = query_model(question, port, max_tokens=48)
        if response is not None:
            score = score_math_response(correct, response)
            parsed = parse_number_from_response(response)
            scores.append(score)
        else:
            score = 0.0
            parsed = None
            scores.append(0.0)
        answers.append({
            "probe": "math",
            "question": question,
            "expected": str(correct),
            "response": response,
            "parsed": str(parsed) if parsed is not None else None,
            "score": round(score, 4),
        })
    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, answers


def run_eq_probe(port: int) -> tuple[float, list[dict]]:
    """Run all EQ scenarios and return (average score, answer details)."""
    scores = []
    answers = []
    for scenario in EQ_SCENARIOS:
        prompt = build_eq_prompt(scenario)
        response = query_model(prompt, port, max_tokens=48)
        if response is not None:
            predicted = parse_eq_response(response, len(scenario["emotions"]))
            score = score_eq_response(scenario["reference"], predicted)
            scores.append(score)
        else:
            predicted = None
            score = 0.0
            scores.append(0.0)
        answers.append({
            "probe": f"eq/{scenario['id']}",
            "question": prompt,
            "expected": str(scenario["reference"]),
            "response": response,
            "parsed": str(predicted) if predicted is not None else None,
            "score": round(score / 100.0, 4),
        })
    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, answers


def run_reasoning_probe(port: int) -> tuple[dict, list[dict]]:
    """Run all reasoning questions, return (scores dict, answer details)."""
    by_category = {}
    answers = []
    for q in REASONING_QUESTIONS:
        cat = q["type"]
        if cat not in by_category:
            by_category[cat] = []
        response = query_model(q["prompt"], port, max_tokens=512)
        score = score_reasoning_response(q, response)
        by_category[cat].append(score)
        parsed = extract_final_answer(response) if response else None
        answers.append({
            "probe": f"reasoning/{cat}",
            "question": q["prompt"],
            "expected": q["answer"],
            "response": response,
            "parsed": parsed,
            "score": round(score, 4),
        })

    # Per-category averages
    cat_scores = {}
    for cat, scores in by_category.items():
        cat_scores[cat] = sum(scores) / len(scores) if scores else 0.0

    # Overall reasoning score (0-1)
    all_scores = [s for scores in by_category.values() for s in scores]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

    return {"categories": cat_scores, "overall": overall}, answers


def run_general_probe(port: int, probe_dir: str) -> tuple[dict, list[dict]]:
    """Run all general probe categories, return (scores dict, answer details)."""
    categories = load_probe_files(probe_dir)
    if not categories:
        return {"categories": {}, "overall": 0.0}, []

    cat_scores = {}
    all_scores = []
    answers = []

    for cat in categories:
        scores = []
        for q in cat["questions"]:
            response = query_model(q["prompt"], port, max_tokens=q["max_tokens"])
            score = score_general_response(q["answer"], response)
            parsed = extract_answer(response) if response else None
            scores.append(score)
            all_scores.append(score)
            answers.append({
                "probe": f"general/{cat['name']}",
                "question": q["prompt"],
                "expected": q["answer"],
                "response": response,
                "parsed": parsed,
                "score": round(score, 4),
            })

        avg = sum(scores) / len(scores) if scores else 0.0
        cat_scores[cat["name"]] = avg

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {"categories": cat_scores, "overall": overall}, answers


def run_evaluation(port: int, probe_dir: str = "probes") -> dict:
    """Run all probes and return results including answer details."""
    math_score, math_answers = run_math_probe(port)
    eq_score, eq_answers = run_eq_probe(port)
    reasoning, reasoning_answers = run_reasoning_probe(port)
    general, general_answers = run_general_probe(port, probe_dir)
    return {
        "math_score": math_score,
        "eq_score": eq_score,
        "reasoning_score": reasoning["overall"],
        "reasoning_cats": reasoning["categories"],
        "general_score": general["overall"],
        "general_cats": general["categories"],
        "answers": math_answers + eq_answers + reasoning_answers + general_answers,
    }


def load_answers_db(path: Path) -> dict:
    """Load the answers database. Returns dict keyed by (probe, question)."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    db = {}
    for entry in data:
        key = (entry["probe"], entry["question"])
        db[key] = entry
    return db


def save_answers_db(path: Path, db: dict):
    """Save the answers database."""
    data = sorted(db.values(), key=lambda e: (e["probe"], e["question"]))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def merge_answers(db: dict, config_label: str, answers: list[dict]):
    """Merge a config's answers into the database."""
    for a in answers:
        key = (a["probe"], a["question"])
        if key not in db:
            db[key] = {
                "probe": a["probe"],
                "question": a["question"],
                "expected": a["expected"],
                "answers": [],
            }
        # Avoid duplicate config entries on resume
        entry = db[key]
        existing_configs = {ans["config"] for ans in entry["answers"]}
        if config_label not in existing_configs:
            entry["answers"].append({
                "config": config_label,
                "response": a["response"],
                "parsed": a["parsed"],
                "score": a["score"],
            })


def _print_general_cats(general_cats: dict, baseline_cats: dict | None = None):
    """Print per-category sub-rows under a result row."""
    for name in sorted(general_cats):
        val = general_cats[name]
        if baseline_cats and name in baseline_cats:
            delta = val - baseline_cats[name]
            delta_str = f"{delta:>+8.2%}"
        else:
            delta_str = f"{'---':>8}"
        # Config column gets the name, General column gets the value, rest blank
        print(f"{'  ' + name:>12} {'':>8} "
              f"{'':>8} {'':>8} {'':>8} {val:>8.2%} "
              f"{'':>8} {'':>8} {'':>8} {delta_str} {'':>7}")


def print_results_table(results: list[dict], baseline: dict | None = None):
    """Print a live-updating results table."""
    print("\n" + "=" * 123)
    print(f"{'Config':>12} {'Layers':>8} {'Math':>8} {'EQ':>8} {'Reason':>8} {'General':>8} "
          f"{'Math Δ':>8} {'EQ Δ':>8} {'Reas Δ':>8} {'Gen Δ':>8} {'Time':>7}")
    print("-" * 123)

    baseline_gcats = baseline.get('general_cats', {}) if baseline else {}

    if baseline:
        brs = baseline.get('reasoning_score', 0)
        bgs = baseline.get('general_score', 0)
        bt = baseline.get('eval_time', 0)
        print(f"{'BASELINE':>12} {'0':>8} "
              f"{baseline['math_score']:>8.4f} {baseline['eq_score']:>8.2f} {brs:>8.2%} {bgs:>8.2%} "
              f"{'---':>8} {'---':>8} {'---':>8} {'---':>8} {bt:>6.1f}s")
        _print_general_cats(baseline_gcats)
        print("-" * 123)

    for r in results:
        config = f"({r['dup_start']},{r['dup_end']})"
        mode = r.get('mode', 'add')
        if mode in ("replace-next", "replace-same"):
            layers_str = f"{r['dup_start']}>{r['dup_end']}"
        elif mode == "add":
            n_affected = r['dup_end'] - r['dup_start']
            layers_str = f"+{n_affected}"
        else:
            n_affected = r['dup_end'] - r['dup_start']
            layers_str = f"-{n_affected}"
        rs = r.get('reasoning_score', 0)
        gs = r.get('general_score', 0)

        if baseline:
            math_delta = r['math_score'] - baseline['math_score']
            eq_delta = r['eq_score'] - baseline['eq_score']
            reas_delta = rs - baseline.get('reasoning_score', 0)
            gen_delta = gs - baseline.get('general_score', 0)
            math_d = f"{math_delta:>+8.4f}"
            eq_d = f"{eq_delta:>+8.2f}"
            reas_d = f"{reas_delta:>+8.2%}"
            gen_d = f"{gen_delta:>+8.2%}"
        else:
            math_d = eq_d = reas_d = gen_d = "---"

        print(f"{config:>12} {layers_str:>8} "
              f"{r['math_score']:>8.4f} {r['eq_score']:>8.2f} {rs:>8.2%} {gs:>8.2%} "
              f"{math_d} {eq_d} {reas_d} {gen_d} {r.get('eval_time', 0):>6.1f}s")
        _print_general_cats(r.get('general_cats', {}), baseline_gcats)

    print("=" * 123)
    sys.stdout.flush()


def is_attention_layer(layer_idx: int, full_attention_interval: int | None) -> bool:
    """Check if a layer is a pure-attention layer in a hybrid model.
    Returns False for pure transformer models (no interval)."""
    if full_attention_interval is None:
        return False
    return (layer_idx + 1) % full_attention_interval == 0


def generate_replace_configs(n_layers: int, full_attention_interval: int | None,
                             mode: str, start_min: int = 1,
                             start_max: int | None = None) -> list[tuple[int, int]]:
    """
    Generate (target_layer, replacement_layer) configs for replace modes.

    Skips attention layers as targets (they stay untouched).

    mode='replace-next': replace with the very next layer.
    mode='replace-same': replace with the next layer of the same type
                         (skips attention layers as replacement source).
    """
    if start_max is None:
        start_max = n_layers - 2  # need at least one layer after

    configs = []
    for target in range(start_min, start_max + 1):
        # Skip attention layers as targets
        if is_attention_layer(target, full_attention_interval):
            continue

        if mode == "replace-next":
            replacement = target + 1
            if replacement < n_layers:
                configs.append((target, replacement))
        elif mode == "replace-same":
            # Find next non-attention layer
            replacement = target + 1
            while replacement < n_layers and is_attention_layer(replacement, full_attention_interval):
                replacement += 1
            if replacement < n_layers:
                configs.append((target, replacement))

    return configs


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
    parser = argparse.ArgumentParser(description="RYS Layer Sweep")
    parser.add_argument("--model", required=True, help="Path to input GGUF model")
    parser.add_argument("--llama-server", required=True, help="Path to llama-server binary")
    parser.add_argument("--mode", choices=["add", "remove", "replace-next", "replace-same"],
                        default="add",
                        help="'add' duplicates layers, 'remove' skips them, "
                             "'replace-next' replaces a layer with the next one, "
                             "'replace-same' replaces with the next same-type layer "
                             "(default: add)")
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
                        help="Earliest layer to start (default: 4)")
    parser.add_argument("--start-max", type=int, default=None,
                        help="Latest layer to start")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline run (use if already in results)")
    parser.add_argument("--probe-dir", default="probes",
                        help="Directory with general_*.json probe files (default: probes)")
    parser.add_argument("--server-args", nargs=argparse.REMAINDER, default=[],
                        help="Extra args to pass to llama-server (must be last)")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    results_path = Path(args.results)
    answers_path = results_path.with_suffix('.answers.json')
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

    # Load existing answers database
    answers_db = load_answers_db(answers_path)

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
            eval_start = time.time()
            eval_result = run_evaluation(args.port, args.probe_dir)
            eval_elapsed = time.time() - eval_start
            baseline = {
                "is_baseline": True,
                "dup_start": -1,
                "dup_end": -1,
                "math_score": eval_result["math_score"],
                "eq_score": eval_result["eq_score"],
                "reasoning_score": eval_result["reasoning_score"],
                "reasoning_cats": eval_result.get("reasoning_cats", {}),
                "general_score": eval_result["general_score"],
                "general_cats": eval_result.get("general_cats", {}),
                "eval_time": round(eval_elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            }

            with open(results_path, "a") as f:
                f.write(json.dumps(baseline) + "\n")

            merge_answers(answers_db, "BASELINE", eval_result["answers"])
            save_answers_db(answers_path, answers_db)

            brs = baseline['reasoning_score']
            bgs = baseline['general_score']
            print(f"  Baseline: math={baseline['math_score']:.4f} eq={baseline['eq_score']:.2f} reasoning={brs:.2%} general={bgs:.2%} ({eval_elapsed:.1f}s)")
            gcats = baseline.get('general_cats', {})
            if gcats:
                parts = " ".join(f"{k}={v:.2%}" for k, v in sorted(gcats.items()))
                print(f"  General:  {parts}")
        finally:
            stop_server(proc)

    # Get model layer count from the GGUF metadata
    model_info = get_model_info(str(model_path))
    arch = model_info["architecture"]
    n_layers = model_info["block_count"]
    fai = model_info["full_attention_interval"]
    print(f"\nModel: {model_path.name}")
    print(f"Architecture: {arch}, Layers: {n_layers}")
    if fai:
        print(f"Hybrid model: full_attention_interval={fai}")

    # Generate sweep configurations
    if args.mode in ("replace-next", "replace-same"):
        configs = generate_replace_configs(
            n_layers=n_layers,
            full_attention_interval=fai,
            mode=args.mode,
            start_min=args.start_min,
            start_max=args.start_max,
        )
    else:
        configs = generate_sweep_configs(
            n_layers=n_layers,
            block_sizes=args.block_sizes,
            start_min=args.start_min,
            start_max=args.start_max,
            stride=args.stride,
        )

    # Filter out already-completed configs (mode-aware)
    done = {(r.get("mode", "add"), r["dup_start"], r["dup_end"]) for r in results}
    configs = [(s, e) for s, e in configs if (args.mode, s, e) not in done]

    print(f"Mode: {args.mode}")
    print(f"Configs to test: {len(configs)}")
    if configs:
        print(f"  Range: ({configs[0][0]},{configs[0][1]}) to ({configs[-1][0]},{configs[-1][1]})")

    print_results_table(results, baseline)

    for idx, (dup_start, dup_end) in enumerate(configs):
        config_str = f"({dup_start},{dup_end})"

        if args.mode == "add":
            n_affected = dup_end - dup_start
            mode_label = f"+{n_affected}"
            desc = f"+{n_affected} layers"
        elif args.mode == "remove":
            n_affected = dup_end - dup_start
            mode_label = f"-{n_affected}"
            desc = f"-{n_affected} layers"
        else:
            # replace modes: dup_start=target, dup_end=replacement
            mode_label = f"{dup_start}>{dup_end}"
            desc = f"layer {dup_start} -> {dup_end}"

        print(f"\n>>> [{idx+1}/{len(configs)}] Testing config {config_str} "
              f"({desc})...")

        # Generate modified GGUF
        modified_path = tmpdir / f"rys_{args.mode}_{dup_start}_{dup_end}.gguf"
        print(f"  Generating modified GGUF ({args.mode} mode)...")
        try:
            if args.mode == "add":
                layer_path = (list(range(dup_end))
                              + list(range(dup_start, dup_end))
                              + list(range(dup_end, n_layers)))
            elif args.mode == "remove":
                layer_path = (list(range(dup_start))
                              + list(range(dup_end, n_layers)))
            else:
                # replace modes: swap dup_start with dup_end's weights
                layer_path = list(range(n_layers))
                layer_path[dup_start] = dup_end
            build_gguf_from_path(
                str(model_path), str(modified_path),
                layer_path, verbose=False
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
            eval_start = time.time()
            eval_result = run_evaluation(args.port, args.probe_dir)
            eval_elapsed = time.time() - eval_start

            entry = {
                "mode": args.mode,
                "dup_start": dup_start,
                "dup_end": dup_end,
                "math_score": eval_result["math_score"],
                "eq_score": eval_result["eq_score"],
                "reasoning_score": eval_result["reasoning_score"],
                "reasoning_cats": eval_result.get("reasoning_cats", {}),
                "general_score": eval_result["general_score"],
                "general_cats": eval_result.get("general_cats", {}),
                "eval_time": round(eval_elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            }

            results.append(entry)

            # Append to results file
            with open(results_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            config_label = f"{args.mode} ({dup_start},{dup_end})"
            merge_answers(answers_db, config_label, eval_result["answers"])
            save_answers_db(answers_path, answers_db)

            gcats = entry.get('general_cats', {})
            if gcats:
                parts = " ".join(f"{k}={v:.2%}" for k, v in sorted(gcats.items()))
                print(f"  General:  {parts}")

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
