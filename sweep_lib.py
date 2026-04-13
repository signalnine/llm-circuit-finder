#!/usr/bin/env python3
"""
Shared infrastructure for layer surgery sweeps.

Provides server management, probe execution, GGUF helpers, and JSONL I/O
used by sweep.py, prune_sweep.py, and experiment scripts.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import requests

from math_probe import MATH_QUESTIONS, score_math_response
from eq_probe import EQ_SCENARIOS, build_eq_prompt, parse_eq_response, score_eq_response
from reasoning_probe import REASONING_QUESTIONS, score_reasoning_response
from code_probe import CODE_TASKS, score_code_response
from swe_probe import SWE_TASKS, score_swe_response
from general_probe import load_probe_files, score_general_response


# Defaults
DEFAULT_PORT = 8099
SERVER_STARTUP_TIMEOUT = 120  # seconds
REQUEST_TIMEOUT = 60  # seconds per completion


# ---------------------------------------------------------------------------
# GGUF helpers
# ---------------------------------------------------------------------------

def get_layer_count(gguf_path: str) -> tuple[int, str]:
    """Read layer count and architecture from GGUF metadata."""
    from gguf import GGUFReader
    reader = GGUFReader(gguf_path, 'r')
    arch = reader.get_field('general.architecture').contents()
    n_layers = reader.get_field(f'{arch}.block_count').contents()
    return n_layers, arch


def get_model_info(gguf_path: str) -> dict:
    """Read architecture, block_count, full_attention_interval (if hybrid)."""
    from layer_path import get_model_info as _gmi
    return _gmi(gguf_path)


def is_attention_layer(layer: int, full_attention_interval: int | None) -> bool:
    """Return True if the layer is a full-attention step in a hybrid model.

    Hybrid models (e.g. Qwen3.5) interleave attention layers every
    full_attention_interval steps. Pure transformers have no interval
    and this always returns False.
    """
    if not full_attention_interval:
        return False
    return (layer + 1) % full_attention_interval == 0


def create_modified_gguf(source: str, output: str, layer_path_str: str,
                         timeout: int = 180) -> bool:
    """Create a modified GGUF using layer_path.py with the given path string."""
    script = str(Path(__file__).parent / "layer_path.py")
    cmd = f'python3 {script} "{source}" "{output}" -p "{layer_path_str}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                            timeout=timeout)
    if result.returncode != 0:
        print(f"  ERROR: layer_path.py failed", file=sys.stderr)
        if result.stderr:
            print(result.stderr[-500:], file=sys.stderr)
        return False
    return True


def build_layer_path(mode: str, start: int, end: int, n_layers: int) -> str:
    """Build a layer_path.py path string for prune or dup mode.

    Args:
        mode: "prune" to remove layers, "dup" to duplicate them
        start: first layer of the block (inclusive)
        end: last layer of the block (exclusive for prune, inclusive end for dup)
        n_layers: total layer count in the original model
    """
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


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

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
        "-c", "4096",
        "-ngl", "99",
        "--flash-attn", "on",
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--no-warmup",
        "-np", "1",
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  [CMD] {' '.join(cmd)}", flush=True)

    log_path = Path(f"/tmp/rys_server_{port}.log")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    proc._log_file = log_file
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
        "messages": [{"role": "user", "content": prompt}],
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


# ---------------------------------------------------------------------------
# Probe runners
# ---------------------------------------------------------------------------

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

    cat_scores = {}
    for cat, scores in by_category.items():
        cat_scores[cat] = sum(scores) / len(scores) if scores else 0.0

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


def run_general_probe(port: int, probe_dir: str) -> dict:
    """Run every category in probe_dir; return per-category + overall."""
    cats = load_probe_files(probe_dir)
    if not cats:
        return {"categories": {}, "overall": 0.0}
    cat_scores = {}
    all_scores = []
    for cat in cats:
        scores = []
        for q in cat["questions"]:
            resp = query_model(q["prompt"], port, max_tokens=q["max_tokens"])
            s = score_general_response(q["answer"], resp or "")
            scores.append(s)
            all_scores.append(s)
        cat_scores[cat["name"]] = sum(scores) / len(scores) if scores else 0.0
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {"categories": cat_scores, "overall": overall}


def run_evaluation(port: int, probe_dir: str | None = None) -> dict:
    """Run all probes and return results.

    If probe_dir is provided and contains general_*.json files, also run
    the general probe suite and include its scores.
    """
    math_score = run_math_probe(port)
    eq_score = run_eq_probe(port)
    reasoning = run_reasoning_probe(port)
    code = run_code_probe(port)
    swe = run_swe_probe(port)
    result = {
        "math_score": math_score,
        "eq_score": eq_score,
        "reasoning_score": reasoning["overall"],
        "reasoning_cats": reasoning["categories"],
        "code_score": code["overall"],
        "code_tasks": code["tasks"],
        "swe_score": swe["overall"],
        "swe_tasks": swe["tasks"],
    }
    if probe_dir:
        general = run_general_probe(port, probe_dir)
        if general["categories"]:
            result["general_score"] = general["overall"]
            result["general_cats"] = general["categories"]
    return result


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> tuple[dict | None, list[dict]]:
    """Load results from a JSONL file. Returns (baseline, results)."""
    baseline = None
    results = []
    p = Path(path)
    if not p.exists():
        return baseline, results
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("is_baseline"):
                baseline = entry
            else:
                results.append(entry)
    return baseline, results


def append_jsonl(path: str, entry: dict):
    """Append a single result entry to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Per-question answer database (optional inspection aid)
# ---------------------------------------------------------------------------

def load_answers_db(path) -> dict:
    """Load an answers DB keyed by (probe, question). Returns {} if missing."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    return {(e["probe"], e["question"]): e for e in data}


def save_answers_db(path, db: dict):
    """Write the answers DB as a stable-sorted JSON array."""
    data = sorted(db.values(), key=lambda e: (e["probe"], e["question"]))
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def merge_answers(db: dict, config_label: str, answers: list[dict]):
    """Merge per-question answers from a single config into the DB."""
    for a in answers:
        key = (a["probe"], a["question"])
        if key not in db:
            db[key] = {
                "probe": a["probe"],
                "question": a["question"],
                "expected": a.get("expected", ""),
                "answers": [],
            }
        entry = db[key]
        entry["answers"] = [x for x in entry["answers"] if x.get("config") != config_label]
        entry["answers"].append({
            "config": config_label,
            "response": a.get("response"),
            "parsed": a.get("parsed"),
            "score": a.get("score"),
        })
