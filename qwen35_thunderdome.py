#!/usr/bin/env python3
"""Run Thunderdome 8-task suite for baseline vs circuit Qwen3.5 via llama-server.

Assumes llama-server will be spawned per-orchestrator (user restarts
between phases). Just drives `thunderdome run --task X --orchestrator Y`
and aggregates results into JSONL.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


TASKS = [
    "bench-phantom-invoice",
    "bench-ecommerce-backend",
    "bench-analytics-dashboard",
    "bench-collab-server",
    "bench-plugin-marketplace",
    "bench-time-tracker",
    "bench-task-queue",
    "bench-financial-ledger",
]


def run_task(td_dir: str, orch: str, task: str, timeout: int = 3600):
    cmd = ["./thunderdome", "run", "--orchestrator", orch, "--task", task, "--trials", "1"]
    print(f"  [{orch}] {task}...", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=td_dir, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    return proc.returncode, proc.stdout, proc.stderr, dt


def collect_latest_score(td_dir: str, orch: str, task: str):
    runs = Path(td_dir) / "results" / "runs"
    if not runs.exists():
        return None
    # Latest run dir
    candidates = sorted(runs.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for run in candidates[:5]:
        # Each run dir has trial dirs; look for this task + orchestrator
        for trial in run.glob("**/result.json"):
            try:
                data = json.loads(trial.read_text())
                if data.get("orchestrator") == orch and data.get("task") == task:
                    return data.get("score", 0.0)
            except Exception:
                continue
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thunderdome-dir", default="/mnt/ai/projects/thunderdome")
    p.add_argument("--orchestrator", required=True,
                   help="e.g. aider-local-qwen35-baseline or aider-local-qwen35-circuit")
    p.add_argument("--results", required=True)
    p.add_argument("--tasks", nargs="+", default=TASKS)
    args = p.parse_args()

    results = {"orchestrator": args.orchestrator, "tasks": {}, "elapsed": {}}
    out = Path(args.results)
    for task in args.tasks:
        rc, stdout, stderr, dt = run_task(args.thunderdome_dir, args.orchestrator, task)
        score = collect_latest_score(args.thunderdome_dir, args.orchestrator, task)
        print(f"    rc={rc} score={score} time={dt:.0f}s", flush=True)
        results["tasks"][task] = score
        results["elapsed"][task] = round(dt, 1)
        # Incremental write
        out.write_text(json.dumps(results, indent=2))

    scores = [v for v in results["tasks"].values() if v is not None]
    avg = sum(scores) / len(scores) if scores else 0.0
    results["avg_score"] = avg
    out.write_text(json.dumps(results, indent=2))
    print(f"\nAverage: {avg:.3f} over {len(scores)}/{len(args.tasks)} tasks")


if __name__ == "__main__":
    main()
