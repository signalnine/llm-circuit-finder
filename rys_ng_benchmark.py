#!/usr/bin/env python3
"""
Benchmark David Ng's RYS-Qwen3.5-27B models on Thunderdome.

Tests whether Ng's math+EQ probe gains transfer to multi-turn agentic SWE tasks.
Downloads pre-built RYS models (S/M/L/XL) from HuggingFace, converts to GGUF,
and runs through Thunderdome via the swe_sweep.py pattern.

Prerequisites:
    - ollama running locally
    - Thunderdome repo cloned with aider orchestrator configured
    - huggingface-cli or wget for model download
    - llama.cpp convert_hf_to_gguf.py and llama-quantize for conversion

Usage:
    python rys_ng_benchmark.py \
        --thunderdome-dir /path/to/thunderdome \
        --gguf-dir /path/to/store/ggufs \
        --llama-cpp-dir /path/to/llama.cpp \
        --tasks "greenfield/simple" "bugfix/medium" "features/medium"
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# RYS model variants — layer configs from Ng's RYS-II blog post
RYS_VARIANTS = {
    "S":  {"layers_dup": (33, 34), "overhead": "1.6%",  "desc": "Single layer duplication"},
    "M":  {"layers_dup": (31, 34), "overhead": "4.7%",  "desc": "Three layer duplication"},
    "L":  {"layers_dup": (30, 35), "overhead": "7.8%",  "desc": "Five layer duplication"},
    "XL": {"layers_dup": (26, 34), "overhead": "12.5%", "desc": "Eight layer duplication"},
}

HF_BASE = "dnhkng/RYS-Qwen3.5-27B"


def run_cmd(cmd, timeout=1800):
    """Run a shell command, return (stdout, returncode)."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout, errors="replace"
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        print(f"  WARNING: timed out after {timeout}s", flush=True)
        return "", 1


def download_and_convert(variant, gguf_dir, llama_cpp_dir, quantize="Q4_K_M"):
    """Download a RYS variant from HuggingFace, convert to GGUF, quantize."""
    hf_repo = f"{HF_BASE}-{variant}"
    hf_dir = Path(gguf_dir) / f"rys-{variant}-hf"
    gguf_f16 = Path(gguf_dir) / f"rys-{variant}-f16.gguf"
    gguf_quant = Path(gguf_dir) / f"rys-{variant}-{quantize}.gguf"

    if gguf_quant.exists():
        print(f"  {variant}: Already converted at {gguf_quant}", flush=True)
        return str(gguf_quant)

    # Download from HuggingFace
    print(f"  {variant}: Downloading from {hf_repo}...", flush=True)
    _, rc = run_cmd(
        f"huggingface-cli download {hf_repo} --local-dir {hf_dir}",
        timeout=3600,
    )
    if rc != 0:
        print(f"  ERROR: Failed to download {hf_repo}", flush=True)
        return None

    # Convert to GGUF
    if not gguf_f16.exists():
        print(f"  {variant}: Converting to GGUF...", flush=True)
        convert_script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
        _, rc = run_cmd(
            f"python3 {convert_script} {hf_dir} --outfile {gguf_f16}",
            timeout=1800,
        )
        if rc != 0:
            print(f"  ERROR: Conversion failed", flush=True)
            return None

    # Quantize
    print(f"  {variant}: Quantizing to {quantize}...", flush=True)
    quantize_bin = Path(llama_cpp_dir) / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = Path(llama_cpp_dir) / "llama-quantize"
    _, rc = run_cmd(f"{quantize_bin} {gguf_f16} {gguf_quant} {quantize}", timeout=1800)
    if rc != 0:
        print(f"  ERROR: Quantization failed", flush=True)
        return None

    # Cleanup f16
    if gguf_f16.exists():
        gguf_f16.unlink()

    return str(gguf_quant)


def import_to_ollama(gguf_path, model_name, template_model):
    """Import GGUF into ollama, copying template from an existing model."""
    stdout, rc = run_cmd(f"ollama show {template_model} --modelfile 2>/dev/null")
    if rc != 0:
        modelfile = f"FROM {gguf_path}\n"
    else:
        lines = stdout.split("\n")
        new_lines = []
        for line in lines:
            if line.startswith("FROM ") and not line.startswith("# FROM"):
                new_lines.append(f"FROM {gguf_path}")
            elif not line.startswith("#"):
                new_lines.append(line)
        modelfile = "\n".join(new_lines)

    mf_path = "/tmp/rys_ng_modelfile"
    with open(mf_path, "w") as f:
        f.write(modelfile)

    run_cmd(f"ollama rm {model_name} 2>/dev/null", timeout=30)
    _, rc = run_cmd(f"ollama create {model_name} -f {mf_path}", timeout=180)
    if rc != 0:
        return False

    # Pre-load
    run_cmd(
        f'curl -s http://localhost:11434/api/generate -d \'{{"model":"{model_name}","prompt":"hi","stream":false,"options":{{"num_predict":1}}}}\' > /dev/null',
        timeout=120,
    )
    return True


def run_thunderdome(orchestrator, task_categories, thunderdome_dir):
    """Run Thunderdome tasks and collect scores."""
    scores = {}
    for cat in task_categories:
        cmd = (
            f"cd {thunderdome_dir} && "
            f"export PATH=$PATH:/usr/local/go/bin && "
            f"./thunderdome run --orchestrator {orchestrator} --category \"{cat}\" --trials 1 2>&1"
        )
        run_cmd(cmd, timeout=1800)

    runs_dir = Path(thunderdome_dir) / "results" / "runs"
    cutoff = time.time() - 3600
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        try:
            if run_dir.stat().st_mtime < cutoff:
                break
        except OSError:
            continue
        for meta in run_dir.rglob("meta.json"):
            if orchestrator not in str(meta):
                continue
            task_name = meta.parent.parent.name
            with open(meta) as f:
                data = json.load(f)
            score = data.get("composite_score", 0)
            if task_name not in scores:
                scores[task_name] = score

    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Ng's RYS-Qwen3.5-27B models on Thunderdome"
    )
    parser.add_argument("--thunderdome-dir", required=True)
    parser.add_argument("--gguf-dir", required=True,
                        help="Directory to store downloaded/converted GGUFs")
    parser.add_argument("--llama-cpp-dir", required=True,
                        help="Path to llama.cpp source (for convert/quantize)")
    parser.add_argument("--template-model", default="qwen3:27b",
                        help="Ollama model to copy chat template from")
    parser.add_argument("--results", default="rys_ng_benchmark.jsonl")
    parser.add_argument("--tasks", nargs="+",
                        default=["greenfield/simple", "bugfix/medium", "features/medium"])
    parser.add_argument("--quantize", default="Q4_K_M")
    parser.add_argument("--variants", nargs="+", default=["S", "M", "L", "XL"],
                        help="Which RYS variants to test")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, assume GGUFs already exist in gguf-dir")
    args = parser.parse_args()

    os.makedirs(args.gguf_dir, exist_ok=True)
    results_path = Path(args.results)

    # Load existing results
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                existing[entry.get("variant", "baseline")] = entry

    baseline_orch = f"aider-local-qwen3-27b"
    rys_orch = f"aider-local-qwen3-27b-rys"

    # Baseline: standard Qwen3.5-27B
    if not args.skip_baseline and "baseline" not in existing:
        print("\n>>> BASELINE (Qwen3.5-27B)...", flush=True)
        scores = run_thunderdome(baseline_orch, args.tasks, args.thunderdome_dir)
        avg = sum(scores.values()) / max(len(scores), 1)
        entry = {
            "variant": "baseline", "is_baseline": True,
            "task_scores": scores, "avg_score": avg,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        existing["baseline"] = entry
        with open(results_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"  Baseline avg: {avg:.3f}", flush=True)

    b_avg = existing.get("baseline", {}).get("avg_score", 0)

    # Test each RYS variant
    for variant in args.variants:
        if variant in existing:
            print(f"\n>>> Skipping {variant} (already done)", flush=True)
            continue

        info = RYS_VARIANTS[variant]
        print(f"\n>>> RYS-{variant}: {info['desc']} (overhead: {info['overhead']})...",
              flush=True)

        # Get GGUF
        if args.skip_download:
            gguf_path = str(Path(args.gguf_dir) / f"rys-{variant}-{args.quantize}.gguf")
            if not Path(gguf_path).exists():
                print(f"  ERROR: {gguf_path} not found", flush=True)
                continue
        else:
            gguf_path = download_and_convert(
                variant, args.gguf_dir, args.llama_cpp_dir, args.quantize
            )
            if not gguf_path:
                continue

        # Import to ollama
        ollama_name = f"qwen3-27b-rys-{variant.lower()}"
        print(f"  Importing to ollama as {ollama_name}...", flush=True)
        if not import_to_ollama(gguf_path, ollama_name, args.template_model):
            print(f"  ERROR: Ollama import failed", flush=True)
            continue

        # Ensure orchestrator adapter exists with this model
        td = Path(args.thunderdome_dir)
        rys_adapter_dir = td / "adapters" / rys_orch
        if not rys_adapter_dir.exists():
            baseline_adapter_dir = td / "adapters" / baseline_orch
            if baseline_adapter_dir.exists():
                rys_adapter_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy(baseline_adapter_dir / "adapter.sh",
                            rys_adapter_dir / "adapter.sh")

        # Patch adapter to use current variant
        adapter_file = rys_adapter_dir / "adapter.sh"
        if adapter_file.exists():
            text = adapter_file.read_text()
            # Replace any openai/qwen* model reference with current variant
            import re
            text = re.sub(r'openai/[^\s"]+', f'openai/{ollama_name}', text)
            adapter_file.write_text(text)

        # Run Thunderdome
        print(f"  Running Thunderdome...", flush=True)
        scores = run_thunderdome(rys_orch, args.tasks, args.thunderdome_dir)
        avg = sum(scores.values()) / max(len(scores), 1)
        delta = avg - b_avg

        entry = {
            "variant": variant, "is_baseline": False,
            "layers_dup": list(info["layers_dup"]),
            "overhead": info["overhead"],
            "task_scores": scores, "avg_score": avg,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        existing[variant] = entry
        with open(results_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"  Avg: {avg:.3f} ({delta:+.3f} vs baseline)", flush=True)

        # Unload from ollama
        run_cmd(
            f'curl -s http://localhost:11434/api/generate -d \'{{"model":"{ollama_name}","keep_alive":0}}\' > /dev/null',
            timeout=30,
        )

    # Summary
    print("\n" + "=" * 70, flush=True)
    print(f"{'Variant':>10s}  {'Overhead':>10s}  {'Avg':>7s}  {'Delta':>7s}  Tasks", flush=True)
    print("-" * 70, flush=True)
    if "baseline" in existing:
        e = existing["baseline"]
        print(f"{'baseline':>10s}  {'0%':>10s}  {e['avg_score']:>7.3f}  {'---':>7s}  {e['task_scores']}",
              flush=True)
    for variant in ["S", "M", "L", "XL"]:
        if variant in existing:
            e = existing[variant]
            d = e["avg_score"] - b_avg
            print(f"{'RYS-' + variant:>10s}  {e['overhead']:>10s}  {e['avg_score']:>7.3f}  {d:>+7.3f}  {e['task_scores']}",
                  flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
