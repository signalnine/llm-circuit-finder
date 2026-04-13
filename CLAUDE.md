# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

llm-circuit-finder discovers and exploits "reasoning circuits" in transformer models through layer duplication and pruning. Based on David Ng's RYS (Repeat Your Steps) method: certain contiguous blocks of transformer layers form functional units that, when duplicated in the forward pass (no training), measurably improve model performance on specific reasoning tasks.

## Key Commands

### Run a sweep to find circuits
```bash
python sweep.py \
    --model /path/to/model.gguf \
    --llama-server /path/to/llama-server \
    --tmpdir /dev/shm/rys \
    --results pass.jsonl \
    --block-sizes 3 4 5 \
    --stride 1 \
    --start-min 10 --start-max 20 \
    --port 8099 \
    --server-args --device Vulkan1,Vulkan2
```

### Apply a known circuit (layer duplication)
```bash
python layer_path.py model.gguf improved.gguf -p "0..14,12,13,14,15..39" -v
```

### Run individual probes against a running llama-server
```bash
python math_probe.py      # Arithmetic without chain-of-thought
python reasoning_probe.py # BBH-derived benchmarks
python code_probe.py      # Hard algorithmic coding tasks
python swe_probe.py       # SWE agentic reasoning tasks
python eq_probe.py        # Emotional intelligence assessment
```

### Visualize results
```bash
python visualize.py results.jsonl
```

### Run lm-evaluation-harness validation
```bash
lm_eval --model local-chat-completions \
    --model_args model=test,base_url=http://localhost:8089/v1/chat/completions \
    --tasks gsm8k_cot,ifeval,mbpp,bbh_cot_fewshot_logical_deduction_five_objects \
    --apply_chat_template --limit 50 \
    --output_path ./eval_results
```

## Architecture

The system follows a pipeline: **GGUF surgery → llama-server → probe evaluation → scoring → JSONL results**.

**Orchestration layer** (`sweep.py`, `prune_sweep.py`, `swe_sweep.py`, `multi_repeat.py`):
For each candidate config, creates a modified GGUF, launches llama-server, runs probe suites, records scores to JSONL, then tears down.

**GGUF manipulation** (`layer_path.py`, `gguf_surgery.py`):
`layer_path.py` is the primary tool — supports range notation like `0..16,13,14,15,16,17..39` for arbitrary layer routing (duplication, pruning, reordering). `gguf_surgery.py` is the lower-level predecessor. Both are architecture-aware (handle deepseek2 `q_lora_rank`, etc).

**Probe suites** (`math_probe.py`, `reasoning_probe.py`, `code_probe.py`, `eq_probe.py`, `swe_probe.py`):
Each probe sends HTTP requests to a running llama-server's OpenAI-compatible API, parses responses with deterministic scoring, and returns a score dict. Code/SWE probes strip `<think>` blocks for reasoning model compatibility. All probes target `http://localhost:8088` by default.

**Analysis** (`visualize.py`, `compare_eval.py`, `comprehensive_probe.py`):
Read JSONL sweep output or lm-eval JSON results and produce ranked tables/charts.

## Key Design Patterns

- All probes communicate via HTTP to llama-server's OpenAI-compatible `/v1/chat/completions` endpoint
- Sweep results are stored as JSONL (one JSON object per line) — each line contains the config and all probe scores
- `layer_path.py` range notation: `0..N` means layers 0 through N inclusive; comma-separated for arbitrary sequences
- Model strategy depends on architecture: high-expert MoE models benefit from pruning (`prune_sweep.py`), low-expert MoE from duplication (`sweep.py`), dense models are generally incompatible
- Probes use partial-credit scoring where applicable (e.g., math probe penalizes proportional digit error)

## Dependencies

Core: `gguf`, `requests`, `tqdm`, `numpy`. Optional: `lm-eval`, `matplotlib`, `aider-chat`.
External: `llama.cpp` (any backend: CPU/CUDA/Vulkan/Metal).

No pyproject.toml or requirements.txt — install dependencies directly via pip.
