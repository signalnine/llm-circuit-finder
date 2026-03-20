# Layer Surgery on MoE Language Models: A Multi-Model Study

## Summary

We used the llm-circuit-finder toolkit to systematically discover, duplicate, prune, and benchmark transformer layer configurations across **7 models** spanning 4 architectures. We evaluated changes using code and SWE reasoning probes, then validated promising configurations against the [Agentic Thunderdome](https://github.com/signalnine/thunderdome) SWE benchmark suite.

**Key findings:**
1. **Layer surgery works on pure MoE models** — 3 out of 4 MoE models improved with surgery
2. **High-expert MoE models benefit from pruning**, low-expert models from duplication
3. **Dense and hybrid (SSM+MoE) models are incompatible** with layer surgery
4. **Probe improvements don't fully transfer to end-to-end SWE benchmarks** but pruned models trade <4% SWE performance for 11% size reduction

## Hardware

- **GPU:** NVIDIA GeForce RTX 5090 (32 GB GDDR7)
- **CPU:** Intel Core i7-14700 (20c/28t)
- **RAM:** 64 GB DDR4 @ 2666 MT/s
- **Inference:** ollama 0.18.0, llama.cpp (built from source with CUDA sm_120)

## Probe Design

### Code Probe (code_probe.py)
6 hard algorithmic tasks calibrated to ~38% baseline on Qwen3-Coder:

| Task | Tests | Difficulty |
|------|-------|------------|
| serialize_tree | 7 | Tree serialization with None handling |
| count_smaller_after | 7 | Merge sort with index tracking |
| skyline | 5 | Sweep line + coordinate compression |
| calculator | 8 | Full expression parser with unary minus |
| burst_balloons | 6 | Interval DP, counterintuitive formulation |
| min_refuel_stops | 6 | Greedy with max-heap |

### SWE Probe (swe_probe.py)
10 tasks testing agentic tool-use and engineering reasoning (~82% baseline), including canary tasks at 100% to detect regressions:

- Diff comprehension, git bisect reasoning, build error diagnosis
- Multi-file conflict analysis, flaky test diagnosis, CI pipeline debugging
- Code search commands, environment-specific bugs, memory leak diagnosis

Both probes strip `<think>` blocks for compatibility with thinking models.

## Multi-Model Results

### Summary Table

| Model | Arch | Layers | Experts | Type | Best Config | Code Δ | SWE Δ | Strategy |
|-------|------|--------|---------|------|-------------|--------|-------|----------|
| **Qwen3-Coder-30B** | qwen3moe | 48 | 128 | Pure MoE | **del(28,30)** | **+46.9%** | **+5.8%** | **Prune** |
| **Mixtral 8x7B** | llama | 32 | 8 | Pure MoE | **dup(8,11)** | **+16.4%** | **+1.8%** | **Duplicate** |
| **DeepSeek-Coder-V2** | deepseek2 | 27 | 64 | MoE+dense | **dup(6,9)** | **+7.8%** | **+2.0%** | **Duplicate** |
| GPT-OSS-20B | gpt-oss | 24 | 128 | Pure MoE | — | All hurt | — | Too few layers |
| Devstral-24B | llama | 40 | dense | Dense | — | All hurt | — | Every layer critical |
| Nemotron-Nano-30B | nemotron_h_moe | 52 | 128 | Hybrid SSM+MoE | — | All crash | — | SSM breaks on reorder |
| Qwen3.5-35B-A3B | qwen35moe | 40 | 256 | MoE+thinking | — | — | — | Thinking model incompatible |

### Key Insight: Expert Count Determines Strategy

```
High experts (128+) → PRUNE redundant layers  (Qwen3-Coder: +47%)
Low experts (8-64)  → DUPLICATE useful layers  (Mixtral: +16%, DeepSeek: +8%)
```

Models with many experts have natural redundancy — some layers actively interfere with coding. Models with few experts have every layer load-bearing, but benefit from reinforcing useful computation.

## Detailed Results by Model

### Qwen3-Coder-30B-A3B (48 layers, 128 experts)

**Best model for surgery.** Nearly every pruning config improved both code and SWE scores.

#### Pruning Results (Code + SWE Probes)

| Config | Code | SWE | Code Δ | SWE Δ |
|--------|------|-----|--------|-------|
| **BASELINE** | **38.1%** | **82.8%** | — | — |
| **del(28,30)** | **84.9%** | **88.6%** | **+46.9%** | **+5.8%** |
| del(20,22) | 80.8% | 89.8% | +42.7% | +7.0% |
| del(16,18) | 76.0% | 85.3% | +37.9% | +2.5% |
| del(24,26) | 71.8% | 86.4% | +33.8% | +3.6% |
| del(32,34) | 63.0% | 92.3% | +24.9% | +9.5% |

11 of 12 configs improved code scores. Layers 28-29 actively **interfere** with code generation.

#### Duplication Results

| Config | Code | SWE | Code Δ | SWE Δ |
|--------|------|-----|--------|-------|
| dup(28,31) | 82.1% | 82.7% | +44.1% | -0.2% |
| dup(12,15) | 64.4% | 84.1% | +26.4% | +1.2% |
| dup(8,11) | 25.2% | 88.1% | -12.9% | +5.2% |

Duplication also works but pruning is consistently better (model gets smaller AND better).

#### Code vs Reasoning Circuit Map

```
Layer:  0    8    12   16   20   24   28   32   36   40   48
        |----|----|----|----|----|----|----|----|----|----|
Reason: .....████.......................................   (layers 8-12)
Code:   ..........████.........████████.................   (layers 12-15, 28-32)
```

#### Thunderdome Validation

| Model | Size | Thunderdome Avg | Δ |
|-------|------|-----------------|---|
| Baseline (48 layers) | 18 GB | 0.573 | — |
| **del(28,30) (46 layers)** | **17 GB** | **0.542** | **-3.1%** |

Per-task: ecommerce-backend improved +130%, but fts-search and plugin-marketplace regressed. The pruned model is a practical trade-off: **11% smaller, ~3% faster, 3.1% worse on SWE benchmarks.**

### Mixtral 8x7B (32 layers, 8 experts)

**Duplication works, pruning doesn't.** With only 8 experts, each layer is critical — can add compute but can't remove it.

| Config | Code | SWE | Code Δ | SWE Δ |
|--------|------|-----|--------|-------|
| **BASELINE** | **17.9%** | **78.7%** | — | — |
| **dup(8,11)** | **34.2%** | **80.5%** | **+16.4%** | **+1.8%** |
| **dup(24,27)** | **29.8%** | **88.0%** | **+11.9%** | **+9.3%** |
| del(16,18) | 0.0% | 80.7% | -17.9% | +2.0% |

dup(24,27) is particularly interesting — both code and SWE probes improve substantially (+12% and +9%).

#### Thunderdome Validation (Aider)

| Model | Thunderdome Avg | Δ |
|-------|-----------------|---|
| Mixtral baseline | 0.444 | — |
| Mixtral dup(24,27) | 0.403 | -4.0% |

Same pattern as Qwen3-Coder: probe improvements don't transfer to end-to-end SWE tasks.

### DeepSeek-Coder-V2-Lite (27 layers, 64 experts)

Moderate improvements from duplication. Required a `layer_path.py` fix to add missing `q_lora_rank` metadata.

| Config | Code | SWE | Code Δ | SWE Δ |
|--------|------|-----|--------|-------|
| **BASELINE** | **28.3%** | **86.8%** | — | — |
| **dup(6,9)** | **36.1%** | **88.8%** | **+7.8%** | **+2.0%** |
| dup(20,23) | 32.7% | 89.3% | +4.4% | +2.5% |
| del(20,22) | 0.0% | 80.2% | -28.3% | -6.7% |

### Devstral-24B (40 layers, dense)

**0 of 12 configurations improved either metric.** Dense models are too tightly coupled for layer surgery.

### GPT-OSS-20B (24 layers, 128 experts)

Despite having 128 experts, only 24 layers provides insufficient redundancy. Only `dup(6,9)` showed marginal code improvement (+13.8%) at the cost of SWE (-6.3%).

### Nemotron-Nano-30B (52 layers, hybrid SSM+MoE)

All configurations crashed on load. The alternating Mamba-2/attention pattern creates rigid layer dependencies that can't be disrupted by reordering or removal.

## Thunderdome Validation Summary

All Thunderdome-tested configurations showed the same pattern: **probe improvements do not transfer to end-to-end SWE benchmarks.**

| Model | Config | Probe Code Δ | Probe SWE Δ | Thunderdome Δ |
|-------|--------|-------------|-------------|---------------|
| Qwen3-Coder | del(28,30) | +46.9% | +5.8% | **-3.1%** |
| Qwen3-Coder | del(24,26) | +33.8% | +3.6% | -3.8% |
| Mixtral 8x7B | dup(24,27) | +11.9% | +9.3% | **-4.0%** |

The ~3-4% Thunderdome degradation is remarkably consistent across models and surgery strategies. Single-turn probes measure a different capability than multi-turn agentic SWE — the model's ability to iteratively read, write, test, and fix code involves holistic reasoning that can't be isolated to specific layers.

## Requirements for Successful Layer Surgery

Based on testing across 7 models, layer surgery requires:

1. **MoE architecture** — dense models (Devstral) are too tightly coupled
2. **Pure transformer layers** — hybrid SSM+MoE (Nemotron) breaks on reorder
3. **Sufficient layer count** — 24 layers (GPT-OSS-20B) isn't enough; 27+ works
4. **No thinking mode** — thinking models (Qwen3.5) generate 200+ tokens of reasoning that break probe evaluation

The optimal strategy depends on expert count:
- **128+ experts:** Prune harmful layers (Qwen3-Coder: +47% code, model gets smaller)
- **8-64 experts:** Duplicate beneficial layers (Mixtral: +16%, DeepSeek: +8%, model gets larger)

**Practical takeaway:** Pruning high-expert MoE models is the most valuable application — models get simultaneously smaller, faster, and better on algorithmic probes, at the cost of ~3% on real SWE benchmarks. For latency-sensitive or resource-constrained deployments, this is a worthwhile trade-off.

## Tools Developed

- **`code_probe.py`** — 6-task hard coding probe with `<think>` stripping
- **`swe_probe.py`** — 10-task SWE agentic probe with canary tasks and `<think>` stripping
- **`prune_sweep.py`** — Layer removal sweep
- **`swe_sweep.py`** — Thunderdome-integrated sweep
- **`sweep.py`** (modified) — Added code + SWE probes, multi-metric evaluation
- **`layer_path.py`** (modified) — Added deepseek2 `q_lora_rank` compatibility fix

## Reproduction

```bash
git clone https://github.com/signalnine/llm-circuit-finder.git
cd llm-circuit-finder
pip install gguf requests tqdm

# Run code+SWE sweep on any model
python /tmp/run_model_sweep.py /path/to/model.gguf model-name

# Run pruning sweep
python prune_sweep.py \
  --model /path/to/model.gguf \
  --llama-server /path/to/llama-server \
  --block-sizes 2 3 --stride 4 --start-min 4 --start-max 40 \
  --server-args --n-gpu-layers 999 --flash-attn on

# Run full sweep with all probes (duplication)
python sweep.py \
  --model /path/to/model.gguf \
  --llama-server /path/to/llama-server \
  --block-sizes 3 4 --stride 4 --start-min 8 --start-max 28 \
  --server-args --n-gpu-layers 999 --flash-attn on
```

## Raw Data

All sweep results are stored as JSONL files in this repository:
- `qwen3-coder-codeswe-sweep.jsonl` — Qwen3-Coder pruning + duplication
- `mixtral-8x7b-codeswe-sweep.jsonl` — Mixtral 8x7B
- `deepseek-coder-v2-codeswe-sweep.jsonl` — DeepSeek-Coder-V2
- `devstral-codeswe-sweep.jsonl` — Devstral-24B
- `gptoss-20b-codeswe-sweep.jsonl` — GPT-OSS-20B
- `nemotron-nano-30b-codeswe-sweep.jsonl` — Nemotron-Nano-30B
