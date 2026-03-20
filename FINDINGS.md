# Circuit Discovery & Layer Surgery on Qwen3-Coder-30B-A3B

## Summary

We used the llm-circuit-finder toolkit to systematically discover, duplicate, prune, and benchmark transformer layer configurations on Qwen3-Coder-30B-A3B (a 48-layer MoE model with 128 experts, 8 active per token). We evaluated changes across four probe dimensions (math, EQ, reasoning, code) and validated promising configurations against the [Agentic Thunderdome](https://github.com/signalnine/thunderdome) SWE benchmark suite using the Crush orchestrator.

**Key finding:** Layer surgery produces dramatic improvements on isolated probes (+35% code, +35% reasoning) but these gains do not transfer to end-to-end software engineering tasks. Probe-optimized models score within ~4% of baseline on real SWE benchmarks, suggesting that agentic coding ability is more holistic than what single-layer circuits can capture.

## Hardware

- **GPU:** NVIDIA GeForce RTX 5090 (32 GB GDDR7)
- **CPU:** Intel Core i7-14700 (20c/28t)
- **RAM:** 64 GB DDR4 @ 2666 MT/s
- **Inference:** ollama 0.18.0, llama.cpp (built from source with CUDA sm_120)

## Model Under Test

- **Qwen3-Coder-30B-A3B-Instruct** (GGUF via ollama)
- 48 transformer layers, 128 MoE experts, 8 active per token
- ~18.6 GB model size (GGUF Q4)
- Baseline inference: ~100 tok/s (ollama), ~210 tok/s (llama.cpp)

## Code Probe Design

We developed a hard coding probe (`code_probe.py`) calibrated to baseline around 38% for this model. After iterating through several difficulty levels (initial probes scored 92-97%), we settled on 6 tasks with 39 test cases:

| Task | Tests | Baseline Score | Difficulty |
|------|-------|---------------|------------|
| serialize_tree | 7 | 57% | Tree serialization with None handling |
| count_smaller_after | 7 | 43% | Merge sort with index tracking |
| skyline | 5 | 20% | Sweep line + coordinate compression |
| calculator | 8 | 25% | Full expression parser with unary minus |
| burst_balloons | 6 | 17% | Interval DP, counterintuitive formulation |
| min_refuel_stops | 6 | 67% | Greedy with max-heap |

Scoring is deterministic: extract code from model response, execute against test cases, report fraction passed.

## Experiment 1: Layer Duplication (RYS Method)

Duplicated contiguous blocks of 3-4 layers and measured impact across all probes.

### Reasoning-Optimized Circuits

| Config | Reasoning Δ | Code Δ | Math Δ |
|--------|-------------|--------|--------|
| **(8,12) +4** | **+23.5%** | +0.0% | +0.041 |
| (8,11) +3 | +17.6% | +0.0% | +0.028 |
| (18,21) +3 | +23.5% | — | +0.044 |

Reasoning circuits concentrate in **layers 8-12** (early-mid network).

### Code-Optimized Circuits

| Config | Code Δ | Reasoning Δ | Math Δ |
|--------|--------|-------------|--------|
| **(28,32) +4** | **+41.3%** | +5.9% | +0.031 |
| (12,16) +4 | +35.0% | +5.9% | +0.066 |
| (12,15) +3 | +34.2% | +0.0% | +0.033 |
| (20,23) +3 | +33.5% | +0.0% | +0.020 |

Code circuits concentrate in **layers 12-15 and 28-32** — completely independent from reasoning circuits. Notably, **(8,12) +4** (best reasoning) actually **hurts** code score (-12.9%).

### Thunderdome Validation (Layer Duplication)

Tested the best reasoning circuit (18,21) and best code circuit (28,32) against 10 SWE benchmark tasks using Crush:

| Model | Thunderdome Avg | Δ vs Baseline |
|-------|-----------------|---------------|
| Baseline (48 layers) | 0.573 | — |
| +dup(18,21) reasoning circuit | 0.531 | -0.042 |
| +dup(28,32) code circuit | 0.526 | -0.047 |

**Result:** Probe improvements did not transfer to SWE tasks. Both variants performed slightly worse than baseline.

## Experiment 2: Layer Pruning

Instead of duplicating layers, we **removed** 2-layer blocks to find harmful/redundant layers.

### Pruning Sweep Results (Block-Size-2)

| Config | Layers | Code Δ | Reasoning Δ | Combined |
|--------|--------|--------|-------------|----------|
| **del(28,30)** | **46** | **+31.9%** | **+35.3%** | **Best overall** |
| **del(24,26)** | **46** | **+33.8%** | **+17.6%** | **Best code** |
| del(4,6) | 46 | +35.9% | -5.9% | Code only |
| del(32,34) | 46 | +24.9% | +5.9% | |
| del(16,18) | 46 | +27.0% | +0.0% | |
| del(8,10) | 46 | +24.5% | +5.9% | |
| del(12,14) | 46 | +4.4% | +5.9% | Least impact |
| del(20,22) | 46 | +29.4% | -17.6% | Hurts reasoning |

Every single pruning configuration improved code scores. Most improved reasoning too. Removing layers 28-29 achieved **100% reasoning score** (up from 64.7%) — a perfect score on all reasoning probes.

### Thunderdome Validation (Layer Pruning)

| Model | Size | Thunderdome Avg | Δ |
|-------|------|-----------------|---|
| Baseline (48 layers) | 18.6 GB | 0.573 | — |
| del(24,26) (46 layers) | 16.6 GB | 0.535 | -0.038 |

**Result:** The pruned model scored 3.8% worse on SWE tasks while being 11% smaller and ~3% faster. Per-task breakdown shows wins on 3 tasks and losses on 3.

## Experiment 3: Cross-Model Comparison (Devstral)

Ran the same circuit sweep on Devstral-Small-2-24B (dense architecture, 40 layers) for comparison.

### Devstral Findings

- Reasoning circuits are **deeper** (layers 26-29) vs Qwen3-Coder (layers 8-12)
- Effects are **weaker** (+5.9% max reasoning improvement vs +23.5% on Qwen3-Coder)
- Dense models are more tightly coupled — layer surgery is more disruptive
- Code probe not run on Devstral (focused on Qwen3-Coder)

## Experiment 4: Qwen3-Coder vs Devstral on Thunderdome

Head-to-head comparison of both models on SWE tasks using Crush:

| Model | Avg Score | Wins |
|-------|-----------|------|
| Qwen3-Coder | **0.573** | **5** |
| Devstral | 0.531 | 4 |

Qwen3-Coder edges out overall. Devstral wins on more "agentic" tasks (collab-server, analytics-dashboard) while Qwen3-Coder wins on precision tasks (phantom-invoice, plugin-marketplace).

## Conclusions

1. **Transformers have task-specific circuits** in distinct layer ranges. Coding circuits (layers 12-15, 28-32) are independent from reasoning circuits (layers 8-12) in Qwen3-Coder.

2. **Layer surgery produces real, reproducible probe improvements.** Duplicating or removing the right layers can double code probe scores and achieve perfect reasoning scores.

3. **Probe improvements do not transfer to end-to-end SWE tasks.** Every circuit-modified model scored within ~4% of baseline on Thunderdome. Agentic software engineering requires holistic capabilities that can't be isolated to specific layer blocks.

4. **Layer pruning is surprisingly effective.** Removing 2 layers from a 48-layer MoE model improves probe scores across the board while making the model 11% smaller and faster. This suggests significant redundancy in the middle-to-late layers.

5. **MoE models are more amenable to layer surgery than dense models.** Qwen3-Coder showed much larger effects from layer manipulation than Devstral, likely because MoE routing provides natural modularity.

6. **Hard, discriminating probes are essential.** Our initial code probes scored 92-97% baseline and couldn't differentiate configurations. Only after calibrating to ~38% baseline did we see meaningful signal.

## Tools Developed

- **`code_probe.py`** — 6-task hard coding probe calibrated for strong code models
- **`prune_sweep.py`** — Layer removal sweep with multi-metric evaluation
- **`swe_sweep.py`** — Thunderdome-integrated sweep using real SWE tasks as fitness function
- **`sweep.py`** (modified) — Added code probe to existing math/EQ/reasoning probes

## Reproduction

```bash
# Clone the fork
git clone https://github.com/signalnine/llm-circuit-finder.git
cd llm-circuit-finder

# Install deps
pip install gguf requests tqdm

# Run duplication sweep with all probes
python sweep.py \
  --model /path/to/model.gguf \
  --llama-server /path/to/llama-server \
  --block-sizes 3 4 --stride 4 --start-min 8 --start-max 28 \
  --server-args --n-gpu-layers 999 --flash-attn on

# Run pruning sweep
python prune_sweep.py \
  --model /path/to/model.gguf \
  --llama-server /path/to/llama-server \
  --block-sizes 2 3 --stride 4 --start-min 4 --start-max 40 \
  --server-args --n-gpu-layers 999 --flash-attn on
```
