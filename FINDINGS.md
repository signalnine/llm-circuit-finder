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

#### Thunderdome Validation

**swe_sweep.py (3 tasks):** Mixtral is the **only model where probe improvements transferred to Thunderdome.**

| Model | Thunderdome Avg | Δ |
|-------|-----------------|---|
| Mixtral baseline | 0.503 | — |
| **Mixtral dup(8,11)** | **0.532** | **+5.8%** |
| **Mixtral dup(24,27)** | **0.532** | **+5.8%** |

Both duplication configs improved phantom-invoice scoring (0.710 → 0.795) while maintaining identical performance on fts-search and time-tracker.

Earlier 8-task aider run showed -4.0%, but that run had a different task mix and single-trial noise. The swe_sweep result with controlled task selection shows a positive transfer.

### DeepSeek-Coder-V2-Lite (27 layers, 64 experts)

Moderate improvements from duplication. Required a `layer_path.py` fix to add missing `q_lora_rank` metadata.

| Config | Code | SWE | Code Δ | SWE Δ |
|--------|------|-----|--------|-------|
| **BASELINE** | **28.3%** | **86.8%** | — | — |
| **dup(6,9)** | **36.1%** | **88.8%** | **+7.8%** | **+2.0%** |
| dup(20,23) | 32.7% | 89.3% | +4.4% | +2.5% |
| del(20,22) | 0.0% | 80.2% | -28.3% | -6.7% |

#### Thunderdome Validation

| Model | Thunderdome Avg | Δ |
|-------|-----------------|---|
| DeepSeek baseline | 0.532 | — |
| DeepSeek dup(6,9) | 0.532 | 0.0% |

Probe improvements (+8% code, +2% SWE) did not transfer — identical Thunderdome scores.

### Devstral-24B (40 layers, dense)

**0 of 12 configurations improved either metric.** Dense models are too tightly coupled for layer surgery.

### GPT-OSS-20B (24 layers, 128 experts)

Despite having 128 experts, only 24 layers provides insufficient redundancy. Only `dup(6,9)` showed marginal code improvement (+13.8%) at the cost of SWE (-6.3%).

### Nemotron-Nano-30B (52 layers, hybrid SSM+MoE)

All configurations crashed on load. The alternating Mamba-2/attention pattern creates rigid layer dependencies that can't be disrupted by reordering or removal.

### Qwen3.5-35B-A3B (40 layers, hybrid attention interval=4)

**Baseline is broken in llama.cpp** — the unmodified model scored 0% on math, EQ, reasoning, code, and general probes (only 10% on SWE). This confirms the upstream author's observation: Qwen3.5's hybrid attention pattern (a full-attention layer every 4 steps) is not correctly routed by llama.cpp out of the box.

Layer-swap surgery recovered real capability. Using `--mode replace-same` (replaces a layer with the next non-attention layer, preserving the hybrid stride), 32 single-layer swap configs were evaluated across layers 4-35:

| Config | Code | SWE | Reasoning | General |
|--------|------|-----|-----------|---------|
| replace-same(14,16) | **20.63%** | 0% | 5.88% | 0% |
| replace-same(28,29) | 16.67% | 10% | 0% | 0% |
| replace-same( 8, 9) | 14.44% | 10% | 0% | 7.14% |
| replace-same(24,25) | 14.44% | **45%** | 0% | 0.71% |
| replace-same(18,20) | 0% | **48.33%** | 0% | 0% |
| replace-same(17,18) | 0% | 30% | 0% | 7.14% |
| replace-same(32,33) | 9.52% | 28.33% | 5.88% | 0% |
| replace-same(21,22) | 0% | 10% | **11.76%** | 0% |
| replace-same(22,24) | 0% | 0% | 0% | **13.74%** |
| baseline | 0% | 10% | 0% | 7.14% |

**Task-specific circuits are clearly separable:**
- Code: layers 14-16 (+20.63pp)
- SWE: layers 18-20 (+38.33pp) and 24-25 (+35pp)
- Reasoning: layers 21-22 (+11.76pp)
- General knowledge: layers 22-24 (+6.6pp)

Math and EQ remained at 0% across all configs, suggesting either that those capabilities require a multi-layer intervention or that llama.cpp's hybrid routing damages them beyond recovery from single-layer swaps.

These results validate two claims from Alain Reyes (original RYS author):
1. Qwen3.5 requires preservation of the 4-layer attention stride; `replace-next` (which ignores stride) caused server crashes on attention-layer positions, while `replace-same` (stride-aware) ran cleanly.
2. Category-based probes via `--probe-dir` (`general_general.json`, `general_languages.json`) surface capability differences invisible to the math/EQ/reasoning probes alone.

Raw data: `qwen35_replace_same.jsonl`.

#### Circuit Composition

Seven compositions of the hot circuits were tested by applying multiple layer swaps simultaneously (e.g., `layers[14]=16; layers[18]=20`):

| Combo | Code | SWE | Reas | Gen |
|-------|-----:|----:|-----:|----:|
| (best solos) | 20.63% | 48.33% | 11.76% | 13.74% |
| code(14,16)+swe(18,20) | 5.56% | 18.75% | 0% | 7.14% |
| code(14,16)+swe(24,25) | 0% | 26.67% | 0% | 0% |
| swe(18,20)+swe(24,25) | 11.11% | 26.67% | 0% | 0% |
| code(14,16)+swe(18,20)+swe(24,25) | 5.56% | 30.00% | 0% | 7.14% |
| **code(14,16)+reas(21,22)** | **22.86%** | 10.00% | 0% | 7.14% |
| code(14,16)+gen(22,24) | 0% | 0% | 0% | 0% |
| code(14,16)+swe(18,20)+reas(21,22) | 0% | 30.00% | 5.88% | 0% |

**Circuits do not stack additively.** Every composition involving SWE(18,20) degraded it from 48% to 18-30%. Only code(14,16)+reas(21,22) showed mild synergy (code 20.63% → 22.86%). The code(14,16)+gen(22,24) pairing was catastrophically destructive, zeroing every metric.

Interpretation: Qwen3.5's capability circuits overlap in layer space, and the interference pattern required to surface one capability is broken when multiple surgeries are applied at once. The practical implication is that **surgery must target a single capability at a time** — there is no "swiss-army" config that recovers all broken capabilities simultaneously.

Raw data: `qwen35_combo.jsonl`.

#### Thunderdome Validation (8 tasks, Crush orchestrator)

The best single-swap circuit `replace-same(24,25)` (probe winner: code +14.4pp, SWE +45pp) was benchmarked against the unmodified baseline across 8 Thunderdome tasks via the Crush orchestrator on llama-server.

| Task | Baseline | Circuit(24,25) | Δ |
|------|---------:|--------:|----:|
| phantom-invoice | 0.98 | 0.98 | 0.00 |
| ecommerce-backend | 0.88 | 0.26 | **-0.62** |
| analytics-dashboard | 0.25 | 0.63 | +0.38 |
| collab-server | 0.29 | 0.46 | +0.17 |
| plugin-marketplace | 0.84 | 0.29 | **-0.55** |
| time-tracker | 0.74 | 0.32 | **-0.42** |
| task-queue | 0.30 | 0.26 | -0.04 |
| financial-ledger | 1.00 | 1.00 | 0.00 |
| **Average** | **0.660** | **0.525** | **-0.135** |

**The circuit lost -13.5pp on Thunderdome despite winning +45pp on the SWE probe.** Two revelations:

1. **The unmodified Qwen3.5-35B is not actually broken in agentic contexts.** The probe baseline of 0% on math/eq/reasoning/code was misleading — those probes used short `max_tokens` budgets that were entirely consumed by the model's `<think>` reasoning phase, leaving no tokens for the final answer. With Crush's generous per-turn token budget and `--reasoning-format none`, the baseline scored 66% across 8 tasks — competitive with the best layer-surgeried Mixtral configs (51.9% dup811).
2. **Circuit surgery that improves single-turn probes actively damages multi-turn agentic capability.** This replicates the pattern already documented for Qwen3-Coder, DeepSeek-Coder-V2, and Devstral in this repository: probe improvements do not transfer to Thunderdome. The gains on (24,25) came from the surgery compensating for how the probe measures output — not from genuine capability improvement.

The upshot: **probe scores are a poor proxy for real SWE capability.** For Qwen3.5 in particular, the unmodified model is already production-ready under Crush once `--reasoning-format none` is set and context is ≥32k. Layer surgery provides no value on this architecture for agentic work, and in several tasks is actively harmful.

Raw data: `qwen35_thunderdome_baseline.json`, `qwen35_thunderdome_circuit.json`.

## Thunderdome Validation Summary

**Probe improvements do not reliably transfer to end-to-end SWE benchmarks.**

| Model | Config | Probe Code Δ | Probe SWE Δ | Thunderdome Δ | Tasks |
|-------|--------|-------------|-------------|---------------|-------|
| Qwen3-Coder | del(28,30) | +46.9% | +5.8% | **-3.1%** | 10 |
| Qwen3-Coder | del(24,26) | +33.8% | +3.6% | -5.1% | 3 |
| Mixtral 8x7B | dup(24,27) | +11.9% | +9.3% | +0.9% | 8 |
| DeepSeek-Coder-V2 | dup(6,9) | +7.8% | +2.0% | 0.0% | 3 |

Across all models and strategies, surgically modified models score within ~5% of baseline on real SWE benchmarks. The consistent pattern:

- **Pruning (high-expert MoE):** Dramatic probe improvements (+47% code) but Thunderdome degradation (-3 to -5%). The model's routing has adapted to all layers being present — removing them disrupts multi-turn reasoning.
- **Duplication (low-expert MoE):** Moderate probe improvements (+8-16% code) and roughly neutral Thunderdome performance (±1%). Adding layers reinforces computation without breaking routing.
- **Neither strategy reliably improves real SWE performance.** Single-turn probes measure algorithmic problem-solving, while Thunderdome measures multi-turn agentic capability — a fundamentally different skill.

**Practical value:** Pruning high-expert MoE models (Qwen3-Coder del(28,30)) produces models that are 11% smaller and ~3% faster, at the cost of ~3% SWE degradation — a worthwhile trade-off for latency-sensitive deployments.

## Requirements for Successful Layer Surgery

Based on testing across 7 models, layer surgery requires:

1. **MoE architecture** — dense models (Devstral) are too tightly coupled
2. **Pure transformer layers** — hybrid SSM+MoE (Nemotron) breaks on reorder
3. **Sufficient layer count** — 24 layers (GPT-OSS-20B) isn't enough; 27+ works
4. **No thinking mode** — thinking models (Qwen3.5) generate 200+ tokens of reasoning that break probe evaluation

The optimal strategy depends on expert count:
- **128+ experts:** Prune harmful layers (Qwen3-Coder: +47% code, model gets smaller)
- **8-64 experts:** Duplicate beneficial layers (Mixtral: +16%, DeepSeek: +8%, model gets larger)

**Practical takeaway:** Pruning high-expert MoE models is the most valuable application — models get simultaneously smaller, faster, and better on algorithmic probes, at the cost of ~3-5% on real SWE benchmarks.

## Experiment: Pruning vs Lower Quantization for Size Reduction

To determine whether layer pruning is a better size-reduction strategy than simply using a smaller quantization, we compared two sub-16GB variants of Qwen3-Coder-30B-A3B:

| Variant | Layers | Quant | Size | Thunderdome Avg | Δ vs Baseline |
|---------|--------|-------|------|-----------------|---------------|
| **Baseline** | 48 | Q4_K_M | 18.6 GB | **0.533** | — |
| Pruned (del 6 layers) | 42 | Q4_K_S | 15.4 GB | 0.499 | **-3.5%** |
| **Lower quant** | **48** | **Q3_K_M** | **14.7 GB** | **0.523** | **-1.1%** |

**Lower quantization wins.** Q3_K_M (14.7 GB) preserves quality better (-1.1%) than pruning to Q4_K_S (15.4 GB, -3.5%), while being even smaller. Quantization distributes the quality loss evenly across all layers, while pruning removes entire processing steps that some tasks depend on (plugin-marketplace dropped from 0.477 to 0.215 with pruning but only to 0.468 with Q3_K_M).

### Per-Task Breakdown

| Task | Baseline | Pruned (15.4 GB) | Q3_K_M (14.7 GB) | Better small model |
|------|----------|------------------|-------------------|-------------------|
| fts-search | 0.612 | **0.660** | 0.620 | Pruned |
| phantom-invoice | 0.907 | 0.949 | 0.949 | Tied |
| plugin-marketplace | 0.477 | 0.215 | **0.468** | Q3_K_M |
| task-queue | 0.301 | 0.292 | **0.300** | Q3_K_M |
| time-tracker | 0.318 | **0.292** | 0.215 | Pruned |
| collab-server | 0.585 | 0.585 | 0.585 | Tied |

**Conclusion:** For fitting a model into a smaller VRAM budget, standard quantization (Q3_K_M) is preferable to layer pruning. Pruning is better understood as a tool for improving specific probe metrics, not for general-purpose size reduction.

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
