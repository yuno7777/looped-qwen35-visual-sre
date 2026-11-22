<div align="center">

# Looped Hybrid Qwen3.5 for Visual SRE Incident Triage

*Pushing a 0.8B multimodal hybrid model from 0.0% to 86.7% held-out Grafana triage accuracy on a single 8 GB RTX 4060 Laptop GPU.*

<p>
  <img src="https://img.shields.io/badge/PyTorch-2.11_CUDA_12.8-0A0A0A?style=for-the-badge&logo=pytorch&logoColor=A78BFA&labelColor=0A0A0A&color=7C3AED" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Model-Qwen3.5--0.8B-0A0A0A?style=for-the-badge&labelColor=0A0A0A&color=7C3AED" alt="Qwen3.5-0.8B" />
  <img src="https://img.shields.io/badge/v2.0-ETD_+_ARLoop-0A0A0A?style=for-the-badge&labelColor=0A0A0A&color=7C3AED" alt="v2.0 ETD + ARLoop" />
  <img src="https://img.shields.io/badge/RLVR-Dr._GRPO_+_DAPO-0A0A0A?style=for-the-badge&labelColor=0A0A0A&color=7C3AED" alt="Dr. GRPO + DAPO" />
  <img src="https://img.shields.io/badge/KV_Cache-Vis--Exempt_TurboQuant-0A0A0A?style=for-the-badge&labelColor=0A0A0A&color=7C3AED" alt="Vision-Exempt TurboQuant" />
  <img src="https://img.shields.io/badge/Hardware-RTX_4060_8GB-0A0A0A?style=for-the-badge&logo=nvidia&logoColor=A78BFA&labelColor=0A0A0A&color=7C3AED" alt="RTX 4060 8GB" />
</p>

<p>
  <a href="docs/Looped_Hybrid_Qwen35_Visual_SRE_Paper.pdf">
    <img src="https://img.shields.io/badge/Read_the_Paper-arXiv_PDF-7C3AED?style=for-the-badge&logo=adobeacrobatreader&logoColor=white&labelColor=0A0A0A" alt="Read the Paper" />
  </a>
  <a href="docs/Visual_SRE_Looped_Qwen35_Dev_Diary.pdf">
    <img src="https://img.shields.io/badge/Read_the_Dev_Diary-Engineering_Log-A78BFA?style=for-the-badge&logo=readthedocs&logoColor=0A0A0A&labelColor=0A0A0A" alt="Read the Dev Diary" />
  </a>
  <a href="HARD_BENCHMARK_RESULTS.md">
    <img src="https://img.shields.io/badge/3--Tier_Benchmarks-Stock_vs._v1_vs._v2-0A0A0A?style=for-the-badge&labelColor=0A0A0A&color=7C3AED" alt="3-Tier Benchmarks" />
  </a>
  <a href="https://orcid.org/0009-0003-4847-3288">
    <img src="https://img.shields.io/badge/ORCID-0009--0003--4847--3288-0A0A0A?style=for-the-badge&logo=orcid&logoColor=A6CE39&labelColor=0A0A0A&color=7C3AED" alt="ORCID" />
  </a>
</p>

</div>

---

## Visual Showcase

<table>
  <tr>
    <td width="46%" align="center">
      <img src="cot_dataset_builder/samples/sre_042_incident.png" alt="Dark-mode 2x2 SEV1 Grafana Telemetry Dashboard" width="100%" />
      <br />
      <sub><b>Input Telemetry:</b> Synthetic dark-mode 2x2 Grafana dashboard (<code>sre_042_incident.png</code>) encoding a cascading SEV1 multi-service SLO breach.</sub>
    </td>
    <td width="54%" align="center">
      <img src="docs/assets/ablation_and_kv_figure.png" alt="Held-Out Visual SRE Ablation and Hybrid TurboQuant KV Cache" width="100%" />
      <br />
      <sub><b>Results:</b> Held-out Visual SRE progression (0.0% to 73.3% to 86.7%) and 2064-token hybrid KV compression (4.58x Attention KV, untouched DeltaNet).</sub>
    </td>
  </tr>
</table>

---

## Key Results & Architecture Highlights

- **Non-Destructive Looped Hybrid Wrapper (`LoopedQwen3_5ForConditionalGeneration`):** Loops hidden states through Qwen3.5's 24 hybrid decoder layers (**18 Gated DeltaNet** recurrent layers + **6 Gated Attention** layers) `num_loops` times (effective depth 48 at `num_loops=2`) while remaining bit-exact with stock `Qwen/Qwen3.5-0.8B` at `num_loops=1` (`max_abs_diff = 0.000000`).
- **Scalar Input Re-Injection (`reinject_logit`):** Captures `base_inputs_embeds` after the native vision encoder merges image patches and blends `h = sigmoid(reinject_logit) * base_inputs_embeds + (1 - sigmoid(reinject_logit)) * h` before every loop `t >= 1` (initialized at `4.6`, `w ≈ 0.990`), keeping visual charts grounded across loops.
- **Per-Loop Hybrid Cache Isolation (`PerLoopHybridCache`):** Allocates independent `DynamicCache` instances per loop so Gated DeltaNet causal conv and recurrent states are never double- or triple-counted during cached autoregressive decoding.
- **Leakage-Proof Visual SRE Distillation & Looped SFT:** Harvested 34 validated DeepSeek CoT traces (including 8 multi-turn tool-error recovery traces) strictly on the 35 training prompts (SHA-256 verified `0` overlap with the 15 held-out prompts). Assistant-only masked PEFT LoRA (`r=16`, `alpha=32`, 10.82M trainable params = 1.25% across all 12 text-decoder linear projections) made held-out Loop 2 loss (`0.8897`) beat Loop 1 (`0.9562`), lifting tool accuracy from **0.0% to 73.3%** and cutting `<think>` truncation from **100.0% to 0.0%**.
- **Pure-PyTorch GRPO (RLVR) in 3.37 GB Peak VRAM:** Loads a frozen `ref` LoRA adapter (~21 MB overhead) on the same base weights for KL regularization (`beta=0.02`), micro-batches clipped surrogate updates one rollout at a time, and pushes held-out tool accuracy from **73.3% to 86.7% (13/15)** with mean reward `+0.8667`.
- **Weight-Space Programming (WSP) Rollback Guardrail:** Contrastively probed all 24 MLP `down_proj` layers on GRPO passing vs. failing traces (localizing layers 23 and 22, editing `0.0152%` of weights) and cleanly reverted because SFT + GRPO had already eliminated the infinite thinking-loop bug.
- **Hybrid-Safe TurboQuant KV Compression & Vision-Prefix Protection (v2.0):** `TurboQuantHybridCache` applies Haar orthogonal rotation, `uint8` bit-packed Lloyd-Max centroids, and 1-bit QJL key residuals strictly to the **6 Gated Attention layers** while keeping all **18 DeltaNet states** untouched in native `bf16`/`fp32` (**4.58x** Attention KV compression at 3-bit, **3.59x** at 4-bit on 2064 tokens). In **v2.0**, `--protect-vision-prefix` (`--sink-tokens 4`) pins attention sinks and visual patch tokens permanently in native `bf16`, recovering **+10.0 points** on Tier 1 OOD under 4-bit TurboQuant (`62.0% -> 72.0%`, `85.0%` Visual Shift matching `bf16`).
- **ETD Middle-Stack Looping + ARLoop Depth-Attention Residuals (v2.0):** Upgrades full-stack 24-layer looping to **Encode-Think-Decode (ETD)** (`layers 0..3` once, `layers 4..19` looped `num_loops` times with isolated per-loop caches, `layers 20..23` once) paired with **ARLoop** (`arXiv:2611.05234`) RMSNorm depth-attention over `[E_enc, h^(t-1), h^(t)]`, reducing DeltaNet recurrent cache overhead (`37.69 MB -> 31.41 MB`) and cutting 2-loop inference latency to `13.70 s/prompt` (`9.68 tok/s`).
- **Dr. GRPO + DAPO Clip-Higher + Negative Refusal & Active `crop_zoom` (v2.0):** Upgrades RLVR to variance-unnormalized **Dr. GRPO** (`A_i = r_i - mean(r)`) with asymmetric **DAPO** (`arXiv:2503.14476`) clipping (`[-0.20, +0.28]`) in `3.36 GB` peak VRAM, plus 12 negative/no-tool refusal traces and 8 active `crop_zoom` multi-turn traces. Lifts **BFCL v3** tool-calling accuracy to **74.0%** (`+6.0 pts` over v1, `+26.0 pts` over Stock), repairs **BFCL Irrelevance** from `60.0%` to **73.3%**, and recovers **ChartQA** from `70.0%` to **72.0%**.
- **v2.1 Anti-Abstention Dr. GRPO, Scaled 0.8B Training & Fused FLA Triton Kernels:** Scales SFT to 180 validated traces (rebalancing refusal to 8.9%, adding light-mode themes, 1x3/3x1 layouts, and adversarial Conflict Traps with SHA-256 verified 0 overlap), adds a `-0.50` false-abstention penalty in 60-step Dr. GRPO (`3.37 GB` peak VRAM), and dispatches all 18 Gated DeltaNet layers to fused `fla==0.6.0` Triton kernels (`1.24x` decode speedup: `3.93 -> 4.88 tok/s`). Sets new records on **Tier 1 OOD SRE (`84.0%`, `90.0%` schema valid, `86.7%` Conflict Traps)**, **4-bit Vision-Exempt OOD (`76.0%`)**, and fine-tuned **ChartQA (`74.0%`)**.
- **2B Scaling Readiness:** Verified identical 24-layer hybrid topology and 12/12 LoRA target matching (0 vision overlap) for `Qwen/Qwen3.5-2B` with `--model-size 2b` and optional `--load-in-4bit` QLoRA flags.

---

## Benchmark Tables

### 1. Held-Out Visual SRE Evaluation Ablation (`eval_heldout.jsonl`, 15 Unseen Prompts)

| Stage / Configuration | Loops | KV Quant | Tool Acc (%) | Schema Valid (%) | Format (%) | Mean Reward | Trunc Rate (%) | Loop Losses (L1 / L2 / L3) | Peak VRAM (GB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock Qwen3.5-0.8B (1-Loop) | 1 | `none` | 0.0% | 0.0% | 0.0% | -0.2000 | 100.0% | 4.6063 / 4.6418 / 4.651 | 6.47 |
| Stock Qwen3.5-0.8B (2-Loop) | 2 | `none` | 0.0% | 0.0% | 0.0% | -0.2000 | 100.0% | 4.6063 / 4.6418 / 4.651 | 6.51 |
| Looped SFT (2-Loop LoRA) | 2 | `none` | 73.3% | 73.3% | 73.3% | 0.7333 | 0.0% | 4.494 / 4.4882 / 4.5084 | 6.55 |
| **Looped GRPO (2-Loop RLVR)** | **2** | **`none`** | **86.7%** | **86.7%** | **86.7%** | **0.8667** | **0.0%** | **4.4914 / 4.4869 / 4.4983** | **6.55** |
| Looped GRPO + WSP (Reverted) | 2 | `none` | 86.7% | 86.7% | 86.7% | 0.8667 | 0.0% | 4.4914 / 4.4869 / 4.4983 | 6.55 |
| Final Merged + TurboQuant 3-bit | 2 | `turboquant_3bit` | 33.3% | 40.0% | 66.7% | 0.3933 | 6.7% | 4.4928 / 4.4871 / 4.5013 | 6.47 |

### 2. Long-Context TurboQuant KV-Cache Benchmark (`2048 Prefill + 16 Decode`, `num_loops=2`, `residual_window=64`)

| KV Quant Mode | Cached Tokens | 6x Attention KV (MB) | 18x DeltaNet States (MB) | Total Cache (MB) | Attention KV Compression | Peak VRAM (GB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `none` | 2064 | 48.38 MB | 37.69 MB | 86.06 MB | **1.00x** | 2.11 GB |
| `turboquant_3bit` | 2064 | 10.56 MB | 37.69 MB | 48.25 MB | **4.58x** | 2.07 GB |
| `turboquant_4bit` | 2064 | 13.49 MB | 37.69 MB | 51.18 MB | **3.59x** | 2.08 GB |

### 3. 3-Tier Hard Stress-Test & Generalization Gauntlet (Stock vs. v1 vs. v2 vs. v2.1)

Deterministic evaluation (`seed=1337`, greedy `do_sample=False`, `bfloat16`) on a single **RTX 4060 Laptop GPU (8.00 GB VRAM)**. Deployment tables standardize on `bf16` and `4-bit Vision-Exempt` TurboQuant. Full logs live in [HARD_BENCHMARK_RESULTS.md](HARD_BENCHMARK_RESULTS.md).

#### Tier 1: Hard OOD Visual SRE (50 Unseen Dashboards) + Vision-Prefix-Exempt 4-bit TurboQuant

| Configuration | Loops | KV Quant | Vis-Exempt | Overall Acc | Schema Valid | Visual Shift (20) | Pixel-Only (15) | Conflict Trap (15) | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock Qwen3.5-0.8B (1-Loop, bf16) | 1 | `none` | N/A | **0.0%** | 0.0% | 0.0% | 0.0% | 0.0% | 2.02 GB |
| v1 Merged (2-Loop, bf16) | 2 | `none` | N/A | **80.0%** | 88.0% | **85.0%** | **86.7%** | 66.7% | 2.10 GB |
| v1 Merged (2-Loop, Naive TQ 4-bit) | 2 | `turboquant_4bit` | No | 62.0% | 66.0% | 60.0% | 66.7% | 60.0% | 2.04 GB |
| v1 Merged (2-Loop, TQ 4-bit Vis-Exempt) | 2 | `turboquant_4bit` | Yes | 72.0% | 86.0% | **85.0%** | 73.3% | 53.3% | 2.06 GB |
| v2 ETD+ARLoop (2-Loop, bf16) | 2 | `none` | N/A | 66.0% | 84.0% | 70.0% | 66.7% | 60.0% | 2.09 GB |
| v2 ETD+ARLoop (2-Loop, TQ 4-bit Vis-Exempt) | 2 | `turboquant_4bit` | Yes | 62.0% | 82.0% | 75.0% | 53.3% | 53.3% | 2.04 GB |
| **v2.1 ETD+ARLoop (2-Loop, bf16)** | **2** | **`none`** | **N/A** | **84.0%** | **90.0%** | 80.0% | **86.7%** | **86.7%** | **2.09 GB** |
| **v2.1 ETD+ARLoop (2-Loop, TQ 4-bit Vis-Exempt)** | **2** | **`turboquant_4bit`** | **Yes** | **76.0%** | 84.0% | 75.0% | 80.0% | 73.3% | **2.04 GB** |

#### Fused DeltaNet Kernel (`fla` Triton) & `torch.compile` Decode Speedup (`num_loops=2`, 50 Tokens, RTX 4060 8 GB)

| Backend Configuration | DeltaNet Kernel | Mean Latency (s/prompt) | Throughput (tok/s) | Speedup | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Eager (Pure-PyTorch Fallback Baseline) | `pytorch_fallback` | 12.711 s | 3.93 tok/s | 1.00x | 1.74 GB |
| **Fused Kernels (`fla==0.6.0` + `triton-windows==3.9.0`)** | **`fla_triton_fused`** | **10.255 s** | **4.88 tok/s** | **1.24x** | **1.72 GB** |
| Fused Kernels + `torch.compile` (`cudagraphs` fallback) | `fla_triton_fused` | 17.568 s | 2.85 tok/s | 0.73x | 1.73 GB |

#### Tier 2 (`ChartQA`, 50 Samples) & Tier 3 (`BFCL v3`, 50 Samples)

| Model | ChartQA Relaxed Acc | BFCL v3 Overall | BFCL Simple (20) | BFCL Multiple (15) | BFCL Irrelevance (15) | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock Qwen3.5-0.8B (1-Loop) | **80.0%** | 48.0% | 40.0% | 6.7% | **100.0%** | 1.76 GB |
| v1 Merged (2-Loop, bf16) | 70.0% | 68.0% | 65.0% | **80.0%** | 60.0% | 1.81 GB |
| v2 ETD+ARLoop (2-Loop, bf16) | 72.0% | **74.0%** | **75.0%** | 73.3% | **73.3%** | 1.80 GB |
| **v2.1 ETD+ARLoop (2-Loop, bf16)** | **74.0%** | 68.0% | 70.0% | 73.3% | 60.0% | **1.79 GB** |

---

## Reproducible Quickstart

<details>
<summary><b>1. Dataset Generation, Stratified Splits & DeepSeek CoT Harvest</b></summary>

```bash
# Generate 50 balanced Visual SRE prompts (40 dark-mode 2x2 Grafana PNGs + 10 text alerts)
python cot_dataset_builder/generate_sre_dataset.py --count 50

# Create leakage-proof 35 train / 15 held-out stratified splits + SHA-256 manifest
python cot_dataset_builder/build_splits.py \
  --prompts cot_dataset_builder/niche_prompts.jsonl \
  --clean-dataset cot_dataset_builder/dataset_clean.jsonl \
  --eval-ratio 0.30

# Harvest DeepSeek CoT + 25% multi-turn error-recovery traces strictly on training prompts
cd cot_dataset_builder
python generate_dataset.py --input splits/train_grpo_prompts.jsonl --recovery-mode --recovery-ratio 0.25
```

</details>

<details>
<summary><b>2. Looped PEFT LoRA SFT (Phase 3)</b></summary>

```bash
python looped_qwen35/train_looped_sft.py \
  --dataset cot_dataset_builder/splits/train_sft.jsonl \
  --eval-dataset cot_dataset_builder/splits/eval_heldout.jsonl \
  --output-dir looped_qwen35/checkpoints/sft_sre \
  --num-loops 2 --epochs 3 --lr 2e-4 --grad-accum 4
```

</details>

<details>
<summary><b>3. Pure-PyTorch GRPO RLVR (Phase 4)</b></summary>

```bash
python looped_qwen35/train_looped_grpo.py \
  --sft-checkpoint looped_qwen35/checkpoints/sft_sre \
  --prompts cot_dataset_builder/splits/train_grpo_prompts.jsonl \
  --output-dir looped_qwen35/checkpoints/grpo_sre \
  --num-loops 2 --group-size 4 --steps 10 --save-every 5 --lr 1e-5 --kl-beta 0.02
```

</details>

<details>
<summary><b>4. Weight-Space Programming Audit (Phase 5)</b></summary>

```bash
python looped_qwen35/weight_editor.py \
  --checkpoint looped_qwen35/checkpoints/grpo_sre \
  --eval-report looped_qwen35/eval_results/grpo_sre.json \
  --eval-file cot_dataset_builder/splits/eval_heldout.jsonl \
  --run-name grpo_sre --num-loops 2 --top-layers 2 --top-columns 64 --alphas 0.10 0.25
```

</details>

<details>
<summary><b>5. Held-Out Evaluation Suite, Merge & Ablation Benchmark (Phases 2 & 7)</b></summary>

```bash
# Evaluate any checkpoint (stock, PEFT, or merged) on eval_heldout.jsonl
python looped_qwen35/eval_suite.py \
  --checkpoint looped_qwen35/checkpoints/grpo_sre \
  --run-name grpo_sre --num-loops 2 --kv-quant none

# Merge winning LoRA adapter + export self-contained model with loop_state.pt
python looped_qwen35/merge_and_export.py \
  --checkpoint looped_qwen35/checkpoints/grpo_sre \
  --output-dir looped_qwen35/exports/looped_qwen35_sre_final

# Aggregate ablation table + run 2048-token TurboQuant 3-bit/4-bit KV benchmark
python looped_qwen35/benchmark_ablation.py \
  --merged-model looped_qwen35/exports/looped_qwen35_sre_final
```

</details>

<details>
<summary><b>6. Inference with Hybrid-Safe TurboQuant KV Compression</b></summary>

```bash
# Uncompressed bf16 (recommended for dense visual Grafana OCR)
python looped_qwen35/infer_looped.py \
  --checkpoint looped_qwen35/checkpoints/grpo_sre \
  --image cot_dataset_builder/samples/sre_042_incident.png \
  --kv-quant none

# 3-bit or 4-bit Vision-Prefix-Exempt TurboQuant (pins sinks + visual patches in bf16)
python looped_qwen35/infer_looped.py \
  --checkpoint looped_qwen35/checkpoints/grpo_sre \
  --image cot_dataset_builder/samples/sre_042_incident.png \
  --kv-quant turboquant_4bit --residual-window 64 --protect-vision-prefix --sink-tokens 4
```

</details>

<details>
<summary><b>7. v2.0 Pipeline: ETD + ARLoop, Dr. GRPO/DAPO & 3-Tier Hard Benchmark Suite</b></summary>

```bash
# Build v2 training splits (34 core SRE + 12 negative refusal + 8 active crop_zoom)
python cot_dataset_builder/build_v2_dataset.py

# Train v2 SFT (ETD middle-stack 4..19 + ARLoop depth-attention residuals)
python looped_qwen35/train_looped_sft.py \
  --dataset cot_dataset_builder/splits/train_sft_v2.jsonl \
  --eval-dataset cot_dataset_builder/splits/eval_heldout.jsonl \
  --output-dir looped_qwen35/checkpoints/sft_sre_v2 \
  --num-loops 2 --etd-middle-loop --reinject-mode arloop --epochs 3 --lr 2e-4 --grad-accum 4

# Train v2 Dr. GRPO + DAPO Clip-Higher (3.36 GB peak VRAM)
python looped_qwen35/train_looped_grpo.py \
  --sft-checkpoint looped_qwen35/checkpoints/sft_sre_v2 \
  --prompts cot_dataset_builder/splits/train_grpo_prompts_v2.jsonl \
  --output-dir looped_qwen35/checkpoints/grpo_sre_v2 \
  --num-loops 2 --group-size 4 --steps 15 --dr-grpo --clip-low 0.20 --clip-high 0.28

# Merge v2 & run the full Stock vs. v1 vs. v2 3-tier benchmark gauntlet
python looped_qwen35/merge_and_export.py \
  --checkpoint looped_qwen35/checkpoints/grpo_sre_v2 \
  --output-dir looped_qwen35/exports/looped_qwen35_sre_v2
python looped_qwen35/hard_benchmark.py --include-v2 --include-loop-sweep --resume
```

</details>

---

<div align="center">
  <sub>Abhishek Satarkar | November 2026 | <a href="https://orcid.org/0009-0003-4847-3288">ORCID: 0009-0003-4847-3288</a> | <a href="https://github.com/yuno7777/looped-qwen35-visual-sre">GitHub</a></sub>
</div>
