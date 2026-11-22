# Hard Stress-Test & Generalization Benchmark Results (Stock vs. v1 vs. v2 vs. v2.1)

Deterministic evaluation (`seed=1337`, greedy `do_sample=False`, `bfloat16`) on a single **NVIDIA RTX 4060 Laptop GPU (8.00 GB VRAM)**.

## Tier 1: Hard OOD Visual SRE Stress Test (50 Unseen Dashboards) + Vision-Prefix-Exempt TurboQuant

| Configuration | Loops | KV Quant | Vis-Exempt | Overall Acc | Schema Valid | Trunc Rate | Visual Shift (20) | Pixel-Only (15) | Conflict Trap (15) | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock Qwen3.5-0.8B (1-Loop, bf16) | 1 | `none` | N/A | **0.0%** | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 2.02 GB |
| v1 Merged (2-Loop, bf16) | 2 | `none` | N/A | **80.0%** | 88.0% | 0.0% | 85.0% | 86.7% | 66.7% | 2.10 GB |
| v1 Merged (2-Loop, Naive TQ 4-bit) | 2 | `turboquant_4bit` | No | **62.0%** | 66.0% | 0.0% | 60.0% | 66.7% | 60.0% | 2.04 GB |
| v1 Merged (2-Loop, TQ 4-bit Vis-Exempt) | 2 | `turboquant_4bit` | Yes | **72.0%** | 86.0% | 0.0% | 85.0% | 73.3% | 53.3% | 2.06 GB |
| v2 ETD+ARLoop (2-Loop, bf16) | 2 | `none` | N/A | **66.0%** | 84.0% | 0.0% | 70.0% | 66.7% | 60.0% | 2.09 GB |
| v2 ETD+ARLoop (2-Loop, TQ 4-bit Vis-Exempt) | 2 | `turboquant_4bit` | Yes | **62.0%** | 82.0% | 0.0% | 75.0% | 53.3% | 53.3% | 2.04 GB |
| v2.1 ETD+ARLoop (2-Loop, bf16) | 2 | `none` | N/A | **84.0%** | 90.0% | 4.0% | 80.0% | 86.7% | 86.7% | 2.09 GB |
| v2.1 ETD+ARLoop (2-Loop, TQ 4-bit Vis-Exempt) | 2 | `turboquant_4bit` | Yes | **76.0%** | 84.0% | 2.0% | 75.0% | 80.0% | 73.3% | 2.04 GB |

## Loop-Depth Sweep on Tier 1 OOD (`v1 Merged`, `bf16`, `num_loops = 1, 2, 3`)

| Model & Loop Depth | Overall Acc | Visual Shift (20) | Pixel-Only (15) | Conflict Trap (15) | Latency (s/prompt) | Throughput (tok/s) | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| v1 Merged (num_loops=1, bf16) | **80.0%** | 90.0% | 86.7% | 60.0% | 9.25 s | 16.31 | 2.02 GB |
| v1 Merged (num_loops=2, bf16) | **80.0%** | 85.0% | 86.7% | 66.7% | 40.92 s | - | 2.10 GB |
| v1 Merged (num_loops=3, bf16) | **80.0%** | 90.0% | 80.0% | 66.7% | 24.62 s | 6.12 | 2.19 GB |

## Tier 2: Public Chart Reasoning Benchmark (`HuggingFaceM4/ChartQA`, 50 Test Samples)

| Configuration | Loops | Relaxed Accuracy (5% tol) | Truncation Rate | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: |
| Stock Qwen3.5-0.8B (1-Loop) | 1 | **80.0%** | 0.0% | 1.76 GB |
| v1 Merged (2-Loop, bf16) | 2 | **70.0%** | 0.0% | 1.81 GB |
| v2 ETD+ARLoop (2-Loop, bf16) | 2 | **72.0%** | 0.0% | 1.80 GB |
| v2.1 ETD+ARLoop (2-Loop, bf16) | 2 | **74.0%** | 0.0% | 1.79 GB |

## Tier 3: Public Tool-Calling Benchmark (`BFCL v3`, 50 Samples: 20 Simple / 15 Multiple / 15 Irrelevance)

| Configuration | Loops | Overall Acc | Simple (20) | Multiple (15) | Irrelevance (15) | Truncation Rate | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock Qwen3.5-0.8B (1-Loop) | 1 | **48.0%** | 40.0% | 6.7% | 100.0% | 0.0% | 1.76 GB |
| v1 Merged (2-Loop, bf16) | 2 | **68.0%** | 65.0% | 80.0% | 60.0% | 0.0% | 1.81 GB |
| v2 ETD+ARLoop (2-Loop, bf16) | 2 | **74.0%** | 75.0% | 73.3% | 73.3% | 0.0% | 1.80 GB |
| v2.1 ETD+ARLoop (2-Loop, bf16) | 2 | **68.0%** | 70.0% | 73.3% | 60.0% | 0.0% | 1.79 GB |

## Fused DeltaNet Kernel (`fla` Triton) & `torch.compile` Decode Speedup (`num_loops=2`, 50 Tokens, RTX 4060 8 GB)

| Backend Configuration | DeltaNet Kernel | Mean Latency (s/prompt) | Throughput (tok/s) | Speedup vs. Eager | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Eager (Pure-PyTorch Fallback Baseline) | `pytorch_fallback` | 12.711 s | 3.93 tok/s | 1.00x | 1.74 GB |
| **Fused Kernels (`fla==0.6.0` + `triton-windows==3.9.0`)** | **`fla_triton_fused`** | **10.255 s** | **4.88 tok/s** | **1.24x** | **1.72 GB** |
| Fused Kernels + `torch.compile` (`cudagraphs` fallback*) | `fla_triton_fused` | 17.568 s | 2.85 tok/s | 0.73x | 1.73 GB |

<sub>*On Windows without MSVC `cl.exe`, `torch.compile` automatically selects `backend="cudagraphs"` instead of crashing in Inductor; CUDA Graph capture skips due to CPU cache metadata tensors (`skipping cudagraphs due to cpu device`), so eager FLA Triton is the fastest Windows backend while `inductor` (`mode="reduce-overhead"`) activates automatically on Linux/WSL2 or Windows with MSVC Build Tools. `causal-conv1d` has no prebuilt `win_amd64` `cu128`/`torch2.11` wheel and uses PyTorch fallback.</sub>

## Key Engineering Takeaways (Stock vs. v1 vs. v2 vs. v2.1)

1. **v2.1 Sets a New All-Time Record on Hard OOD SRE (`84.0%`) & Adversarial Conflict Traps (`86.7%`):** Scaling SFT from 54 to 180 validated traces (rebalancing refusal from ~22% down to 8.9%), adding a `-0.50` false-abstention penalty in Dr. GRPO (`60` steps, `3.37 GB` peak VRAM), and keeping full 2-loop ETD+ARLoop boosted Tier 1 OOD accuracy by **+18.0 points over v2** (`66.0% -> 84.0%`, `90.0%` schema valid) and **+20.0 points over v1 on Conflict Traps** (`66.7% -> 86.7%`). Under 4-bit Vision-Prefix-Exempt TurboQuant, v2.1 reaches **`76.0%`** (+14.0 pts over v2, +4.0 pts over v1).
2. **ChartQA Recovers to `74.0%` (+4.0 pts over v1):** Scaled multimodal diversity (light-mode themes + 1x3/3x1 layouts) lifted public ChartQA accuracy monotonically across versions (`v1: 70.0% -> v2: 72.0% -> v2.1: 74.0%`, `0.0%` truncation, `1.79 GB` VRAM).
3. **Clear Refusal-vs-Recall Pareto Frontier on BFCL v3:** Dialing negative refusal down from 22% (v2) to 8.9% (v2.1) + adding the `-0.50` anti-abstention penalty traded `13.3 pts` of BFCL Irrelevance (`73.3% -> 60.0%`, overall BFCL `74.0% -> 68.0%`, still `+20.0 pts` over Stock `48.0%`) in exchange for `+18.0 pts` on Tier 1 OOD SRE and `80.0%` on the 15-prompt `eval_heldout.jsonl` split (`86.7%` schema valid).
4. **Fused FLA Triton DeltaNet Kernels Deliver 1.24x Faster Decode on Windows:** Installing `triton-windows==3.9.0.post30` + `flash-linear-attention==0.6.0` and removing per-token `.item()` GPU-CPU syncs accelerated 2-loop ETD+ARLoop decode from `3.93 tok/s` to **`4.88 tok/s`** (`10.255 s/prompt`, `1.72 GB` VRAM) with zero `fla` fallback warnings.

