# Looped Hybrid Qwen3.5 (Visual SRE Incident Triage) — Benchmark & Ablation Results

## 1. Held-Out Visual SRE Evaluation Ablation (`eval_heldout.jsonl`)

| Stage / Configuration | Loops | KV Quant | Tool Acc (%) | Schema Valid (%) | Format (%) | Mean Reward | Trunc Rate (%) | Loop Losses (L1 / L2 / L3) | Peak VRAM (GB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Stock Qwen3.5-0.8B (1-Loop) | 1 | `none` | 0.0% | 0.0% | 0.0% | -0.2000 | 100.0% | 4.6063 / 4.6418 / 4.651 | 6.47 |
| Stock Qwen3.5-0.8B (2-Loop) | 2 | `none` | 0.0% | 0.0% | 0.0% | -0.2000 | 100.0% | 4.6063 / 4.6418 / 4.651 | 6.51 |
| Looped SFT (2-Loop LoRA) | 2 | `none` | 73.3% | 73.3% | 73.3% | 0.7333 | 0.0% | 4.494 / 4.4882 / 4.5084 | 6.55 |
| Looped GRPO (2-Loop RLVR) | 2 | `none` | 86.7% | 86.7% | 86.7% | 0.8667 | 0.0% | 4.4914 / 4.4869 / 4.4983 | 6.55 |
| Looped GRPO + WSP | 2 | `none` | 86.7% | 86.7% | 86.7% | 0.8667 | 0.0% | 4.4914 / 4.4869 / 4.4983 | 6.55 |
| Final Merged + TurboQuant 3-bit | 2 | `turboquant_3bit` | 33.3% | 40.0% | 66.7% | 0.3933 | 6.7% | 4.4928 / 4.4871 / 4.5013 | 6.47 |

## 2. Long-Context TurboQuant KV-Cache Benchmark (2048 Prefill + 16 Decode, `num_loops=2`, `residual_window=64`)

| KV Quant Mode | Cached Tokens | 6x Attention KV (MB) | 18x DeltaNet States (MB) | Total Cache (MB) | Attention KV Compression | Peak VRAM (GB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `none` | 2064 | 48.38 MB | 37.69 MB | 86.06 MB | **1.00x** | 2.11 GB |
| `turboquant_3bit` | 2064 | 10.56 MB | 37.69 MB | 48.25 MB | **4.58x** | 2.07 GB |
| `turboquant_4bit` | 2064 | 13.49 MB | 37.69 MB | 51.18 MB | **3.59x** | 2.08 GB |
