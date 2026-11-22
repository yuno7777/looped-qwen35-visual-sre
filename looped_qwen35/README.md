# Looped Hybrid Qwen3.5 (`Qwen/Qwen3.5-0.8B`)

A non-destructive PyTorch wrapper that converts `Qwen3_5ForConditionalGeneration` into a **Looped Hybrid Transformer** while preserving its native vision encoder, 3D M-RoPE, thinking capabilities, and 18 Gated DeltaNet + 6 Gated Attention hybrid decoder layers.

## Architecture Summary

1. **Multimodal Input Grounding**: Runs Qwen3.5's native vision encoder (`self.base_model.model.visual`) once and scatters image/video patch embeddings into the token sequence to form `base_inputs_embeds`.
2. **Looped Decoder Execution (`num_loops`)**: Passes `hidden_states` through all 24 hybrid decoder layers `num_loops` times (`1`, `2`, `3`, ...).
3. **Scalar Input Re-Injection**: Before every loop step `t >= 1` (0-indexed), blends the previous loop's hidden state back with `base_inputs_embeds`:
   ```python
   w = torch.sigmoid(self.reinject_logit)  # init logit=4.6 -> w ≈ 0.990
   hidden_states = w * base_inputs_embeds + (1.0 - w) * hidden_states
   ```
   When `num_loops == 1`, re-injection is skipped, giving exact equivalence to stock `Qwen3_5ForConditionalGeneration`.
4. **Per-Loop Hybrid Cache Isolation (`PerLoopHybridCache`)**: Maintains one independent `DynamicCache(config=text_config)` per loop during cached autoregressive decoding (`use_cache=True`), preventing Gated DeltaNet conv/recurrent states from double- or triple-counting tokens across loops. During training (`use_cache=False`), no recurrent state is shared across loops.
5. **Loop-Aware Loss & Diagnostics**: Supports auxiliary middle-loop cross-entropy loss (`aux_loop_loss_weight`) so intermediate loops receive direct gradients, plus `compare_loop_losses(loop_counts=(1, 2, 3), ...)` to measure whether extra loops reduce loss on the same batch.

## Project Structure

```text
looped_qwen35/
├── config.py                  # LoopedQwen35Config dataclass
├── modeling_looped_qwen35.py  # LoopedQwen3_5ForConditionalGeneration, PerLoopHybridCache, generate_looped
├── dataset.py                 # LoopedSFTDataset & MultimodalSFTCollator (assistant-only loss masking)
├── train_looped_sft.py        # PEFT LoRA SFT trainer (Gated Attention + Gated DeltaNet + SwiGLU MLP + reinject_logit)
├── rewards.py                 # Verifiable deterministic tool-calling RLVR rewards
├── train_looped_grpo.py       # Pure-PyTorch GRPO RLVR trainer with PerLoopHybridCache + KL ref
├── turboquant_cache.py        # TurboQuant (~3-bit/4-bit) KV compression for 6 Attention layers only
├── verify_turboquant.py       # 4-check TurboQuant compression, DeltaNet isolation, and generation suite
├── infer_looped.py            # Checkpoint loader + generate_looped inference (<think> + tool call)
├── verify_looped_qwen35.py    # 5-check GPU verification suite (8GB VRAM, bf16)
└── README.md
```

## PEFT LoRA SFT Training & Inference

### 1. Run a 5-Step Smoke Test
```powershell
.venv\Scripts\python.exe looped_qwen35\train_looped_sft.py --smoke-test
```

### 2. Full LoRA SFT Run
```powershell
.venv\Scripts\python.exe looped_qwen35\train_looped_sft.py `
  --dataset cot_dataset_builder\dataset_clean.jsonl `
  --num-loops 2 --epochs 3 --lr 2e-4 --grad-accum 4
```

### 3. Inference from Saved Checkpoint
```powershell
.venv\Scripts\python.exe looped_qwen35\infer_looped.py --checkpoint looped_qwen35\checkpoints\smoke_test
```

Pass `--image path\to\screenshot.png` for multimodal inference.

## Run Verification

From the repository root using the local `.venv` (PyTorch CUDA + Transformers 5.24+):

```powershell
.venv\Scripts\python.exe looped_qwen35\verify_looped_qwen35.py
```

Or from inside `looped_qwen35/`:

```powershell
cd looped_qwen35
..\.venv\Scripts\python.exe verify_looped_qwen35.py
```

The verification script downloads `Qwen/Qwen3.5-0.8B` (first run only) and runs:
1. **Loop=1 Equivalence Test** (vs stock `Qwen3_5ForConditionalGeneration`, tolerance `1e-3`)
2. **Multimodal Forward Pass** (`num_loops=2` and `num_loops=3`, zero `NaN`/`Inf`, plus `compare_loop_losses`)
3. **Gradient Flow Test** (`reinject_logit` + last decoder layer backward pass)
4. **Autoregressive Generation Smoke Test** (`generate_looped` with `use_cache=True`, `num_loops=2` on text and image+text prompts)
5. **VRAM Footprint** (peak allocated GPU memory for `num_loops=2` vs 8GB VRAM)
