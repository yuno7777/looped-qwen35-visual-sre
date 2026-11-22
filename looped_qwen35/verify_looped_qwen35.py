"""End-to-end verification suite for LoopedQwen3_5ForConditionalGeneration on an 8GB CUDA GPU."""

from __future__ import annotations

import gc
import sys
import time
from typing import Any

import torch
from PIL import Image, ImageDraw
import transformers
import accelerate

from config import LoopedQwen35Config
from modeling_looped_qwen35 import LoopedQwen3_5ForConditionalGeneration


def make_synthetic_image(width: int = 224, height: int = 224) -> Image.Image:
    """Create a simple high-contrast synthetic PIL image for multimodal testing."""
    img = Image.new("RGB", (width, height), color=(245, 248, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([24, 24, 200, 110], fill=(30, 100, 220), outline=(10, 40, 120), width=4)
    draw.ellipse([64, 125, 160, 210], fill=(230, 70, 50), outline=(120, 20, 10), width=4)
    return img


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def build_text_inputs(processor: Any, prompt: str, device: torch.device) -> dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return move_batch_to_device(dict(inputs), device)


def build_multimodal_inputs(
    processor: Any,
    image: Image.Image,
    prompt: str,
    device: torch.device,
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return move_batch_to_device(dict(inputs), device)


def gb(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is required for this verification script.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    total_vram_gb = gb(torch.cuda.get_device_properties(0).total_memory)

    print("=" * 72)
    print("PREFLIGHT ENVIRONMENT")
    print("=" * 72)
    print(f"PyTorch      : {torch.__version__}")
    print(f"Transformers : {transformers.__version__}")
    print(f"Accelerate   : {accelerate.__version__}")
    print(f"GPU          : {torch.cuda.get_device_name(0)} ({total_vram_gb:.2f} GB VRAM)")

    cfg = LoopedQwen35Config(
        base_model_id="Qwen/Qwen3.5-0.8B",
        num_loops=2,
        reinject_init_logit=4.6,
        aux_loop_loss_weight=0.1,
        dtype="bfloat16",
    )

    print(f"\nLoading {cfg.base_model_id} in {cfg.dtype} on {device}...")
    t0 = time.perf_counter()
    processor = LoopedQwen3_5ForConditionalGeneration.load_processor(cfg.base_model_id)
    model = LoopedQwen3_5ForConditionalGeneration.from_pretrained(cfg, device=device)
    model.eval()
    load_vram_gb = gb(torch.cuda.memory_allocated(device))
    print(
        f"Loaded in {time.perf_counter() - t0:.1f}s | "
        f"layers={model.text_config.num_hidden_layers} "
        f"(linear={sum(t == 'linear_attention' for t in model.text_config.layer_types)}, "
        f"full={sum(t == 'full_attention' for t in model.text_config.layer_types)}) | "
        f"reinject_w={model.reinject_weight.item():.5f} | "
        f"base_weights_vram={load_vram_gb:.2f} GB"
    )

    synthetic_img = make_synthetic_image()
    text_inputs = build_text_inputs(processor, "What is 12 + 30? Answer concisely.", device)
    mm_inputs = build_multimodal_inputs(
        processor,
        synthetic_img,
        "Describe the shapes and colors visible in this image in one short sentence.",
        device,
    )

    # ------------------------------------------------------------------
    # CHECK 1: Loop=1 Equivalence Test
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("CHECK 1: Loop=1 Equivalence Test (stock vs wrapped num_loops=1)")
    print("=" * 72)
    with torch.no_grad():
        model.base_model.model.rope_deltas = None
        stock_out = model.base_model(**mm_inputs, use_cache=False)

        model.base_model.model.rope_deltas = None
        loop1_out = model(**mm_inputs, use_cache=False, num_loops=1)

    diff = (stock_out.logits.float() - loop1_out.logits.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"Multimodal logits shape : {tuple(loop1_out.logits.shape)}")
    print(f"Max abs diff            : {max_diff:.6e} (tolerance: 1e-3)")
    print(f"Mean abs diff           : {mean_diff:.6e}")
    assert max_diff <= 1e-3, f"Loop=1 equivalence failed: max_diff={max_diff}"
    print("CHECK 1 PASSED")
    del stock_out, loop1_out, diff
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # CHECK 2: Multimodal Forward Pass (num_loops=2 and num_loops=3)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("CHECK 2: Multimodal Forward Pass (num_loops=2 & num_loops=3)")
    print("=" * 72)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for k in (2, 3):
            model.base_model.model.rope_deltas = None
            out = model(**mm_inputs, use_cache=False, num_loops=k)
            logits = out.logits
            assert logits is not None
            has_nan = bool(torch.isnan(logits).any().item())
            has_inf = bool(torch.isinf(logits).any().item())
            print(
                f"num_loops={k} | logits_shape={tuple(logits.shape)} | "
                f"NaN={has_nan} | Inf={has_inf} | reinject_w={out.reinject_weight:.5f}"
            )
            assert not has_nan and not has_inf, f"Found NaN/Inf for num_loops={k}"
            if k == 2:
                fwd_loop2_peak_gb = gb(torch.cuda.max_memory_allocated(device))
            del out, logits

        # Also test compare_loop_losses diagnostic helper
        diag_labels = mm_inputs["input_ids"].clone()
        loop_losses = model.compare_loop_losses(loop_counts=(1, 2, 3), labels=diag_labels, **mm_inputs)
        print(f"compare_loop_losses(1, 2, 3): { {k: round(v, 4) for k, v in loop_losses.items()} }")
        del diag_labels

    print("CHECK 2 PASSED")
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # CHECK 3: Gradient Flow Test
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("CHECK 3: Gradient Flow Test (reinject_logit + last decoder layer)")
    print("=" * 72)
    for p in model.parameters():
        p.requires_grad_(False)

    model.reinject_logit.requires_grad_(True)
    target_layer = model.base_model.model.language_model.layers[-1]
    for p in target_layer.parameters():
        p.requires_grad_(True)

    model.train()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)

    labels = mm_inputs["input_ids"].clone()
    model.base_model.model.rope_deltas = None
    out = model(**mm_inputs, labels=labels, use_cache=False, num_loops=2)
    assert out.loss is not None
    out.loss.backward()

    bwd_loop2_peak_gb = gb(torch.cuda.max_memory_allocated(device))

    reinject_grad = model.reinject_logit.grad
    assert reinject_grad is not None, "reinject_logit.grad is None!"
    reinject_grad_val = reinject_grad.detach().float().item()
    assert torch.isfinite(reinject_grad).all(), "reinject_logit.grad is not finite!"
    assert abs(reinject_grad_val) > 0.0, "reinject_logit.grad is zero!"

    decoder_grad_norms = []
    for name, param in target_layer.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name}"
            decoder_grad_norms.append((name, float(param.grad.detach().float().norm().item())))

    nonzero_decoder_grads = [(n, g) for n, g in decoder_grad_norms if g > 0.0]
    assert len(nonzero_decoder_grads) > 0, "No non-zero gradients reached decoder layer!"

    print(
        f"total_loss={out.loss.item():.4f} | "
        f"final_loop_loss={out.final_loop_loss.item():.4f} | "
        f"aux_loop_loss={out.aux_loop_loss.item():.4f}"
    )
    print(f"reinject_logit.grad     : {reinject_grad_val:.6e} (finite & non-zero)")
    print(
        f"last_layer params w/grad: {len(nonzero_decoder_grads)}/{len(decoder_grad_norms)} "
        f"(sample: {nonzero_decoder_grads[0][0]} norm={nonzero_decoder_grads[0][1]:.4e})"
    )
    print("CHECK 3 PASSED")

    # Clean up gradients and switch back to eval mode
    model.zero_grad(set_to_none=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    del out, labels
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # CHECK 4: Autoregressive Generation Smoke Test (use_cache=True, num_loops=2)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("CHECK 4: Autoregressive Generation Smoke Test (num_loops=2, use_cache=True)")
    print("=" * 72)
    torch.cuda.reset_peak_memory_stats(device)

    gen_text_ids = model.generate_looped(
        **text_inputs,
        max_new_tokens=32,
        num_loops=2,
        use_cache=True,
        do_sample=False,
    )
    new_text_ids = gen_text_ids[:, text_inputs["input_ids"].shape[1] :]
    decoded_text = processor.batch_decode(new_text_ids, skip_special_tokens=True)[0]
    print(f"[Text-only output ({new_text_ids.shape[1]} tokens)]:\n{decoded_text!r}")
    assert new_text_ids.shape[1] > 0 and decoded_text.strip(), "Empty text generation!"

    gen_mm_ids = model.generate_looped(
        **mm_inputs,
        max_new_tokens=32,
        num_loops=2,
        use_cache=True,
        do_sample=False,
    )
    new_mm_ids = gen_mm_ids[:, mm_inputs["input_ids"].shape[1] :]
    decoded_mm = processor.batch_decode(new_mm_ids, skip_special_tokens=True)[0]
    print(f"\n[Image+text output ({new_mm_ids.shape[1]} tokens)]:\n{decoded_mm!r}")
    assert new_mm_ids.shape[1] > 0 and decoded_mm.strip(), "Empty multimodal generation!"

    gen_loop2_peak_gb = gb(torch.cuda.max_memory_allocated(device))
    print("\nCHECK 4 PASSED")

    # ------------------------------------------------------------------
    # CHECK 5: VRAM Footprint Summary (num_loops=2)
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("CHECK 5: VRAM Footprint (num_loops=2, bfloat16)")
    print("=" * 72)
    overall_peak_gb = max(fwd_loop2_peak_gb, gen_loop2_peak_gb, bwd_loop2_peak_gb)
    headroom_gb = total_vram_gb - overall_peak_gb
    print(f"Base weights (bf16)               : {load_vram_gb:.2f} GB")
    print(f"Peak forward  (num_loops=2)       : {fwd_loop2_peak_gb:.2f} GB")
    print(f"Peak generate (num_loops=2, cache): {gen_loop2_peak_gb:.2f} GB")
    print(f"Peak backward (num_loops=2, train): {bwd_loop2_peak_gb:.2f} GB")
    print(f"Overall peak / GPU total          : {overall_peak_gb:.2f} GB / {total_vram_gb:.2f} GB")
    print(f"Estimated headroom for LoRA       : {headroom_gb:.2f} GB")
    print("CHECK 5 PASSED")

    print("\nALL 5 VERIFICATION CHECKS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
