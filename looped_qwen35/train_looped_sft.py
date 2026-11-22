"""PEFT LoRA SFT trainer for LoopedQwen3_5ForConditionalGeneration on Qwen3.5 multimodal JSONL datasets."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from config import LoopedQwen35Config, resolve_model_id
from dataset import LoopedSFTDataset, MultimodalSFTCollator
from modeling_looped_qwen35 import LoopedQwen3_5ForConditionalGeneration
from rewards import BASE_SYSTEM_PROMPT, TOOL_SCHEMAS

# Verified via preflight inspection of Qwen/Qwen3.5-0.8B and Qwen/Qwen3.5-2B
# (zero overlap with vision encoder Linear leaf names ['linear_fc1', 'linear_fc2', 'proj', 'qkv']):
QWEN35_TEXT_LORA_TARGET_MODULES: list[str] = [
    # Gated Attention layers (full_attention)
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # Gated DeltaNet layers (linear_attention)
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    # SwiGLU MLP layers
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "cot_dataset_builder" / "dataset_clean.jsonl"
DEFAULT_CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"


def gb(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def attach_lora_and_freeze(
    model: LoopedQwen3_5ForConditionalGeneration,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> tuple[int, int]:
    """Freeze vision encoder + base weights, attach LoRA to text decoder, and unfreeze `reinject_logit`."""
    for param in model.parameters():
        param.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=QWEN35_TEXT_LORA_TARGET_MODULES,
    )
    model.base_model = get_peft_model(model.base_model, lora_cfg)

    for param in model._core_model.model.visual.parameters():
        param.requires_grad_(False)
    model.reinject_logit.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_looped_checkpoint(
    model: LoopedQwen3_5ForConditionalGeneration,
    output_dir: Path,
) -> None:
    """Save PEFT LoRA adapter weights and `loop_state.pt` (reinject_logit + config metadata)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(str(output_dir))

    loop_state = {
        "reinject_logit": model.reinject_logit.detach().cpu(),
        "reinject_weight": float(model.reinject_weight.detach().float().item()),
        "loop_config": dataclasses.asdict(model.loop_config),
        "target_modules": QWEN35_TEXT_LORA_TARGET_MODULES,
    }
    torch.save(loop_state, output_dir / "loop_state.pt")


def build_heldout_eval_batches(
    eval_path: Path,
    processor: Any,
    collator: MultimodalSFTCollator,
    device: torch.device,
    max_eval_batches: int = 6,
) -> list[dict[str, torch.Tensor]]:
    """Build held-out evaluation batches from either an SFT JSONL or a prompt JSONL (`eval_heldout.jsonl`)."""
    raw_lines = [l.strip() for l in eval_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not raw_lines:
        return []

    first = json.loads(raw_lines[0])
    if "messages" in first:
        eval_ds = LoopedSFTDataset(eval_path)
        return [
            move_batch_to_device(collator([eval_ds[i]]), device)
            for i in range(min(max_eval_batches, len(eval_ds)))
        ]

    # Prompt-only JSONL (`eval_heldout.jsonl`) with optional `expected_tool` / `expected_args`
    cot_root = eval_path.parent.parent if eval_path.parent.name == "splits" else eval_path.parent
    schema_json = json.dumps(TOOL_SCHEMAS, indent=2, ensure_ascii=False)
    sys_prompt = f"{BASE_SYSTEM_PROMPT}\n\nAVAILABLE TOOL SCHEMAS:\n{schema_json}"
    batches: list[dict[str, torch.Tensor]] = []

    for raw in raw_lines[:max_eval_batches]:
        obj = json.loads(raw)
        raw_img = obj.get("image_path")
        img_paths: list[Path] = []
        if raw_img:
            cand = Path(raw_img)
            if not cand.is_absolute():
                for base in (eval_path.parent, cot_root):
                    if (base / cand).exists():
                        cand = base / cand
                        break
            if cand.exists():
                img_paths.append(cand)

        if obj.get("expected_tool") and isinstance(obj.get("expected_args"), dict):
            target_json = json.dumps(
                {"tool": obj["expected_tool"], "arguments": obj["expected_args"]},
                ensure_ascii=False,
            )
            fake_item = {
                "id": obj.get("id", "eval"),
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": obj["prompt"]},
                    {"role": "assistant", "content": f"<think>\nAnalyze telemetry and call {obj['expected_tool']}.\n</think>\n{target_json}"},
                ],
                "image_paths": img_paths,
            }
            batches.append(move_batch_to_device(collator([fake_item]), device))
        else:
            pil_img = Image.open(img_paths[0]).convert("RGB") if img_paths else None
            try:
                user_content = (
                    [{"type": "image", "image": pil_img}, {"type": "text", "text": obj["prompt"]}]
                    if pil_img is not None
                    else [{"type": "text", "text": obj["prompt"]}]
                )
                inp = processor.apply_chat_template(
                    [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            finally:
                if pil_img is not None:
                    pil_img.close()
            d = move_batch_to_device(dict(inp), device)
            d["labels"] = d["input_ids"].clone()
            batches.append(d)

    return batches


def evaluate_avg_loop_losses(
    model: LoopedQwen3_5ForConditionalGeneration,
    eval_batches: list[dict[str, torch.Tensor]],
    loop_counts: tuple[int, ...] = (1, 2, 3),
) -> dict[int, float]:
    """Average `compare_loop_losses(1, 2, 3)` across multiple held-out batches."""
    accum: dict[int, list[float]] = {k: [] for k in loop_counts}
    for batch in eval_batches:
        res = model.compare_loop_losses(loop_counts=loop_counts, **batch)
        for k, v in res.items():
            accum[k].append(v)
    return {k: round(sum(vs) / max(1, len(vs)), 4) for k, vs in accum.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PEFT LoRA SFT training for Looped Qwen3.5.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to train_sft.jsonl")
    parser.add_argument("--eval-dataset", type=Path, default=None, help="Optional held-out eval JSONL for epoch selection")
    parser.add_argument("--output-dir", type=Path, default=None, help="Checkpoint output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Subdirectory under checkpoints/ if --output-dir omitted")
    parser.add_argument("--model-size", type=str, default="0.8b", choices=["0.8b", "2b"], help="Qwen3.5 size preset")
    parser.add_argument("--base-model-id", type=str, default=None, help="Explicit HF model ID override")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit QLoRA loading (for 2B on 8GB VRAM)")
    parser.add_argument("--num-loops", type=int, default=2, help="Number of decoder loops per forward pass")
    parser.add_argument("--aux-weight", type=float, default=0.1, help="Auxiliary middle-loop loss weight")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional cap on optimization steps (-1 = full epochs)")
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size per device")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak AdamW learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Cosine LR warmup ratio")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum tokenized sequence length")
    parser.add_argument("--smoke-test", action="store_true", help="Run a 5-step end-to-end GPU smoke test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is required for Looped Qwen3.5 LoRA training.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    total_vram_gb = gb(torch.cuda.get_device_properties(0).total_memory)
    model_id = resolve_model_id(args.model_size, args.base_model_id)

    if args.smoke_test:
        if args.max_steps <= 0:
            args.max_steps = 5
        args.grad_accum = 1
        args.batch_size = 1
        args.epochs = max(args.epochs, 1)
        run_name = args.run_name or "smoke_test"
    else:
        run_name = args.run_name or f"looped_qwen35_L{args.num_loops}_{time.strftime('%Y%m%d_%H%M%S')}"

    output_dir = args.output_dir if args.output_dir is not None else (DEFAULT_CHECKPOINTS_DIR / run_name)

    loop_cfg = LoopedQwen35Config(
        base_model_id=model_id,
        num_loops=args.num_loops,
        reinject_init_logit=4.6,
        aux_loop_loss_weight=args.aux_weight,
        dtype="bfloat16",
        load_in_4bit=args.load_in_4bit,
    )

    print("=" * 76)
    print("LOOPED QWEN3.5 PEFT LORA SFT")
    print("=" * 76)
    print(f"GPU             : {torch.cuda.get_device_name(0)} ({total_vram_gb:.2f} GB VRAM)")
    print(f"Model           : {model_id} (4bit={args.load_in_4bit})")
    print(f"Dataset         : {args.dataset}")
    if args.eval_dataset:
        print(f"Eval dataset    : {args.eval_dataset}")
    print(f"Output dir      : {output_dir}")
    print(f"num_loops       : {loop_cfg.num_loops} | aux_weight={loop_cfg.aux_loop_loss_weight} | dtype={loop_cfg.dtype}")
    print(f"LR / GradAccum  : lr={args.lr:.2e} | batch_size={args.batch_size} | grad_accum={args.grad_accum}")

    processor = LoopedQwen3_5ForConditionalGeneration.load_processor(loop_cfg.base_model_id)
    model = LoopedQwen3_5ForConditionalGeneration.from_pretrained(loop_cfg, device=device)

    trainable_params, total_params = attach_lora_and_freeze(model, r=16, lora_alpha=32, lora_dropout=0.05)
    model.enable_gradient_checkpointing()
    print(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100.0 * trainable_params / total_params:.2f}%) | reinject_logit=trainable"
    )

    min_samples = (args.max_steps * args.grad_accum * args.batch_size) if args.max_steps > 0 else 0
    dataset = LoopedSFTDataset(args.dataset, min_samples=min_samples)
    collator = MultimodalSFTCollator(processor=processor, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.smoke_test,
        collate_fn=collator,
    )

    if args.eval_dataset is not None:
        eval_batches = build_heldout_eval_batches(args.eval_dataset, processor, collator, device, max_eval_batches=6)
    else:
        eval_batches = [move_batch_to_device(collator([dataset[0]]), device)]

    valid_label_tokens = int((eval_batches[0]["labels"] != -100).sum().item())
    total_tokens = int(eval_batches[0]["input_ids"].numel())
    print(
        f"Loaded rows     : {len(dataset.base_records)} unique -> {len(dataset)} epoch rows | "
        f"eval_batches={len(eval_batches)} | sample_tokens={total_tokens} (loss_tokens={valid_label_tokens})"
    )

    pre_losses = evaluate_avg_loop_losses(model, eval_batches, loop_counts=(1, 2, 3))
    print(f"Pre-train  compare_loop_losses(1, 2, 3): {pre_losses}")

    steps_per_epoch = max(1, math.ceil(len(dataloader) / args.grad_accum))
    total_opt_steps = args.max_steps if args.max_steps > 0 else (steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_opt_steps * args.warmup_ratio)) if total_opt_steps > 5 else 0

    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and n != "reinject_logit"]
    no_decay_params = [model.reinject_logit]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "lr": args.lr, "weight_decay": 0.01},
            {"params": no_decay_params, "lr": args.lr * 10.0, "weight_decay": 0.0},
        ]
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    print("-" * 76)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)

    opt_step = 0
    micro_step = 0
    accum_total = 0.0
    accum_final = 0.0
    accum_aux = 0.0
    step_losses: list[float] = []
    step_history: list[dict[str, Any]] = []
    epoch_history: list[dict[str, Any]] = []

    best_eval_loop_loss = float("inf")
    best_epoch = 0

    epoch = 0
    while opt_step < total_opt_steps:
        epoch += 1
        for raw_batch in dataloader:
            if opt_step >= total_opt_steps:
                break
            batch = move_batch_to_device(raw_batch, device)
            model._core_model.model.rope_deltas = None

            out = model(**batch, use_cache=False)
            assert out.loss is not None and out.final_loop_loss is not None
            if not torch.isfinite(out.loss):
                raise RuntimeError(f"Non-finite loss encountered at step {opt_step + 1}: {out.loss.item()}")

            scaled_loss = out.loss / args.grad_accum
            scaled_loss.backward()

            accum_total += float(out.loss.detach().float().item())
            accum_final += float(out.final_loop_loss.detach().float().item())
            accum_aux += float(out.aux_loop_loss.detach().float().item()) if out.aux_loop_loss is not None else 0.0
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                avg_total = accum_total / args.grad_accum
                avg_final = accum_final / args.grad_accum
                avg_aux = accum_aux / args.grad_accum
                step_losses.append(avg_total)

                reinject_w = float(model.reinject_weight.detach().float().item())
                cur_lr = scheduler.get_last_lr()[0]
                peak_vram_gb = gb(torch.cuda.max_memory_allocated(device))

                step_history.append(
                    {
                        "step": opt_step,
                        "epoch": epoch,
                        "total_loss": round(avg_total, 5),
                        "final_loss": round(avg_final, 5),
                        "aux_loss": round(avg_aux, 5),
                        "reinject_w": round(reinject_w, 6),
                        "lr": cur_lr,
                        "peak_vram_gb": round(peak_vram_gb, 3),
                    }
                )
                print(
                    f"step {opt_step:>3d}/{total_opt_steps} | "
                    f"total_loss={avg_total:.4f} | final_loss={avg_final:.4f} | aux_loss={avg_aux:.4f} | "
                    f"reinject_w={reinject_w:.5f} | lr={cur_lr:.2e} | peak_vram={peak_vram_gb:.2f} GB"
                )

                accum_total = 0.0
                accum_final = 0.0
                accum_aux = 0.0

        # Flush any remaining partial gradient accumulation at end of epoch
        if micro_step % args.grad_accum != 0 and opt_step < total_opt_steps:
            rem = micro_step % args.grad_accum
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            avg_total = accum_total / rem
            avg_final = accum_final / rem
            avg_aux = accum_aux / rem
            step_losses.append(avg_total)
            reinject_w = float(model.reinject_weight.detach().float().item())
            cur_lr = scheduler.get_last_lr()[0]
            peak_vram_gb = gb(torch.cuda.max_memory_allocated(device))
            step_history.append(
                {
                    "step": opt_step,
                    "epoch": epoch,
                    "total_loss": round(avg_total, 5),
                    "final_loss": round(avg_final, 5),
                    "aux_loss": round(avg_aux, 5),
                    "reinject_w": round(reinject_w, 6),
                    "lr": cur_lr,
                    "peak_vram_gb": round(peak_vram_gb, 3),
                }
            )
            print(
                f"step {opt_step:>3d}/{total_opt_steps} | "
                f"total_loss={avg_total:.4f} | final_loss={avg_final:.4f} | aux_loss={avg_aux:.4f} | "
                f"reinject_w={reinject_w:.5f} | lr={cur_lr:.2e} | peak_vram={peak_vram_gb:.2f} GB"
            )
            accum_total = accum_final = accum_aux = 0.0
            micro_step = 0

        # End-of-epoch held-out loop-loss check & best-checkpoint selection
        epoch_losses = evaluate_avg_loop_losses(model, eval_batches, loop_counts=(1, 2, 3))
        target_loop_loss = epoch_losses.get(args.num_loops, epoch_losses[2])
        is_best = target_loop_loss < best_eval_loop_loss
        if is_best:
            best_eval_loop_loss = target_loop_loss
            best_epoch = epoch
            save_looped_checkpoint(model, output_dir)
        epoch_history.append(
            {
                "epoch": epoch,
                "compare_loop_losses": {str(k): v for k, v in epoch_losses.items()},
                "reinject_w": round(float(model.reinject_weight.detach().float().item()), 6),
                "is_best": is_best,
            }
        )
        print(
            f"[Epoch {epoch}] held-out compare_loop_losses(1, 2, 3): {epoch_losses} "
            f"{'(NEW BEST -> saved)' if is_best else ''}"
        )

        if args.max_steps <= 0 and epoch >= args.epochs:
            break

    print("-" * 76)
    post_losses = epoch_history[-1]["compare_loop_losses"] if epoch_history else {}
    peak_vram_gb = gb(torch.cuda.max_memory_allocated(device))
    print(f"Best Epoch      : {best_epoch} (held-out loop={args.num_loops} loss={best_eval_loop_loss:.4f})")
    print(f"Peak GPU VRAM   : {peak_vram_gb:.2f} GB / {total_vram_gb:.2f} GB")

    metrics_payload = {
        "model_id": model_id,
        "num_loops": args.num_loops,
        "aux_weight": args.aux_weight,
        "load_in_4bit": args.load_in_4bit,
        "pre_train_loop_losses": {str(k): v for k, v in pre_losses.items()},
        "post_train_loop_losses": post_losses,
        "best_epoch": best_epoch,
        "best_eval_loop_loss": best_eval_loop_loss,
        "peak_vram_gb": round(peak_vram_gb, 3),
        "epochs": epoch_history,
        "steps": step_history,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {output_dir} (PEFT adapter + loop_state.pt + training_metrics.json)")

    if args.smoke_test:
        assert len(step_losses) == total_opt_steps, f"Expected {total_opt_steps} steps, ran {len(step_losses)}"
        assert step_losses[-1] < step_losses[0], (
            f"Smoke test expected loss to decrease: start={step_losses[0]:.4f}, end={step_losses[-1]:.4f}"
        )
        assert peak_vram_gb < 8.0, f"Peak VRAM {peak_vram_gb:.2f} GB exceeded 8.00 GB budget"
        assert (output_dir / "loop_state.pt").exists(), "loop_state.pt missing!"
        assert (output_dir / "adapter_config.json").exists(), "adapter_config.json missing!"
        print("SMOKE TEST PASSED!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
