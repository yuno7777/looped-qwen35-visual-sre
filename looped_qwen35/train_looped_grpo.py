"""Pure-PyTorch GRPO (Group Relative Policy Optimization) RLVR trainer for Looped Qwen3.5.

Works natively with `LoopedQwen3_5ForConditionalGeneration`, `PerLoopHybridCache`
(18 Gated DeltaNet + 6 Gated Attention layers), PEFT LoRA adapters, and trainable
`reinject_logit` on a single 8GB VRAM GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration

from config import LoopedQwen35Config, resolve_model_id
from modeling_looped_qwen35 import LoopedQwen3_5ForConditionalGeneration
from rewards import BASE_SYSTEM_PROMPT, TOOL_SCHEMAS, score_completion

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "cot_dataset_builder" / "prompts.jsonl"
DEFAULT_SFT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "smoke_test"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "grpo_run"


def gb(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def build_grpo_system_prompt() -> str:
    """Include the exact niche tool schemas in the system prompt so rollouts target valid schemas."""
    schema_json = json.dumps(TOOL_SCHEMAS, indent=2, ensure_ascii=False)
    return (
        f"{BASE_SYSTEM_PROMPT}\n\n"
        f"AVAILABLE TOOL SCHEMAS:\n{schema_json}"
    )


def load_grpo_prompts(prompts_path: Path) -> list[dict[str, Any]]:
    """Load prompts from `prompts.jsonl`, resolving optional local images."""
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    loaded: list[dict[str, Any]] = []
    search_roots = [prompts_path.parent]
    if prompts_path.parent.name == "splits":
        search_roots.append(prompts_path.parent.parent)

    with prompts_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw_img = obj.get("image_path")
            resolved_img: Path | None = None
            if raw_img:
                cand = Path(raw_img)
                if not cand.is_absolute():
                    for root in search_roots:
                        if (root / cand).exists():
                            cand = root / cand
                            break
                if not cand.exists():
                    print(
                        f"[WARN] Skipping prompt {obj.get('id', f'line_{line_no}')}: "
                        f"image file not found ({cand})"
                    )
                    continue
                resolved_img = cand

            loaded.append(
                {
                    "id": obj.get("id", f"prompt_{line_no}"),
                    "prompt": obj["prompt"],
                    "expected_tool": obj.get("expected_tool"),
                    "image_path": resolved_img,
                }
            )

    if not loaded:
        raise ValueError(f"No runnable prompts found in {prompts_path}")
    return loaded


def prepare_prompt_inputs(
    processor: Any,
    system_prompt: str,
    item: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Tokenize a text-only or multimodal prompt with `enable_thinking=True` for GRPO rollouts."""
    pil_img = None
    try:
        if item["image_path"] is not None:
            pil_img = Image.open(item["image_path"]).convert("RGB")
            user_content: list[dict[str, Any]] = [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": item["prompt"]},
            ]
        else:
            user_content = [{"type": "text", "text": item["prompt"]}]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=True,
            return_tensors="pt",
        )
    finally:
        if pil_img is not None:
            pil_img.close()

    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dict(inputs).items()}


def load_sft_model_for_grpo(
    sft_checkpoint: Path,
    device: torch.device,
    num_loops_override: int | None = None,
    need_ref_adapter: bool = True,
    base_model_id_override: str | None = None,
    load_in_4bit: bool = False,
) -> tuple[LoopedQwen3_5ForConditionalGeneration, Any, torch.Tensor]:
    """Load SFT checkpoint with trainable `default` adapter and optional frozen `ref` adapter."""
    state_path = sft_checkpoint / "loop_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing loop_state.pt in SFT checkpoint: {sft_checkpoint}")

    loop_state = torch.load(state_path, map_location="cpu", weights_only=True)
    cfg_dict = dict(loop_state["loop_config"])
    if num_loops_override is not None:
        cfg_dict["num_loops"] = num_loops_override
    if base_model_id_override is not None:
        cfg_dict["base_model_id"] = base_model_id_override
    if load_in_4bit:
        cfg_dict["load_in_4bit"] = True
    loop_cfg = LoopedQwen35Config(**cfg_dict)

    processor = LoopedQwen3_5ForConditionalGeneration.load_processor(loop_cfg.base_model_id)
    load_kwargs: dict[str, Any] = {"dtype": loop_cfg.torch_dtype}
    if loop_cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=loop_cfg.torch_dtype,
        )
        load_kwargs["device_map"] = "auto"

    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(loop_cfg.base_model_id, **load_kwargs)
    if not loop_cfg.load_in_4bit:
        base_model = base_model.to(device)

    peft_model = PeftModel.from_pretrained(
        base_model,
        str(sft_checkpoint),
        adapter_name="default",
        is_trainable=True,
    )
    if need_ref_adapter:
        peft_model.load_adapter(str(sft_checkpoint), adapter_name="ref", is_trainable=False)
    peft_model.set_adapter("default")

    model = LoopedQwen3_5ForConditionalGeneration(base_model=peft_model, loop_config=loop_cfg)
    sft_reinject_logit = loop_state["reinject_logit"].detach().clone().to(device=device, dtype=torch.float32)
    with torch.no_grad():
        model.reinject_logit.copy_(sft_reinject_logit)

    # Ensure vision encoder stays frozen while default LoRA + reinject_logit are trainable
    for param in model._core_model.model.visual.parameters():
        param.requires_grad_(False)
    model.reinject_logit.requires_grad_(True)
    model.enable_gradient_checkpointing()

    return model, processor, sft_reinject_logit


def compute_completion_logps(
    model: LoopedQwen3_5ForConditionalGeneration,
    full_ids: torch.LongTensor,
    prompt_len: int,
    prompt_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute per-token log-probabilities for the completion slice `[prompt_len:]` only."""
    comp_len = full_ids.shape[1] - prompt_len
    if comp_len <= 0:
        return torch.zeros((1, 0), dtype=torch.float32, device=full_ids.device)

    device = full_ids.device
    full_mask = torch.ones_like(full_ids, dtype=torch.long, device=device)

    fwd_kwargs: dict[str, Any] = {
        "input_ids": full_ids,
        "attention_mask": full_mask,
        "use_cache": False,
    }
    if "pixel_values" in prompt_inputs:
        fwd_kwargs["pixel_values"] = prompt_inputs["pixel_values"]
    if "image_grid_thw" in prompt_inputs:
        fwd_kwargs["image_grid_thw"] = prompt_inputs["image_grid_thw"]
    if "mm_token_type_ids" in prompt_inputs:
        prompt_mm = prompt_inputs["mm_token_type_ids"]
        comp_mm = torch.zeros((1, comp_len), dtype=prompt_mm.dtype, device=device)
        fwd_kwargs["mm_token_type_ids"] = torch.cat([prompt_mm, comp_mm], dim=1)

    model._core_model.model.rope_deltas = None
    out = model(**fwd_kwargs)
    assert out.logits is not None

    # Shift logits to predict each completion token t from prefix [:t]
    shift_logits = out.logits[:, prompt_len - 1 : -1, :].float()
    target_ids = full_ids[:, prompt_len:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return token_logps


def save_grpo_checkpoint(
    model: LoopedQwen3_5ForConditionalGeneration,
    output_dir: Path,
    drop_ref_adapter: bool = True,
) -> None:
    """Save RL-tuned PEFT adapter (`default` only) and `loop_state.pt`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if drop_ref_adapter and "ref" in getattr(model.base_model, "peft_config", {}):
        model.base_model.delete_adapter("ref")
    model.base_model.set_adapter("default")
    model.base_model.save_pretrained(str(output_dir), selected_adapters=["default"])

    loop_state = {
        "reinject_logit": model.reinject_logit.detach().cpu(),
        "reinject_weight": float(model.reinject_weight.detach().float().item()),
        "loop_config": dataclasses.asdict(model.loop_config),
    }
    torch.save(loop_state, output_dir / "loop_state.pt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure-PyTorch GRPO RLVR trainer for Looped Qwen3.5.")
    parser.add_argument("--sft-checkpoint", type=Path, default=DEFAULT_SFT_CHECKPOINT, help="SFT checkpoint dir")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH, help="Prompts JSONL path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="GRPO checkpoint output dir")
    parser.add_argument("--model-size", type=str, default=None, choices=["0.8b", "2b"], help="Optional model size preset")
    parser.add_argument("--base-model-id", type=str, default=None, help="Optional HF base model ID override")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit QLoRA loading (for 2B on 8GB VRAM)")
    parser.add_argument("--num-loops", type=int, default=None, help="Override num_loops (default: from SFT state)")
    parser.add_argument("--group-size", type=int, default=4, help="Number of rollouts G per prompt")
    parser.add_argument("--steps", type=int, default=20, help="Number of GRPO optimization steps")
    parser.add_argument("--save-every", type=int, default=0, help="Save periodic checkpoint every N steps (0=off)")
    parser.add_argument("--lr", type=float, default=1e-5, help="AdamW learning rate")
    parser.add_argument("--kl-beta", type=float, default=0.02, help="KL regularization coefficient vs SFT ref")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="GRPO PPO-style ratio clip epsilon")
    parser.add_argument("--max-new-tokens", type=int, default=192, help="Max new tokens per rollout")
    parser.add_argument("--max-think-tokens", type=int, default=256, help="Anti-ramble <think> token budget")
    parser.add_argument("--temperature", type=float, default=0.8, help="Rollout sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Rollout nucleus sampling top-p")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--smoke-test", action="store_true", help="Run a 3-step, G=4 GPU smoke test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is required for Looped Qwen3.5 GRPO training.", file=sys.stderr)
        return 1

    if args.smoke_test:
        args.steps = 3
        args.group_size = 4

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    total_vram_gb = gb(torch.cuda.get_device_properties(0).total_memory)

    print("=" * 76)
    print("LOOPED QWEN3.5 PURE-PYTORCH GRPO (RLVR)")
    print("=" * 76)
    print(f"GPU            : {torch.cuda.get_device_name(0)} ({total_vram_gb:.2f} GB VRAM)")
    print(f"SFT checkpoint : {args.sft_checkpoint}")
    print(f"Prompts        : {args.prompts}")
    print(f"Output dir     : {args.output_dir}")
    print(
        f"Config         : steps={args.steps} | G={args.group_size} | lr={args.lr:.1e} | "
        f"kl_beta={args.kl_beta} | clip_eps={args.clip_eps} | max_new_tokens={args.max_new_tokens}"
    )

    base_override = resolve_model_id(args.model_size, args.base_model_id) if (args.model_size or args.base_model_id) else None
    model, processor, sft_reinject_logit = load_sft_model_for_grpo(
        sft_checkpoint=args.sft_checkpoint,
        device=device,
        num_loops_override=args.num_loops,
        need_ref_adapter=args.kl_beta > 0.0,
        base_model_id_override=base_override,
        load_in_4bit=args.load_in_4bit,
    )
    prompts = load_grpo_prompts(args.prompts)
    system_prompt = build_grpo_system_prompt()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(
        f"Loaded model   : num_loops={model.num_loops} | reinject_w={model.reinject_weight.item():.5f} | "
        f"trainable={trainable_count:,} | runnable_prompts={len(prompts)}"
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and n != "reinject_logit"], "lr": args.lr, "weight_decay": 0.01},
            {"params": [model.reinject_logit], "lr": args.lr * 10.0, "weight_decay": 0.0},
        ]
    )

    torch.cuda.reset_peak_memory_stats(device)
    print("-" * 76)
    grpo_history: list[dict[str, Any]] = []

    for step in range(1, args.steps + 1):
        item = prompts[(step - 1) % len(prompts)]
        prompt_inputs = prepare_prompt_inputs(processor, system_prompt, item, device)
        prompt_len = int(prompt_inputs["input_ids"].shape[1])

        # ------------------------------------------------------------------
        # 1. Rollout Phase (No Grad, PerLoopHybridCache enabled)
        # ------------------------------------------------------------------
        model.eval()
        model.base_model.set_adapter("default")

        rollout_ids_list: list[torch.LongTensor] = []
        old_logps_list: list[torch.Tensor] = []
        ref_logps_list: list[torch.Tensor] = []
        reward_dicts: list[dict[str, Any]] = []
        decoded_list: list[str] = []

        with torch.no_grad():
            for _ in range(args.group_size):
                full_ids = model.generate_looped(
                    **prompt_inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_loops=model.num_loops,
                    use_cache=True,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                comp_ids = full_ids[:, prompt_len:]
                decoded = processor.batch_decode(comp_ids, skip_special_tokens=True)[0]
                score = score_completion(
                    decoded,
                    expected_tool=item["expected_tool"],
                    max_think_tokens=args.max_think_tokens,
                    tokenizer=processor.tokenizer,
                )

                old_lp = compute_completion_logps(model, full_ids, prompt_len, prompt_inputs)

                rollout_ids_list.append(full_ids)
                old_logps_list.append(old_lp)
                reward_dicts.append(score)
                decoded_list.append(decoded)

            # Compute frozen SFT reference log-probs using shared base + frozen "ref" adapter
            if args.kl_beta > 0.0:
                saved_policy_logit = model.reinject_logit.detach().clone()
                model.base_model.set_adapter("ref")
                model.reinject_logit.copy_(sft_reinject_logit)
                try:
                    for full_ids in rollout_ids_list:
                        ref_lp = compute_completion_logps(model, full_ids, prompt_len, prompt_inputs)
                        ref_logps_list.append(ref_lp)
                finally:
                    model.reinject_logit.copy_(saved_policy_logit)
                    model.base_model.set_adapter("default")

        rewards = [d["total_reward"] for d in reward_dicts]
        mean_r = sum(rewards) / len(rewards)
        max_r = max(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = math.sqrt(var_r)

        # Print compact rollout summary for visibility
        tools_seen = [d["parsed_tool"] for d in reward_dicts]
        print(
            f"[step {step}/{args.steps}] prompt={item['id']} (expected={item['expected_tool']}) | "
            f"rewards={[round(r, 3) for r in rewards]} | tools={tools_seen}"
        )

        # ------------------------------------------------------------------
        # 2. Group Advantages & Micro-Batched Policy Update (micro_batch=1)
        # ------------------------------------------------------------------
        if std_r < 1e-6:
            peak_vram_gb = gb(torch.cuda.max_memory_allocated(device))
            reinject_w = float(model.reinject_weight.detach().float().item())
            grpo_history.append(
                {
                    "step": step,
                    "prompt_id": item["id"],
                    "expected_tool": item["expected_tool"],
                    "mean_reward": round(mean_r, 4),
                    "max_reward": round(max_r, 4),
                    "reward_std": 0.0,
                    "policy_loss": 0.0,
                    "kl": 0.0,
                    "skipped_update": True,
                    "reinject_w": round(reinject_w, 6),
                    "peak_vram_gb": round(peak_vram_gb, 3),
                }
            )
            print(
                f"step {step:>3d}/{args.steps} | mean_reward={mean_r:+.3f} | max_reward={max_r:+.3f} | "
                f"reward_std=0.000 | policy_loss=SKIP | kl=0.0000 | "
                f"reinject_w={reinject_w:.5f} | peak_vram={peak_vram_gb:.2f} GB"
            )
            if args.save_every > 0 and step % args.save_every == 0 and step < args.steps:
                ckpt_dir = args.output_dir / f"step_{step}"
                save_grpo_checkpoint(model, ckpt_dir, drop_ref_adapter=False)
                print(f"  -> periodic checkpoint saved: {ckpt_dir}")
            continue

        advantages = [(r - mean_r) / (std_r + 1e-6) for r in rewards]

        model.train()
        model.base_model.set_adapter("default")
        optimizer.zero_grad(set_to_none=True)

        total_pg_loss = 0.0
        total_kl = 0.0
        valid_rollouts = 0

        for idx in range(args.group_size):
            full_ids = rollout_ids_list[idx]
            if full_ids.shape[1] <= prompt_len:
                continue

            old_lp = old_logps_list[idx].detach()
            new_lp = compute_completion_logps(model, full_ids, prompt_len, prompt_inputs)

            log_ratio = (new_lp - old_lp).clamp(-20.0, 20.0)
            ratio = torch.exp(log_ratio)
            adv = advantages[idx]

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
            pg_loss = -torch.min(surr1, surr2).mean()

            if args.kl_beta > 0.0:
                ref_lp = ref_logps_list[idx].detach()
                ref_diff = (ref_lp - new_lp).clamp(-20.0, 20.0)
                kl = (torch.exp(ref_diff) - ref_diff - 1.0).mean()
            else:
                kl = torch.zeros((), dtype=torch.float32, device=device)

            loss = (pg_loss + args.kl_beta * kl) / args.group_size
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite GRPO loss at step {step}, rollout {idx}: {loss.item()}")

            loss.backward()
            total_pg_loss += float(pg_loss.detach().item())
            total_kl += float(kl.detach().item())
            valid_rollouts += 1

            del new_lp, log_ratio, ratio, surr1, surr2, pg_loss, kl, loss

        if valid_rollouts > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at step {step}: {grad_norm}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        avg_pg = total_pg_loss / max(1, valid_rollouts)
        avg_kl = total_kl / max(1, valid_rollouts)
        reinject_w = float(model.reinject_weight.detach().float().item())
        peak_vram_gb = gb(torch.cuda.max_memory_allocated(device))

        grpo_history.append(
            {
                "step": step,
                "prompt_id": item["id"],
                "expected_tool": item["expected_tool"],
                "mean_reward": round(mean_r, 4),
                "max_reward": round(max_r, 4),
                "reward_std": round(std_r, 4),
                "policy_loss": round(avg_pg, 5),
                "kl": round(avg_kl, 5),
                "skipped_update": False,
                "reinject_w": round(reinject_w, 6),
                "peak_vram_gb": round(peak_vram_gb, 3),
            }
        )
        print(
            f"step {step:>3d}/{args.steps} | mean_reward={mean_r:+.3f} | max_reward={max_r:+.3f} | "
            f"reward_std={std_r:.3f} | policy_loss={avg_pg:+.4f} | kl={avg_kl:.4f} | "
            f"reinject_w={reinject_w:.5f} | peak_vram={peak_vram_gb:.2f} GB"
        )

        # Print best rollout snippet on step 1 for quick inspection
        if step == 1:
            best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
            snippet = decoded_list[best_idx].replace("\n", "\\n")
            print(f"  -> best rollout (r={rewards[best_idx]:+.2f}): {snippet[:220]}")

        if args.save_every > 0 and step % args.save_every == 0 and step < args.steps:
            ckpt_dir = args.output_dir / f"step_{step}"
            save_grpo_checkpoint(model, ckpt_dir, drop_ref_adapter=False)
            print(f"  -> periodic checkpoint saved: {ckpt_dir}")

    print("-" * 76)
    peak_vram_gb = gb(torch.cuda.max_memory_allocated(device))
    print(f"Overall Peak GPU VRAM : {peak_vram_gb:.2f} GB / {total_vram_gb:.2f} GB")

    save_grpo_checkpoint(model, args.output_dir, drop_ref_adapter=True)
    metrics_payload = {
        "sft_checkpoint": str(args.sft_checkpoint),
        "num_loops": model.num_loops,
        "group_size": args.group_size,
        "steps": args.steps,
        "lr": args.lr,
        "kl_beta": args.kl_beta,
        "clip_eps": args.clip_eps,
        "peak_vram_gb": round(peak_vram_gb, 3),
        "final_reinject_w": round(float(model.reinject_weight.detach().float().item()), 6),
        "history": grpo_history,
    }
    (args.output_dir / "grpo_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"Saved GRPO checkpoint : {args.output_dir} (PEFT adapter + loop_state.pt + grpo_metrics.json)")

    if args.smoke_test:
        assert peak_vram_gb < 8.0, f"Peak VRAM {peak_vram_gb:.2f} GB exceeded 8.00 GB budget"
        assert (args.output_dir / "loop_state.pt").exists(), "loop_state.pt missing!"
        assert (args.output_dir / "adapter_config.json").exists(), "adapter_config.json missing!"
        print("GRPO SMOKE TEST PASSED!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
