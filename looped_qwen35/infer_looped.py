"""Inference script for loading a saved Looped Qwen3.5 PEFT LoRA checkpoint + loop_state.pt."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration

from config import LoopedQwen35Config, resolve_model_id
from modeling_looped_qwen35 import LoopedQwen3_5ForConditionalGeneration

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise tool-calling assistant. Analyze the user's request step by step, "
    "select the single best tool from the available tool schemas, and respond with ONLY a "
    "valid JSON object of the form:\n"
    '{"tool": "<tool_name>", "arguments": { ... }}\n'
    "Do not include any prose, explanation, or markdown formatting outside the JSON object."
)

DEFAULT_TEST_PROMPT = (
    "Enterprise customer CUST-88412 reports that SSO login via Okta is returning HTTP 500 "
    "across their entire organization since 09:15 UTC. Open an urgent support ticket and flag it for escalation."
)

THINK_SPLIT_RE = re.compile(r"^(.*?)</think>\s*(.*)$", re.DOTALL | re.IGNORECASE)


def load_looped_checkpoint(
    checkpoint_dir: Path,
    device: torch.device,
    num_loops_override: int | None = None,
    base_model_id_override: str | None = None,
) -> tuple[LoopedQwen3_5ForConditionalGeneration, Any]:
    """Load base Qwen3.5 + PEFT adapter (or merged export) and restore `reinject_logit` from `loop_state.pt`."""
    state_path = checkpoint_dir / "loop_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing loop_state.pt in checkpoint directory: {checkpoint_dir}")

    loop_state = torch.load(state_path, map_location="cpu", weights_only=True)
    cfg_dict = dict(loop_state["loop_config"])
    if num_loops_override is not None:
        cfg_dict["num_loops"] = num_loops_override
    if base_model_id_override is not None:
        cfg_dict["base_model_id"] = base_model_id_override
    loop_cfg = LoopedQwen35Config(**cfg_dict)

    is_peft = (checkpoint_dir / "adapter_config.json").exists()
    if is_peft:
        processor = LoopedQwen3_5ForConditionalGeneration.load_processor(loop_cfg.base_model_id)
        base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            loop_cfg.base_model_id,
            dtype=loop_cfg.torch_dtype,
        ).to(device)
        wrapped = PeftModel.from_pretrained(base_model, str(checkpoint_dir))
        wrapped.eval()
    else:
        processor = LoopedQwen3_5ForConditionalGeneration.load_processor(str(checkpoint_dir))
        wrapped = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(checkpoint_dir),
            dtype=loop_cfg.torch_dtype,
        ).to(device)
        wrapped.eval()

    model = LoopedQwen3_5ForConditionalGeneration(base_model=wrapped, loop_config=loop_cfg)
    model.restore_loop_state(loop_state, device)
    model.eval()
    return model, processor


def crop_zoom_panel(image: Image.Image, panel: str) -> Image.Image:
    """Active Chart Zoom helper (DeepEyes-style): crop a 2x2 dashboard quadrant at 2x resolution."""
    w, h = image.size
    boxes = {
        "top_left": (0, int(h * 0.12), w // 2, int(h * 0.58)),
        "top_right": (w // 2, int(h * 0.12), w, int(h * 0.58)),
        "bottom_left": (0, int(h * 0.52), w // 2, h),
        "bottom_right": (w // 2, int(h * 0.52), w, h),
    }
    box = boxes.get(panel, (0, 0, w, h))
    cropped = image.crop(box)
    return cropped.resize((cropped.width * 2, cropped.height * 2), Image.Resampling.LANCZOS)


def split_thinking_and_answer(decoded: str) -> tuple[str, str]:
    """Separate `<think>...</think>` reasoning from the final tool call output."""
    text = decoded.strip()
    if text.lower().startswith("<think>"):
        text = text[len("<think>") :].lstrip("\n")

    match = THINK_SPLIT_RE.match(text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a saved Looped Qwen3.5 LoRA checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to saved checkpoint directory")
    parser.add_argument("--model-size", type=str, default=None, choices=["0.8b", "2b"], help="Optional size preset")
    parser.add_argument("--base-model-id", type=str, default=None, help="Optional HF base model ID override")
    parser.add_argument("--prompt", type=str, default=DEFAULT_TEST_PROMPT, help="User prompt")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt")
    parser.add_argument("--image", type=Path, default=None, help="Optional path to an input image")
    parser.add_argument("--num-loops", type=int, default=None, help="Override num_loops from loop_state.pt")
    parser.add_argument(
        "--kv-quant",
        type=str,
        default="none",
        choices=["none", "turboquant_3bit", "turboquant_4bit"],
        help="KV-cache quantization mode for the 6 Gated Attention layers (18 DeltaNet layers stay untouched)",
    )
    parser.add_argument(
        "--residual-window",
        type=int,
        default=64,
        help="Recent bf16 token window before flushing into bit-packed TurboQuant store",
    )
    parser.add_argument(
        "--protect-vision-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep sink tokens and visual patch tokens in native bf16 under TurboQuant (default: True)",
    )
    parser.add_argument("--sink-tokens", type=int, default=4, help="Number of initial sink tokens kept in bf16")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Maximum tokens to generate")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling (default: greedy)")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling top-p")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Looped Qwen3.5 checkpoint from: {args.checkpoint} (device={device})")
    base_override = resolve_model_id(args.model_size, args.base_model_id) if (args.model_size or args.base_model_id) else None
    model, processor = load_looped_checkpoint(
        checkpoint_dir=args.checkpoint,
        device=device,
        num_loops_override=args.num_loops,
        base_model_id_override=base_override,
    )
    print(
        f"Loaded | num_loops={model.num_loops} | kv_quant={args.kv_quant} | "
        f"reinject_logit={model.reinject_logit.item():.5f} | "
        f"reinject_w={model.reinject_weight.item():.5f}"
    )

    pil_img = None
    try:
        if args.image is not None:
            if not args.image.exists():
                raise FileNotFoundError(f"Input image not found: {args.image}")
            pil_img = Image.open(args.image).convert("RGB")
            user_content: list[dict[str, Any]] = [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": args.prompt},
            ]
        else:
            user_content = [{"type": "text", "text": args.prompt}]

        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # enable_thinking=True appends `<|im_start|>assistant\n<think>\n` so Qwen3.5 emits CoT first
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dict(inputs).items()}

        generated_ids = model.generate_looped(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_loops=model.num_loops,
            use_cache=True,
            kv_quant=args.kv_quant,
            residual_window=args.residual_window,
            protect_vision_prefix=args.protect_vision_prefix,
            sink_tokens=args.sink_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    finally:
        if pil_img is not None:
            pil_img.close()

    prompt_len = inputs["input_ids"].shape[1]
    new_ids = generated_ids[:, prompt_len:]
    raw_decoded = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    thinking, final_answer = split_thinking_and_answer(raw_decoded)

    print("\n" + "=" * 72)
    print(f"GENERATED OUTPUT ({new_ids.shape[1]} tokens, num_loops={model.num_loops})")
    print("=" * 72)
    print("<think>")
    print(thinking if thinking else "(no </think> delimiter emitted within max_new_tokens)")
    print("</think>")
    print("\n[Final Tool Call / Answer]")
    print(final_answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
