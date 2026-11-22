"""Held-out evaluation suite for stock, Looped SFT, Looped GRPO, weight-edited, and merged Qwen3.5 models."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration

from config import LoopedQwen35Config, resolve_model_id
from modeling_looped_qwen35 import LoopedQwen3_5ForConditionalGeneration
from rewards import BASE_SYSTEM_PROMPT, TOOL_SCHEMAS, score_completion

DEFAULT_EVAL_FILE = Path(__file__).resolve().parent.parent / "cot_dataset_builder" / "splits" / "eval_heldout.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "eval_results"


def gb(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def build_eval_system_prompt() -> str:
    schema_json = json.dumps(TOOL_SCHEMAS, indent=2, ensure_ascii=False)
    return f"{BASE_SYSTEM_PROMPT}\n\nAVAILABLE TOOL SCHEMAS:\n{schema_json}"


def load_eval_records(eval_file: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    if not eval_file.exists():
        raise FileNotFoundError(f"Held-out eval file not found: {eval_file}")

    records: list[dict[str, Any]] = []
    # Resolve relative image paths against eval_file parent or cot_dataset_builder root
    cot_root = eval_file.parent.parent if eval_file.parent.name == "splits" else eval_file.parent
    with eval_file.open("r", encoding="utf-8") as fh:
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
                    for base in (eval_file.parent, cot_root):
                        if (base / cand).exists():
                            cand = base / cand
                            break
                if cand.exists():
                    resolved_img = cand
                else:
                    print(f"[WARN] Missing eval image for {obj.get('id', line_no)}: {cand} (skipping)")
                    continue
            records.append(
                {
                    "id": obj.get("id", f"eval_{line_no}"),
                    "prompt": obj["prompt"],
                    "expected_tool": obj.get("expected_tool"),
                    "expected_args": obj.get("expected_args"),
                    "image_path": resolved_img,
                    "raw_image_path": raw_img,
                }
            )
            if max_samples is not None and len(records) >= max_samples:
                break

    if not records:
        raise ValueError(f"No runnable eval records loaded from {eval_file}")
    return records


def load_model_for_eval(
    checkpoint: Path | None,
    base_model_id: str,
    num_loops: int,
    kv_quant: str,
    residual_window: int,
    device: torch.device,
    protect_vision_prefix: bool = True,
    sink_tokens: int = 4,
) -> tuple[LoopedQwen3_5ForConditionalGeneration, Any, str]:
    """Load stock, PEFT adapter checkpoint, or merged exported model (v1 scalar or v2 ETD+ARLoop)."""
    if checkpoint is None:
        cfg = LoopedQwen35Config(
            base_model_id=base_model_id,
            num_loops=num_loops,
            kv_quant=kv_quant,
            kv_residual_window=residual_window,
            kv_protect_vision_prefix=protect_vision_prefix,
            kv_sink_tokens=sink_tokens,
            dtype="bfloat16",
        )
        processor = LoopedQwen3_5ForConditionalGeneration.load_processor(base_model_id)
        model = LoopedQwen3_5ForConditionalGeneration.from_pretrained(cfg, device=device)
        model.eval()
        return model, processor, "stock"

    state_path = checkpoint / "loop_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing loop_state.pt in checkpoint: {checkpoint}")
    loop_state = torch.load(state_path, map_location="cpu", weights_only=True)
    cfg_dict = dict(loop_state["loop_config"])
    cfg_dict["num_loops"] = num_loops
    cfg_dict["kv_quant"] = kv_quant
    cfg_dict["kv_residual_window"] = residual_window
    cfg_dict["kv_protect_vision_prefix"] = protect_vision_prefix
    cfg_dict["kv_sink_tokens"] = sink_tokens
    cfg = LoopedQwen35Config(**cfg_dict)

    is_peft = (checkpoint / "adapter_config.json").exists()
    if is_peft:
        processor = LoopedQwen3_5ForConditionalGeneration.load_processor(cfg.base_model_id)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            cfg.base_model_id, dtype=cfg.torch_dtype
        ).to(device)
        wrapped_base = PeftModel.from_pretrained(base, str(checkpoint))
        wrapped_base.eval()
        kind = "peft"
    else:
        processor = LoopedQwen3_5ForConditionalGeneration.load_processor(str(checkpoint))
        wrapped_base = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(checkpoint), dtype=cfg.torch_dtype
        ).to(device)
        wrapped_base.eval()
        kind = "merged"

    model = LoopedQwen3_5ForConditionalGeneration(base_model=wrapped_base, loop_config=cfg)
    model.restore_loop_state(loop_state, device)
    model.eval()
    return model, processor, kind


def prepare_eval_inputs(
    processor: Any,
    system_prompt: str,
    rec: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    pil_img = None
    try:
        if rec["image_path"] is not None:
            pil_img = Image.open(rec["image_path"]).convert("RGB")
            user_content: list[dict[str, Any]] = [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": rec["prompt"]},
            ]
        else:
            user_content = [{"type": "text", "text": rec["prompt"]}]

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


def evaluate_model_on_heldout(
    model: LoopedQwen3_5ForConditionalGeneration,
    processor: Any,
    eval_records: list[dict[str, Any]],
    device: torch.device,
    num_loops: int = 2,
    kv_quant: str = "none",
    residual_window: int = 64,
    max_new_tokens: int = 192,
    compute_loop_loss_samples: int = 4,
    run_name: str = "eval",
    checkpoint_label: str = "stock",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run full held-out evaluation and return `(report_dict, failures_list)`."""
    system_prompt = build_eval_system_prompt()
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    sample_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    tool_matches = 0
    schema_valids = 0
    format_valids = 0
    truncations = 0
    total_reward_sum = 0.0

    max_attn_kv_bytes = 0
    max_deltanet_bytes = 0
    loop_loss_accum: dict[int, list[float]] = {1: [], 2: [], 3: []}

    for idx, rec in enumerate(eval_records):
        inputs = prepare_eval_inputs(processor, system_prompt, rec, device)
        prompt_len = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            gen_ids = model.generate_looped(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_loops=num_loops,
                use_cache=True,
                kv_quant=kv_quant,
                residual_window=residual_window,
                do_sample=False,
            )

        last_caches = getattr(model, "_last_per_loop_caches", None)
        if last_caches is not None:
            cstats = last_caches.kv_cache_memory_bytes()
            max_attn_kv_bytes = max(max_attn_kv_bytes, cstats["attention_kv_bytes"])
            max_deltanet_bytes = max(max_deltanet_bytes, cstats["deltanet_state_bytes"])

        comp_ids = gen_ids[:, prompt_len:]
        decoded = processor.batch_decode(comp_ids, skip_special_tokens=True)[0]
        score = score_completion(
            decoded,
            expected_tool=rec["expected_tool"],
            max_think_tokens=256,
            tokenizer=processor.tokenizer,
        )

        # Evaluate compare_loop_losses(1, 2, 3) on up to `compute_loop_loss_samples` held-out samples
        if idx < compute_loop_loss_samples:
            labels = inputs["input_ids"].clone()
            ll = model.compare_loop_losses(loop_counts=(1, 2, 3), labels=labels, **inputs)
            for k, val in ll.items():
                loop_loss_accum[k].append(val)

        is_tool_match = score["tool_match_reward"] > 0.0
        is_schema_valid = score["schema_reward"] > 0.0
        is_format_valid = score["format_reward"] > 0.0
        is_truncated = not score["has_closed_think"]

        tool_matches += int(is_tool_match)
        schema_valids += int(is_schema_valid)
        format_valids += int(is_format_valid)
        truncations += int(is_truncated)
        total_reward_sum += score["total_reward"]

        row_summary = {
            "id": rec["id"],
            "expected_tool": rec["expected_tool"],
            "parsed_tool": score["parsed_tool"],
            "image_path": rec["raw_image_path"],
            "prompt": rec["prompt"],
            "completion": decoded,
            "generated_tokens": int(comp_ids.shape[1]),
            "total_reward": score["total_reward"],
            "format_valid": is_format_valid,
            "schema_valid": is_schema_valid,
            "tool_match": is_tool_match,
            "truncated_think": is_truncated,
            "schema_errors": score["schema_errors"],
        }
        sample_results.append(row_summary)

        if not (is_tool_match and is_schema_valid and is_format_valid and not is_truncated):
            failures.append(row_summary)

        del inputs, gen_ids, comp_ids
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n = len(eval_records)
    elapsed_s = time.perf_counter() - t0
    peak_vram_gb = gb(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0

    avg_loop_losses = {
        str(k): round(sum(vals) / len(vals), 4)
        for k, vals in loop_loss_accum.items()
        if vals
    }

    report: dict[str, Any] = {
        "run_name": run_name,
        "checkpoint": checkpoint_label,
        "num_loops": num_loops,
        "kv_quant": kv_quant,
        "residual_window": residual_window,
        "reinject_weight": round(float(model.reinject_weight.detach().float().item()), 6),
        "num_samples": n,
        "elapsed_seconds": round(elapsed_s, 2),
        "metrics": {
            "tool_accuracy_pct": round(100.0 * tool_matches / n, 2),
            "schema_validity_pct": round(100.0 * schema_valids / n, 2),
            "format_compliance_pct": round(100.0 * format_valids / n, 2),
            "mean_reward": round(total_reward_sum / n, 4),
            "truncation_rate_pct": round(100.0 * truncations / n, 2),
            "num_failures": len(failures),
        },
        "loop_loss_comparison": avg_loop_losses,
        "memory": {
            "peak_vram_gb": round(peak_vram_gb, 3),
            "max_attention_kv_bytes": max_attn_kv_bytes,
            "max_deltanet_state_bytes": max_deltanet_bytes,
            "max_attention_kv_mb": round(max_attn_kv_bytes / (1024**2), 3),
            "max_deltanet_state_mb": round(max_deltanet_bytes / (1024**2), 3),
        },
        "samples": sample_results,
    }
    return report, failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Held-out evaluation suite for Looped Hybrid Qwen3.5.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE, help="Held-out eval JSONL file")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint dir (omit for stock base model)")
    parser.add_argument("--model-size", type=str, default="0.8b", choices=["0.8b", "2b"], help="Qwen3.5 size preset")
    parser.add_argument("--base-model-id", type=str, default=None, help="Explicit HF base model ID override")
    parser.add_argument("--run-name", type=str, required=True, help="Output report name under eval_results/")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for JSON reports")
    parser.add_argument("--num-loops", type=int, default=2, help="Decoder loop count")
    parser.add_argument(
        "--kv-quant",
        type=str,
        default="none",
        choices=["none", "turboquant_3bit", "turboquant_4bit"],
        help="KV-cache quantization mode",
    )
    parser.add_argument("--residual-window", type=int, default=64, help="TurboQuant recent bf16 window")
    parser.add_argument("--max-new-tokens", type=int, default=192, help="Max new tokens per sample")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of eval samples")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model_id = resolve_model_id(args.model_size, args.base_model_id)

    eval_records = load_eval_records(args.eval_file, max_samples=args.max_samples)
    model, processor, kind = load_model_for_eval(
        checkpoint=args.checkpoint,
        base_model_id=base_model_id,
        num_loops=args.num_loops,
        kv_quant=args.kv_quant,
        residual_window=args.residual_window,
        device=device,
    )

    print("=" * 74)
    print(f"HELD-OUT EVAL: {args.run_name} ({kind}, num_loops={args.num_loops}, kv_quant={args.kv_quant})")
    print("=" * 74)
    print(f"Eval file : {args.eval_file} ({len(eval_records)} samples)")

    report, failures = evaluate_model_on_heldout(
        model=model,
        processor=processor,
        eval_records=eval_records,
        device=device,
        num_loops=args.num_loops,
        kv_quant=args.kv_quant,
        residual_window=args.residual_window,
        max_new_tokens=args.max_new_tokens,
        run_name=args.run_name,
        checkpoint_label=str(args.checkpoint) if args.checkpoint else f"stock:{args.base_model_id}",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{args.run_name}.json"
    failures_path = args.output_dir / f"{args.run_name}_failures.jsonl"

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with failures_path.open("w", encoding="utf-8") as fh:
        for fail in failures:
            fh.write(json.dumps(fail, ensure_ascii=False) + "\n")

    m = report["metrics"]
    mem = report["memory"]
    print(f"Tool Accuracy      : {m['tool_accuracy_pct']:.1f}%")
    print(f"Schema Validity    : {m['schema_validity_pct']:.1f}%")
    print(f"Format Compliance  : {m['format_compliance_pct']:.1f}%")
    print(f"Mean Reward        : {m['mean_reward']:.4f}")
    print(f"Truncation Rate    : {m['truncation_rate_pct']:.1f}% ({m['num_failures']}/{report['num_samples']} failures)")
    print(f"Loop Loss (1,2,3)  : {report['loop_loss_comparison']}")
    print(
        f"Peak VRAM / KV     : {mem['peak_vram_gb']:.2f} GB | "
        f"Attn KV={mem['max_attention_kv_mb']:.2f} MB | DeltaNet={mem['max_deltanet_state_mb']:.2f} MB"
    )
    print(f"Saved report       : {report_path}")
    print(f"Saved failures     : {failures_path}")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
