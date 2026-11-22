"""Merge PEFT LoRA adapter into Qwen3.5 base weights and export self-contained Looped model."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import Qwen3_5ForConditionalGeneration

from config import LoopedQwen35Config
from modeling_looped_qwen35 import LoopedQwen3_5ForConditionalGeneration

DEFAULT_EXPORTS_DIR = Path(__file__).resolve().parent / "exports" / "looped_qwen35_sre_final"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export Looped Qwen3.5 model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Winning checkpoint dir (PEFT or WSP)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORTS_DIR, help="Export directory")
    parser.add_argument("--num-loops", type=int, default=None, help="Optional num_loops override")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_path = args.checkpoint / "loop_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing loop_state.pt in {args.checkpoint}")

    loop_state = torch.load(state_path, map_location="cpu", weights_only=True)
    cfg_dict = dict(loop_state["loop_config"])
    if args.num_loops is not None:
        cfg_dict["num_loops"] = args.num_loops
    loop_cfg = LoopedQwen35Config(**cfg_dict)

    is_peft = (args.checkpoint / "adapter_config.json").exists()
    print("=" * 74)
    print(f"MERGE & EXPORT: {args.checkpoint} -> {args.output_dir}")
    print("=" * 74)
    print(f"Source type : {'PEFT LoRA adapter' if is_peft else 'Already-merged checkpoint (WSP)'}")

    test_ids = torch.tensor([[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13]], dtype=torch.long, device=device)
    test_mask = torch.ones_like(test_ids)

    if is_peft:
        processor = LoopedQwen3_5ForConditionalGeneration.load_processor(loop_cfg.base_model_id)
        base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            loop_cfg.base_model_id, dtype=loop_cfg.torch_dtype
        ).to(device)
        peft_model = PeftModel.from_pretrained(base_model, str(args.checkpoint))
        peft_model.eval()

        unmerged = LoopedQwen3_5ForConditionalGeneration(base_model=peft_model, loop_config=loop_cfg)
        unmerged.restore_loop_state(loop_state, device)
        with torch.no_grad():
            unmerged._core_model.model.rope_deltas = None
            ref_logits = unmerged(input_ids=test_ids, attention_mask=test_mask, use_cache=False).logits.detach().float()

        merged_base = peft_model.merge_and_unload(safe_merge=True)
        merged_base.eval()
        merged_wrapper = LoopedQwen3_5ForConditionalGeneration(base_model=merged_base, loop_config=loop_cfg)
        merged_wrapper.restore_loop_state(loop_state, device)
        with torch.no_grad():
            merged_wrapper._core_model.model.rope_deltas = None
            merged_logits = merged_wrapper(input_ids=test_ids, attention_mask=test_mask, use_cache=False).logits.detach().float()

        max_diff = (ref_logits - merged_logits).abs().max().item()
        mean_diff = (ref_logits - merged_logits).abs().mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(ref_logits, merged_logits, dim=-1).mean().item()
        argmax_match = (ref_logits.argmax(dim=-1) == merged_logits.argmax(dim=-1)).float().mean().item() * 100.0
        print(f"Merged vs Unmerged LoRA max abs diff  : {max_diff:.6e}")
        print(f"Merged vs Unmerged LoRA mean abs diff : {mean_diff:.6e}")
        print(f"Merged vs Unmerged LoRA cosine sim    : {cos_sim:.6f} | argmax match: {argmax_match:.1f}%")
        assert cos_sim >= 0.9995 and argmax_match == 100.0 and mean_diff <= 0.1, (
            f"Merge verification failed: cos={cos_sim}, argmax={argmax_match}%, mean_diff={mean_diff}"
        )
    else:
        processor = LoopedQwen3_5ForConditionalGeneration.load_processor(str(args.checkpoint))
        merged_base = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(args.checkpoint), dtype=loop_cfg.torch_dtype
        ).to(device)
        merged_base.eval()
        merged_wrapper = LoopedQwen3_5ForConditionalGeneration(base_model=merged_base, loop_config=loop_cfg)
        merged_wrapper.restore_loop_state(loop_state, device)
        print("Source checkpoint already merged; exporting weights + loop_state.pt cleanly.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_base.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    export_state = merged_wrapper.export_loop_state()
    export_state["source_checkpoint"] = str(args.checkpoint)
    if "wsp_patch" in loop_state:
        export_state["wsp_patch"] = loop_state["wsp_patch"]
    torch.save(export_state, args.output_dir / "loop_state.pt")
    print(f"Exported merged model + processor + loop_state.pt -> {args.output_dir}")
    print("MERGE & EXPORT PASSED!")

    del merged_base, merged_wrapper
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
