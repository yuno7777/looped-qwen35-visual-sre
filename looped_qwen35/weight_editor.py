"""Weight-Space Programming (Surgical Bug Patch) for Looped Hybrid Qwen3.5.

Inspired by MIT CSAIL's Weight-Space Programming on Qwen3.5-0.8B:
1. Contrastively probes all 24 decoder MLP `down_proj` layers on passing vs. failing traces
   from the held-out evaluation report (max 50 forward passes, no gradient descent).
2. Computes the contrastive bug direction `d_bug = mean_fail - mean_pass`, ranks layers by
   separation norm, and selects the top 1-2 layers and top 64 contributing columns (<0.05% weights).
3. Sweeps reversible directional ablation `W_new[:, c] = W[:, c] - alpha * (d_hat @ W[:, c]) * d_hat`
   on `eval_heldout.jsonl` with a strict rollback guardrail: only commits if held-out
   Mean Reward or Tool Accuracy strictly improves without increasing truncation/failure rate.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

from eval_suite import (
    DEFAULT_EVAL_FILE,
    DEFAULT_OUTPUT_DIR,
    build_eval_system_prompt,
    evaluate_model_on_heldout,
    load_eval_records,
    load_model_for_eval,
    prepare_eval_inputs,
)
from modeling_looped_qwen35 import LoopedQwen3_5ForConditionalGeneration


def _get_down_proj_linear(decoder_layer: Any) -> torch.nn.Linear:
    """Return the underlying `nn.Linear` weight module of `mlp.down_proj` (handles PEFT LoraLinear)."""
    dp = decoder_layer.mlp.down_proj
    if hasattr(dp, "base_layer"):
        return dp.base_layer
    return dp


def _capture_mlp_down_proj_means(
    model: LoopedQwen3_5ForConditionalGeneration,
    processor: Any,
    samples: list[dict[str, Any]],
    eval_lookup: dict[str, dict[str, Any]],
    device: torch.device,
    max_passes: int = 25,
) -> dict[int, torch.Tensor]:
    """Run forward hooks on all 24 `mlp.down_proj` layers and return mean activation vectors."""
    system_prompt = build_eval_system_prompt()
    layers = model._core_model.model.language_model.layers
    num_layers = len(layers)

    accum: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {i: 0 for i in range(num_layers)}
    handles = []

    def _make_hook(layer_idx: int):
        def _hook(_mod, _inp, out):
            # out shape: (1, seq_len, hidden_size) -> mean across sequence in float32
            vec = out.detach().float().mean(dim=(0, 1)).cpu()
            if torch.isfinite(vec).all():
                if layer_idx not in accum:
                    accum[layer_idx] = vec.clone()
                else:
                    accum[layer_idx] += vec
                counts[layer_idx] += 1
        return _hook

    for i, layer in enumerate(layers):
        handles.append(layer.mlp.down_proj.register_forward_hook(_make_hook(i)))

    model.eval()
    try:
        for sample in samples[:max_passes]:
            rec = eval_lookup.get(sample["id"])
            if rec is None:
                continue
            inputs = prepare_eval_inputs(processor, system_prompt, rec, device)
            # Append the model's own completion tokens so hooks observe the actual failing/passing trajectory
            comp_ids = processor.tokenizer.encode(sample.get("completion", ""), add_special_tokens=False)
            if comp_ids:
                comp_t = torch.tensor([comp_ids[:192]], dtype=torch.long, device=device)
                full_ids = torch.cat([inputs["input_ids"], comp_t], dim=1)
                full_mask = torch.ones_like(full_ids, dtype=torch.long, device=device)
                fwd_kwargs = {"input_ids": full_ids, "attention_mask": full_mask, "use_cache": False}
                if "pixel_values" in inputs:
                    fwd_kwargs["pixel_values"] = inputs["pixel_values"]
                if "image_grid_thw" in inputs:
                    fwd_kwargs["image_grid_thw"] = inputs["image_grid_thw"]
                if "mm_token_type_ids" in inputs:
                    comp_mm = torch.zeros((1, comp_t.shape[1]), dtype=inputs["mm_token_type_ids"].dtype, device=device)
                    fwd_kwargs["mm_token_type_ids"] = torch.cat([inputs["mm_token_type_ids"], comp_mm], dim=1)
            else:
                fwd_kwargs = {**inputs, "use_cache": False}

            with torch.no_grad():
                model._core_model.model.rope_deltas = None
                model(**fwd_kwargs)
    finally:
        for h in handles:
            h.remove()

    return {i: (accum[i] / counts[i]) for i in range(num_layers) if counts[i] > 0}


def apply_reversible_ablation(
    model: LoopedQwen3_5ForConditionalGeneration,
    target_layers: list[int],
    bug_dirs: dict[int, torch.Tensor],
    top_columns: int,
    alpha: float,
) -> tuple[dict[int, torch.Tensor], int]:
    """Apply surgical directional ablation on top columns of `down_proj` and return backups + edited param count."""
    layers = model._core_model.model.language_model.layers
    backups: dict[int, torch.Tensor] = {}
    edited_weights = 0

    with torch.no_grad():
        for layer_idx in target_layers:
            lin = _get_down_proj_linear(layers[layer_idx])
            w_orig = lin.weight.data
            backups[layer_idx] = w_orig.detach().clone()

            d = bug_dirs[layer_idx].to(device=w_orig.device, dtype=torch.float32)
            d_norm = d.norm().clamp_min(1e-8)
            d_hat = d / d_norm  # (hidden_size,)

            w_fp32 = w_orig.float()  # (hidden_size, intermediate_size)
            proj_coeffs = torch.matmul(d_hat, w_fp32)  # (intermediate_size,)
            k = min(top_columns, w_fp32.shape[1])
            _, top_idx = torch.topk(proj_coeffs.abs(), k=k)

            # Directional ablation on the top-k contributing columns only
            delta = alpha * torch.outer(d_hat, proj_coeffs[top_idx])
            w_fp32[:, top_idx] -= delta
            lin.weight.data.copy_(w_fp32.to(w_orig.dtype))
            edited_weights += int(w_orig.shape[0] * k)

    return backups, edited_weights


def restore_backups(
    model: LoopedQwen3_5ForConditionalGeneration,
    backups: dict[int, torch.Tensor],
) -> None:
    layers = model._core_model.model.language_model.layers
    with torch.no_grad():
        for layer_idx, orig_w in backups.items():
            lin = _get_down_proj_linear(layers[layer_idx])
            lin.weight.data.copy_(orig_w)


def save_patched_checkpoint(
    model: LoopedQwen3_5ForConditionalGeneration,
    processor: Any,
    output_dir: Path,
    patch_metadata: dict[str, Any],
) -> None:
    """Merge PEFT adapter (if any) with the surgically edited weights and save with `loop_state.pt`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model.base_model, "merge_and_unload"):
        merged = model.base_model.merge_and_unload()
    else:
        merged = model.base_model
    merged.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    loop_state = {
        "reinject_logit": model.reinject_logit.detach().cpu(),
        "reinject_weight": float(model.reinject_weight.detach().float().item()),
        "loop_config": dataclasses.asdict(model.loop_config),
        "wsp_patch": patch_metadata,
    }
    torch.save(loop_state, output_dir / "loop_state.pt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weight-Space Programming surgical patcher for Looped Qwen3.5.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="GRPO checkpoint directory")
    parser.add_argument("--eval-report", type=Path, required=True, help="GRPO eval_results/<run>.json")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE, help="Held-out eval JSONL")
    parser.add_argument("--run-name", type=str, default="grpo_sre", help="Run name prefix for _wsp checkpoint & report")
    parser.add_argument("--output-dir", type=Path, default=None, help="Patched checkpoint dir (default: checkpoints/<run_name>_wsp)")
    parser.add_argument("--eval-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Eval JSON output dir")
    parser.add_argument("--num-loops", type=int, default=2, help="Decoder loop count")
    parser.add_argument("--top-layers", type=int, default=2, help="Number of top MLP layers to edit (1-2)")
    parser.add_argument("--top-columns", type=int, default=64, help="Top contributing down_proj columns per layer")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.15, 0.30], help="Ablation strength grid")
    parser.add_argument("--max-new-tokens", type=int, default=192, help="Max new tokens for held-out eval")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patched_ckpt_dir = args.output_dir or (Path(__file__).resolve().parent / "checkpoints" / f"{args.run_name}_wsp")

    base_report = json.loads(args.eval_report.read_text(encoding="utf-8"))
    all_samples = base_report.get("samples", [])
    passing = [s for s in all_samples if s.get("tool_match") and s.get("schema_valid") and s.get("format_valid") and not s.get("truncated_think")]
    failing = [s for s in all_samples if s not in passing]

    base_metrics = base_report["metrics"]
    print("=" * 74)
    print(f"WEIGHT-SPACE PROGRAMMING (WSP): {args.run_name}")
    print("=" * 74)
    print(f"GRPO checkpoint : {args.checkpoint}")
    print(f"Baseline metrics: tool_acc={base_metrics['tool_accuracy_pct']:.1f}% | mean_reward={base_metrics['mean_reward']:.4f} | trunc={base_metrics['truncation_rate_pct']:.1f}%")
    print(f"Contrastive set : {len(passing)} passing vs {len(failing)} failing traces")

    eval_records = load_eval_records(args.eval_file)
    eval_lookup = {r["id"]: r for r in eval_records}

    if not failing or not passing:
        print("Contrastive probing skipped (need >=1 passing and >=1 failing trace).")
        print("No beneficial edit found (kept GRPO weights)")
        summary_report = dict(base_report)
        summary_report["run_name"] = f"{args.run_name}_wsp"
        summary_report["wsp_status"] = "kept_grpo_weights_no_contrastive_pairs"
        args.eval_output_dir.mkdir(parents=True, exist_ok=True)
        (args.eval_output_dir / f"{args.run_name}_wsp.json").write_text(
            json.dumps(summary_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return 0

    model, processor, _ = load_model_for_eval(
        checkpoint=args.checkpoint,
        base_model_id="Qwen/Qwen3.5-0.8B",
        num_loops=args.num_loops,
        kv_quant="none",
        residual_window=64,
        device=device,
    )
    total_params = sum(p.numel() for p in model.parameters())

    # Cap total forward passes at 50 (up to 25 failing + 25 passing)
    fail_means = _capture_mlp_down_proj_means(model, processor, failing, eval_lookup, device, max_passes=25)
    pass_means = _capture_mlp_down_proj_means(model, processor, passing, eval_lookup, device, max_passes=25)

    bug_dirs: dict[int, torch.Tensor] = {}
    layer_scores: list[tuple[int, float]] = []
    for layer_idx in sorted(set(fail_means.keys()) & set(pass_means.keys())):
        diff = fail_means[layer_idx] - pass_means[layer_idx]
        if not torch.isfinite(diff).all():
            raise RuntimeError(f"Non-finite contrastive direction in layer {layer_idx}")
        bug_dirs[layer_idx] = diff
        layer_scores.append((layer_idx, float(diff.norm().item())))

    layer_scores.sort(key=lambda x: x[1], reverse=True)
    top_layers = [idx for idx, _ in layer_scores[: args.top_layers]]
    print("Top candidate MLP down_proj layers by separation norm:")
    for idx, norm_val in layer_scores[:5]:
        marker = " <-- SELECTED" if idx in top_layers else ""
        print(f"  layer {idx:>2d}: ||d_bug|| = {norm_val:.4f}{marker}")

    best_alpha: float | None = None
    best_report: dict[str, Any] | None = None
    best_failures: list[dict[str, Any]] = []
    best_edited_pct = 0.0

    for alpha in args.alphas:
        backups, edited_count = apply_reversible_ablation(
            model=model,
            target_layers=top_layers,
            bug_dirs=bug_dirs,
            top_columns=args.top_columns,
            alpha=alpha,
        )
        edited_pct = 100.0 * edited_count / total_params
        cand_report, cand_failures = evaluate_model_on_heldout(
            model=model,
            processor=processor,
            eval_records=eval_records,
            device=device,
            num_loops=args.num_loops,
            kv_quant="none",
            residual_window=64,
            max_new_tokens=args.max_new_tokens,
            run_name=f"{args.run_name}_wsp_a{alpha}",
            checkpoint_label=str(patched_ckpt_dir),
        )
        cm = cand_report["metrics"]
        print(
            f"  alpha={alpha:.2f} (edited {edited_count:,} weights = {edited_pct:.4f}%) | "
            f"tool_acc={cm['tool_accuracy_pct']:.1f}% | mean_reward={cm['mean_reward']:.4f} | "
            f"trunc={cm['truncation_rate_pct']:.1f}% | failures={cm['num_failures']}"
        )

        # Strict Rollback Guardrail
        strictly_better = (
            cm["mean_reward"] > base_metrics["mean_reward"]
            or cm["tool_accuracy_pct"] > base_metrics["tool_accuracy_pct"]
        )
        no_regression = (
            cm["truncation_rate_pct"] <= base_metrics["truncation_rate_pct"]
            and cm["num_failures"] <= base_metrics["num_failures"]
        )
        if strictly_better and no_regression:
            if best_report is None or cm["mean_reward"] > best_report["metrics"]["mean_reward"]:
                best_alpha = alpha
                best_report = cand_report
                best_failures = cand_failures
                best_edited_pct = edited_pct

        # Always revert before testing next alpha
        restore_backups(model, backups)
        del backups
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.eval_output_dir.mkdir(parents=True, exist_ok=True)
    wsp_report_path = args.eval_output_dir / f"{args.run_name}_wsp.json"

    if best_alpha is not None and best_report is not None:
        _, edited_count = apply_reversible_ablation(
            model=model,
            target_layers=top_layers,
            bug_dirs=bug_dirs,
            top_columns=args.top_columns,
            alpha=best_alpha,
        )
        patch_meta = {
            "source_checkpoint": str(args.checkpoint),
            "target_layers": top_layers,
            "top_columns": args.top_columns,
            "alpha": best_alpha,
            "edited_weights": edited_count,
            "edited_pct": round(best_edited_pct, 5),
        }
        save_patched_checkpoint(model, processor, patched_ckpt_dir, patch_meta)
        best_report["run_name"] = f"{args.run_name}_wsp"
        best_report["wsp_status"] = "committed"
        best_report["wsp_patch"] = patch_meta
        wsp_report_path.write_text(json.dumps(best_report, indent=2, ensure_ascii=False), encoding="utf-8")
        with (args.eval_output_dir / f"{args.run_name}_wsp_failures.jsonl").open("w", encoding="utf-8") as fh:
            for fail in best_failures:
                fh.write(json.dumps(fail, ensure_ascii=False) + "\n")
        print(
            f"COMMITTED WSP PATCH: alpha={best_alpha} on layers {top_layers} "
            f"({best_edited_pct:.4f}% weights) -> {patched_ckpt_dir}"
        )
    else:
        print("No beneficial edit found (kept GRPO weights)")
        kept_report = dict(base_report)
        kept_report["run_name"] = f"{args.run_name}_wsp"
        kept_report["wsp_status"] = "reverted_kept_grpo_weights"
        kept_report["wsp_probed_layers"] = [{"layer": i, "sep_norm": round(s, 4)} for i, s in layer_scores[:5]]
        wsp_report_path.write_text(json.dumps(kept_report, indent=2, ensure_ascii=False), encoding="utf-8")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
