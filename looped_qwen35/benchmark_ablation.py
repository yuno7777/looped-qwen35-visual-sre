"""Aggregate held-out ablation reports and run final 2048-token TurboQuant KV benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

from infer_looped import load_looped_checkpoint
from verify_turboquant import mb, run_prefill_and_decode

DEFAULT_EVAL_DIR = Path(__file__).resolve().parent / "eval_results"
DEFAULT_MERGED_MODEL = Path(__file__).resolve().parent / "exports" / "looped_qwen35_sre_final"
DEFAULT_OUT_MD = Path(__file__).resolve().parent / "BENCHMARK_RESULTS.md"

ORDERED_RUNS: list[tuple[str, str]] = [
    ("stock_sre_loop1", "Stock Qwen3.5-0.8B (1-Loop)"),
    ("stock_sre_loop2", "Stock Qwen3.5-0.8B (2-Loop)"),
    ("sft_sre", "Looped SFT (2-Loop LoRA)"),
    ("grpo_sre", "Looped GRPO (2-Loop RLVR)"),
    ("grpo_sre_wsp", "Looped GRPO + WSP"),
    ("final_merged_tq3", "Final Merged + TurboQuant 3-bit"),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ablation table and run final TurboQuant benchmark.")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR, help="Directory with eval_results/*.json")
    parser.add_argument("--merged-model", type=Path, default=DEFAULT_MERGED_MODEL, help="Final merged export dir")
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD, help="Output BENCHMARK_RESULTS.md path")
    parser.add_argument("--prefill-tokens", type=int, default=2048, help="Long-context prefill length")
    parser.add_argument("--decode-tokens", type=int, default=16, help="Cached decode steps")
    parser.add_argument("--residual-window", type=int, default=64, help="TurboQuant bf16 residual window")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load ordered held-out eval reports
    eval_rows: list[tuple[str, dict[str, Any]]] = []
    for run_key, display_name in ORDERED_RUNS:
        p = args.eval_dir / f"{run_key}.json"
        if p.exists():
            eval_rows.append((display_name, json.loads(p.read_text(encoding="utf-8"))))

    # 2. Run 2048-token prefill + 16-token decode TurboQuant benchmark on final merged model
    kv_bench_rows: list[dict[str, Any]] = []
    if args.merged_model.exists() and device.type == "cuda":
        model, _ = load_looped_checkpoint(args.merged_model, device=device, num_loops_override=2)
        torch.manual_seed(42)
        long_ids = torch.randint(100, 5000, (1, args.prefill_tokens), dtype=torch.long, device=device)

        baseline_attn_bytes: int | None = None
        for mode in ("none", "turboquant_3bit", "turboquant_4bit"):
            caches, peak_gb = run_prefill_and_decode(
                model,
                long_ids,
                decode_steps=args.decode_tokens,
                kv_quant=mode,
                residual_window=args.residual_window,
            )
            stats = caches.kv_cache_memory_bytes()
            if baseline_attn_bytes is None:
                baseline_attn_bytes = stats["attention_kv_bytes"]
            ratio = baseline_attn_bytes / max(1, stats["attention_kv_bytes"])
            kv_bench_rows.append(
                {
                    "mode": mode,
                    "seq_len": caches.get_seq_length(),
                    "attn_kv_mb": mb(stats["attention_kv_bytes"]),
                    "deltanet_mb": mb(stats["deltanet_state_bytes"]),
                    "total_cache_mb": mb(stats["total_bytes"]),
                    "attn_compression": ratio,
                    "peak_vram_gb": peak_gb,
                }
            )
            del caches
            gc.collect()
            torch.cuda.empty_cache()

        del model, long_ids
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Render publication-ready Markdown
    lines: list[str] = [
        "# Looped Hybrid Qwen3.5 (Visual SRE Incident Triage) — Benchmark & Ablation Results",
        "",
        "## 1. Held-Out Visual SRE Evaluation Ablation (`eval_heldout.jsonl`)",
        "",
        "| Stage / Configuration | Loops | KV Quant | Tool Acc (%) | Schema Valid (%) | Format (%) | Mean Reward | Trunc Rate (%) | Loop Losses (L1 / L2 / L3) | Peak VRAM (GB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for label, rep in eval_rows:
        m = rep["metrics"]
        mem = rep["memory"]
        ll = rep.get("loop_loss_comparison", {})
        ll_str = f"{ll.get('1', '-')} / {ll.get('2', '-')} / {ll.get('3', '-')}"
        lines.append(
            f"| {label} | {rep['num_loops']} | `{rep['kv_quant']}` | "
            f"{m['tool_accuracy_pct']:.1f}% | {m['schema_validity_pct']:.1f}% | "
            f"{m['format_compliance_pct']:.1f}% | {m['mean_reward']:.4f} | "
            f"{m['truncation_rate_pct']:.1f}% | {ll_str} | {mem['peak_vram_gb']:.2f} |"
        )

    lines.extend(
        [
            "",
            f"## 2. Long-Context TurboQuant KV-Cache Benchmark ({args.prefill_tokens} Prefill + {args.decode_tokens} Decode, `num_loops=2`, `residual_window={args.residual_window}`)",
            "",
            "| KV Quant Mode | Cached Tokens | 6x Attention KV (MB) | 18x DeltaNet States (MB) | Total Cache (MB) | Attention KV Compression | Peak VRAM (GB) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for row in kv_bench_rows:
        lines.append(
            f"| `{row['mode']}` | {row['seq_len']} | {row['attn_kv_mb']:.2f} MB | "
            f"{row['deltanet_mb']:.2f} MB | {row['total_cache_mb']:.2f} MB | "
            f"**{row['attn_compression']:.2f}x** | {row['peak_vram_gb']:.2f} GB |"
        )

    md_text = "\n".join(lines) + "\n"
    args.output_md.write_text(md_text, encoding="utf-8")
    print(md_text)
    print(f"Saved Markdown report -> {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
