"""Phase 1 Decode Speed Benchmark: Eager PyTorch Fallback vs. Fused FLA Triton Kernels vs. Fused + torch.compile."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35_mod

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from looped_qwen35.config import LoopedQwen35Config
from looped_qwen35.modeling_looped_qwen35 import LoopedQwen3_5ForConditionalGeneration


def detect_kernel_backend() -> dict[str, Any]:
    triton_spec = importlib.util.find_spec("triton")
    fla_spec = importlib.util.find_spec("fla")
    cc1d_spec = importlib.util.find_spec("causal_conv1d")

    triton_ver = None
    if triton_spec is not None:
        import triton
        triton_ver = getattr(triton, "__version__", "unknown")

    fla_ver = None
    if fla_spec is not None:
        import fla
        fla_ver = getattr(fla, "__version__", "unknown")

    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "triton_version": triton_ver,
        "fla_installed": fla_spec is not None,
        "fla_version": fla_ver,
        "deltanet_backend": "fla_triton_fused" if fla_spec is not None else "pytorch_fallback",
        "causal_conv1d_installed": cc1d_spec is not None,
        "causal_conv1d_backend": "causal_conv1d_cuda" if cc1d_spec is not None else "pytorch_fallback",
    }


def load_v2_model(export_dir: Path) -> tuple[LoopedQwen3_5ForConditionalGeneration, Any]:
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(export_dir),
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    loop_state_path = export_dir / "loop_state.pt"
    state = torch.load(str(loop_state_path), map_location="cuda", weights_only=True) if loop_state_path.exists() else {}
    cfg = LoopedQwen35Config(
        num_loops=2,
        loop_start_layer=int(state.get("loop_start_layer", 4)),
        loop_end_layer=int(state.get("loop_end_layer", 20)),
        reinject_mode=str(state.get("reinject_mode", "arloop")),
    )
    model = LoopedQwen3_5ForConditionalGeneration(base, loop_config=cfg)
    if state:
        model.restore_loop_state(state, device=torch.device("cuda"))
    model.eval()
    processor = AutoProcessor.from_pretrained(str(export_dir))
    return model, processor


def build_benchmark_inputs(processor: Any, image_path: Path) -> dict[str, torch.Tensor]:
    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "system",
            "content": "You are an expert Visual SRE assistant. Analyze the dashboard and output <think>...</think> followed by a JSON tool call.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Alert fired on checkout-api. Inspect the dashboard and select the right remediation tool."},
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt")
    return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}


def run_timed_decode(
    model: LoopedQwen3_5ForConditionalGeneration,
    inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
    runs: int,
    use_compile: bool,
    eos_disable: bool = True,
) -> dict[str, Any]:
    # Disable early EOS so every run decodes exactly `max_new_tokens` tokens
    eos_id = [] if eos_disable else None

    # Warmup pass
    _ = model.generate_looped(
        **inputs,
        max_new_tokens=8 if not use_compile else max_new_tokens,
        num_loops=2,
        do_sample=False,
        use_compile=use_compile,
        eos_token_id=eos_id,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    decode_tokens = 0
    prompt_len = int(inputs["input_ids"].shape[1])

    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_ids = model.generate_looped(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_loops=2,
            do_sample=False,
            use_compile=use_compile,
            eos_token_id=eos_id,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        decode_tokens += int(out_ids.shape[1]) - prompt_len

    total_time = sum(latencies)
    mean_latency = total_time / len(latencies)
    tok_per_s = decode_tokens / max(total_time, 1e-6)
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    compile_fallback = bool(getattr(model, "_compile_fallback_triggered", False)) if use_compile else False

    return {
        "runs": runs,
        "tokens_per_run": max_new_tokens,
        "mean_latency_s": round(mean_latency, 3),
        "tokens_per_sec": round(tok_per_s, 2),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "compile_fallback_triggered": compile_fallback,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="looped_qwen35/exports/looped_qwen35_sre_v2")
    parser.add_argument("--image", type=str, default="cot_dataset_builder/samples/sre_001_rollback.png")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=str, default="looped_qwen35/eval_results/decode_speed_benchmark.json")
    args = parser.parse_args()

    backend = detect_kernel_backend()
    print("=== DeltaNet Kernel Backend ===")
    print(json.dumps(backend, indent=2))

    export_dir = ROOT / args.model_dir
    image_path = ROOT / args.image
    model, processor = load_v2_model(export_dir)
    inputs = build_benchmark_inputs(processor, image_path)

    results: dict[str, Any] = {"backend": backend, "benchmarks": {}}

    # 1. Pure-PyTorch Fallback Baseline (temporarily bypass fla hub wrapper)
    orig_chunk = qwen35_mod.torch_chunk_gated_delta_rule
    orig_rec = qwen35_mod.torch_recurrent_gated_delta_rule
    try:
        if hasattr(orig_chunk, "__wrapped__"):
            qwen35_mod.torch_chunk_gated_delta_rule = orig_chunk.__wrapped__
        if hasattr(orig_rec, "__wrapped__"):
            qwen35_mod.torch_recurrent_gated_delta_rule = orig_rec.__wrapped__
        print("\n[1/3] Running Eager (pure-PyTorch DeltaNet fallback baseline)...")
        res_eager = run_timed_decode(model, inputs, args.max_new_tokens, args.runs, use_compile=False)
        print("  ->", res_eager)
        results["benchmarks"]["eager_pytorch_fallback"] = res_eager
    finally:
        qwen35_mod.torch_chunk_gated_delta_rule = orig_chunk
        qwen35_mod.torch_recurrent_gated_delta_rule = orig_rec

    # 2. Fused Kernels (FLA Triton DeltaNet)
    print("\n[2/3] Running Fused Kernels (FLA Triton DeltaNet)...")
    res_fused = run_timed_decode(model, inputs, args.max_new_tokens, args.runs, use_compile=False)
    res_fused["speedup_vs_eager"] = round(res_fused["tokens_per_sec"] / max(res_eager["tokens_per_sec"], 1e-6), 2)
    print("  ->", res_fused)
    results["benchmarks"]["fused_fla_triton"] = res_fused

    # 3. Fused Kernels + torch.compile
    print("\n[3/3] Running Fused Kernels + torch.compile (mode='reduce-overhead')...")
    res_compile = run_timed_decode(model, inputs, args.max_new_tokens, args.runs, use_compile=True)
    res_compile["speedup_vs_eager"] = round(res_compile["tokens_per_sec"] / max(res_eager["tokens_per_sec"], 1e-6), 2)
    print("  ->", res_compile)
    results["benchmarks"]["fused_fla_triton_compile"] = res_compile

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved speed benchmark -> {out_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
