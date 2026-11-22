"""3-Tier Hard Stress-Test & Generalization Benchmark Suite (Stock vs. v1 vs. v2 + Vision-Exempt TQ + Loop Sweep).

Tier 1: Hard OOD Visual SRE Stress Test (50 unseen dashboards: Visual Shift, Pixel-Only, Conflict Traps)
        - Stock (1-Loop bf16) vs. v1 (2-Loop bf16, naive TQ4/TQ3, Vision-Exempt TQ4/TQ3) vs. v2 (ETD+ARLoop bf16, Vision-Exempt TQ4/TQ3)
        - Loop-Depth Sweep (num_loops=1, 2, 3) with latency (s/prompt) and throughput (tokens/sec).
Tier 2: Public Chart Reasoning Benchmark (HuggingFaceM4/ChartQA, 50 test samples, seed=1337).
Tier 3: Public Tool-Calling Benchmark (BFCL v3: 20 simple + 15 multiple + 15 irrelevance, seed=1337).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

from eval_suite import build_eval_system_prompt, load_model_for_eval, prepare_eval_inputs
from rewards import score_completion

ROOT = Path(__file__).resolve().parent.parent
OOD_PROMPTS_FILE = ROOT / "cot_dataset_builder" / "ood_sre_prompts.jsonl"
OOD_BASE_DIR = ROOT / "cot_dataset_builder"
V1_MERGED_DIR = ROOT / "looped_qwen35" / "exports" / "looped_qwen35_sre_final"
V2_MERGED_DIR = ROOT / "looped_qwen35" / "exports" / "looped_qwen35_sre_v2"
EVAL_RESULTS_DIR = ROOT / "looped_qwen35" / "eval_results"
OUT_MD = ROOT / "HARD_BENCHMARK_RESULTS.md"

THINK_CLOSE_RE = re.compile(r"^(.*?)</think>\s*(.*)$", re.DOTALL | re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def free_gpu(model: Any = None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def split_think(text: str) -> tuple[bool, str, str]:
    cleaned = text.strip()
    if cleaned.lower().startswith("<think>"):
        cleaned = cleaned[len("<think>") :].lstrip("\n")
    m = THINK_CLOSE_RE.match(cleaned)
    if m:
        return True, m.group(1).strip(), m.group(2).strip()
    return False, cleaned, ""


def load_ood_sre_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with OOD_PROMPTS_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            img_path = OOD_BASE_DIR / obj["image_path"]
            if not img_path.exists():
                raise FileNotFoundError(f"Missing OOD image: {img_path}")
            records.append(
                {
                    "id": obj["id"],
                    "subset": obj["subset"],
                    "layout": obj.get("layout", ""),
                    "theme": obj.get("theme", ""),
                    "prompt": obj["prompt"],
                    "expected_tool": obj["expected_tool"],
                    "expected_args": obj.get("expected_args"),
                    "image_path": img_path,
                    "raw_image_path": obj["image_path"],
                }
            )
    return records


def run_tier1_config(
    config_tag: str,
    display_name: str,
    checkpoint: Path | None,
    num_loops: int,
    kv_quant: str,
    residual_window: int,
    records: list[dict[str, Any]],
    device: torch.device,
    protect_vision_prefix: bool = False,
    sink_tokens: int = 4,
    max_new_tokens: int = 192,
    resume: bool = False,
) -> dict[str, Any]:
    out_path = EVAL_RESULTS_DIR / f"hard_bench_tier1_{config_tag}.json"
    if resume and out_path.exists():
        print(f"[Tier 1] Resuming cached result: {out_path.name}")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        data["display_name"] = display_name
        return data

    print(
        f"\n[Tier 1] Running {display_name} "
        f"(L={num_loops}, kv={kv_quant}, vis_exempt={protect_vision_prefix}, n={len(records)})..."
    )
    model, processor, _ = load_model_for_eval(
        checkpoint=checkpoint,
        base_model_id="Qwen/Qwen3.5-0.8B",
        num_loops=num_loops,
        kv_quant=kv_quant,
        residual_window=residual_window,
        device=device,
        protect_vision_prefix=protect_vision_prefix,
        sink_tokens=sink_tokens,
    )
    sys_prompt = build_eval_system_prompt()
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    samples_out: list[dict[str, Any]] = []
    subset_hits: dict[str, list[int]] = {"visual_shift": [], "pixel_only": [], "conflict_trap": []}
    tool_hits = 0
    schema_hits = 0
    trunc_hits = 0
    total_gen_tokens = 0
    last_cache_breakdown: dict[str, int] = {}

    for idx, rec in enumerate(records, start=1):
        inputs = prepare_eval_inputs(processor, sys_prompt, rec, device)
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            gen_ids = model.generate_looped(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_loops=num_loops,
                use_cache=True,
                kv_quant=kv_quant,
                residual_window=residual_window,
                protect_vision_prefix=protect_vision_prefix,
                sink_tokens=sink_tokens,
                do_sample=False,
            )
        if getattr(model, "_last_per_loop_caches", None) is not None:
            last_cache_breakdown = model._last_per_loop_caches.kv_cache_memory_bytes()

        comp_ids = gen_ids[:, prompt_len:]
        n_toks = int(comp_ids.shape[1])
        total_gen_tokens += n_toks
        decoded = processor.batch_decode(comp_ids, skip_special_tokens=True)[0]
        score = score_completion(
            decoded,
            expected_tool=rec["expected_tool"],
            max_think_tokens=256,
            tokenizer=processor.tokenizer,
        )
        t_ok = score["tool_match_reward"] > 0.0
        s_ok = score["schema_reward"] > 0.0
        trunc = not score["has_closed_think"]

        tool_hits += int(t_ok)
        schema_hits += int(s_ok)
        trunc_hits += int(trunc)
        subset_hits[rec["subset"]].append(int(t_ok))

        samples_out.append(
            {
                "id": rec["id"],
                "subset": rec["subset"],
                "expected_tool": rec["expected_tool"],
                "parsed_tool": score["parsed_tool"],
                "tool_match": t_ok,
                "schema_valid": s_ok,
                "truncated_think": trunc,
                "total_reward": score["total_reward"],
                "generated_tokens": n_toks,
                "completion": decoded,
            }
        )
        del inputs, gen_ids, comp_ids
        if idx % 10 == 0:
            print(f"  [{idx}/{len(records)}] running tool_acc={100.0 * tool_hits / idx:.1f}%")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    n = len(records)
    elapsed_s = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    result = {
        "tier": 1,
        "config_tag": config_tag,
        "display_name": display_name,
        "num_loops": num_loops,
        "kv_quant": kv_quant,
        "protect_vision_prefix": protect_vision_prefix,
        "residual_window": residual_window,
        "num_samples": n,
        "elapsed_seconds": round(elapsed_s, 1),
        "latency_s_per_prompt": round(elapsed_s / max(1, n), 2),
        "throughput_tok_per_s": round(total_gen_tokens / max(1e-6, elapsed_s), 2),
        "total_generated_tokens": total_gen_tokens,
        "peak_vram_gb": round(peak_gb, 2),
        "cache_breakdown_bytes": last_cache_breakdown,
        "metrics": {
            "tool_accuracy_pct": round(100.0 * tool_hits / n, 1),
            "schema_validity_pct": round(100.0 * schema_hits / n, 1),
            "truncation_rate_pct": round(100.0 * trunc_hits / n, 1),
            "visual_shift_acc_pct": round(100.0 * sum(subset_hits["visual_shift"]) / max(1, len(subset_hits["visual_shift"])), 1),
            "pixel_only_acc_pct": round(100.0 * sum(subset_hits["pixel_only"]) / max(1, len(subset_hits["pixel_only"])), 1),
            "conflict_trap_acc_pct": round(100.0 * sum(subset_hits["conflict_trap"]) / max(1, len(subset_hits["conflict_trap"])), 1),
        },
        "samples": samples_out,
    }
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"  DONE {display_name}: {result['metrics']} | "
        f"{result['latency_s_per_prompt']}s/prompt | {result['throughput_tok_per_s']} tok/s | peak {peak_gb:.2f} GB"
    )
    free_gpu(model)
    return result


# =========================================================================
# TIER 2: CHARTQA (50 SAMPLES)
# =========================================================================
def _to_float_or_none(s: str) -> float | None:
    cleaned = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        m = NUM_RE.search(cleaned)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
        return None


def chartqa_relaxed_match(pred: str, targets: list[str]) -> bool:
    pred_clean = pred.strip().strip(".").strip()
    if not pred_clean:
        return False
    pred_norm = " ".join(pred_clean.lower().split())

    for gt in targets:
        gt_clean = str(gt).strip().strip(".")
        gt_norm = " ".join(gt_clean.lower().split())
        if pred_norm == gt_norm or gt_norm in pred_norm:
            if len(pred_norm) <= len(gt_norm) + 20:
                return True
        gt_num = _to_float_or_none(gt_clean)
        pred_num = _to_float_or_none(pred_clean)
        if gt_num is not None and pred_num is not None:
            if abs(gt_num) < 1e-9:
                if abs(pred_num) < 1e-6:
                    return True
            elif abs(pred_num - gt_num) / abs(gt_num) <= 0.05:
                return True
    return False


def load_chartqa_slice(n: int = 50, seed: int = 1337) -> list[dict[str, Any]]:
    ds = load_dataset("HuggingFaceM4/ChartQA", split="test")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(ds)), k=min(n, len(ds))))
    items: list[dict[str, Any]] = []
    for idx in indices:
        row = ds[idx]
        lbl = row["label"]
        targets = lbl if isinstance(lbl, list) else [str(lbl)]
        items.append({"id": f"chartqa_{idx}", "image": row["image"], "query": row["query"], "targets": targets})
    return items


def run_tier2_config(
    config_tag: str,
    display_name: str,
    checkpoint: Path | None,
    num_loops: int,
    items: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int = 128,
    resume: bool = False,
) -> dict[str, Any]:
    out_path = EVAL_RESULTS_DIR / f"hard_bench_tier2_{config_tag}.json"
    if resume and out_path.exists():
        print(f"[Tier 2] Resuming cached result: {out_path.name}")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        data["display_name"] = display_name
        return data

    print(f"\n[Tier 2 - ChartQA] Running {display_name} ({len(items)} samples)...")
    model, processor, _ = load_model_for_eval(
        checkpoint=checkpoint,
        base_model_id="Qwen/Qwen3.5-0.8B",
        num_loops=num_loops,
        kv_quant="none",
        residual_window=64,
        device=device,
    )
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sys_prompt = "You are a helpful visual chart analysis assistant. Answer the user's chart question accurately and concisely with only the final value or short phrase."
    correct = 0
    truncated = 0
    samples_out: list[dict[str, Any]] = []

    for idx, item in enumerate(items, start=1):
        img = item["image"].convert("RGB")
        if max(img.size) > 768:
            img.thumbnail((768, 768), Image.Resampling.LANCZOS)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": item["query"]}]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dict(inputs).items()}
        prompt_len = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            gen_ids = model.generate_looped(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_loops=num_loops,
                use_cache=True,
                kv_quant="none",
                do_sample=False,
            )
        comp_ids = gen_ids[:, prompt_len:]
        decoded = processor.batch_decode(comp_ids, skip_special_tokens=True)[0].strip()

        has_close, _, after_think = split_think(decoded)
        if "</think>" in decoded.lower():
            final_ans = after_think
            is_trunc = not has_close
        elif "<think>" in decoded.lower():
            final_ans = ""
            is_trunc = True
        else:
            final_ans = decoded.splitlines()[-1].strip() if decoded else ""
            is_trunc = False

        ok = chartqa_relaxed_match(final_ans, item["targets"])
        correct += int(ok)
        truncated += int(is_trunc)
        samples_out.append(
            {
                "id": item["id"],
                "query": item["query"],
                "targets": item["targets"],
                "prediction": final_ans,
                "raw_completion": decoded,
                "correct": ok,
                "truncated": is_trunc,
            }
        )
        del inputs, gen_ids, comp_ids
        if idx % 10 == 0:
            print(f"  [{idx}/{len(items)}] running ChartQA relaxed_acc={100.0 * correct / idx:.1f}%")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    n = len(items)
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    result = {
        "tier": 2,
        "config_tag": config_tag,
        "display_name": display_name,
        "num_loops": num_loops,
        "num_samples": n,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "peak_vram_gb": round(peak_gb, 2),
        "metrics": {
            "relaxed_accuracy_pct": round(100.0 * correct / n, 1),
            "truncation_rate_pct": round(100.0 * truncated / n, 1),
        },
        "samples": samples_out,
    }
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  DONE {display_name}: {result['metrics']} (peak {peak_gb:.2f} GB)")
    free_gpu(model)
    return result


# =========================================================================
# TIER 3: BFCL v3 (50 SAMPLES)
# =========================================================================
def load_bfcl_slice(seed: int = 1337) -> list[dict[str, Any]]:
    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []

    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]

    for split_name, fname, ans_fname, k in [
        ("simple", "BFCL_v3_simple.json", "possible_answer/BFCL_v3_simple.json", 20),
        ("multiple", "BFCL_v3_multiple.json", "possible_answer/BFCL_v3_multiple.json", 15),
        ("irrelevance", "BFCL_v3_irrelevance.json", None, 15),
    ]:
        q_path = hf_hub_download(repo, fname, repo_type="dataset")
        q_rows = _load_jsonl(q_path)
        ans_map: dict[str, Any] = {}
        if ans_fname is not None:
            a_path = hf_hub_download(repo, ans_fname, repo_type="dataset")
            for a in _load_jsonl(a_path):
                ans_map[a["id"]] = a["ground_truth"]

        chosen = [q_rows[i] for i in sorted(rng.sample(range(len(q_rows)), k=min(k, len(q_rows))))]
        for row in chosen:
            q_text = row["question"][0][0]["content"]
            items.append(
                {
                    "id": row["id"],
                    "category": split_name,
                    "question": q_text,
                    "functions": row["function"],
                    "ground_truth": ans_map.get(row["id"]),
                }
            )
    return items


def _arg_value_matches(pred_val: Any, allowed_vals: list[Any]) -> bool:
    for av in allowed_vals:
        if pred_val == av:
            return True
        if isinstance(pred_val, (int, float)) and isinstance(av, (int, float)) and not isinstance(pred_val, bool):
            if abs(float(pred_val) - float(av)) <= 1e-4:
                return True
        if isinstance(pred_val, str) and isinstance(av, str):
            if pred_val.strip().lower() == av.strip().lower():
                return True
    return False


def score_bfcl_item(decoded: str, item: dict[str, Any]) -> dict[str, Any]:
    has_close, _, after_think = split_think(decoded)
    if "</think>" in decoded.lower():
        ans = after_think
        truncated = not has_close
    elif "<think>" in decoded.lower():
        ans = ""
        truncated = True
    else:
        ans = decoded.strip()
        truncated = False

    parsed: dict[str, Any] | None = None
    if ans:
        cand = ans.strip()
        fm = CODE_FENCE_RE.match(cand)
        if fm:
            cand = fm.group(1).strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                parsed = obj
        except json.JSONDecodeError:
            pass

    func_names = {f["name"] for f in item["functions"]}

    if item["category"] == "irrelevance":
        if parsed is not None:
            tool = parsed.get("tool") or parsed.get("name")
            correct = (tool is None) or (str(tool).strip().lower() in ("none", "null", "no_tool", "")) or (tool not in func_names)
        else:
            correct = (not truncated) and (not any(f"{fn}(" in ans for fn in func_names))
        return {"correct": bool(correct), "truncated": truncated, "parsed": parsed}

    if parsed is None:
        return {"correct": False, "truncated": truncated, "parsed": None}

    pred_tool = parsed.get("tool") or parsed.get("name")
    pred_args = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
    gt_list = item["ground_truth"] or []
    if not gt_list:
        return {"correct": False, "truncated": truncated, "parsed": parsed}

    matched = False
    for gt_entry in gt_list:
        if not isinstance(gt_entry, dict) or len(gt_entry) != 1:
            continue
        exp_func, exp_args_map = next(iter(gt_entry.items()))
        if pred_tool != exp_func:
            continue
        all_args_ok = True
        for arg_name, allowed in exp_args_map.items():
            is_optional = ("" in allowed) or (None in allowed)
            if arg_name not in pred_args:
                if not is_optional:
                    all_args_ok = False
                    break
            else:
                if not _arg_value_matches(pred_args[arg_name], allowed):
                    all_args_ok = False
                    break
        if all_args_ok:
            matched = True
            break

    return {"correct": matched, "truncated": truncated, "parsed": parsed}


def run_tier3_config(
    config_tag: str,
    display_name: str,
    checkpoint: Path | None,
    num_loops: int,
    items: list[dict[str, Any]],
    device: torch.device,
    max_new_tokens: int = 192,
    resume: bool = False,
) -> dict[str, Any]:
    out_path = EVAL_RESULTS_DIR / f"hard_bench_tier3_{config_tag}.json"
    if resume and out_path.exists():
        print(f"[Tier 3] Resuming cached result: {out_path.name}")
        data = json.loads(out_path.read_text(encoding="utf-8"))
        data["display_name"] = display_name
        return data

    print(f"\n[Tier 3 - BFCL v3] Running {display_name} ({len(items)} samples)...")
    model, processor, _ = load_model_for_eval(
        checkpoint=checkpoint,
        base_model_id="Qwen/Qwen3.5-0.8B",
        num_loops=num_loops,
        kv_quant="none",
        residual_window=64,
        device=device,
    )
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    cat_hits: dict[str, list[int]] = {"simple": [], "multiple": [], "irrelevance": []}
    trunc_hits = 0
    samples_out: list[dict[str, Any]] = []

    for idx, item in enumerate(items, start=1):
        tools_json = json.dumps(item["functions"], ensure_ascii=False)
        sys_prompt = (
            "You are a precise function-calling assistant. Given the available functions below, "
            "respond with ONLY a valid JSON object of the form:\n"
            '{"tool": "<function_name>", "arguments": { ... }}\n'
            "If NONE of the available functions can fulfill the user's request, respond with:\n"
            '{"tool": "none", "arguments": {}}\n\n'
            f"AVAILABLE FUNCTIONS:\n{tools_json}"
        )
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": [{"type": "text", "text": item["question"]}]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dict(inputs).items()}
        prompt_len = int(inputs["input_ids"].shape[1])

        with torch.no_grad():
            gen_ids = model.generate_looped(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_loops=num_loops,
                use_cache=True,
                kv_quant="none",
                do_sample=False,
            )
        comp_ids = gen_ids[:, prompt_len:]
        decoded = processor.batch_decode(comp_ids, skip_special_tokens=True)[0]
        res = score_bfcl_item(decoded, item)

        cat_hits[item["category"]].append(int(res["correct"]))
        trunc_hits += int(res["truncated"])
        samples_out.append(
            {
                "id": item["id"],
                "category": item["category"],
                "correct": res["correct"],
                "truncated": res["truncated"],
                "parsed": res["parsed"],
                "completion": decoded,
            }
        )
        del inputs, gen_ids, comp_ids
        if idx % 10 == 0:
            total_ok = sum(sum(v) for v in cat_hits.values())
            print(f"  [{idx}/{len(items)}] running BFCL overall_acc={100.0 * total_ok / idx:.1f}%")
            if device.type == "cuda":
                torch.cuda.empty_cache()

    n = len(items)
    total_ok = sum(sum(v) for v in cat_hits.values())
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    result = {
        "tier": 3,
        "config_tag": config_tag,
        "display_name": display_name,
        "num_loops": num_loops,
        "num_samples": n,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "peak_vram_gb": round(peak_gb, 2),
        "metrics": {
            "overall_accuracy_pct": round(100.0 * total_ok / n, 1),
            "simple_acc_pct": round(100.0 * sum(cat_hits["simple"]) / max(1, len(cat_hits["simple"])), 1),
            "multiple_acc_pct": round(100.0 * sum(cat_hits["multiple"]) / max(1, len(cat_hits["multiple"])), 1),
            "irrelevance_acc_pct": round(100.0 * sum(cat_hits["irrelevance"]) / max(1, len(cat_hits["irrelevance"])), 1),
            "truncation_rate_pct": round(100.0 * trunc_hits / n, 1),
        },
        "samples": samples_out,
    }
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  DONE {display_name}: {result['metrics']} (peak {peak_gb:.2f} GB)")
    free_gpu(model)
    return result


def write_markdown_report(
    t1_main: list[dict[str, Any]],
    t1_loop_sweep: list[dict[str, Any]],
    t2_results: list[dict[str, Any]],
    t3_results: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# Hard Stress-Test & Generalization Benchmark Results (Stock vs. v1 vs. v2)",
        "",
        "Deterministic evaluation (`seed=1337`, greedy `do_sample=False`, `bfloat16`) on a single **NVIDIA RTX 4060 Laptop GPU (8.00 GB VRAM)**.",
        "",
        "## Tier 1: Hard OOD Visual SRE Stress Test (50 Unseen Dashboards) + Vision-Prefix-Exempt TurboQuant",
        "",
        "| Configuration | Loops | KV Quant | Vis-Exempt | Overall Acc | Schema Valid | Trunc Rate | Visual Shift (20) | Pixel-Only (15) | Conflict Trap (15) | Peak VRAM |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in t1_main:
        m = r["metrics"]
        ve = "Yes" if r.get("protect_vision_prefix") else ("No" if r["kv_quant"] != "none" else "N/A")
        lines.append(
            f"| {r['display_name']} | {r['num_loops']} | `{r['kv_quant']}` | {ve} | "
            f"**{m['tool_accuracy_pct']:.1f}%** | {m['schema_validity_pct']:.1f}% | {m['truncation_rate_pct']:.1f}% | "
            f"{m['visual_shift_acc_pct']:.1f}% | {m['pixel_only_acc_pct']:.1f}% | {m['conflict_trap_acc_pct']:.1f}% | "
            f"{r['peak_vram_gb']:.2f} GB |"
        )

    if t1_loop_sweep:
        lines.extend(
            [
                "",
                "## Loop-Depth Sweep on Tier 1 OOD (`v1 Merged`, `bf16`, `num_loops = 1, 2, 3`)",
                "",
                "| Model & Loop Depth | Overall Acc | Visual Shift (20) | Pixel-Only (15) | Conflict Trap (15) | Latency (s/prompt) | Throughput (tok/s) | Peak VRAM |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for r in t1_loop_sweep:
            m = r["metrics"]
            lat = r.get("latency_s_per_prompt", round(r.get("elapsed_seconds", 0.0) / max(1, r["num_samples"]), 2))
            tps = r.get("throughput_tok_per_s", "-")
            lines.append(
                f"| {r['display_name']} | **{m['tool_accuracy_pct']:.1f}%** | "
                f"{m['visual_shift_acc_pct']:.1f}% | {m['pixel_only_acc_pct']:.1f}% | {m['conflict_trap_acc_pct']:.1f}% | "
                f"{lat} s | {tps} | {r['peak_vram_gb']:.2f} GB |"
            )

    lines.extend(
        [
            "",
            "## Tier 2: Public Chart Reasoning Benchmark (`HuggingFaceM4/ChartQA`, 50 Test Samples)",
            "",
            "| Configuration | Loops | Relaxed Accuracy (5% tol) | Truncation Rate | Peak VRAM |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in t2_results:
        m = r["metrics"]
        lines.append(
            f"| {r['display_name']} | {r['num_loops']} | **{m['relaxed_accuracy_pct']:.1f}%** | "
            f"{m['truncation_rate_pct']:.1f}% | {r['peak_vram_gb']:.2f} GB |"
        )

    lines.extend(
        [
            "",
            "## Tier 3: Public Tool-Calling Benchmark (`BFCL v3`, 50 Samples: 20 Simple / 15 Multiple / 15 Irrelevance)",
            "",
            "| Configuration | Loops | Overall Acc | Simple (20) | Multiple (15) | Irrelevance (15) | Truncation Rate | Peak VRAM |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for r in t3_results:
        m = r["metrics"]
        lines.append(
            f"| {r['display_name']} | {r['num_loops']} | **{m['overall_accuracy_pct']:.1f}%** | "
            f"{m['simple_acc_pct']:.1f}% | {m['multiple_acc_pct']:.1f}% | {m['irrelevance_acc_pct']:.1f}% | "
            f"{m['truncation_rate_pct']:.1f}% | {r['peak_vram_gb']:.2f} GB |"
        )

    md_text = "\n".join(lines) + "\n"
    OUT_MD.write_text(md_text, encoding="utf-8")
    return md_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stock vs. v1 vs. v2 3-Tier Hard Benchmark Suite.")
    parser.add_argument("--tiers", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--include-v2", action="store_true", help="Include v2 ETD+ARLoop merged model evaluations")
    parser.add_argument("--include-loop-sweep", action="store_true", help="Include num_loops=1,2,3 sweep on v1")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    t1_main: list[dict[str, Any]] = []
    t1_loop_sweep: list[dict[str, Any]] = []
    t2_results: list[dict[str, Any]] = []
    t3_results: list[dict[str, Any]] = []

    if 1 in args.tiers:
        ood_records = load_ood_sre_records()
        configs = [
            ("stock_l1_bf16", "Stock Qwen3.5-0.8B (1-Loop, bf16)", None, 1, "none", 64, False),
            ("merged_l2_bf16", "v1 Merged (2-Loop, bf16)", V1_MERGED_DIR, 2, "none", 64, False),
            ("merged_l2_tq4", "v1 Merged (2-Loop, Naive TQ 4-bit)", V1_MERGED_DIR, 2, "turboquant_4bit", 64, False),
            ("merged_l2_tq3", "v1 Merged (2-Loop, Naive TQ 3-bit)", V1_MERGED_DIR, 2, "turboquant_3bit", 64, False),
            ("v1_l2_tq4_vis", "v1 Merged (2-Loop, TQ 4-bit Vis-Exempt)", V1_MERGED_DIR, 2, "turboquant_4bit", 64, True),
            ("v1_l2_tq3_vis", "v1 Merged (2-Loop, TQ 3-bit Vis-Exempt)", V1_MERGED_DIR, 2, "turboquant_3bit", 64, True),
        ]
        if args.include_v2 and V2_MERGED_DIR.exists():
            configs.extend(
                [
                    ("v2_l2_bf16", "v2 ETD+ARLoop (2-Loop, bf16)", V2_MERGED_DIR, 2, "none", 64, False),
                    ("v2_l2_tq4_vis", "v2 ETD+ARLoop (2-Loop, TQ 4-bit Vis-Exempt)", V2_MERGED_DIR, 2, "turboquant_4bit", 64, True),
                    ("v2_l2_tq3_vis", "v2 ETD+ARLoop (2-Loop, TQ 3-bit Vis-Exempt)", V2_MERGED_DIR, 2, "turboquant_3bit", 64, True),
                ]
            )
        for tag, name, ckpt, loops, quant, win, prot in configs:
            t1_main.append(
                run_tier1_config(
                    tag, name, ckpt, loops, quant, win, ood_records, device,
                    protect_vision_prefix=prot, resume=args.resume,
                )
            )

        if args.include_loop_sweep:
            for tag, name, loops in [
                ("v1_l1_bf16", "v1 Merged (num_loops=1, bf16)", 1),
                ("merged_l2_bf16", "v1 Merged (num_loops=2, bf16)", 2),
                ("v1_l3_bf16", "v1 Merged (num_loops=3, bf16)", 3),
            ]:
                t1_loop_sweep.append(
                    run_tier1_config(
                        tag, name, V1_MERGED_DIR, loops, "none", 64, ood_records, device,
                        protect_vision_prefix=False, resume=args.resume,
                    )
                )

    if 2 in args.tiers:
        cqa_items = load_chartqa_slice(n=50, seed=args.seed)
        cqa_configs = [
            ("stock_l1_bf16", "Stock Qwen3.5-0.8B (1-Loop)", None, 1),
            ("merged_l2_bf16", "v1 Merged (2-Loop, bf16)", V1_MERGED_DIR, 2),
        ]
        if args.include_v2 and V2_MERGED_DIR.exists():
            cqa_configs.append(("v2_l2_bf16", "v2 ETD+ARLoop (2-Loop, bf16)", V2_MERGED_DIR, 2))
        for tag, name, ckpt, loops in cqa_configs:
            t2_results.append(run_tier2_config(tag, name, ckpt, loops, cqa_items, device, resume=args.resume))

    if 3 in args.tiers:
        bfcl_items = load_bfcl_slice(seed=args.seed)
        bfcl_configs = [
            ("stock_l1_bf16", "Stock Qwen3.5-0.8B (1-Loop)", None, 1),
            ("merged_l2_bf16", "v1 Merged (2-Loop, bf16)", V1_MERGED_DIR, 2),
        ]
        if args.include_v2 and V2_MERGED_DIR.exists():
            bfcl_configs.append(("v2_l2_bf16", "v2 ETD+ARLoop (2-Loop, bf16)", V2_MERGED_DIR, 2))
        for tag, name, ckpt, loops in bfcl_configs:
            t3_results.append(run_tier3_config(tag, name, ckpt, loops, bfcl_items, device, resume=args.resume))

    md_text = write_markdown_report(t1_main, t1_loop_sweep, t2_results, t3_results)
    print("\n" + "=" * 78)
    print(md_text)
    print("=" * 78)
    print(f"Saved report -> {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
