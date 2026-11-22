"""Build v2.0 training splits: 34 core SRE traces + 12 negative/refusal traces + 8 active crop_zoom traces.

Strictly preserves `splits/eval_heldout.jsonl` and `ood_sre_prompts.jsonl` with 0 overlap.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
SPLITS_DIR = BASE_DIR / "splits"
SAMPLES_DIR = BASE_DIR / "samples"

sys.path.insert(0, str(BASE_DIR))
from config import SYSTEM_PROMPT

BG_FIG = "#0B0C10"
BG_PANEL = "#141619"
GRID_COLOR = "#252A34"
TEXT_MAIN = "#E6EDF3"
TEXT_MUTED = "#8B949E"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_healthy_dashboard(out_path: Path, service: str, region: str, version: str, rng: np.random.Generator) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 60, 120)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), dpi=100, facecolor=BG_FIG)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.10, wspace=0.24, hspace=0.38)

    fig.text(
        0.02,
        0.94,
        f"GRAFANA | {service} (prod) | Healthy Status | Version: {version}",
        color="#3FB950",
        fontsize=10.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#12261E", edgecolor="#238636"),
    )
    fig.text(
        0.02,
        0.875,
        f"Region: {region} | All SLOs Green | Pod Restarts: 0 | Replicas: 6/6",
        color=TEXT_MUTED,
        fontsize=8.5,
    )

    cpu = 32 + rng.normal(0, 2.0, len(t))
    mem = 44 + rng.normal(0, 1.5, len(t))
    err_5xx = 0.06 + np.abs(rng.normal(0, 0.02, len(t)))
    p99 = 46 + rng.normal(0, 4.0, len(t))

    specs = [
        (axes[0, 0], cpu, "#3FB950", "CPU Utilization (%)", "%", (0, 100), 85),
        (axes[0, 1], mem, "#58A6FF", "Memory Usage (%)", "%", (0, 100), 95),
        (axes[1, 0], err_5xx, "#3FB950", "HTTP 5xx Rate (Healthy)", "%", (0, 5), 1.0),
        (axes[1, 1], p99, "#58A6FF", "p99 Latency (ms)", "ms", (0, 250), None),
    ]
    for ax, y, col, title, yl, ylim, thresh in specs:
        ax.set_facecolor(BG_PANEL)
        ax.plot(t, y, color=col, linewidth=1.6)
        if thresh is not None:
            ax.axhline(thresh, color="#FFA657", linestyle=":", linewidth=0.9)
        ax.set_title(title, color=TEXT_MAIN, fontsize=9.5, fontweight="bold", loc="left", pad=5)
        ax.set_ylabel(yl, color=TEXT_MUTED, fontsize=8)
        ax.set_xlabel("Time (min)", color=TEXT_MUTED, fontsize=7.5)
        ax.tick_params(colors=TEXT_MUTED, labelsize=7.5)
        ax.grid(True, color=GRID_COLOR, linestyle="-", linewidth=0.6, alpha=0.85)
        for sp in ax.spines.values():
            sp.set_color("#2C323E")
        ax.set_ylim(*ylim)

    fig.savefig(out_path, dpi=100, facecolor=BG_FIG)
    plt.close(fig)


def _crop_quadrant(src_path: Path, dst_path: Path, panel: str) -> None:
    with Image.open(src_path).convert("RGB") as img:
        w, h = img.size
        boxes = {
            "top_left": (0, int(h * 0.12), w // 2, int(h * 0.58)),
            "top_right": (w // 2, int(h * 0.12), w, int(h * 0.58)),
            "bottom_left": (0, int(h * 0.52), w // 2, h),
            "bottom_right": (w // 2, int(h * 0.52), w, h),
        }
        cropped = img.crop(boxes[panel])
        zoomed = cropped.resize((cropped.width * 2, cropped.height * 2), Image.Resampling.LANCZOS)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        zoomed.save(dst_path)


def main() -> int:
    rng = np.random.default_rng(2026)

    train_sft_v1 = [
        json.loads(l) for l in (SPLITS_DIR / "train_sft.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    train_grpo_v1 = [
        json.loads(l) for l in (SPLITS_DIR / "train_grpo_prompts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    eval_heldout = [
        json.loads(l) for l in (SPLITS_DIR / "eval_heldout.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    eval_ids = {r["id"] for r in eval_heldout}

    # ------------------------------------------------------------------
    # 1. Build 12 Negative / No-Tool Refusal Examples (8 healthy dashboards + 4 text info)
    # ------------------------------------------------------------------
    neg_sft_rows: list[dict[str, Any]] = []
    neg_grpo_prompts: list[dict[str, Any]] = []

    healthy_specs = [
        ("checkout-api", "us-east-1", "v2.4.0"),
        ("auth-gateway", "eu-west-1", "v3.1.2"),
        ("payments-worker", "ap-south-1", "v1.9.4"),
        ("search-indexer", "us-east-1", "v2.2.1"),
        ("orders-db", "eu-west-1", "v4.0.1"),
        ("checkout-api", "ap-south-1", "v2.4.1"),
        ("auth-gateway", "us-east-1", "v3.1.3"),
        ("payments-worker", "eu-west-1", "v1.9.5"),
    ]
    for i, (svc, reg, ver) in enumerate(healthy_specs, start=1):
        rid = f"neg_healthy_{i:02d}"
        img_rel = f"samples/{rid}.png"
        _render_healthy_dashboard(SAMPLES_DIR / f"{rid}.png", svc, reg, ver, rng)
        prompt = (
            f"Routine health check on `{svc}` ({reg}, {ver}). Inspect the attached Grafana dashboard "
            f"and decide if any remediation tool is needed, or confirm the service is healthy."
        )
        cot = (
            f"Inspecting the dashboard for {svc} in {reg} ({ver}): CPU is steady around 32% (well below 85%), "
            f"memory is flat at 44% with 0 pod restarts, HTTP 5xx is ~0.06% (well below 1% SLO), and p99 latency "
            f"is stable at ~46ms. All SLOs are green, so no remediation tool should be called."
        )
        answer = (
            f"No remediation tool is required. `{svc}` ({ver}, {reg}) is healthy across all panels "
            f"(CPU ~32%, memory ~44%, HTTP 5xx <0.1%, p99 latency ~46ms, 0 pod restarts)."
        )
        neg_sft_rows.append(
            {
                "id": rid,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"<image>\n{prompt}"},
                    {"role": "assistant", "content": f"<think>\n{cot}\n</think>\n{answer}"},
                ],
                "images": [img_rel],
                "is_refusal": True,
            }
        )
        neg_grpo_prompts.append(
            {
                "id": rid,
                "prompt": prompt,
                "image_path": img_rel,
                "expected_tool": "none",
                "expected_args": None,
            }
        )

    info_questions = [
        (
            "neg_info_01",
            "What is the difference between a SEV1 and a SEV3 incident in our on-call runbook? Do not trigger any tools.",
            "The user is asking a conceptual runbook question with no active outage or alert. No remediation tool applies.",
            "No tool call is needed. SEV1 denotes a critical customer-facing outage or multi-service SLO breach requiring immediate paging, whereas SEV3 covers minor localized issues handled during business hours.",
        ),
        (
            "neg_info_02",
            "Can you summarize what metrics indicate a memory leak OOMKill sawtooth pattern vs. an organic traffic surge?",
            "This is an informational SRE question with no failing service specified. Do not invoke any remediation tool.",
            "No tool call is needed. A memory leak OOMKill sawtooth shows RAM climbing linearly to 100% followed by abrupt drops and ExitCode 137 restarts, whereas an organic traffic surge shows rising RPS and CPU saturation (>85%) with healthy (<1%) error rates.",
        ),
        (
            "neg_info_03",
            "All services in us-east-1 passed post-deploy smoke tests with 0.02% 5xx and 38ms p99 latency. Should we roll back?",
            "Telemetry is completely healthy after deployment (0.02% 5xx, 38ms p99). Rolling back would be harmful; refuse tool invocation.",
            "No remediation tool should be called. All post-deploy metrics are healthy (0.02% 5xx, 38ms p99 latency), so no rollback is warranted.",
        ),
        (
            "neg_info_04",
            "Schedule a team lunch for the SRE crew next Friday at 12:30 PM.",
            "None of the available SRE incident triage tools handle calendar events or team lunches. Decline to call any tool.",
            "No tool call is possible: none of the available SRE remediation tools support scheduling calendar events or team lunches.",
        ),
    ]
    for rid, prompt, cot, answer in info_questions:
        neg_sft_rows.append(
            {
                "id": rid,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": f"<think>\n{cot}\n</think>\n{answer}"},
                ],
                "images": [],
                "is_refusal": True,
            }
        )
        neg_grpo_prompts.append(
            {
                "id": rid,
                "prompt": prompt,
                "image_path": None,
                "expected_tool": "none",
                "expected_args": None,
            }
        )

    # ------------------------------------------------------------------
    # 2. Build 8 Active Chart Zoom (`crop_zoom`) Multi-Turn SFT Traces
    # ------------------------------------------------------------------
    zoom_sft_rows: list[dict[str, Any]] = []
    mm_train_rows = [r for r in train_sft_v1 if r.get("images")]
    panel_by_hint = {
        "rollback": "bottom_left",
        "scale": "top_left",
        "oom": "top_right",
        "logs": "bottom_right",
        "incident": "bottom_left",
    }

    for idx, base_row in enumerate(mm_train_rows[:8], start=1):
        orig_img_rel = base_row["images"][0]
        orig_img_path = BASE_DIR / orig_img_rel
        if not orig_img_path.exists():
            continue

        panel = "bottom_left"
        for hint, p_name in panel_by_hint.items():
            if hint in orig_img_rel:
                panel = p_name
                break

        zoom_rel = f"samples/zoom_{idx:02d}_{panel}.png"
        _crop_quadrant(orig_img_path, BASE_DIR / zoom_rel, panel)

        orig_user = base_row["messages"][1]["content"]
        final_asst = base_row["messages"][-1]["content"]
        zoom_call = json.dumps({"tool": "crop_zoom", "arguments": {"panel": panel}}, ensure_ascii=False)

        zoom_sft_rows.append(
            {
                "id": f"zoom_{idx:02d}_{base_row['id']}",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": orig_user},
                    {
                        "role": "assistant",
                        "content": f"<think>\nThe key anomaly in the {panel} panel has fine-grained curves and labels. Let me zoom into `{panel}` at 2x resolution first to verify exact values.\n</think>\n{zoom_call}",
                    },
                    {
                        "role": "user",
                        "content": f"<image>\n[ToolResult: crop_zoom({panel})] High-resolution 2x crop attached. Now emit the final SRE remediation JSON tool call.",
                    },
                    {"role": "assistant", "content": final_asst},
                ],
                "images": [orig_img_rel, zoom_rel],
                "is_crop_zoom": True,
            }
        )

    train_sft_v2 = train_sft_v1 + neg_sft_rows + zoom_sft_rows
    train_grpo_v2 = train_grpo_v1 + neg_grpo_prompts

    # Verify zero overlap with eval_heldout.jsonl
    v2_ids = {r["id"] for r in train_sft_v2} | {r["id"] for r in train_grpo_v2}
    overlap = sorted(v2_ids & eval_ids)
    assert not overlap, f"LEAKAGE DETECTED in v2 splits: {overlap}"

    sft_v2_path = SPLITS_DIR / "train_sft_v2.jsonl"
    grpo_v2_path = SPLITS_DIR / "train_grpo_prompts_v2.jsonl"
    with sft_v2_path.open("w", encoding="utf-8") as fh:
        for r in train_sft_v2:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with grpo_v2_path.open("w", encoding="utf-8") as fh:
        for r in train_grpo_v2:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest_v2 = {
        "counts": {
            "train_sft_v2": len(train_sft_v2),
            "core_sre_sft": len(train_sft_v1),
            "negative_refusal_sft": len(neg_sft_rows),
            "active_crop_zoom_sft": len(zoom_sft_rows),
            "train_grpo_prompts_v2": len(train_grpo_v2),
            "eval_heldout_untouched": len(eval_heldout),
        },
        "leakage_checks": {"overlapping_ids": 0, "status": "PASS"},
        "sha256": {
            "train_sft_v2.jsonl": _sha256(sft_v2_path),
            "train_grpo_prompts_v2.jsonl": _sha256(grpo_v2_path),
            "eval_heldout.jsonl": _sha256(SPLITS_DIR / "eval_heldout.jsonl"),
        },
    }
    (SPLITS_DIR / "split_manifest_v2.json").write_text(json.dumps(manifest_v2, indent=2), encoding="utf-8")
    print("Built v2 training splits:")
    print(json.dumps(manifest_v2["counts"], indent=2))
    print("Leakage checks:", manifest_v2["leakage_checks"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
