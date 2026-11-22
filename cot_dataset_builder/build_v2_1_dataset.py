"""Build v2.1 scaled & rebalanced training splits (180 validated SFT rows, ~8.9% refusal).

Adds diverse light-mode themes, 1x3 / 3x1 / 2x2 panel layouts, zero-hint pixel-only prompts,
and adversarial text-vs-pixel Conflict Traps strictly using training-only service names.
Verifies 0 overlap (IDs, prompts, and images) with `splits/eval_heldout.jsonl` and `ood_sre_prompts.jsonl`.
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

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
SPLITS_DIR = BASE_DIR / "splits"
SAMPLES_V21_DIR = BASE_DIR / "samples_v2_1"

sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(ROOT_DIR))
from config import SYSTEM_PROMPT
from looped_qwen35.rewards import score_completion

# Strictly disjoint from OOD services (billing-engine, edge-proxy, ml-feature-store, notif-dispatcher, ledger-sync)
TRAIN_SERVICES = [
    "cart-service",
    "catalog-api",
    "order-router",
    "session-gateway",
    "inventory-worker",
    "pricing-engine",
    "shipping-api",
    "identity-broker",
    "webhook-relay",
    "analytics-ingest",
]

REGIONS = ["us-east-1", "eu-west-1", "ap-south-1"]
NAMESPACES = ["prod-core", "prod-edge", "prod-data", "prod-payments"]
VERSIONS = [
    ("v1.8.2", "v1.9.0"),
    ("v2.3.1", "v2.4.0"),
    ("v3.0.4", "v3.1.0"),
    ("v4.1.2", "v4.2.0"),
    ("v2.7.5", "v2.8.0"),
]

THEMES = [
    # Dark Grafana classic
    {"name": "dark", "bg": "#0B0C10", "panel": "#141619", "grid": "#252A34", "text": "#E6EDF3", "muted": "#8B949E", "c1": "#58A6FF", "c2": "#F85149", "c3": "#3FB950", "c4": "#D2A8FF"},
    # Light mode Grafana
    {"name": "light_clean", "bg": "#FFFFFF", "panel": "#F8FAFC", "grid": "#E2E8F0", "text": "#0F172A", "muted": "#475569", "c1": "#2563EB", "c2": "#DC2626", "c3": "#16A34A", "c4": "#9333EA"},
    # Light warm slate
    {"name": "light_slate", "bg": "#F4F5F7", "panel": "#FFFFFF", "grid": "#CBD5E1", "text": "#1E293B", "muted": "#64748B", "c1": "#0284C7", "c2": "#E11D48", "c3": "#059669", "c4": "#D97706"},
]

TOOLS = [
    "trigger_rollback",
    "scale_service",
    "restart_pod_pool",
    "query_logs",
    "create_incident",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_dashboard(
    out_path: Path,
    tool: str,
    service: str,
    region: str,
    namespace: str,
    prev_ver: str,
    bad_ver: str,
    layout: str,
    theme: dict[str, str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Render a realistic Grafana dashboard in 2x2, 1x3, or 3x1 layout and return ground-truth metrics."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 60, 120)

    if layout == "1x3":
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.1), dpi=100, facecolor=theme["bg"])
        fig.subplots_adjust(left=0.06, right=0.98, top=0.76, bottom=0.16, wspace=0.26)
        ax_list = list(axes)
    elif layout == "3x1":
        fig, axes = plt.subplots(3, 1, figsize=(6.2, 5.8), dpi=100, facecolor=theme["bg"])
        fig.subplots_adjust(left=0.11, right=0.96, top=0.86, bottom=0.07, hspace=0.46)
        ax_list = list(axes)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), dpi=100, facecolor=theme["bg"])
        fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.10, wspace=0.24, hspace=0.38)
        ax_list = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    deploy_t = 34.0
    extra_svc = TRAIN_SERVICES[(TRAIN_SERVICES.index(service) + 3) % len(TRAIN_SERVICES)]
    extra_svc2 = TRAIN_SERVICES[(TRAIN_SERVICES.index(service) + 5) % len(TRAIN_SERVICES)]

    if tool == "trigger_rollback":
        cpu = 42 + rng.normal(0, 2.0, len(t))
        mem = 50 + rng.normal(0, 1.5, len(t))
        err = np.where(t < deploy_t, 0.1 + np.abs(rng.normal(0, 0.04, len(t))), 14.5 + rng.normal(0, 1.2, len(t)))
        p99 = np.where(t < deploy_t, 55 + rng.normal(0, 5.0, len(t)), 780 + rng.normal(0, 45.0, len(t)))
        banner = f"GRAFANA | {service} (prod) | Deploy {bad_ver} (prev {prev_ver})"
        sub = f"Region: {region} | Namespace: {namespace} | Deploy Marker @ t=34m"
        panels = [
            (err, theme["c2"], "HTTP 5xx Rate (%)", "%", (0, 22), deploy_t),
            (p99, theme["c4"], "p99 Latency (ms)", "ms", (0, 1000), deploy_t),
            (cpu, theme["c1"], "CPU Utilization (%)", "%", (0, 100), deploy_t),
            (mem, theme["c3"], "Memory Usage (%)", "%", (0, 100), None),
        ]
        gt_args = {
            "service": service,
            "target_version": prev_ver,
            "environment": "prod",
            "reason": f"HTTP 5xx spiked to ~14.5% and p99 latency surged to ~780ms immediately after deploying {bad_ver}.",
        }
        think = (
            f"Looking at the dashboard for {service} (prod), there is a clear deployment marker for {bad_ver} (previous {prev_ver}) at t=34m. "
            f"Right after the deploy, HTTP 5xx jumps from 0.1% to ~14.5% and p99 latency spikes to ~780ms while CPU and memory stay flat. "
            f"This is a bad release regression, so I should roll back {service} to {prev_ver} using trigger_rollback."
        )
    elif tool == "scale_service":
        rps = np.linspace(1200, 5600, len(t)) + rng.normal(0, 90, len(t))
        cpu = np.clip(np.linspace(38, 96, len(t)) + rng.normal(0, 1.8, len(t)), 0, 100)
        err = 0.12 + np.abs(rng.normal(0, 0.04, len(t)))
        mem = 54 + rng.normal(0, 2.0, len(t))
        banner = f"GRAFANA | {service} (prod) | High Traffic Saturation ({prev_ver})"
        sub = f"Region: {region} | Current Replicas: 4 | Error Rate < 0.2%"
        panels = [
            (cpu, theme["c2"], "CPU Utilization (%)", "%", (0, 105), None),
            (rps, theme["c1"], "Inbound Traffic (RPS)", "rps", (0, 6500), None),
            (err, theme["c3"], "HTTP 5xx Rate (Healthy <1%)", "%", (0, 5), None),
            (mem, theme["c4"], "Memory Usage (%)", "%", (0, 100), None),
        ]
        gt_args = {
            "service": service,
            "target_replicas": 10,
            "region": region,
            "reason": f"Organic traffic climbed to ~5600 RPS driving CPU to 96% while HTTP 5xx remains healthy (<0.2%).",
        }
        think = (
            f"Inspecting the {service} telemetry in {region}: inbound traffic rises steadily from 1200 to ~5600 RPS and CPU saturates at 96%. "
            f"Crucially, HTTP 5xx stays healthy under 0.2% with no deploy marker or OOM sawtooth. "
            f"This is organic load saturation, so I should scale {service} out to 10 replicas in {region} via scale_service."
        )
    elif tool == "restart_pod_pool":
        saw = ((t % 15.0) / 15.0) * 62.0 + 38.0 + rng.normal(0, 1.2, len(t))
        mem = np.clip(saw, 0, 100)
        cpu = 40 + rng.normal(0, 2.5, len(t))
        restarts = np.floor(t / 15.0) * 2
        err = 0.4 + np.abs(rng.normal(0, 0.15, len(t)))
        banner = f"GRAFANA | {service} (prod) | OOMKill CrashLoop Sawtooth"
        sub = f"Region: {region} | Namespace: {namespace} | OOMKills: 8 in 60m"
        panels = [
            (mem, theme["c2"], "Container Memory (% - OOM Sawtooth)", "%", (0, 105), None),
            (restarts, theme["c4"], "Cumulative Pod OOM Restarts", "count", (0, 10), None),
            (cpu, theme["c1"], "CPU Utilization (%)", "%", (0, 100), None),
            (err, theme["c3"], "HTTP 5xx Rate (%)", "%", (0, 5), None),
        ]
        gt_args = {
            "service": service,
            "namespace": namespace,
            "strategy": "rolling",
            "reason": "Memory climbs linearly to 100% every 15 minutes with repeated OOMKill pod restarts.",
        }
        think = (
            f"The {service} dashboard in namespace {namespace} shows a textbook 15-minute memory leak sawtooth hitting 100% RAM followed by "
            f"repeated OOMKill pod restarts (8 restarts in 60m), while CPU stays moderate around 40%. "
            f"I should perform a rolling restart of the pod pool using restart_pod_pool."
        )
    elif tool == "query_logs":
        p99 = 65 + rng.normal(0, 8.0, len(t))
        for spike_idx in [24, 58, 92]:
            p99[spike_idx : spike_idx + 3] += rng.uniform(220, 340)
        cpu = 36 + rng.normal(0, 2.0, len(t))
        mem = 48 + rng.normal(0, 1.5, len(t))
        err = 0.25 + np.abs(rng.normal(0, 0.08, len(t)))
        banner = f"GRAFANA | {service} (prod) | Intermittent Latency Jitter"
        sub = f"Region: {region} | Namespace: {namespace} | No Recent Deploy"
        panels = [
            (p99, theme["c4"], "p99 Latency (Intermittent Spikes)", "ms", (0, 480), None),
            (err, theme["c3"], "HTTP 5xx Rate (%)", "%", (0, 5), None),
            (cpu, theme["c1"], "CPU Utilization (%)", "%", (0, 100), None),
            (mem, theme["c2"], "Memory Usage (%)", "%", (0, 100), None),
        ]
        gt_args = {
            "service": service,
            "time_window_minutes": 30,
            "severity": "ERROR",
            "filter_query": "timeout OR downstream OR latency",
        }
        think = (
            f"Looking at {service}, CPU (36%), memory (48%), and HTTP 5xx (0.25%) are all normal with no deploy marker or OOM pattern. "
            f"However, p99 latency shows brief intermittent spikes up to ~350ms. Without an obvious single remediation trigger, "
            f"the right diagnostic action is to inspect recent ERROR logs over the last 30 minutes using query_logs."
        )
    else:  # create_incident
        err = np.where(t < 25, 0.15 + np.abs(rng.normal(0, 0.05, len(t))), 38.0 + rng.normal(0, 3.5, len(t)))
        p99 = np.where(t < 25, 50 + rng.normal(0, 5, len(t)), 1850 + rng.normal(0, 120, len(t)))
        cpu = np.where(t < 25, 40 + rng.normal(0, 2, len(t)), 92 + rng.normal(0, 3, len(t)))
        mem = np.where(t < 25, 48 + rng.normal(0, 2, len(t)), 89 + rng.normal(0, 3, len(t)))
        banner = f"GRAFANA | SEV1 MULTI-SERVICE OUTAGE | {service} + {extra_svc} + {extra_svc2}"
        sub = f"Region: {region} | Cascading SLO Breach | 3 Services Failing Simultaneously"
        panels = [
            (err, theme["c2"], "Fleet HTTP 5xx Rate (% - Critical)", "%", (0, 55), None),
            (p99, theme["c4"], "Fleet p99 Latency (ms)", "ms", (0, 2300), None),
            (cpu, theme["c1"], "Cluster CPU (%)", "%", (0, 105), None),
            (mem, theme["c3"], "Cluster Memory (%)", "%", (0, 105), None),
        ]
        gt_args = {
            "severity": "SEV1",
            "affected_services": [service, extra_svc, extra_svc2],
            "summary": f"Fleet HTTP 5xx spiked to ~38% and p99 latency exceeded 1800ms across {service}, {extra_svc}, and {extra_svc2} in {region}.",
            "page_oncall": True,
        }
        think = (
            f"The dashboard shows a severe fleet-wide cascading outage affecting three services ({service}, {extra_svc}, and {extra_svc2}) in {region}. "
            f"HTTP 5xx jumped to ~38% and p99 latency surged past 1800ms simultaneously across the cluster without a single-service deploy trigger. "
            f"This requires immediate SEV1 escalation via create_incident."
        )

    fig.text(
        0.02, 0.93, banner, color=theme["text"], fontsize=9.8, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=theme["panel"], edgecolor=theme["grid"]),
    )
    fig.text(0.02, 0.855 if layout != "1x3" else 0.83, sub, color=theme["muted"], fontsize=8.0)

    for ax, (y, col, title, yl, ylim, vline) in zip(ax_list, panels[: len(ax_list)]):
        ax.set_facecolor(theme["panel"])
        ax.plot(t, y, color=col, linewidth=1.7)
        if vline is not None:
            ax.axvline(vline, color="#F59E0B", linestyle="--", linewidth=1.4, label=f"Deploy {bad_ver}")
            ax.legend(loc="upper left", fontsize=7.0, facecolor=theme["panel"], edgecolor=theme["grid"], labelcolor=theme["text"])
        ax.set_title(title, color=theme["text"], fontsize=8.8, fontweight="bold", loc="left", pad=4)
        ax.set_ylabel(yl, color=theme["muted"], fontsize=7.5)
        ax.set_xlabel("Time (min)", color=theme["muted"], fontsize=7.0)
        ax.tick_params(colors=theme["muted"], labelsize=7.0)
        ax.grid(True, color=theme["grid"], linestyle="-", linewidth=0.55, alpha=0.85)
        for sp in ax.spines.values():
            sp.set_color(theme["grid"])
        ax.set_ylim(*ylim)

    fig.savefig(out_path, dpi=100, facecolor=theme["bg"])
    plt.close(fig)
    return {"gt_args": gt_args, "think": think}


def _render_light_healthy(out_path: Path, service: str, region: str, version: str, theme: dict[str, str], rng: np.random.Generator) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 60, 120)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), dpi=100, facecolor=theme["bg"])
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.10, wspace=0.24, hspace=0.38)
    fig.text(0.02, 0.93, f"GRAFANA | {service} (prod) | Healthy Status | Version: {version}", color=theme["c3"], fontsize=10.0, fontweight="bold")
    fig.text(0.02, 0.87, f"Region: {region} | All SLOs Green | Pod Restarts: 0 | Replicas: 6/6", color=theme["muted"], fontsize=8.2)
    specs = [
        (axes[0, 0], 29 + rng.normal(0, 1.8, len(t)), theme["c3"], "CPU Utilization (%)", "%", (0, 100)),
        (axes[0, 1], 41 + rng.normal(0, 1.4, len(t)), theme["c1"], "Memory Usage (%)", "%", (0, 100)),
        (axes[1, 0], 0.05 + np.abs(rng.normal(0, 0.02, len(t))), theme["c3"], "HTTP 5xx Rate (Healthy)", "%", (0, 5)),
        (axes[1, 1], 42 + rng.normal(0, 3.5, len(t)), theme["c1"], "p99 Latency (ms)", "ms", (0, 250)),
    ]
    for ax, y, col, title, yl, ylim in specs:
        ax.set_facecolor(theme["panel"])
        ax.plot(t, y, color=col, linewidth=1.6)
        ax.set_title(title, color=theme["text"], fontsize=9.0, fontweight="bold", loc="left", pad=4)
        ax.set_ylabel(yl, color=theme["muted"], fontsize=7.5)
        ax.set_xlabel("Time (min)", color=theme["muted"], fontsize=7.0)
        ax.tick_params(colors=theme["muted"], labelsize=7.0)
        ax.grid(True, color=theme["grid"], linewidth=0.55)
        ax.set_ylim(*ylim)
    fig.savefig(out_path, dpi=100, facecolor=theme["bg"])
    plt.close(fig)


def _conflict_text(true_tool: str, service: str) -> str:
    """Create a misleading alert text that contradicts the visual dashboard pixels."""
    wrong_hints = {
        "trigger_rollback": f"Alert note: On-call suspects organic traffic overload on {service} and suggests scaling replicas. Verify against the attached Grafana dashboard before acting!",
        "scale_service": f"Alert note: Pager says {service} might have a memory leak causing OOMKills. Check the actual dashboard metrics carefully and choose the right tool.",
        "restart_pod_pool": f"Alert note: A teammate thinks the recent deploy broke {service} and asked for a rollback. Inspect the dashboard curves first and pick the true remediation.",
        "query_logs": f"Alert note: Automated triage flagged a possible SEV1 fleet outage on {service}. Inspect the attached dashboard closely before taking action.",
        "create_incident": f"Alert note: Ticket claims {service} only needs a minor log query for slight latency jitter. Look at the actual dashboard panels and respond appropriately.",
    }
    return wrong_hints[true_tool]


def main() -> int:
    rng = np.random.default_rng(202602)

    train_sft_v2 = [
        json.loads(l) for l in (SPLITS_DIR / "train_sft_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    eval_heldout = [
        json.loads(l) for l in (SPLITS_DIR / "eval_heldout.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    ood_prompts = [
        json.loads(l) for l in (BASE_DIR / "ood_sre_prompts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    ]

    # Start with all 54 validated v2 rows (34 core + 12 negative + 8 crop_zoom)
    sft_rows: list[dict[str, Any]] = list(train_sft_v2)
    grpo_rows: list[dict[str, Any]] = [
        json.loads(l) for l in (SPLITS_DIR / "train_grpo_prompts_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    ]

    # Add 4 fresh light-mode healthy negative refusal rows (total negative = 16 / 180 = 8.9%)
    for i in range(4):
        svc = TRAIN_SERVICES[i]
        reg = REGIONS[i % len(REGIONS)]
        ver = VERSIONS[i % len(VERSIONS)][0]
        theme = THEMES[1 + (i % 2)]
        img_rel = f"samples_v2_1/v21_neg_light_{i+1:02d}.png"
        _render_light_healthy(BASE_DIR / img_rel, svc, reg, ver, theme, rng)
        prompt = f"Routine check on {svc} ({reg}). Review the attached Grafana dashboard and determine if any remediation tool is required."
        assistant = (
            f"<think>\nInspecting the light-mode Grafana dashboard for {svc} in {reg}: CPU (~29%), memory (~41%), "
            f"HTTP 5xx (~0.05%), and p99 latency (~42ms) are all well within healthy SLO thresholds with zero pod restarts. "
            f"No remediation tool should be called.\n</think>\n"
            f"No remediation action is required. All telemetry metrics for {svc} in {reg} are healthy and within normal SLOs."
        )
        row_id = f"v21_neg_light_{i+1:02d}"
        sft_rows.append({
            "id": row_id,
            "category": "negative_refusal",
            "expected_tool": "none",
            "image_path": img_rel,
            "prompt": prompt,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant},
            ],
        })
        grpo_rows.append({
            "id": row_id,
            "prompt": prompt,
            "image_path": img_rel,
            "expected_tool": "none",
            "category": "negative_refusal",
        })

    # Add 122 fresh positive Visual SRE rows (45 visual_shift + 37 pixel_only + 40 conflict_trap)
    # Total = 54 + 4 + 122 = 180 SFT rows (16 negative = 8.89%)
    plan = (
        [("visual_shift", TOOLS[i % 5]) for i in range(45)]
        + [("pixel_only", TOOLS[i % 5]) for i in range(37)]
        + [("conflict_trap", TOOLS[i % 5]) for i in range(40)]
    )

    pixel_only_prompts = [
        "Inspect the attached Grafana telemetry dashboard and emit the right remediation tool call.",
        "Analyze the attached production dashboard screenshot and select the appropriate SRE tool.",
        "Review this telemetry dashboard and execute the exact required remediation action.",
    ]

    for idx, (subset, tool) in enumerate(plan, start=1):
        svc = TRAIN_SERVICES[(idx * 3) % len(TRAIN_SERVICES)]
        reg = REGIONS[idx % len(REGIONS)]
        ns = NAMESPACES[idx % len(NAMESPACES)]
        prev_ver, bad_ver = VERSIONS[idx % len(VERSIONS)]

        if subset == "visual_shift":
            theme = THEMES[1 + (idx % 2)]  # light themes
            layout = ["1x3", "3x1", "2x2"][idx % 3]
            prompt = f"Production alert fired for {svc} in {reg} (namespace {ns}). Inspect the dashboard and select the correct remediation tool."
        elif subset == "pixel_only":
            theme = THEMES[idx % len(THEMES)]
            layout = ["2x2", "1x3", "3x1"][idx % 3]
            prompt = pixel_only_prompts[idx % len(pixel_only_prompts)]
        else:  # conflict_trap
            theme = THEMES[idx % len(THEMES)]
            layout = ["2x2", "1x3", "3x1"][idx % 3]
            prompt = _conflict_text(tool, svc)

        row_id = f"v21_{subset}_{idx:03d}_{tool}"
        img_rel = f"samples_v2_1/{row_id}.png"
        meta = _render_dashboard(
            BASE_DIR / img_rel, tool, svc, reg, ns, prev_ver, bad_ver, layout, theme, rng
        )

        tool_json = json.dumps({"tool": tool, "arguments": meta["gt_args"]})
        if subset == "conflict_trap":
            think = (
                f"The alert text suggests a different cause, so I must trust the actual Grafana pixels over the text hint. "
                + meta["think"]
            )
        else:
            think = meta["think"]

        assistant = f"<think>\n{think}\n</think>\n{tool_json}"

        # Deterministic schema + reward validation gate
        score = score_completion(assistant, expected_tool=tool, expected_args=meta["gt_args"])
        assert score["schema_reward"] == 0.35 and score["tool_match_reward"] == 0.50, (
            f"Validation failed for {row_id}: {score}"
        )

        sft_rows.append({
            "id": row_id,
            "category": subset,
            "expected_tool": tool,
            "expected_args": meta["gt_args"],
            "image_path": img_rel,
            "prompt": prompt,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant},
            ],
        })
        grpo_rows.append({
            "id": row_id,
            "prompt": prompt,
            "image_path": img_rel,
            "expected_tool": tool,
            "expected_args": meta["gt_args"],
            "category": subset,
        })

    # Strict leakage verification against BOTH eval_heldout.jsonl AND ood_sre_prompts.jsonl
    heldout_ids = {r["id"] for r in eval_heldout}
    ood_ids = {r["id"] for r in ood_prompts}
    train_ids = {r["id"] for r in sft_rows}

    heldout_imgs = {r.get("image_path") for r in eval_heldout if r.get("image_path")}
    ood_imgs = {r.get("image_path") for r in ood_prompts if r.get("image_path")}
    train_imgs = {r.get("image_path") for r in sft_rows if r.get("image_path")}

    overlap_heldout_ids = train_ids & heldout_ids
    overlap_ood_ids = train_ids & ood_ids
    overlap_heldout_imgs = train_imgs & heldout_imgs
    overlap_ood_imgs = train_imgs & ood_imgs

    assert not overlap_heldout_ids and not overlap_ood_ids, "ID leakage detected!"
    assert not overlap_heldout_imgs and not overlap_ood_imgs, "Image leakage detected!"

    sft_out = SPLITS_DIR / "train_sft_v2_1.jsonl"
    grpo_out = SPLITS_DIR / "train_grpo_prompts_v2_1.jsonl"
    manifest_out = SPLITS_DIR / "split_manifest_v2_1.json"

    sft_out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in sft_rows) + "\n", encoding="utf-8")
    grpo_out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in grpo_rows) + "\n", encoding="utf-8")

    neg_count = sum(1 for r in sft_rows if r.get("expected_tool") == "none")
    manifest = {
        "version": "2.1",
        "train_sft_v2_1_count": len(sft_rows),
        "train_grpo_prompts_v2_1_count": len(grpo_rows),
        "negative_refusal_count": neg_count,
        "negative_refusal_pct": round(100.0 * neg_count / len(sft_rows), 2),
        "eval_heldout_count": len(eval_heldout),
        "ood_sre_count": len(ood_prompts),
        "overlapping_heldout_ids": len(overlap_heldout_ids),
        "overlapping_ood_ids": len(overlap_ood_ids),
        "overlapping_heldout_images": len(overlap_heldout_imgs),
        "overlapping_ood_images": len(overlap_ood_imgs),
        "sha256": {
            "train_sft_v2_1.jsonl": _sha256_file(sft_out),
            "train_grpo_prompts_v2_1.jsonl": _sha256_file(grpo_out),
            "eval_heldout.jsonl": _sha256_file(SPLITS_DIR / "eval_heldout.jsonl"),
            "ood_sre_prompts.jsonl": _sha256_file(BASE_DIR / "ood_sre_prompts.jsonl"),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
