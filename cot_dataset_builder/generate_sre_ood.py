"""Generate 50 Hard Out-of-Distribution (OOD) Visual SRE stress-test dashboards & prompts.

Subsets (balanced across all 5 SRE tools):
1. visual_shift (20): Light-mode Grafana (#FFFFFF/#F4F5F7), unseen 1x3 and 3x1 layouts,
   shifted color palettes, and unseen service names/versions/namespaces.
2. pixel_only (15): Neutral zero-hint prompt ("Inspect the attached telemetry dashboard
   and execute the appropriate remediation tool.") forcing pure visual chart + header reading.
3. conflict_trap (15): Noisy/misleading automated alert text contradicting the chart,
   where the visual dashboard is ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_JSONL = BASE_DIR / "ood_sre_prompts.jsonl"
DEFAULT_SAMPLES_DIR = BASE_DIR / "samples_ood"

OOD_SERVICES = [
    "billing-engine",
    "edge-proxy",
    "ml-feature-store",
    "notif-dispatcher",
    "ledger-sync",
]
OOD_NAMESPACES = ["fin-core", "edge-mesh", "ml-serving", "messaging-prod", "ledger-ops"]
REGIONS = ["us-east-1", "eu-west-1", "ap-south-1"]
ENVIRONMENTS = ["prod", "staging"]

LIGHT_THEME = {
    "fig_bg": "#FFFFFF",
    "panel_bg": "#F4F5F7",
    "grid": "#D1D5DB",
    "spine": "#9CA3AF",
    "text": "#111827",
    "muted": "#4B5563",
    "hdr_bg": "#EEF2FF",
    "hdr_fg": "#3730A3",
    "hdr_crit_bg": "#FEE2E2",
    "hdr_crit_fg": "#991B1B",
    "c_cpu": "#0D9488",
    "c_mem": "#4F46E5",
    "c_5xx": "#BE123C",
    "c_lat": "#B45309",
    "c_rps": "#7E22CE",
    "c_deploy": "#D97706",
    "c_multi": ["#BE123C", "#0369A1", "#7E22CE"],
}

DARK_ALT_THEME = {
    "fig_bg": "#0D1117",
    "panel_bg": "#161B22",
    "grid": "#2A313C",
    "spine": "#3B434F",
    "text": "#F0F6FC",
    "muted": "#9CA3AF",
    "hdr_bg": "#1F2937",
    "hdr_fg": "#38BDF8",
    "hdr_crit_bg": "#450A0A",
    "hdr_crit_fg": "#FCA5A5",
    "c_cpu": "#2DD4BF",
    "c_mem": "#818CF8",
    "c_5xx": "#FB7185",
    "c_lat": "#FBBF24",
    "c_rps": "#C084FC",
    "c_deploy": "#FDE047",
    "c_multi": ["#FB7185", "#38BDF8", "#C084FC"],
}


def _style_ood_ax(ax: Any, title: str, ylabel: str, theme: dict[str, Any], ylim: tuple[float, float]) -> None:
    ax.set_facecolor(theme["panel_bg"])
    ax.set_title(title, color=theme["text"], fontsize=9.0, fontweight="bold", loc="left", pad=4)
    ax.set_ylabel(ylabel, color=theme["muted"], fontsize=7.5)
    ax.set_xlabel("Time (min)", color=theme["muted"], fontsize=7.0)
    ax.tick_params(colors=theme["muted"], labelsize=7.0)
    ax.grid(True, color=theme["grid"], linestyle="-", linewidth=0.6, alpha=0.85)
    for sp in ax.spines.values():
        sp.set_color(theme["spine"])
    ax.set_ylim(*ylim)


def render_ood_dashboard(
    out_path: Path,
    scenario: str,
    meta: dict[str, Any],
    layout: str,
    light_mode: bool,
    rng: np.random.Generator,
) -> None:
    """Render an OOD Grafana dashboard in `1x3` or `3x1` layout with light or dark alt theme."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    theme = LIGHT_THEME if light_mode else DARK_ALT_THEME
    t = np.linspace(0, 60, 120)

    if layout == "1x3":
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.3), dpi=100, facecolor=theme["fig_bg"])
        fig.subplots_adjust(left=0.06, right=0.98, top=0.73, bottom=0.16, wspace=0.26)
        hdr_y, sub_y = 0.90, 0.81
    else:  # 3x1
        fig, axes = plt.subplots(3, 1, figsize=(5.8, 6.0), dpi=100, facecolor=theme["fig_bg"])
        fig.subplots_adjust(left=0.12, right=0.96, top=0.85, bottom=0.07, hspace=0.48)
        hdr_y, sub_y = 0.955, 0.905

    is_crit = scenario == "create_incident"
    fig.text(
        0.02,
        hdr_y,
        meta["banner_title"],
        color=theme["hdr_crit_fg"] if is_crit else theme["hdr_fg"],
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor=theme["hdr_crit_bg"] if is_crit else theme["hdr_bg"],
            edgecolor=theme["spine"],
        ),
    )
    fig.text(0.02, sub_y, meta["banner_sub"], color=theme["muted"], fontsize=7.8)

    ax1, ax2, ax3 = axes[0], axes[1], axes[2]

    if scenario == "trigger_rollback":
        deploy_t = 28.0
        post = t >= deploy_t
        cpu = 44 + rng.normal(0, 2.2, len(t)) + post * 6.0
        err_5xx = np.where(post, rng.uniform(13.0, 26.0, len(t)), rng.uniform(0.05, 0.25, len(t)))
        p99 = np.where(post, rng.uniform(980, 1550, len(t)), rng.uniform(45, 70, len(t)))

        ax1.plot(t, cpu, color=theme["c_cpu"], linewidth=1.6)
        ax2.plot(t, err_5xx, color=theme["c_5xx"], linewidth=1.8)
        ax3.plot(t, p99, color=theme["c_lat"], linewidth=1.8)
        for ax in (ax1, ax2, ax3):
            ax.axvline(deploy_t, color=theme["c_deploy"], linestyle="--", linewidth=1.4)
        ax2.text(deploy_t + 1.2, 26.5, f"Deploy {meta['new_version']}", color=theme["c_deploy"], fontsize=7.2, fontweight="bold")
        ax2.axhline(1.0, color=theme["c_lat"], linestyle=":", linewidth=1.0)

        _style_ood_ax(ax1, "CPU Utilization (%)", "%", theme, (0, 100))
        _style_ood_ax(ax2, "HTTP 5xx Error Rate (%)", "%", theme, (0, 32))
        _style_ood_ax(ax3, "p99 Latency (ms)", "ms", theme, (0, 1750))

    elif scenario == "scale_service":
        ramp = 1.0 / (1.0 + np.exp(-(t - 24.0) / 5.5))
        cpu = 34 + 61 * ramp + rng.normal(0, 1.7, len(t))
        err_5xx = 0.12 + 0.22 * ramp + rng.normal(0, 0.04, len(t))
        rps = 1200 + 3900 * ramp + rng.normal(0, 95, len(t))

        ax1.plot(t, np.clip(cpu, 0, 100), color=theme["c_5xx"], linewidth=1.8)
        ax1.axhline(85, color=theme["c_lat"], linestyle="--", linewidth=1.1)
        ax1.text(2, 87.5, "85% Saturation", color=theme["c_lat"], fontsize=7.0, fontweight="bold")
        ax2.plot(t, np.clip(err_5xx, 0, 5), color=theme["c_cpu"], linewidth=1.6)
        ax2.axhline(1.0, color=theme["c_lat"], linestyle=":", linewidth=1.0)
        ax3.plot(t, rps, color=theme["c_rps"], linewidth=1.8)

        _style_ood_ax(ax1, "CPU Saturation (%)", "%", theme, (0, 100))
        _style_ood_ax(ax2, "HTTP 5xx (Healthy <1%)", "%", theme, (0, 5))
        _style_ood_ax(ax3, "Organic Traffic (RPS)", "req/s", theme, (0, 5800))

    elif scenario == "restart_pod_pool":
        period = 15.0
        saw = (t % period) / period
        mem = 32 + 68 * saw + rng.normal(0, 1.2, len(t))
        cpu = 47 + 9 * saw + rng.normal(0, 1.8, len(t))
        oom_times = [15.0, 30.0, 45.0]
        err_5xx = 0.2 + rng.normal(0, 0.05, len(t))
        for ot in oom_times:
            err_5xx += 2.0 * np.exp(-0.5 * ((t - ot) / 0.8) ** 2)

        ax1.plot(t, np.clip(mem, 0, 100), color=theme["c_5xx"], linewidth=1.8)
        ax1.axhline(95, color=theme["c_lat"], linestyle="--", linewidth=1.0)
        for ot in oom_times:
            ax1.scatter([ot], [99], color=theme["c_5xx"], s=24, zorder=5)
            ax1.text(ot - 3.8, 86, "OOMKill", color=theme["c_5xx"], fontsize=6.8, fontweight="bold")
        ax2.plot(t, cpu, color=theme["c_cpu"], linewidth=1.6)
        ax3.plot(t, np.clip(err_5xx, 0, 10), color=theme["c_lat"], linewidth=1.6)

        _style_ood_ax(ax1, "Memory OOMKill Sawtooth (%)", "%", theme, (0, 105))
        _style_ood_ax(ax2, "CPU Utilization (%)", "%", theme, (0, 100))
        _style_ood_ax(ax3, "HTTP 5xx Rate (%)", "%", theme, (0, 10))

    elif scenario == "query_logs":
        cpu = 42 + rng.normal(0, 2.8, len(t))
        err_5xx = 0.30 + rng.normal(0, 0.10, len(t))
        p99 = 82 + rng.normal(0, 11, len(t))
        for spike_t in (10.0, 25.0, 41.0, 52.0):
            p99 += rng.uniform(280, 430) * np.exp(-0.5 * ((t - spike_t) / 1.3) ** 2)

        ax1.plot(t, cpu, color=theme["c_cpu"], linewidth=1.6)
        ax2.plot(t, np.clip(err_5xx, 0, 5), color=theme["c_mem"], linewidth=1.6)
        ax2.axhline(1.0, color=theme["c_lat"], linestyle=":", linewidth=1.0)
        ax3.plot(t, p99, color=theme["c_lat"], linewidth=1.8)
        ax3.text(3, 530, "Intermittent Jitter (Unconfirmed)", color=theme["c_lat"], fontsize=7.0, fontweight="bold")

        _style_ood_ax(ax1, "CPU Utilization (%)", "%", theme, (0, 100))
        _style_ood_ax(ax2, "HTTP 5xx Rate (<1%)", "%", theme, (0, 5))
        _style_ood_ax(ax3, "p99 Latency Bursts (ms)", "ms", theme, (0, 640))

    else:  # create_incident
        onset = 20.0
        svcs = meta["affected_services"][:3]
        for idx, (svc_name, col) in enumerate(zip(svcs, theme["c_multi"])):
            p_curve = 1.0 / (1.0 + np.exp(-(t - (onset + idx * 1.5)) / 2.0))
            ax1.plot(t, 44 + 49 * p_curve + rng.normal(0, 1.5, len(t)), color=col, linewidth=1.5, label=svc_name)
            ax2.plot(t, 0.2 + (24 + idx * 5) * p_curve + rng.normal(0, 0.8, len(t)), color=col, linewidth=1.7)
            ax3.plot(t, 65 + (1950 + idx * 320) * p_curve + rng.normal(0, 40, len(t)), color=col, linewidth=1.7)

        ax1.legend(loc="upper left", fontsize=6.3, facecolor=theme["panel_bg"], edgecolor=theme["grid"], labelcolor=theme["text"])
        ax2.axhline(5.0, color=theme["c_5xx"], linestyle="--", linewidth=1.0)
        ax2.text(2, 34, "CRITICAL SLO BREACH", color=theme["c_5xx"], fontsize=7.2, fontweight="bold")

        _style_ood_ax(ax1, "Multi-Service CPU (%)", "%", theme, (0, 100))
        _style_ood_ax(ax2, "Cascading 5xx Rate (%)", "%", theme, (0, 42))
        _style_ood_ax(ax3, "Cascading p99 Latency (ms)", "ms", theme, (0, 3100))

    fig.savefig(out_path, dpi=100, facecolor=theme["fig_bg"])
    plt.close(fig)


def generate_ood_dataset(
    output_path: Path = DEFAULT_OUTPUT_JSONL,
    samples_dir: Path = DEFAULT_SAMPLES_DIR,
    seed: int = 1337,
) -> list[dict[str, Any]]:
    """Generate 50 balanced OOD Visual SRE prompts (20 visual_shift, 15 pixel_only, 15 conflict_trap)."""
    np_rng = np.random.default_rng(seed)
    tools = [
        "trigger_rollback",
        "scale_service",
        "restart_pod_pool",
        "query_logs",
        "create_incident",
    ]
    # Exact 10 per tool: 4 visual_shift + 3 pixel_only + 3 conflict_trap = 50 total
    subset_schedule = (
        ["visual_shift"] * 4 + ["pixel_only"] * 3 + ["conflict_trap"] * 3
    )

    records: list[dict[str, Any]] = []
    global_idx = 0

    for t_idx, tool in enumerate(tools):
        for i, subset in enumerate(subset_schedule):
            global_idx += 1
            item_id = f"ood_{global_idx:03d}"

            svc = OOD_SERVICES[(i + t_idx) % len(OOD_SERVICES)]
            ns = OOD_NAMESPACES[(i + t_idx) % len(OOD_NAMESPACES)]
            region = REGIONS[(i + t_idx) % len(REGIONS)]
            env = ENVIRONMENTS[0 if i % 4 != 3 else 1]

            # Light mode for all visual_shift + half of pixel_only/conflict_trap; 1x3 and 3x1 alternating
            light_mode = True if subset == "visual_shift" else (i % 2 == 0)
            layout = "1x3" if (global_idx % 2 == 1) else "3x1"

            if tool == "trigger_rollback":
                minor = 1 + (i % 7)
                prev_ver = f"v4.{minor}.0"
                new_ver = f"v4.{minor + 1}.0"
                expected_args = {
                    "service": svc,
                    "target_version": prev_ver,
                    "environment": env,
                    "reason": f"HTTP 5xx and p99 latency spiked immediately after {new_ver} deployment",
                }
                meta = {
                    "new_version": new_ver,
                    "banner_title": f"TELEMETRY | {svc} | env={env} | cur={new_ver} | prev={prev_ver}",
                    "banner_sub": f"Region: {region} | Namespace: {ns} | Deploy {new_ver} at t=28m | Replicas: 10/10",
                }
                normal_prompt = (
                    f"Telemetry alert on `{svc}` ({env}, {region}). Previous stable version was `{prev_ver}`, "
                    f"`{new_ver}` deployed at t=28m. Inspect the dashboard and execute the right SRE tool."
                )
                conflict_prompt = (
                    f"[AUTOMATED NOTE - UNVERIFIED] Low-severity latency jitter reported on `{svc}` ({env}, {region}); "
                    f"initial triage bot suspects harmless downstream log warnings. Check the attached dashboard "
                    f"(ground truth; prev={prev_ver}, cur={new_ver}) and invoke the correct remediation tool."
                )

            elif tool == "scale_service":
                cur_rep = 8 + (i % 4) * 2
                target_rep = cur_rep * 2
                expected_args = {
                    "service": svc,
                    "target_replicas": target_rep,
                    "region": region,
                    "reason": f"Organic traffic surge saturated CPU >85% with healthy 5xx (<1%); scale to {target_rep}",
                }
                meta = {
                    "banner_title": f"TELEMETRY | {svc} | region={region} | replicas={cur_rep} -> target={target_rep}",
                    "banner_sub": f"Env: prod | No recent deploys | Organic RPS Surge | 5xx Healthy (<1%)",
                }
                normal_prompt = (
                    f"High CPU alert on `{svc}` in `{region}` (current={cur_rep}, target={target_rep} replicas). "
                    f"Inspect the attached telemetry dashboard and execute the appropriate SRE tool."
                )
                conflict_prompt = (
                    f"[AUTOMATED NOTE - UNVERIFIED] Alert rule flagged a possible memory leak or bad release on `{svc}` "
                    f"in `{region}` (replicas={cur_rep}, target={target_rep}). Trust the attached dashboard curves "
                    f"as ground truth and invoke the correct SRE tool."
                )

            elif tool == "restart_pod_pool":
                strategy = "rolling" if i % 3 != 2 else "immediate"
                restarts = 10 + (i % 6) * 2
                expected_args = {
                    "service": svc,
                    "namespace": ns,
                    "strategy": strategy,
                    "reason": f"Memory leak OOMKill sawtooth reaching 100% RAM with {restarts} pod restarts",
                }
                meta = {
                    "banner_title": f"TELEMETRY | {svc} | ns={ns} | Restarts: {restarts} (OOMKilled)",
                    "banner_sub": f"Region: {region} | Restart policy: {strategy} | No recent deploys",
                }
                normal_prompt = (
                    f"Pod stability alert on `{svc}` in namespace `{ns}` (policy: `{strategy}`). "
                    f"Inspect the attached dashboard and invoke the right SRE remediation tool."
                )
                conflict_prompt = (
                    f"[AUTOMATED NOTE - UNVERIFIED] Auto-triage suggests scaling replicas for high load on `{svc}` "
                    f"in namespace `{ns}` (policy={strategy}). Verify against the attached telemetry dashboard "
                    f"as ground truth and execute the right SRE tool."
                )

            elif tool == "query_logs":
                win_mins = [15, 30, 45, 60][i % 4]
                sev = ["ERROR", "WARN", "FATAL"][i % 3]
                filt = ["grpc_deadline", "socket_hangup", "db_pool_wait", "dns_lookup_retry"][i % 4]
                expected_args = {
                    "service": svc,
                    "time_window_minutes": win_mins,
                    "severity": sev,
                    "filter_query": filt,
                }
                meta = {
                    "banner_title": f"TELEMETRY | {svc} | {region} | Intermittent Latency ({win_mins}m)",
                    "banner_sub": f"Root cause unconfirmed | Filter: severity={sev} query='{filt}' | No deploys",
                }
                normal_prompt = (
                    f"Intermittent p99 spikes on `{svc}` over the last {win_mins}m (suspected `{filt}` {sev} events). "
                    f"Inspect the attached dashboard and invoke the right SRE tool."
                )
                conflict_prompt = (
                    f"[AUTOMATED NOTE - UNVERIFIED] Escalation bot warns of a possible deployment regression on `{svc}` "
                    f"(window={win_mins}m, severity={sev}, filter='{filt}'). Trust the attached dashboard as ground truth "
                    f"and execute the right SRE tool."
                )

            else:  # create_incident
                affected = [
                    OOD_SERVICES[(i + 0) % len(OOD_SERVICES)],
                    OOD_SERVICES[(i + 1) % len(OOD_SERVICES)],
                    OOD_SERVICES[(i + 2) % len(OOD_SERVICES)],
                ]
                sev = "SEV1" if i % 3 != 2 else "SEV2"
                expected_args = {
                    "severity": sev,
                    "affected_services": affected,
                    "summary": f"Cascading {sev} outage across {', '.join(affected)} with critical 5xx and latency breach",
                    "page_oncall": True,
                }
                meta = {
                    "affected_services": affected,
                    "banner_title": f"CRITICAL SLO BREACH ({sev}) | Cascading Multi-Service Outage",
                    "banner_sub": f"Affected: {', '.join(affected)} | Region: {region} | page_oncall=true",
                }
                normal_prompt = (
                    f"Multi-service SLO alert ({sev}) impacting {', '.join(affected)} in `{region}`. "
                    f"Inspect the attached dashboard and execute the appropriate SRE tool."
                )
                conflict_prompt = (
                    f"[AUTOMATED NOTE - UNVERIFIED] Minor single-service warning logged for `{affected[0]}` ({sev}, "
                    f"neighbors: {', '.join(affected[1:])} in {region}); bot suggests querying logs first. "
                    f"Trust the attached telemetry dashboard as ground truth and execute the right SRE tool."
                )

            if subset == "pixel_only":
                prompt = "Inspect the attached telemetry dashboard and execute the appropriate remediation tool."
            elif subset == "conflict_trap":
                prompt = conflict_prompt
            else:
                prompt = normal_prompt

            img_name = f"{item_id}_{subset}_{tool}.png"
            render_ood_dashboard(
                out_path=samples_dir / img_name,
                scenario=tool,
                meta=meta,
                layout=layout,
                light_mode=light_mode,
                rng=np_rng,
            )

            records.append(
                {
                    "id": item_id,
                    "subset": subset,
                    "layout": layout,
                    "theme": "light" if light_mode else "dark_alt",
                    "prompt": prompt,
                    "image_path": f"samples_ood/{img_name}",
                    "expected_tool": tool,
                    "expected_args": expected_args,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate 50 Hard OOD Visual SRE stress-test prompts.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    recs = generate_ood_dataset(args.output, args.samples_dir, args.seed)
    by_subset: dict[str, int] = {}
    by_tool: dict[str, int] = {}
    for r in recs:
        by_subset[r["subset"]] = by_subset.get(r["subset"], 0) + 1
        by_tool[r["expected_tool"]] = by_tool.get(r["expected_tool"], 0) + 1
    print(f"Generated {len(recs)} OOD Visual SRE dashboards -> {args.output}")
    print(f"Subsets : {by_subset}")
    print(f"Tools   : {by_tool}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
