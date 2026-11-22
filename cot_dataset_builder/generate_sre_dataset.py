"""Generate 50 Visual SRE / DevOps Incident Triage prompts + dark-mode Grafana dashboards."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_JSONL = BASE_DIR / "niche_prompts.jsonl"
DEFAULT_SAMPLES_DIR = BASE_DIR / "samples"

SERVICES = ["checkout-api", "auth-gateway", "payments-worker", "search-indexer", "orders-db"]
REGIONS = ["us-east-1", "eu-west-1", "ap-south-1"]
NAMESPACES = ["prod-core", "prod-payments", "prod-edge", "prod-search", "prod-data"]
ENVIRONMENTS = ["prod", "staging"]

BG_FIG = "#0B0C10"
BG_PANEL = "#141619"
GRID_COLOR = "#252A34"
TEXT_MAIN = "#E6EDF3"
TEXT_MUTED = "#8B949E"


def _style_ax(ax: Any, title: str, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color=TEXT_MAIN, fontsize=9.5, fontweight="bold", loc="left", pad=5)
    ax.set_ylabel(ylabel, color=TEXT_MUTED, fontsize=8)
    ax.set_xlabel("Time (min)", color=TEXT_MUTED, fontsize=7.5)
    ax.tick_params(colors=TEXT_MUTED, labelsize=7.5)
    ax.grid(True, color=GRID_COLOR, linestyle="-", linewidth=0.6, alpha=0.85)
    for spine in ax.spines.values():
        spine.set_color("#2C323E")
    if ylim is not None:
        ax.set_ylim(*ylim)


def render_grafana_dashboard(
    out_path: Path,
    scenario: str,
    meta: dict[str, Any],
    rng: np.random.Generator,
) -> None:
    """Render a crisp 2x2 dark-mode Grafana telemetry dashboard PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 60, 120)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), dpi=100, facecolor=BG_FIG)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.10, wspace=0.24, hspace=0.38)

    header_bg = "#3D1418" if scenario == "create_incident" else "#161B22"
    header_fg = "#FF7B72" if scenario == "create_incident" else "#58A6FF"
    fig.text(
        0.02,
        0.94,
        meta["banner_title"],
        color=header_fg,
        fontsize=10.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=header_bg, edgecolor="#30363D"),
    )
    fig.text(0.02, 0.875, meta["banner_sub"], color=TEXT_MUTED, fontsize=8.5)

    ax_cpu, ax_mem, ax_5xx, ax_lat = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    if scenario == "trigger_rollback":
        deploy_t = 30.0
        post = t >= deploy_t
        cpu = 42 + rng.normal(0, 2.5, len(t)) + post * 8.0
        mem = 54 + rng.normal(0, 1.5, len(t)) + post * 4.0
        err_5xx = np.where(post, rng.uniform(11.0, 24.0, len(t)), rng.uniform(0.05, 0.25, len(t)))
        p99 = np.where(post, rng.uniform(920, 1480, len(t)), rng.uniform(42, 68, len(t)))

        ax_cpu.plot(t, cpu, color="#3FB950", linewidth=1.6)
        ax_mem.plot(t, mem, color="#58A6FF", linewidth=1.6)
        ax_5xx.plot(t, err_5xx, color="#F85149", linewidth=1.8)
        ax_lat.plot(t, p99, color="#D29922", linewidth=1.8)

        for ax in (ax_cpu, ax_mem, ax_5xx, ax_lat):
            ax.axvline(deploy_t, color="#F0C674", linestyle="--", linewidth=1.4)
        ax_5xx.text(deploy_t + 1.2, 25.5, f"Deploy {meta['new_version']}", color="#F0C674", fontsize=7.5, fontweight="bold")
        ax_cpu.axhline(85, color="#F85149", linestyle=":", linewidth=1.0)
        ax_5xx.axhline(1.0, color="#FFA657", linestyle=":", linewidth=1.0)

        _style_ax(ax_cpu, "CPU Utilization (%)", "%", (0, 100))
        _style_ax(ax_mem, "Memory Usage (%)", "%", (0, 100))
        _style_ax(ax_5xx, "HTTP 5xx Error Rate (%)", "%", (0, 30))
        _style_ax(ax_lat, "p99 Latency (ms)", "ms", (0, 1700))

    elif scenario == "scale_service":
        ramp = 1.0 / (1.0 + np.exp(-(t - 26.0) / 6.0))
        cpu = 36 + 58 * ramp + rng.normal(0, 1.8, len(t))
        mem = 50 + 8 * ramp + rng.normal(0, 1.2, len(t))
        err_5xx = 0.15 + 0.25 * ramp + rng.normal(0, 0.04, len(t))
        rps = 1100 + 3600 * ramp + rng.normal(0, 90, len(t))

        ax_cpu.plot(t, np.clip(cpu, 0, 100), color="#F85149", linewidth=1.8)
        ax_cpu.axhline(85, color="#FFA657", linestyle="--", linewidth=1.1)
        ax_cpu.text(2, 87.5, "85% Saturation Threshold", color="#FFA657", fontsize=7)
        ax_mem.plot(t, mem, color="#58A6FF", linewidth=1.6)
        ax_5xx.plot(t, np.clip(err_5xx, 0, 5), color="#3FB950", linewidth=1.6)
        ax_5xx.axhline(1.0, color="#FFA657", linestyle=":", linewidth=1.0)
        ax_lat.plot(t, rps, color="#A371F7", linewidth=1.8)

        _style_ax(ax_cpu, "CPU Utilization (%)", "%", (0, 100))
        _style_ax(ax_mem, "Memory Usage (%)", "%", (0, 100))
        _style_ax(ax_5xx, "HTTP 5xx Rate (Healthy <1%)", "%", (0, 5))
        _style_ax(ax_lat, "Incoming Traffic (RPS Surge)", "req/s", (0, 5500))

    elif scenario == "restart_pod_pool":
        period = 15.0
        saw = (t % period) / period
        mem = 34 + 66 * saw + rng.normal(0, 1.2, len(t))
        cpu = 46 + 10 * saw + rng.normal(0, 2.0, len(t))
        oom_times = [15.0, 30.0, 45.0, 60.0]
        err_5xx = 0.2 + rng.normal(0, 0.05, len(t))
        for ot in oom_times:
            err_5xx += 2.2 * np.exp(-0.5 * ((t - ot) / 0.8) ** 2)
        p99 = 65 + 18 * saw + rng.normal(0, 5, len(t))

        ax_cpu.plot(t, cpu, color="#3FB950", linewidth=1.6)
        ax_mem.plot(t, np.clip(mem, 0, 100), color="#F85149", linewidth=1.8)
        ax_mem.axhline(95, color="#FFA657", linestyle="--", linewidth=1.0)
        for ot in oom_times[:-1]:
            ax_mem.scatter([ot], [99], color="#FF7B72", s=26, zorder=5)
            ax_mem.text(ot - 3.5, 86, "OOMKill", color="#FF7B72", fontsize=6.8, fontweight="bold")
        ax_5xx.plot(t, np.clip(err_5xx, 0, 10), color="#D29922", linewidth=1.6)
        ax_lat.plot(t, p99, color="#58A6FF", linewidth=1.6)

        _style_ax(ax_cpu, "CPU Utilization (%)", "%", (0, 100))
        _style_ax(ax_mem, "Memory Usage (OOMKill Sawtooth)", "%", (0, 105))
        _style_ax(ax_5xx, "HTTP 5xx Rate (%)", "%", (0, 10))
        _style_ax(ax_lat, "p99 Latency (ms)", "ms", (0, 250))

    elif scenario == "query_logs":
        cpu = 44 + rng.normal(0, 3.0, len(t))
        mem = 52 + rng.normal(0, 1.5, len(t))
        err_5xx = 0.35 + rng.normal(0, 0.12, len(t))
        p99 = 85 + rng.normal(0, 12, len(t))
        for spike_t in (11.0, 24.0, 39.0, 51.0):
            p99 += rng.uniform(260, 420) * np.exp(-0.5 * ((t - spike_t) / 1.4) ** 2)

        ax_cpu.plot(t, cpu, color="#3FB950", linewidth=1.6)
        ax_mem.plot(t, mem, color="#58A6FF", linewidth=1.6)
        ax_5xx.plot(t, np.clip(err_5xx, 0, 5), color="#3FB950", linewidth=1.6)
        ax_5xx.axhline(1.0, color="#FFA657", linestyle=":", linewidth=1.0)
        ax_lat.plot(t, p99, color="#D29922", linewidth=1.8)
        ax_lat.text(4, 520, "Intermittent Downstream Jitter", color="#D29922", fontsize=7.5)

        _style_ax(ax_cpu, "CPU Utilization (%)", "%", (0, 100))
        _style_ax(ax_mem, "Memory Usage (%)", "%", (0, 100))
        _style_ax(ax_5xx, "HTTP 5xx Rate (%)", "%", (0, 5))
        _style_ax(ax_lat, "p99 Latency (Intermittent Bursts)", "ms", (0, 620))

    else:  # create_incident (multi-service cascading outage)
        onset = 22.0
        post = 1.0 / (1.0 + np.exp(-(t - onset) / 2.2))
        colors = ["#F85149", "#FFA657", "#A371F7"]
        svcs = meta["affected_services"][:3]
        for idx, (svc_name, col) in enumerate(zip(svcs, colors)):
            shift = idx * 1.5
            p_curve = 1.0 / (1.0 + np.exp(-(t - (onset + shift)) / 2.0))
            ax_cpu.plot(t, 45 + 48 * p_curve + rng.normal(0, 1.5, len(t)), color=col, linewidth=1.5, label=svc_name)
            ax_mem.plot(t, 52 + 34 * p_curve + rng.normal(0, 1.5, len(t)), color=col, linewidth=1.5)
            ax_5xx.plot(t, 0.2 + (22 + idx * 6) * p_curve + rng.normal(0, 0.8, len(t)), color=col, linewidth=1.7)
            ax_lat.plot(t, 60 + (1900 + idx * 350) * p_curve + rng.normal(0, 40, len(t)), color=col, linewidth=1.7)

        ax_cpu.legend(loc="upper left", fontsize=6.5, facecolor=BG_PANEL, edgecolor=GRID_COLOR, labelcolor=TEXT_MAIN)
        ax_5xx.axhline(5.0, color="#FF7B72", linestyle="--", linewidth=1.0)
        ax_5xx.text(2, 34, "CRITICAL SLO BREACH", color="#FF7B72", fontsize=7.5, fontweight="bold")

        _style_ax(ax_cpu, "Multi-Service CPU (%)", "%", (0, 100))
        _style_ax(ax_mem, "Multi-Service Memory (%)", "%", (0, 100))
        _style_ax(ax_5xx, "Cascading 5xx Error Rate (%)", "%", (0, 42))
        _style_ax(ax_lat, "Cascading p99 Latency (ms)", "ms", (0, 3100))

    fig.savefig(out_path, dpi=100, facecolor=BG_FIG)
    plt.close(fig)


def generate_sre_dataset(
    count: int = 50,
    output_path: Path = DEFAULT_OUTPUT_JSONL,
    samples_dir: Path = DEFAULT_SAMPLES_DIR,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate `count` balanced Visual SRE prompts (80% multimodal Grafana PNGs, 20% text alerts)."""
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    tools = [
        "trigger_rollback",
        "scale_service",
        "restart_pod_pool",
        "query_logs",
        "create_incident",
    ]
    per_tool = count // len(tools)
    remainder = count % len(tools)

    records: list[dict[str, Any]] = []
    global_idx = 0

    for t_idx, tool in enumerate(tools):
        n_for_tool = per_tool + (1 if t_idx < remainder else 0)
        for i in range(n_for_tool):
            global_idx += 1
            item_id = f"sre_{global_idx:03d}"
            # 80% multimodal (8 of 10), 20% text-only alert logs (2 of 10)
            is_multimodal = (i % 5) != 4

            svc = SERVICES[(i + t_idx) % len(SERVICES)]
            region = REGIONS[(i + t_idx) % len(REGIONS)]
            ns = NAMESPACES[(i + t_idx) % len(NAMESPACES)]
            env = ENVIRONMENTS[0 if i % 4 != 3 else 1]

            img_rel: str | None = None
            meta: dict[str, Any] = {}

            if tool == "trigger_rollback":
                minor = 4 + (i % 6)
                prev_ver = f"v2.{minor}.0"
                new_ver = f"v2.{minor + 1}.0"
                expected_args = {
                    "service": svc,
                    "target_version": prev_ver,
                    "environment": env,
                    "reason": f"HTTP 5xx and p99 latency spiked immediately after {new_ver} deployment",
                }
                meta = {
                    "new_version": new_ver,
                    "banner_title": f"GRAFANA | {svc} ({env}) | Current: {new_ver} (prev: {prev_ver})",
                    "banner_sub": f"Region: {region} | Namespace: {ns} | Deploy {new_ver} at t=30m | Replicas: 8/8",
                }
                if is_multimodal:
                    img_rel = f"samples/{item_id}_rollback.png"
                    render_grafana_dashboard(samples_dir / f"{item_id}_rollback.png", tool, meta, np_rng)
                    prompt = (
                        f"Alert triggered on `{svc}` ({env}, {region}). Inspect the attached Grafana dashboard "
                        f"(previous stable release was `{prev_ver}`, rolled out `{new_ver}` at t=30m) and execute "
                        f"the single best remediation tool."
                    )
                else:
                    prompt = (
                        f"[ALERT-LOG] service={svc} env={env} region={region} | deploy_event={new_ver} (prev={prev_ver}) "
                        f"at 14:30 UTC | 14:31 UTC http_5xx jumped from 0.12% to 19.4%, p99_latency jumped from 52ms to 1340ms; "
                        f"cpu=48%, mem=55%. Choose and invoke the correct SRE remediation tool."
                    )

            elif tool == "scale_service":
                cur_rep = 6 + (i % 4) * 2
                target_rep = cur_rep * 2
                expected_args = {
                    "service": svc,
                    "target_replicas": target_rep,
                    "region": region,
                    "reason": f"Organic RPS surge saturated CPU above 85% with healthy 5xx (<1%); scaling {cur_rep}->{target_rep}",
                }
                meta = {
                    "banner_title": f"GRAFANA | {svc} ({region}) | Replicas: {cur_rep} (HPA max: {target_rep})",
                    "banner_sub": f"Env: prod | No recent deploys | Traffic surge detected | Target scale: {target_rep} replicas",
                }
                if is_multimodal:
                    img_rel = f"samples/{item_id}_scale.png"
                    render_grafana_dashboard(samples_dir / f"{item_id}_scale.png", tool, meta, np_rng)
                    prompt = (
                        f"High CPU alert on `{svc}` in `{region}` (currently {cur_rep} replicas; recommended capacity "
                        f"is {target_rep} replicas). Inspect the attached Grafana dashboard and invoke the right SRE tool."
                    )
                else:
                    prompt = (
                        f"[ALERT-LOG] service={svc} region={region} replicas={cur_rep} | no deployments in 72h | "
                        f"rps surged 1150 -> 4650 req/s, cpu=93.4%, mem=57%, http_5xx=0.28% (healthy). "
                        f"Recommended target_replicas={target_rep}. Choose and invoke the correct SRE tool."
                    )

            elif tool == "restart_pod_pool":
                strategy = "rolling" if i % 3 != 2 else "immediate"
                restarts = 8 + (i % 7) * 2
                expected_args = {
                    "service": svc,
                    "namespace": ns,
                    "strategy": strategy,
                    "reason": f"Memory leak OOMKill sawtooth reaching 100% RAM with {restarts} pod restarts",
                }
                meta = {
                    "banner_title": f"GRAFANA | {svc} | Namespace: {ns} | Pod Restarts: {restarts}",
                    "banner_sub": f"Region: {region} | Strategy policy: {strategy} | No recent deploys",
                }
                if is_multimodal:
                    img_rel = f"samples/{item_id}_oom.png"
                    render_grafana_dashboard(samples_dir / f"{item_id}_oom.png", tool, meta, np_rng)
                    prompt = (
                        f"Kubernetes pod health alert on `{svc}` in namespace `{ns}` (restart policy: `{strategy}`). "
                        f"Inspect the attached Grafana dashboard and invoke the single best SRE tool."
                    )
                else:
                    prompt = (
                        f"[ALERT-LOG] service={svc} namespace={ns} | pod_restarts={restarts} (ExitCode=137 OOMKilled) | "
                        f"memory climbs linearly 35% -> 100% every 15m in a sawtooth pattern; cpu=51%, 5xx=0.4%. "
                        f"Runbook specifies strategy='{strategy}'. Invoke the right SRE tool."
                    )

            elif tool == "query_logs":
                win_mins = [15, 30, 45, 60][i % 4]
                sev = ["ERROR", "WARN", "FATAL"][i % 3]
                filt = ["upstream_timeout", "connection_reset", "pool_exhausted", "tls_handshake_retry"][i % 4]
                expected_args = {
                    "service": svc,
                    "time_window_minutes": win_mins,
                    "severity": sev,
                    "filter_query": filt,
                }
                meta = {
                    "banner_title": f"GRAFANA | {svc} ({region}) | Intermittent Latency Bursts",
                    "banner_sub": f"Window: {win_mins}m | Suggested log filter: severity={sev} query='{filt}' | No deploys",
                }
                if is_multimodal:
                    img_rel = f"samples/{item_id}_logs.png"
                    render_grafana_dashboard(samples_dir / f"{item_id}_logs.png", tool, meta, np_rng)
                    prompt = (
                        f"Intermittent latency warning on `{svc}` over the past {win_mins} minutes "
                        f"(suspected `{filt}` {sev} events). Inspect the attached Grafana dashboard and choose the right SRE tool."
                    )
                else:
                    prompt = (
                        f"[ALERT-LOG] service={svc} window={win_mins}m | cpu=43%, mem=51%, http_5xx=0.38% (within SLO), "
                        f"no deploys, but sporadic p99 latency bursts (85ms -> 470ms) every ~12m. Root cause unconfirmed; "
                        f"check {sev} logs matching '{filt}'. Invoke the right SRE tool."
                    )

            else:  # create_incident
                affected = [
                    SERVICES[(i + 0) % len(SERVICES)],
                    SERVICES[(i + 1) % len(SERVICES)],
                    SERVICES[(i + 2) % len(SERVICES)],
                ]
                sev = "SEV1" if i % 3 != 2 else "SEV2"
                page = True
                expected_args = {
                    "severity": sev,
                    "affected_services": affected,
                    "summary": f"Cascading {sev} outage across {', '.join(affected)} with critical 5xx and latency SLO breach",
                    "page_oncall": page,
                }
                meta = {
                    "affected_services": affected,
                    "banner_title": f"CRITICAL SLO BREACH ({sev}) | Cascading Multi-Service Outage",
                    "banner_sub": f"Affected: {', '.join(affected)} | Region: {region} | Page On-Call Required",
                }
                if is_multimodal:
                    img_rel = f"samples/{item_id}_incident.png"
                    render_grafana_dashboard(samples_dir / f"{item_id}_incident.png", tool, meta, np_rng)
                    prompt = (
                        f"Global SLO burn alert ({sev}) affecting {', '.join(affected)} in `{region}`. "
                        f"Inspect the attached multi-service Grafana dashboard and invoke the appropriate SRE tool."
                    )
                else:
                    prompt = (
                        f"[CRITICAL-ALERT] severity={sev} region={region} | affected_services={affected} | "
                        f"all three services redlining simultaneously: http_5xx=24-38%, p99_latency=2100-2850ms, "
                        f"cascading SLO breach requiring immediate on-call paging. Invoke the right SRE tool."
                    )

            records.append(
                {
                    "id": item_id,
                    "prompt": prompt,
                    "image_path": img_rel,
                    "expected_tool": tool,
                    "expected_args": expected_args,
                }
            )

    py_rng.shuffle(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Visual SRE Grafana dashboards and prompts.")
    parser.add_argument("--count", type=int, default=50, help="Number of SRE prompts to generate (default: 50)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path")
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR, help="Directory for Grafana PNGs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recs = generate_sre_dataset(
        count=args.count,
        output_path=args.output,
        samples_dir=args.samples_dir,
        seed=args.seed,
    )
    n_mm = sum(1 for r in recs if r.get("image_path"))
    by_tool: dict[str, int] = {}
    for r in recs:
        by_tool[r["expected_tool"]] = by_tool.get(r["expected_tool"], 0) + 1
    print(f"Generated {len(recs)} Visual SRE prompts ({n_mm} multimodal Grafana PNGs, {len(recs) - n_mm} text alerts)")
    print(f"Tool balance: {by_tool}")
    print(f"Saved JSONL : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
