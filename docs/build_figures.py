"""Generate high-DPI (300 DPI) figures for the Dev Diary PDF and arXiv LaTeX paper."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_dev_diary_progression_chart() -> Path:
    """Crisp progression chart (0.0% -> 73.3% -> 86.7%) for the Dev Diary PDF."""
    stages = ["Stock 0.8B\n(1-Loop)", "Stock 0.8B\n(2-Loop)", "Looped SFT\n(2-Loop LoRA)", "Looped GRPO\n(2-Loop RLVR)"]
    acc = [0.0, 0.0, 73.3, 86.7]
    trunc = [100.0, 100.0, 0.0, 0.0]

    x = np.arange(len(stages))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.6, 3.6), dpi=300, facecolor="white")
    ax.set_facecolor("#FAFBFD")

    bars_acc = ax.bar(x - width / 2, acc, width, label="Tool Accuracy (%)", color="#1D4ED8", edgecolor="#1E3A8A", linewidth=0.8, zorder=3)
    bars_tr = ax.bar(x + width / 2, trunc, width, label="<think> Truncation Rate (%)", color="#EF4444", alpha=0.82, edgecolor="#991B1B", linewidth=0.8, zorder=3)

    for b, v in zip(bars_acc, acc):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1E3A8A")
    for b, v in zip(bars_tr, trunc):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#991B1B")

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9, color="#1F2937")
    ax.set_ylabel("Percentage (%)", fontsize=9.5, fontweight="bold", color="#1F2937")
    ax.set_ylim(0, 118)
    ax.set_title("Visual SRE Held-Out Progression (15 Unseen Prompts, Qwen3.5-0.8B)", fontsize=10.5, fontweight="bold", color="#111827", pad=16)
    ax.grid(axis="y", color="#E5E7EB", linestyle="--", linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#D1D5DB", fontsize=8.5)

    out = FIG_DIR / "progression_chart.png"
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


def build_paper_ablation_and_kv_figure() -> Path:
    """Publication-grade Figure 2: Held-out stage ablation + 2064-token TurboQuant KV compression."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0), dpi=300, facecolor="white")

    # Left panel: Accuracy & Truncation across stages
    stages = ["Stock\n(L=1)", "Stock\n(L=2)", "Looped\nSFT", "Looped\nGRPO", "Merged\n+ TQ3"]
    acc = [0.0, 0.0, 73.3, 86.7, 33.3]
    trunc = [100.0, 100.0, 0.0, 0.0, 6.7]

    x = np.arange(len(stages))
    w = 0.36
    b1 = ax1.bar(x - w / 2, acc, w, label="Tool Accuracy (%)", color="#1D4ED8", edgecolor="#1E3A8A", linewidth=0.7, zorder=3)
    b2 = ax1.bar(x + w / 2, trunc, w, label="Truncation Rate (%)", color="#DC2626", alpha=0.8, edgecolor="#7F1D1D", linewidth=0.7, zorder=3)

    for b, v in zip(b1, acc):
        ax1.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1E3A8A")
    for b, v in zip(b2, trunc):
        ax1.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#7F1D1D")

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=8.2)
    ax1.set_ylabel("Percentage (%)", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 118)
    ax1.set_title("(a) Held-Out Visual SRE Ablation (n = 15)", fontsize=9.8, fontweight="bold", pad=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", fontsize=7.8, frameon=True)

    # Right panel: Stacked Attention KV vs Untouched DeltaNet states (2064 tokens, L=2)
    modes = ["Uncompressed\n(bf16)", "TurboQuant\n3-bit", "TurboQuant\n4-bit"]
    attn_mb = np.array([48.38, 10.56, 13.49])
    delta_mb = np.array([37.69, 37.69, 37.69])
    ratios = ["1.00x", "4.58x Attn", "3.59x Attn"]

    x2 = np.arange(len(modes))
    w2 = 0.48
    ax2.bar(x2, delta_mb, w2, label="18x DeltaNet States (Native bf16/fp32)", color="#64748B", edgecolor="#334155", linewidth=0.7, zorder=3)
    ax2.bar(x2, attn_mb, w2, bottom=delta_mb, label="6x Gated Attention KV Cache", color="#0EA5E9", edgecolor="#0369A1", linewidth=0.7, zorder=3)

    for i, (a, d, r) in enumerate(zip(attn_mb, delta_mb, ratios)):
        ax2.text(i, d / 2, f"{d:.1f} MB", ha="center", va="center", fontsize=7.8, color="white", fontweight="bold", zorder=4)
        ax2.text(i, d + a / 2, f"{a:.1f} MB", ha="center", va="center", fontsize=7.6, color="white", fontweight="bold", zorder=4)
        ax2.text(i, d + a + 2.2, f"Total: {d + a:.1f} MB\n({r})", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#0F172A")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(modes, fontsize=8.5)
    ax2.set_ylabel("Cache Memory (MB)", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.set_title("(b) Hybrid KV Cache at 2064 Tokens (num_loops = 2)", fontsize=9.8, fontweight="bold", pad=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", fontsize=7.5, frameon=True)

    out = FIG_DIR / "ablation_and_kv_figure.png"
    fig.tight_layout(w_pad=2.4)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


def build_paper_sre_trace_figure() -> Path:
    """Publication-grade Figure 1: Real dark-mode Grafana dashboard + verbatim GRPO trace."""
    img_path = ROOT / "cot_dataset_builder" / "samples" / "sre_042_incident.png"
    img = mpimg.imread(str(img_path))

    rep = json.loads((ROOT / "looped_qwen35" / "eval_results" / "grpo_sre.json").read_text(encoding="utf-8"))
    sample = next(s for s in rep["samples"] if s["id"] == "sre_042")
    completion = sample["completion"].strip()

    fig = plt.figure(figsize=(10.2, 4.3), dpi=300, facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.08)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title("(a) Input Telemetry Dashboard (sre_042_incident.png)", fontsize=9.2, fontweight="bold", pad=6, color="#111827")

    ax_txt = fig.add_subplot(gs[0, 1])
    ax_txt.set_facecolor("#F8FAFC")
    ax_txt.set_xticks([])
    ax_txt.set_yticks([])
    for sp in ax_txt.spines.values():
        sp.set_color("#CBD5E1")
        sp.set_linewidth(1.0)
    ax_txt.set_title("(b) Looped GRPO Output (Held-Out, Reward = 1.00)", fontsize=9.2, fontweight="bold", pad=6, color="#111827")

    # Format verbatim completion cleanly wrapped for readability
    parts = completion.split("</think>")
    think_body = parts[0].strip()
    json_body = parts[1].strip() if len(parts) > 1 else ""
    try:
        json_pretty = json.dumps(json.loads(json_body), indent=2)
    except Exception:
        json_pretty = json_body

    import textwrap
    wrapped_think = "\n".join(textwrap.wrap(think_body, width=50))
    wrapped_json_lines = []
    for jline in json_pretty.splitlines():
        if len(jline) <= 50:
            wrapped_json_lines.append(jline)
        else:
            indent = " " * (len(jline) - len(jline.lstrip(" ")) + 2)
            wrapped_json_lines.extend(
                textwrap.wrap(jline, width=50, subsequent_indent=indent)
            )
    display_text = (
        "<think>\n"
        f"{wrapped_think}\n"
        "</think>\n\n"
        + "\n".join(wrapped_json_lines)
    )

    ax_txt.text(
        0.04,
        0.95,
        display_text,
        transform=ax_txt.transAxes,
        va="top",
        ha="left",
        fontsize=7.3,
        family="monospace",
        color="#0F172A",
        linespacing=1.32,
    )

    out = FIG_DIR / "sre_trace_figure.png"
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p1 = build_dev_diary_progression_chart()
    p2 = build_paper_ablation_and_kv_figure()
    p3 = build_paper_sre_trace_figure()
    print("Generated figures:")
    for p in (p1, p2, p3):
        print(" -", p)
