"""Build the polished 3-4 page Dev Diary PDF (`docs/Visual_SRE_Looped_Qwen35_Dev_Diary.pdf`)."""

from __future__ import annotations

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
OUT_PDF = DOCS_DIR / "Visual_SRE_Looped_Qwen35_Dev_Diary.pdf"

NAVY = colors.HexColor("#0F172A")
BLUE = colors.HexColor("#1D4ED8")
SLATE = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748B")
LIGHT_BG = colors.HexColor("#F8FAFC")
CALLOUT_BG = colors.HexColor("#EFF6FF")
BORDER = colors.HexColor("#CBD5E1")
ACCENT_RED = colors.HexColor("#DC2626")
ACCENT_GREEN = colors.HexColor("#15803D")


def _assert_no_em_dash(text: str) -> str:
    if "\u2014" in text or "\u2013" in text:
        raise ValueError(f"Found forbidden dash character in prose: {text[:120]!r}")
    return text


def _draw_header_footer(canvas, doc) -> None:
    canvas.saveState()
    w, h = LETTER
    if doc.page > 1:
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(0.75 * inch, h - 0.55 * inch, w - 0.75 * inch, h - 0.55 * inch)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.75 * inch, h - 0.46 * inch, "Dev Diary: Pushing Qwen3.5-0.8B to 86.7% on Visual SRE Triage")
        canvas.drawRightString(w - 0.75 * inch, h - 0.46 * inch, "Abhishek Satarkar")

    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(0.75 * inch, 0.55 * inch, w - 0.75 * inch, 0.55 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.75 * inch, 0.38 * inch, "Independent Research & Engineering Log  |  November 2026")
    canvas.drawRightString(w - 0.75 * inch, 0.38 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf() -> Path:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.72 * inch,
        title="Dev Diary: Visual SRE Looped Hybrid Qwen3.5",
        author="Abhishek Satarkar",
    )

    styles = getSampleStyleSheet()
    kicker_style = ParagraphStyle(
        "Kicker",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        textColor=BLUE,
        spaceAfter=4,
        leading=11,
    )
    title_style = ParagraphStyle(
        "DiaryTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=NAVY,
        alignment=TA_LEFT,
        leading=24,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DiarySubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        textColor=SLATE,
        leading=14.5,
        spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "Meta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        textColor=MUTED,
        spaceAfter=10,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=5,
        leading=16,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.6,
        textColor=colors.HexColor("#1E293B"),
        leading=14.0,
        spaceAfter=7,
        alignment=TA_JUSTIFY,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-8,
        spaceAfter=4.5,
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=body_style,
        fontName="Helvetica",
        fontSize=9.2,
        textColor=NAVY,
        leading=13.5,
        spaceAfter=0,
        alignment=TA_LEFT,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.2,
        textColor=MUTED,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=10,
        leading=11,
    )

    def P(text: str, style=body_style) -> Paragraph:
        return Paragraph(_assert_no_em_dash(text), style)

    def make_callout(title_text: str, body_text: str) -> Table:
        content = P(f"<b>{title_text}</b><br/>{body_text}", callout_style)
        tbl = Table([[content]], colWidths=[7.0 * inch])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), CALLOUT_BG),
                    ("LINEBEFORE", (0, 0), (0, -1), 3.0, BLUE),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 11),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        return tbl

    story = []

    # ------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------
    story.append(P("BUILD-IN-PUBLIC ENGINEERING LOG  |  MULTIMODAL SMALL LANGUAGE MODELS", kicker_style))
    story.append(P("Teaching a 0.8B Hybrid Model to Loop, Reason, and Triage Live Grafana Dashboards on an 8 GB Laptop GPU", title_style))
    story.append(
        P(
            "How I wrapped Qwen3.5-0.8B into a Looped Hybrid Transformer, wrote pure-PyTorch SFT and GRPO loops from scratch, "
            "tested Weight-Space Programming, and built hybrid-safe 3-bit TurboQuant KV compression without ever leaving my local RTX 4060.",
            subtitle_style,
        )
    )
    story.append(P("By <b>Abhishek Satarkar</b> (Independent)  |  November 2026  |  Hardware: NVIDIA RTX 4060 Laptop GPU (8.00 GB VRAM)", meta_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=BLUE, spaceAfter=10))

    # Hero KPI strip
    kpi_data = [
        [
            P("<b>0.0% -> 86.7%</b><br/><font size='7.5' color='#64748B'>Held-Out SRE Tool Acc</font>", ParagraphStyle("K", alignment=TA_CENTER, leading=12, fontSize=10.5, textColor=BLUE)),
            P("<b>100% -> 0.0%</b><br/><font size='7.5' color='#64748B'>&lt;think&gt; Truncation Rate</font>", ParagraphStyle("K", alignment=TA_CENTER, leading=12, fontSize=10.5, textColor=ACCENT_GREEN)),
            P("<b>3.37 GB Peak</b><br/><font size='7.5' color='#64748B'>GRPO RLVR VRAM (8 GB GPU)</font>", ParagraphStyle("K", alignment=TA_CENTER, leading=12, fontSize=10.5, textColor=NAVY)),
            P("<b>4.58x Smaller</b><br/><font size='7.5' color='#64748B'>TurboQuant 3-bit Attn KV</font>", ParagraphStyle("K", alignment=TA_CENTER, leading=12, fontSize=10.5, textColor=BLUE)),
        ]
    ]
    kpi_tbl = Table(kpi_data, colWidths=[1.75 * inch] * 4)
    kpi_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
                ("BOX", (0, 0), (-1, -1), 0.6, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.append(kpi_tbl)
    story.append(Spacer(1, 10))

    # ------------------------------------------------------------------
    # SECTION 1: VISION & PROBLEM STATEMENT
    # ------------------------------------------------------------------
    story.append(P("1. The Late-Night Bet: Pushing a 0.8B Model Past Its Weight Class", h2_style))
    story.append(
        P(
            "Most multimodal agent demos assume you have a frontier API key or a cluster of H100s. I wanted to see what happens "
            "when you flip the constraint: one consumer 8 GB RTX 4060 laptop GPU, one tiny 0.8B multimodal model (<b>Qwen3.5-0.8B</b>), "
            "and a genuinely visual, high-stakes ops domain. I picked <b>Visual SRE / DevOps Incident Triage</b>: looking at a dark-mode "
            "Grafana telemetry dashboard (CPU, Memory, HTTP 5xx, p99 latency, deployment markers) and deciding which exact remediation "
            "tool to fire (<code>trigger_rollback</code>, <code>scale_service</code>, <code>restart_pod_pool</code>, <code>query_logs</code>, "
            "or <code>create_incident</code>) with schema-valid JSON arguments."
        )
    )
    story.append(
        P(
            "Why this niche? Because text-only alert logs hide the story, whereas a dashboard makes the root cause visually obvious: "
            "a 5xx cliff right after a dashed yellow deploy line means rollback, an organic RPS ramp saturating CPU with healthy errors "
            "means scale out, and a 15-minute memory sawtooth hitting OOMKill means restart the pod pool. If a sub-1B model running "
            "locally can read those patterns, reason cleanly inside <code>&lt;think&gt;</code> tags, and emit strict JSON tool calls, "
            "that is a real systems win."
        )
    )

    # ------------------------------------------------------------------
    # SECTION 2: BRAINSTORMING WITH GEMINI & ARCHITECTURE DECISIONS
    # ------------------------------------------------------------------
    story.append(P("2. Brainstorming the Architecture (Back-and-Forth with Gemini)", h2_style))
    story.append(
        P(
            "Before writing a single line of PyTorch, I spent a late-night session bouncing architectural ideas back and forth with "
            "Gemini as a thought partner. Instead of just slapping a standard LoRA on Qwen3.5 and calling it a day, we mapped out "
            "four deliberate systems bets:"
        )
    )
    story.append(
        P(
            "<b>1. Looped Hybrid Transformer with Scalar Input Re-Injection:</b> Rather than adding parameters I could not fit in 8 GB VRAM, "
            "I wrapped Qwen3.5's 24 decoder layers non-destructively so hidden states pass through the stack <code>num_loops</code> times "
            "(effective depth 48 at <code>num_loops=2</code>). To keep visual patches grounded across loops, I added one learnable scalar "
            "<code>reinject_logit</code> initialized to <code>4.6</code> (so <code>w = sigmoid(4.6) = 0.990</code>) and blended "
            "<code>h = w * base_inputs_embeds + (1 - w) * h</code> after the native vision encoder merges image patches. At "
            "<code>num_loops=1</code>, the wrapper is bit-exact with stock Qwen3.5 (max logit diff <code>0.000000</code>).",
            bullet_style,
        )
    )
    story.append(
        P(
            "<b>2. Per-Loop Hybrid Cache Isolation:</b> Inspecting <code>transformers==5.24.0</code> source code revealed that Qwen3.5-0.8B "
            "mixes <b>18 Gated DeltaNet recurrent layers</b> with <b>6 Gated Attention layers</b> inside a unified <code>DynamicCache</code>. "
            "Sharing one cache across loops would feed the same token into every DeltaNet recurrent state twice per decode step and corrupt "
            "the recurrence. My <code>PerLoopHybridCache</code> gives each loop its own isolated hybrid cache.",
            bullet_style,
        )
    )
    story.append(
        P(
            "<b>3. Pure PyTorch Over Unsloth / TRL:</b> Wrapping custom recurrent caches and a trainable <code>reinject_logit</code> inside "
            "high-level trainers is a recipe for silent bugs. Writing clean pure-PyTorch SFT and GRPO loops gave me total control over "
            "assistant-only loss masking, non-reentrant gradient checkpointing, and micro-batched rollouts.",
            bullet_style,
        )
    )
    story.append(
        P(
            "<b>4. Testing Weight-Space Programming &amp; Hybrid-Safe TurboQuant:</b> We also debated MIT CSAIL's Weight-Space Programming "
            "(WSP) paper on Qwen3.5-0.8B (surgical directional ablation in MLP <code>down_proj</code> layers to kill infinite thinking loops) "
            "and TurboQuant 3-bit KV compression restricted strictly to the 6 Gated Attention layers so DeltaNet stays untouched.",
            bullet_style,
        )
    )
    story.append(Spacer(1, 4))

    # Visual: Two real dark-mode Grafana dashboards side-by-side
    img_rollback = Image(str(ROOT / "cot_dataset_builder" / "samples" / "sre_001_rollback.png"), width=3.42 * inch, height=2.28 * inch)
    img_incident = Image(str(ROOT / "cot_dataset_builder" / "samples" / "sre_042_incident.png"), width=3.42 * inch, height=2.28 * inch)
    img_tbl = Table([[img_rollback, img_incident]], colWidths=[3.5 * inch, 3.5 * inch])
    img_tbl.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    story.append(KeepTogether([
        img_tbl,
        P("Figure 1: Real dark-mode 2x2 Grafana telemetry dashboards generated for this project: (left) deploy-triggered 5xx spike requiring trigger_rollback; (right) multi-service cascading outage requiring create_incident.", caption_style),
    ]))

    # ------------------------------------------------------------------
    # SECTION 3: THE BUILD JOURNEY (PHASES 1 TO 7)
    # ------------------------------------------------------------------
    story.append(P("3. The Build Journey: From 0.0% Baseline Collapse to 86.7% Held-Out Accuracy", h2_style))
    story.append(
        P(
            "<b>Phases 1 &amp; 2 (Data Harvesting &amp; Leakage-Proof Splits):</b> I built <code>generate_sre_dataset.py</code> to render 50 "
            "balanced Visual SRE scenarios (40 dark-mode 2x2 Grafana PNGs + 10 text alerts, 10 per tool) and split them deterministically "
            "into 35 training prompts and 15 strictly isolated held-out eval prompts (<code>split_manifest.json</code> verified 0 overlapping "
            "IDs and 0 overlapping prompts). I then wrapped <code>opencode run --format json --thinking --pure</code> around DeepSeek "
            "(<code>opencode/deepseek-v4.1-flash-free</code>) with rejection sampling and 25% multi-turn error-recovery mode, harvesting 34 "
            "validated CoT traces (including 8 self-correction traces) strictly from the training split. Along the way I caught two real "
            "Windows systems bugs: <code>opencode.CMD</code> truncating multiline prompts at newlines (fixed by piping via <code>stdin</code>) "
            "and 13-digit Unix timestamps accidentally triggering my HTTP 429/503 throttle detector."
        )
    )
    story.append(
        P(
            "<b>The Stock Baseline Reality Check:</b> Before touching a single weight, I evaluated stock <code>Qwen3.5-0.8B</code> on the "
            "15 unseen SRE held-out prompts at both <code>num_loops=1</code> and <code>num_loops=2</code>. The result was brutal: "
            "<b>0.0% tool accuracy</b>, <b>-0.2000 mean reward</b>, and <b>100.0% truncation rate</b>. On every single held-out dashboard, "
            "the stock model got trapped in an infinite <code>&lt;think&gt;</code> ramble loop until hitting <code>max_new_tokens=192</code>, "
            "never closing <code>&lt;/think&gt;</code> or emitting JSON. Worse, untrained Loop 2 loss (<code>4.6418</code>) was higher than Loop 1 "
            "(<code>4.6063</code>)."
        )
    )
    story.append(
        P(
            "<b>Phase 3 (Looped SFT Breakthrough):</b> I froze the vision encoder, attached PEFT LoRA (<code>r=16, alpha=32</code>, 10.82M "
            "trainable params = 1.25%) across all 12 verified text-decoder linear projections (Gated Attention + Gated DeltaNet + SwiGLU MLP), "
            "kept <code>reinject_logit</code> trainable, and trained for 3 epochs with assistant-only loss masking and auxiliary middle-loop "
            "loss (<code>aux_weight=0.1</code>). Peak VRAM stayed at <b>7.25 GB / 8.00 GB</b>. By Epoch 1, held-out Loop 2 loss "
            "(<code>0.8897</code>) decisively beat Loop 1 (<code>0.9562</code>), held-out Tool Accuracy rocketed from <b>0.0% to 73.3%</b> "
            "(11/15), and <code>&lt;think&gt;</code> truncation dropped from <b>100.0% to 0.0%</b>."
        )
    )
    story.append(
        P(
            "<b>Phase 4 (Pure-PyTorch GRPO RLVR in 3.37 GB VRAM):</b> To push past SFT imitation, I wrote a pure-PyTorch GRPO loop with "
            "deterministic tool-calling rewards (<code>+0.15</code> format, <code>+0.35</code> schema validity, <code>+0.50</code> exact tool "
            "match, up to <code>-0.20</code> anti-ramble penalty). The cleanest memory trick in the project: instead of loading a second "
            "1.62 GB model for KL regularization, I loaded a frozen <code>ref</code> LoRA adapter on the <i>same</i> base weights "
            "(~21 MB overhead) and micro-batched clipped policy updates one rollout at a time. Peak GRPO VRAM was just <b>3.37 GB</b>, "
            "and 10 GRPO steps lifted held-out Tool Accuracy from <b>73.3% to 86.7% (13/15)</b> with mean reward <b>+0.8667</b>."
        )
    )

    # Visual: Progression Chart
    prog_img = Image(str(DOCS_DIR / "figures" / "progression_chart.png"), width=6.4 * inch, height=3.03 * inch)
    story.append(Spacer(1, 6))
    story.append(KeepTogether([
        prog_img,
        P("Figure 2: Held-out Visual SRE progression across training stages (15 strictly isolated prompts). SFT eliminates 100% of thinking truncation; GRPO pushes accuracy to 86.7%.", caption_style),
    ]))

    # ------------------------------------------------------------------
    # SECTION 4: HONEST ENGINEERING FINDINGS (WSP & TURBOQUANT)
    # ------------------------------------------------------------------
    story.append(P("4. Honest Engineering Findings: Weight-Space Programming &amp; TurboQuant", h2_style))
    story.append(
        P(
            "Not every shiny idea should be forced into a green checkmark. Two of my favorite moments in this build came from "
            "guardrails telling me the truth:"
        )
    )
    story.append(
        make_callout(
            "Finding A: Weight-Space Programming (WSP) Validated Its Own Guardrail",
            "I built <code>weight_editor.py</code> to contrastively probe all 24 MLP <code>down_proj</code> layers on the GRPO model's "
            "13 passing vs. 2 failing traces. Probing cleanly localized separation to layers 23 (<code>||d_bug||=0.1498</code>) and 22 "
            "(<code>0.0942</code>) and tested reversible directional ablation on just 131,072 weights (0.0152% of the model). Both "
            "<code>alpha=0.10</code> and <code>0.25</code> held 86.7% without improving over GRPO, so the rollback guardrail reverted "
            "cleanly. Why? Because SFT + GRPO had already crushed the infinite thinking-loop bug to 0.0% truncation. WSP is a scalpel "
            "for localized failure circuits, not a magic accuracy booster.",
        )
    )
    story.append(Spacer(1, 7))
    story.append(
        make_callout(
            "Finding B: Hybrid-Safe TurboQuant (4.58x Compression vs. Dense Visual OCR Trade-Off)",
            "My <code>TurboQuantHybridCache</code> compresses strictly the 6 Gated Attention layers using orthogonal rotation, "
            "bit-packed <code>uint8</code> Lloyd-Max centroids, and 1-bit QJL key residuals, while leaving all 18 Gated DeltaNet states "
            "bit-for-bit untouched in native <code>bf16</code>/<code>fp32</code>. On a 2064-token benchmark (<code>num_loops=2</code>), "
            "3-bit TurboQuant cut Attention KV by <b>4.58x</b> (48.38 MB to 10.56 MB) and 4-bit achieved <b>3.59x</b> (13.49 MB). "
            "However, evaluating the merged model with 3-bit TurboQuant on dense Grafana screenshots (roughly 1,100 visual patch tokens "
            "flushed past the 64-token bf16 window) dropped tool accuracy from <b>86.7% to 33.3%</b>. Text-only tasks stayed sharp, "
            "but dense chart OCR is sensitive to 3-bit KV noise. Keeping vision prefix tokens unquantized or using 4-bit/bf16 is the "
            "clear production call.",
        )
    )
    story.append(Spacer(1, 10))

    # ------------------------------------------------------------------
    # SECTION 5: SUMMARY ABLATION TABLE & WHAT'S NEXT
    # ------------------------------------------------------------------
    sec5_heading = P("5. Final Held-Out Ablation &amp; Scaling Readiness", h2_style)

    table_data = [
        ["Stage / Configuration", "Loops", "KV Quant", "Tool Acc", "Mean Reward", "Trunc Rate", "Peak VRAM"],
        ["Stock Qwen3.5-0.8B (1-Loop)", "1", "none", "0.0%", "-0.2000", "100.0%", "6.47 GB"],
        ["Stock Qwen3.5-0.8B (2-Loop)", "2", "none", "0.0%", "-0.2000", "100.0%", "6.51 GB"],
        ["Looped SFT (2-Loop LoRA)", "2", "none", "73.3%", "+0.7333", "0.0%", "6.55 GB"],
        ["Looped GRPO (2-Loop RLVR)", "2", "none", "86.7%", "+0.8667", "0.0%", "6.55 GB"],
        ["Looped GRPO + WSP (Reverted)", "2", "none", "86.7%", "+0.8667", "0.0%", "6.55 GB"],
        ["Final Merged + TurboQuant 3-bit", "2", "tq_3bit", "33.3%", "+0.3933", "6.7%", "6.47 GB"],
    ]
    summary_tbl = Table(
        table_data,
        colWidths=[2.35 * inch, 0.55 * inch, 0.78 * inch, 0.78 * inch, 0.92 * inch, 0.80 * inch, 0.82 * inch],
    )
    summary_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#ECFDF5")),
                ("FONTNAME", (0, 4), (-1, 4), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(KeepTogether([sec5_heading, summary_tbl]))
    story.append(Spacer(1, 10))

    story.append(
        P(
            "Finally, I merged the winning GRPO LoRA adapter cleanly into the base weights (cosine similarity <code>0.999743</code>, "
            "<code>100.0%</code> argmax match) and verified full scaling readiness for <b>Qwen/Qwen3.5-2B</b> (identical 24-layer "
            "18 DeltaNet + 6 Gated Attention layout, <code>head_dim=256</code>, 12/12 LoRA target match with 0 vision overlap, plus "
            "optional <code>--load-in-4bit</code> QLoRA support)."
        )
    )
    story.append(Spacer(1, 8))

    # ------------------------------------------------------------------
    # SECTION 6: v2.0 UPGRADE & 3-TIER GENERALIZATION GAUNTLET
    # ------------------------------------------------------------------
    sec6_heading = P("6. v2.0: The 3-Tier Generalization Gauntlet, ETD+ARLoop &amp; Dr. GRPO/DAPO", h2_style)
    story.append(sec6_heading)
    story.append(
        P(
            "After shipping v1, I wanted to stress-test whether a 0.8B model was genuinely generalizing or just memorizing dark-mode "
            "Grafana layouts. I built a deterministic 3-tier gauntlet (<b>Tier 1:</b> 50 unseen OOD dashboards with light-mode themes, "
            "zero-hint pixel-only prompts, and adversarial text-vs-pixel Conflict Traps; <b>Tier 2:</b> 50 public <code>ChartQA</code> "
            "charts; <b>Tier 3:</b> 50 public <code>BFCL v3</code> tool-calling prompts) and shipped three v2.0 upgrades:"
        )
    )
    story.append(
        P(
            "<b>1. Vision-Prefix-Exempt TurboQuant:</b> Pinning the 4 attention sink tokens and all visual patch tokens permanently in "
            "native <code>bf16</code> inside <code>TurboQuantLayer</code> recovered <b>+10.0 points</b> on v1 Tier 1 OOD under 4-bit "
            "TurboQuant (<b>62.0% to 72.0%</b>, matching <code>bf16</code> at <b>85.0%</b> on Visual Shift), while confirming 3-bit "
            "remains noisy for sub-1B multimodal keys (44.0% to 58.0%).",
            bullet_style,
        )
    )
    story.append(
        P(
            "<b>2. ETD Middle-Stack Looping + ARLoop Depth-Attention:</b> Instead of looping all 24 layers, v2 runs layers 0..3 once, "
            "loops core layers 4..19 with isolated per-loop caches and an RMSNorm <code>ARLoopDepthGate</code> over "
            "<code>[E_enc, h^(t-1), h^(t)]</code>, and decodes layers 20..23 once. This trims 2-loop DeltaNet state from 37.69 MB to "
            "<b>31.41 MB</b> and brings 2-loop inference latency down to <b>13.70 s/prompt (9.68 tok/s)</b>.",
            bullet_style,
        )
    )
    story.append(
        P(
            "<b>3. Dr. GRPO + DAPO Clip-Higher + Refusal/Zoom Data:</b> Adding 12 negative/no-tool refusal traces and 8 multi-turn "
            "<code>crop_zoom</code> traces (54 SFT rows total, 0 overlap) and upgrading RLVR to variance-unnormalized Dr. GRPO with "
            "asymmetric DAPO bounds (<code>[-0.20, +0.28]</code>, <b>3.36 GB peak VRAM</b>) lifted <b>BFCL v3</b> from 68.0% to "
            "<b>74.0%</b> (+26.0 pts over Stock, with Irrelevance jumping from 60.0% to <b>73.3%</b>) and recovered <b>ChartQA</b> "
            "from 70.0% to <b>72.0%</b>. On Tier 1 OOD SRE, v1 still leads overall (80.0% vs. 66.0%) because v2's refusal training "
            "made it cautiously abstain on 8 of its 17 OOD misses, while a v1 loop-depth sweep (<code>L=1, 2, 3</code>) proved latent "
            "depth pays off specifically on adversarial Conflict Traps (<b>60.0% at L=1 to 66.7% at L=2, 3</b>).",
            bullet_style,
        )
    )
    story.append(Spacer(1, 6))

    v2_table_data = [
        ["Model / Configuration", "OOD All", "Vis-Shift", "Conflict", "ChartQA", "BFCL All", "BFCL Irrel."],
        ["Stock Qwen3.5-0.8B (1-Loop)", "0.0%", "0.0%", "0.0%", "80.0%", "48.0%", "100.0%"],
        ["v1 Merged (2-Loop, bf16)", "80.0%", "85.0%", "66.7%", "70.0%", "68.0%", "60.0%"],
        ["v1 Merged (Naive TQ 4-bit)", "62.0%", "60.0%", "60.0%", "-", "-", "-"],
        ["v1 Merged (TQ 4-bit Vis-Exempt)", "72.0%", "85.0%", "53.3%", "-", "-", "-"],
        ["v2 ETD+ARLoop (2-Loop, bf16)", "66.0%", "70.0%", "60.0%", "72.0%", "74.0%", "73.3%"],
        ["v2 ETD+ARLoop (TQ 4-bit Vis-Ex.)", "62.0%", "75.0%", "53.3%", "-", "-", "-"],
    ]
    v2_tbl = Table(
        v2_table_data,
        colWidths=[2.32 * inch, 0.72 * inch, 0.78 * inch, 0.75 * inch, 0.80 * inch, 0.80 * inch, 0.83 * inch],
    )
    v2_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.0),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#ECFDF5")),
                ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#EEF2FF")),
                ("FONTNAME", (0, 2), (-1, 2), "Helvetica-Bold"),
                ("FONTNAME", (0, 5), (-1, 5), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(KeepTogether([
        v2_tbl,
        P("Table 2: Stock vs. v1 vs. v2 across Tier 1 OOD SRE (50 dashboards), Tier 2 ChartQA (50), and Tier 3 BFCL v3 (50) on RTX 4060 8 GB.", caption_style),
    ]))
    story.append(Spacer(1, 6))
    story.append(
        P(
            "Building v1 and v2 end-to-end on one 8 GB laptop GPU proved that thoughtful architecture, honest ablations, and clean "
            "pure-PyTorch RLVR beat throwing raw compute at a problem. Next up: exploring in-layer Hardware ALU Experts routed via RLVR "
            "for exact numerical reasoning inside looped blocks."
        )
    )

    doc.build(story, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)
    return OUT_PDF


if __name__ == "__main__":
    pdf_path = build_pdf()
    print("Built Dev Diary PDF:", pdf_path)
