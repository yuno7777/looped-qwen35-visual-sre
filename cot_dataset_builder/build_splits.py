"""Niche prompt pack generator and leakage-proof stratified train/eval splitter."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_NICHE_PROMPTS = BASE_DIR / "niche_prompts.jsonl"
DEFAULT_CLEAN_DATASET = BASE_DIR / "dataset_clean.jsonl"
DEFAULT_SAMPLES_DIR = BASE_DIR / "samples"
DEFAULT_SPLITS_DIR = BASE_DIR / "splits"

IMAGE_TAG_RE = re.compile(r"(?:<image>\s*\n?)+")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_prompt_text(text: str) -> str:
    cleaned = IMAGE_TAG_RE.sub("", text or "")
    return " ".join(cleaned.strip().lower().split())


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _render_invoice_image(
    out_path: Path,
    vendor: str,
    invoice_num: str,
    amount: float,
    currency: str,
    due_date: str,
    line_items: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (448, 320), color=(252, 253, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([16, 16, 432, 304], outline=(40, 60, 100), width=3)
    draw.rectangle([16, 16, 432, 64], fill=(30, 70, 140))
    draw.text((28, 32), f"INVOICE - {vendor}", fill=(255, 255, 255))
    draw.text((28, 84), f"Invoice Number: {invoice_num}", fill=(20, 20, 20))
    draw.text((28, 114), f"Due Date: {due_date}", fill=(20, 20, 20))
    draw.text((28, 144), f"Line Items: {line_items}", fill=(20, 20, 20))
    draw.line([(28, 180), (416, 180)], fill=(180, 190, 210), width=2)
    draw.text((28, 200), f"TOTAL DUE: {currency} {amount:.2f}", fill=(180, 30, 30))
    img.save(out_path)


def _render_ui_image(
    out_path: Path,
    button_label: str,
    bx: int,
    by: int,
    width: int = 448,
    height: int = 280,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color=(244, 246, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([12, 12, width - 12, 50], fill=(35, 42, 56))
    draw.text((24, 24), "Admin Operations Console", fill=(255, 255, 255))
    # Secondary decoy button
    draw.rectangle([30, height - 80, 150, height - 38], fill=(210, 214, 222), outline=(120, 125, 135), width=2)
    draw.text((58, height - 66), "Cancel", fill=(40, 40, 40))
    # Target primary button centered around (bx, by)
    x1, y1 = max(20, bx - 75), max(60, by - 22)
    x2, y2 = min(width - 20, bx + 75), min(height - 20, by + 22)
    draw.rectangle([x1, y1, x2, y2], fill=(24, 118, 210), outline=(12, 68, 130), width=3)
    draw.text((x1 + 14, y1 + 14), button_label, fill=(255, 255, 255))
    img.save(out_path)


def generate_starter_pack(
    num_prompts: int,
    output_path: Path,
    samples_dir: Path,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate a realistic starter niche prompt pack covering all 5 tools + multimodal images."""
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    ticket_scenarios = [
        ("CUST-88412", "authentication", "urgent", True, "Enterprise customer CUST-88412 reports that SSO login via Okta is returning HTTP 500 across their entire organization since 09:15 UTC. Open an urgent support ticket and flag it for escalation."),
        ("CUST-44109", "billing", "high", False, "Customer CUST-44109 was double-charged $4,200 on their annual renewal invoice INV-9921. Open a high-priority billing support ticket without escalation."),
        ("CUST-71920", "outage", "urgent", True, "Production webhook delivery for CUST-71920 has been 100% failing in us-east for 25 minutes. Create an urgent outage ticket and escalate immediately."),
        ("CUST-10284", "bug", "medium", False, "User CUST-10284 found that CSV export truncates UTF-8 emojis in column headers. File a medium-priority bug ticket; no escalation needed."),
        ("CUST-55391", "feature_request", "low", False, "Account CUST-55391 requested SAML group sync filtering for read-only auditors. Log a low-priority feature request ticket."),
        ("CUST-63218", "authentication", "high", True, "Customer CUST-63218 says MFA TOTP codes are rejected for newly onboarded staff. Create a high-priority authentication ticket and escalate."),
        ("CUST-90811", "billing", "medium", False, "CUST-90811 needs a VAT tax ID added retroactively to their October receipt. Create a medium-priority billing ticket."),
        ("CUST-31744", "bug", "high", True, "CUST-31744 reports that saving a workflow triggers an infinite spinner on Safari 18. Open a high-priority bug ticket with escalation."),
        ("CUST-20419", "outage", "urgent", True, "API gateway returns 503 for all endpoints under tenant CUST-20419. Create an urgent outage ticket and flag requires_escalation."),
        ("CUST-66702", "feature_request", "medium", False, "Enterprise client CUST-66702 asked for dark-mode PDF audit exports. Open a medium-priority feature_request ticket."),
    ]

    inventory_scenarios = [
        ("SKU-AX904", "eu-central", True, 25),
        ("SKU-NV409", "us-east", False, 10),
        ("SKU-QP112", "ap-south", True, 50),
        ("SKU-LT880", "us-west", False, 5),
        ("SKU-ZR301", "eu-central", False, 100),
        ("SKU-MK772", "us-east", True, 0),
        ("SKU-BX215", "ap-south", False, 15),
        ("SKU-HD640", "us-west", True, 40),
        ("SKU-CR508", "eu-central", True, 12),
        ("SKU-VT919", "us-east", False, 75),
    ]

    maint_scenarios = [
        ("payments-ledger", "2026-12-01T02:00:00Z", 45, ["#ops-alerts", "statuspage"], True),
        ("auth-gateway", "2026-12-03T04:30:00Z", 30, ["#security-ops", "pagerduty"], True),
        ("search-indexer", "2026-12-05T01:00:00Z", 120, ["#search-team"], False),
        ("billing-worker", "2026-12-07T03:15:00Z", 60, ["#billing-alerts", "statuspage"], True),
        ("cdn-edge-router", "2026-12-10T00:00:00Z", 20, ["#infra-noc", "statuspage", "#eng-all"], True),
        ("analytics-warehouse", "2026-12-12T05:00:00Z", 180, ["#data-eng"], False),
        ("webhook-dispatcher", "2026-12-14T02:30:00Z", 40, ["#platform-ops", "statuspage"], True),
        ("object-storage-proxy", "2026-12-16T01:45:00Z", 90, ["#storage-oncall"], True),
        ("notification-mailer", "2026-12-18T06:00:00Z", 15, ["#comms-ops"], False),
        ("feature-flag-store", "2026-12-20T03:00:00Z", 25, ["#release-eng", "statuspage"], True),
    ]

    invoice_scenarios = [
        ("Acme Cloud Ltd", "INV-2026-104", 1285.50, "USD", "2026-12-15", 4),
        ("Nordic Fiber GmbH", "INV-2026-209", 4920.00, "EUR", "2026-12-22", 7),
        ("Thames Data PLC", "INV-2026-318", 875.25, "GBP", "2026-12-28", 2),
        ("Deccan Compute Pvt", "INV-2026-441", 64500.00, "INR", "2027-01-05", 5),
        ("Maple Hosting Inc", "INV-2026-552", 2140.75, "CAD", "2027-01-10", 3),
        ("Helios Security Co", "INV-2026-667", 3499.00, "USD", "2027-01-14", 6),
    ]

    ui_scenarios = [
        ("Confirm Payment", "left_click", 310, 190),
        ("Deploy Release", "left_click", 295, 165),
        ("Approve Refund", "left_click", 320, 140),
        ("Inspect Logs", "double_click", 275, 175),
        ("Context Menu", "right_click", 300, 155),
        ("Rotate API Key", "left_click", 315, 205),
    ]

    tools_cycle = [
        "create_support_ticket",
        "query_inventory_db",
        "schedule_maintenance_window",
        "extract_invoice_fields",
        "click_ui_element",
    ]
    counters = defaultdict(int)

    for idx in range(num_prompts):
        tool = tools_cycle[idx % len(tools_cycle)]
        c = counters[tool]
        counters[tool] += 1
        item_id = f"niche_{idx + 1:03d}"

        if tool == "create_support_ticket":
            cust, cat, prio, esc, base_prompt = ticket_scenarios[c % len(ticket_scenarios)]
            variant_tag = f" [Ref #{c + 1}]" if c >= len(ticket_scenarios) else ""
            records.append(
                {
                    "id": item_id,
                    "prompt": f"{base_prompt}{variant_tag}",
                    "image_path": None,
                    "expected_tool": tool,
                    "expected_args": {
                        "customer_id": cust,
                        "category": cat,
                        "priority": prio,
                        "requires_escalation": esc,
                    },
                }
            )
        elif tool == "query_inventory_db":
            sku, region, inc_res, min_q = inventory_scenarios[c % len(inventory_scenarios)]
            if c >= len(inventory_scenarios):
                sku = f"{sku}-{c}"
            res_phrase = "Include reserved units" if inc_res else "Exclude reserved units"
            prompt = (
                f"Query warehouse inventory for {sku} in the {region} region. "
                f"{res_phrase} and require a minimum quantity of {min_q}."
            )
            records.append(
                {
                    "id": item_id,
                    "prompt": prompt,
                    "image_path": None,
                    "expected_tool": tool,
                    "expected_args": {
                        "sku": sku,
                        "warehouse_region": region,
                        "include_reserved": inc_res,
                        "min_quantity": min_q,
                    },
                }
            )
        elif tool == "schedule_maintenance_window":
            svc, start_iso, dur, chans, rollback = maint_scenarios[c % len(maint_scenarios)]
            if c >= len(maint_scenarios):
                svc = f"{svc}-v{c}"
            rb_phrase = "enable automatic rollback" if rollback else "disable automatic rollback"
            prompt = (
                f"Schedule a {dur}-minute maintenance window for {svc} starting {start_iso}. "
                f"Notify {', '.join(chans)} and {rb_phrase}."
            )
            records.append(
                {
                    "id": item_id,
                    "prompt": prompt,
                    "image_path": None,
                    "expected_tool": tool,
                    "expected_args": {
                        "service_name": svc,
                        "start_iso": start_iso,
                        "duration_minutes": dur,
                        "notify_channels": chans,
                        "auto_rollback": rollback,
                    },
                }
            )
        elif tool == "extract_invoice_fields":
            vendor, inv_no, amt, curr, due, items = invoice_scenarios[c % len(invoice_scenarios)]
            if c >= len(invoice_scenarios):
                inv_no = f"{inv_no}-R{c}"
                amt = round(amt + rng.uniform(10.0, 250.0), 2)
            rel_img = f"samples/invoice_{c + 1:03d}.png"
            _render_invoice_image(samples_dir / f"invoice_{c + 1:03d}.png", vendor, inv_no, amt, curr, due, items)
            prompt = (
                f"Extract the vendor name, invoice number, total amount, currency, due date, and line item count "
                f"from the attached invoice screenshot (reference batch {c + 1})."
            )
            records.append(
                {
                    "id": item_id,
                    "prompt": prompt,
                    "image_path": rel_img,
                    "expected_tool": tool,
                    "expected_args": {
                        "vendor_name": vendor,
                        "invoice_number": inv_no,
                        "total_amount": amt,
                        "currency": curr,
                        "due_date": due,
                        "line_item_count": items,
                    },
                }
            )
        else:  # click_ui_element
            label, action, bx, by = ui_scenarios[c % len(ui_scenarios)]
            if c >= len(ui_scenarios):
                bx = max(140, min(340, bx + rng.randint(-15, 15)))
                by = max(100, min(220, by + rng.randint(-15, 15)))
            rel_img = f"samples/ui_{c + 1:03d}.png"
            _render_ui_image(samples_dir / f"ui_{c + 1:03d}.png", label, bx, by)
            prompt = (
                f"Locate the primary '{label}' button in the attached console screenshot and perform a "
                f"{action} at its center coordinates with your confidence score (screen #{c + 1})."
            )
            records.append(
                {
                    "id": item_id,
                    "prompt": prompt,
                    "image_path": rel_img,
                    "expected_tool": tool,
                    "expected_args": {
                        "element_label": label,
                        "action": action,
                        "x": bx,
                        "y": by,
                    },
                }
            )

    write_jsonl(output_path, records)
    return records


def build_stratified_splits(
    prompts_path: Path,
    clean_dataset_path: Path,
    splits_dir: Path,
    eval_ratio: float = 0.20,
    seed: int = 42,
) -> dict[str, Any]:
    """Deterministically split prompts & clean SFT traces by stratified `expected_tool` with zero leakage."""
    prompts = load_jsonl(prompts_path)
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_path}")

    rng = random.Random(seed)
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in prompts:
        by_tool[rec.get("expected_tool") or "unspecified"].append(rec)

    train_prompts: list[dict[str, Any]] = []
    eval_prompts: list[dict[str, Any]] = []
    per_tool_counts: dict[str, dict[str, int]] = {}

    for tool_name in sorted(by_tool.keys()):
        group = list(by_tool[tool_name])
        rng.shuffle(group)
        if len(group) == 1:
            n_eval = 0
        else:
            n_eval = max(1, int(round(len(group) * eval_ratio)))
            n_eval = min(n_eval, len(group) - 1)
        eval_slice = group[:n_eval]
        train_slice = group[n_eval:]
        eval_prompts.extend(eval_slice)
        train_prompts.extend(train_slice)
        per_tool_counts[tool_name] = {"train": len(train_slice), "eval": len(eval_slice)}

    eval_ids = {r["id"] for r in eval_prompts}
    train_ids = {r["id"] for r in train_prompts}
    eval_norm_prompts = {normalize_prompt_text(r["prompt"]) for r in eval_prompts}
    train_norm_prompts = {normalize_prompt_text(r["prompt"]) for r in train_prompts}

    # Filter clean SFT traces strictly excluding any held-out ID or held-out prompt text
    clean_rows = load_jsonl(clean_dataset_path)
    train_sft_rows: list[dict[str, Any]] = []
    excluded_sft_ids: list[str] = []

    for row in clean_rows:
        row_id = row.get("id", "")
        user_msgs = [m.get("content", "") for m in row.get("messages", []) if m.get("role") == "user"]
        first_user_norm = normalize_prompt_text(user_msgs[0]) if user_msgs else ""
        if row_id in eval_ids or (first_user_norm and first_user_norm in eval_norm_prompts):
            excluded_sft_ids.append(row_id)
            continue
        train_sft_rows.append(row)

    splits_dir.mkdir(parents=True, exist_ok=True)
    train_sft_path = splits_dir / "train_sft.jsonl"
    train_grpo_path = splits_dir / "train_grpo_prompts.jsonl"
    eval_heldout_path = splits_dir / "eval_heldout.jsonl"
    manifest_path = splits_dir / "split_manifest.json"

    write_jsonl(train_sft_path, train_sft_rows)
    write_jsonl(train_grpo_path, train_prompts)
    write_jsonl(eval_heldout_path, eval_prompts)

    # Hard leakage assertions
    sft_ids = {r["id"] for r in train_sft_rows}
    overlap_train_eval_ids = sorted((train_ids | sft_ids) & eval_ids)
    overlap_prompts = sorted(train_norm_prompts & eval_norm_prompts)

    if overlap_train_eval_ids:
        raise AssertionError(f"LEAKAGE DETECTED: overlapping IDs between train and eval: {overlap_train_eval_ids}")
    if overlap_prompts:
        raise AssertionError(f"LEAKAGE DETECTED: overlapping prompts between train and eval: {len(overlap_prompts)}")

    manifest = {
        "seed": seed,
        "eval_ratio": eval_ratio,
        "source_prompts": str(prompts_path),
        "source_clean_dataset": str(clean_dataset_path),
        "counts": {
            "total_prompts": len(prompts),
            "train_grpo_prompts": len(train_prompts),
            "eval_heldout": len(eval_prompts),
            "train_sft_rows": len(train_sft_rows),
            "excluded_sft_rows_from_eval": len(excluded_sft_ids),
        },
        "per_tool_stratification": per_tool_counts,
        "leakage_checks": {
            "overlapping_ids": len(overlap_train_eval_ids),
            "overlapping_prompts": len(overlap_prompts),
            "status": "PASS",
        },
        "sha256": {
            "train_sft.jsonl": sha256_file(train_sft_path),
            "train_grpo_prompts.jsonl": sha256_file(train_grpo_path),
            "eval_heldout.jsonl": sha256_file(eval_heldout_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate niche prompt pack and leakage-proof train/eval splits.")
    parser.add_argument(
        "--generate-starter-pack",
        type=int,
        default=0,
        help="Generate N starter niche prompts (text + synthetic multimodal images) into --prompts",
    )
    parser.add_argument("--prompts", type=Path, default=DEFAULT_NICHE_PROMPTS, help="Niche prompt bank JSONL")
    parser.add_argument("--clean-dataset", type=Path, default=DEFAULT_CLEAN_DATASET, help="Harvested clean SFT JSONL")
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR, help="Directory for starter images")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR, help="Output directory for splits")
    parser.add_argument("--eval-ratio", type=float, default=0.20, help="Fraction of prompts reserved for eval_heldout")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--skip-splits", action="store_true", help="Only generate starter pack without writing splits")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.generate_starter_pack > 0:
        recs = generate_starter_pack(
            num_prompts=args.generate_starter_pack,
            output_path=args.prompts,
            samples_dir=args.samples_dir,
            seed=args.seed,
        )
        n_mm = sum(1 for r in recs if r.get("image_path"))
        print(f"Generated starter pack: {len(recs)} prompts ({n_mm} multimodal) -> {args.prompts}")

    if not args.skip_splits:
        manifest = build_stratified_splits(
            prompts_path=args.prompts,
            clean_dataset_path=args.clean_dataset,
            splits_dir=args.splits_dir,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
        )
        print(f"Built stratified splits in: {args.splits_dir}")
        print(json.dumps(manifest["counts"], indent=2))
        print(f"Leakage checks: {manifest['leakage_checks']}")
        print(f"Manifest saved: {args.splits_dir / 'split_manifest.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
