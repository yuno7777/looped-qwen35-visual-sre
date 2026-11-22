"""Production-grade OpenCode CLI wrapper for collecting DeepSeek CoT + tool-call SFT datasets."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    BACKOFF_BASE_SECONDS,
    BACKOFF_JITTER_RATIO,
    BACKOFF_MAX_SECONDS,
    BASE_DIR,
    CLEAN_DATASET_FILE,
    FAILED_TRACES_FILE,
    MAX_RETRIES,
    MODEL_ID,
    PROMPTS_FILE,
    RAW_DATASET_FILE,
    SLEEP_SECONDS,
    SUBPROCESS_TIMEOUT_SECONDS,
    SYSTEM_PROMPT,
    THROTTLE_PATTERNS,
    build_teacher_prompt,
)
from validator import validate_trace

THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cot_dataset_builder")


@dataclass
class ParsedEventStream:
    """Structured extraction from `opencode run --format json` NDJSON stdout."""

    cot: str = ""
    answer: str = ""
    finish_reason: str | None = None
    session_id: str | None = None
    tokens: dict[str, Any] = field(default_factory=dict)
    cost: float | None = None
    errors: list[str] = field(default_factory=list)
    event_count: int = 0
    unparsed_lines: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunResult:
    """Result of invoking `opencode run` with retries."""

    ok: bool
    returncode: int | None
    stdout: str
    stderr: str
    parsed: ParsedEventStream
    attempts: int
    error_reason: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_opencode_binary() -> str:
    """Locate the `opencode` executable (handles `.cmd` shim on Windows)."""
    binary = shutil.which("opencode")
    if not binary:
        raise FileNotFoundError(
            "Could not find `opencode` on PATH. Install OpenCode CLI and ensure it is on PATH."
        )
    return binary


def run_preflight_check(model_id: str, opencode_bin: str) -> bool:
    """Run `opencode models` to verify the configured `provider/model` ID exists locally."""
    logger.info("Running preflight check: `opencode models` for model %r ...", model_id)
    try:
        proc = subprocess.run(
            [opencode_bin, "models"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        logger.error("Preflight check timed out while running `opencode models`.")
        return False

    if proc.returncode != 0:
        logger.error("`opencode models` failed (exit %d): %s", proc.returncode, proc.stderr.strip())
        return False

    available_models = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if model_id in available_models:
        logger.info("Preflight passed: model %r is available (%d total models).", model_id, len(available_models))
        return True

    deepseek_models = [m for m in available_models if "deepseek" in m.lower()]
    logger.error("Configured model %r was NOT found in `opencode models`.", model_id)
    if deepseek_models:
        logger.error("Available DeepSeek models in your local OpenCode setup:")
        for m in deepseek_models:
            logger.error("  - %s", m)
    else:
        logger.error("No models containing 'deepseek' were found. Run `opencode models` to inspect all models.")
    logger.error("Update MODEL_ID in config.py or pass `--model <provider/model>`.")
    return False


def load_prompts(prompts_path: Path) -> list[dict[str, Any]]:
    """Load prompt items from a JSONL file."""
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with prompts_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {prompts_path} at line {line_no}: {exc}") from exc

            item_id = obj.get("id")
            prompt = obj.get("prompt")
            if not isinstance(item_id, str) or not item_id.strip():
                raise ValueError(f"Missing string 'id' in {prompts_path} at line {line_no}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Missing string 'prompt' in {prompts_path} at line {line_no}")
            if item_id in seen_ids:
                raise ValueError(f"Duplicate id {item_id!r} in {prompts_path} at line {line_no}")

            seen_ids.add(item_id)
            items.append(obj)

    return items


def load_completed_ids(*jsonl_paths: Path) -> set[str]:
    """Load already-processed IDs from checkpoint JSONL files for clean resume."""
    completed: set[str] = set()
    for path in jsonl_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Ignoring malformed/partial line %d in %s", line_no, path.name)
                    continue
                item_id = obj.get("id")
                if isinstance(item_id, str) and item_id:
                    completed.add(item_id)
    return completed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Durably append a single JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(serialized)
        fh.flush()
        os.fsync(fh.fileno())


def parse_opencode_json_events(stdout: str) -> ParsedEventStream:
    """Parse newline-delimited JSON events emitted by `opencode run --format json --thinking`."""
    parsed = ParsedEventStream()
    reasoning_chunks: list[str] = []
    text_chunks: list[str] = []

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            parsed.unparsed_lines.append(line)
            continue

        if not isinstance(event, dict):
            parsed.unparsed_lines.append(line)
            continue

        parsed.events.append(event)
        parsed.event_count += 1

        if not parsed.session_id and isinstance(event.get("sessionID"), str):
            parsed.session_id = event["sessionID"]

        event_type = str(event.get("type", "")).lower()
        part = event.get("part") if isinstance(event.get("part"), dict) else {}
        part_type = str(part.get("type", "")).lower()

        if event_type in ("reasoning", "thinking") or part_type in ("reasoning", "thinking"):
            chunk = part.get("text") or event.get("text") or ""
            if isinstance(chunk, str) and chunk:
                reasoning_chunks.append(chunk)

        elif event_type == "text" or part_type == "text":
            chunk = part.get("text") or event.get("text") or ""
            if isinstance(chunk, str) and chunk:
                text_chunks.append(chunk)

        elif event_type in ("step_finish", "step-finish") or part_type in ("step_finish", "step-finish"):
            reason = part.get("reason") or event.get("reason")
            if isinstance(reason, str):
                parsed.finish_reason = reason
            tokens = part.get("tokens")
            if isinstance(tokens, dict):
                parsed.tokens = tokens
            cost = part.get("cost")
            if isinstance(cost, (int, float)):
                parsed.cost = float(cost)

        elif event_type == "error" or "error" in event:
            err_payload = event.get("error") or part.get("error") or event.get("message") or event
            parsed.errors.append(json.dumps(err_payload, ensure_ascii=False) if isinstance(err_payload, dict) else str(err_payload))

    cot = "\n".join(reasoning_chunks).strip()
    answer = "\n".join(text_chunks).strip()

    # Fallback: if a model emits inline <think>...</think> inside text instead of reasoning events
    if not cot and answer:
        think_matches = THINK_TAG_RE.findall(answer)
        if think_matches:
            cot = "\n".join(m.strip() for m in think_matches if m.strip()).strip()
            answer = THINK_TAG_RE.sub("", answer).strip()

    parsed.cot = cot
    parsed.answer = answer
    return parsed


def is_throttle_error(stderr: str, stdout: str, parsed: ParsedEventStream, returncode: int = 0) -> bool:
    """Detect provider rate-limit / throttle signals in stderr, JSON error events, or failed stdout.

    Avoids scanning valid JSON event stdout (whose 13-digit timestamps or IDs can
    coincidentally contain substrings like '429', '503', or '529').
    """
    sources: list[str] = [stderr, *parsed.errors]
    if returncode != 0 or parsed.event_count == 0:
        sources.append(stdout)
    haystack = "\n".join(sources).lower()
    return any(pattern in haystack for pattern in THROTTLE_PATTERNS)


def compute_backoff(attempt: int) -> float:
    """Compute exponential backoff with jitter (1-indexed attempt)."""
    delay = min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
    jitter = delay * BACKOFF_JITTER_RATIO * random.uniform(-1.0, 1.0)
    return max(1.0, delay + jitter)


def build_opencode_command(
    opencode_bin: str,
    model_id: str,
    image_path: Path | None = None,
) -> list[str]:
    """Construct the exact non-interactive `opencode run` command.

    The prompt is passed via stdin (`input=prompt_text`) rather than positional CLI args
    so multiline prompts, JSON quotes, and long tool schemas are never truncated by
    Windows `.cmd` shims or OS command-line length limits.
    """
    cmd = [
        opencode_bin,
        "run",
        "-m",
        model_id,
        "--format",
        "json",
        "--thinking",
        "--pure",
    ]
    if image_path is not None:
        cmd.extend(["-f", str(image_path)])
    return cmd


def execute_with_retries(
    opencode_bin: str,
    model_id: str,
    prompt_text: str,
    image_path: Path | None = None,
    max_retries: int = MAX_RETRIES,
    timeout_seconds: int = SUBPROCESS_TIMEOUT_SECONDS,
) -> RunResult:
    """Invoke `opencode run` with exponential backoff on errors, empty stdout, or throttling."""
    cmd = build_opencode_command(opencode_bin, model_id, image_path)

    last_rc: int | None = None
    last_stdout = ""
    last_stderr = ""
    last_parsed = ParsedEventStream()
    last_error = "unknown failure"

    for attempt in range(1, max_retries + 1):
        try:
            proc = subprocess.run(
                cmd,
                input=prompt_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
            )
            last_rc = proc.returncode
            last_stdout = proc.stdout or ""
            last_stderr = proc.stderr or ""
            last_parsed = parse_opencode_json_events(last_stdout)

            throttled = is_throttle_error(last_stderr, last_stdout, last_parsed, returncode=proc.returncode)
            if proc.returncode == 0 and last_stdout.strip() and last_parsed.event_count > 0 and not last_parsed.errors and not throttled:
                return RunResult(
                    ok=True,
                    returncode=proc.returncode,
                    stdout=last_stdout,
                    stderr=last_stderr,
                    parsed=last_parsed,
                    attempts=attempt,
                )

            if throttled:
                last_error = f"throttle/rate-limit detected (exit={proc.returncode})"
            elif proc.returncode != 0:
                last_error = f"non-zero exit code {proc.returncode}"
            elif not last_stdout.strip():
                last_error = "empty stdout from opencode run"
            elif last_parsed.errors:
                last_error = f"opencode JSON error event: {'; '.join(last_parsed.errors)}"
            else:
                last_error = "no valid JSON events parsed from stdout"

        except subprocess.TimeoutExpired as exc:
            last_rc = None
            last_stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            last_stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            last_parsed = parse_opencode_json_events(last_stdout)
            last_error = f"subprocess timed out after {timeout_seconds}s"

        if attempt < max_retries:
            wait_s = compute_backoff(attempt)
            logger.warning(
                "Attempt %d/%d failed (%s). Backing off %.1fs before retry...",
                attempt,
                max_retries,
                last_error,
                wait_s,
            )
            time.sleep(wait_s)

    return RunResult(
        ok=False,
        returncode=last_rc,
        stdout=last_stdout,
        stderr=last_stderr,
        parsed=last_parsed,
        attempts=max_retries,
        error_reason=last_error,
    )


def build_simulated_tool_failure(expected_tool: str | None) -> tuple[str, str]:
    """Return `(failed_call_json, tool_error_feedback)` for multi-turn SRE error-recovery harvesting."""
    failures: dict[str, tuple[dict[str, Any], str]] = {
        "trigger_rollback": (
            {
                "tool": "trigger_rollback",
                "arguments": {
                    "service": "unknown-svc",
                    "target_version": "v0",
                    "reason": "errors",
                    "environment": "production",
                },
            },
            "ToolError: Schema validation failed — arguments.environment='production' is not in allowed enum "
            "['prod', 'staging']; verify exact service name and previous stable target_version from the dashboard.",
        ),
        "scale_service": (
            {
                "tool": "scale_service",
                "arguments": {
                    "service": "api",
                    "target_replicas": "20",
                    "region": "us-east",
                    "reason": "cpu",
                },
            },
            "ToolError: Schema validation failed — arguments.target_replicas must be integer (not string); "
            "arguments.region='us-east' is not in allowed enum ['us-east-1', 'eu-west-1', 'ap-south-1']. "
            "Fix the arguments using exact values from the telemetry alert.",
        ),
        "restart_pod_pool": (
            {
                "tool": "restart_pod_pool",
                "arguments": {
                    "service": "worker",
                    "namespace": "",
                    "strategy": "graceful",
                    "reason": "oom",
                },
            },
            "ToolError: Schema validation failed — arguments.strategy='graceful' is not in allowed enum "
            "['rolling', 'immediate']; arguments.namespace must be non-empty; reason must be at least 5 chars.",
        ),
        "query_logs": (
            {
                "tool": "query_logs",
                "arguments": {
                    "service": "gateway",
                    "time_window_minutes": 0,
                    "severity": "WARNING",
                    "filter_query": "",
                },
            },
            "ToolError: Schema validation failed — arguments.time_window_minutes=0 < minimum 1; "
            "arguments.severity='WARNING' is not in allowed enum ['ERROR', 'WARN', 'FATAL']; "
            "arguments.filter_query must be non-empty.",
        ),
        "create_incident": (
            {
                "tool": "create_incident",
                "arguments": {
                    "severity": "P0",
                    "affected_services": [],
                    "summary": "down",
                    "page_oncall": "yes",
                },
            },
            "ToolError: Schema validation failed — arguments.severity='P0' is not in allowed enum "
            "['SEV1', 'SEV2', 'SEV3']; arguments.affected_services must have minItems=1; "
            "arguments.page_oncall must be boolean.",
        ),
    }
    bad_obj, err_msg = failures.get(
        expected_tool or "",
        (
            {"tool": expected_tool or "unknown_tool", "arguments": {}},
            "ToolError: Schema validation failed — missing required arguments. "
            "Inspect the tool schema carefully and emit the corrected JSON tool call.",
        ),
    )
    return json.dumps(bad_obj, ensure_ascii=False), err_msg


def build_recovery_teacher_prompt(
    user_prompt: str,
    bad_call_json: str,
    tool_error_msg: str,
    has_image: bool = False,
) -> str:
    """Build a multi-turn error-recovery prompt for DeepSeek to self-correct a failed tool call."""
    base = build_teacher_prompt(user_prompt, has_image=has_image)
    return (
        f"{base}\n\n"
        f"PREVIOUS FAILED TOOL CALL:\n{bad_call_json}\n\n"
        f"TOOL RUNTIME ERROR:\n{tool_error_msg}\n\n"
        "SELF-CORRECTION INSTRUCTION:\n"
        "Diagnose why the previous tool call failed in your reasoning, then output ONLY the corrected valid JSON object."
    )


def format_qwen35_sft_record(
    item_id: str,
    user_prompt: str,
    cot: str,
    valid_tool_call_json: str,
    image_path_str: str | None = None,
    recovery_turn: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Format a passing trace for Qwen3.5 multimodal SFT (single-turn or multi-turn recovery)."""
    user_content = f"<image>\n{user_prompt}" if image_path_str else user_prompt
    images = [image_path_str] if image_path_str else []
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if recovery_turn is not None:
        bad_call_json, tool_error_msg = recovery_turn
        messages.append({"role": "assistant", "content": bad_call_json})
        messages.append({"role": "user", "content": tool_error_msg})
    messages.append(
        {"role": "assistant", "content": f"<think>\n{cot.strip()}\n</think>\n{valid_tool_call_json}"}
    )
    record: dict[str, Any] = {
        "id": item_id,
        "messages": messages,
        "images": images,
    }
    if recovery_turn is not None:
        record["is_recovery"] = True
    return record


def resolve_image_path(raw_image_path: str | None, prompts_file: Path) -> Path | None:
    """Resolve an optional image path relative to the prompts file directory (or CWD)."""
    if not raw_image_path:
        return None
    candidate = Path(raw_image_path)
    if candidate.is_absolute():
        return candidate
    rel_to_prompts = prompts_file.parent / candidate
    if rel_to_prompts.exists():
        return rel_to_prompts
    return candidate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CoT + tool-calling SFT datasets via non-interactive `opencode run`."
    )
    parser.add_argument("--model", default=MODEL_ID, help=f"OpenCode provider/model ID (default: {MODEL_ID})")
    parser.add_argument("--input", type=Path, default=PROMPTS_FILE, help="Input prompts JSONL file")
    parser.add_argument("--raw-out", type=Path, default=RAW_DATASET_FILE, help="Raw traces JSONL output")
    parser.add_argument("--clean-out", type=Path, default=CLEAN_DATASET_FILE, help="Clean Qwen3.5 SFT JSONL output")
    parser.add_argument("--failed-out", type=Path, default=FAILED_TRACES_FILE, help="Failed traces JSONL log")
    parser.add_argument("--sleep", type=float, default=SLEEP_SECONDS, help="Seconds to sleep between requests")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Max retries per prompt")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N pending prompts")
    parser.add_argument(
        "--recovery-mode",
        action="store_true",
        help="Harvest ~25%% of traces as multi-turn tool-error recovery examples",
    )
    parser.add_argument(
        "--recovery-ratio",
        type=float,
        default=0.25,
        help="Fraction of prompts harvested in recovery mode when --recovery-mode is set (default: 0.25)",
    )
    parser.add_argument("--preflight-only", action="store_true", help="Verify `opencode models` and exit")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip the `opencode models` preflight check")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        opencode_bin = resolve_opencode_binary()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not args.skip_preflight:
        if not run_preflight_check(args.model, opencode_bin):
            return 1
        if args.preflight_only:
            return 0
    elif args.preflight_only:
        logger.info("`--preflight-only` requested alongside `--skip-preflight`; running preflight anyway.")
        return 0 if run_preflight_check(args.model, opencode_bin) else 1

    prompts = load_prompts(args.input)
    completed_ids = load_completed_ids(args.raw_out, args.clean_out)
    pending = [p for p in prompts if p["id"] not in completed_ids]

    if args.limit is not None:
        pending = pending[: args.limit]

    logger.info(
        "Loaded %d prompts (%d already completed, %d pending). Model=%s",
        len(prompts),
        len(completed_ids),
        len(pending),
        args.model,
    )

    if not pending:
        logger.info("Nothing to do — all prompts are already checkpointed.")
        return 0

    stats = {"processed": 0, "clean": 0, "rejected": 0, "failed_exec": 0, "recovery": 0}

    recovery_indices: set[int] = set()
    if args.recovery_mode and pending:
        stride = max(1, round(1.0 / max(0.05, min(1.0, args.recovery_ratio))))
        # Guarantee at least 1 recovery item (index 1 if len >= 2 else 0) on small --limit runs
        for pos in range(len(pending)):
            if (pos + 1) % stride == 0:
                recovery_indices.add(pos)
        if not recovery_indices:
            recovery_indices.add(len(pending) - 1)
        logger.info(
            "Recovery mode enabled (ratio=%.2f): %d/%d pending prompts scheduled for multi-turn recovery.",
            args.recovery_ratio,
            len(recovery_indices),
            len(pending),
        )

    try:
        for idx, item in enumerate(pending, start=1):
            item_id: str = item["id"]
            user_prompt: str = item["prompt"]
            raw_image_path: str | None = item.get("image_path")
            expected_tool: str | None = item.get("expected_tool")
            is_recovery = ((idx - 1) in recovery_indices) or (item.get("mode") == "recovery")

            resolved_img = resolve_image_path(raw_image_path, args.input)
            if resolved_img is not None and not resolved_img.exists():
                msg = f"Image file not found: {resolved_img}"
                logger.error("[%d/%d] %s — %s", idx, len(pending), item_id, msg)
                append_jsonl(
                    args.failed_out,
                    {
                        "id": item_id,
                        "timestamp": utc_now_iso(),
                        "stage": "input_validation",
                        "model": args.model,
                        "image_path": raw_image_path,
                        "reasons": [msg],
                    },
                )
                stats["failed_exec"] += 1
                continue

            recovery_turn: tuple[str, str] | None = None
            if is_recovery:
                recovery_turn = build_simulated_tool_failure(expected_tool)
                teacher_prompt = build_recovery_teacher_prompt(
                    user_prompt,
                    bad_call_json=recovery_turn[0],
                    tool_error_msg=recovery_turn[1],
                    has_image=resolved_img is not None,
                )
            else:
                teacher_prompt = build_teacher_prompt(user_prompt, has_image=resolved_img is not None)

            logger.info(
                "[%d/%d] Running id=%s (image=%s, mode=%s)...",
                idx,
                len(pending),
                item_id,
                raw_image_path or "none",
                "recovery" if is_recovery else "standard",
            )

            result = execute_with_retries(
                opencode_bin=opencode_bin,
                model_id=args.model,
                prompt_text=teacher_prompt,
                image_path=resolved_img,
                max_retries=args.max_retries,
            )

            if not result.ok:
                logger.error(
                    "[%d/%d] Execution failed for id=%s after %d attempt(s): %s",
                    idx,
                    len(pending),
                    item_id,
                    result.attempts,
                    result.error_reason,
                )
                append_jsonl(
                    args.failed_out,
                    {
                        "id": item_id,
                        "timestamp": utc_now_iso(),
                        "stage": "execution",
                        "model": args.model,
                        "attempts": result.attempts,
                        "returncode": result.returncode,
                        "reasons": [result.error_reason or "execution failed"],
                        "stderr": result.stderr,
                        "raw_stdout": result.stdout,
                    },
                )
                stats["failed_exec"] += 1
            else:
                validation = validate_trace(
                    cot=result.parsed.cot,
                    answer=result.parsed.answer,
                    expected_tool=expected_tool,
                    finish_reason=result.parsed.finish_reason,
                )

                raw_record = {
                    "id": item_id,
                    "timestamp": utc_now_iso(),
                    "model": args.model,
                    "prompt": user_prompt,
                    "image_path": raw_image_path,
                    "expected_tool": expected_tool,
                    "session_id": result.parsed.session_id,
                    "finish_reason": result.parsed.finish_reason,
                    "tokens": result.parsed.tokens,
                    "cost": result.parsed.cost,
                    "attempts": result.attempts,
                    "cot": result.parsed.cot,
                    "answer": result.parsed.answer,
                    "is_valid": validation.is_valid,
                    "is_recovery": is_recovery,
                    "validation_reasons": validation.reasons,
                }
                append_jsonl(args.raw_out, raw_record)
                stats["processed"] += 1

                if validation.is_valid and validation.normalized_tool_json is not None:
                    sft_record = format_qwen35_sft_record(
                        item_id=item_id,
                        user_prompt=user_prompt,
                        cot=result.parsed.cot,
                        valid_tool_call_json=validation.normalized_tool_json,
                        image_path_str=raw_image_path,
                        recovery_turn=recovery_turn,
                    )
                    append_jsonl(args.clean_out, sft_record)
                    stats["clean"] += 1
                    if is_recovery:
                        stats["recovery"] += 1
                    logger.info(
                        "[%d/%d] PASS id=%s (cot=%d chars, tool=%s)",
                        idx,
                        len(pending),
                        item_id,
                        len(result.parsed.cot),
                        validation.parsed_tool_call.get("tool") if validation.parsed_tool_call else "?",
                    )
                else:
                    stats["rejected"] += 1
                    logger.warning(
                        "[%d/%d] REJECT id=%s: %s",
                        idx,
                        len(pending),
                        item_id,
                        "; ".join(validation.reasons),
                    )
                    append_jsonl(
                        args.failed_out,
                        {
                            "id": item_id,
                            "timestamp": utc_now_iso(),
                            "stage": "rejection_sampling",
                            "model": args.model,
                            "expected_tool": expected_tool,
                            "finish_reason": result.parsed.finish_reason,
                            "reasons": validation.reasons,
                            "cot": result.parsed.cot,
                            "answer": result.parsed.answer,
                            "raw_stdout": result.stdout,
                        },
                    )

            # Rate-limit delay between requests (skip sleep after final item)
            if idx < len(pending) and args.sleep > 0:
                logger.info("Sleeping %.1fs before next request...", args.sleep)
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C). All completed items are safely checkpointed.")

    logger.info(
        "Run complete | processed=%d | clean=%d | rejected=%d | failed_exec=%d",
        stats["processed"],
        stats["clean"],
        stats["rejected"],
        stats["failed_exec"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
