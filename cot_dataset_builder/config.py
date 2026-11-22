"""Configuration for the OpenCode DeepSeek Visual SRE / DevOps Incident Triage CoT pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = BASE_DIR / "prompts.jsonl"
RAW_DATASET_FILE = BASE_DIR / "dataset_raw.jsonl"
CLEAN_DATASET_FILE = BASE_DIR / "dataset_clean.jsonl"
FAILED_TRACES_FILE = BASE_DIR / "failed_traces.jsonl"

# ---------------------------------------------------------------------------
# OpenCode CLI & Model Configuration
# ---------------------------------------------------------------------------
MODEL_ID = "opencode/deepseek-v4.1-flash-free"

# Rate-limiting & retry configuration
SLEEP_SECONDS: float = 10.0
MAX_RETRIES: int = 5
BACKOFF_BASE_SECONDS: float = 15.0
BACKOFF_MAX_SECONDS: float = 300.0
BACKOFF_JITTER_RATIO: float = 0.2
SUBPROCESS_TIMEOUT_SECONDS: int = 180

# Rejection-sampling thresholds
MIN_COT_CHARS: int = 20
REQUIRE_EXPECTED_TOOL_MATCH: bool = True

# Lowercase substrings used to detect rate-limiting / transient provider throttling
THROTTLE_PATTERNS: tuple[str, ...] = (
    "429",
    "rate limit",
    "rate_limit",
    "too many requests",
    "overloaded",
    "quota",
    "throttle",
    "temporarily unavailable",
    "capacity",
    "503",
    "529",
)

# ---------------------------------------------------------------------------
# Visual SRE / DevOps Incident Triage JSON Tool Schemas
# Every tool call emitted by the model must match:
#   {"tool": "<tool_name>", "arguments": { ... }}
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "trigger_rollback": {
        "description": (
            "Roll back a service to a previous known-good version when a sharp HTTP 5xx or "
            "p99 latency spike occurs immediately after a recent deployment marker."
        ),
        "parameters": {
            "type": "object",
            "required": ["service", "target_version", "reason", "environment"],
            "additionalProperties": False,
            "properties": {
                "service": {"type": "string", "minLength": 1},
                "target_version": {"type": "string", "minLength": 2},
                "reason": {"type": "string", "minLength": 5},
                "environment": {"type": "string", "enum": ["prod", "staging"]},
            },
        },
    },
    "scale_service": {
        "description": (
            "Scale horizontal replicas for a service experiencing an organic traffic/RPS surge "
            "causing CPU saturation (>85%) while error rates remain healthy (<1%)."
        ),
        "parameters": {
            "type": "object",
            "required": ["service", "target_replicas", "region", "reason"],
            "additionalProperties": False,
            "properties": {
                "service": {"type": "string", "minLength": 1},
                "target_replicas": {"type": "integer", "minimum": 1, "maximum": 500},
                "region": {"type": "string", "enum": ["us-east-1", "eu-west-1", "ap-south-1"]},
                "reason": {"type": "string", "minLength": 5},
            },
        },
    },
    "restart_pod_pool": {
        "description": (
            "Restart a service's pod pool when telemetry shows a memory leak / OOMKill sawtooth "
            "pattern with RAM climbing linearly to 100% and repeated pod crash-loops."
        ),
        "parameters": {
            "type": "object",
            "required": ["service", "namespace", "strategy", "reason"],
            "additionalProperties": False,
            "properties": {
                "service": {"type": "string", "minLength": 1},
                "namespace": {"type": "string", "minLength": 1},
                "strategy": {"type": "string", "enum": ["rolling", "immediate"]},
                "reason": {"type": "string", "minLength": 5},
            },
        },
    },
    "query_logs": {
        "description": (
            "Query structured service logs when telemetry shows ambiguous intermittent latency "
            "or downstream dependency jitter where the dashboard alone cannot confirm root cause."
        ),
        "parameters": {
            "type": "object",
            "required": ["service", "time_window_minutes", "severity", "filter_query"],
            "additionalProperties": False,
            "properties": {
                "service": {"type": "string", "minLength": 1},
                "time_window_minutes": {"type": "integer", "minimum": 1, "maximum": 1440},
                "severity": {"type": "string", "enum": ["ERROR", "WARN", "FATAL"]},
                "filter_query": {"type": "string", "minLength": 1},
            },
        },
    },
    "create_incident": {
        "description": (
            "Declare a formal incident and page on-call when telemetry shows a multi-service "
            "cascading outage or critical SLO breach across multiple panels."
        ),
        "parameters": {
            "type": "object",
            "required": ["severity", "affected_services", "summary", "page_oncall"],
            "additionalProperties": False,
            "properties": {
                "severity": {"type": "string", "enum": ["SEV1", "SEV2", "SEV3"]},
                "affected_services": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "summary": {"type": "string", "minLength": 5},
                "page_oncall": {"type": "boolean"},
            },
        },
    },
}

# ---------------------------------------------------------------------------
# System Prompt (stored in SFT dataset & sent to OpenCode teacher model)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert Site Reliability Engineering (SRE) incident triage assistant. "
    "Inspect the telemetry dashboard or alert logs step by step to distinguish between "
    "deployment regressions (trigger_rollback), organic traffic saturation (scale_service), "
    "memory leak OOMKill sawtooths (restart_pod_pool), ambiguous downstream jitter (query_logs), "
    "and multi-service cascading outages (create_incident). "
    "Select the single best tool from the available schemas and respond with ONLY a valid JSON object:\n"
    '{"tool": "<tool_name>", "arguments": { ... }}\n'
    "Do not include any prose, explanation, or markdown formatting outside the JSON object."
)


def build_teacher_prompt(user_prompt: str, has_image: bool = False) -> str:
    """Build the full non-interactive prompt passed to `opencode run`."""
    schema_json = json.dumps(TOOL_SCHEMAS, indent=2, ensure_ascii=False)
    image_note = (
        "A dark-mode Grafana telemetry dashboard screenshot is attached via `-f`. "
        "Carefully read the visual curves, deploy markers, thresholds, and header metadata before deciding.\n"
        if has_image
        else ""
    )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "IMPORTANT EXECUTION RULES:\n"
        "- Do NOT call any filesystem, shell, web, or OpenCode built-in tools.\n"
        "- Think carefully through the telemetry evidence, then output ONLY the single raw JSON object.\n"
        "- Every required argument must match the schema types and enum values exactly.\n\n"
        f"AVAILABLE TOOL SCHEMAS:\n{schema_json}\n\n"
        f"{image_note}"
        f"INCIDENT ALERT / REQUEST:\n{user_prompt}"
    )
