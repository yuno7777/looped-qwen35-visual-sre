"""Rejection-sampling validator for CoT reasoning traces and structured tool calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from config import MIN_COT_CHARS, REQUIRE_EXPECTED_TOOL_MATCH, TOOL_SCHEMAS

TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens", "max_output_tokens", "truncated"})
CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)


@dataclass
class ValidationResult:
    """Outcome of validating a single model trace."""

    is_valid: bool
    reasons: list[str] = field(default_factory=list)
    normalized_tool_json: str | None = None
    parsed_tool_call: dict[str, Any] | None = None


def _is_int_strict(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number_strict(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_against_schema(value: Any, schema: dict[str, Any], path: str = "arguments") -> list[str]:
    """Validate a Python object against a subset of JSON Schema (stdlib-only)."""
    errors: list[str] = []
    expected_type = schema.get("type")

    if expected_type == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected object, got {type(value).__name__}"]

        required_fields = schema.get("required", [])
        for req in required_fields:
            if req not in value:
                errors.append(f"{path}.{req}: missing required field")

        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra_keys = sorted(set(value.keys()) - set(properties.keys()))
            for key in extra_keys:
                errors.append(f"{path}.{key}: unexpected additional property")

        for key, prop_schema in properties.items():
            if key in value:
                errors.extend(validate_against_schema(value[key], prop_schema, f"{path}.{key}"))

    elif expected_type == "array":
        if not isinstance(value, list):
            return [f"{path}: expected array, got {type(value).__name__}"]
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            errors.append(f"{path}: array length {len(value)} < minItems {min_items}")
        if max_items is not None and len(value) > max_items:
            errors.append(f"{path}: array length {len(value)} > maxItems {max_items}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                errors.extend(validate_against_schema(item, item_schema, f"{path}[{idx}]"))

    elif expected_type == "string":
        if not isinstance(value, str):
            return [f"{path}: expected string, got {type(value).__name__}"]
        min_len = schema.get("minLength")
        max_len = schema.get("maxLength")
        if min_len is not None and len(value) < min_len:
            errors.append(f"{path}: string length {len(value)} < minLength {min_len}")
        if max_len is not None and len(value) > max_len:
            errors.append(f"{path}: string length {len(value)} > maxLength {max_len}")

    elif expected_type == "integer":
        if not _is_int_strict(value):
            return [f"{path}: expected integer, got {type(value).__name__}"]
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            errors.append(f"{path}: value {value} < minimum {minimum}")
        if maximum is not None and value > maximum:
            errors.append(f"{path}: value {value} > maximum {maximum}")

    elif expected_type == "number":
        if not _is_number_strict(value):
            return [f"{path}: expected number, got {type(value).__name__}"]
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            errors.append(f"{path}: value {value} < minimum {minimum}")
        if maximum is not None and value > maximum:
            errors.append(f"{path}: value {value} > maximum {maximum}")

    elif expected_type == "boolean":
        if not isinstance(value, bool):
            return [f"{path}: expected boolean, got {type(value).__name__}"]

    elif expected_type == "null":
        if value is not None:
            return [f"{path}: expected null, got {type(value).__name__}"]

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        errors.append(f"{path}: value {value!r} not in allowed enum {enum_values}")

    return errors


def validate_cot(
    cot: str | None,
    finish_reason: str | None = None,
    min_cot_chars: int = MIN_COT_CHARS,
) -> list[str]:
    """Check that Chain-of-Thought reasoning is present and not truncated."""
    reasons: list[str] = []

    if finish_reason and finish_reason.strip().lower() in TRUNCATED_FINISH_REASONS:
        reasons.append(f"trace truncated (finish_reason={finish_reason!r})")

    if not cot or not cot.strip():
        reasons.append("empty CoT reasoning block")
        return reasons

    cleaned = cot.strip()
    if len(cleaned) < min_cot_chars:
        reasons.append(f"CoT too short ({len(cleaned)} chars < {min_cot_chars})")

    # Detect unclosed <think> tags or dangling code fences inside CoT
    if cleaned.count("<think>") != cleaned.count("</think>"):
        reasons.append("CoT contains unclosed <think> tag")

    if cleaned.count("```") % 2 != 0:
        reasons.append("CoT contains unclosed code fence (likely truncated)")

    # Detect obvious abrupt mid-line truncation (e.g., ends with lone backslash or connector)
    if cleaned.endswith(("\\", ",", ":", ";", "->")):
        reasons.append("CoT ends abruptly with trailing delimiter (likely truncated)")

    return reasons


def extract_tool_call_json(answer: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Parse the assistant answer as a JSON object, allowing a single markdown code fence."""
    if not answer or not answer.strip():
        return None, "empty assistant answer"

    candidate = answer.strip()
    fence_match = CODE_FENCE_RE.match(candidate)
    if fence_match:
        candidate = fence_match.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"invalid tool-call JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})"

    if not isinstance(parsed, dict):
        return None, f"tool call must be a JSON object, got {type(parsed).__name__}"

    return parsed, None


def validate_tool_call(
    answer: str | None,
    expected_tool: str | None = None,
    tool_schemas: dict[str, dict[str, Any]] = TOOL_SCHEMAS,
    require_expected_match: bool = REQUIRE_EXPECTED_TOOL_MATCH,
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """Parse and schema-validate the final tool call JSON."""
    parsed, parse_error = extract_tool_call_json(answer)
    if parse_error or parsed is None:
        return None, None, [parse_error or "failed to parse tool-call JSON"]

    errors: list[str] = []

    allowed_top_keys = {"tool", "arguments"}
    extra_top = sorted(set(parsed.keys()) - allowed_top_keys)
    if extra_top:
        errors.append(f"unexpected top-level keys in tool call: {extra_top}")

    tool_name = parsed.get("tool")
    arguments = parsed.get("arguments")

    if not isinstance(tool_name, str) or not tool_name.strip():
        errors.append("missing or invalid 'tool' name (must be non-empty string)")
        return parsed, None, errors

    if tool_name not in tool_schemas:
        errors.append(f"unknown tool {tool_name!r}; allowed tools: {sorted(tool_schemas.keys())}")
        return parsed, None, errors

    if require_expected_match and expected_tool and tool_name != expected_tool:
        errors.append(f"tool mismatch: expected {expected_tool!r}, got {tool_name!r}")

    if not isinstance(arguments, dict):
        errors.append("missing or invalid 'arguments' field (must be JSON object)")
        return parsed, None, errors

    param_schema = tool_schemas[tool_name].get("parameters", {})
    errors.extend(validate_against_schema(arguments, param_schema, path="arguments"))

    if errors:
        return parsed, None, errors

    canonical = {"tool": tool_name, "arguments": arguments}
    normalized_json = json.dumps(canonical, ensure_ascii=False)
    return canonical, normalized_json, []


def validate_trace(
    cot: str | None,
    answer: str | None,
    expected_tool: str | None = None,
    finish_reason: str | None = None,
    tool_schemas: dict[str, dict[str, Any]] = TOOL_SCHEMAS,
) -> ValidationResult:
    """Run full rejection-sampling validation on a collected reasoning + tool-call trace."""
    reasons: list[str] = []
    reasons.extend(validate_cot(cot, finish_reason=finish_reason))

    parsed, normalized_json, tool_errors = validate_tool_call(
        answer=answer,
        expected_tool=expected_tool,
        tool_schemas=tool_schemas,
    )
    reasons.extend(tool_errors)

    is_valid = len(reasons) == 0 and normalized_json is not None
    return ValidationResult(
        is_valid=is_valid,
        reasons=reasons,
        normalized_tool_json=normalized_json if is_valid else None,
        parsed_tool_call=parsed,
    )
