"""Verifiable deterministic RLVR reward functions for Qwen3.5 tool-calling GRPO."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

COT_BUILDER_DIR = Path(__file__).resolve().parent.parent / "cot_dataset_builder"
THINK_CLOSE_RE = re.compile(r"^(.*?)</think>\s*(.*)$", re.DOTALL | re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL | re.IGNORECASE)


def _load_cot_builder_schemas() -> tuple[dict[str, dict[str, Any]], str]:
    """Load TOOL_SCHEMAS and SYSTEM_PROMPT from `cot_dataset_builder/config.py` without module-name collision."""
    cfg_path = COT_BUILDER_DIR / "config.py"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Could not find {cfg_path} to import TOOL_SCHEMAS.")
    spec = importlib.util.spec_from_file_location("_cot_dataset_builder_config", str(cfg_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TOOL_SCHEMAS, module.SYSTEM_PROMPT


TOOL_SCHEMAS, BASE_SYSTEM_PROMPT = _load_cot_builder_schemas()


def _is_int_strict(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number_strict(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_against_schema(value: Any, schema: dict[str, Any], path: str = "arguments") -> list[str]:
    """Validate a parsed JSON object against a tool's parameter JSON Schema."""
    errors: list[str] = []
    expected_type = schema.get("type")

    if expected_type == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected object, got {type(value).__name__}"]
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}.{req}: missing required field")
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for extra in sorted(set(value.keys()) - set(props.keys())):
                errors.append(f"{path}.{extra}: unexpected property")
        for key, prop_schema in props.items():
            if key in value:
                errors.extend(_validate_against_schema(value[key], prop_schema, f"{path}.{key}"))

    elif expected_type == "array":
        if not isinstance(value, list):
            return [f"{path}: expected array, got {type(value).__name__}"]
        if (min_items := schema.get("minItems")) is not None and len(value) < min_items:
            errors.append(f"{path}: len {len(value)} < minItems {min_items}")
        if (max_items := schema.get("maxItems")) is not None and len(value) > max_items:
            errors.append(f"{path}: len {len(value)} > maxItems {max_items}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                errors.extend(_validate_against_schema(item, item_schema, f"{path}[{idx}]"))

    elif expected_type == "string":
        if not isinstance(value, str):
            return [f"{path}: expected string, got {type(value).__name__}"]
        if (min_len := schema.get("minLength")) is not None and len(value) < min_len:
            errors.append(f"{path}: len {len(value)} < minLength {min_len}")
        if (max_len := schema.get("maxLength")) is not None and len(value) > max_len:
            errors.append(f"{path}: len {len(value)} > maxLength {max_len}")

    elif expected_type == "integer":
        if not _is_int_strict(value):
            return [f"{path}: expected integer, got {type(value).__name__}"]
        if (minimum := schema.get("minimum")) is not None and value < minimum:
            errors.append(f"{path}: {value} < minimum {minimum}")
        if (maximum := schema.get("maximum")) is not None and value > maximum:
            errors.append(f"{path}: {value} > maximum {maximum}")

    elif expected_type == "number":
        if not _is_number_strict(value):
            return [f"{path}: expected number, got {type(value).__name__}"]
        if (minimum := schema.get("minimum")) is not None and value < minimum:
            errors.append(f"{path}: {value} < minimum {minimum}")
        if (maximum := schema.get("maximum")) is not None and value > maximum:
            errors.append(f"{path}: {value} > maximum {maximum}")

    elif expected_type == "boolean":
        if not isinstance(value, bool):
            return [f"{path}: expected boolean, got {type(value).__name__}"]

    if (enum_vals := schema.get("enum")) is not None and value not in enum_vals:
        errors.append(f"{path}: {value!r} not in enum {enum_vals}")

    return errors


def _parse_think_and_json(text: str) -> tuple[bool, str, str]:
    """Extract `(has_closed_think, think_text, answer_text)` from a model completion.

    Supports both completions that begin after the prompt's `<think>\\n` prefix
    and full assistant strings starting with `<think>`.
    """
    cleaned = text.strip()
    if cleaned.lower().startswith("<think>"):
        cleaned = cleaned[len("<think>") :].lstrip("\n")

    match = THINK_CLOSE_RE.match(cleaned)
    if match:
        return True, match.group(1).strip(), match.group(2).strip()
    return False, cleaned, ""


def _parse_json_object(answer_text: str) -> tuple[dict[str, Any] | None, str | None]:
    if not answer_text:
        return None, "empty answer after </think>"
    candidate = answer_text.strip()
    fence = CODE_FENCE_RE.match(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    if not isinstance(parsed, dict):
        return None, f"expected JSON object, got {type(parsed).__name__}"
    return parsed, None


def _estimate_token_count(text: str, tokenizer: Any = None) -> int:
    if not text:
        return 0
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return max(1, int(len(text.split()) * 1.3))


def score_completion(
    text: str,
    expected_tool: str | None = None,
    max_think_tokens: int = 256,
    tokenizer: Any = None,
    tool_schemas: dict[str, dict[str, Any]] = TOOL_SCHEMAS,
) -> dict[str, Any]:
    """Compute verifiable tool-calling reward and sub-scores for a single completion.

    Reward breakdown:
    - Format reward (+0.15): Closed `</think>` with non-empty reasoning + valid JSON object.
    - Schema validity (+0.35): Valid JSON passing `TOOL_SCHEMAS` validation.
    - Tool match (+0.50): Exact match with `expected_tool` (when provided).
    - Length / anti-ramble penalty (-0.05 to -0.20): Truncated without `</think>` (-0.20)
      or `<think>` exceeding `max_think_tokens` (-0.05 to -0.20).
    """
    has_closed_think, think_text, answer_text = _parse_think_and_json(text)
    think_tokens = _estimate_token_count(think_text, tokenizer=tokenizer)

    parsed_obj, json_err = _parse_json_object(answer_text) if has_closed_think else (None, "missing </think>")

    # 1. Format reward (+0.15)
    format_reward = 0.15 if (has_closed_think and len(think_text) >= 10 and parsed_obj is not None) else 0.0

    # 2. Schema validity reward (+0.35)
    schema_reward = 0.0
    schema_errors: list[str] = []
    parsed_tool: str | None = None

    if parsed_obj is not None:
        extra_top = sorted(set(parsed_obj.keys()) - {"tool", "arguments"})
        if extra_top:
            schema_errors.append(f"unexpected top-level keys: {extra_top}")

        tool_name = parsed_obj.get("tool")
        arguments = parsed_obj.get("arguments")
        if not isinstance(tool_name, str) or not tool_name.strip():
            schema_errors.append("missing or invalid 'tool'")
        else:
            parsed_tool = tool_name
            if tool_name not in tool_schemas:
                schema_errors.append(f"unknown tool {tool_name!r}")
            elif not isinstance(arguments, dict):
                schema_errors.append("missing or invalid 'arguments'")
            else:
                schema_errors.extend(
                    _validate_against_schema(arguments, tool_schemas[tool_name].get("parameters", {}))
                )

        if not schema_errors:
            schema_reward = 0.35
    elif json_err:
        schema_errors.append(json_err)

    # 3. Tool match reward (+0.50)
    tool_match_reward = 0.50 if (expected_tool and parsed_tool == expected_tool) else 0.0

    # 4. Length / anti-ramble penalty (-0.05 to -0.20)
    length_penalty = 0.0
    if not has_closed_think:
        length_penalty = -0.20
    elif think_tokens > max_think_tokens:
        overflow_ratio = (think_tokens - max_think_tokens) / max(1, max_think_tokens)
        length_penalty = -min(0.20, 0.05 + 0.15 * overflow_ratio)

    total_reward = format_reward + schema_reward + tool_match_reward + length_penalty

    return {
        "total_reward": round(float(total_reward), 6),
        "format_reward": format_reward,
        "schema_reward": schema_reward,
        "tool_match_reward": tool_match_reward,
        "length_penalty": round(float(length_penalty), 6),
        "has_closed_think": has_closed_think,
        "think_tokens": think_tokens,
        "parsed_tool": parsed_tool,
        "schema_errors": schema_errors,
    }
