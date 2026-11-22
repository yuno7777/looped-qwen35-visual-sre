"""Configuration dataclass for the Looped Hybrid Qwen3.5 wrapper (v1 & v2 ETD + ARLoop)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

VALID_KV_QUANT_MODES: tuple[str, ...] = ("none", "turboquant_3bit", "turboquant_4bit")
VALID_REINJECT_MODES: tuple[str, ...] = ("scalar", "arloop")

MODEL_SIZE_PRESETS: dict[str, str] = {
    "0.8b": "Qwen/Qwen3.5-0.8B",
    "2b": "Qwen/Qwen3.5-2B",
}


def resolve_model_id(model_size: str | None = None, base_model_id: str | None = None) -> str:
    """Resolve HF model ID from `--base-model-id` override or `--model-size` preset."""
    if base_model_id:
        return base_model_id
    if model_size:
        key = model_size.strip().lower()
        if key not in MODEL_SIZE_PRESETS:
            raise ValueError(f"Unsupported model_size {model_size!r}; choose from {sorted(MODEL_SIZE_PRESETS)}")
        return MODEL_SIZE_PRESETS[key]
    return MODEL_SIZE_PRESETS["0.8b"]


@dataclass
class LoopedQwen35Config:
    """Configuration for converting Qwen3.5 into a Looped Hybrid Transformer (v1 & v2)."""

    base_model_id: str = "Qwen/Qwen3.5-0.8B"
    num_loops: int = 2
    reinject_init_logit: float = 4.6
    reinject_mode: str = "scalar"  # "scalar" (v1) or "arloop" (v2 sliding-window depth attention)
    loop_start_layer: int = 0  # 0..24 full-stack (v1); 4..20 ETD middle-stack (v2)
    loop_end_layer: int = 24
    aux_loop_loss_weight: float = 0.1
    dtype: str = "bfloat16"
    kv_quant: str = "none"
    kv_residual_window: int = 64
    kv_protect_vision_prefix: bool = True
    kv_sink_tokens: int = 4
    load_in_4bit: bool = False

    def __post_init__(self) -> None:
        if self.num_loops < 1:
            raise ValueError(f"num_loops must be >= 1, got {self.num_loops}")
        if self.reinject_mode not in VALID_REINJECT_MODES:
            raise ValueError(
                f"Unsupported reinject_mode {self.reinject_mode!r}; choose from {VALID_REINJECT_MODES}"
            )
        if self.loop_start_layer < 0 or self.loop_end_layer <= self.loop_start_layer:
            raise ValueError(
                f"Invalid loop layer range [{self.loop_start_layer}, {self.loop_end_layer})"
            )
        if not (0.0 <= self.aux_loop_loss_weight <= 1.0):
            raise ValueError(
                f"aux_loop_loss_weight must be in [0.0, 1.0], got {self.aux_loop_loss_weight}"
            )
        if self.dtype.lower() not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype {self.dtype!r}; choose from {sorted(DTYPE_MAP)}")
        if self.kv_quant not in VALID_KV_QUANT_MODES:
            raise ValueError(
                f"Unsupported kv_quant {self.kv_quant!r}; choose from {VALID_KV_QUANT_MODES}"
            )
        if self.kv_residual_window < 0:
            raise ValueError(f"kv_residual_window must be >= 0, got {self.kv_residual_window}")
        if self.kv_sink_tokens < 0:
            raise ValueError(f"kv_sink_tokens must be >= 0, got {self.kv_sink_tokens}")

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPE_MAP[self.dtype.lower()]

    @property
    def is_etd_middle_loop(self) -> bool:
        return self.loop_start_layer > 0 or self.loop_end_layer < 24
