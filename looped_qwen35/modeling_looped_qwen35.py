"""Non-destructive Looped Hybrid Transformer wrapper for Qwen3.5 (v1 & v2: ETD + ARLoop + Vision-Exempt TurboQuant).

Built directly against `transformers.models.qwen3_5.modeling_qwen3_5` (transformers 5.24.0):
- Supports both v1 full-stack looping (layers 0..23) and v2 ETD middle-stack looping
  (Encoder 0..3 once, Looped Reasoning Core 4..19 x num_loops, Decoder 20..23 once).
- Supports both v1 scalar input re-injection (`reinject_mode="scalar"`) and v2 ARLoop
  sliding-window depth-attention residuals (`reinject_mode="arloop"`).
- Uses isolated `PerLoopHybridCache` per loop for the looped core (while sharing a single
  cache for non-looped ETD Encoder/Decoder layers), with optional Vision-Prefix-Exempt
  TurboQuant (~3-bit / 4-bit) strictly on the 6 Gated Attention layers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import nn
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import slice_logits_to_keep

from config import LoopedQwen35Config
from turboquant_cache import (
    TurboQuantCompressor,
    TurboQuantHybridCache,
    measure_hybrid_cache_bytes,
)


@dataclass
class LoopedCausalLMOutput:
    """Output container for `LoopedQwen3_5ForConditionalGeneration.forward`."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    per_loop_caches: PerLoopHybridCache | None = None
    per_loop_losses: list[torch.Tensor] = field(default_factory=list)
    final_loop_loss: torch.Tensor | None = None
    aux_loop_loss: torch.Tensor | None = None
    reinject_weight: float | None = None
    image_outputs: BaseModelOutputWithPooling | None = None
    video_outputs: BaseModelOutputWithPooling | None = None


class ARLoopDepthGate(nn.Module):
    """Sliding-window depth-attention residual gate (ARLoop, arXiv:2611.05234).

    At loop step `t >= 1`, computes per-token softmax attention weights over the depth
    window `[E_enc, h^(t-1), h^(t)]` using an RMSNorm + learned query vector `q_t`.
    Initialized smoothly (`q_t = 0`, `anchor_bias = 4.6`) so step 0 training starts
    near `0.99 * E_enc + 0.01 * h^(t)` with zero NaNs.
    """

    def __init__(
        self,
        hidden_size: int,
        max_loops: int = 8,
        init_anchor_bias: float = 4.6,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_loops = max_loops
        self.scale = hidden_size**-0.5

        self.norm = nn.RMSNorm(hidden_size, eps=1e-6, device=device, dtype=torch.float32)
        self.loop_queries = nn.Parameter(torch.zeros((max_loops, hidden_size), dtype=torch.float32, device=device))
        self.anchor_bias = nn.Parameter(torch.tensor(init_anchor_bias, dtype=torch.float32, device=device))

    def forward(self, loop_idx: int, states: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate depth states `[E_enc, ..., h^(t)]` -> `(blended_hidden, mean_anchor_weight)`."""
        # V: (B, S, K, D)
        v_stack = torch.stack(states, dim=-2)
        k_norm = self.norm(v_stack.float())
        q_idx = min(loop_idx, self.max_loops - 1)
        q_vec = self.loop_queries[q_idx]  # (D,)

        # scores: (B, S, K)
        scores = torch.einsum("bskd,d->bsk", k_norm, q_vec) * self.scale
        bias = torch.zeros((v_stack.shape[-2],), dtype=torch.float32, device=scores.device)
        bias[0] = self.anchor_bias
        scores = scores + bias.view(1, 1, -1)

        weights = torch.softmax(scores, dim=-1)
        blended = torch.einsum("bsk,bskd->bsd", weights.to(v_stack.dtype), v_stack)
        mean_anchor_w = weights[..., 0].detach().mean()
        return blended, mean_anchor_w


class PerLoopHybridCache:
    """Holds one independent hybrid cache (`DynamicCache` or `TurboQuantHybridCache`) per loop iteration.

    In v2 ETD mode (`loop_start_layer=4, loop_end_layer=20`), Encoder layers (`0..3`) and
    Decoder layers (`20..23`) use `caches[0]` exclusively while `caches[1:]` only populate
    the looped Reasoning Core (`4..19`), cutting extra-loop cache memory by 33%.
    """

    def __init__(
        self,
        text_config: Any,
        num_loops: int,
        kv_quant: str = "none",
        residual_window: int = 64,
        protect_vision_prefix: bool = True,
        sink_tokens: int = 4,
        seed: int = 42,
    ) -> None:
        if num_loops < 1:
            raise ValueError(f"num_loops must be >= 1, got {num_loops}")
        if kv_quant not in ("none", "turboquant_3bit", "turboquant_4bit"):
            raise ValueError(f"Unsupported kv_quant mode: {kv_quant!r}")

        self.text_config = text_config
        self.num_loops = num_loops
        self.kv_quant = kv_quant
        self.residual_window = residual_window
        self.protect_vision_prefix = protect_vision_prefix
        self.sink_tokens = sink_tokens
        self.seed = seed

        self._shared_compressor: TurboQuantCompressor | None = None
        if kv_quant in ("turboquant_3bit", "turboquant_4bit"):
            bits = 3 if kv_quant == "turboquant_3bit" else 4
            head_dim = getattr(text_config, "head_dim", None) or (
                text_config.hidden_size // text_config.num_attention_heads
            )
            self._shared_compressor = TurboQuantCompressor(head_dim=head_dim, bits=bits, seed=seed)

        self.caches: list[DynamicCache] = self._build_caches()

    def _build_caches(self) -> list[DynamicCache]:
        if self._shared_compressor is not None:
            return [
                TurboQuantHybridCache(
                    config=self.text_config,
                    compressor=self._shared_compressor,
                    residual_window=self.residual_window,
                    protect_vision_prefix=self.protect_vision_prefix,
                    sink_tokens=self.sink_tokens,
                    seed=self.seed,
                )
                for _ in range(self.num_loops)
            ]
        return [DynamicCache(config=self.text_config) for _ in range(self.num_loops)]

    def set_prefill_protected_mask(self, mask: torch.Tensor | None) -> None:
        for cache in self.caches:
            if isinstance(cache, TurboQuantHybridCache):
                cache.set_prefill_protected_mask(mask)

    def __getitem__(self, loop_idx: int) -> DynamicCache:
        return self.caches[loop_idx]

    def __len__(self) -> int:
        return len(self.caches)

    def get_seq_length(self) -> int:
        """Return cached sequence length (caches[0] always tracks all 24 layers)."""
        return self.caches[0].get_seq_length() if self.caches else 0

    def reset(self) -> None:
        self.caches = self._build_caches()

    def kv_cache_memory_bytes(self) -> dict[str, int]:
        """Return exact byte totals across all loops split by Attention KV (protected/quant/res) vs DeltaNet."""
        attn_total = 0
        prot_total = 0
        quant_total = 0
        res_total = 0
        delta_total = 0
        for cache in self.caches:
            stats = measure_hybrid_cache_bytes(cache)
            attn_total += stats["attention_kv_bytes"]
            prot_total += stats["protected_bf16_bytes"]
            quant_total += stats["quantized_bytes"]
            res_total += stats["residual_bf16_bytes"]
            delta_total += stats["deltanet_state_bytes"]
        return {
            "attention_kv_bytes": attn_total,
            "protected_bf16_bytes": prot_total,
            "quantized_bytes": quant_total,
            "residual_bf16_bytes": res_total,
            "deltanet_state_bytes": delta_total,
            "deltanet_bytes": delta_total,
            "total_bytes": attn_total + delta_total,
        }


class LoopedQwen3_5ForConditionalGeneration(nn.Module):
    """Non-destructive wrapper turning `Qwen3_5ForConditionalGeneration` into a Looped Hybrid Transformer."""

    def __init__(
        self,
        base_model: Qwen3_5ForConditionalGeneration,
        loop_config: LoopedQwen35Config | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.loop_config = loop_config or LoopedQwen35Config()
        self.num_loops = self.loop_config.num_loops
        self.aux_loop_loss_weight = self.loop_config.aux_loop_loss_weight
        self.reinject_mode = self.loop_config.reinject_mode

        num_layers = int(self.text_config.num_hidden_layers)
        self.loop_start_layer = max(0, min(self.loop_config.loop_start_layer, num_layers - 1))
        self.loop_end_layer = max(self.loop_start_layer + 1, min(self.loop_config.loop_end_layer, num_layers))

        param_device = next(base_model.parameters()).device
        self.reinject_logit = nn.Parameter(
            torch.tensor(self.loop_config.reinject_init_logit, dtype=torch.float32, device=param_device)
        )
        self.arloop_gate: ARLoopDepthGate | None = None
        if self.reinject_mode == "arloop":
            self.arloop_gate = ARLoopDepthGate(
                hidden_size=int(self.text_config.hidden_size),
                max_loops=max(8, self.num_loops + 2),
                init_anchor_bias=self.loop_config.reinject_init_logit,
                device=param_device,
            )

        self._last_per_loop_caches: PerLoopHybridCache | None = None
        self._last_reinject_weight: float = float(torch.sigmoid(self.reinject_logit).item())

    @property
    def _core_model(self) -> Qwen3_5ForConditionalGeneration:
        """Unwrap PeftModel if `self.base_model` has LoRA adapters attached."""
        if hasattr(self.base_model, "get_base_model"):
            return self.base_model.get_base_model()
        return self.base_model

    @property
    def config(self) -> Any:
        return self._core_model.config

    @property
    def text_config(self) -> Any:
        return self._core_model.config.text_config

    @property
    def reinject_weight(self) -> torch.Tensor:
        if self.reinject_mode == "arloop" and self.arloop_gate is not None:
            return torch.sigmoid(self.arloop_gate.anchor_bias)
        return torch.sigmoid(self.reinject_logit)

    def get_loop_trainable_parameters(self) -> list[nn.Parameter]:
        """Return trainable loop-control parameters (`reinject_logit` or `arloop_gate` params)."""
        if self.reinject_mode == "arloop" and self.arloop_gate is not None:
            return list(self.arloop_gate.parameters())
        return [self.reinject_logit]

    def export_loop_state(self) -> dict[str, Any]:
        """Serialize loop parameters and config for `loop_state.pt`."""
        import dataclasses

        state: dict[str, Any] = {
            "reinject_logit": self.reinject_logit.detach().cpu(),
            "reinject_weight": float(self.reinject_weight.detach().float().item()),
            "loop_config": dataclasses.asdict(self.loop_config),
        }
        if self.arloop_gate is not None:
            state["arloop_gate_state"] = {
                k: v.detach().cpu() for k, v in self.arloop_gate.state_dict().items()
            }
        return state

    def restore_loop_state(self, loop_state: dict[str, Any], device: torch.device) -> None:
        """Restore `reinject_logit` and optional `arloop_gate` weights from `loop_state.pt`."""
        with torch.no_grad():
            if "reinject_logit" in loop_state:
                self.reinject_logit.copy_(loop_state["reinject_logit"].to(device=device, dtype=torch.float32))
            if self.arloop_gate is not None and "arloop_gate_state" in loop_state:
                sd = {k: v.to(device=device) for k, v in loop_state["arloop_gate_state"].items()}
                self.arloop_gate.load_state_dict(sd)

    def enable_gradient_checkpointing(self) -> None:
        """Enable non-reentrant gradient checkpointing on the text decoder (compatible with looping + PEFT)."""
        core = self._core_model
        core.model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(core, "enable_input_require_grads"):
            core.enable_input_require_grads()

    @classmethod
    def from_pretrained(
        cls,
        loop_config: LoopedQwen35Config | None = None,
        device_map: str | dict[str, Any] | None = None,
        device: str | torch.device | None = None,
        **from_pretrained_kwargs: Any,
    ) -> LoopedQwen3_5ForConditionalGeneration:
        """Load a standard pretrained Qwen3.5 checkpoint (bf16 or optional 4-bit QLoRA) and wrap it."""
        cfg = loop_config or LoopedQwen35Config()
        load_kwargs: dict[str, Any] = {
            "dtype": cfg.torch_dtype,
            **from_pretrained_kwargs,
        }
        if cfg.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise ImportError(
                    "`load_in_4bit=True` requires `bitsandbytes`. Install with `pip install bitsandbytes`."
                ) from exc
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=cfg.torch_dtype,
            )
            if device_map is None:
                device_map = "auto"

        if device_map is not None:
            load_kwargs["device_map"] = device_map

        base_model = Qwen3_5ForConditionalGeneration.from_pretrained(cfg.base_model_id, **load_kwargs)
        if device is not None and device_map is None and not cfg.load_in_4bit:
            base_model = base_model.to(device)
        return cls(base_model=base_model, loop_config=cfg)

    @staticmethod
    def load_processor(model_id: str = "Qwen/Qwen3.5-0.8B", **kwargs: Any) -> Any:
        """Load the matching `AutoProcessor` for Qwen3.5 multimodal inputs."""
        return AutoProcessor.from_pretrained(model_id, **kwargs)

    def create_per_loop_caches(
        self,
        num_loops: int | None = None,
        kv_quant: str | None = None,
        residual_window: int | None = None,
        protect_vision_prefix: bool | None = None,
        sink_tokens: int | None = None,
    ) -> PerLoopHybridCache:
        """Create isolated hybrid caches (Gated DeltaNet + Gated Attention) for each loop."""
        loops = self.num_loops if num_loops is None else num_loops
        quant_mode = self.loop_config.kv_quant if kv_quant is None else kv_quant
        window = self.loop_config.kv_residual_window if residual_window is None else residual_window
        prot_vis = (
            self.loop_config.kv_protect_vision_prefix
            if protect_vision_prefix is None
            else protect_vision_prefix
        )
        sinks = self.loop_config.kv_sink_tokens if sink_tokens is None else sink_tokens
        return PerLoopHybridCache(
            self.text_config,
            loops,
            kv_quant=quant_mode,
            residual_window=window,
            protect_vision_prefix=prot_vis,
            sink_tokens=sinks,
        )

    def _compute_prefill_protected_mask(
        self,
        input_ids: torch.LongTensor | None,
        seq_len: int,
        sink_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Identify sink positions (`0..sink_tokens-1`) + all vision token positions for bf16 protection."""
        mask = torch.zeros((seq_len,), dtype=torch.bool, device=device)
        if sink_tokens > 0:
            mask[: min(sink_tokens, seq_len)] = True
        if input_ids is not None:
            ids_1d = input_ids[0]
            special_vis_ids = [
                getattr(self.config, attr, None)
                for attr in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
            ]
            for tid in special_vis_ids:
                if tid is not None:
                    mask |= ids_1d == int(tid)
        return mask

    def _build_multimodal_inputs(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, BaseModelOutputWithPooling]]:
        """Run Qwen3.5's native vision encoder and scatter visual embeddings into `inputs_embeds`."""
        qwen_model = self._core_model.model

        if (pixel_values is not None or pixel_values_videos is not None) and mm_encoder_outputs is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos and mm_encoder_outputs at the same time"
            )

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must specify either `input_ids` or `inputs_embeds`.")
            inputs_embeds = qwen_model.get_input_embeddings()(input_ids)

        mm_outputs = dict(mm_encoder_outputs) if mm_encoder_outputs is not None else {}

        if mm_outputs.get("image") is None and pixel_values is not None:
            mm_outputs["image"] = qwen_model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True, **kwargs
            )

        if mm_outputs.get("video") is None and pixel_values_videos is not None:
            mm_outputs["video"] = qwen_model.get_video_features(
                pixel_values_videos, video_grid_thw, return_dict=True, **kwargs
            )

        if mm_outputs.get("image") is not None:
            image_embeds = torch.cat(mm_outputs["image"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = qwen_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if mm_outputs.get("video") is not None:
            video_embeds = torch.cat(mm_outputs["video"].pooler_output, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = qwen_model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return inputs_embeds, mm_outputs

    def _prepare_decoder_context(
        self,
        base_inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor | None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
        position_ids: torch.LongTensor | None,
        reference_cache: DynamicCache | None,
        image_grid_thw: torch.LongTensor | None,
        video_grid_thw: torch.LongTensor | None,
        mm_token_type_ids: torch.IntTensor | None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor | None], torch.LongTensor | None]:
        """Compute 3D M-RoPE embeddings and dual hybrid masks once for all loop steps."""
        qwen_model = self._core_model.model
        text_model = qwen_model.language_model

        if position_ids is None:
            position_ids = qwen_model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=base_inputs_embeds,
                attention_mask=attention_mask if not isinstance(attention_mask, dict) else None,
                cache=reference_cache,
                mm_token_type_ids=mm_token_type_ids,
            )

        if position_ids is None:
            past_seen_tokens = reference_cache.get_seq_length() if reference_cache is not None else 0
            position_ids = (
                torch.arange(base_inputs_embeds.shape[1], device=base_inputs_embeds.device) + past_seen_tokens
            )
            position_ids = position_ids.view(1, 1, -1).expand(4, base_inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            mrope_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            mrope_position_ids = position_ids

        if isinstance(attention_mask, dict):
            causal_mask_mapping = attention_mask
        else:
            mask_kwargs = {
                "config": text_model.config,
                "inputs_embeds": base_inputs_embeds,
                "attention_mask": attention_mask,
                "cache": reference_cache,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        position_embeddings = text_model.rotary_emb(base_inputs_embeds, mrope_position_ids)
        return position_embeddings, causal_mask_mapping, text_position_ids

    def _compute_ce_loss(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        return self._core_model.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.text_config.vocab_size,
        )

    def _run_layer_range(
        self,
        hidden_states: torch.Tensor,
        decoder_layers: Any,
        layer_types: Sequence[str],
        start_idx: int,
        end_idx: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        causal_mask_mapping: dict[str, torch.Tensor | None],
        text_position_ids: torch.LongTensor | None,
        cache: DynamicCache | None,
        use_cache: bool,
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        for layer_idx in range(start_idx, end_idx):
            hidden_states = decoder_layers[layer_idx](
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[layer_types[layer_idx]],
                position_ids=text_position_ids,
                cache=cache,
                use_cache=use_cache,
                **kwargs,
            )
        return hidden_states

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        per_loop_caches: PerLoopHybridCache | list[DynamicCache] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool = False,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        mm_encoder_outputs: dict[str, BaseModelOutputWithPooling] | None = None,
        num_loops: int | None = None,
        aux_loop_loss_weight: float | None = None,
        **kwargs: Any,
    ) -> LoopedCausalLMOutput:
        """Forward pass supporting v1 full-stack looping and v2 ETD middle-stack + ARLoop."""
        loops = self.num_loops if num_loops is None else num_loops
        if loops < 1:
            raise ValueError(f"num_loops must be >= 1, got {loops}")
        aux_weight = self.aux_loop_loss_weight if aux_loop_loss_weight is None else aux_loop_loss_weight

        # 1. Run native vision encoder & merge visual embeddings -> base_inputs_embeds
        base_inputs_embeds, mm_outputs = self._build_multimodal_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_encoder_outputs=mm_encoder_outputs,
            **kwargs,
        )

        # 2. Prepare isolated per-loop hybrid caches (only when use_cache=True)
        cache_container: PerLoopHybridCache | None = None
        if use_cache:
            if per_loop_caches is None:
                cache_container = self.create_per_loop_caches(loops)
            elif isinstance(per_loop_caches, PerLoopHybridCache):
                if len(per_loop_caches) != loops:
                    raise ValueError(
                        f"per_loop_caches has {len(per_loop_caches)} caches, expected {loops}"
                    )
                cache_container = per_loop_caches
            else:
                if len(per_loop_caches) != loops:
                    raise ValueError(
                        f"per_loop_caches list has {len(per_loop_caches)} caches, expected {loops}"
                    )
                cache_container = PerLoopHybridCache(self.text_config, loops)
                cache_container.caches = list(per_loop_caches)

            # Register sink + vision protected mask on prefill when TurboQuant protection is active
            if cache_container.get_seq_length() == 0 and cache_container.protect_vision_prefix:
                prot_mask = self._compute_prefill_protected_mask(
                    input_ids=input_ids,
                    seq_len=int(base_inputs_embeds.shape[1]),
                    sink_tokens=cache_container.sink_tokens,
                    device=base_inputs_embeds.device,
                )
                cache_container.set_prefill_protected_mask(prot_mask)

        reference_cache = cache_container[0] if cache_container is not None else None

        # 3. Compute 3D M-RoPE and dual attention masks once across all loops
        position_embeddings, causal_mask_mapping, text_position_ids = self._prepare_decoder_context(
            base_inputs_embeds=base_inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            reference_cache=reference_cache,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

        text_model = self._core_model.model.language_model
        total_layers = int(text_model.config.num_hidden_layers)
        decoder_layers = text_model.layers[:total_layers]
        layer_types = text_model.config.layer_types

        # 4a. ETD Encoder stage (layers 0 .. loop_start_layer - 1), run ONCE using shared caches[0]
        hidden_states = base_inputs_embeds
        if self.loop_start_layer > 0:
            hidden_states = self._run_layer_range(
                hidden_states,
                decoder_layers,
                layer_types,
                0,
                self.loop_start_layer,
                position_embeddings,
                causal_mask_mapping,
                text_position_ids,
                cache=reference_cache,
                use_cache=use_cache,
                kwargs=kwargs,
            )

        encoder_anchor = hidden_states  # E_base (v1) or E_enc (v2 ETD)

        # 4b. Looped Reasoning Core (layers loop_start_layer .. loop_end_layer - 1)
        w_scalar = torch.sigmoid(self.reinject_logit).to(
            dtype=encoder_anchor.dtype, device=encoder_anchor.device
        )
        observed_anchor_w: float | None = None if use_cache else float(w_scalar.detach().float().item())
        per_loop_losses: list[torch.Tensor] = []
        final_logits: torch.Tensor | None = None
        prev_loop_hidden: torch.Tensor | None = None

        for loop_idx in range(loops):
            if loop_idx >= 1:
                if self.reinject_mode == "arloop" and self.arloop_gate is not None:
                    # Sliding window: [E_enc, h^(t-1), h^(t)]
                    window_states = [encoder_anchor]
                    if prev_loop_hidden is not None:
                        window_states.append(prev_loop_hidden)
                    window_states.append(hidden_states)
                    hidden_states, mean_w = self.arloop_gate(loop_idx, window_states)
                    if not use_cache:
                        observed_anchor_w = float(mean_w.item())
                else:
                    hidden_states = w_scalar * encoder_anchor + (1.0 - w_scalar) * hidden_states

            loop_cache = cache_container[loop_idx] if cache_container is not None else None
            core_out = self._run_layer_range(
                hidden_states,
                decoder_layers,
                layer_types,
                self.loop_start_layer,
                self.loop_end_layer,
                position_embeddings,
                causal_mask_mapping,
                text_position_ids,
                cache=loop_cache,
                use_cache=use_cache,
                kwargs=kwargs,
            )
            prev_loop_hidden = hidden_states if loop_idx >= 1 else None
            hidden_states = core_out

            is_final_loop = loop_idx == loops - 1
            if is_final_loop:
                # 4c. ETD Output Decoder stage (layers loop_end_layer .. total_layers - 1), run ONCE
                if self.loop_end_layer < total_layers:
                    hidden_states = self._run_layer_range(
                        hidden_states,
                        decoder_layers,
                        layer_types,
                        self.loop_end_layer,
                        total_layers,
                        position_embeddings,
                        causal_mask_mapping,
                        text_position_ids,
                        cache=reference_cache,
                        use_cache=use_cache,
                        kwargs=kwargs,
                    )
                normed_hidden = text_model.norm(hidden_states)
                sliced_hidden = slice_logits_to_keep(normed_hidden, logits_to_keep)
                final_logits = self._core_model.lm_head(sliced_hidden)
                if labels is not None:
                    per_loop_losses.append(self._compute_ce_loss(final_logits, labels))
            elif labels is not None and aux_weight > 0.0:
                # For auxiliary middle-loop supervision in ETD mode, project through decoder layers without mutating cache
                mid_hidden = hidden_states
                if self.loop_end_layer < total_layers:
                    mid_hidden = self._run_layer_range(
                        mid_hidden,
                        decoder_layers,
                        layer_types,
                        self.loop_end_layer,
                        total_layers,
                        position_embeddings,
                        causal_mask_mapping,
                        text_position_ids,
                        cache=None,
                        use_cache=False,
                        kwargs=kwargs,
                    )
                mid_normed = text_model.norm(mid_hidden)
                mid_logits = self._core_model.lm_head(mid_normed)
                per_loop_losses.append(self._compute_ce_loss(mid_logits, labels))

        self._last_reinject_weight = observed_anchor_w

        # 5. Combine final loop loss + optional auxiliary middle-loop loss
        total_loss: torch.Tensor | None = None
        final_loop_loss: torch.Tensor | None = None
        aux_loop_loss: torch.Tensor | None = None

        if labels is not None and per_loop_losses:
            final_loop_loss = per_loop_losses[-1]
            if len(per_loop_losses) > 1 and aux_weight > 0.0:
                aux_loop_loss = torch.stack(per_loop_losses[:-1]).mean()
                total_loss = (1.0 - aux_weight) * final_loop_loss + aux_weight * aux_loop_loss
            else:
                total_loss = final_loop_loss

        return LoopedCausalLMOutput(
            loss=total_loss,
            logits=final_logits,
            per_loop_caches=cache_container,
            per_loop_losses=per_loop_losses,
            final_loop_loss=final_loop_loss,
            aux_loop_loss=aux_loop_loss,
            reinject_weight=observed_anchor_w,
            image_outputs=mm_outputs.get("image"),
            video_outputs=mm_outputs.get("video"),
        )

    @torch.no_grad()
    def compare_loop_losses(
        self,
        loop_counts: Sequence[int] = (1, 2, 3),
        **batch_inputs: Any,
    ) -> dict[int, float]:
        """Evaluate pure final-loop CE loss across multiple loop counts on the same batch."""
        if "labels" not in batch_inputs or batch_inputs["labels"] is None:
            raise ValueError("`compare_loop_losses` requires `labels` in batch_inputs.")

        was_training = self.training
        saved_rope_deltas = self._core_model.model.rope_deltas
        self.eval()
        results: dict[int, float] = {}
        try:
            for k in loop_counts:
                self._core_model.model.rope_deltas = None
                out = self.forward(
                    **batch_inputs,
                    use_cache=False,
                    num_loops=k,
                    aux_loop_loss_weight=0.0,
                )
                assert out.final_loop_loss is not None
                results[int(k)] = float(out.final_loop_loss.detach().float().item())
        finally:
            self._core_model.model.rope_deltas = saved_rope_deltas
            if was_training:
                self.train()
        return results

    def _build_decode_position_ids(self, attention_mask: torch.Tensor) -> torch.LongTensor:
        """Build `[4, batch_size, 1]` position IDs (text + 3D vision mRoPE) for cached decode steps."""
        batch_size = attention_mask.shape[0]
        device = attention_mask.device

        text_pos = (attention_mask.long().cumsum(-1)[:, -1:] - 1).clamp_min(0)
        rope_deltas = self._core_model.model.rope_deltas
        if rope_deltas is None:
            rope_deltas = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        else:
            rope_deltas = rope_deltas.to(device=device, dtype=torch.long)
            if rope_deltas.shape[0] != batch_size:
                rope_deltas = rope_deltas.repeat_interleave(batch_size // rope_deltas.shape[0], dim=0)

        vision_pos = text_pos.unsqueeze(0).expand(3, -1, -1) + rope_deltas.unsqueeze(0)
        return torch.cat([text_pos.unsqueeze(0), vision_pos], dim=0)

    def _resolve_eos_token_ids(self, eos_token_id: int | Sequence[int] | None) -> set[int]:
        if eos_token_id is None:
            eos_token_id = getattr(self._core_model.generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.text_config, "eos_token_id", None)
        if eos_token_id is None:
            return set()
        if isinstance(eos_token_id, int):
            return {eos_token_id}
        return {int(tok) for tok in eos_token_id}

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.LongTensor:
        if not do_sample or temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        scaled = logits.float() / temperature
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative_probs > top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove_mask = sorted_remove.scatter(dim=-1, index=sorted_indices, src=sorted_remove)
            scaled = scaled.masked_fill(remove_mask, float("-inf"))

        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def generate_looped(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        max_new_tokens: int = 32,
        num_loops: int | None = None,
        use_cache: bool = True,
        kv_quant: str | None = None,
        residual_window: int | None = None,
        protect_vision_prefix: bool | None = None,
        sink_tokens: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | Sequence[int] | None = None,
        use_compile: bool = False,
        **kwargs: Any,
    ) -> torch.LongTensor:
        """Autoregressive decoding with isolated per-loop hybrid caches for text and multimodal prompts."""
        if max_new_tokens <= 0:
            return input_ids

        loops = self.num_loops if num_loops is None else num_loops
        batch_size = input_ids.shape[0]
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        eos_ids = self._resolve_eos_token_ids(eos_token_id)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated_ids = input_ids
        cur_attention_mask = attention_mask

        self._core_model.model.rope_deltas = None
        per_loop_caches = (
            self.create_per_loop_caches(
                loops,
                kv_quant=kv_quant,
                residual_window=residual_window,
                protect_vision_prefix=protect_vision_prefix,
                sink_tokens=sink_tokens,
            )
            if use_cache
            else None
        )

        # Optional torch.compile wrapper for single-token cached decode steps
        decode_fn = None
        if use_compile and use_cache and not getattr(self, "_compile_fallback_triggered", False):
            if not hasattr(self, "_compiled_decode_step") or self._compiled_decode_step is None:
                import os
                import shutil

                def _eager_decode_step(
                    tok: torch.LongTensor,
                    attn_mask: torch.Tensor,
                    pos_ids: torch.LongTensor,
                    caches: PerLoopHybridCache,
                    n_loops: int,
                ) -> LoopedCausalLMOutput:
                    return self.forward(
                        input_ids=tok,
                        attention_mask=attn_mask,
                        position_ids=pos_ids,
                        per_loop_caches=caches,
                        use_cache=True,
                        logits_to_keep=1,
                        num_loops=n_loops,
                    )

                try:
                    if os.name == "nt" and shutil.which("cl") is None:
                        self._compiled_decode_step = torch.compile(
                            _eager_decode_step,
                            backend="cudagraphs",
                            dynamic=True,
                        )
                        self._compile_backend_used = "cudagraphs"
                    else:
                        self._compiled_decode_step = torch.compile(
                            _eager_decode_step,
                            mode="reduce-overhead",
                            dynamic=True,
                        )
                        self._compile_backend_used = "inductor_reduce_overhead"
                    self._compile_fallback_triggered = False
                except Exception as exc:
                    print(f"[generate_looped] torch.compile unavailable ({exc}); falling back to eager decode.")
                    self._compiled_decode_step = None
                    self._compile_backend_used = "eager_fallback"
                    self._compile_fallback_triggered = True
            decode_fn = self._compiled_decode_step

        # Step 0: Prefill
        out = self.forward(
            input_ids=generated_ids,
            attention_mask=cur_attention_mask,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            per_loop_caches=per_loop_caches,
            use_cache=use_cache,
            logits_to_keep=1,
            num_loops=loops,
            **kwargs,
        )
        per_loop_caches = out.per_loop_caches
        assert out.logits is not None
        next_token = self._sample_next_token(
            out.logits[:, -1, :], do_sample=do_sample, temperature=temperature, top_p=top_p
        )
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        cur_attention_mask = torch.cat(
            [cur_attention_mask, torch.ones((batch_size, 1), dtype=cur_attention_mask.dtype, device=device)],
            dim=-1,
        )

        if eos_ids:
            for eid in eos_ids:
                finished |= next_token.squeeze(-1) == eid
            if bool(finished.all()):
                self._last_per_loop_caches = per_loop_caches
                return generated_ids

        # Steps 1..max_new_tokens-1: Incremental cached decode
        for _ in range(max_new_tokens - 1):
            if use_cache:
                step_position_ids = self._build_decode_position_ids(cur_attention_mask)
                if decode_fn is not None and not kwargs:
                    try:
                        out = decode_fn(
                            next_token,
                            cur_attention_mask,
                            step_position_ids,
                            per_loop_caches,
                            loops,
                        )
                    except Exception as exc:
                        if not getattr(self, "_compile_fallback_triggered", False):
                            print(
                                f"[generate_looped] torch.compile decode graph failed ({type(exc).__name__}: {exc}); "
                                "falling back to eager decode."
                            )
                            self._compile_fallback_triggered = True
                        decode_fn = None
                        self._compiled_decode_step = None
                        out = self.forward(
                            input_ids=next_token,
                            attention_mask=cur_attention_mask,
                            position_ids=step_position_ids,
                            per_loop_caches=per_loop_caches,
                            use_cache=True,
                            logits_to_keep=1,
                            num_loops=loops,
                            **kwargs,
                        )
                else:
                    out = self.forward(
                        input_ids=next_token,
                        attention_mask=cur_attention_mask,
                        position_ids=step_position_ids,
                        per_loop_caches=per_loop_caches,
                        use_cache=True,
                        logits_to_keep=1,
                        num_loops=loops,
                        **kwargs,
                    )
                per_loop_caches = out.per_loop_caches
            else:
                self._core_model.model.rope_deltas = None
                out = self.forward(
                    input_ids=generated_ids,
                    attention_mask=cur_attention_mask,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                    use_cache=False,
                    logits_to_keep=1,
                    num_loops=loops,
                    **kwargs,
                )

            assert out.logits is not None
            next_token = self._sample_next_token(
                out.logits[:, -1, :], do_sample=do_sample, temperature=temperature, top_p=top_p
            )
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            cur_attention_mask = torch.cat(
                [cur_attention_mask, torch.ones((batch_size, 1), dtype=cur_attention_mask.dtype, device=device)],
                dim=-1,
            )

            if eos_ids:
                for eid in eos_ids:
                    finished |= next_token.squeeze(-1) == eid
                if bool(finished.all()):
                    break

        self._last_per_loop_caches = per_loop_caches
        return generated_ids
