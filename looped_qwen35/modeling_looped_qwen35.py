"""Non-destructive Looped Hybrid Transformer wrapper for Qwen3.5 (Gated DeltaNet + Gated Attention).

Built directly against `transformers.models.qwen3_5.modeling_qwen3_5` (transformers 5.24.0):
- Uses `DynamicCache(config=text_config)` per loop to isolate both Gated DeltaNet
  conv/recurrent states and Gated Attention KV states across loop iterations.
- Preserves native vision encoding (`self.base_model.model.visual`), placeholder scattering,
  3D M-RoPE (`compute_3d_position_ids` / `Qwen3_5TextRotaryEmbedding`), and dual masks
  (`create_causal_mask` + `create_recurrent_attention_mask`).
"""

from __future__ import annotations

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


class PerLoopHybridCache:
    """Holds one independent hybrid cache (`DynamicCache` or `TurboQuantHybridCache`) per loop iteration.

    Qwen3.5 mixes 18 Gated DeltaNet recurrent layers with 6 Gated Attention layers.
    Sharing a single cache across loops would feed the same token into each DeltaNet
    conv/recurrent state `num_loops` times per decode step. Keeping one isolated cache
    per loop guarantees loop `t` only reads and updates `caches[t]`.
    """

    def __init__(
        self,
        text_config: Any,
        num_loops: int,
        kv_quant: str = "none",
        residual_window: int = 64,
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
                    seed=self.seed,
                )
                for _ in range(self.num_loops)
            ]
        return [DynamicCache(config=self.text_config) for _ in range(self.num_loops)]

    def __getitem__(self, loop_idx: int) -> DynamicCache:
        return self.caches[loop_idx]

    def __len__(self) -> int:
        return len(self.caches)

    def get_seq_length(self) -> int:
        """Return cached sequence length (all loops advance in lockstep)."""
        return self.caches[0].get_seq_length() if self.caches else 0

    def reset(self) -> None:
        self.caches = self._build_caches()

    def kv_cache_memory_bytes(self) -> dict[str, int]:
        """Return exact byte totals across all loops split by 6 Attention KV layers vs 18 DeltaNet layers."""
        attn_total = 0
        delta_total = 0
        for cache in self.caches:
            stats = measure_hybrid_cache_bytes(cache)
            attn_total += stats["attention_kv_bytes"]
            delta_total += stats["deltanet_state_bytes"]
        return {
            "attention_kv_bytes": attn_total,
            "deltanet_state_bytes": delta_total,
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

        # Keep scalar re-injection logit in fp32 for clean optimizer updates; cast during blending.
        # sigmoid(4.6) ~= 0.9900 -> starts very close to pure input grounding before fine-tuning.
        param_device = next(base_model.parameters()).device
        self.reinject_logit = nn.Parameter(
            torch.tensor(self.loop_config.reinject_init_logit, dtype=torch.float32, device=param_device)
        )

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
        return torch.sigmoid(self.reinject_logit)

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
    ) -> PerLoopHybridCache:
        """Create isolated hybrid caches (Gated DeltaNet + Gated Attention) for each loop."""
        loops = self.num_loops if num_loops is None else num_loops
        quant_mode = self.loop_config.kv_quant if kv_quant is None else kv_quant
        window = self.loop_config.kv_residual_window if residual_window is None else residual_window
        return PerLoopHybridCache(
            self.text_config,
            loops,
            kv_quant=quant_mode,
            residual_window=window,
        )

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

        # Mirror Qwen3_5TextModel.forward lines 1241-1269 exactly
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
        """Forward pass looping hidden states through all 24 Qwen3.5 decoder layers `num_loops` times."""
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

        # 4. Run `loops` passes through the 24 hybrid decoder layers
        text_model = self._core_model.model.language_model
        decoder_layers = text_model.layers[: text_model.config.num_hidden_layers]
        layer_types = text_model.config.layer_types

        hidden_states = base_inputs_embeds
        w_scalar = torch.sigmoid(self.reinject_logit).to(
            dtype=base_inputs_embeds.dtype, device=base_inputs_embeds.device
        )

        per_loop_losses: list[torch.Tensor] = []
        final_logits: torch.Tensor | None = None

        for loop_idx in range(loops):
            if loop_idx >= 1:
                hidden_states = w_scalar * base_inputs_embeds + (1.0 - w_scalar) * hidden_states

            loop_cache = cache_container[loop_idx] if cache_container is not None else None

            for layer_idx, decoder_layer in enumerate(decoder_layers):
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask_mapping[layer_types[layer_idx]],
                    position_ids=text_position_ids,
                    cache=loop_cache,
                    use_cache=use_cache,
                    **kwargs,
                )

            is_final_loop = loop_idx == loops - 1
            if is_final_loop:
                normed_hidden = text_model.norm(hidden_states)
                sliced_hidden = slice_logits_to_keep(normed_hidden, logits_to_keep)
                final_logits = self._core_model.lm_head(sliced_hidden)
                if labels is not None:
                    per_loop_losses.append(self._compute_ce_loss(final_logits, labels))
            elif labels is not None and aux_weight > 0.0:
                mid_normed = text_model.norm(hidden_states)
                mid_logits = self._core_model.lm_head(mid_normed)
                per_loop_losses.append(self._compute_ce_loss(mid_logits, labels))

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
            reinject_weight=float(w_scalar.detach().float().item()),
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

        text_pos = (attention_mask.long().cumsum(-1)[:, -1:] - 1).clamp_min(0)  # (bs, 1)
        rope_deltas = self._core_model.model.rope_deltas
        if rope_deltas is None:
            rope_deltas = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        else:
            rope_deltas = rope_deltas.to(device=device, dtype=torch.long)
            if rope_deltas.shape[0] != batch_size:
                rope_deltas = rope_deltas.repeat_interleave(batch_size // rope_deltas.shape[0], dim=0)

        vision_pos = text_pos.unsqueeze(0).expand(3, -1, -1) + rope_deltas.unsqueeze(0)  # (3, bs, 1)
        return torch.cat([text_pos.unsqueeze(0), vision_pos], dim=0)  # (4, bs, 1)

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
        """Select the next token ID from `(batch_size, vocab_size)` logits."""
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
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id: int | Sequence[int] | None = None,
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

        # Reset stale mRoPE deltas from any prior call
        self._core_model.model.rope_deltas = None
        per_loop_caches = (
            self.create_per_loop_caches(loops, kv_quant=kv_quant, residual_window=residual_window)
            if use_cache
            else None
        )

        # Step 0: Prefill (processes full prompt + vision encoder once)
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

        # Steps 1..max_new_tokens-1: Incremental cached decode (or uncached fallback)
        for _ in range(max_new_tokens - 1):
            if use_cache:
                step_position_ids = self._build_decode_position_ids(cur_attention_mask)
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
