"""TurboQuant (~3-bit / 4-bit) data-oblivious KV-cache compression for Hybrid Qwen3.5.

Compresses strictly the 6 Gated Attention (`full_attention`) layers while leaving all
18 Gated DeltaNet (`linear_attention`) conv/recurrent states untouched in full precision.

Algorithm (Zandieh et al., TurboQuant):
- Stage 0: L2-normalize each KV vector (store fp16 norm) and apply a fixed Haar-random
  orthogonal rotation `R` so coordinates concentrate around 0 with std ~= 1/sqrt(head_dim).
- Stage 1: Quantize rotated coordinates using fixed symmetric Gaussian Lloyd-Max centroids
  (2-bit for 3-bit Keys, 3-bit for 3-bit Values; 3-bit / 4-bit for 4-bit mode) and
  bit-pack indices directly into `torch.uint8`.
- Stage 2 (Keys only): Project the Stage-1 residual `e = x_rot - x_hat_rot` through a
  fixed Gaussian matrix `P`, store `||e||` in fp16 and 1-bit signs bit-packed into
  `torch.uint8` (8 signs/byte) for unbiased inner-product reconstruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from transformers.cache_utils import CacheLayerMixin, DynamicCache, LinearAttentionLayer

# Standard normal N(0, 1) optimal symmetric Lloyd-Max centroids (scaled by 1/sqrt(head_dim))
LLOYD_MAX_CENTROIDS: dict[int, tuple[float, ...]] = {
    2: (-1.5104176, -0.4527800, 0.4527800, 1.5104176),
    3: (
        -2.1519457,
        -1.3439093,
        -0.7560052,
        -0.2450942,
        0.2450942,
        0.7560052,
        1.3439093,
        2.1519457,
    ),
    4: (
        -2.7326,
        -2.0690,
        -1.6180,
        -1.2562,
        -0.9423,
        -0.6568,
        -0.3881,
        -0.1284,
        0.1284,
        0.3881,
        0.6568,
        0.9423,
        1.2562,
        1.6180,
        2.0690,
        2.7326,
    ),
}


def pack_bits(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-pack integer indices in `[0, 2^bits - 1]` along the last dimension into `torch.uint8`."""
    d = indices.shape[-1]
    u8 = indices.to(torch.uint8)

    if bits == 1:
        if d % 8 != 0:
            raise ValueError(f"head_dim ({d}) must be divisible by 8 for 1-bit packing")
        grouped = u8.reshape(*u8.shape[:-1], d // 8, 8)
        shifts = torch.arange(8, device=u8.device, dtype=torch.uint8)
        return (grouped << shifts).sum(dim=-1, dtype=torch.uint8)

    if bits == 2:
        if d % 4 != 0:
            raise ValueError(f"head_dim ({d}) must be divisible by 4 for 2-bit packing")
        grouped = u8.reshape(*u8.shape[:-1], d // 4, 4)
        return (
            grouped[..., 0]
            | (grouped[..., 1] << 2)
            | (grouped[..., 2] << 4)
            | (grouped[..., 3] << 6)
        )

    if bits == 3:
        if d % 8 != 0:
            raise ValueError(f"head_dim ({d}) must be divisible by 8 for 3-bit packing")
        grouped = indices.to(torch.int32).reshape(*indices.shape[:-1], d // 8, 8)
        shifts = torch.arange(0, 24, 3, device=indices.device, dtype=torch.int32)
        packed24 = (grouped << shifts).sum(dim=-1, dtype=torch.int32)
        b0 = (packed24 & 0xFF).to(torch.uint8)
        b1 = ((packed24 >> 8) & 0xFF).to(torch.uint8)
        b2 = ((packed24 >> 16) & 0xFF).to(torch.uint8)
        return torch.stack((b0, b1, b2), dim=-1).reshape(*indices.shape[:-1], (d // 8) * 3)

    if bits == 4:
        if d % 2 != 0:
            raise ValueError(f"head_dim ({d}) must be divisible by 2 for 4-bit packing")
        grouped = u8.reshape(*u8.shape[:-1], d // 2, 2)
        return grouped[..., 0] | (grouped[..., 1] << 4)

    raise ValueError(f"Unsupported bit-width for packing: {bits}")


def unpack_bits(packed: torch.Tensor, bits: int, orig_dim: int) -> torch.Tensor:
    """Unpack `torch.uint8` bit-packed tensor back to `torch.long` indices of shape `(..., orig_dim)`."""
    prefix_shape = packed.shape[:-1]

    if bits == 1:
        shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
        unpacked = (packed.unsqueeze(-1) >> shifts) & 1
        return unpacked.reshape(*prefix_shape, orig_dim).to(torch.long)

    if bits == 2:
        shifts = torch.tensor([0, 2, 4, 6], device=packed.device, dtype=torch.uint8)
        unpacked = (packed.unsqueeze(-1) >> shifts) & 0x3
        return unpacked.reshape(*prefix_shape, orig_dim).to(torch.long)

    if bits == 3:
        triples = packed.reshape(*prefix_shape, orig_dim // 8, 3).to(torch.int32)
        packed24 = triples[..., 0] | (triples[..., 1] << 8) | (triples[..., 2] << 16)
        shifts = torch.arange(0, 24, 3, device=packed.device, dtype=torch.int32)
        unpacked = (packed24.unsqueeze(-1) >> shifts) & 0x7
        return unpacked.reshape(*prefix_shape, orig_dim).to(torch.long)

    if bits == 4:
        lo = packed & 0xF
        hi = (packed >> 4) & 0xF
        return torch.stack((lo, hi), dim=-1).reshape(*prefix_shape, orig_dim).to(torch.long)

    raise ValueError(f"Unsupported bit-width for unpacking: {bits}")


@dataclass
class CompressedKeys:
    """Bit-packed TurboQuant representation for Attention Keys."""

    packed_indices: torch.Tensor  # uint8: (B, H, S, packed_k_bytes)
    packed_qjl_signs: torch.Tensor  # uint8: (B, H, S, head_dim // 8)
    norms: torch.Tensor  # float16: (B, H, S, 1)
    residual_norms: torch.Tensor  # float16: (B, H, S, 1)
    head_dim: int
    dtype: torch.dtype

    @property
    def seq_len(self) -> int:
        return int(self.packed_indices.shape[-2])

    def memory_bytes(self) -> int:
        return (
            self.packed_indices.numel() * self.packed_indices.element_size()
            + self.packed_qjl_signs.numel() * self.packed_qjl_signs.element_size()
            + self.norms.numel() * self.norms.element_size()
            + self.residual_norms.numel() * self.residual_norms.element_size()
        )

    @classmethod
    def concat(cls, a: CompressedKeys, b: CompressedKeys) -> CompressedKeys:
        return cls(
            packed_indices=torch.cat([a.packed_indices, b.packed_indices], dim=-2),
            packed_qjl_signs=torch.cat([a.packed_qjl_signs, b.packed_qjl_signs], dim=-2),
            norms=torch.cat([a.norms, b.norms], dim=-2),
            residual_norms=torch.cat([a.residual_norms, b.residual_norms], dim=-2),
            head_dim=a.head_dim,
            dtype=a.dtype,
        )


@dataclass
class CompressedValues:
    """Bit-packed TurboQuant representation for Attention Values."""

    packed_indices: torch.Tensor  # uint8: (B, H, S, packed_v_bytes)
    norms: torch.Tensor  # float16: (B, H, S, 1)
    head_dim: int
    dtype: torch.dtype

    @property
    def seq_len(self) -> int:
        return int(self.packed_indices.shape[-2])

    def memory_bytes(self) -> int:
        return (
            self.packed_indices.numel() * self.packed_indices.element_size()
            + self.norms.numel() * self.norms.element_size()
        )

    @classmethod
    def concat(cls, a: CompressedValues, b: CompressedValues) -> CompressedValues:
        return cls(
            packed_indices=torch.cat([a.packed_indices, b.packed_indices], dim=-2),
            norms=torch.cat([a.norms, b.norms], dim=-2),
            head_dim=a.head_dim,
            dtype=a.dtype,
        )


class TurboQuantCompressor:
    """Data-oblivious TurboQuant KV-cache compressor supporting `bits=3` and `bits=4`."""

    def __init__(
        self,
        head_dim: int = 256,
        bits: int = 3,
        seed: int = 42,
        device: str | torch.device | None = None,
    ) -> None:
        if bits not in (3, 4):
            raise ValueError(f"TurboQuantCompressor supports bits in (3, 4), got {bits}")
        if head_dim % 8 != 0:
            raise ValueError(f"head_dim must be divisible by 8, got {head_dim}")

        self.head_dim = head_dim
        self.bits = bits
        self.key_stage1_bits = bits - 1  # 2-bit for bits=3; 3-bit for bits=4 (+ 1-bit QJL)
        self.val_bits = bits  # 3-bit for bits=3; 4-bit for bits=4
        self.seed = seed
        self._qjl_scale = math.sqrt(math.pi / 2.0) / float(head_dim)

        # Build deterministic Haar orthogonal matrix R and Gaussian projection P on CPU
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

        g_mat = torch.randn((head_dim, head_dim), generator=gen, dtype=torch.float32)
        q_mat, r_mat = torch.linalg.qr(g_mat)
        diag_sign = torch.sign(torch.diag(r_mat))
        diag_sign[diag_sign == 0] = 1.0
        self._R_cpu = (q_mat * diag_sign.unsqueeze(0)).contiguous()
        self._P_cpu = torch.randn((head_dim, head_dim), generator=gen, dtype=torch.float32).contiguous()

        scale = 1.0 / math.sqrt(float(head_dim))
        self._key_centroids_cpu = torch.tensor(
            LLOYD_MAX_CENTROIDS[self.key_stage1_bits], dtype=torch.float32
        ) * scale
        self._key_boundaries_cpu = 0.5 * (self._key_centroids_cpu[:-1] + self._key_centroids_cpu[1:])

        self._val_centroids_cpu = torch.tensor(
            LLOYD_MAX_CENTROIDS[self.val_bits], dtype=torch.float32
        ) * scale
        self._val_boundaries_cpu = 0.5 * (self._val_centroids_cpu[:-1] + self._val_centroids_cpu[1:])

        # Per-device tensor cache (fp32 for accurate rotation/quantization)
        self._device_cache: dict[torch.device, dict[str, torch.Tensor]] = {}
        if device is not None:
            self._get_device_tensors(torch.device(device))

    def _get_device_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        if device not in self._device_cache:
            self._device_cache[device] = {
                "R": self._R_cpu.to(device=device),
                "R_T": self._R_cpu.T.contiguous().to(device=device),
                "P": self._P_cpu.to(device=device),
                "P_T": self._P_cpu.T.contiguous().to(device=device),
                "key_centroids": self._key_centroids_cpu.to(device=device),
                "key_boundaries": self._key_boundaries_cpu.to(device=device),
                "val_centroids": self._val_centroids_cpu.to(device=device),
                "val_boundaries": self._val_boundaries_cpu.to(device=device),
            }
        return self._device_cache[device]

    def compress_keys(self, keys: torch.Tensor) -> CompressedKeys:
        """Compress `(B, H, S, head_dim)` Keys via rotation + Lloyd-Max + 1-bit QJL residual."""
        tensors = self._get_device_tensors(keys.device)
        x = keys.float()
        norms = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x_rot = torch.matmul(x / norms, tensors["R"])

        # Stage 1: Fixed Lloyd-Max quantization (bit-packed uint8)
        indices = torch.bucketize(x_rot, tensors["key_boundaries"])
        x_hat_rot = tensors["key_centroids"][indices]
        packed_indices = pack_bits(indices, self.key_stage1_bits)

        # Stage 2: 1-bit QJL sign projection of the quantization residual
        residual = x_rot - x_hat_rot
        res_norms = residual.norm(dim=-1, keepdim=True)
        qjl_bits = (torch.matmul(residual, tensors["P"]) >= 0).to(torch.uint8)
        packed_qjl_signs = pack_bits(qjl_bits, bits=1)

        return CompressedKeys(
            packed_indices=packed_indices,
            packed_qjl_signs=packed_qjl_signs,
            norms=norms.to(torch.float16),
            residual_norms=res_norms.to(torch.float16),
            head_dim=self.head_dim,
            dtype=keys.dtype,
        )

    def dequantize_keys(self, compressed: CompressedKeys) -> torch.Tensor:
        """Reconstruct Keys from bit-packed Stage-1 Lloyd-Max + Stage-2 QJL residual."""
        tensors = self._get_device_tensors(compressed.packed_indices.device)

        indices = unpack_bits(compressed.packed_indices, self.key_stage1_bits, self.head_dim)
        x_hat_rot = tensors["key_centroids"][indices]

        qjl_bits = unpack_bits(compressed.packed_qjl_signs, bits=1, orig_dim=self.head_dim)
        signs = qjl_bits.to(torch.float32) * 2.0 - 1.0  # {-1.0, +1.0}
        e_hat = (self._qjl_scale * compressed.residual_norms.float()) * torch.matmul(signs, tensors["P_T"])

        x_rot_rec = x_hat_rot + e_hat
        x_rec = torch.matmul(x_rot_rec, tensors["R_T"]) * compressed.norms.float()
        return x_rec.to(compressed.dtype)

    def compress_values(self, values: torch.Tensor) -> CompressedValues:
        """Compress `(B, H, S, head_dim)` Values via rotation + bit-packed Lloyd-Max."""
        tensors = self._get_device_tensors(values.device)
        x = values.float()
        norms = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x_rot = torch.matmul(x / norms, tensors["R"])

        indices = torch.bucketize(x_rot, tensors["val_boundaries"])
        packed_indices = pack_bits(indices, self.val_bits)

        return CompressedValues(
            packed_indices=packed_indices,
            norms=norms.to(torch.float16),
            head_dim=self.head_dim,
            dtype=values.dtype,
        )

    def dequantize_values(self, compressed: CompressedValues) -> torch.Tensor:
        """Reconstruct Values from bit-packed rotated Lloyd-Max indices."""
        tensors = self._get_device_tensors(compressed.packed_indices.device)
        indices = unpack_bits(compressed.packed_indices, self.val_bits, self.head_dim)
        x_hat_rot = tensors["val_centroids"][indices]
        x_rec = torch.matmul(x_hat_rot, tensors["R_T"]) * compressed.norms.float()
        return x_rec.to(compressed.dtype)

    @staticmethod
    def memory_bytes(compressed_keys: CompressedKeys | None, compressed_values: CompressedValues | None) -> int:
        total = 0
        if compressed_keys is not None:
            total += compressed_keys.memory_bytes()
        if compressed_values is not None:
            total += compressed_values.memory_bytes()
        return total


class TurboQuantLayer(CacheLayerMixin):
    """Drop-in replacement for `DynamicLayer` storing older tokens in bit-packed TurboQuant format.

    When `protect_vision_prefix=True`, sink tokens (`0..sink_tokens-1`) and all visual patch
    tokens (flagged via `set_prefill_protected_mask`) are kept permanently in native `bf16`
    while non-protected text / `<think>` / decode tokens past `residual_window` are compressed.
    """

    is_sliding = False
    is_croppable = False

    def __init__(
        self,
        compressor: TurboQuantCompressor,
        residual_window: int = 64,
        protect_vision_prefix: bool = True,
    ) -> None:
        super().__init__()
        self.compressor = compressor
        self.residual_window = max(0, int(residual_window))
        self.protect_vision_prefix = bool(protect_vision_prefix)

        self.compressed_keys: CompressedKeys | None = None
        self.compressed_values: CompressedValues | None = None
        self.recent_keys: torch.Tensor | None = None
        self.recent_values: torch.Tensor | None = None

        # Protected bf16 sink + vision prefix state (set on prefill)
        self._prefill_protected_mask: torch.Tensor | None = None
        self.protected_keys: torch.Tensor | None = None
        self.protected_values: torch.Tensor | None = None
        self.protected_positions: torch.Tensor | None = None
        self.prefill_unprotected_positions: torch.Tensor | None = None
        self.prefill_len: int = 0

        self.is_initialized = False

    def set_prefill_protected_mask(self, mask: torch.Tensor | None) -> None:
        """Register a 1D boolean mask `(prefill_seq_len,)` of positions to protect in bf16."""
        self._prefill_protected_mask = mask

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.device = key_states.device
        self.is_initialized = True

    def _append_unprotected(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if k_new.shape[-2] == 0:
            return
        if self.recent_keys is None:
            self.recent_keys = k_new.contiguous()
            self.recent_values = v_new.contiguous()
        else:
            self.recent_keys = torch.cat([self.recent_keys, k_new], dim=-2)
            self.recent_values = torch.cat([self.recent_values, v_new], dim=-2)

        if self.recent_keys.shape[-2] > self.residual_window:
            if self.residual_window > 0:
                k_flush = self.recent_keys[:, :, : -self.residual_window, :]
                v_flush = self.recent_values[:, :, : -self.residual_window, :]
                self.recent_keys = self.recent_keys[:, :, -self.residual_window :, :].contiguous()
                self.recent_values = self.recent_values[:, :, -self.residual_window :, :].contiguous()
            else:
                k_flush = self.recent_keys
                v_flush = self.recent_values
                self.recent_keys = None
                self.recent_values = None

            new_ck = self.compressor.compress_keys(k_flush)
            new_cv = self.compressor.compress_values(v_flush)
            self.compressed_keys = (
                new_ck if self.compressed_keys is None else CompressedKeys.concat(self.compressed_keys, new_ck)
            )
            self.compressed_values = (
                new_cv if self.compressed_values is None else CompressedValues.concat(self.compressed_values, new_cv)
            )

    def _reconstruct_unprotected(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.compressed_keys is not None:
            k_unprot = self.compressor.dequantize_keys(self.compressed_keys)
            v_unprot = self.compressor.dequantize_values(self.compressed_values)
            if self.recent_keys is not None:
                k_unprot = torch.cat([k_unprot, self.recent_keys], dim=-2)
                v_unprot = torch.cat([v_unprot, self.recent_values], dim=-2)
            return k_unprot, v_unprot
        return self.recent_keys, self.recent_values

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_prefill = not self.is_initialized
        if is_prefill:
            self.lazy_initialization(key_states, value_states)

        seq_len = int(key_states.shape[-2])
        if (
            is_prefill
            and self.protect_vision_prefix
            and self._prefill_protected_mask is not None
            and int(self._prefill_protected_mask.numel()) == seq_len
            and bool(self._prefill_protected_mask.any().item())
        ):
            mask = self._prefill_protected_mask.to(device=key_states.device, dtype=torch.bool)
            self.prefill_len = seq_len
            self.protected_positions = mask.nonzero(as_tuple=False).squeeze(-1)
            self.prefill_unprotected_positions = (~mask).nonzero(as_tuple=False).squeeze(-1)
            self.protected_keys = key_states[:, :, self.protected_positions, :].contiguous()
            self.protected_values = value_states[:, :, self.protected_positions, :].contiguous()
            if self.prefill_unprotected_positions.numel() > 0:
                self._append_unprotected(
                    key_states[:, :, self.prefill_unprotected_positions, :],
                    value_states[:, :, self.prefill_unprotected_positions, :],
                )
            self._prefill_protected_mask = None
        else:
            self._append_unprotected(key_states, value_states)

        k_unprot, v_unprot = self._reconstruct_unprotected()

        # Reconstruct exact original sequence order when protected prefill slots exist
        if self.protected_keys is not None and self.protected_positions is not None:
            bs, heads, _, head_dim = self.protected_keys.shape
            k_prefill = torch.empty(
                (bs, heads, self.prefill_len, head_dim),
                dtype=self.protected_keys.dtype,
                device=self.protected_keys.device,
            )
            v_prefill = torch.empty_like(k_prefill)
            k_prefill[:, :, self.protected_positions, :] = self.protected_keys
            v_prefill[:, :, self.protected_positions, :] = self.protected_values

            n_unprot_prefill = (
                int(self.prefill_unprotected_positions.numel())
                if self.prefill_unprotected_positions is not None
                else 0
            )
            if n_unprot_prefill > 0 and k_unprot is not None and v_unprot is not None:
                k_prefill[:, :, self.prefill_unprotected_positions, :] = k_unprot[:, :, :n_unprot_prefill, :]
                v_prefill[:, :, self.prefill_unprotected_positions, :] = v_unprot[:, :, :n_unprot_prefill, :]

            if k_unprot is not None and v_unprot is not None and k_unprot.shape[-2] > n_unprot_prefill:
                k_full = torch.cat([k_prefill, k_unprot[:, :, n_unprot_prefill:, :]], dim=-2)
                v_full = torch.cat([v_prefill, v_unprot[:, :, n_unprot_prefill:, :]], dim=-2)
                return k_full, v_full
            return k_prefill, v_prefill

        assert k_unprot is not None and v_unprot is not None
        return k_unprot, v_unprot

    def get_seq_length(self) -> int:
        total = 0
        if self.protected_keys is not None:
            total += int(self.protected_keys.shape[-2])
        if self.compressed_keys is not None:
            total += self.compressed_keys.seq_len
        if self.recent_keys is not None:
            total += int(self.recent_keys.shape[-2])
        return total

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_max_length(self) -> int:
        return -1

    def reset(self) -> None:
        self.compressed_keys = None
        self.compressed_values = None
        self.recent_keys = None
        self.recent_values = None
        self._prefill_protected_mask = None
        self.protected_keys = None
        self.protected_values = None
        self.protected_positions = None
        self.prefill_unprotected_positions = None
        self.prefill_len = 0
        self.is_initialized = False
        super().reset()

    def memory_breakdown(self) -> dict[str, int]:
        protected_bytes = 0
        if self.protected_keys is not None:
            protected_bytes += self.protected_keys.numel() * self.protected_keys.element_size()
        if self.protected_values is not None:
            protected_bytes += self.protected_values.numel() * self.protected_values.element_size()

        quantized_bytes = TurboQuantCompressor.memory_bytes(self.compressed_keys, self.compressed_values)

        residual_bytes = 0
        if self.recent_keys is not None:
            residual_bytes += self.recent_keys.numel() * self.recent_keys.element_size()
        if self.recent_values is not None:
            residual_bytes += self.recent_values.numel() * self.recent_values.element_size()

        return {
            "protected_bf16_bytes": protected_bytes,
            "quantized_bytes": quantized_bytes,
            "residual_bf16_bytes": residual_bytes,
            "total_bytes": protected_bytes + quantized_bytes + residual_bytes,
        }

    def memory_bytes(self) -> int:
        return self.memory_breakdown()["total_bytes"]


class TurboQuantHybridCache(DynamicCache):
    """Hybrid cache compressing only the 6 Gated Attention layers; 18 DeltaNet layers stay untouched."""

    def __init__(
        self,
        config: Any,
        compressor: TurboQuantCompressor | None = None,
        bits: int = 3,
        residual_window: int = 64,
        protect_vision_prefix: bool = True,
        sink_tokens: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__(config=config)
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        self.compressor = compressor or TurboQuantCompressor(head_dim=head_dim, bits=bits, seed=seed)
        self.residual_window = residual_window
        self.protect_vision_prefix = protect_vision_prefix
        self.sink_tokens = sink_tokens

        # Replace ONLY full-attention DynamicLayer slots; leave all LinearAttentionLayer slots untouched
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, CacheLayerMixin):
                self.layers[idx] = TurboQuantLayer(
                    compressor=self.compressor,
                    residual_window=residual_window,
                    protect_vision_prefix=protect_vision_prefix,
                )

    def set_prefill_protected_mask(self, mask: torch.Tensor | None) -> None:
        for layer in self.layers:
            if isinstance(layer, TurboQuantLayer):
                layer.set_prefill_protected_mask(mask)


def measure_hybrid_cache_bytes(cache: DynamicCache) -> dict[str, int]:
    """Measure exact allocated tensor bytes split by protected/quantized/residual Attention KV vs DeltaNet."""
    protected_bytes = 0
    quantized_bytes = 0
    residual_bytes = 0
    attention_bytes = 0
    deltanet_bytes = 0

    for layer in cache.layers:
        if isinstance(layer, TurboQuantLayer):
            b = layer.memory_breakdown()
            protected_bytes += b["protected_bf16_bytes"]
            quantized_bytes += b["quantized_bytes"]
            residual_bytes += b["residual_bf16_bytes"]
            attention_bytes += b["total_bytes"]
        elif isinstance(layer, CacheLayerMixin):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if isinstance(keys, torch.Tensor):
                sz = keys.numel() * keys.element_size()
                attention_bytes += sz
                residual_bytes += sz
            if isinstance(values, torch.Tensor):
                sz = values.numel() * values.element_size()
                attention_bytes += sz
                residual_bytes += sz
        elif isinstance(layer, LinearAttentionLayer):
            for state_dict in (getattr(layer, "conv_states", None), getattr(layer, "recurrent_states", None)):
                if isinstance(state_dict, dict):
                    for tensor in state_dict.values():
                        if isinstance(tensor, torch.Tensor):
                            deltanet_bytes += tensor.numel() * tensor.element_size()
                elif isinstance(state_dict, (list, tuple)):
                    for tensor in state_dict:
                        if isinstance(tensor, torch.Tensor):
                            deltanet_bytes += tensor.numel() * tensor.element_size()

    return {
        "attention_kv_bytes": attention_bytes,
        "protected_bf16_bytes": protected_bytes,
        "quantized_bytes": quantized_bytes,
        "residual_bf16_bytes": residual_bytes,
        "deltanet_state_bytes": deltanet_bytes,
        "deltanet_bytes": deltanet_bytes,
        "total_bytes": attention_bytes + deltanet_bytes,
    }
