"""Automated 4-check GPU verification suite for TurboQuant (~3-bit) Hybrid KV-Cache compression."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers.cache_utils import LinearAttentionLayer

from infer_looped import DEFAULT_SYSTEM_PROMPT, DEFAULT_TEST_PROMPT, load_looped_checkpoint, split_thinking_and_answer
from rewards import score_completion
from turboquant_cache import TurboQuantCompressor, TurboQuantLayer

GRPO_CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "grpo_smoke"


def mb(num_bytes: int) -> float:
    return num_bytes / (1024**2)


def run_prefill_and_decode(
    model,
    input_ids: torch.LongTensor,
    decode_steps: int,
    kv_quant: str,
    residual_window: int = 64,
):
    """Run prefill + `decode_steps` greedy cached decode steps and return `(caches, peak_vram_gb)`."""
    device = input_ids.device
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    model._core_model.model.rope_deltas = None
    caches = model.create_per_loop_caches(
        num_loops=model.num_loops,
        kv_quant=kv_quant,
        residual_window=residual_window,
    )

    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            per_loop_caches=caches,
            use_cache=True,
            logits_to_keep=1,
        )
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        cur_mask = torch.cat(
            [attention_mask, torch.ones((input_ids.shape[0], 1), dtype=attention_mask.dtype, device=device)],
            dim=-1,
        )
        for _ in range(decode_steps):
            pos_ids = model._build_decode_position_ids(cur_mask)
            out = model(
                input_ids=next_tok,
                attention_mask=cur_mask,
                position_ids=pos_ids,
                per_loop_caches=caches,
                use_cache=True,
                logits_to_keep=1,
            )
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            cur_mask = torch.cat(
                [cur_mask, torch.ones((input_ids.shape[0], 1), dtype=cur_mask.dtype, device=device)],
                dim=-1,
            )

    peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
    return caches, peak_gb


def main() -> int:
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is required for verify_turboquant.py.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # ------------------------------------------------------------------
    # CHECK 1: Unit Compression & Fidelity Check (bits=3, 2048 tokens)
    # ------------------------------------------------------------------
    print("=" * 74)
    print("CHECK 1: Unit Compression & Fidelity Check (TurboQuant bits=3)")
    print("=" * 74)
    head_dim = 256
    compressor = TurboQuantCompressor(head_dim=head_dim, bits=3, seed=42, device=device)

    keys = torch.randn((1, 2, 2048, head_dim), dtype=torch.bfloat16, device=device)
    values = torch.randn((1, 2, 2048, head_dim), dtype=torch.bfloat16, device=device)
    queries = torch.randn((1, 2, 32, head_dim), dtype=torch.bfloat16, device=device)

    ck = compressor.compress_keys(keys)
    cv = compressor.compress_values(values)
    k_rec = compressor.dequantize_keys(ck)
    v_rec = compressor.dequantize_values(cv)

    orig_k_bytes = keys.numel() * keys.element_size()
    orig_v_bytes = values.numel() * values.element_size()
    comp_k_bytes = ck.memory_bytes()
    comp_v_bytes = cv.memory_bytes()
    total_ratio = (orig_k_bytes + orig_v_bytes) / (comp_k_bytes + comp_v_bytes)

    k_cos = F.cosine_similarity(keys.float(), k_rec.float(), dim=-1).mean().item()
    v_cos = F.cosine_similarity(values.float(), v_rec.float(), dim=-1).mean().item()

    # Inner-product error (scaled dot-product attention logits Q @ K^T / sqrt(d))
    scale = head_dim**-0.5
    ip_true = torch.matmul(queries.float(), keys.float().transpose(-2, -1)) * scale
    ip_est = torch.matmul(queries.float(), k_rec.float().transpose(-2, -1)) * scale
    ip_rmse = torch.sqrt(F.mse_loss(ip_est, ip_true)).item()
    ip_bias = (ip_est - ip_true).mean().item()

    assert ck.packed_indices.dtype == torch.uint8 and ck.packed_qjl_signs.dtype == torch.uint8
    assert cv.packed_indices.dtype == torch.uint8
    assert total_ratio >= 4.0, f"Expected >=4x compression, got {total_ratio:.2f}x"
    assert not torch.isnan(k_rec).any() and not torch.isnan(v_rec).any()

    print(f"Keys   bf16={mb(orig_k_bytes):.3f} MB -> TurboQuant={mb(comp_k_bytes):.3f} MB ({orig_k_bytes / comp_k_bytes:.2f}x)")
    print(f"Values bf16={mb(orig_v_bytes):.3f} MB -> TurboQuant={mb(comp_v_bytes):.3f} MB ({orig_v_bytes / comp_v_bytes:.2f}x)")
    print(f"Combined KV compression : {total_ratio:.2f}x")
    print(f"Key   cosine similarity : {k_cos:.4f}")
    print(f"Value cosine similarity : {v_cos:.4f}")
    print(f"Scaled Q@K^T RMSE       : {ip_rmse:.4f} | mean bias: {ip_bias:+.5f}")
    print("CHECK 1 PASSED")

    del keys, values, queries, ck, cv, k_rec, v_rec
    torch.cuda.empty_cache()

    # Load GRPO checkpoint once for Checks 2, 3, and 4
    print(f"\nLoading GRPO checkpoint: {GRPO_CHECKPOINT_DIR} (num_loops=2, bf16)...")
    model, processor = load_looped_checkpoint(GRPO_CHECKPOINT_DIR, device=device, num_loops_override=2)

    # ------------------------------------------------------------------
    # CHECK 2: DeltaNet Untouched Check (18 linear_attention layers)
    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("CHECK 2: DeltaNet Untouched Check (18 linear_attention layers)")
    print("=" * 74)
    short_ids = torch.randint(100, 5000, (1, 32), dtype=torch.long, device=device)
    caches_none, _ = run_prefill_and_decode(model, short_ids, decode_steps=0, kv_quant="none", residual_window=64)
    caches_tq, _ = run_prefill_and_decode(
        model, short_ids, decode_steps=0, kv_quant="turboquant_3bit", residual_window=64
    )

    linear_count = 0
    attn_tq_count = 0
    max_conv_diff = 0.0
    max_rec_diff = 0.0
    conv_dtypes: set[str] = set()
    rec_dtypes: set[str] = set()

    for loop_idx in range(model.num_loops):
        for layer_idx, (l_none, l_tq) in enumerate(zip(caches_none[loop_idx].layers, caches_tq[loop_idx].layers)):
            if isinstance(l_none, LinearAttentionLayer):
                assert isinstance(l_tq, LinearAttentionLayer), f"Layer {layer_idx} was unexpectedly wrapped!"
                if loop_idx == 0:
                    linear_count += 1
                c_none = l_none.conv_states[0]
                c_tq = l_tq.conv_states[0]
                r_none = l_none.recurrent_states[0]
                r_tq = l_tq.recurrent_states[0]
                assert c_none.dtype == c_tq.dtype and r_none.dtype == r_tq.dtype
                conv_dtypes.add(str(c_tq.dtype))
                rec_dtypes.add(str(r_tq.dtype))
                max_conv_diff = max(max_conv_diff, (c_none - c_tq).abs().max().item())
                max_rec_diff = max(max_rec_diff, (r_none - r_tq).abs().max().item())
            else:
                assert isinstance(l_tq, TurboQuantLayer), f"Attention layer {layer_idx} not TurboQuantLayer!"
                if loop_idx == 0:
                    attn_tq_count += 1

    print(f"Verified layer split    : {linear_count} LinearAttentionLayer (untouched) + {attn_tq_count} TurboQuantLayer")
    print(f"DeltaNet state dtypes   : conv={sorted(conv_dtypes)}, recurrent={sorted(rec_dtypes)}")
    print(f"Max DeltaNet diff step0 : conv={max_conv_diff:.6e}, recurrent={max_rec_diff:.6e}")
    assert linear_count == 18 and attn_tq_count == 6
    assert max_conv_diff == 0.0 and max_rec_diff == 0.0
    print("CHECK 2 PASSED")

    del caches_none, caches_tq, short_ids
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # CHECK 3: Long-Context Memory Benchmark (2048 prefill + 16 decode, num_loops=2)
    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("CHECK 3: Long-Context Memory Benchmark (2048 prefill + 16 decode, L=2)")
    print("=" * 74)
    long_ids = torch.randint(100, 5000, (1, 2048), dtype=torch.long, device=device)

    caches_fp, peak_fp_gb = run_prefill_and_decode(
        model, long_ids, decode_steps=16, kv_quant="none", residual_window=64
    )
    stats_fp = caches_fp.kv_cache_memory_bytes()
    del caches_fp
    gc.collect()
    torch.cuda.empty_cache()

    caches_tq, peak_tq_gb = run_prefill_and_decode(
        model, long_ids, decode_steps=16, kv_quant="turboquant_3bit", residual_window=64
    )
    stats_tq = caches_tq.kv_cache_memory_bytes()
    seq_len_tq = caches_tq.get_seq_length()
    del caches_tq, long_ids
    gc.collect()
    torch.cuda.empty_cache()

    attn_ratio = stats_fp["attention_kv_bytes"] / stats_tq["attention_kv_bytes"]
    print(f"Total cached seq_len    : {seq_len_tq} tokens (2048 prefill + 16 decode) x {model.num_loops} loops")
    print(
        f"kv_quant='none'         : Attn KV={mb(stats_fp['attention_kv_bytes']):.2f} MB | "
        f"DeltaNet={mb(stats_fp['deltanet_state_bytes']):.2f} MB | Peak VRAM={peak_fp_gb:.2f} GB"
    )
    print(
        f"kv_quant='turboquant_3b': Attn KV={mb(stats_tq['attention_kv_bytes']):.2f} MB | "
        f"DeltaNet={mb(stats_tq['deltanet_state_bytes']):.2f} MB | Peak VRAM={peak_tq_gb:.2f} GB"
    )
    print(f"Attention KV compression: {attn_ratio:.2f}x (DeltaNet bytes identical: {stats_fp['deltanet_state_bytes'] == stats_tq['deltanet_state_bytes']})")
    assert attn_ratio >= 4.0, f"Expected >=4x attention KV compression at 2064 tokens, got {attn_ratio:.2f}x"
    assert stats_fp["deltanet_state_bytes"] == stats_tq["deltanet_state_bytes"]
    print("CHECK 3 PASSED")

    # ------------------------------------------------------------------
    # CHECK 4: End-to-End GRPO Checkpoint Generation (task_001, turboquant_3bit)
    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("CHECK 4: End-to-End GRPO Generation (kv_quant='turboquant_3bit', task_001)")
    print("=" * 74)
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": DEFAULT_TEST_PROMPT}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in dict(inputs).items()}

    # Use residual_window=32 so prompt tokens (>100 tokens) are actively compressed by TurboQuant during generation
    gen_ids = model.generate_looped(
        **inputs,
        max_new_tokens=160,
        num_loops=2,
        use_cache=True,
        kv_quant="turboquant_3bit",
        residual_window=32,
        do_sample=False,
    )
    new_ids = gen_ids[:, inputs["input_ids"].shape[1] :]
    decoded = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    thinking, final_answer = split_thinking_and_answer(decoded)
    score = score_completion(decoded, expected_tool="create_support_ticket", tokenizer=processor.tokenizer)

    print(f"Generated tokens : {new_ids.shape[1]} (prompt_len={inputs['input_ids'].shape[1]}, residual_window=32)")
    print(f"<think>\n{thinking}\n</think>")
    print(f"\n[Tool Call]\n{final_answer}")
    print(f"\nReward score     : total={score['total_reward']} | tool={score['parsed_tool']} | errors={score['schema_errors']}")
    assert score["total_reward"] == 1.0, f"Expected perfect tool-call reward=1.0, got {score}"
    print("CHECK 4 PASSED")

    print("\nALL 4 TURBOQUANT VERIFICATION CHECKS PASSED!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
