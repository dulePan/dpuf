import os
import time
import math
import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
import numpy as np
import sentencepiece as spm

import test_lr_flac_numpy as t
import compress_full_model as cfm


# ==============================================================================
# 1. TAČNA T5 ARHITEKTURA SA RELATIVE POSITION BIAS-OM I KAUZALNOM MASKOM
# ==============================================================================
def t5_relative_position_bucket(
    relative_position: np.ndarray,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> np.ndarray:
    """Tačna HuggingFace / Google T5 formula za relativne pozicione bakete."""
    relative_buckets = np.zeros_like(relative_position, dtype=np.int32)
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).astype(np.int32) * num_buckets
        relative_position = np.abs(relative_position)
    else:
        relative_position = -np.minimum(relative_position, np.zeros_like(relative_position))

    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    relative_position_if_large = max_exact + (
        np.log(np.maximum(relative_position.astype(np.float32), 1.0) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype(np.int32)
    relative_position_if_large = np.minimum(relative_position_if_large, num_buckets - 1)

    relative_buckets += np.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


def compute_t5_position_bias(
    q_len: int,
    kv_len: int,
    rel_bias_u16: np.ndarray,
    bidirectional: bool,
) -> np.ndarray:
    """Računa [1, n_heads, q_len, kv_len] tenzor pozicionog biasa iz [32, 12] tabele."""
    context_position = np.arange(q_len, dtype=np.int32)[:, None]
    memory_position = np.arange(kv_len, dtype=np.int32)[None, :]
    relative_position = memory_position - context_position
    rp_bucket = t5_relative_position_bucket(relative_position, bidirectional=bidirectional)
    rel_bias_fp32 = t.bf16_u16_to_fp32(rel_bias_u16)  # [32, 12]
    values = rel_bias_fp32[rp_bucket]  # [q_len, kv_len, 12]
    return values.transpose(2, 0, 1)[None, :, :, :]  # [1, 12, q_len, kv_len]


def t5_attention_with_bias(
    hidden: np.ndarray,
    kv_state: np.ndarray,
    q_u16: np.ndarray,
    k_u16: np.ndarray,
    v_u16: np.ndarray,
    o_u16: np.ndarray,
    position_bias: np.ndarray,
    causal_mask: np.ndarray = None,
    n_heads: int = 12,
) -> np.ndarray:
    """T5 Multi-Head Attention sa Relative Position Bias-om i opcionom kauzalnom maskom."""
    bsz, q_len, d_model = hidden.shape
    kv_len = kv_state.shape[1]
    d_kv = d_model // n_heads

    Wq = t.bf16_u16_to_fp32(q_u16)
    Wk = t.bf16_u16_to_fp32(k_u16)
    Wv = t.bf16_u16_to_fp32(v_u16)
    Wo = t.bf16_u16_to_fp32(o_u16)

    Q = (hidden @ Wq.T).reshape(bsz, q_len, n_heads, d_kv).transpose(0, 2, 1, 3)
    K = (kv_state @ Wk.T).reshape(bsz, kv_len, n_heads, d_kv).transpose(0, 2, 1, 3)
    V = (kv_state @ Wv.T).reshape(bsz, kv_len, n_heads, d_kv).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) + position_bias
    if causal_mask is not None:
        scores = scores + causal_mask

    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    context = (probs @ V).transpose(0, 2, 1, 3).reshape(bsz, q_len, d_model)
    return context @ Wo.T


def encode_t5_prompt(tensors: Dict[str, np.ndarray], input_ids: np.ndarray) -> np.ndarray:
    """Propušta ulazni prompt kroz svih 12 Encoder Transformer blokova."""
    shared_fp32 = t.bf16_u16_to_fp32(tensors["shared.weight"])
    enc_hidden = shared_fp32[input_ids]
    q_len = input_ids.shape[1]

    pos_bias = compute_t5_position_bias(
        q_len,
        q_len,
        tensors["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
        bidirectional=True,
    )

    for i in range(12):
        pfx = f"encoder.block.{i}."
        normed = cfm.t5_rms_norm(enc_hidden, tensors[pfx + "layer.0.layer_norm.weight"])
        attn_out = t5_attention_with_bias(
            normed,
            normed,
            tensors[pfx + "layer.0.SelfAttention.q.weight"],
            tensors[pfx + "layer.0.SelfAttention.k.weight"],
            tensors[pfx + "layer.0.SelfAttention.v.weight"],
            tensors[pfx + "layer.0.SelfAttention.o.weight"],
            position_bias=pos_bias,
        )
        enc_hidden = enc_hidden + attn_out

        normed_ff = cfm.t5_rms_norm(enc_hidden, tensors[pfx + "layer.1.layer_norm.weight"])
        wi = t.bf16_u16_to_fp32(tensors[pfx + "layer.1.DenseReluDense.wi.weight"])
        wo = t.bf16_u16_to_fp32(tensors[pfx + "layer.1.DenseReluDense.wo.weight"])
        ff_out = np.maximum(normed_ff @ wi.T, 0.0) @ wo.T
        enc_hidden = enc_hidden + ff_out

    return cfm.t5_rms_norm(enc_hidden, tensors["encoder.final_layer_norm.weight"])


def generate_t5_greedy(
    tensors: Dict[str, np.ndarray],
    input_ids: np.ndarray,
    max_new_tokens: int = 32,
) -> Tuple[List[int], np.ndarray]:
    """
    Autoregresivno (reč po reč / token po token) generiše odgovor kroz svih 24 sloja T5-Base modela!
    Vraća listu generisanih token ID-jeva i vektor logita prvog koraka za poređenje kosinusne sličnosti.
    """
    shared_fp32 = t.bf16_u16_to_fp32(tensors["shared.weight"])
    enc_hidden = encode_t5_prompt(tensors, input_ids)
    enc_len = enc_hidden.shape[1]

    # T5 dekoder počinje sa <pad> tokenom (id = 0)
    dec_tokens = [0]
    first_step_logits = None
    scale = 768 ** -0.5

    for step in range(max_new_tokens):
        dec_arr = np.array([dec_tokens], dtype=np.int32)
        dec_hidden = shared_fp32[dec_arr]
        dec_len = len(dec_tokens)

        self_pos_bias = compute_t5_position_bias(
            dec_len,
            dec_len,
            tensors["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
            bidirectional=False,
        )
        causal_mask = np.triu(np.full((1, 1, dec_len, dec_len), -1e9, dtype=np.float32), k=1)
        cross_pos_bias = np.zeros((1, 12, dec_len, enc_len), dtype=np.float32)

        for i in range(12):
            pfx = f"decoder.block.{i}."
            normed = cfm.t5_rms_norm(dec_hidden, tensors[pfx + "layer.0.layer_norm.weight"])
            self_attn = t5_attention_with_bias(
                normed,
                normed,
                tensors[pfx + "layer.0.SelfAttention.q.weight"],
                tensors[pfx + "layer.0.SelfAttention.k.weight"],
                tensors[pfx + "layer.0.SelfAttention.v.weight"],
                tensors[pfx + "layer.0.SelfAttention.o.weight"],
                position_bias=self_pos_bias,
                causal_mask=causal_mask,
            )
            dec_hidden = dec_hidden + self_attn

            normed_cross = cfm.t5_rms_norm(dec_hidden, tensors[pfx + "layer.1.layer_norm.weight"])
            cross_attn = t5_attention_with_bias(
                normed_cross,
                enc_hidden,
                tensors[pfx + "layer.1.EncDecAttention.q.weight"],
                tensors[pfx + "layer.1.EncDecAttention.k.weight"],
                tensors[pfx + "layer.1.EncDecAttention.v.weight"],
                tensors[pfx + "layer.1.EncDecAttention.o.weight"],
                position_bias=cross_pos_bias,
            )
            dec_hidden = dec_hidden + cross_attn

            normed_ff = cfm.t5_rms_norm(dec_hidden, tensors[pfx + "layer.2.layer_norm.weight"])
            wi = t.bf16_u16_to_fp32(tensors[pfx + "layer.2.DenseReluDense.wi.weight"])
            wo = t.bf16_u16_to_fp32(tensors[pfx + "layer.2.DenseReluDense.wo.weight"])
            ff_out = np.maximum(normed_ff @ wi.T, 0.0) @ wo.T
            dec_hidden = dec_hidden + ff_out

        dec_hidden = cfm.t5_rms_norm(dec_hidden, tensors["decoder.final_layer_norm.weight"])
        next_logits = (dec_hidden[:, -1, :] * scale) @ shared_fp32.T  # [1, 32128]
        if first_step_logits is None:
            first_step_logits = next_logits[0].copy()

        next_token = int(np.argmax(next_logits[0]))
        if next_token == 1:  # </s> EOS token
            break
        dec_tokens.append(next_token)

    return dec_tokens[1:], first_step_logits


# ==============================================================================
# 2. GLAVNI TEST GENERISANJA PRAVIH REČENICA (KORAK B)
# ==============================================================================
def main():
    base_dir = (
        pathlib.Path.home()
        / ".cache/huggingface/hub/models--t5-base/snapshots/fe6d9bf207cd3337512ca838a8b453f87a9178ef"
    )
    safetensors_path = base_dir / "model.safetensors"
    spiece_path = base_dir / "spiece.model"

    sp = spm.SentencePieceProcessor(model_file=str(spiece_path))
    orig_tensors = cfm.load_full_safetensors_bf16(safetensors_path)
    names = list(orig_tensors.keys())
    total_params = sum(arr.size for arr in orig_tensors.values())
    raw_mb = sum(arr.nbytes for arr in orig_tensors.values()) / (1024 ** 2)
    workers = min(16, os.cpu_count() or 8)

    print("=" * 96)
    print("KORAK B: UŽIVO GENERISANJE TEKSTA I PREVOD REČENICA NA KOMPRESOVANIM MODELIMA (T5-Base 222.9M)")
    print("=" * 96)

    # Pripremamo sve verzije modela (Original + Lossless + 5 AV1 BFP nivoa)
    models_to_test: List[Tuple[str, float, float, Dict[str, np.ndarray]]] = [
        ("0. Originalni BFloat16 ('WAV')", raw_mb, 0.0, orig_tensors),
    ]

    # 100% Lossless
    with ThreadPoolExecutor(max_workers=workers) as ex:
        lossless_blobs = dict(zip(names, ex.map(lambda k: cfm.encode_tensor_lossless_fast(orig_tensors[k], 6), names)))
        rec_lossless = dict(zip(names, ex.map(cfm.decode_tensor_lossless_fast, [lossless_blobs[k] for k in names])))
    ll_mb = sum(len(b) for b in lossless_blobs.values()) / (1024 ** 2)
    models_to_test.append(("1. LR-FLAC 100% Lossless", ll_mb, (1 - ll_mb / raw_mb) * 100, rec_lossless))

    # AV1 Near-Lossless nivoi
    for idx, (label_cfg, step_mult, use_nibble) in enumerate(cfm.BFP_CONFIGS, start=2):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            blobs = dict(
                zip(names, ex.map(lambda k: cfm.encode_tensor_bfp_fast(orig_tensors[k], step_mult, use_nibble, 6), names))
            )
            rec_tensors = dict(zip(names, ex.map(cfm.decode_tensor_bfp_fast, [blobs[k] for k in names])))
        mb = sum(len(b) for b in blobs.values()) / (1024 ** 2)
        sav = (1 - mb / raw_mb) * 100
        models_to_test.append((f"{idx}. {label_cfg}", mb, sav, rec_tensors))

    prompts = [
        (
            "Prevod EN -> DE (Nemački)",
            "translate English to German: Artificial intelligence and data compression are transforming modern technology.",
        ),
        (
            "Prevod EN -> FR (Francuski)",
            "translate English to French: The new compression algorithm reduces memory usage without losing accuracy.",
        ),
        (
            "Prevod EN -> RO (Rumunski)",
            "translate English to Romanian: Large language models require fast decompression and high precision.",
        ),
        (
            "Sažimanje teksta (Summarization)",
            "summarize: Researchers developed a new compression format for large language models called LR-FLAC. By combining two-dimensional scaling with nibble packing, the full model size is reduced by over seventy percent while maintaining fast decompression speed and accurate text generation.",
        ),
    ]

    for task_title, prompt_text in prompts:
        print("\n" + "-" * 96)
        print(f"ZADATAK: {task_title}")
        print(f"ULAZ:    \"{prompt_text}\"")
        print("-" * 96)

        input_ids = np.array([sp.encode(prompt_text, out_type=int) + [1]], dtype=np.int32)
        ref_tokens: List[int] = []

        for label, mb, sav, m_tensors in models_to_test:
            out_tokens, _ = generate_t5_greedy(m_tensors, input_ids, max_new_tokens=30)
            decoded_text = sp.decode(out_tokens)
            if not ref_tokens:
                ref_tokens = out_tokens
                match_str = "REFERENCA (100%)"
            else:
                if out_tokens == ref_tokens:
                    match_str = "✅ 100% IDENTIČNO"
                else:
                    # Procenat poklapanja reči
                    ref_words = sp.decode(ref_tokens).split()
                    out_words = decoded_text.split()
                    common = sum(1 for w in out_words if w in ref_words)
                    pct = (common / max(len(ref_words), 1)) * 100
                    match_str = f"~{pct:.0f}% poklapanje reči"

            print(
                f"  [{label:<41} | {mb:6.1f} MB (-{sav:4.1f}%) | {match_str:<18}]:\n"
                f"     -> \"{decoded_text}\""
            )


if __name__ == "__main__":
    main()
