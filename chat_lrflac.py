import os
import sys
import time
import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple
import numpy as np
import sentencepiece as spm

import test_lr_flac_numpy as t
import compress_full_model as cfm
import test_live_text_generation as tltg
import test_qwen_llama_lrflac as tql


# ==============================================================================
# 1. DIREKTNI ZERO-DECODE T5 TRANSFORMER ENGINE (BEZ DEKOMPRESIJE MATRICA U BF16!)
# ==============================================================================
def t5_rms_norm_zd(x: np.ndarray, weight: Any, eps: float = 1e-6) -> np.ndarray:
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    w_f32 = t.bf16_u16_to_fp32(weight) if weight.dtype == np.uint16 else weight
    return (x / np.sqrt(variance + eps)) * w_f32


def t5_attention_zero_decode(
    hidden: np.ndarray,
    kv_state: np.ndarray,
    wq: Any,
    wk: Any,
    wv: Any,
    wo: Any,
    position_bias: np.ndarray,
    causal_mask: np.ndarray = None,
    n_heads: int = 12,
) -> np.ndarray:
    """
    Izvršava T5 Multi-Head Attention DIREKTNO nad ZeroDecodeWeight objektima iz .lrflac fajla:
      Y = ((X * col_rms) @ Q^T) * (row_rms * step_mult)
    Bez kreiranja BFloat16 ili Float32 matrica težina u memoriji!
    """
    bsz, q_len, d_model = hidden.shape
    kv_len = kv_state.shape[1]
    d_kv = d_model // n_heads

    Q = tql.linear_op(hidden, wq).reshape(bsz, q_len, n_heads, d_kv).transpose(0, 2, 1, 3)
    K = tql.linear_op(kv_state, wk).reshape(bsz, kv_len, n_heads, d_kv).transpose(0, 2, 1, 3)
    V = tql.linear_op(kv_state, wv).reshape(bsz, kv_len, n_heads, d_kv).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) + position_bias
    if causal_mask is not None:
        scores = scores + causal_mask

    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    context = (probs @ V).transpose(0, 2, 1, 3).reshape(bsz, q_len, d_model)
    return tql.linear_op(context, wo)


def compute_pos_bias_zd(q_len: int, kv_len: int, rel_bias_w: Any, bidirectional: bool) -> np.ndarray:
    context_position = np.arange(q_len, dtype=np.int32)[:, None]
    memory_position = np.arange(kv_len, dtype=np.int32)[None, :]
    relative_position = memory_position - context_position
    rp_bucket = tltg.t5_relative_position_bucket(relative_position, bidirectional=bidirectional)
    if isinstance(rel_bias_w, tql.ZeroDecodeWeight):
        rel_bias_fp32 = (
            rel_bias_w.q_int8.astype(np.float32)
            * rel_bias_w.row_scale[:, None]
            * rel_bias_w.col_scale[None, :]
        )
    elif rel_bias_w.dtype == np.uint16:
        rel_bias_fp32 = t.bf16_u16_to_fp32(rel_bias_w)
    else:
        rel_bias_fp32 = rel_bias_w
    values = rel_bias_fp32[rp_bucket]
    return values.transpose(2, 0, 1)[None, :, :, :]


def generate_direct_from_lrflac(
    zd_weights: Dict[str, Any],
    input_ids: np.ndarray,
    max_new_tokens: int = 40,
) -> Tuple[List[int], np.ndarray]:
    """
    Direktna autoregresivna generacija teksta (token-po-token) iz .lrflac ZeroDecodeWeight objekata
    BEZ dekompresije modela u BFloat16/Float32!
    """
    shared_w = zd_weights["shared.weight"]
    enc_hidden = tql.embedding_lookup(shared_w, input_ids)
    q_len = input_ids.shape[1]

    pos_bias = compute_pos_bias_zd(
        q_len,
        q_len,
        zd_weights["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
        bidirectional=True,
    )

    for i in range(12):
        pfx = f"encoder.block.{i}."
        normed = t5_rms_norm_zd(enc_hidden, zd_weights[pfx + "layer.0.layer_norm.weight"])
        attn_out = t5_attention_zero_decode(
            normed,
            normed,
            zd_weights[pfx + "layer.0.SelfAttention.q.weight"],
            zd_weights[pfx + "layer.0.SelfAttention.k.weight"],
            zd_weights[pfx + "layer.0.SelfAttention.v.weight"],
            zd_weights[pfx + "layer.0.SelfAttention.o.weight"],
            position_bias=pos_bias,
        )
        enc_hidden = enc_hidden + attn_out

        normed_ff = t5_rms_norm_zd(enc_hidden, zd_weights[pfx + "layer.1.layer_norm.weight"])
        wi_out = np.maximum(tql.linear_op(normed_ff, zd_weights[pfx + "layer.1.DenseReluDense.wi.weight"]), 0.0)
        ff_out = tql.linear_op(wi_out, zd_weights[pfx + "layer.1.DenseReluDense.wo.weight"])
        enc_hidden = enc_hidden + ff_out

    enc_hidden = t5_rms_norm_zd(enc_hidden, zd_weights["encoder.final_layer_norm.weight"])
    enc_len = enc_hidden.shape[1]

    dec_tokens = [0]
    first_step_logits = None
    scale = np.float32(768 ** -0.5)

    for _ in range(max_new_tokens):
        dec_arr = np.array([dec_tokens], dtype=np.int32)
        dec_hidden = tql.embedding_lookup(shared_w, dec_arr)
        dec_len = len(dec_tokens)

        self_pos_bias = compute_pos_bias_zd(
            dec_len,
            dec_len,
            zd_weights["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"],
            bidirectional=False,
        )
        causal_mask = np.triu(np.full((1, 1, dec_len, dec_len), -1e9, dtype=np.float32), k=1)
        cross_pos_bias = np.zeros((1, 12, dec_len, enc_len), dtype=np.float32)

        for i in range(12):
            pfx = f"decoder.block.{i}."
            normed = t5_rms_norm_zd(dec_hidden, zd_weights[pfx + "layer.0.layer_norm.weight"])
            self_attn = t5_attention_zero_decode(
                normed,
                normed,
                zd_weights[pfx + "layer.0.SelfAttention.q.weight"],
                zd_weights[pfx + "layer.0.SelfAttention.k.weight"],
                zd_weights[pfx + "layer.0.SelfAttention.v.weight"],
                zd_weights[pfx + "layer.0.SelfAttention.o.weight"],
                position_bias=self_pos_bias,
                causal_mask=causal_mask,
            )
            dec_hidden = dec_hidden + self_attn

            normed_cross = t5_rms_norm_zd(dec_hidden, zd_weights[pfx + "layer.1.layer_norm.weight"])
            cross_attn = t5_attention_zero_decode(
                normed_cross,
                enc_hidden,
                zd_weights[pfx + "layer.1.EncDecAttention.q.weight"],
                zd_weights[pfx + "layer.1.EncDecAttention.k.weight"],
                zd_weights[pfx + "layer.1.EncDecAttention.v.weight"],
                zd_weights[pfx + "layer.1.EncDecAttention.o.weight"],
                position_bias=cross_pos_bias,
            )
            dec_hidden = dec_hidden + cross_attn

            normed_ff = t5_rms_norm_zd(dec_hidden, zd_weights[pfx + "layer.2.layer_norm.weight"])
            wi_out = np.maximum(tql.linear_op(normed_ff, zd_weights[pfx + "layer.2.DenseReluDense.wi.weight"]), 0.0)
            ff_out = tql.linear_op(wi_out, zd_weights[pfx + "layer.2.DenseReluDense.wo.weight"])
            dec_hidden = dec_hidden + ff_out

        dec_hidden = t5_rms_norm_zd(dec_hidden, zd_weights["decoder.final_layer_norm.weight"])
        next_logits = tql.linear_op(dec_hidden[:, -1, :] * scale, shared_w)
        if first_step_logits is None:
            first_step_logits = next_logits[0].copy()

        next_token = int(np.argmax(next_logits[0]))
        if next_token == 1:
            break
        dec_tokens.append(next_token)

    return dec_tokens[1:], first_step_logits


def build_or_load_chat_lrflac(
    safetensors_path: pathlib.Path,
    lrflac_path: pathlib.Path,
    step_mult: float = 0.200,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], float, float]:
    """
    Pretvara ceo model (.safetensors, 425.16 MB BF16) u Direct Zero-Decode .lrflac fajl
    (ako već ne postoji) i učitava ga direktno u ZeroDecodeWeight strukture sa CRC32 verifikacijom!
    """
    orig_tensors = cfm.load_full_safetensors_bf16(safetensors_path)
    names = list(orig_tensors.keys())
    workers = min(16, os.cpu_count() or 8)

    if not lrflac_path.exists():
        print(f"Pakujem kompletan model u Direct Zero-Decode .lrflac fajl: {lrflac_path} ...")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            zd_blobs = dict(
                zip(
                    names,
                    ex.map(lambda k: tql.encode_tensor_direct_zero_decode_4bit(orig_tensors[k], step_mult), names),
                )
            )
        tql.save_secure_lrflac_container(lrflac_path, zd_blobs)

    t0 = time.perf_counter()
    loaded_blobs = tql.load_secure_lrflac_container(lrflac_path, verify_crc=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        zd_weights = dict(zip(names, ex.map(tql.unpack_zero_decode_tensor, [loaded_blobs[k] for k in names])))
    load_s = time.perf_counter() - t0
    file_mb = lrflac_path.stat().st_size / (1024 ** 2)
    return orig_tensors, zd_weights, file_mb, load_s


def main():
    base_dir = (
        pathlib.Path.home()
        / ".cache/huggingface/hub/models--t5-base/snapshots/fe6d9bf207cd3337512ca838a8b453f87a9178ef"
    )
    safetensors_path = base_dir / "model.safetensors"
    spiece_path = base_dir / "spiece.model"
    lrflac_chat_path = pathlib.Path("artifacts_lrflac/t5_base_zero_decode_chat.lrflac")

    sp = spm.SentencePieceProcessor(model_file=str(spiece_path))
    orig_tensors, zd_weights, lrflac_mb, load_s = build_or_load_chat_lrflac(
        safetensors_path, lrflac_chat_path, step_mult=0.200
    )
    orig_mb = sum(v.nbytes for v in orig_tensors.values()) / (1024 ** 2)
    savings_pct = (1.0 - lrflac_mb / orig_mb) * 100.0

    print("\n" + "=" * 108)
    print("DIREKTAN CHAT & KOMUNIKACIJA SA MODELOM IZ .LRFLAC FAJLA (BEZ ZLIB DEKODERA I BEZ BF16 REKONSTRUKCIJE!)")
    print("=" * 108)
    print(f"Originalni .safetensors (BFloat16): {orig_mb:.2f} MB")
    print(f"Naš Direct Zero-Decode .lrflac:     {lrflac_mb:.2f} MB (Ušteda: -{savings_pct:.1f}%, CRC32 Verifikovan = True)")
    print(f"Vreme učitavanja .lrflac modela:    {load_s:.3f} s ({orig_mb / max(load_s, 1e-4):.1f} MB/s efektivni protok)")
    print("Način izvršavanja:                  Y = ((X * col_rms) @ Q^T) * (row_rms * step_mult) direktno nad int MatMul!")
    print("-" * 108)

    test_prompts = [
        (
            "Prevod EN -> DE (AI & Kompresija)",
            "translate English to German: Artificial intelligence and data compression are transforming modern technology.",
        ),
        (
            "Prevod EN -> FR (Brza komunikacija)",
            "translate English to French: We can now communicate directly with the compressed language model without decoding weights.",
        ),
        (
            "Prevod EN -> RO (Preciznost modela)",
            "translate English to Romanian: Large language models require fast decompression and high precision.",
        ),
        (
            "Prevod EN -> DE (Bezbednost fajla)",
            "translate English to German: The new file format is completely safe from malicious code and memory attacks.",
        ),
        (
            "Sažimanje teksta (Summarization)",
            "summarize: The new LR-FLAC format reduces large language model size by over seventy percent while allowing direct matrix multiplication on quantized weights without dequantizing the full matrix into memory. This saves gigabytes of RAM and enables fast local inference on laptops and phones.",
        ),
    ]

    for title, prompt in test_prompts:
        ids = sp.encode(prompt, out_type=int) + [1]
        inp = np.array([ids], dtype=np.int32)

        t0 = time.perf_counter()
        orig_toks, orig_logits = tltg.generate_t5_greedy(orig_tensors, inp, max_new_tokens=36)
        t_orig = time.perf_counter() - t0

        t0 = time.perf_counter()
        zd_toks, zd_logits = generate_direct_from_lrflac(zd_weights, inp, max_new_tokens=36)
        t_zd = time.perf_counter() - t0

        a = orig_logits.astype(np.float64)
        b = zd_logits.astype(np.float64)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        orig_text = sp.decode(orig_toks)
        zd_text = sp.decode(zd_toks)

        print(f"\n[{title}]")
        print(f"  Upit (Prompt):                 \"{prompt}\"")
        print(f"  Original .safetensors (425MB): \"{orig_text}\"  ({t_orig:.2f}s)")
        print(f"  Direktni .lrflac     (117MB): \"{zd_text}\"  ({t_zd:.2f}s | Logits CosSim = {cos:.6f})")

    print("\n" + "=" * 108)
    if "--chat" in sys.argv:
        print("INTERAKTIVNI .LRFLAC CHAT REŽIM (unesite 'izlaz' ili 'exit' za kraj):")
        print("Primeri: 'translate English to German: Hello, how are you today?' ili 'summarize: <tekst>'")
        while True:
            try:
                user_in = input("\nVi (.lrflac prompt) > ").strip()
            except EOFError:
                break
            if not user_in or user_in.lower() in ("exit", "quit", "izlaz"):
                break
            ids = sp.encode(user_in, out_type=int) + [1]
            inp = np.array([ids], dtype=np.int32)
            t0 = time.perf_counter()
            zd_toks, _ = generate_direct_from_lrflac(zd_weights, inp, max_new_tokens=48)
            dt = time.perf_counter() - t0
            print(f"Model (.lrflac Zero-Decode, {dt:.2f}s) > {sp.decode(zd_toks)}")


if __name__ == "__main__":
    main()
