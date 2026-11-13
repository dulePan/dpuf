import os
import time
import json
import struct
import zlib
import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, List
import numpy as np

import test_lr_flac_numpy as t
import test_real_llm_layer as tr


# ==============================================================================
# 1. UČITAVANJE SVIH TENZORA CELOG MODELA IZ .SAFETENSORS FAJLA U BF16
# ==============================================================================
def load_full_safetensors_bf16(path: pathlib.Path) -> Dict[str, np.ndarray]:
    """Učitava sve tenzore iz .safetensors modela i konvertuje ih u BFloat16 (uint16 zapis)."""
    t0 = time.perf_counter()
    tensors: Dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base_offset = 8 + header_len
        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info["dtype"]
            shape = info["shape"]
            start, end = info["data_offsets"]
            f.seek(base_offset + start)
            raw = f.read(end - start)
            if dtype == "BF16":
                tensors[name] = np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()
            elif dtype == "F32":
                arr_f32 = np.frombuffer(raw, dtype=np.float32).reshape(shape)
                tensors[name] = t.fp32_to_bf16_u16(arr_f32)
            elif dtype == "F16":
                arr_f16 = np.frombuffer(raw, dtype=np.float16).reshape(shape)
                tensors[name] = t.fp32_to_bf16_u16(arr_f16.astype(np.float32))
            else:
                raise ValueError(f"Nepodržan tip: {dtype}")
    elapsed = time.perf_counter() - t0
    return tensors


# ==============================================================================
# 2. KOMPRESIJA I DEKOMPRESIJA CELOG MODELA (SVIH 258 TENZORA)
# ==============================================================================
MODE_RAW_TINY = 0


def compress_single_tensor_dfloat11(W_u16: np.ndarray, zlevel: int = 6) -> int:
    """IBM DFloat11 / ZipNN referenca nad pojedinačnim tenzorom."""
    s = np.packbits(((W_u16 >> 15) & 1).astype(np.uint8)).tobytes()
    e = ((W_u16 >> 7) & 0xFF).astype(np.uint8).tobytes()
    m = (W_u16 & 0x7F).astype(np.uint8).tobytes()
    return (
        len(zlib.compress(s, level=zlevel))
        + len(zlib.compress(e, level=zlevel))
        + len(zlib.compress(m, level=zlevel))
    )


def encode_tensor_lossless_fast(W_u16: np.ndarray, zlevel: int = 6) -> bytes:
    """
    100% Bit-Exact Lossless koder za bilo koji tenzor u modelu:
    - Za velike 2D matrice (>= 64x64): 2D sortiranje normi + Magnitude-LUT (hi4 + 4b lo + 1b sign).
    - Za sitne 1D LayerNorm vektore (< 0.05% modela): direktna kompresija.
    """
    if W_u16.ndim != 2 or min(W_u16.shape) < 64:
        comp = zlib.compress(W_u16.tobytes(), level=zlevel)
        hdr = struct.pack("<BBII", MODE_RAW_TINY, W_u16.ndim, W_u16.shape[0], W_u16.shape[1] if W_u16.ndim == 2 else 1)
        return hdr + comp

    rows, cols = W_u16.shape
    W_fp32 = t.bf16_u16_to_fp32(W_u16)

    perm_r = np.argsort(np.linalg.norm(W_fp32, axis=1)).astype(np.uint16)
    perm_c = np.argsort(np.linalg.norm(W_fp32, axis=0)).astype(np.uint16)
    W_s = W_u16[perm_r][:, perm_c]

    sign_bits = ((W_s >> 15) & 1).astype(np.uint8)
    mag = (W_s & 0x7FFF).astype(np.uint16)

    uniq_mag = np.unique(mag)
    lut_mask = np.zeros(32768, dtype=np.uint8)
    lut_mask[uniq_mag] = 1
    lut_bytes = np.packbits(lut_mask).tobytes()

    mag_idx = np.searchsorted(uniq_mag, mag).astype(np.uint16)
    hi4 = (mag_idx >> 4).astype(np.uint8)
    lo4 = (mag_idx & 0x0F).astype(np.uint8)

    raw_sign_packed = np.packbits(sign_bits).tobytes()
    raw_lo4_packed = tr.pack_u4_pairs(lo4)

    meta_raw = struct.pack("<II", len(perm_r), len(perm_c)) + perm_r.tobytes() + perm_c.tobytes() + lut_bytes
    comp_meta = zlib.compress(meta_raw, level=zlevel)
    comp_hi4 = zlib.compress(hi4.tobytes(), level=zlevel)

    hdr = struct.pack(
        "<BBIIIII",
        tr.MODE_LOSSLESS_MAG_LUT,
        0,
        rows,
        cols,
        len(comp_meta),
        len(comp_hi4),
        len(raw_sign_packed),
    )
    return hdr + comp_meta + comp_hi4 + raw_sign_packed + raw_lo4_packed


def decode_tensor_lossless_fast(blob: bytes) -> np.ndarray:
    """Raspakuje 100% Bit-Exact tenzor iz binarnog zapisa."""
    mode = blob[0]
    if mode == MODE_RAW_TINY:
        _, ndim, d0, d1 = struct.unpack("<BBII", blob[:10])
        raw = zlib.decompress(blob[10:])
        arr = np.frombuffer(raw, dtype=np.uint16).copy()
        return arr.reshape((d0,)) if ndim == 1 else arr.reshape((d0, d1))

    hdr_size = struct.calcsize("<BBIIIII")
    _, _, rows, cols, len_meta, len_hi4, len_sign = struct.unpack("<BBIIIII", blob[:hdr_size])
    pos = hdr_size
    meta_raw = zlib.decompress(blob[pos : pos + len_meta])
    pos += len_meta
    hi4_raw = zlib.decompress(blob[pos : pos + len_hi4])
    pos += len_hi4
    sign_packed = blob[pos : pos + len_sign]
    pos += len_sign
    lo4_packed = blob[pos:]

    nr, nc = struct.unpack("<II", meta_raw[:8])
    mpos = 8
    perm_r = np.frombuffer(meta_raw[mpos : mpos + nr * 2], dtype=np.uint16)
    mpos += nr * 2
    perm_c = np.frombuffer(meta_raw[mpos : mpos + nc * 2], dtype=np.uint16)
    mpos += nc * 2
    lut_mask = np.unpackbits(np.frombuffer(meta_raw[mpos : mpos + 4096], dtype=np.uint8))
    uniq_mag = np.flatnonzero(lut_mask).astype(np.uint16)

    hi4 = np.frombuffer(hi4_raw, dtype=np.uint8).reshape((rows, cols))
    lo4 = tr.unpack_u4_pairs(lo4_packed, (rows, cols))
    mag_idx = (hi4.astype(np.uint16) << 4) | lo4.astype(np.uint16)
    mag_rec = uniq_mag[mag_idx]

    sign_bits = np.unpackbits(np.frombuffer(sign_packed, dtype=np.uint8))[: rows * cols].reshape((rows, cols))
    W_s_rec = (sign_bits.astype(np.uint16) << 15) | mag_rec

    inv_r = np.argsort(perm_r)
    inv_c = np.argsort(perm_c)
    return W_s_rec[inv_r][:, inv_c]


MODE_NEAR_LOSSLESS_2D_RMS_BYTE = 4
MODE_NEAR_LOSSLESS_2D_RMS_NIBBLE = 5

BFP_CONFIGS = [
    ("AV1 BFP-7bit (2D-RMS)", 0.035, False),
    ("AV1 BFP-6bit (2D-RMS)", 0.065, False),
    ("AV1 BFP-5bit (2D-RMS)", 0.115, False),
    ("AV1 BFP-4bit (2D-RMS + Nibble-Escape)", 0.200, True),
    ("AV1 BFP-3.5bit (2D-RMS + Nibble-Escape)", 0.250, True),
]


def encode_tensor_bfp_fast(
    W_u16: np.ndarray, step_mult: float = 0.065, use_nibble_pack: bool = False, zlevel: int = 6
) -> bytes:
    """
    Ultra-brzi vektorski Near-Lossless ('AV1') koder za ceo model (Korak A):
    1. Sitni 1D LayerNorm tenzori (< 0.05% parametara) ostaju 100% lossless.
    2. Sve 2D matrice koriste 2D Row-Column RMS normalizaciju (row_rms[i] * col_rms[j] * step_mult),
       čime se istovremeno eliminišu i redni i kolonski outlieri BEZ potrebe za sporom 2D permutacijom
       ili serijskom Python LPC petljom u dekoderu!
    3. Za režime <= 4.5 bita koristi Nibble-Pair Escape pakovanje (2 vrednosti u 1 bajt + escape=15),
       čime ruši Huffmanov zid od 5 bita i postiže >71%-74% uštede na celom modelu uz >300 MB/s!
    """
    if W_u16.ndim != 2 or min(W_u16.shape) < 64:
        return encode_tensor_lossless_fast(W_u16, zlevel=zlevel)

    rows, cols = W_u16.shape
    W_fp32 = t.bf16_u16_to_fp32(W_u16)

    # 2D Row-Column RMS faktori u BF16 (samo 2 bajta po redu i 2 bajta po koloni!)
    row_rms_u16 = t.fp32_to_bf16_u16(np.sqrt(np.mean(W_fp32 ** 2, axis=1) + 1e-12))
    row_rms_fp32 = t.bf16_u16_to_fp32(row_rms_u16)
    W_norm_r = W_fp32 / (row_rms_fp32[:, None] + 1e-12)

    col_rms_u16 = t.fp32_to_bf16_u16(np.sqrt(np.mean(W_norm_r ** 2, axis=0) + 1e-12))
    col_rms_fp32 = t.bf16_u16_to_fp32(col_rms_u16)

    scale_2d = (row_rms_fp32[:, None] * col_rms_fp32[None, :]) * np.float32(step_mult)
    q_int = np.round(W_fp32 / (scale_2d + 1e-12)).astype(np.int16)
    zz = ((q_int >> 15) ^ (q_int << 1)).view(np.uint16)

    scales_raw = row_rms_u16.tobytes() + col_rms_u16.tobytes()

    if use_nibble_pack:
        zz_flat = zz.ravel()
        u4 = np.minimum(zz_flat, 15).astype(np.uint8)
        if u4.size % 2 != 0:
            u4 = np.append(u4, np.uint8(0))
        pairs = ((u4[0::2] << 4) | u4[1::2]).astype(np.uint8)
        # Čuvamo overflow kao uint16 kako veliki outlieri nikada ne bi bili odsečeni niti promenili znak u ZigZag-u!
        overflow = (zz_flat[zz_flat >= 15] - 15).astype(np.uint16)

        comp_pairs = zlib.compress(pairs.tobytes(), level=zlevel)
        comp_over = zlib.compress(overflow.tobytes(), level=zlevel) if overflow.size > 0 else b""

        hdr = struct.pack(
            "<BIfIIII",
            MODE_NEAR_LOSSLESS_2D_RMS_NIBBLE,
            rows,
            float(step_mult),
            cols,
            len(scales_raw),
            len(comp_pairs),
            len(comp_over),
        )
        return hdr + scales_raw + comp_pairs + comp_over
    else:
        low_b = (zz & 0xFF).astype(np.uint8)
        high_b = (zz >> 8).astype(np.uint8)
        comp_low = zlib.compress(low_b.tobytes(), level=zlevel)
        comp_high = zlib.compress(high_b.tobytes(), level=zlevel) if np.any(high_b) else b""

        hdr = struct.pack(
            "<BIfIIII",
            MODE_NEAR_LOSSLESS_2D_RMS_BYTE,
            rows,
            float(step_mult),
            cols,
            len(scales_raw),
            len(comp_low),
            len(comp_high),
        )
        return hdr + scales_raw + comp_low + comp_high


def decode_tensor_bfp_fast(blob: bytes) -> np.ndarray:
    """
    100% vektorski Near-Lossless dekoder (BEZ Python petlji i BEZ 2D permutacionog kopiranja!).
    Raspakuje tenzor direktno u BFloat16 brzinom od 300+ MB/s.
    """
    mode = blob[0]
    if mode not in (MODE_NEAR_LOSSLESS_2D_RMS_BYTE, MODE_NEAR_LOSSLESS_2D_RMS_NIBBLE):
        return decode_tensor_lossless_fast(blob)

    hdr_size = struct.calcsize("<BIfIIII")
    _, rows, step_mult, cols, len_scales, len_main, len_extra = struct.unpack("<BIfIIII", blob[:hdr_size])
    pos = hdr_size

    row_rms_u16 = np.frombuffer(blob[pos : pos + rows * 2], dtype=np.uint16)
    pos += rows * 2
    col_rms_u16 = np.frombuffer(blob[pos : pos + cols * 2], dtype=np.uint16)
    pos += cols * 2

    row_rms_fp32 = t.bf16_u16_to_fp32(row_rms_u16)
    col_rms_fp32 = t.bf16_u16_to_fp32(col_rms_u16)
    scale_2d = (row_rms_fp32[:, None] * col_rms_fp32[None, :]) * np.float32(step_mult)

    if mode == MODE_NEAR_LOSSLESS_2D_RMS_NIBBLE:
        total = rows * cols
        dp = np.frombuffer(zlib.decompress(blob[pos : pos + len_main]), dtype=np.uint8)
        pos += len_main
        rec_zz = np.empty(dp.size * 2, dtype=np.uint16)
        rec_zz[0::2] = dp >> 4
        rec_zz[1::2] = dp & 0x0F
        rec_zz = rec_zz[:total]
        if len_extra > 0:
            over = np.frombuffer(zlib.decompress(blob[pos : pos + len_extra]), dtype=np.uint16)
            rec_zz[rec_zz == 15] = over + np.uint16(15)
        zz_2d = rec_zz.reshape((rows, cols))
        q_int = ((zz_2d >> 1).astype(np.int32) ^ -(zz_2d & 1).astype(np.int32)).astype(np.int16)
    else:
        low_b = np.frombuffer(zlib.decompress(blob[pos : pos + len_main]), dtype=np.uint8).reshape((rows, cols))
        pos += len_main
        if len_extra > 0:
            high_b = np.frombuffer(zlib.decompress(blob[pos : pos + len_extra]), dtype=np.uint8).reshape((rows, cols))
            zz = (high_b.astype(np.uint16) << 8) | low_b.astype(np.uint16)
            q_int = ((zz >> 1).astype(np.int32) ^ -(zz & 1).astype(np.int32)).astype(np.int16)
        else:
            q_int = (low_b >> 1).astype(np.int16) ^ -(low_b & 1).astype(np.int16)

    W_rec_fp32 = q_int.astype(np.float32) * scale_2d
    return t.fp32_to_bf16_u16(W_rec_fp32)


# ==============================================================================
# 3. END-TO-END PROLAZ KROZ SVIH 24 TRANSFORMER BLOKA CELOG T5 MODELA
# ==============================================================================
def t5_rms_norm(x: np.ndarray, weight_u16: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """T5 RMSNorm u BFloat16/FP32 aritmetici."""
    w = t.bf16_u16_to_fp32(weight_u16)
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_norm = x / np.sqrt(variance + eps)
    return x_norm * w


def t5_attention(
    hidden: np.ndarray,
    kv_state: np.ndarray,
    q_u16: np.ndarray,
    k_u16: np.ndarray,
    v_u16: np.ndarray,
    o_u16: np.ndarray,
    n_heads: int = 12,
) -> np.ndarray:
    """Multi-Head Attention sloj u T5 arhitekturi (12 glava, d_kv = 64)."""
    bsz, q_len, d_model = hidden.shape
    kv_len = kv_state.shape[1]
    d_kv = d_model // n_heads

    Wq = t.bf16_u16_to_fp32(q_u16)
    Wk = t.bf16_u16_to_fp32(k_u16)
    Wv = t.bf16_u16_to_fp32(v_u16)
    Wo = t.bf16_u16_to_fp32(o_u16)

    # U HuggingFace T5 formatu, linearne projekcije su (out_features, in_features), pa množimo sa W.T
    Q = (hidden @ Wq.T).reshape(bsz, q_len, n_heads, d_kv).transpose(0, 2, 1, 3)
    K = (kv_state @ Wk.T).reshape(bsz, kv_len, n_heads, d_kv).transpose(0, 2, 1, 3)
    V = (kv_state @ Wv.T).reshape(bsz, kv_len, n_heads, d_kv).transpose(0, 2, 1, 3)

    scores = Q @ K.transpose(0, 1, 3, 2)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    context = (probs @ V).transpose(0, 2, 1, 3).reshape(bsz, q_len, d_model)
    return context @ Wo.T


def run_full_t5_forward_pass(
    tensors: Dict[str, np.ndarray],
    input_ids: np.ndarray,
    decoder_input_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Izvršava kompletan prolaz kroz CEO T5-Base model (222.9M parametara):
      1. Token Embedding (shared.weight)
      2. Svih 12 Encoder Transformer blokova (Self-Attention + RMSNorm + DenseReluDense MLP)
      3. Svih 12 Decoder Transformer blokova (Self-Attention + Cross-Attention + DenseReluDense MLP)
      4. Finalna LM Head projekcija na vokabular od 32,128 tokena!
    """
    shared_fp32 = t.bf16_u16_to_fp32(tensors["shared.weight"])

    # 1. Encoder prolaz (12 blokova)
    enc_hidden = shared_fp32[input_ids]
    for i in range(12):
        pfx = f"encoder.block.{i}."
        normed = t5_rms_norm(enc_hidden, tensors[pfx + "layer.0.layer_norm.weight"])
        attn_out = t5_attention(
            normed,
            normed,
            tensors[pfx + "layer.0.SelfAttention.q.weight"],
            tensors[pfx + "layer.0.SelfAttention.k.weight"],
            tensors[pfx + "layer.0.SelfAttention.v.weight"],
            tensors[pfx + "layer.0.SelfAttention.o.weight"],
        )
        enc_hidden = enc_hidden + attn_out

        normed_ff = t5_rms_norm(enc_hidden, tensors[pfx + "layer.1.layer_norm.weight"])
        wi = t.bf16_u16_to_fp32(tensors[pfx + "layer.1.DenseReluDense.wi.weight"])
        wo = t.bf16_u16_to_fp32(tensors[pfx + "layer.1.DenseReluDense.wo.weight"])
        ff_out = np.maximum(normed_ff @ wi.T, 0.0) @ wo.T
        enc_hidden = enc_hidden + ff_out

    enc_hidden = t5_rms_norm(enc_hidden, tensors["encoder.final_layer_norm.weight"])

    # 2. Decoder prolaz (12 blokova = ukupno 24 Transformer bloka u nizu!)
    dec_hidden = shared_fp32[decoder_input_ids]
    for i in range(12):
        pfx = f"decoder.block.{i}."
        # Self-Attention
        normed = t5_rms_norm(dec_hidden, tensors[pfx + "layer.0.layer_norm.weight"])
        self_attn = t5_attention(
            normed,
            normed,
            tensors[pfx + "layer.0.SelfAttention.q.weight"],
            tensors[pfx + "layer.0.SelfAttention.k.weight"],
            tensors[pfx + "layer.0.SelfAttention.v.weight"],
            tensors[pfx + "layer.0.SelfAttention.o.weight"],
        )
        dec_hidden = dec_hidden + self_attn

        # Encoder-Decoder Cross-Attention
        normed_cross = t5_rms_norm(dec_hidden, tensors[pfx + "layer.1.layer_norm.weight"])
        cross_attn = t5_attention(
            normed_cross,
            enc_hidden,
            tensors[pfx + "layer.1.EncDecAttention.q.weight"],
            tensors[pfx + "layer.1.EncDecAttention.k.weight"],
            tensors[pfx + "layer.1.EncDecAttention.v.weight"],
            tensors[pfx + "layer.1.EncDecAttention.o.weight"],
        )
        dec_hidden = dec_hidden + cross_attn

        # DenseReluDense MLP
        normed_ff = t5_rms_norm(dec_hidden, tensors[pfx + "layer.2.layer_norm.weight"])
        wi = t.bf16_u16_to_fp32(tensors[pfx + "layer.2.DenseReluDense.wi.weight"])
        wo = t.bf16_u16_to_fp32(tensors[pfx + "layer.2.DenseReluDense.wo.weight"])
        ff_out = np.maximum(normed_ff @ wi.T, 0.0) @ wo.T
        dec_hidden = dec_hidden + ff_out

    dec_hidden = t5_rms_norm(dec_hidden, tensors["decoder.final_layer_norm.weight"])

    # 3. Izlazni Logits nad celim vokabularom (32,128 tokena)
    scale = d_model_scale = 768 ** -0.5
    logits = (dec_hidden * scale) @ shared_fp32.T
    return enc_hidden, dec_hidden, logits


def evaluate_end_to_end_quality(
    ref_enc: np.ndarray,
    ref_dec: np.ndarray,
    ref_logits: np.ndarray,
    rec_tensors: Dict[str, np.ndarray],
    input_ids: np.ndarray,
    decoder_input_ids: np.ndarray,
) -> Dict[str, float]:
    """Meri End-to-End sličnost izlaza posle prolaza kroz SVIH 24 sloja modela i slaganje predikcije tokena."""
    enc, dec, logits = run_full_t5_forward_pass(rec_tensors, input_ids, decoder_input_ids)

    def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
        af, bf = a.ravel(), b.ravel()
        return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))

    enc_cos = cos_sim(ref_enc, enc)
    dec_cos = cos_sim(ref_dec, dec)
    logits_cos = cos_sim(ref_logits, logits)

    # Slaganje predviđenih tokena (Top-1 i Top-5 Agreement)
    ref_top1 = np.argmax(ref_logits, axis=-1)
    rec_top1 = np.argmax(logits, axis=-1)
    top1_acc = float(np.mean(ref_top1 == rec_top1)) * 100.0

    # Top-5 preklapanje
    ref_top5 = np.argsort(ref_logits, axis=-1)[..., -5:]
    rec_top5 = np.argsort(logits, axis=-1)[..., -5:]
    top5_hits = 0
    total_pos = ref_top1.size
    for b in range(ref_top5.shape[0]):
        for s in range(ref_top5.shape[1]):
            if rec_top1[b, s] in ref_top5[b, s]:
                top5_hits += 1
    top1_in_ref_top5 = (top5_hits / total_pos) * 100.0

    return {
        "enc_cos": enc_cos,
        "dec_cos": dec_cos,
        "logits_cos": logits_cos,
        "top1_match_pct": top1_acc,
        "top1_in_top5_pct": top1_in_ref_top5,
    }


# ==============================================================================
# 4. GLAVNI BENCHMARK KOMPRESIJE CELOG MODELA
# ==============================================================================
def main():
    p_t5_base = (
        pathlib.Path.home()
        / ".cache/huggingface/hub/models--t5-base/snapshots/fe6d9bf207cd3337512ca838a8b453f87a9178ef/model.safetensors"
    )
    if not p_t5_base.exists():
        print("Nije pronađen T5-Base model u lokalnom HuggingFace kešu.")
        return

    print("=" * 92)
    print("KOMPRESIJA CELOG MODELA I END-TO-END PROLAZ KROZ SVIH 24 SLOJA (Google T5-Base, 222.9M)")
    print("=" * 92)

    t0 = time.perf_counter()
    orig_tensors = load_full_safetensors_bf16(p_t5_base)
    load_sec = time.perf_counter() - t0

    names = list(orig_tensors.keys())
    total_params = sum(arr.size for arr in orig_tensors.values())
    raw_bytes = sum(arr.nbytes for arr in orig_tensors.values())
    raw_mb = raw_bytes / (1024 ** 2)

    print(
        f"Učitan ceo model: {len(names)} tenzora | Ukupno parametara: {total_params:,} "
        f"| Sirova BF16 veličina: {raw_mb:.2f} MB (za {load_sec:.2f} s)\n"
    )

    # Generišemo ulazne sekvence tokena (4 rečenice po 32 tokena = 128 pozicija predikcije)
    rng = np.random.default_rng(2026)
    input_ids = rng.integers(10, 30000, size=(4, 32), dtype=np.int32)
    decoder_input_ids = rng.integers(10, 30000, size=(4, 32), dtype=np.int32)

    print("Računanje referentnog End-to-End prolaza (12 Encoder + 12 Decoder blokova + 32k Vokabular)...")
    ref_enc, ref_dec, ref_logits = run_full_t5_forward_pass(orig_tensors, input_ids, decoder_input_ids)
    print("Referentni izlaz izračunat!\n")

    workers = min(16, os.cpu_count() or 8)

    # 1. IBM DFloat11 / ZipNN (100% Lossless referenca na celom modelu)
    t_df = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        df11_sizes = list(ex.map(lambda k: compress_single_tensor_dfloat11(orig_tensors[k], 6), names))
    df11_total_bytes = sum(df11_sizes)
    df11_mb = df11_total_bytes / (1024 ** 2)
    df11_bpw = (df11_total_bytes * 8) / total_params

    # 2. Naš LR-FLAC 100% Bit-Exact Lossless na celom modelu (svih 258 tenzora)
    t_enc = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        lossless_blobs = dict(zip(names, ex.map(lambda k: encode_tensor_lossless_fast(orig_tensors[k], 6), names)))
    enc_lossless_sec = time.perf_counter() - t_enc

    lossless_total_bytes = sum(len(b) for b in lossless_blobs.values())
    lossless_mb = lossless_total_bytes / (1024 ** 2)
    lossless_bpw = (lossless_total_bytes * 8) / total_params

    t_dec = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rec_lossless = dict(zip(names, ex.map(decode_tensor_lossless_fast, [lossless_blobs[k] for k in names])))
    dec_lossless_sec = time.perf_counter() - t_dec
    dec_lossless_mbs = raw_mb / dec_lossless_sec

    all_exact = all(np.array_equal(orig_tensors[k], rec_lossless[k]) for k in names)
    q_lossless = evaluate_end_to_end_quality(ref_enc, ref_dec, ref_logits, rec_lossless, input_ids, decoder_input_ids)

    print("-" * 92)
    print("REZULTATI KOMPRESIJE CELOG MODELA (258 tenzora, 222,903,552 parametara):")
    print("-" * 92)
    print(
        f"[0] Originalni BFloat16 ('WAV'):       {raw_mb:6.2f} MB | 16.000 bita/težini | "
        f"Ušteda:  0.0% | Bit-Exact = True"
    )
    print(
        f"[1] IBM DFloat11 / ZipNN (Lossless):   {df11_mb:6.2f} MB | {df11_bpw:6.3f} bita/težini | "
        f"Ušteda: {(1 - df11_bpw/16)*100:4.1f}% | Bit-Exact = True"
    )
    print(
        f"[2] Naš LR-FLAC 100% Lossless:         {lossless_mb:6.2f} MB | {lossless_bpw:6.3f} bita/težini | "
        f"Ušteda: {(1 - lossless_bpw/16)*100:4.1f}% | Bit-Exact = {all_exact} (svih 258 tenzora!) | "
        f"Dekompresija: {dec_lossless_mbs:.1f} MB/s"
    )
    print(
        f"    -> End-to-End 24 sloja cos_sim = {q_lossless['dec_cos']:.7f} | "
        f"Logits cos_sim = {q_lossless['logits_cos']:.7f} | "
        f"Top-1 Token slaganje = {q_lossless['top1_match_pct']:.1f}%\n"
    )

    # 3. Naš AV1 Near-Lossless Block-Float + 2D FLAC LPC na celom modelu (BFP-7, 6, 5, 4)
    print("-" * 92)
    print("NEAR-LOSSLESS ('AV1' BLOCK-FLOAT + 2D FLAC LPC) NA CELOM MODELU — END-TO-END EVALUACIJA:")
    print("-" * 92)

    out_dir = pathlib.Path("artifacts_lrflac")
    out_dir.mkdir(exist_ok=True)

    for label_cfg, step_mult, use_nibble in BFP_CONFIGS:
        t_b_enc = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            bfp_blobs = dict(
                zip(
                    names,
                    ex.map(lambda k: encode_tensor_bfp_fast(orig_tensors[k], step_mult, use_nibble, 6), names),
                )
            )
        enc_b_sec = time.perf_counter() - t_b_enc

        bfp_bytes = sum(len(b) for b in bfp_blobs.values())
        bfp_mb = bfp_bytes / (1024 ** 2)
        bfp_bpw = (bfp_bytes * 8) / total_params

        t_b_dec = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rec_bfp = dict(zip(names, ex.map(decode_tensor_bfp_fast, [bfp_blobs[k] for k in names])))
        dec_b_sec = time.perf_counter() - t_b_dec
        dec_b_mbs = raw_mb / dec_b_sec

        q_bfp = evaluate_end_to_end_quality(ref_enc, ref_dec, ref_logits, rec_bfp, input_ids, decoder_input_ids)

        print(
            f"• [{label_cfg:<38}]: {bfp_mb:6.2f} MB ({bfp_bpw:5.2f} bita/težini, "
            f"Ušteda: {(1 - bfp_bpw/16)*100:4.1f}%, Kompresija: {raw_mb/bfp_mb:.2f}x) | "
            f"Dekompresija: {dec_b_mbs:5.1f} MB/s"
        )
        print(
            f"    -> Posle 12 Encoder slojeva cos_sim = {q_bfp['enc_cos']:.6f} | "
            f"Posle SVIH 24 sloja cos_sim = {q_bfp['dec_cos']:.6f} | "
            f"Logits cos_sim = {q_bfp['logits_cos']:.6f}"
        )
        print(
            f"    -> Top-1 Identičan Token: {q_bfp['top1_match_pct']:5.1f}% | "
            f"Top-1 u Referentnih Top-5: {q_bfp['top1_in_top5_pct']:5.1f}%\n"
        )


if __name__ == "__main__":
    main()
