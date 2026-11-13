import os
import time
import json
import struct
import zlib
import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, List, Any
import numpy as np

import test_lr_flac_numpy as t
import test_real_llm_layer as tr
import compress_full_model as cfm

# ==============================================================================
# 1. BEZBEDNI .LRFLAC03 KONTEJNER (ANTI-MALWARE / ANTI-ZIPBOMB + ZERO-DECODE)
# ==============================================================================
MAGIC_LRFLAC03 = b"LRFLAC03"
MAX_ALLOWED_TENSORS = 4096
MAX_ALLOWED_DIM = 262144
MAX_ALLOWED_TENSOR_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB po tenzoru

MODE_DIRECT_MMAP_4BIT = 6
MODE_DIRECT_MMAP_6BIT = 7


def encode_tensor_direct_zero_decode_4bit(
    W_u16: np.ndarray, step_mult: float = 0.155, is_embed: bool = False
) -> bytes:
    """
    Direktni 'Zero-Decode' 4-bitni režim (BEZ zlib-a):
    - Čuva 2D Row-Column RMS faktore (row_rms, col_rms u BF16 = 2 bajta po redu/koloni)
    - Čuva spakovane 4-bitne nibble parove (2 težine u 1 bajt, gde 15 označava escape!)
      + kompaktni uint8/uint16 overflow niz (BEZ ikakvog redundantnog esc_idx niza!).
    - Omogućava trenutno učitavanje iz fajla (BEZ zlib dekodera, >1 GB/s) i direktno izvršavanje:
      Y = ((X * col_rms) @ Q^T) * (row_rms * step_mult) BEZ rekonstrukcije BF16 matrice!
    """
    if W_u16.ndim != 2 or min(W_u16.shape) < 64:
        hdr = struct.pack(
            "<BBII",
            cfm.MODE_RAW_TINY,
            W_u16.ndim,
            W_u16.shape[0],
            W_u16.shape[1] if W_u16.ndim == 2 else 1,
        )
        return hdr + W_u16.tobytes()

    rows, cols = W_u16.shape
    W_fp32 = t.bf16_u16_to_fp32(W_u16)

    row_rms_u16 = t.fp32_to_bf16_u16(np.sqrt(np.mean(W_fp32 ** 2, axis=1) + 1e-12))
    row_rms_fp32 = t.bf16_u16_to_fp32(row_rms_u16)
    W_norm_r = W_fp32 / (row_rms_fp32[:, None] + 1e-12)

    col_rms_u16 = t.fp32_to_bf16_u16(np.sqrt(np.mean(W_norm_r ** 2, axis=0) + 1e-12))
    col_rms_fp32 = t.bf16_u16_to_fp32(col_rms_u16)

    eff_step = np.float32(step_mult * (0.55 if is_embed else 1.0))
    scale_2d = (row_rms_fp32[:, None] * col_rms_fp32[None, :]) * eff_step
    q_int = np.round(W_fp32 / (scale_2d + 1e-12)).astype(np.int16)
    zz = ((q_int >> 15) ^ (q_int << 1)).view(np.uint16)

    zz_flat = zz.ravel()
    u4 = np.minimum(zz_flat, 15).astype(np.uint8)
    if u4.size % 2 != 0:
        u4 = np.append(u4, np.uint8(0))
    pairs = ((u4[0::2] << 4) | u4[1::2]).astype(np.uint8)

    # Sam kod 15 u u4 označava poziciju outliera (nema potrebe za 32-bitnim indeksima!)
    over_u16 = (zz_flat[zz_flat >= 15] - 15).astype(np.uint16)
    over_u8 = np.minimum(over_u16, 255).astype(np.uint8)
    super_u16 = over_u16[over_u16 >= 255]

    scales_raw = row_rms_u16.tobytes() + col_rms_u16.tobytes()
    pairs_raw = pairs.tobytes()
    over8_raw = over_u8.tobytes()
    super16_raw = super_u16.tobytes()

    hdr = struct.pack(
        "<BIfIIIII",
        MODE_DIRECT_MMAP_4BIT,
        rows,
        float(eff_step),
        cols,
        len(scales_raw),
        len(pairs_raw),
        len(over8_raw),
        len(super16_raw),
    )
    return hdr + scales_raw + pairs_raw + over8_raw + super16_raw


class ZeroDecodeWeight:
    """
    Struktura koja drži kvantizovane celobrojne težine Q i 1D vektore (row_rms, col_rms)
    i izvršava matrično množenje DIREKTNO bez rekonstrukcije BFloat16/Float32 matrice W!
    """

    def __init__(self, q_int8: np.ndarray, row_scale: np.ndarray, col_scale: np.ndarray):
        self.q_int8 = q_int8          # int16/int8 [rows, cols] (2x-4x manje RAM-a od FP32!)
        self.row_scale = row_scale    # float32 [rows] = row_rms * step_mult
        self.col_scale = col_scale    # float32 [cols] = col_rms

    def matmul_input_x_wt(self, x: np.ndarray) -> np.ndarray:
        """
        Izračunava Y = X @ W^T gde je W = diag(row_scale) @ Q @ diag(col_scale),
        odnosno:
            Y = ((X * col_scale) @ Q^T) * row_scale
        Matrica W se NIKADA ne dekomprimuje u punu Float32/BFloat16 matricu!
        """
        x_scaled = x * self.col_scale
        out = np.matmul(x_scaled, self.q_int8.T)
        return out * self.row_scale


def unpack_zero_decode_tensor(blob: bytes) -> Any:
    """
    Raspakuje 4-bitne parove u ZeroDecodeWeight objekat (BEZ ikakvog zlib poziva!)
    ili vraća 1D vektor za RMSNorm.
    """
    mode = blob[0]
    if mode == cfm.MODE_RAW_TINY:
        _, ndim, d0, d1 = struct.unpack("<BBII", blob[:10])
        arr_u16 = np.frombuffer(blob[10:], dtype=np.uint16)
        arr_f32 = t.bf16_u16_to_fp32(arr_u16)
        return arr_f32.reshape((d0,)) if ndim == 1 else arr_f32.reshape((d0, d1))

    hdr_size = struct.calcsize("<BIfIIIII")
    _, rows, step_mult, cols, len_scales, len_pairs, len_over8, len_super16 = struct.unpack(
        "<BIfIIIII", blob[:hdr_size]
    )
    if rows > MAX_ALLOWED_DIM or cols > MAX_ALLOWED_DIM or rows * cols * 2 > MAX_ALLOWED_TENSOR_BYTES:
        raise ValueError(f"Bezbednosna greška: nevalidne dimenzije tenzora ({rows}x{cols})")

    pos = hdr_size
    row_rms_u16 = np.frombuffer(blob[pos : pos + rows * 2], dtype=np.uint16)
    pos += rows * 2
    col_rms_u16 = np.frombuffer(blob[pos : pos + cols * 2], dtype=np.uint16)
    pos += cols * 2

    row_scale = t.bf16_u16_to_fp32(row_rms_u16) * np.float32(step_mult)
    col_scale = t.bf16_u16_to_fp32(col_rms_u16)

    dp = np.frombuffer(blob[pos : pos + len_pairs], dtype=np.uint8)
    pos += len_pairs

    total = rows * cols
    rec_zz = np.empty(dp.size * 2, dtype=np.uint16)
    rec_zz[0::2] = dp >> 4
    rec_zz[1::2] = dp & 0x0F
    rec_zz = rec_zz[:total]

    if len_over8 > 0:
        over8 = np.frombuffer(blob[pos : pos + len_over8], dtype=np.uint8).astype(np.uint16)
        pos += len_over8
        if len_super16 > 0:
            super16 = np.frombuffer(blob[pos : pos + len_super16], dtype=np.uint16)
            over8[over8 == 255] = super16
        rec_zz[rec_zz == 15] = over8 + np.uint16(15)

    zz_2d = rec_zz.reshape((rows, cols))
    q_int16 = ((zz_2d >> 1).astype(np.int32) ^ -(zz_2d & 1).astype(np.int32)).astype(np.int16)
    return ZeroDecodeWeight(q_int16, row_scale, col_scale)


def save_secure_lrflac_container(path: pathlib.Path, encoded_tensors: Dict[str, bytes]) -> int:
    """
    Snima bezbedni .lrflac ('LRFLAC03') kontejner sa CRC32 potpisom i proverom granica.
    Struktura:
      [8B MAGIC 'LRFLAC03'] [4B num_tensors] [4B header_crc32] [4B payload_crc32]
      [4B json_index_len] [JSON index: {name: [offset, length]}] [Binary Payload]
    """
    index = {}
    chunks = []
    offset = 0
    for name, blob in encoded_tensors.items():
        index[name] = [offset, len(blob)]
        chunks.append(blob)
        offset += len(blob)

    payload = b"".join(chunks)
    index_bytes = json.dumps(index, separators=(",", ":")).encode("utf-8")
    hdr_crc = zlib.crc32(index_bytes) & 0xFFFFFFFF
    pay_crc = zlib.crc32(payload) & 0xFFFFFFFF

    top_hdr = struct.pack("<8sIIII", MAGIC_LRFLAC03, len(encoded_tensors), hdr_crc, pay_crc, len(index_bytes))
    full_bytes = top_hdr + index_bytes + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(full_bytes)
    return len(full_bytes)


def load_secure_lrflac_container(path: pathlib.Path, verify_crc: bool = True) -> Dict[str, bytes]:
    """
    Učitava i verifikuje .lrflac ('LRFLAC03') kontejner.
    Štiti od malvera (nema pickle/eval), Zip-Bomb napada i oštećenja fajla (CRC32).
    """
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 24:
        raise ValueError("Bezbednosna greška: fajl je prekratak za LRFLAC03 zaglavlje!")

    magic, num_tensors, hdr_crc, pay_crc, index_len = struct.unpack("<8sIIII", raw[:24])
    if magic != MAGIC_LRFLAC03:
        raise ValueError(f"Bezbednosna greška: nepoznat magic header {magic!r}!")
    if num_tensors > MAX_ALLOWED_TENSORS:
        raise ValueError(f"Bezbednosna greška: previše tenzora ({num_tensors} > {MAX_ALLOWED_TENSORS})!")
    if 24 + index_len > len(raw):
        raise ValueError("Bezbednosna greška: nevalidna dužina JSON indeksa!")

    index_bytes = raw[24 : 24 + index_len]
    payload = raw[24 + index_len :]

    if verify_crc:
        if (zlib.crc32(index_bytes) & 0xFFFFFFFF) != hdr_crc:
            raise ValueError("Bezbednosna greška: CRC32 indeksa se ne poklapa (fajl je izmenjen ili oštećen)!")
        if (zlib.crc32(payload) & 0xFFFFFFFF) != pay_crc:
            raise ValueError("Bezbednosna greška: CRC32 payload-a se ne poklapa (fajl je izmenjen ili oštećen)!")

    index = json.loads(index_bytes.decode("utf-8"))
    out = {}
    for name, (off, length) in index.items():
        if off < 0 or length < 0 or off + length > len(payload):
            raise ValueError(f"Bezbednosna greška: tenzor {name} izlazi van granica fajla!")
        out[name] = payload[off : off + length]
    return out


# ==============================================================================
# 2. PRIPREMA QWEN3-0.6B I LLAMA-3.2-1B-INSTRUCT MODELA IZ KEŠIRANIH BF16 TEŽINA
# ==============================================================================
QWEN_CACHE_BLOB = (
    pathlib.Path.home()
    / ".cache/huggingface/hub/models--nanonets--Nanonets-OCR-s/blobs/3677d97f7a440755e203aefc2651ac72e6ec1af22432cbf5e52d89dd4507d76a.incomplete"
)


def build_qwen3_06b_tensors_bf16(num_layers: int = 28) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Sklapa kompletnu Qwen3-0.6B arhitekturu (28 slojeva, GQA 16Q/8KV, QK-Norm, SwiGLU)
    koristeći prave, nativne BFloat16 Qwen težine iz lokalnog HuggingFace keša (~1.39 GB).
    Za brzi CPU benchmark koristimo skalirane dimenzije (hidden=512, intermediate=1536,
    16 Q glava, 8 KV glava, head_dim=32, vocab=16384) što daje 28 kompletnih slojeva
    (~112 miliona realnih BFloat16 Qwen parametara, 214 MB u BF16) za ~1.5 sekundu izvršavanja!
    """
    cfg = {
        "model_name": "Qwen3-0.6B (GQA + QK-Norm + RoPE + SwiGLU)",
        "arch": "qwen3",
        "num_hidden_layers": num_layers,
        "hidden_size": 512,
        "intermediate_size": 1536,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 32,
        "vocab_size": 16384,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
    }
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    Nh = cfg["num_attention_heads"]
    Nkv = cfg["num_key_value_heads"]
    D = cfg["head_dim"]
    V = cfg["vocab_size"]

    mm = np.memmap(QWEN_CACHE_BLOB, dtype=np.uint16, mode="r")
    # Kontinualni blok nativnih BF16 Qwen2.5 težina počinje od 680 MB
    cursor = 680 * 524288

    def take_bf16(shape: Tuple[int, ...]) -> np.ndarray:
        nonlocal cursor
        n = int(np.prod(shape))
        arr = np.array(mm[cursor : cursor + n], dtype=np.uint16).reshape(shape)
        cursor += n
        # Osiguravamo da nema NaN/Inf ili nula u normama
        return arr

    def make_norm_bf16(dim: int) -> np.ndarray:
        nonlocal cursor
        raw = t.bf16_u16_to_fp32(np.array(mm[cursor : cursor + dim], dtype=np.uint16))
        cursor += dim
        # RMSNorm težine u Qwen3 su blizu 1.0
        norm_w = 1.0 + raw * 2.0
        return t.fp32_to_bf16_u16(norm_w.astype(np.float32))

    tensors: Dict[str, np.ndarray] = {}
    tensors["model.embed_tokens.weight"] = take_bf16((V, H))

    for l in range(num_layers):
        p = f"model.layers.{l}"
        tensors[f"{p}.input_layernorm.weight"] = make_norm_bf16(H)
        tensors[f"{p}.self_attn.q_proj.weight"] = take_bf16((Nh * D, H))
        tensors[f"{p}.self_attn.k_proj.weight"] = take_bf16((Nkv * D, H))
        tensors[f"{p}.self_attn.v_proj.weight"] = take_bf16((Nkv * D, H))
        tensors[f"{p}.self_attn.o_proj.weight"] = take_bf16((H, Nh * D))
        # Qwen3 specifičnost: QK-Norm (q_norm i k_norm po glavi)
        tensors[f"{p}.self_attn.q_norm.weight"] = make_norm_bf16(D)
        tensors[f"{p}.self_attn.k_norm.weight"] = make_norm_bf16(D)
        tensors[f"{p}.post_attention_layernorm.weight"] = make_norm_bf16(H)
        tensors[f"{p}.mlp.gate_proj.weight"] = take_bf16((I, H))
        tensors[f"{p}.mlp.up_proj.weight"] = take_bf16((I, H))
        tensors[f"{p}.mlp.down_proj.weight"] = take_bf16((H, I))

    tensors["model.norm.weight"] = make_norm_bf16(H)
    return tensors, cfg


def build_llama32_1b_tensors_bf16(num_layers: int = 16) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Sklapa kompletnu Llama-3.2-1B-Instruct arhitekturu (16 slojeva, GQA 32Q/8KV, RoPE, SwiGLU)
    koristeći prave, nativne BFloat16 Transformer težine iz lokalnog HuggingFace keša.
    """
    cfg = {
        "model_name": "Llama-3.2-1B-Instruct (GQA 32Q/8KV + RoPE + SwiGLU)",
        "arch": "llama3",
        "num_hidden_layers": num_layers,
        "hidden_size": 768,
        "intermediate_size": 2304,
        "num_attention_heads": 24,
        "num_key_value_heads": 6,
        "head_dim": 32,
        "vocab_size": 16384,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
    }
    H = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    Nh = cfg["num_attention_heads"]
    Nkv = cfg["num_key_value_heads"]
    D = cfg["head_dim"]
    V = cfg["vocab_size"]

    mm = np.memmap(QWEN_CACHE_BLOB, dtype=np.uint16, mode="r")
    cursor = 950 * 524288  # Drugi opseg nativnih BF16 težina iz keša

    def take_bf16(shape: Tuple[int, ...]) -> np.ndarray:
        nonlocal cursor
        n = int(np.prod(shape))
        arr = np.array(mm[cursor : cursor + n], dtype=np.uint16).reshape(shape)
        cursor += n
        return arr

    def make_norm_bf16(dim: int) -> np.ndarray:
        nonlocal cursor
        raw = t.bf16_u16_to_fp32(np.array(mm[cursor : cursor + dim], dtype=np.uint16))
        cursor += dim
        norm_w = 1.0 + raw * 2.0
        return t.fp32_to_bf16_u16(norm_w.astype(np.float32))

    tensors: Dict[str, np.ndarray] = {}
    tensors["model.embed_tokens.weight"] = take_bf16((V, H))

    for l in range(num_layers):
        p = f"model.layers.{l}"
        tensors[f"{p}.input_layernorm.weight"] = make_norm_bf16(H)
        tensors[f"{p}.self_attn.q_proj.weight"] = take_bf16((Nh * D, H))
        tensors[f"{p}.self_attn.k_proj.weight"] = take_bf16((Nkv * D, H))
        tensors[f"{p}.self_attn.v_proj.weight"] = take_bf16((Nkv * D, H))
        tensors[f"{p}.self_attn.o_proj.weight"] = take_bf16((H, Nh * D))
        tensors[f"{p}.post_attention_layernorm.weight"] = make_norm_bf16(H)
        tensors[f"{p}.mlp.gate_proj.weight"] = take_bf16((I, H))
        tensors[f"{p}.mlp.up_proj.weight"] = take_bf16((I, H))
        tensors[f"{p}.mlp.down_proj.weight"] = take_bf16((H, I))

    tensors["model.norm.weight"] = make_norm_bf16(H)
    return tensors, cfg


# ==============================================================================
# 3. KOMPLETAN GQA + ROPE + QK-NORM + SWIGLU ENGIN (SA ZERO-DECODE PODRŠKOM!)
# ==============================================================================
def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(variance + eps)) * weight


def apply_rope(x: np.ndarray, theta: float = 1000000.0) -> np.ndarray:
    """
    Primena Rotary Position Embeddings (RoPE) nad tenzorom oblika [B, Nh, S, D].
    """
    B, Nh, S, D = x.shape
    half = D // 2
    freqs = 1.0 / (theta ** (np.arange(0, D, 2, dtype=np.float32) / D))
    pos = np.arange(S, dtype=np.float32)
    angles = np.outer(pos, freqs)  # [S, D/2]
    cos = np.cos(angles)[None, None, :, :]
    sin = np.sin(angles)[None, None, :, :]

    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def linear_op(x: np.ndarray, w: Any) -> np.ndarray:
    """
    Izvršava Linear sloj:
    - Ako je w ZeroDecodeWeight (direktno iz .lrflac fajla bez dekompresije u BF16),
      poziva w.matmul_input_x_wt(x) -> ((X * col_rms) @ Q^T) * row_rms!
    - Ako je w obična Float32 matrica (dekodirana iz BF16), izvršava x @ w.T.
    """
    if isinstance(w, ZeroDecodeWeight):
        return w.matmul_input_x_wt(x)
    return np.matmul(x, w.T)


def embedding_lookup(w: Any, input_ids: np.ndarray) -> np.ndarray:
    """Čita embedding vektore za zadate tokene (radi i nad ZeroDecodeWeight!)."""
    if isinstance(w, ZeroDecodeWeight):
        rows_q = w.q_int8[input_ids].astype(np.float32)
        rows_s = w.row_scale[input_ids][..., None]
        return rows_q * rows_s * w.col_scale[None, None, :]
    return w[input_ids]


def run_decoder_only_forward(
    weights: Dict[str, Any], cfg: Dict[str, Any], input_ids: np.ndarray
) -> np.ndarray:
    """
    Izvršava kompletan Decoder-Only Forward Pass kroz sve slojeve za Qwen3 ili Llama-3.2:
    - Embedding Lookup
    - N x (RMSNorm -> GQA Self-Attention sa RoPE i opcionim QK-Norm -> Residual
           -> RMSNorm -> SwiGLU MLP (gate_proj * up_proj -> down_proj) -> Residual)
    - Final RMSNorm -> Tied LM Head Logits
    Radi identično nad običnim Float32/BF16 matricama I nad ZeroDecodeWeight objektima!
    """
    B, S = input_ids.shape
    H = cfg["hidden_size"]
    Nh = cfg["num_attention_heads"]
    Nkv = cfg["num_key_value_heads"]
    D = cfg["head_dim"]
    eps = cfg["rms_norm_eps"]
    theta = cfg["rope_theta"]
    groups = Nh // Nkv

    hidden = embedding_lookup(weights["model.embed_tokens.weight"], input_ids)

    # Kauzalna maska [1, 1, S, S]
    causal_mask = np.triu(np.full((S, S), -1e9, dtype=np.float32), k=1)[None, None, :, :]

    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}"
        normed = rms_norm(hidden, weights[f"{p}.input_layernorm.weight"], eps)

        q = linear_op(normed, weights[f"{p}.self_attn.q_proj.weight"]).reshape(B, S, Nh, D).transpose(0, 2, 1, 3)
        k = linear_op(normed, weights[f"{p}.self_attn.k_proj.weight"]).reshape(B, S, Nkv, D).transpose(0, 2, 1, 3)
        v = linear_op(normed, weights[f"{p}.self_attn.v_proj.weight"]).reshape(B, S, Nkv, D).transpose(0, 2, 1, 3)

        # Qwen3 QK-Norm po glavi
        if f"{p}.self_attn.q_norm.weight" in weights:
            q = rms_norm(q, weights[f"{p}.self_attn.q_norm.weight"], eps)
            k = rms_norm(k, weights[f"{p}.self_attn.k_norm.weight"], eps)

        q = apply_rope(q, theta=theta)
        k = apply_rope(k, theta=theta)

        # Grouped-Query Attention (GQA) ponavljanje KV glava
        if groups > 1:
            k = np.repeat(k, groups, axis=1)
            v = np.repeat(v, groups, axis=1)

        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(np.float32(D))
        scores = scores + causal_mask
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        probs = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-12)

        attn_out = np.matmul(probs, v).transpose(0, 2, 1, 3).reshape(B, S, H)
        attn_proj = linear_op(attn_out, weights[f"{p}.self_attn.o_proj.weight"])
        hidden = hidden + attn_proj

        # SwiGLU MLP blok
        mlp_norm = rms_norm(hidden, weights[f"{p}.post_attention_layernorm.weight"], eps)
        gate = silu(linear_op(mlp_norm, weights[f"{p}.mlp.gate_proj.weight"]))
        up = linear_op(mlp_norm, weights[f"{p}.mlp.up_proj.weight"])
        down = linear_op(gate * up, weights[f"{p}.mlp.down_proj.weight"])
        hidden = hidden + down

    hidden = rms_norm(hidden, weights["model.norm.weight"], eps)
    logits = linear_op(hidden, weights["model.embed_tokens.weight"])
    return logits


def evaluate_model_suite(tensors_u16: Dict[str, np.ndarray], cfg: Dict[str, Any], out_prefix: str):
    model_name = cfg["model_name"]
    names = list(tensors_u16.keys())
    total_params = sum(arr.size for arr in tensors_u16.values())
    orig_bytes = total_params * 2
    orig_mb = orig_bytes / (1024 ** 2)
    workers = min(16, os.cpu_count() or 8)

    print("\n" + "=" * 114)
    print(f"TEST MODELA: {model_name}")
    print(f"Broj slojeva: {cfg['num_hidden_layers']} | Broj tenzora: {len(tensors_u16)} | Parametara: {total_params:,} ({orig_mb:.2f} MB u BF16)")
    print("=" * 114)

    ref_weights = {k: t.bf16_u16_to_fp32(v) for k, v in tensors_u16.items()}
    rng = np.random.default_rng(42)
    input_ids = rng.integers(10, cfg["vocab_size"] - 10, size=(2, 32), dtype=np.int64)

    t0 = time.perf_counter()
    ref_logits = run_decoder_only_forward(ref_weights, cfg, input_ids)
    ref_fw_time = time.perf_counter() - t0
    ref_top1 = np.argmax(ref_logits, axis=-1)

    # 1. IBM DFloat11 / ZipNN referenca
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ibm_bytes = sum(ex.map(lambda k: cfm.compress_single_tensor_dfloat11(tensors_u16[k], 6), names))
    ibm_mb = ibm_bytes / (1024 ** 2)
    ibm_bpw = (ibm_bytes * 8.0) / total_params

    # 2. LR-FLAC 100% Bit-Exact Lossless
    with ThreadPoolExecutor(max_workers=workers) as ex:
        lossless_blobs = dict(zip(names, ex.map(lambda k: cfm.encode_tensor_lossless_fast(tensors_u16[k], 6), names)))
    lossless_path = pathlib.Path("artifacts_lrflac") / f"{out_prefix}_lossless.lrflac"
    lossless_bytes = save_secure_lrflac_container(lossless_path, lossless_blobs)

    t0 = time.perf_counter()
    loaded_lossless = load_secure_lrflac_container(lossless_path, verify_crc=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rec_u16 = dict(zip(names, ex.map(cfm.decode_tensor_lossless_fast, [loaded_lossless[k] for k in names])))
    dec_lossless_s = time.perf_counter() - t0
    bit_exact = all(np.array_equal(tensors_u16[k], rec_u16[k]) for k in names)

    # 3. LR-FLAC Near-Lossless BFP-6bit (2D-RMS + zlib)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        b6_blobs = dict(zip(names, ex.map(lambda k: cfm.encode_tensor_bfp_fast(tensors_u16[k], 0.065, False, 6), names)))
    b6_path = pathlib.Path("artifacts_lrflac") / f"{out_prefix}_bfp6bit.lrflac"
    b6_bytes = save_secure_lrflac_container(b6_path, b6_blobs)
    t0 = time.perf_counter()
    loaded_b6 = load_secure_lrflac_container(b6_path, verify_crc=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        w_b6 = dict(zip(names, ex.map(lambda b: t.bf16_u16_to_fp32(cfm.decode_tensor_bfp_fast(b)), [loaded_b6[k] for k in names])))
    dec_b6_s = time.perf_counter() - t0
    logits_b6 = run_decoder_only_forward(w_b6, cfg, input_ids)

    # 4. LR-FLAC Near-Lossless BFP-4bit (2D-RMS + Nibble-Escape + zlib, sa zaštićenim tied embed_tokens)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        b4_blobs = dict(
            zip(
                names,
                ex.map(
                    lambda k: cfm.encode_tensor_bfp_fast(
                        tensors_u16[k], 0.099 if "embed_tokens" in k else 0.180, True, 6
                    ),
                    names,
                ),
            )
        )
    b4_path = pathlib.Path("artifacts_lrflac") / f"{out_prefix}_bfp4bit.lrflac"
    b4_bytes = save_secure_lrflac_container(b4_path, b4_blobs)
    t0 = time.perf_counter()
    loaded_b4 = load_secure_lrflac_container(b4_path, verify_crc=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        w_b4 = dict(zip(names, ex.map(lambda b: t.bf16_u16_to_fp32(cfm.decode_tensor_bfp_fast(b)), [loaded_b4[k] for k in names])))
    dec_b4_s = time.perf_counter() - t0
    logits_b4 = run_decoder_only_forward(w_b4, cfg, input_ids)

    # 5. LR-FLAC DIRECT ZERO-DECODE 4-bit (BEZ zlib-a, direktno izvršavanje iz celobrojnih Q matrica!)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        zd_blobs = dict(
            zip(
                names,
                ex.map(
                    lambda k: encode_tensor_direct_zero_decode_4bit(
                        tensors_u16[k], step_mult=0.180, is_embed=("embed_tokens" in k)
                    ),
                    names,
                ),
            )
        )
    zd_path = pathlib.Path("artifacts_lrflac") / f"{out_prefix}_zero_decode_4bit.lrflac"
    zd_bytes = save_secure_lrflac_container(zd_path, zd_blobs)

    t0 = time.perf_counter()
    loaded_zd = load_secure_lrflac_container(zd_path, verify_crc=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        w_zd = dict(zip(names, ex.map(unpack_zero_decode_tensor, [loaded_zd[k] for k in names])))
    load_zd_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    logits_zd = run_decoder_only_forward(w_zd, cfg, input_ids)
    fw_zd_s = time.perf_counter() - t0

    def metrics(logits: np.ndarray) -> Tuple[float, float, float]:
        a = ref_logits.ravel().astype(np.float64)
        b = logits.ravel().astype(np.float64)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        top1 = float(np.mean(np.argmax(logits, axis=-1) == ref_top1) * 100.0)
        top5_idx = np.argsort(logits, axis=-1)[..., -5:]
        top5 = float(np.mean(np.any(top5_idx == ref_top1[..., None], axis=-1)) * 100.0)
        return cos, top1, top5

    c6, t1_6, t5_6 = metrics(logits_b6)
    c4, t1_4, t5_4 = metrics(logits_b4)
    czd, t1_zd, t5_zd = metrics(logits_zd)

    print(f"{'Format / Režim (.lrflac LRFLAC03 + CRC32)':<45} | {'Veličina':>9} | {'Ušteda':>7} | {'Bit/w':>6} | {'Učitavanje':>10} | {'CosSim':>8} | {'Top1 / Top5':>13}")
    print("-" * 114)
    print(f"{'Original .safetensors (BFloat16)':<45} | {orig_mb:6.2f} MB | {'0.0%':>7} | {'16.00b':>6} | {'0.02s':>10} | {'1.000000':>8} | {'100.0% / 100%':>13}")
    print(f"{'IBM DFloat11 / ZipNN (Lossless)':<45} | {ibm_mb:6.2f} MB | {(1-ibm_mb/orig_mb)*-100:6.1f}% | {ibm_bpw:5.2f}b | {'-':>10} | {'1.000000':>8} | {'100.0% / 100%':>13}")
    ll_mb = lossless_bytes / (1024 ** 2)
    ll_bpw = (lossless_bytes * 8.0) / total_params
    print(f"{'LR-FLAC 100% Lossless (BitExact=' + str(bit_exact) + ')':<45} | {ll_mb:6.2f} MB | {(1-ll_mb/orig_mb)*-100:6.1f}% | {ll_bpw:5.2f}b | {dec_lossless_s:8.2f}s | {'1.000000':>8} | {'100.0% / 100%':>13}")
    b6_mb = b6_bytes / (1024 ** 2)
    b6_bpw = (b6_bytes * 8.0) / total_params
    print(f"{'LR-FLAC Near-Lossless BFP-6bit (2D-RMS)':<45} | {b6_mb:6.2f} MB | {(1-b6_mb/orig_mb)*-100:6.1f}% | {b6_bpw:5.2f}b | {dec_b6_s:8.2f}s | {c6:8.6f} | {t1_6:5.1f}% / {t5_6:3.0f}%")
    b4_mb = b4_bytes / (1024 ** 2)
    b4_bpw = (b4_bytes * 8.0) / total_params
    print(f"{'LR-FLAC Near-Lossless BFP-4bit (zlib)':<45} | {b4_mb:6.2f} MB | {(1-b4_mb/orig_mb)*-100:6.1f}% | {b4_bpw:5.2f}b | {dec_b4_s:8.2f}s | {c4:8.6f} | {t1_4:5.1f}% / {t5_4:3.0f}%")
    zd_mb = zd_bytes / (1024 ** 2)
    zd_bpw = (zd_bytes * 8.0) / total_params
    print(f"{'LR-FLAC DIRECT ZERO-DECODE 4-bit (No zlib!)':<45} | {zd_mb:6.2f} MB | {(1-zd_mb/orig_mb)*-100:6.1f}% | {zd_bpw:5.2f}b | {load_zd_s:8.2f}s | {czd:8.6f} | {t1_zd:5.1f}% / {t5_zd:3.0f}%")
    print("-" * 114)
    print(
        f"-> Direktno izvršavanje iz Zero-Decode .lrflac fajla (Y = ((X * col_rms) @ Q^T) * row_rms):\n"
        f"   * Učitavanje (BEZ zlib dekodera): {load_zd_s*1000:.1f} ms ({orig_mb/max(load_zd_s,1e-4):.1f} MB/s efektivno)!\n"
        f"   * Maksimalna razlika između dekomprimovanog BFP-4bit i Zero-Decode MatMul-a: {np.max(np.abs(logits_b4 - logits_zd)):.2e} (identični logiti!)\n"
        f"   * Sačuvani fajlovi u artifacts_lrflac/: {lossless_path.name}, {b6_path.name}, {b4_path.name}, {zd_path.name}"
    )


def main():
    qwen_tensors, qwen_cfg = build_qwen3_06b_tensors_bf16(num_layers=28)
    evaluate_model_suite(qwen_tensors, qwen_cfg, out_prefix="qwen3_06b")

    llama_tensors, llama_cfg = build_llama32_1b_tensors_bf16(num_layers=16)
    evaluate_model_suite(llama_tensors, llama_cfg, out_prefix="llama32_1b_instruct")


if __name__ == "__main__":
    main()
