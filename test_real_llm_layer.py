import os
import sys
import time
import json
import struct
import zlib
import lzma
import pathlib
from typing import Dict, List, Tuple, Optional
import numpy as np

import test_lr_flac_numpy as t


# ==============================================================================
# 1. ZERO-COPY ČITAČ .SAFETENSORS FAJLOVA (Direktan seek na pojedinačni sloj)
# ==============================================================================
def find_local_safetensors() -> List[Tuple[str, pathlib.Path]]:
    """Pronalazi lokalno keširane .safetensors modele (HuggingFace / Torch keš)."""
    home = pathlib.Path.home()
    candidates = [
        (
            "T5-Large (770M, Google Transformer)",
            home
            / ".cache/huggingface/hub/models--t5-large/snapshots/150ebc2c4b72291e770f58e6057481c8d2ed331a/model.safetensors",
        ),
        (
            "Multilingual-E5-Large (560M, XLM-RoBERTa)",
            home / ".cache/torch/sentence_transformers/intfloat_multilingual-e5-large/model.safetensors",
        ),
        (
            "T5-Base (220M, Google Transformer)",
            home
            / ".cache/huggingface/hub/models--t5-base/snapshots/fe6d9bf207cd3337512ca838a8b453f87a9178ef/model.safetensors",
        ),
    ]
    found = [(label, p) for label, p in candidates if p.exists()]
    return found


def inspect_safetensors_header(path: pathlib.Path) -> Dict[str, dict]:
    """Čita samo JSON zaglavlje .safetensors fajla (prvih par KB) bez učitavanja modela."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return {k: v for k, v in header.items() if k != "__metadata__"}


def load_safetensors_tensor_bf16(path: pathlib.Path, tensor_name: str) -> Tuple[np.ndarray, str, float]:
    """
    Direktno pozicionira fajl pokazivač (f.seek) na traženi tenzor unutar .safetensors
    fajla od više gigabajta i vraća BFloat16 (uint16 bit-exact) matricu i vreme čitanja.
    """
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        if tensor_name not in header:
            raise KeyError(f"Tenzor '{tensor_name}' nije pronađen u {path.name}")
        info = header[tensor_name]
        dtype = info["dtype"]
        shape = info["shape"]
        start, end = info["data_offsets"]
        f.seek(8 + header_len + start)
        raw = f.read(end - start)

    if dtype == "BF16":
        w_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()
    elif dtype == "F32":
        arr_f32 = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        w_u16 = t.fp32_to_bf16_u16(arr_f32)
    elif dtype == "F16":
        arr_f16 = np.frombuffer(raw, dtype=np.float16).reshape(shape)
        w_u16 = t.fp32_to_bf16_u16(arr_f16.astype(np.float32))
    else:
        raise ValueError(f"Nepodržan tip tenzora: {dtype}")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return w_u16, dtype, elapsed_ms


# ==============================================================================
# 2. KORAK 2: GREEDY NEAREST-NEIGHBOR SORTIRANJE I 2D FLAC ADAPTIVNI LPC FILTER
# ==============================================================================
def greedy_nn_sort_cols(W_fp32: np.ndarray) -> np.ndarray:
    """
    Sortira kolone matrice po maksimalnoj kosinusnoj sličnosti (Greedy Nearest-Neighbor),
    tako da srodni neuroni (npr. unutar iste Attention glave ili semantičkog klastera)
    stoje jedan pored drugog.
    """
    norms = np.linalg.norm(W_fp32, axis=0) + 1e-8
    W_norm = W_fp32 / norms
    sim = W_norm.T @ W_norm
    cols = W_fp32.shape[1]
    visited = np.zeros(cols, dtype=bool)
    perm = np.empty(cols, dtype=np.int32)

    curr = int(np.argmax(norms))
    perm[0] = curr
    visited[curr] = True

    for i in range(1, cols):
        row_sim = sim[curr]
        # Privremeno maskiranje posećenih kolona
        masked_sim = np.where(visited, -1e9, row_sim)
        nxt = int(np.argmax(masked_sim))
        perm[i] = nxt
        visited[nxt] = True
        curr = nxt
    return perm


def flac_2d_lpc_encode(q_int: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D FLAC Adaptivni LPC-1 prostorni filter (100% celobrojno reverzibilan!).
    Za razliku od naivnog PNG oduzimanja (res[j] - res[j-1], gde je koeficijent fiksiran na 1.0
    pa pojačava šum za 2*(1-rho) kada je korelacija suseda rho < 0.5),
    FLAC LPC-1 računa optimalni Q7 fiksni koeficijent predikcije:
        alpha_i = round(128 * rho_i) / 128  u [-127/128, +127/128]
    čime UVEK smanjuje varijansu za faktor (1 - rho^2) bez obzira na jačinu korelacije!
    """
    # Prolaz 1: Predikcija po redovima (Up LPC-1)
    q32 = q_int.astype(np.int32)
    num_r = np.sum(q32[1:] * q32[:-1], axis=1, dtype=np.float64)
    den_r = np.sum(q32[:-1] ** 2, axis=1, dtype=np.float64) + 1e-8
    rho_r = np.clip(np.round((num_r / den_r) * 128.0), -127, 127).astype(np.int8)

    r1 = q_int.copy()
    pred_r = (q32[:-1] * rho_r.astype(np.int32)[:, None]) >> 7
    r1[1:] = (q32[1:] - pred_r).astype(np.int16)

    # Prolaz 2: Predikcija po kolonama (Sub/Left LPC-1) nad ostatkom prvog prolaza
    r1_32 = r1.astype(np.int32)
    num_c = np.sum(r1_32[:, 1:] * r1_32[:, :-1], axis=0, dtype=np.float64)
    den_c = np.sum(r1_32[:, :-1] ** 2, axis=0, dtype=np.float64) + 1e-8
    rho_c = np.clip(np.round((num_c / den_c) * 128.0), -127, 127).astype(np.int8)

    r2 = r1.copy()
    pred_c = (r1_32[:, :-1] * rho_c.astype(np.int32)[None, :]) >> 7
    r2[:, 1:] = (r1_32[:, 1:] - pred_c).astype(np.int16)

    return r2, rho_r, rho_c


def flac_2d_lpc_decode(r2: np.ndarray, rho_r: np.ndarray, rho_c: np.ndarray) -> np.ndarray:
    """100% bit-exact celobrojna rekonstrukcija 2D FLAC LPC-1 filtera."""
    rows, cols = r2.shape
    r1 = r2.copy()
    rc_i32 = rho_c.astype(np.int32)
    for j in range(1, cols):
        pred_c = (r1[:, j - 1].astype(np.int32) * rc_i32[j - 1]) >> 7
        r1[:, j] = (r1[:, j].astype(np.int32) + pred_c).astype(np.int16)

    q_rec = r1.copy()
    rr_i32 = rho_r.astype(np.int32)
    for i in range(1, rows):
        pred_r = (q_rec[i - 1, :].astype(np.int32) * rr_i32[i - 1]) >> 7
        q_rec[i, :] = (q_rec[i, :].astype(np.int32) + pred_r).astype(np.int16)

    return q_rec


# ==============================================================================
# 3. KORAK 3: .LRFLAC BINARNI FORMAT (Pakovanje i merenje brzine u MB/s)
# ==============================================================================
MAGIC_HEADER = b"LRFLAC02"
MODE_LOSSLESS_MAG_LUT = 1
MODE_LOSSLESS_SVD_LUT = 2
MODE_NEAR_LOSSLESS_BFP = 3


def pack_u4_pairs(arr_u8: np.ndarray) -> bytes:
    """Pakuje niz 4-bitnih vrednosti (0..15) po dve u jedan bajt (tačno 4.00 bita/elementu)."""
    flat = arr_u8.ravel()
    if flat.size % 2 != 0:
        flat = np.append(flat, np.uint8(0))
    packed = (flat[0::2] << 4) | (flat[1::2] & 0x0F)
    return packed.astype(np.uint8).tobytes()


def unpack_u4_pairs(data: bytes, shape: Tuple[int, int]) -> np.ndarray:
    """Raspakuje 4-bitne parove iz bajtova nazad u 2D uint8 matricu."""
    total = shape[0] * shape[1]
    packed = np.frombuffer(data, dtype=np.uint8)
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2] = packed >> 4
    out[1::2] = packed & 0x0F
    return out[:total].reshape(shape)


def encode_lrflac_lossless_mag_lut(W_u16: np.ndarray, use_lzma: bool = False) -> Tuple[bytes, float]:
    """
    Pakuje BFloat16 matricu u 100% Bit-Exact Lossless .lrflac binarni format
    koristeći 2D sortiranje normi + Magnitude-LUT (hi4 + 4b lo + 1b sign).
    """
    t0 = time.perf_counter()
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
    lut_bytes = np.packbits(lut_mask).tobytes()  # 4096 bajtova sirovo

    mag_idx = np.searchsorted(uniq_mag, mag).astype(np.uint16)
    hi4 = (mag_idx >> 4).astype(np.uint8)
    lo4 = (mag_idx & 0x0F).astype(np.uint8)

    # Sirovi (već maksimalno entropijski gusti) bit-streamovi za znak (1b) i donjih 4 bita (4b)
    raw_sign_packed = np.packbits(sign_bits).tobytes()
    raw_lo4_packed = pack_u4_pairs(lo4)

    # Kompresija strukture (perm_r, perm_c, lut_bytes, hi4)
    meta_raw = (
        struct.pack("<II", len(perm_r), len(perm_c))
        + perm_r.tobytes()
        + perm_c.tobytes()
        + lut_bytes
    )
    comp_meta = zlib.compress(meta_raw, level=9)
    comp_hi4 = lzma.compress(hi4.tobytes()) if use_lzma else zlib.compress(hi4.tobytes(), level=9)

    header = struct.pack(
        "<8sBBIIIII",
        MAGIC_HEADER,
        MODE_LOSSLESS_MAG_LUT,
        1 if use_lzma else 0,
        rows,
        cols,
        len(comp_meta),
        len(comp_hi4),
        len(raw_sign_packed),
    )
    blob = header + comp_meta + comp_hi4 + raw_sign_packed + raw_lo4_packed
    enc_ms = (time.perf_counter() - t0) * 1000.0
    return blob, enc_ms


def decode_lrflac_lossless_mag_lut(blob: bytes) -> Tuple[np.ndarray, float]:
    """Raspakuje .lrflac binarni bafer nazad u 100% Bit-Exact BFloat16 matricu i meri brzinu."""
    t0 = time.perf_counter()
    hdr_size = struct.calcsize("<8sBBIIIII")
    magic, mode, codec_flag, rows, cols, len_meta, len_hi4, len_sign = struct.unpack(
        "<8sBBIIIII", blob[:hdr_size]
    )
    assert magic == MAGIC_HEADER and mode == MODE_LOSSLESS_MAG_LUT

    pos = hdr_size
    meta_raw = zlib.decompress(blob[pos : pos + len_meta])
    pos += len_meta

    hi4_raw = (
        lzma.decompress(blob[pos : pos + len_hi4])
        if codec_flag == 1
        else zlib.decompress(blob[pos : pos + len_hi4])
    )
    pos += len_hi4

    sign_packed = blob[pos : pos + len_sign]
    pos += len_sign
    lo4_packed = blob[pos:]

    # Raspakivanje meta podataka
    nr, nc = struct.unpack("<II", meta_raw[:8])
    mpos = 8
    perm_r = np.frombuffer(meta_raw[mpos : mpos + nr * 2], dtype=np.uint16)
    mpos += nr * 2
    perm_c = np.frombuffer(meta_raw[mpos : mpos + nc * 2], dtype=np.uint16)
    mpos += nc * 2
    lut_mask = np.unpackbits(np.frombuffer(meta_raw[mpos : mpos + 4096], dtype=np.uint8))
    uniq_mag = np.flatnonzero(lut_mask).astype(np.uint16)

    # Rekonstrukcija indeksa i amplituda
    hi4 = np.frombuffer(hi4_raw, dtype=np.uint8).reshape((rows, cols))
    lo4 = unpack_u4_pairs(lo4_packed, (rows, cols))
    mag_idx = (hi4.astype(np.uint16) << 4) | lo4.astype(np.uint16)
    mag_rec = uniq_mag[mag_idx]

    sign_bits = np.unpackbits(np.frombuffer(sign_packed, dtype=np.uint8))[: rows * cols].reshape(
        (rows, cols)
    )
    W_s_rec = (sign_bits.astype(np.uint16) << 15) | mag_rec

    inv_r = np.argsort(perm_r)
    inv_c = np.argsort(perm_c)
    W_rec = W_s_rec[inv_r][:, inv_c]

    dec_ms = (time.perf_counter() - t0) * 1000.0
    return W_rec, dec_ms


def encode_lrflac_near_lossless_bfp(
    W_u16: np.ndarray, mantissa_bits: int = 6, use_2d_lpc: bool = True
) -> Tuple[bytes, float]:
    """
    Pakuje BFloat16 matricu u .lrflac Near-Lossless Block-Float ('AV1') format
    uz Greedy-NN sortiranje i 2D FLAC Adaptivni LPC filter.
    """
    t0 = time.perf_counter()
    rows, cols = W_u16.shape
    W_fp32 = t.bf16_u16_to_fp32(W_u16)

    perm_c = greedy_nn_sort_cols(W_fp32).astype(np.uint16)
    perm_r = np.argsort(np.linalg.norm(W_fp32, axis=1)).astype(np.uint16)

    W_s_u16 = W_u16[perm_r][:, perm_c]
    W_s_fp32 = W_fp32[perm_r][:, perm_c]

    exps = ((W_s_u16 >> 7) & 0xFF).astype(np.int32)
    col_max_exp = np.max(exps, axis=0).astype(np.uint8)
    col_step = np.ldexp(np.ones(cols, dtype=np.float32), col_max_exp.astype(np.int32) - 127 - mantissa_bits)

    q_int = np.round(W_s_fp32 / col_step[None, :]).astype(np.int16)

    if use_2d_lpc:
        res_int, rho_r, rho_c = flac_2d_lpc_encode(q_int)
    else:
        res_int = q_int
        rho_r = np.zeros(rows - 1, dtype=np.int8)
        rho_c = np.zeros(cols - 1, dtype=np.int8)

    zz = ((res_int >> 15) ^ (res_int << 1)).view(np.uint16)
    low_b = (zz & 0xFF).astype(np.uint8)
    high_b = (zz >> 8).astype(np.uint8)

    meta_raw = (
        perm_r.tobytes()
        + perm_c.tobytes()
        + col_max_exp.tobytes()
        + rho_r.tobytes()
        + rho_c.tobytes()
    )
    comp_meta = zlib.compress(meta_raw, level=9)
    comp_low = zlib.compress(low_b.tobytes(), level=9)
    comp_high = zlib.compress(high_b.tobytes(), level=9) if np.any(high_b) else b""

    header = struct.pack(
        "<8sBBBIIII",
        MAGIC_HEADER,
        MODE_NEAR_LOSSLESS_BFP,
        mantissa_bits,
        1 if use_2d_lpc else 0,
        rows,
        cols,
        len(comp_meta),
        len(comp_low),
    )
    blob = header + comp_meta + comp_low + comp_high
    enc_ms = (time.perf_counter() - t0) * 1000.0
    return blob, enc_ms


def decode_lrflac_near_lossless_bfp(blob: bytes) -> Tuple[np.ndarray, float]:
    """Raspakuje .lrflac Near-Lossless Block-Float bafer nazad u BFloat16 matricu."""
    t0 = time.perf_counter()
    hdr_size = struct.calcsize("<8sBBBIIII")
    magic, mode, mantissa_bits, lpc_flag, rows, cols, len_meta, len_low = struct.unpack(
        "<8sBBBIIII", blob[:hdr_size]
    )
    assert magic == MAGIC_HEADER and mode == MODE_NEAR_LOSSLESS_BFP

    pos = hdr_size
    meta_raw = zlib.decompress(blob[pos : pos + len_meta])
    pos += len_meta
    low_b = np.frombuffer(zlib.decompress(blob[pos : pos + len_low]), dtype=np.uint8).reshape(
        (rows, cols)
    )
    pos += len_low
    if pos < len(blob):
        high_b = np.frombuffer(zlib.decompress(blob[pos:]), dtype=np.uint8).reshape((rows, cols))
    else:
        high_b = np.zeros((rows, cols), dtype=np.uint8)

    mpos = 0
    perm_r = np.frombuffer(meta_raw[mpos : mpos + rows * 2], dtype=np.uint16)
    mpos += rows * 2
    perm_c = np.frombuffer(meta_raw[mpos : mpos + cols * 2], dtype=np.uint16)
    mpos += cols * 2
    col_max_exp = np.frombuffer(meta_raw[mpos : mpos + cols], dtype=np.uint8)
    mpos += cols
    rho_r = np.frombuffer(meta_raw[mpos : mpos + (rows - 1)], dtype=np.int8)
    mpos += rows - 1
    rho_c = np.frombuffer(meta_raw[mpos : mpos + (cols - 1)], dtype=np.int8)

    zz = (high_b.astype(np.uint16) << 8) | low_b.astype(np.uint16)
    res_int = ((zz >> 1).astype(np.int32) ^ -(zz & 1).astype(np.int32)).astype(np.int16)

    if lpc_flag == 1:
        q_int = flac_2d_lpc_decode(res_int, rho_r, rho_c)
    else:
        q_int = res_int

    col_step = np.ldexp(np.ones(cols, dtype=np.float32), col_max_exp.astype(np.int32) - 127 - mantissa_bits)
    W_s_fp32 = q_int.astype(np.float32) * col_step[None, :]

    inv_r = np.argsort(perm_r)
    inv_c = np.argsort(perm_c)
    W_rec_u16 = t.fp32_to_bf16_u16(W_s_fp32)[inv_r][:, inv_c]

    dec_ms = (time.perf_counter() - t0) * 1000.0
    return W_rec_u16, dec_ms


# ==============================================================================
# 4. KOMPLETAN BENCHMARK NA PRAVIM .SAFETENSORS SLOJEVIMA (KORAK 1, 2 i 3)
# ==============================================================================
def benchmark_layer(model_label: str, path: pathlib.Path, tensor_name: str, max_rows: int = 4096):
    W_u16_full, orig_dtype, read_ms = load_safetensors_tensor_bf16(path, tensor_name)
    if W_u16_full.shape[0] > max_rows:
        W_u16 = W_u16_full[:max_rows].copy()
        sliced_note = f" (isečak prvih {max_rows} od {W_u16_full.shape[0]} redova)"
    else:
        W_u16 = W_u16_full
        sliced_note = ""

    rows, cols = W_u16.shape
    total = rows * cols
    raw_bytes = W_u16.nbytes
    raw_mb = raw_bytes / (1024 * 1024)
    W_fp32 = t.bf16_u16_to_fp32(W_u16)

    print("=" * 84)
    print(f"MODEL: {model_label}")
    print(f"SLOJ:  {tensor_name}{sliced_note}")
    print(
        f"DIMENZIJE: {rows} x {cols} ({total:,} težina, {raw_mb:.2f} MB u BF16) | "
        f"Direktno .safetensors čitanje: {read_ms:.1f} ms"
    )
    print("=" * 84)

    # ------------------------------------------------------------------
    # A. Dijagnostika spektra i lokalnih korelacija ("Divlje" vs Sintetičke težine)
    # ------------------------------------------------------------------
    U, S, Vh = np.linalg.svd(W_fp32, full_matrices=False)
    e_tot = float(np.sum(S ** 2))
    e16 = float(np.sum(S[:16] ** 2)) / e_tot * 100
    e64 = float(np.sum(S[:64] ** 2)) / e_tot * 100
    e128 = float(np.sum(S[:128] ** 2)) / e_tot * 100
    e256 = float(np.sum(S[:256] ** 2)) / e_tot * 100

    uniq_bf16 = len(np.unique(W_u16))
    uniq_mag = len(np.unique(W_u16 & 0x7FFF))

    perm_c_gnn = greedy_nn_sort_cols(W_fp32)
    W_gnn = W_fp32[:, perm_c_gnn]
    col_dots = np.sum(W_gnn[:, 1:] * W_gnn[:, :-1], axis=0) / (
        np.linalg.norm(W_gnn[:, 1:], axis=0) * np.linalg.norm(W_gnn[:, :-1], axis=0) + 1e-8
    )

    print("[DIJAGNOSTIKA PRAVOG SLOJA]:")
    print(
        f"  • SVD Energija (Power-Law opadanje): Top-16={e16:.1f}% | Top-64={e64:.1f}% | "
        f"Top-128={e128:.1f}% | Top-256={e256:.1f}%"
    )
    print(
        f"  • Aktivni BF16 vokabular: {uniq_bf16:,} / 65,536 vrednosti ({uniq_bf16/655.36:.1f}%) | "
        f"Unikatnih amplituda |W|: {uniq_mag:,} / 32,768 (staje u {int(np.ceil(np.log2(uniq_mag)))} bita!)"
    )
    print(
        f"  • Kosinusna sličnost susednih kolona posle Greedy-NN: prosek={float(np.mean(col_dots)):.3f}, "
        f"maksimum={float(np.max(col_dots)):.3f}\n"
    )

    # ------------------------------------------------------------------
    # B. Poređenje 100% BIT-EXACT LOSSLESS metoda (DFloat11 vs LR-FLAC .lrflac)
    # ------------------------------------------------------------------
    s_raw = np.packbits(((W_u16 >> 15) & 1).astype(np.uint8)).tobytes()
    e_raw = ((W_u16 >> 7) & 0xFF).astype(np.uint8).tobytes()
    m_raw = (W_u16 & 0x7F).astype(np.uint8).tobytes()

    df11_zlib_bytes = t.compress_bytes(s_raw) + t.compress_bytes(e_raw) + t.compress_bytes(m_raw)
    df11_lzma_bytes = len(lzma.compress(s_raw)) + len(lzma.compress(e_raw)) + len(lzma.compress(m_raw))
    df11_zlib_bpw = (df11_zlib_bytes * 8) / total
    df11_lzma_bpw = (df11_lzma_bytes * 8) / total

    blob_zlib, enc_zl_ms = encode_lrflac_lossless_mag_lut(W_u16, use_lzma=False)
    W_rec_zl, dec_zl_ms = decode_lrflac_lossless_mag_lut(blob_zlib)
    exact_zl = bool(np.array_equal(W_u16, W_rec_zl))
    lrflac_zl_bpw = (len(blob_zlib) * 8) / total
    speed_zl_mbs = raw_mb / (dec_zl_ms / 1000.0)

    blob_lzma, enc_lz_ms = encode_lrflac_lossless_mag_lut(W_u16, use_lzma=True)
    W_rec_lz, dec_lz_ms = decode_lrflac_lossless_mag_lut(blob_lzma)
    exact_lz = bool(np.array_equal(W_u16, W_rec_lz))
    lrflac_lz_bpw = (len(blob_lzma) * 8) / total
    speed_lz_mbs = raw_mb / (dec_lz_ms / 1000.0)

    # Upisujemo pravi .lrflac fajl na disk i proveravamo čitanje
    out_dir = pathlib.Path("artifacts_lrflac")
    out_dir.mkdir(exist_ok=True)
    safe_name = tensor_name.replace(".", "_")
    lrflac_path = out_dir / f"{safe_name}_lossless.lrflac"
    lrflac_path.write_bytes(blob_zlib)

    print("[100% BIT-EXACT LOSSLESS REŽIM ('FLAC' — Nula izgubljenih bitova)]:")
    print(
        f"  [0] Sirovi BFloat16 ('WAV'):            16.000 bita/težini ({raw_Bytes_kb(raw_bytes)})"
    )
    print(
        f"  [1] IBM ZipNN / DFloat11 (zlib):        {df11_zlib_bpw:6.3f} bita/težini "
        f"(Ušteda: {(1 - df11_zlib_bpw/16)*100:4.1f}%)"
    )
    print(
        f"  [2] IBM ZipNN / DFloat11 (LZMA):        {df11_lzma_bpw:6.3f} bita/težini "
        f"(Ušteda: {(1 - df11_lzma_bpw/16)*100:4.1f}%)"
    )
    print(
        f"  [3] Naš LR-FLAC .lrflac (zlib, brzi):   {lrflac_zl_bpw:6.3f} bita/težini "
        f"(Ušteda: {(1 - lrflac_zl_bpw/16)*100:4.1f}%) | Dekompresija: {speed_zl_mbs:6.1f} MB/s | Bit-Exact={exact_zl}"
    )
    print(
        f"  [4] Naš LR-FLAC .lrflac (LZMA, max):    {lrflac_lz_bpw:6.3f} bita/težini "
        f"(Ušteda: {(1 - lrflac_lz_bpw/16)*100:4.1f}%) | Dekompresija: {speed_lz_mbs:6.1f} MB/s | Bit-Exact={exact_lz}\n"
    )

    # ------------------------------------------------------------------
    # C. Poređenje NEAR-LOSSLESS BLOCK-FLOAT ('AV1') + KORAK 2 (2D FLAC LPC)
    # ------------------------------------------------------------------
    print("[NEAR-LOSSLESS BLOCK-FLOAT REŽIM ('AV1') + KORAK 2 (2D FLAC Adaptivni LPC Filter)]:")
    rng_test = np.random.default_rng(123)
    X_test_u16 = t.fp32_to_bf16_u16(rng_test.standard_normal((16, rows), dtype=np.float32))
    Y_true = t.bf16_u16_to_fp32(t.matmul_bf16(X_test_u16, W_u16)).flatten()

    for mbits in [7, 6, 5, 4]:
        blob_bfp, _ = encode_lrflac_near_lossless_bfp(W_u16, mantissa_bits=mbits, use_2d_lpc=True)
        W_rec_bfp, dec_bfp_ms = decode_lrflac_near_lossless_bfp(blob_bfp)
        bpw_bfp = (len(blob_bfp) * 8) / total
        speed_bfp_mbs = raw_mb / (dec_bfp_ms / 1000.0)

        Y_rec = t.bf16_u16_to_fp32(t.matmul_bf16(X_test_u16, W_rec_bfp)).flatten()
        cos_sim = float(np.dot(Y_true, Y_rec) / (np.linalg.norm(Y_true) * np.linalg.norm(Y_rec)))

        if mbits == 6:
            bfp_path = out_dir / f"{safe_name}_bfp6.lrflac"
            bfp_path.write_bytes(blob_bfp)

        print(
            f"  • [AV1 BFP-{mbits}bit + 2D-LPC (.lrflac)]: {bpw_bfp:5.2f} bita/težini "
            f"(Ušteda: {(1 - bpw_bfp/16)*100:4.1f}%) | cos_sim = {cos_sim:.7f} | "
            f"Dekompresija: {speed_bfp_mbs:6.1f} MB/s"
        )
    print()


def raw_Bytes_kb(b: int) -> str:
    return f"{b / 1024:.1f} KB"


def compare_png_delta_vs_flac_lpc_demo(path: pathlib.Path):
    """Eksplicitni eksperiment za Korak 2: Zašto naivni PNG Delta greši, a FLAC 2D-LPC pobeđuje."""
    print("=" * 84)
    print("KORAK 2 DOKAZ: NAIVNI PNG DELTA (alpha=1.0) vs. FLAC ADAPTIVNI 2D-LPC (Q7 koeficijenti)")
    print("=" * 84)

    for tname in [
        "decoder.block.12.layer.0.SelfAttention.q.weight",
        "decoder.block.12.layer.2.DenseReluDense.wi.weight",
        "shared.weight",
    ]:
        W_u16, _, _ = load_safetensors_tensor_bf16(path, tname)
        W_u16 = W_u16[:4096]
        rows, cols = W_u16.shape
        W_fp32 = t.bf16_u16_to_fp32(W_u16)

        perm_c = greedy_nn_sort_cols(W_fp32)
        perm_r = greedy_nn_sort_cols(W_fp32[:, perm_c].T)
        W_s = W_fp32[perm_r][:, perm_c]
        W_s_u16 = W_u16[perm_r][:, perm_c]

        exps = ((W_s_u16 >> 7) & 0xFF).astype(np.int32)
        col_max_exp = np.max(exps, axis=0)
        col_step = np.ldexp(np.ones(cols, dtype=np.float32), col_max_exp - 127 - 6)
        q_int = np.round(W_s / col_step[None, :]).astype(np.int16)

        # 1. Naivni PNG Sub filter (res[:, j] - res[:, j-1], alpha = 1.0)
        png_delta = q_int.copy()
        png_delta[:, 1:] = q_int[:, 1:] - q_int[:, :-1]

        # 2. Naš FLAC 2D-LPC filter (sa verifikacijom 100% tačne rekonstrukcije)
        lpc_res, rho_r, rho_c = flac_2d_lpc_encode(q_int)
        q_rec = flac_2d_lpc_decode(lpc_res, rho_r, rho_c)
        assert np.array_equal(q_int, q_rec), "FLAC 2D-LPC mora biti 100% celobrojno reverzibilan!"

        h_raw = t.shannon_entropy_bits(q_int)
        h_png = t.shannon_entropy_bits(png_delta)
        h_lpc = t.shannon_entropy_bits(lpc_res)

        print(f"Sloj: {tname} ({rows}x{cols})")
        print(
            f"  • Entropija bez prostornog filtera:       {h_raw:.3f} bita\n"
            f"  • Naivni PNG Delta (res[j] - res[j-1]):   {h_png:.3f} bita "
            f"({'POGORŠANJE +' if h_png > h_raw else ''}{h_png - h_raw:.3f}b zbog šuma 2*(1-rho)!)\n"
            f"  • FLAC Adaptivni 2D-LPC (Q7 koeficijenti): {h_lpc:.3f} bita "
            f"(UŠTEDA {h_lpc - h_raw:.3f}b | 100% Reversible = True)\n"
        )


def main():
    models = find_local_safetensors()
    if not models:
        print("Nisu pronađeni lokalni .safetensors fajlovi u kešu.")
        return

    print("\nPronađeni lokalni .safetensors modeli na disku:")
    for label, p in models:
        print(f"  -> {label}: {p} ({p.stat().st_size / (1024**2):.1f} MB)")
    print()

    # Korak 1 + Korak 3: Benchmark nad T5-Large (770M) i Multilingual-E5-Large (560M)
    t5_label, t5_path = models[0]
    benchmark_layer(t5_label, t5_path, "decoder.block.12.layer.0.SelfAttention.q.weight")
    benchmark_layer(t5_label, t5_path, "decoder.block.12.layer.2.DenseReluDense.wi.weight")
    benchmark_layer(t5_label, t5_path, "shared.weight", max_rows=4096)

    if len(models) > 1:
        e5_label, e5_path = models[1]
        benchmark_layer(e5_label, e5_path, "encoder.layer.12.attention.self.query.weight")
        benchmark_layer(e5_label, e5_path, "encoder.layer.12.intermediate.dense.weight")

    # Korak 2: Eksplicitno poređenje PNG Delta vs FLAC 2D-LPC posle Greedy-NN sortiranja
    compare_png_delta_vs_flac_lpc_demo(t5_path)


if __name__ == "__main__":
    main()
