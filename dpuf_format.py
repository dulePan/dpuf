import time
import json
import mmap
import struct
import zlib
import ctypes
import pathlib
from datetime import datetime, timezone
from typing import Dict, Tuple, Any, Optional
import numpy as np

# ==============================================================================
# 1. KONSTANTE FORMATA, VERZIONISANJE (v1.2.0), MAGIČNI BAJTOVI I GRANICE
# ==============================================================================
MAGIC_DPUF = b"DPUF0001"
MAGIC_LRFLAC = b"LRFLAC03"
VALID_MAGICS = (MAGIC_DPUF, MAGIC_LRFLAC)

DPUF_FORMAT_VERSION = "1.2.0"
DPUF_CODEC_VERSION = "1.2.0"

MAX_ALLOWED_TENSORS = 4096
MAX_ALLOWED_DIM = 262144
MAX_ALLOWED_TENSOR_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB po tenzoru

MODE_RAW_TINY = 0
MODE_LOSSLESS_MAG_LUT = 2
MODE_NEAR_LOSSLESS_2D_RMS_BYTE = 4
MODE_NEAR_LOSSLESS_2D_RMS_NIBBLE = 5
MODE_DIRECT_MMAP_4BIT = 6
MODE_DIRECT_ALS_5BIT = 7  # v1.1.0: FP16 ALS-2D-RMS + 5-bit base + 4-bit Nibble-Escape
MODE_QR_TILED_ZD4 = 8     # v1.2.0: QR Self-Contained 2D Tiles + Finder + XOR ECC Parity
MODE_QR_SPLIT_INT4 = 9    # v1.2.0-fast (Variant A): Branchless Split 4-bit [-8..7] + Sparse Gap-Delta CSR + Q8_32 VPMADDUBSW
MODE_QR_HADAMARD_Q4 = 10  # v1.2.0-fast (Variant B): Walsh-Hadamard FWHT-64 + Pure Dense Blk32 4-bit + Q8_32 VPMADDUBSW
MODE_QR_Q8 = 11           # v1.2.1: QR-Tiled Pure Dense Blk32 Symmetric int8 [-127..127] (0% outliers, 8.5 b/w high precision)
MODE_QR_HAD4_PACKED = 12  # v1.2.2: Pre-Packed 8x32 QR Hadamard4 (Zero Transpose, Zero wsum128, L2-Tiled pp256)
MODE_QR_HAD4_ASYM = 13    # v1.3.0: Pre-Packed 8x64 Asymmetric QR Hadamard4 (blk64 d+m, LSQ 4-bit)
MODE_QR_HAD6 = 14         # v1.3.0: Pre-Packed 8x64 Asymmetric QR Hadamard6 (blk64 d+m, LSQ 4+2-bit)
ALL_QR_MODES = (
    MODE_QR_TILED_ZD4, MODE_QR_SPLIT_INT4, MODE_QR_HADAMARD_Q4, MODE_QR_Q8,
    MODE_QR_HAD4_PACKED, MODE_QR_HAD4_ASYM, MODE_QR_HAD6,
)

DEFAULT_LICENSE_NOTICE = (
    "Free for Research, Non-Commercial Use, and Commercial Use <= 1,000,000 MAU. "
    "Enterprise (> 1M MAU) requires a written license from D. P. (dule_pan)."
)


def semver_to_code(version_str: str) -> int:
    """Pretvara SemVer string ('1.2.3') u uporedivi ceo broj (10203) radi provere kompatibilnosti."""
    parts = (str(version_str).split(".") + ["0", "0", "0"])[:3]
    major, minor, patch = (int(p) for p in parts)
    return major * 10000 + minor * 100 + patch


DPUF_FORMAT_VERSION_CODE = semver_to_code(DPUF_FORMAT_VERSION)


def _check_reader_compatibility(meta: Dict[str, Any]) -> None:
    """Proverava da li trenutna verzija biblioteke ume da pročita dati .dpuf fajl."""
    min_ver = meta.get("min_reader_version")
    if min_ver:
        if DPUF_FORMAT_VERSION_CODE < semver_to_code(min_ver):
            raise ValueError(
                f"Nekompatibilna verzija .dpuf fajla: fajl zahteva čitač >= v{min_ver}, "
                f"a instalirana je verzija v{DPUF_FORMAT_VERSION}. Ažurirajte dpuf_format!"
            )


# ==============================================================================
# 2. IEEE-754 BFLOAT16 <-> FLOAT32 KONVERZIJA I 4-BIT / 5-BIT VEKTORSKO PAKOVANJE
# ==============================================================================
def fp32_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    """IEEE-754 Round-to-Nearest-Even konverzija iz Float32 u BFloat16 (uint16 zapis)."""
    f32 = np.asarray(x, dtype=np.float32)
    u32 = f32.view(np.uint32)
    rounding_bias = ((u32 >> 16) & 1) + np.uint32(0x7FFF)
    return ((u32 + rounding_bias) >> 16).astype(np.uint16)


def bf16_u16_to_fp32(u16: np.ndarray) -> np.ndarray:
    """Pretvara BFloat16 (uint16) niz u Float32 bez gubitka tačnosti."""
    u32 = np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16
    return u32.view(np.float32)


def pack_u4_pairs(u4: np.ndarray) -> bytes:
    flat = u4.ravel().astype(np.uint8)
    if flat.size % 2 != 0:
        flat = np.append(flat, np.uint8(0))
    return (((flat[0::2] & 0x0F) << 4) | (flat[1::2] & 0x0F)).astype(np.uint8).tobytes()


def unpack_u4_pairs(buf: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    total = int(np.prod(shape))
    b = np.frombuffer(buf, dtype=np.uint8)
    out = np.empty(b.size * 2, dtype=np.uint8)
    out[0::2] = (b >> 4) & 0x0F
    out[1::2] = b & 0x0F
    return out[:total].reshape(shape)


def pack_u5_values(u5: np.ndarray) -> bytes:
    """Vektorski pakuje niz 5-bitnih vrednosti (0..31): 8 vrednosti u 5 bajtova."""
    flat = u5.ravel().astype(np.uint8)
    rem = (-flat.size) % 8
    if rem:
        flat = np.pad(flat, (0, rem))
    g = flat.reshape(-1, 8).astype(np.uint64)
    w = (
        g[:, 0]
        | (g[:, 1] << 5)
        | (g[:, 2] << 10)
        | (g[:, 3] << 15)
        | (g[:, 4] << 20)
        | (g[:, 5] << 25)
        | (g[:, 6] << 30)
        | (g[:, 7] << 35)
    )
    b = np.empty((g.shape[0], 5), dtype=np.uint8)
    for i in range(5):
        b[:, i] = (w >> (8 * i)) & 0xFF
    return b.tobytes()


def unpack_u5_values(buf: bytes, count: int) -> np.ndarray:
    """Vektorski raspakuje 5-bitne vrednosti (0..31) iz 5-bajtnih blokova."""
    b = np.frombuffer(buf, dtype=np.uint8).reshape(-1, 5).astype(np.uint64)
    w = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24) | (b[:, 4] << 32)
    out = np.empty((b.shape[0], 8), dtype=np.uint8)
    for i in range(8):
        out[:, i] = (w >> (5 * i)) & 0x1F
    return out.ravel()[:count]


def encode_tensor_raw_tiny(W_u16: np.ndarray) -> bytes:
    """Packs 1D or 2D BF16 uint16 tensor uncompressed in MODE_RAW_TINY (exact 0-loss)."""
    arr = np.ascontiguousarray(W_u16, dtype=np.uint16)
    hdr = struct.pack(
        "<BBII",
        MODE_RAW_TINY,
        arr.ndim,
        arr.shape[0],
        arr.shape[1] if arr.ndim == 2 else 1,
    )
    return hdr + arr.tobytes()


# ==============================================================================
# 3. 100% BIT-EXACT LOSSLESS KODER / DEKODER (MAGNITUDE-LUT)
# ==============================================================================
def encode_tensor_lossless(W_u16: np.ndarray, zlevel: int = 6) -> bytes:
    """100% Bit-Exact Lossless koder (2D L2-norm sortiranje + Magnitude-LUT hi4 + 4b lo + 1b sign)."""
    if W_u16.ndim != 2 or min(W_u16.shape) < 64:
        comp = zlib.compress(W_u16.tobytes(), level=zlevel)
        hdr = struct.pack(
            "<BBII",
            MODE_RAW_TINY,
            W_u16.ndim,
            W_u16.shape[0],
            W_u16.shape[1] if W_u16.ndim == 2 else 1,
        )
        return hdr + comp

    rows, cols = W_u16.shape
    perm_dtype = np.uint32 if max(rows, cols) > 65535 else np.uint16
    perm_flag = 1 if perm_dtype is np.uint32 else 0

    chunk_rows = 8192 if rows > 16384 else rows
    row_norms = np.empty(rows, dtype=np.float32)
    col_sq_acc = np.zeros(cols, dtype=np.float64)
    for i0 in range(0, rows, chunk_rows):
        i1 = min(i0 + chunk_rows, rows)
        wf_c = bf16_u16_to_fp32(W_u16[i0:i1])
        row_norms[i0:i1] = np.linalg.norm(wf_c, axis=1)
        col_sq_acc += np.sum(wf_c ** 2, axis=0, dtype=np.float64)

    perm_r = np.argsort(row_norms).astype(perm_dtype)
    perm_c = np.argsort(np.sqrt(col_sq_acc)).astype(perm_dtype)
    del row_norms, col_sq_acc

    W_tmp = W_u16[perm_r]
    W_s = W_tmp[:, perm_c]
    del W_tmp

    # O(1) histogram umesto O(N log N) np.unique i O(1) LUT umesto searchsorted!
    hist = np.bincount((W_s & 0x7FFF).ravel(), minlength=32768)
    uniq_mag = np.flatnonzero(hist).astype(np.uint16)
    del hist
    lut_mask = np.zeros(32768, dtype=np.uint8)
    lut_mask[uniq_mag] = 1
    lut_bytes = np.packbits(lut_mask).tobytes()

    mag_to_idx = np.zeros(32768, dtype=np.uint16)
    mag_to_idx[uniq_mag] = np.arange(len(uniq_mag), dtype=np.uint16)

    raw_sign_packed = np.packbits(((W_s >> 15) & 1).ravel().astype(np.uint8)).tobytes()
    mag_idx = mag_to_idx[W_s & 0x7FFF]
    del W_s

    comp_hi4 = zlib.compress((mag_idx >> 4).astype(np.uint8).tobytes(), level=zlevel)
    raw_lo4_packed = pack_u4_pairs((mag_idx & 0x0F).astype(np.uint8))
    del mag_idx

    meta_raw = struct.pack("<II", len(perm_r), len(perm_c)) + perm_r.tobytes() + perm_c.tobytes() + lut_bytes
    comp_meta = zlib.compress(meta_raw, level=zlevel)

    hdr = struct.pack(
        "<BBIIIII",
        MODE_LOSSLESS_MAG_LUT,
        perm_flag,
        rows,
        cols,
        len(comp_meta),
        len(comp_hi4),
        len(raw_sign_packed),
    )
    return hdr + comp_meta + comp_hi4 + raw_sign_packed + raw_lo4_packed


def decode_tensor_lossless(blob: bytes) -> np.ndarray:
    """Raspakuje 100% Bit-Exact BFloat16 (uint16) tenzor."""
    mode = blob[0]
    if mode == MODE_RAW_TINY:
        _, ndim, d0, d1 = struct.unpack("<BBII", blob[:10])
        raw = zlib.decompress(blob[10:])
        arr = np.frombuffer(raw, dtype=np.uint16).copy()
        return arr.reshape((d0,)) if ndim == 1 else arr.reshape((d0, d1))

    hdr_size = struct.calcsize("<BBIIIII")
    _, perm_flag, rows, cols, len_meta, len_hi4, len_sign = struct.unpack("<BBIIIII", blob[:hdr_size])
    perm_dtype = np.uint32 if perm_flag == 1 else np.uint16
    perm_elem_bytes = 4 if perm_flag == 1 else 2

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
    perm_r = np.frombuffer(meta_raw[mpos : mpos + nr * perm_elem_bytes], dtype=perm_dtype)
    mpos += nr * perm_elem_bytes
    perm_c = np.frombuffer(meta_raw[mpos : mpos + nc * perm_elem_bytes], dtype=perm_dtype)
    mpos += nc * perm_elem_bytes
    lut_mask = np.unpackbits(np.frombuffer(meta_raw[mpos : mpos + 4096], dtype=np.uint8))
    uniq_mag = np.flatnonzero(lut_mask).astype(np.uint16)

    hi4 = np.frombuffer(hi4_raw, dtype=np.uint8).reshape((rows, cols))
    lo4 = unpack_u4_pairs(lo4_packed, (rows, cols))
    mag_idx = (hi4.astype(np.uint16) << 4) | lo4.astype(np.uint16)
    mag_rec = uniq_mag[mag_idx]

    sign_bits = np.unpackbits(np.frombuffer(sign_packed, dtype=np.uint8))[: rows * cols].reshape((rows, cols))
    W_s_rec = (sign_bits.astype(np.uint16) << 15) | mag_rec

    inv_r = np.argsort(perm_r)
    inv_c = np.argsort(perm_c)
    return W_s_rec[inv_r][:, inv_c]


def encode_tensor_bfp_byte(W_u16: np.ndarray, step_mult: float = 0.065, zlevel: int = 6) -> bytes:
    """Near-Lossless 2D-RMS Byte+zlib koder (MODE_NEAR_LOSSLESS_2D_RMS_BYTE, ~6 b/w)."""
    if W_u16.ndim != 2 or min(W_u16.shape) < 64:
        return encode_tensor_lossless(W_u16, zlevel=zlevel)

def _quantize_2d_rms_chunked(
    W_u16: np.ndarray, eff_step: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Memorijski efikasna 2D Row-Column RMS normalizacija i ZigZag kvantizacija (chunked za velike matrice)."""
    rows, cols = W_u16.shape
    chunk_rows = 8192 if rows > 16384 else rows

    row_sq = np.empty(rows, dtype=np.float32)
    for i0 in range(0, rows, chunk_rows):
        i1 = min(i0 + chunk_rows, rows)
        wf_c = bf16_u16_to_fp32(W_u16[i0:i1])
        row_sq[i0:i1] = np.mean(wf_c ** 2, axis=1)
    row_rms_u16 = fp32_to_bf16_u16(np.sqrt(row_sq + 1e-12))
    row_rms_fp32 = bf16_u16_to_fp32(row_rms_u16)

    col_sq_acc = np.zeros(cols, dtype=np.float64)
    for i0 in range(0, rows, chunk_rows):
        i1 = min(i0 + chunk_rows, rows)
        wf_c = bf16_u16_to_fp32(W_u16[i0:i1])
        wn_c = wf_c / (row_rms_fp32[i0:i1, None] + 1e-12)
        col_sq_acc += np.sum(wn_c ** 2, axis=0, dtype=np.float64)
    col_rms_u16 = fp32_to_bf16_u16(np.sqrt(col_sq_acc / rows + 1e-12))
    col_rms_fp32 = bf16_u16_to_fp32(col_rms_u16)

    zz = np.empty((rows, cols), dtype=np.uint16)
    col_scale = col_rms_fp32[None, :] * np.float32(eff_step)
    for i0 in range(0, rows, chunk_rows):
        i1 = min(i0 + chunk_rows, rows)
        wf_c = bf16_u16_to_fp32(W_u16[i0:i1])
        scale_c = row_rms_fp32[i0:i1, None] * col_scale
        q_c = np.round(wf_c / (scale_c + 1e-12)).astype(np.int16)
        zz[i0:i1] = ((q_c >> 15) ^ (q_c << 1)).view(np.uint16)

    return row_rms_u16, col_rms_u16, zz


def encode_tensor_bfp_byte(W_u16: np.ndarray, step_mult: float = 0.065, zlevel: int = 6) -> bytes:
    """Near-Lossless 2D-RMS Byte+zlib koder (MODE_NEAR_LOSSLESS_2D_RMS_BYTE, ~6 b/w)."""
    if W_u16.ndim != 2 or min(W_u16.shape) < 64:
        return encode_tensor_lossless(W_u16, zlevel=zlevel)

    rows, cols = W_u16.shape
    row_rms_u16, col_rms_u16, zz = _quantize_2d_rms_chunked(W_u16, step_mult)

    scales_raw = row_rms_u16.tobytes() + col_rms_u16.tobytes()
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


def decode_tensor_bfp_to_bf16(blob: bytes) -> np.ndarray:
    """Dekodira MODE_NEAR_LOSSLESS_2D_RMS_BYTE nazad u uint16 BFloat16 niz."""
    mode = blob[0]
    if mode != MODE_NEAR_LOSSLESS_2D_RMS_BYTE:
        return decode_tensor_lossless(blob)

    hdr_size = struct.calcsize("<BIfIIII")
    _, rows, step_mult, cols, len_scales, len_main, len_extra = struct.unpack("<BIfIIII", blob[:hdr_size])
    pos = hdr_size
    row_rms_u16 = np.frombuffer(blob[pos : pos + rows * 2], dtype=np.uint16)
    pos += rows * 2
    col_rms_u16 = np.frombuffer(blob[pos : pos + cols * 2], dtype=np.uint16)
    pos += cols * 2

    scale_col = bf16_u16_to_fp32(col_rms_u16)[None, :] * np.float32(step_mult)
    row_fp32 = bf16_u16_to_fp32(row_rms_u16)
    low_b = np.frombuffer(zlib.decompress(blob[pos : pos + len_main]), dtype=np.uint8).reshape((rows, cols))
    pos += len_main
    high_b = (
        np.frombuffer(zlib.decompress(blob[pos : pos + len_extra]), dtype=np.uint8).reshape((rows, cols))
        if len_extra > 0
        else None
    )

    out_u16 = np.empty((rows, cols), dtype=np.uint16)
    chunk_rows = 8192 if rows > 16384 else rows
    for i0 in range(0, rows, chunk_rows):
        i1 = min(i0 + chunk_rows, rows)
        lb = low_b[i0:i1]
        if high_b is not None:
            zz = (high_b[i0:i1].astype(np.uint16) << 8) | lb.astype(np.uint16)
            q_c = ((zz >> 1).astype(np.int32) ^ -(zz & 1).astype(np.int32)).astype(np.float32)
        else:
            q_c = ((lb >> 1).astype(np.int16) ^ -(lb & 1).astype(np.int16)).astype(np.float32)
        out_u16[i0:i1] = fp32_to_bf16_u16(q_c * (row_fp32[i0:i1, None] * scale_col))
    return out_u16


# ==============================================================================
# 4. FAST ZERO-DECODE ENGINE (NATIVE AVX2 KERNEL + FALLBACK)
# ==============================================================================
def _load_avx2_kernel() -> Optional[Any]:
    here = pathlib.Path(__file__).resolve().parent
    for fname in ("dpuf_avx2.dll", "libdpuf_avx2.so", "dpuf_avx2.dylib"):
        p = here / fname
        if p.exists():
            try:
                lib = ctypes.CDLL(str(p))
                u8p = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
                u16p = np.ctypeslib.ndpointer(dtype=np.uint16, flags="C_CONTIGUOUS")
                u32p = np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")
                u64p = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")
                f32p = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
                i32p = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")

                lib.dpuf_index_rows_zd4.argtypes = [ctypes.c_int32, ctypes.c_int32, u8p, u8p, u32p, u32p]
                lib.dpuf_index_rows_zd4.restype = None

                lib.dpuf_matmul_zd4_f32.argtypes = [
                    ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                    f32p, u8p, u8p, u16p, u32p, u32p, f32p, f32p, f32p,
                ]
                lib.dpuf_matmul_zd4_f32.restype = None

                lib.dpuf_embed_zd4_f32.argtypes = [
                    ctypes.c_int32, ctypes.c_int32, i32p, u8p, u8p, u16p, u32p, u32p, f32p, f32p, f32p,
                ]
                lib.dpuf_embed_zd4_f32.restype = None

                if hasattr(lib, "dpuf_qr_verify_tiles"):
                    lib.dpuf_qr_verify_tiles.argtypes = [ctypes.c_int32, ctypes.c_int32, u8p, u64p, u8p]
                    lib.dpuf_qr_verify_tiles.restype = ctypes.c_int32

                    lib.dpuf_matmul_qr_zd4_f32.argtypes = [
                        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                        f32p, u8p, u64p, f32p, f32p,
                    ]
                    lib.dpuf_matmul_qr_zd4_f32.restype = None

                    lib.dpuf_embed_qr_zd4_f32.argtypes = [
                        ctypes.c_int32, ctypes.c_int32, i32p, u8p, u64p, f32p, f32p,
                    ]
                    lib.dpuf_embed_qr_zd4_f32.restype = None

                if hasattr(lib, "dpuf_qr_verify_tiles_ex"):
                    lib.dpuf_qr_verify_tiles_ex.argtypes = [
                        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, u8p, u64p, u8p,
                    ]
                    lib.dpuf_qr_verify_tiles_ex.restype = ctypes.c_int32

                    lib.dpuf_matmul_qr_split4_f32.argtypes = [
                        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                        f32p, u8p, u64p, f32p, f32p,
                    ]
                    lib.dpuf_matmul_qr_split4_f32.restype = None

                    lib.dpuf_embed_qr_split4_f32.argtypes = [
                        ctypes.c_int32, ctypes.c_int32, i32p, u8p, u64p, f32p, f32p,
                    ]
                    lib.dpuf_embed_qr_split4_f32.restype = None

                    lib.dpuf_matmul_qr_hadamard4_f32.argtypes = [
                        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                        f32p, u8p, u64p, u8p, f32p,
                    ]
                    lib.dpuf_matmul_qr_hadamard4_f32.restype = None

                    lib.dpuf_embed_qr_hadamard4_f32.argtypes = [
                        ctypes.c_int32, ctypes.c_int32, i32p, u8p, u64p, u8p, f32p,
                    ]
                    lib.dpuf_embed_qr_hadamard4_f32.restype = None

                if hasattr(lib, "dpuf_matmul_qr_q8_f32"):
                    lib.dpuf_matmul_qr_q8_f32.argtypes = [
                        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                        f32p, u8p, u64p, f32p,
                    ]
                    lib.dpuf_matmul_qr_q8_f32.restype = None

                    lib.dpuf_embed_qr_q8_f32.argtypes = [
                        ctypes.c_int32, ctypes.c_int32, i32p, u8p, u64p, f32p,
                    ]
                    lib.dpuf_embed_qr_q8_f32.restype = None

                if hasattr(lib, "dpuf_matmul_qr_hadamard4_packed_f32"):
                    lib.dpuf_matmul_qr_hadamard4_packed_f32.argtypes = [
                        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                        f32p, u8p, u64p, u8p, f32p,
                    ]
                    lib.dpuf_matmul_qr_hadamard4_packed_f32.restype = None

                    lib.dpuf_embed_qr_hadamard4_packed_f32.argtypes = [
                        ctypes.c_int32, ctypes.c_int32, i32p, u8p, u64p, u8p, f32p,
                    ]
                    lib.dpuf_embed_qr_hadamard4_packed_f32.restype = None
                for fn in ("dpuf_matmul_qr_had4_asym_f32", "dpuf_matmul_qr_had6_f32"):
                    if hasattr(lib, fn) and hasattr(lib, "dpuf_matmul_qr_hadamard4_packed_f32"):
                        getattr(lib, fn).argtypes = lib.dpuf_matmul_qr_hadamard4_packed_f32.argtypes
                        getattr(lib, fn).restype = None
                for fn in ("dpuf_embed_qr_had4_asym_f32", "dpuf_embed_qr_had6_f32"):
                    if hasattr(lib, fn) and hasattr(lib, "dpuf_embed_qr_hadamard4_packed_f32"):
                        getattr(lib, fn).argtypes = lib.dpuf_embed_qr_hadamard4_packed_f32.argtypes
                        getattr(lib, fn).restype = None

                if hasattr(lib, "dpuf_engine_create"):
                    lib.dpuf_engine_create.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ]
                    lib.dpuf_engine_create.restype = ctypes.c_void_p

                    lib.dpuf_engine_destroy.argtypes = [ctypes.c_void_p]
                    lib.dpuf_engine_destroy.restype = None

                    lib.dpuf_engine_reset_cache.argtypes = [ctypes.c_void_p]
                    lib.dpuf_engine_reset_cache.restype = None

                    lib.dpuf_engine_forward_batch.argtypes = [
                        ctypes.c_void_p, ctypes.c_int32, i32p, ctypes.c_int32, ctypes.c_int32, f32p,
                    ]
                    lib.dpuf_engine_forward_batch.restype = ctypes.c_int32

                    lib.dpuf_engine_forward_token.argtypes = [
                        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, f32p,
                    ]
                    lib.dpuf_engine_forward_token.restype = ctypes.c_int32
                return lib
            except OSError:
                pass
    return None


_AVX2_LIB = _load_avx2_kernel()


class ZeroDecodeWeight:
    """
    NumPy referentni Zero-Decode objekat (fallback kada dpuf_avx2.dll nije dostupan).
    Drži float32 keš radi brzog OpenBLAS SGEMM-a, pa troši više RAM-a od PackedZeroDecodeWeight.
    """

    def __init__(self, q_int16: np.ndarray, row_scale: np.ndarray, col_scale: np.ndarray):
        self.q_int8 = q_int16
        self.row_scale = np.asarray(row_scale, dtype=np.float32)
        self.col_scale = np.asarray(col_scale, dtype=np.float32)
        self._qt_f32 = np.ascontiguousarray(q_int16.T, dtype=np.float32)

    def matmul_input_x_wt(self, x: np.ndarray) -> np.ndarray:
        x_scaled = x * self.col_scale
        out = np.matmul(x_scaled, self._qt_f32)
        return out * self.row_scale

    def lookup_embeddings(self, input_ids: np.ndarray) -> np.ndarray:
        rows_q = self.q_int8[input_ids].astype(np.float32)
        return (rows_q * self.row_scale[input_ids][..., None]) * self.col_scale[None, ...]

    def memory_bytes(self) -> int:
        return int(self.q_int8.nbytes + self._qt_f32.nbytes + self.row_scale.nbytes + self.col_scale.nbytes)


class PackedZeroDecodeWeight:
    """
    Nativni C/AVX2+FMA+OpenMP Zero-Decode objekat (Iskorak #1):
    Drži težine ISKLJUČIVO kao spakovane 4-bitne parove + retke overflow nizove (~0.64 B/težini u RAM-u!),
    BEZ ikakve float32 ili int16 kopije matrice u memoriji!
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        pairs: np.ndarray,
        over8: np.ndarray,
        super16: np.ndarray,
        row_scale: np.ndarray,
        col_scale: np.ndarray,
    ):
        if _AVX2_LIB is None:
            raise RuntimeError("dpuf_avx2 native library nije učitan.")
        self.rows = int(rows)
        self.cols = int(cols)
        self.pairs = np.ascontiguousarray(pairs, dtype=np.uint8)
        self.over8 = np.ascontiguousarray(over8, dtype=np.uint8) if over8.size > 0 else np.zeros(1, dtype=np.uint8)
        self.super16 = np.ascontiguousarray(super16, dtype=np.uint16) if super16.size > 0 else np.zeros(1, dtype=np.uint16)
        self.row_scale = np.ascontiguousarray(row_scale, dtype=np.float32)
        self.col_scale = np.ascontiguousarray(col_scale, dtype=np.float32)
        self.row_over_off = np.empty(self.rows + 1, dtype=np.uint32)
        self.row_super_off = np.empty(self.rows + 1, dtype=np.uint32)
        _AVX2_LIB.dpuf_index_rows_zd4(
            self.rows, self.cols, self.pairs, self.over8, self.row_over_off, self.row_super_off
        )

    @property
    def q_int8(self) -> np.ndarray:
        """Retko pozivan fallback samo ako stari kod direktno zahteva punu int16 matricu."""
        total = self.rows * self.cols
        rec_zz = np.empty(self.pairs.size * 2, dtype=np.uint16)
        rec_zz[0::2] = self.pairs >> 4
        rec_zz[1::2] = self.pairs & 0x0F
        rec_zz = rec_zz[:total]
        mask = rec_zz == 15
        if np.any(mask):
            o = self.over8[: int(np.sum(mask))].astype(np.uint16)
            sm = o == 255
            if np.any(sm):
                o[sm] = self.super16[: int(np.sum(sm))]
            rec_zz[mask] = o + np.uint16(15)
        zz_2d = rec_zz.reshape((self.rows, self.cols))
        return ((zz_2d >> 1).astype(np.int32) ^ -(zz_2d & 1).astype(np.int32)).astype(np.int16)

    def matmul_input_x_wt(self, x: np.ndarray) -> np.ndarray:
        orig_shape = x.shape
        x_c = np.ascontiguousarray(x, dtype=np.float32)
        m = int(x_c.size // self.cols)
        out = np.empty((m, self.rows), dtype=np.float32)
        _AVX2_LIB.dpuf_matmul_zd4_f32(
            m,
            self.rows,
            self.cols,
            x_c.ravel(),
            self.pairs,
            self.over8,
            self.super16,
            self.row_over_off,
            self.row_super_off,
            self.row_scale,
            self.col_scale,
            out.ravel(),
        )
        return out.reshape(*orig_shape[:-1], self.rows)

    def lookup_embeddings(self, input_ids: np.ndarray) -> np.ndarray:
        ids_i32 = np.ascontiguousarray(input_ids, dtype=np.int32)
        out = np.empty((ids_i32.size, self.cols), dtype=np.float32)
        _AVX2_LIB.dpuf_embed_zd4_f32(
            ids_i32.size,
            self.cols,
            ids_i32.ravel(),
            self.pairs,
            self.over8,
            self.super16,
            self.row_over_off,
            self.row_super_off,
            self.row_scale,
            self.col_scale,
            out.ravel(),
        )
        return out.reshape(*input_ids.shape, self.cols)

    def memory_bytes(self) -> int:
        return int(
            self.pairs.nbytes
            + self.over8.nbytes
            + self.super16.nbytes
            + self.row_scale.nbytes
            + self.col_scale.nbytes
            + self.row_over_off.nbytes
            + self.row_super_off.nbytes
        )


def encode_tensor_zero_decode_4bit(
    W_u16: np.ndarray, step_mult: float = 0.180, is_embed: bool = False
) -> bytes:
    """v1.0.0 Direct Zero-Decode 4-bit koder (zadržan radi pune backward kompatibilnosti)."""
    if W_u16.ndim != 2 or min(W_u16.shape) < 64:
        hdr = struct.pack(
            "<BBII",
            MODE_RAW_TINY,
            W_u16.ndim,
            W_u16.shape[0],
            W_u16.shape[1] if W_u16.ndim == 2 else 1,
        )
        return hdr + W_u16.tobytes()

    rows, cols = W_u16.shape
    eff_step = float(step_mult * (0.55 if is_embed else 1.0))
    row_rms_u16, col_rms_u16, zz = _quantize_2d_rms_chunked(W_u16, eff_step)

    zz_flat = zz.ravel()
    u4 = np.minimum(zz_flat, 15).astype(np.uint8)
    if u4.size % 2 != 0:
        u4 = np.append(u4, np.uint8(0))
    pairs = ((u4[0::2] << 4) | u4[1::2]).astype(np.uint8)

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


def encode_tensor_zero_decode_v11(
    W_u16: np.ndarray, tensor_name: str = "", base_step: float = 0.112
) -> bytes:
    """
    Iskorak #3 (v1.1.0 High-Precision Zero-Decode Codec - MODE_DIRECT_ALS_5BIT):
    1. Koristi IEEE-754 Float16 (10-bitna mantisa, 8x preciznija od BF16, ista 2 bajta!) za 2D skale.
    2. Izvršava Alternating Least-Squares (ALS) analitičku optimizaciju rednih i kolonskih skala
       tako da diag(r_opt) @ Q @ diag(c_opt) minimizuje Frobeniusovu grešku ||W - W_hat||_F^2 (0 dodatnih bajtova!).
    3. Layer-Adaptive Step (štiti embed_tokens i GQA self_attn slojeve).
    4. Dvoetažno Bit/Nibble-Escape pakovanje bez zlib-a:
       - Baza: 5-bitno pakovanje (0..30, gde je 31 escape marker)
       - Etaža 1: 4-bitni nibble parovi za izbegle vrednosti 31..45 (2 izbegla u 1 bajt!)
       - Etaža 2: uint16 za ekstremne super-outliere (>= 46).
    """
    if W_u16.ndim != 2 or min(W_u16.shape) < 64:
        hdr = struct.pack(
            "<BBII",
            MODE_RAW_TINY,
            W_u16.ndim,
            W_u16.shape[0],
            W_u16.shape[1] if W_u16.ndim == 2 else 1,
        )
        return hdr + W_u16.tobytes()

    rows, cols = W_u16.shape

    if any(t in tensor_name for t in ("embed_tokens", "token_embd", "shared", "lm_head", "output.weight")):
        eff_s = float(base_step * 0.62)
    elif any(t in tensor_name for t in ("self_attn", "SelfAttention", "EncDecAttention", "attn_q", "attn_k", "attn_v", "attn_output")):
        eff_s = float(base_step * 0.76)
    else:
        eff_s = float(base_step)

    # Memorijski efikasan chunked ALS (max ~80 MB RAM cak i za 152k vocab matrice)
    chunk_rows = 8192 if rows > 16384 else rows

    row_sq = np.empty(rows, dtype=np.float32)
    for i0 in range(0, rows, chunk_rows):
        wf_c = bf16_u16_to_fp32(W_u16[i0 : i0 + chunk_rows])
        row_sq[i0 : i0 + chunk_rows] = np.mean(wf_c ** 2, axis=1)
    row_rms0 = np.sqrt(row_sq + 1e-12).astype(np.float32)

    col_sq_acc = np.zeros(cols, dtype=np.float64)
    for i0 in range(0, rows, chunk_rows):
        wf_c = bf16_u16_to_fp32(W_u16[i0 : i0 + chunk_rows])
        wn_c = wf_c / row_rms0[i0 : i0 + chunk_rows, None]
        col_sq_acc += np.sum(wn_c ** 2, axis=0, dtype=np.float64)
    cf = np.sqrt(col_sq_acc / rows + 1e-12).astype(np.float32)
    rf = (row_rms0 * eff_s).astype(np.float32)

    # 2 koraka Alternating Least-Squares (ALS) optimizacije skala
    for _ in range(2):
        new_rf = np.empty(rows, dtype=np.float32)
        num_c = np.zeros(cols, dtype=np.float64)
        den_c = np.zeros(cols, dtype=np.float64)
        for i0 in range(0, rows, chunk_rows):
            wf_c = bf16_u16_to_fp32(W_u16[i0 : i0 + chunk_rows])
            rf_c = rf[i0 : i0 + chunk_rows, None]
            q_c = np.round(wf_c / (rf_c * cf[None, :] + 1e-12)).astype(np.float32)
            qc = q_c * cf[None, :]
            rf_upd = np.sum(wf_c * qc, axis=1) / (np.sum(qc ** 2, axis=1) + 1e-12)
            new_rf[i0 : i0 + chunk_rows] = rf_upd
            qr = q_c * rf_upd[:, None]
            num_c += np.sum(wf_c * qr, axis=0, dtype=np.float64)
            den_c += np.sum(qr ** 2, axis=0, dtype=np.float64)
        rf = new_rf
        cf = (num_c / (den_c + 1e-12)).astype(np.float32)

    rf_f16 = rf.astype(np.float16)
    cf_f16 = cf.astype(np.float16)
    rf_32 = rf_f16.astype(np.float32)
    cf_32 = cf_f16.astype(np.float32)

    # Konacni q_int + zavrsni ALS glanc po blokovima redova
    zz_flat = np.empty(rows * cols, dtype=np.uint16)
    rf_final = np.empty(rows, dtype=np.float16)
    num_c = np.zeros(cols, dtype=np.float64)
    den_c = np.zeros(cols, dtype=np.float64)
    for i0 in range(0, rows, chunk_rows):
        i1 = min(i0 + chunk_rows, rows)
        wf_c = bf16_u16_to_fp32(W_u16[i0:i1])
        q_c = np.round(wf_c / (rf_32[i0:i1, None] * cf_32[None, :] + 1e-12)).astype(np.int16)
        qf_c = q_c.astype(np.float32)
        qc = qf_c * cf_32[None, :]
        rf_upd = (np.sum(wf_c * qc, axis=1) / (np.sum(qc ** 2, axis=1) + 1e-12)).astype(np.float16)
        rf_final[i0:i1] = rf_upd
        qr = qf_c * rf_upd.astype(np.float32)[:, None]
        num_c += np.sum(wf_c * qr, axis=0, dtype=np.float64)
        den_c += np.sum(qr ** 2, axis=0, dtype=np.float64)
        zz_flat[i0 * cols : i1 * cols] = ((q_c >> 15) ^ (q_c << 1)).view(np.uint16).ravel()

    rf_f16 = rf_final
    cf_f16 = (num_c / (den_c + 1e-12)).astype(np.float16)

    u5 = np.minimum(zz_flat, 31).astype(np.uint8)
    base5_raw = pack_u5_values(u5)

    over1 = (zz_flat[zz_flat >= 31] - 31).astype(np.uint16)
    u4_tier1 = np.minimum(over1, 15).astype(np.uint8)
    tier1_raw = pack_u4_pairs(u4_tier1) if u4_tier1.size > 0 else b""

    tier2_u16 = (over1[over1 >= 15] - 15).astype(np.uint16)
    tier2_raw = tier2_u16.tobytes()

    scales_raw = rf_f16.tobytes() + cf_f16.tobytes()

    hdr = struct.pack(
        "<BIIIIIII",
        MODE_DIRECT_ALS_5BIT,
        rows,
        cols,
        int(u4_tier1.size),
        len(scales_raw),
        len(base5_raw),
        len(tier1_raw),
        len(tier2_raw),
    )
    return hdr + scales_raw + base5_raw + tier1_raw + tier2_raw


def _unpack_zero_decode_parts(blob: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vraća sirove (q_int16, row_scale, col_scale) iz MODE_DIRECT_MMAP_4BIT ili MODE_DIRECT_ALS_5BIT."""
    mode = blob[0]
    if mode == MODE_DIRECT_ALS_5BIT:
        hdr_size = struct.calcsize("<BIIIIIII")
        _, rows, cols, n_tier1, len_scales, len_base5, len_tier1, len_tier2 = struct.unpack(
            "<BIIIIIII", blob[:hdr_size]
        )
        if rows > MAX_ALLOWED_DIM or cols > MAX_ALLOWED_DIM or rows * cols * 2 > MAX_ALLOWED_TENSOR_BYTES:
            raise ValueError(f"Bezbednosna greška: nevalidne dimenzije tenzora ({rows}x{cols})")

        pos = hdr_size
        rf_f16 = np.frombuffer(blob[pos : pos + rows * 2], dtype=np.float16)
        pos += rows * 2
        cf_f16 = np.frombuffer(blob[pos : pos + cols * 2], dtype=np.float16)
        pos += cols * 2

        row_scale = rf_f16.astype(np.float32)
        col_scale = cf_f16.astype(np.float32)

        total = rows * cols
        rec_zz = unpack_u5_values(blob[pos : pos + len_base5], total).astype(np.uint16)
        pos += len_base5

        if n_tier1 > 0:
            t1 = unpack_u4_pairs(blob[pos : pos + len_tier1], (n_tier1,)).astype(np.uint16)
            pos += len_tier1
            if len_tier2 > 0:
                t2 = np.frombuffer(blob[pos : pos + len_tier2], dtype=np.uint16)
                t1[t1 == 15] = t2 + np.uint16(15)
            rec_zz[rec_zz == 31] = t1 + np.uint16(31)

        zz_2d = rec_zz.reshape((rows, cols))
        q_int16 = ((zz_2d >> 1).astype(np.int32) ^ -(zz_2d & 1).astype(np.int32)).astype(np.int16)
        return q_int16, row_scale, col_scale

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

    row_scale = bf16_u16_to_fp32(row_rms_u16) * np.float32(step_mult)
    col_scale = bf16_u16_to_fp32(col_rms_u16)

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
    return q_int16, row_scale, col_scale


def unpack_zero_decode_tensor(blob: bytes, prefer_packed_native: bool = True) -> Any:
    """
    Raspakuje Direct Zero-Decode tenzor (v1.0.0 4-bit ili v1.1.0 ALS 5-bit).
    Ako je učitan nativni dpuf_avx2 kernel i režim je 4-bit Zero-Decode, vraća
    PackedZeroDecodeWeight koji drži SAMO spakovane 4-bitne parove u RAM-u!
    """
    mode = blob[0]
    if mode == MODE_RAW_TINY:
        _, ndim, d0, d1 = struct.unpack("<BBII", blob[:10])
        arr_u16 = np.frombuffer(blob[10:], dtype=np.uint16)
        arr_f32 = bf16_u16_to_fp32(arr_u16)
        return arr_f32.reshape((d0,)) if ndim == 1 else arr_f32.reshape((d0, d1))

    if mode == MODE_DIRECT_MMAP_4BIT and prefer_packed_native and _AVX2_LIB is not None:
        hdr_size = struct.calcsize("<BIfIIIII")
        _, rows, step_mult, cols, _, len_pairs, len_over8, len_super16 = struct.unpack(
            "<BIfIIIII", blob[:hdr_size]
        )
        if rows > MAX_ALLOWED_DIM or cols > MAX_ALLOWED_DIM or rows * cols * 2 > MAX_ALLOWED_TENSOR_BYTES:
            raise ValueError(f"Bezbednosna greška: nevalidne dimenzije tenzora ({rows}x{cols})")
        pos = hdr_size
        row_rms_u16 = np.frombuffer(blob[pos : pos + rows * 2], dtype=np.uint16)
        pos += rows * 2
        col_rms_u16 = np.frombuffer(blob[pos : pos + cols * 2], dtype=np.uint16)
        pos += cols * 2
        row_scale = bf16_u16_to_fp32(row_rms_u16) * np.float32(step_mult)
        col_scale = bf16_u16_to_fp32(col_rms_u16)
        pairs = np.frombuffer(blob[pos : pos + len_pairs], dtype=np.uint8).copy()
        pos += len_pairs
        over8 = np.frombuffer(blob[pos : pos + len_over8], dtype=np.uint8).copy() if len_over8 > 0 else np.empty(0, dtype=np.uint8)
        pos += len_over8
        super16 = np.frombuffer(blob[pos : pos + len_super16], dtype=np.uint16).copy() if len_super16 > 0 else np.empty(0, dtype=np.uint16)
        return PackedZeroDecodeWeight(rows, cols, pairs, over8, super16, row_scale, col_scale)

    q_int16, row_scale, col_scale = _unpack_zero_decode_parts(blob)
    return ZeroDecodeWeight(q_int16, row_scale, col_scale)


def decode_any_dpuf_tensor_to_bf16(blob: bytes) -> np.ndarray:
    """Dekodira bilo koji DPUF tenzor (režimi 0, 2, 4, 6, 7) u uint16 BFloat16 niz (chunked radi malog RAM-a)."""
    mode = blob[0]
    if mode in (MODE_RAW_TINY, MODE_LOSSLESS_MAG_LUT):
        return decode_tensor_lossless(blob)
    if mode == MODE_NEAR_LOSSLESS_2D_RMS_BYTE:
        return decode_tensor_bfp_to_bf16(blob)
    if mode in (MODE_DIRECT_MMAP_4BIT, MODE_DIRECT_ALS_5BIT):
        q_int16, row_scale, col_scale = _unpack_zero_decode_parts(blob)
        rows, cols = q_int16.shape
        out_u16 = np.empty((rows, cols), dtype=np.uint16)
        chunk_rows = 8192 if rows > 16384 else rows
        for i0 in range(0, rows, chunk_rows):
            i1 = min(i0 + chunk_rows, rows)
            w_f32 = q_int16[i0:i1].astype(np.float32) * (row_scale[i0:i1, None] * col_scale[None, :])
            out_u16[i0:i1] = fp32_to_bf16_u16(w_f32)
        return out_u16
    if mode in ALL_QR_MODES:
        return _decode_qr_blob_to_bf16(blob)
    raise ValueError(f"Nepoznat DPUF mode: {mode}")


QR_FINDER_FMT = "<B4sBHIIIIQQQ16x"
QR_FINDER_MAGIC = b"QRDP"
QR_TILE_HDR_FMT = "<IHHII"


def _pad64(n: int) -> int:
    return (64 - (n % 64)) % 64


def _qr_tile_body_len(mode: int, n_r: int, cols: int, n_o8: int, n_s16: int) -> int:
    if mode == MODE_QR_TILED_ZD4:
        return n_r * 4 + ((n_r * cols + 1) // 2) + n_o8 + n_s16 * 2
    if mode == MODE_QR_SPLIT_INT4:
        return n_r * 4 + n_r * 2 + (n_r * cols) // 2 + n_o8 * 2 + n_s16 * 2
    if mode == MODE_QR_HADAMARD_Q4:
        return n_r * (cols // 32) * 2 + (n_r * cols) // 2
    if mode == MODE_QR_Q8:
        return n_r * (cols // 32) * 2 + (n_r * cols)
    if mode == MODE_QR_HAD4_PACKED:
        n_8g = (n_r + 7) // 8
        return n_8g * (cols // 32) * 144
    if mode == MODE_QR_HAD4_ASYM:
        n_8g = (n_r + 7) // 8
        return n_8g * (cols // 64) * 288
    if mode == MODE_QR_HAD6:
        n_8g = (n_r + 7) // 8
        return n_8g * (cols // 64) * 416
    raise ValueError(f"Unsupported QR mode: {mode}")


def _fwht64_chunk_f32(w: np.ndarray) -> np.ndarray:
    """Vectorized orthonormal Walsh-Hadamard FWHT along last dimension in blocks of 64."""
    rows, cols = w.shape
    x = w.reshape(-1, 64).copy()
    h = 1
    while h < 64:
        x = x.reshape(-1, 64 // (2 * h), 2, h)
        a = x[..., 0, :] + x[..., 1, :]
        b = x[..., 0, :] - x[..., 1, :]
        x = np.stack([a, b], axis=-2).reshape(-1, 64)
        h *= 2
    return (x * np.float32(0.125)).reshape(rows, cols)


def _decode_qr_blob_to_bf16(blob: bytes) -> np.ndarray:
    """Decodes MODE 8 / 9 / 10 QR-Tiled tensors back into BFloat16 (uint16) for WikiText-2 / GGUF export."""
    (
        mode, magic, tile_rows, ecc_group_size, rows, cols,
        n_tiles, n_ecc, aux_off, tiles_base_off, ecc_base_off,
    ) = struct.unpack(QR_FINDER_FMT, blob[:64])
    if magic != QR_FINDER_MAGIC:
        raise ValueError("Invalid QR Finder Magic!")
    pos = 64
    tile_offsets = np.frombuffer(blob[pos : pos + (n_tiles + 1) * 8], dtype=np.uint64)
    tiles_buf = memoryview(blob)[tiles_base_off:ecc_base_off]
    out_u16 = np.empty((rows, cols), dtype=np.uint16)

    if mode == MODE_QR_TILED_ZD4:
        col_scale = np.frombuffer(blob[aux_off : aux_off + cols * 4], dtype=np.float32)
        half_cols = (cols + 1) // 2
        for t in range(n_tiles):
            off = int(tile_offsets[t])
            tcrc, n_r, _, n_o8, n_s16 = struct.unpack(QR_TILE_HDR_FMT, tiles_buf[off : off + 16])
            p = off + 16
            r_scale = np.frombuffer(tiles_buf[p : p + n_r * 4], dtype=np.float32)
            p += n_r * 4
            p_bytes = (n_r * cols + 1) // 2
            dp = np.frombuffer(tiles_buf[p : p + p_bytes], dtype=np.uint8)
            p += p_bytes
            rec_zz = np.empty(dp.size * 2, dtype=np.uint16)
            rec_zz[0::2] = dp >> 4
            rec_zz[1::2] = dp & 0x0F
            rec_zz = rec_zz[: n_r * cols]
            if n_o8 > 0:
                over8 = np.frombuffer(tiles_buf[p : p + n_o8], dtype=np.uint8).astype(np.uint16)
                p += n_o8
                if n_s16 > 0:
                    super16 = np.frombuffer(tiles_buf[p : p + n_s16 * 2], dtype=np.uint16)
                    over8[over8 == 255] = super16
                rec_zz[rec_zz == 15] = over8 + np.uint16(15)
            zz_2d = rec_zz.reshape(n_r, cols)
            q_i16 = ((zz_2d >> 1).astype(np.int32) ^ -(zz_2d & 1).astype(np.int32)).astype(np.float32)
            r0 = t * tile_rows
            out_u16[r0 : r0 + n_r] = fp32_to_bf16_u16(q_i16 * (r_scale[:, None] * col_scale[None, :]))
        return out_u16

    if mode == MODE_QR_SPLIT_INT4:
        col_scale = np.frombuffer(blob[aux_off : aux_off + cols * 4], dtype=np.float32)
        n_blk = cols // 32
        for t in range(n_tiles):
            off = int(tile_offsets[t])
            tcrc, n_r, _, n_out, n_s16 = struct.unpack(QR_TILE_HDR_FMT, tiles_buf[off : off + 16])
            p = off + 16
            r_scale = np.frombuffer(tiles_buf[p : p + n_r * 4], dtype=np.float32)
            p += n_r * 4
            row_nnz = np.frombuffer(tiles_buf[p : p + n_r * 2], dtype=np.uint16)
            p += n_r * 2
            p_bytes = (n_r * cols) // 2
            pairs = np.frombuffer(tiles_buf[p : p + p_bytes], dtype=np.uint8).reshape(n_r, n_blk, 16)
            p += p_bytes
            ov_pairs = np.frombuffer(tiles_buf[p : p + n_out * 2], dtype=np.uint16)
            p += n_out * 2
            super16 = np.frombuffer(tiles_buf[p : p + n_s16 * 2], dtype=np.int16)

            q_dec = np.empty((n_r, n_blk, 32), dtype=np.int16)
            q_dec[:, :, :16] = (pairs & 0x0F).astype(np.int16) - 8
            q_dec[:, :, 16:] = (pairs >> 4).astype(np.int16) - 8
            q_2d = q_dec.reshape(n_r, cols).astype(np.float32)

            o_idx = 0
            s_idx = 0
            for lr in range(n_r):
                nnz = int(row_nnz[lr])
                if nnz:
                    col = 0
                    for _ in range(nnz):
                        pv = int(ov_pairs[o_idx])
                        o_idx += 1
                        col += pv & 0xFF
                        dv = (pv >> 8)
                        if dv >= 128:
                            dv -= 256
                        if dv == -128:
                            dv = int(super16[s_idx])
                            s_idx += 1
                        q_2d[lr, col] += float(dv)
            r0 = t * tile_rows
            out_u16[r0 : r0 + n_r] = fp32_to_bf16_u16(q_2d * (r_scale[:, None] * col_scale[None, :]))
        return out_u16

    if mode == MODE_QR_HADAMARD_Q4:
        sign_bits = np.frombuffer(blob[aux_off : aux_off + (cols // 8)], dtype=np.uint8)
        signs = np.where(np.unpackbits(sign_bits, bitorder="little")[:cols] > 0, -1.0, 1.0).astype(np.float32)
        n_blk = cols // 32
        for t in range(n_tiles):
            off = int(tile_offsets[t])
            tcrc, n_r, _, _, _ = struct.unpack(QR_TILE_HDR_FMT, tiles_buf[off : off + 16])
            p = off + 16
            sc_bytes = n_r * n_blk * 2
            sc = np.frombuffer(tiles_buf[p : p + sc_bytes], dtype=np.float16).astype(np.float32).reshape(n_r, n_blk, 1)
            p += sc_bytes
            p_bytes = (n_r * cols) // 2
            pairs = np.frombuffer(tiles_buf[p : p + p_bytes], dtype=np.uint8).reshape(n_r, n_blk, 16)
            q_dec = np.empty((n_r, n_blk, 32), dtype=np.float32)
            q_dec[:, :, :16] = (pairs & 0x0F).astype(np.float32) - 8.0
            q_dec[:, :, 16:] = (pairs >> 4).astype(np.float32) - 8.0
            w_rot = (q_dec * sc).reshape(n_r, cols)
            w_rec = _fwht64_chunk_f32(w_rot) * signs[None, :]
            r0 = t * tile_rows
            out_u16[r0 : r0 + n_r] = fp32_to_bf16_u16(w_rec)
        return out_u16

    if mode == MODE_QR_Q8:
        n_blk = cols // 32
        for t in range(n_tiles):
            off = int(tile_offsets[t])
            tcrc, n_r, _, _, _ = struct.unpack(QR_TILE_HDR_FMT, tiles_buf[off : off + 16])
            p = off + 16
            sc_bytes = n_r * n_blk * 2
            sc = np.frombuffer(tiles_buf[p : p + sc_bytes], dtype=np.float16).astype(np.float32).reshape(n_r, n_blk, 1)
            p += sc_bytes
            w_bytes = n_r * cols
            wq = np.frombuffer(tiles_buf[p : p + w_bytes], dtype=np.int8).astype(np.float32).reshape(n_r, n_blk, 32)
            r0 = t * tile_rows
            out_u16[r0 : r0 + n_r] = fp32_to_bf16_u16((wq * sc).reshape(n_r, cols))
        return out_u16

    if mode == MODE_QR_HAD4_PACKED:
        sign_bits = np.frombuffer(blob[aux_off : aux_off + (cols // 8)], dtype=np.uint8)
        signs = np.where(np.unpackbits(sign_bits, bitorder="little")[:cols] > 0, -1.0, 1.0).astype(np.float32)
        n_blk = cols // 32
        for t in range(n_tiles):
            off = int(tile_offsets[t])
            tcrc, n_r, _, _, _ = struct.unpack(QR_TILE_HDR_FMT, tiles_buf[off : off + 16])
            p = off + 16
            r0 = t * tile_rows
            n_8g = (n_r + 7) // 8
            for rg in range(n_8g):
                n_8 = min(8, n_r - rg * 8)
                grp_base = p + rg * n_blk * 144
                sc_bytes = n_blk * 16
                sc = (
                    np.frombuffer(tiles_buf[grp_base : grp_base + sc_bytes], dtype=np.float16)
                    .astype(np.float32)
                    .reshape(n_blk, 8)
                    .T[:n_8, :, None]
                )
                q_grp = np.frombuffer(
                    tiles_buf[grp_base + sc_bytes : grp_base + n_blk * 144], dtype=np.uint8
                ).reshape(n_blk, 4, 8, 4)
                lo = (q_grp[:, :, :n_8, :] & 0x0F).astype(np.float32) - 8.0
                hi = (q_grp[:, :, :n_8, :] >> 4).astype(np.float32) - 8.0
                q_dec = np.concatenate([lo, hi], axis=3).transpose(2, 0, 1, 3).reshape(n_8, n_blk, 32)
                w_rot = (q_dec * sc).reshape(n_8, cols)
                w_rec = _fwht64_chunk_f32(w_rot) * signs[None, :]
                rr0 = r0 + rg * 8
                out_u16[rr0 : rr0 + n_8] = fp32_to_bf16_u16(w_rec)
        return out_u16

    if mode in (MODE_QR_HAD4_ASYM, MODE_QR_HAD6):
        is_had6 = mode == MODE_QR_HAD6
        sign_bits = np.frombuffer(blob[aux_off : aux_off + (cols // 8)], dtype=np.uint8)
        signs = np.where(np.unpackbits(sign_bits, bitorder="little")[:cols] > 0, -1.0, 1.0).astype(np.float32)
        n_64 = cols // 64
        n_blk = n_64 * 2
        grp_bytes = 416 if is_had6 else 288
        for t in range(n_tiles):
            off = int(tile_offsets[t])
            tcrc, n_r, _, _, _ = struct.unpack(QR_TILE_HDR_FMT, tiles_buf[off : off + 16])
            p = off + 16
            r0 = t * tile_rows
            n_8g = (n_r + 7) // 8
            for rg in range(n_8g):
                n_8 = min(8, n_r - rg * 8)
                grp_base = p + rg * n_64 * grp_bytes
                dm_bytes = n_64 * 32
                dm_grp = (
                    np.frombuffer(tiles_buf[grp_base : grp_base + dm_bytes], dtype=np.float16)
                    .astype(np.float32)
                    .reshape(n_64, 16)
                )
                d = dm_grp[:, :8].T[:n_8, :, None]
                m = dm_grp[:, 8:].T[:n_8, :, None]
                q4_off = grp_base + dm_bytes
                q4_bytes = n_64 * 256
                q4_grp = np.frombuffer(tiles_buf[q4_off : q4_off + q4_bytes], dtype=np.uint8).reshape(n_blk, 4, 8, 4)
                lo = (q4_grp[:, :, :n_8, :] & 0x0F).astype(np.int32)
                hi = (q4_grp[:, :, :n_8, :] >> 4).astype(np.int32)
                q_dec = np.concatenate([lo, hi], axis=3).transpose(2, 0, 1, 3).reshape(n_8, n_blk, 32)
                if is_had6:
                    q2_off = q4_off + q4_bytes
                    q2_grp = np.frombuffer(tiles_buf[q2_off : q2_off + n_64 * 128], dtype=np.uint8).reshape(
                        n_blk, 4, 2, 2, 4
                    )
                    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)[:, None, None, None, None, None]
                    bits = ((q2_grp[None, ...] >> shifts) & 0x03).transpose(0, 4, 1, 2, 3, 5).reshape(8, n_blk, 32)
                    q_dec = q_dec | (bits[:n_8].astype(np.int32) << 4)
                w_rot = (q_dec.reshape(n_8, n_64, 64).astype(np.float32) * d + m).reshape(n_8, cols)
                w_rec = _fwht64_chunk_f32(w_rot) * signs[None, :]
                rr0 = r0 + rg * 8
                out_u16[rr0 : rr0 + n_8] = fp32_to_bf16_u16(w_rec)
        return out_u16

    raise ValueError(f"Unsupported QR mode: {mode}")


def _assemble_qr_blob(
    mode: int,
    tile_rows: int,
    ecc_group_size: int,
    rows: int,
    cols: int,
    aux_bytes: bytes,
    tile_chunks: list,
    tile_offsets: np.ndarray,
) -> bytes:
    """Shared QR Finder + 64B Alignment + RAID-XOR Parity ECC builder for Modes 8, 9, 10."""
    n_tiles = len(tile_chunks)
    ecc_chunks = []
    n_ecc = (n_tiles + ecc_group_size - 1) // ecc_group_size if ecc_group_size > 0 else 0
    ecc_offsets = np.empty(n_ecc + 1, dtype=np.uint64)
    cur_ecc_off = 0
    for g in range(n_ecc):
        ecc_offsets[g] = cur_ecc_off
        g0 = g * ecc_group_size
        g1 = min(n_tiles, g0 + ecc_group_size)
        max_len = max(len(tile_chunks[i]) for i in range(g0, g1))
        acc = np.zeros(max_len // 8, dtype=np.uint64)
        for i in range(g0, g1):
            tc = tile_chunks[i]
            n_u64 = len(tc) // 8
            acc[:n_u64] ^= np.frombuffer(tc, dtype=np.uint64)
        p_bytes = acc.tobytes()
        ecc_chunks.append(p_bytes)
        cur_ecc_off += len(p_bytes)
    ecc_offsets[n_ecc] = cur_ecc_off

    hdr_size = 64
    t_off_bytes = tile_offsets.tobytes()
    e_off_bytes = ecc_offsets.tobytes()
    pos_after_tables = hdr_size + len(t_off_bytes) + len(e_off_bytes)
    pad_aux = b"\x00" * _pad64(pos_after_tables)
    aux_off = pos_after_tables + len(pad_aux)

    pos_after_aux = aux_off + len(aux_bytes)
    pad_tiles = b"\x00" * _pad64(pos_after_aux)
    tiles_base_off = pos_after_aux + len(pad_tiles)

    tiles_payload = b"".join(tile_chunks)
    ecc_base_off = tiles_base_off + len(tiles_payload)
    ecc_payload = b"".join(ecc_chunks)

    finder_hdr = struct.pack(
        QR_FINDER_FMT,
        mode,
        QR_FINDER_MAGIC,
        tile_rows,
        ecc_group_size,
        rows,
        cols,
        n_tiles,
        n_ecc,
        aux_off,
        tiles_base_off,
        ecc_base_off,
    )
    return (
        finder_hdr
        + t_off_bytes
        + e_off_bytes
        + pad_aux
        + aux_bytes
        + pad_tiles
        + tiles_payload
        + ecc_payload
    )


def encode_tensor_qr_zd4(
    W_u16: np.ndarray,
    step_mult: float = 0.255,
    is_embed: bool = False,
    tile_rows: int = 16,
    ecc_group_size: int = 64,
) -> bytes:
    """
    QR-Inspired Format (MODE_QR_TILED_ZD4 = 8):
      1. Binary Finder Header (64B aligned): magic 'QRDP', dimensions, tile_offsets, ecc_offsets
      2. Self-Contained 16-row QR Tiles (each 64B aligned):
         [16B TileHdr (CRC32 + n_rows + n_over8 + n_super16)][float32 row_scale][pairs][over8][super16][pad64]
      3. XOR ECC Parity Tiles (1 parity tile per ecc_group_size tiles) for Self-Healing disk/network corruption!
    """
    if W_u16.ndim != 2 or min(W_u16.shape) < 64:
        hdr = struct.pack(
            "<BBII",
            MODE_RAW_TINY,
            W_u16.ndim,
            W_u16.shape[0],
            W_u16.shape[1] if W_u16.ndim == 2 else 1,
        )
        return hdr + W_u16.tobytes()

    rows, cols = W_u16.shape
    eff_step = float(step_mult * (0.55 if is_embed else 1.0))
    row_rms_u16, col_rms_u16, zz = _quantize_2d_rms_chunked(W_u16, eff_step)
    row_scale = (bf16_u16_to_fp32(row_rms_u16) * np.float32(eff_step)).astype(np.float32)
    col_scale = bf16_u16_to_fp32(col_rms_u16).astype(np.float32)

    n_tiles = (rows + tile_rows - 1) // tile_rows
    tile_chunks = []
    tile_offsets = np.empty(n_tiles + 1, dtype=np.uint64)
    cur_off = 0

    for t in range(n_tiles):
        tile_offsets[t] = cur_off
        r0 = t * tile_rows
        r1 = min(rows, r0 + tile_rows)
        n_r = r1 - r0

        zz_t = zz[r0:r1].ravel()
        u4 = np.minimum(zz_t, 15).astype(np.uint8)
        pairs_raw = pack_u4_pairs(u4)

        over_mask = zz_t >= 15
        if np.any(over_mask):
            over_u16 = (zz_t[over_mask] - 15).astype(np.uint16)
            over8 = np.minimum(over_u16, 255).astype(np.uint8)
            super16 = over_u16[over_u16 >= 255]
            over8_raw = over8.tobytes()
            super16_raw = super16.tobytes()
            n_o8, n_s16 = over8.size, super16.size
        else:
            over8_raw, super16_raw = b"", b""
            n_o8, n_s16 = 0, 0

        body = row_scale[r0:r1].tobytes() + pairs_raw + over8_raw + super16_raw
        tcrc = zlib.crc32(body) & 0xFFFFFFFF
        thdr = struct.pack(QR_TILE_HDR_FMT, tcrc, n_r, 0, n_o8, n_s16)
        raw_t = thdr + body
        padded_t = raw_t + b"\x00" * _pad64(len(raw_t))
        tile_chunks.append(padded_t)
        cur_off += len(padded_t)

    tile_offsets[n_tiles] = cur_off
    return _assemble_qr_blob(
        MODE_QR_TILED_ZD4, tile_rows, ecc_group_size, rows, cols, col_scale.tobytes(), tile_chunks, tile_offsets
    )


def encode_tensor_qr_split4(
    W_u16: np.ndarray,
    step_mult: float = 0.255,
    is_embed: bool = False,
    embed_mult: float = 0.55,
    tile_rows: int = 16,
    ecc_group_size: int = 64,
) -> bytes:
    """
    Varijanta A (MODE_QR_SPLIT_INT4 = 9): Branchless 4-bit Base [-8..+7] + Sparse Gap-Delta CSR Outliers!
    Mathematically identical weight accuracy to MODE_QR_TILED_ZD4, zero if(esc) pipeline stalls!
    """
    if W_u16.ndim != 2 or min(W_u16.shape) < 64 or (W_u16.shape[1] % 32) != 0:
        return encode_tensor_qr_zd4(W_u16, step_mult=step_mult, is_embed=is_embed, tile_rows=tile_rows, ecc_group_size=ecc_group_size)

    rows, cols = W_u16.shape
    eff_step = float(step_mult * (embed_mult if is_embed else 1.0))
    row_rms_u16, col_rms_u16, zz = _quantize_2d_rms_chunked(W_u16, eff_step)
    row_scale = (bf16_u16_to_fp32(row_rms_u16) * np.float32(eff_step)).astype(np.float32)
    col_scale = bf16_u16_to_fp32(col_rms_u16).astype(np.float32)

    n_blk = cols // 32
    n_tiles = (rows + tile_rows - 1) // tile_rows
    tile_chunks = []
    tile_offsets = np.empty(n_tiles + 1, dtype=np.uint64)
    cur_off = 0

    for t in range(n_tiles):
        tile_offsets[t] = cur_off
        r0 = t * tile_rows
        r1 = min(rows, r0 + tile_rows)
        n_r = r1 - r0

        zz_t = zz[r0:r1]
        q_t = ((zz_t >> 1).astype(np.int32) ^ -(zz_t & 1).astype(np.int32)).astype(np.int16)
        base_q = np.clip(q_t, -8, 7)
        u4 = (base_q + 8).astype(np.uint8).reshape(n_r, n_blk, 32)
        pairs_raw = ((u4[:, :, 16:] << 4) | u4[:, :, :16]).tobytes()

        delta = q_t - base_q
        row_nnz = np.empty(n_r, dtype=np.uint16)
        ov_parts = []
        s16_parts = []

        if np.any(delta):
            for lr in range(n_r):
                d_row = delta[lr]
                nz_c = np.flatnonzero(d_row)
                if nz_c.size == 0:
                    row_nnz[lr] = 0
                    continue
                vals = d_row[nz_c]
                diffs = np.diff(nz_c, prepend=0)
                if int(diffs.max()) < 255 and int(vals.min()) >= -127 and int(vals.max()) <= 127:
                    u_v = vals.astype(np.int8).view(np.uint8).astype(np.uint16)
                    packed = ((u_v << 8) | diffs.astype(np.uint16)).astype(np.uint16)
                    ov_parts.append(packed.tobytes())
                    row_nnz[lr] = packed.size
                else:
                    row_ov = []
                    row_s = []
                    for gap_i, v_i in zip(diffs.tolist(), vals.tolist()):
                        g = gap_i
                        while g >= 255:
                            row_ov.append(255)  # gap=255, dv=0
                            g -= 255
                        if -127 <= v_i <= 127:
                            u8_v = v_i & 0xFF
                        else:
                            u8_v = 0x80  # -128 -> escape to super16
                            row_s.append(v_i)
                        row_ov.append((u8_v << 8) | g)
                    ov_arr = np.asarray(row_ov, dtype=np.uint16)
                    ov_parts.append(ov_arr.tobytes())
                    if row_s:
                        s16_parts.append(np.asarray(row_s, dtype=np.int16).tobytes())
                    row_nnz[lr] = ov_arr.size
        else:
            row_nnz.fill(0)

        ov_raw = b"".join(ov_parts)
        s16_raw = b"".join(s16_parts)
        n_out = len(ov_raw) // 2
        n_s16 = len(s16_raw) // 2

        body = row_scale[r0:r1].tobytes() + row_nnz.tobytes() + pairs_raw + ov_raw + s16_raw
        tcrc = zlib.crc32(body) & 0xFFFFFFFF
        thdr = struct.pack(QR_TILE_HDR_FMT, tcrc, n_r, 0, n_out, n_s16)
        raw_t = thdr + body
        padded_t = raw_t + b"\x00" * _pad64(len(raw_t))
        tile_chunks.append(padded_t)
        cur_off += len(padded_t)

    tile_offsets[n_tiles] = cur_off
    return _assemble_qr_blob(
        MODE_QR_SPLIT_INT4, tile_rows, ecc_group_size, rows, cols, col_scale.tobytes(), tile_chunks, tile_offsets
    )


def _make_had_sign_bits(cols: int, tensor_name: str):
    seed = zlib.crc32(tensor_name.encode("utf-8")) ^ 0x9E3779B9
    rng = np.random.default_rng(seed)
    sign_flag = rng.integers(0, 2, size=cols, dtype=np.uint8)
    signs = np.where(sign_flag > 0, -1.0, 1.0).astype(np.float32)
    sign_bits_raw = np.packbits(sign_flag, bitorder="little").tobytes()
    return signs, sign_bits_raw


def _encode_qr_hadamard4_common(
    W_u16: np.ndarray,
    tile_rows: int = 16,
    ecc_group_size: int = 64,
    tensor_name: str = "",
    packed_layout: bool = False,
) -> bytes:
    """Shared FWHT-64 + MSE Blk32 [-8..+7] encoder for Mode 10 (row-major) & Mode 12 (8x32 packed)."""
    if W_u16.ndim != 2 or min(W_u16.shape) < 64 or (W_u16.shape[1] % 64) != 0:
        return encode_tensor_qr_zd4(W_u16, tile_rows=tile_rows, ecc_group_size=ecc_group_size)

    rows, cols = W_u16.shape
    signs, sign_bits_raw = _make_had_sign_bits(cols, tensor_name)

    n_blk = cols // 32
    n_tiles = (rows + tile_rows - 1) // tile_rows
    tile_chunks = []
    tile_offsets = np.empty(n_tiles + 1, dtype=np.uint64)
    cur_off = 0

    chunk_tiles = 256
    for t0 in range(0, n_tiles, chunk_tiles):
        t1 = min(n_tiles, t0 + chunk_tiles)
        r0_c = t0 * tile_rows
        r1_c = min(rows, t1 * tile_rows)
        wf_c = bf16_u16_to_fp32(W_u16[r0_c:r1_c]) * signs[None, :]
        wh_c = _fwht64_chunk_f32(wf_c).reshape(-1, n_blk, 32)
        del wf_c

        amax = np.max(np.abs(wh_c), axis=2, keepdims=True)
        base_s = (amax / np.float32(7.5)) + np.float32(1e-12)
        best_s = base_s.astype(np.float16)
        best_err = np.full(amax.shape[:2], np.inf, dtype=np.float32)
        for mult in (0.78, 0.85, 0.92, 1.00):
            cand_s = (base_s * np.float32(mult)).astype(np.float16)
            cand_f = cand_s.astype(np.float32)
            safe_f = np.where(cand_f > 0.0, cand_f, np.float32(1.0))
            q_cand = np.clip(np.round(wh_c / safe_f), -8.0, 7.0)
            err = np.sum((wh_c - q_cand * cand_f) ** 2, axis=2)
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_s = np.where(better[..., None], cand_s, best_s)

        scales_f16 = best_s[..., 0]
        sc_f32 = scales_f16.astype(np.float32)
        safe_sc = np.where(sc_f32 > 0.0, sc_f32, np.float32(1.0))[..., None]
        q_int = np.clip(np.round(wh_c / safe_sc), -8, 7).astype(np.int8)
        u4_chunk = (q_int + 8).astype(np.uint8)

        if not packed_layout:
            pairs_chunk = ((u4_chunk[:, :, 16:] << 4) | u4_chunk[:, :, :16])
            for t in range(t0, t1):
                tile_offsets[t] = cur_off
                lr0 = (t - t0) * tile_rows
                lr1 = min(scales_f16.shape[0], lr0 + tile_rows)
                n_r = lr1 - lr0
                body = scales_f16[lr0:lr1].tobytes() + pairs_chunk[lr0:lr1].tobytes()
                tcrc = zlib.crc32(body) & 0xFFFFFFFF
                thdr = struct.pack(QR_TILE_HDR_FMT, tcrc, n_r, 0, 0, 0)
                raw_t = thdr + body
                padded_t = raw_t + b"\x00" * _pad64(len(raw_t))
                tile_chunks.append(padded_t)
                cur_off += len(padded_t)
        else:
            for t in range(t0, t1):
                tile_offsets[t] = cur_off
                lr0 = (t - t0) * tile_rows
                lr1 = min(scales_f16.shape[0], lr0 + tile_rows)
                n_r = lr1 - lr0
                n_8g = (n_r + 7) // 8
                r_pad = n_8g * 8

                sc_t = scales_f16[lr0:lr1]
                u4_t = u4_chunk[lr0:lr1]
                if n_r < r_pad:
                    sc_p = np.zeros((r_pad, n_blk), dtype=np.float16)
                    sc_p[:n_r] = sc_t
                    u4_p = np.full((r_pad, n_blk, 32), 8, dtype=np.uint8)
                    u4_p[:n_r] = u4_t
                else:
                    sc_p, u4_p = sc_t, u4_t

                sc_grp = sc_p.reshape(n_8g, 8, n_blk).transpose(0, 2, 1)
                u4_4d = u4_p.reshape(n_8g, 8, n_blk, 4, 2, 4)
                packed_5d = ((u4_4d[..., 1, :] << 4) | u4_4d[..., 0, :])  # (n_8g, 8, n_blk, c=4, k=4)
                q_grp = packed_5d.transpose(0, 2, 3, 1, 4)                # (n_8g, n_blk, c=4, ri=8, k=4)

                parts = []
                for rg in range(n_8g):
                    parts.append(sc_grp[rg].tobytes())
                    parts.append(q_grp[rg].tobytes())
                body = b"".join(parts)
                tcrc = zlib.crc32(body) & 0xFFFFFFFF
                thdr = struct.pack(QR_TILE_HDR_FMT, tcrc, n_r, 0, 0, 0)
                raw_t = thdr + body
                padded_t = raw_t + b"\x00" * _pad64(len(raw_t))
                tile_chunks.append(padded_t)
                cur_off += len(padded_t)

    tile_offsets[n_tiles] = cur_off
    out_mode = MODE_QR_HAD4_PACKED if packed_layout else MODE_QR_HADAMARD_Q4
    return _assemble_qr_blob(
        out_mode, tile_rows, ecc_group_size, rows, cols, sign_bits_raw, tile_chunks, tile_offsets
    )


def encode_tensor_qr_hadamard4(
    W_u16: np.ndarray,
    tile_rows: int = 16,
    ecc_group_size: int = 64,
    tensor_name: str = "",
) -> bytes:
    """
    Varijanta B (MODE_QR_HADAMARD_Q4 = 10):
      Walsh-Hadamard FWHT-64 + MSE-tuned Block-32 Symmetric 4-bit [-8..+7] (ZERO outliers!)
    """
    return _encode_qr_hadamard4_common(
        W_u16, tile_rows=tile_rows, ecc_group_size=ecc_group_size, tensor_name=tensor_name, packed_layout=False
    )


def encode_tensor_qr_hadamard4_packed(
    W_u16: np.ndarray,
    tile_rows: int = 16,
    ecc_group_size: int = 64,
    tensor_name: str = "",
) -> bytes:
    """
    MODE_QR_HAD4_PACKED = 12: Pre-Packed 8x32 QR Hadamard4 (Zero Transpose, Zero wsum128, L2-Tiled).
    """
    return _encode_qr_hadamard4_common(
        W_u16, tile_rows=tile_rows, ecc_group_size=ecc_group_size, tensor_name=tensor_name, packed_layout=True
    )


def _encode_qr_had_asym_common(
    W_u16: np.ndarray,
    bits: int,
    w_pow: float = 0.5,
    lsq_iters: int = 2,
    tile_rows: int = 16,
    ecc_group_size: int = 64,
    tensor_name: str = "",
) -> bytes:
    """Shared row-chunked FWHT-64 + weighted LSQ blk64 asymmetric (d, m) encoder for Mode 13 (4-bit) & Mode 14 (6-bit)."""
    if bits not in (4, 6):
        raise ValueError(f"Unsupported asym bit width: {bits}")
    if W_u16.ndim != 2 or min(W_u16.shape) < 64 or (W_u16.shape[1] % 64) != 0:
        return encode_tensor_qr_zd4(W_u16, tile_rows=tile_rows, ecc_group_size=ecc_group_size)

    rows, cols = W_u16.shape
    signs, sign_bits_raw = _make_had_sign_bits(cols, tensor_name)
    n_64 = cols // 64
    n_blk = n_64 * 2
    n_tiles = (rows + tile_rows - 1) // tile_rows
    qmax = (1 << bits) - 1

    tile_chunks = []
    tile_offsets = np.empty(n_tiles + 1, dtype=np.uint64)
    cur_off = 0
    chunk_tiles = 128 if cols >= 2048 else 256

    for t0 in range(0, n_tiles, chunk_tiles):
        t1 = min(n_tiles, t0 + chunk_tiles)
        r0_c = t0 * tile_rows
        r1_c = min(rows, t1 * tile_rows)
        n_rc = r1_c - r0_c
        wf_c = bf16_u16_to_fp32(W_u16[r0_c:r1_c]) * signs[None, :]
        wb = _fwht64_chunk_f32(wf_c).reshape(-1, 64)
        del wf_c

        wt = np.power(np.abs(wb) + np.float32(1e-6), np.float32(w_pow)).astype(np.float32) if w_pow > 0.0 else None
        wmin = wb.min(axis=1)
        wmax = wb.max(axis=1)
        best_err = np.full(wb.shape[0], np.inf, dtype=np.float32)
        best_d16 = np.zeros(wb.shape[0], dtype=np.float16)
        best_m16 = np.zeros(wb.shape[0], dtype=np.float16)
        best_q = np.zeros(wb.shape, dtype=np.uint8)

        for shrink in (0.90, 0.96, 1.00):
            mid = np.float32(0.5) * (wmin + wmax)
            half = np.float32(0.5) * (wmax - wmin) * np.float32(shrink)
            lo = mid - half
            d = np.where(half > np.float32(1e-12), (np.float32(2.0) * half) / np.float32(qmax), np.float32(0.0))

            for _ in range(lsq_iters):
                safe_d = np.where(d > np.float32(1e-12), d, np.float32(1.0))
                qf = np.clip(np.round((wb - lo[:, None]) / safe_d[:, None]), np.float32(0.0), np.float32(qmax))
                if wt is not None:
                    s0 = np.sum(wt, axis=1)
                    sq = np.sum(wt * qf, axis=1)
                    sq2 = np.sum(wt * qf * qf, axis=1)
                    sw = np.sum(wt * wb, axis=1)
                    sqw = np.sum(wt * qf * wb, axis=1)
                else:
                    s0 = np.full(wb.shape[0], np.float32(64.0), dtype=np.float32)
                    sq = np.sum(qf, axis=1)
                    sq2 = np.sum(qf * qf, axis=1)
                    sw = np.sum(wb, axis=1)
                    sqw = np.sum(qf * wb, axis=1)
                det = s0 * sq2 - sq * sq
                valid = det > np.float32(1e-12)
                safe_det = np.where(valid, det, np.float32(1.0))
                d_new = np.where(valid, (s0 * sqw - sq * sw) / safe_det, d)
                lo_new = np.where(valid, (sw - d_new * sq) / np.where(valid, s0, np.float32(1.0)), lo)
                use_new = valid & (d_new > np.float32(1e-12))
                d = np.where(use_new, d_new, d)
                lo = np.where(use_new, lo_new, lo)

            m16 = lo.astype(np.float16)
            d16 = np.maximum(d, np.float32(0.0)).astype(np.float16)
            m32 = m16.astype(np.float32)[:, None]
            d32 = d16.astype(np.float32)[:, None]
            safe_df = np.where(d32 > np.float32(0.0), d32, np.float32(1.0))
            qc = np.clip(np.round((wb - m32) / safe_df), 0, qmax).astype(np.uint8)
            qc = np.where(d32 > np.float32(0.0), qc, np.uint8(0))
            rec = m32 + qc.astype(np.float32) * d32
            err = np.sum((rec - wb) ** 2, axis=1)

            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_d16 = np.where(better, d16, best_d16)
            best_m16 = np.where(better, m16, best_m16)
            best_q = np.where(better[:, None], qc, best_q)

        d_f16 = best_d16.reshape(n_rc, n_64)
        m_f16 = best_m16.reshape(n_rc, n_64)
        q_chunk = best_q.reshape(n_rc, n_64, 64)
        del wb, wt

        for t in range(t0, t1):
            tile_offsets[t] = cur_off
            lr0 = (t - t0) * tile_rows
            lr1 = min(d_f16.shape[0], lr0 + tile_rows)
            n_r = lr1 - lr0
            n_8g = (n_r + 7) // 8
            r_pad = n_8g * 8

            if n_r < r_pad:
                d_p = np.zeros((r_pad, n_64), dtype=np.float16)
                m_p = np.zeros((r_pad, n_64), dtype=np.float16)
                q_p = np.zeros((r_pad, n_64, 64), dtype=np.uint8)
                d_p[:n_r] = d_f16[lr0:lr1]
                m_p[:n_r] = m_f16[lr0:lr1]
                q_p[:n_r] = q_chunk[lr0:lr1]
            else:
                d_p = d_f16[lr0:lr1]
                m_p = m_f16[lr0:lr1]
                q_p = q_chunk[lr0:lr1]

            dm_grp = np.empty((n_8g, n_64, 16), dtype=np.float16)
            dm_grp[:, :, :8] = d_p.reshape(n_8g, 8, n_64).transpose(0, 2, 1)
            dm_grp[:, :, 8:] = m_p.reshape(n_8g, 8, n_64).transpose(0, 2, 1)

            q4_4d = (q_p & np.uint8(0x0F)).reshape(n_8g, 8, n_blk, 4, 2, 4)
            q4_grp = ((q4_4d[..., 1, :] << np.uint8(4)) | q4_4d[..., 0, :]).transpose(0, 2, 3, 1, 4)

            if bits == 6:
                q2_raw = ((q_p >> np.uint8(4)) & np.uint8(0x03)).reshape(n_8g, 4, 2, n_blk, 4, 2, 4)
                q2_t = q2_raw.transpose(0, 1, 3, 4, 5, 2, 6)
                q2_grp = (
                    q2_t[:, 0]
                    | (q2_t[:, 1] << np.uint8(2))
                    | (q2_t[:, 2] << np.uint8(4))
                    | (q2_t[:, 3] << np.uint8(6))
                )

            parts = []
            for rg in range(n_8g):
                parts.append(dm_grp[rg].tobytes())
                parts.append(q4_grp[rg].tobytes())
                if bits == 6:
                    parts.append(q2_grp[rg].tobytes())
            body = b"".join(parts)
            tcrc = zlib.crc32(body) & 0xFFFFFFFF
            thdr = struct.pack(QR_TILE_HDR_FMT, tcrc, n_r, 0, 0, 0)
            raw_t = thdr + body
            padded_t = raw_t + b"\x00" * _pad64(len(raw_t))
            tile_chunks.append(padded_t)
            cur_off += len(padded_t)

    tile_offsets[n_tiles] = cur_off
    out_mode = MODE_QR_HAD6 if bits == 6 else MODE_QR_HAD4_ASYM
    return _assemble_qr_blob(
        out_mode, tile_rows, ecc_group_size, rows, cols, sign_bits_raw, tile_chunks, tile_offsets
    )


def encode_tensor_qr_had4_asym(
    W_u16: np.ndarray,
    tile_rows: int = 16,
    ecc_group_size: int = 64,
    tensor_name: str = "",
    w_pow: float = 0.5,
) -> bytes:
    """MODE_QR_HAD4_ASYM = 13: Pre-Packed 8x64 FWHT-64 + Weighted LSQ Asymmetric blk64 4-bit."""
    return _encode_qr_had_asym_common(
        W_u16, bits=4, w_pow=w_pow, lsq_iters=2,
        tile_rows=tile_rows, ecc_group_size=ecc_group_size, tensor_name=tensor_name,
    )


def encode_tensor_qr_had6(
    W_u16: np.ndarray,
    tile_rows: int = 16,
    ecc_group_size: int = 64,
    tensor_name: str = "",
    w_pow: float = 0.0,
) -> bytes:
    """MODE_QR_HAD6 = 14: Pre-Packed 8x64 FWHT-64 + Weighted LSQ Asymmetric blk64 6-bit."""
    return _encode_qr_had_asym_common(
        W_u16, bits=6, w_pow=w_pow, lsq_iters=2,
        tile_rows=tile_rows, ecc_group_size=ecc_group_size, tensor_name=tensor_name,
    )


def encode_tensor_qr_q8(
    W_u16: np.ndarray,
    tile_rows: int = 16,
    ecc_group_size: int = 64,
) -> bytes:
    """
    MODE_QR_Q8 = 11: QR-Tiled Pure Dense Blk32 Symmetric int8 [-127..127] (8.5 b/w, 0% outliers).
    High-precision embedding / LM-Head codec that keeps speed via VPMADDUBSW and beats Q4_K_M PPL.
    """
    if W_u16.ndim != 2 or (W_u16.shape[1] % 32) != 0:
        return encode_tensor_qr_zd4(W_u16, tile_rows=tile_rows, ecc_group_size=ecc_group_size)

    rows, cols = W_u16.shape
    n_blk = cols // 32
    n_tiles = (rows + tile_rows - 1) // tile_rows
    tile_chunks = []
    tile_offsets = np.empty(n_tiles + 1, dtype=np.uint64)
    cur_off = 0

    chunk_tiles = 256
    for t0 in range(0, n_tiles, chunk_tiles):
        t1 = min(n_tiles, t0 + chunk_tiles)
        r0_c = t0 * tile_rows
        r1_c = min(rows, t1 * tile_rows)
        wf_c = bf16_u16_to_fp32(W_u16[r0_c:r1_c]).reshape(-1, n_blk, 32)
        amax = np.max(np.abs(wf_c), axis=2, keepdims=True)
        scales_f16 = (amax / np.float32(127.0)).astype(np.float16)
        scales_f32 = scales_f16.astype(np.float32)
        safe_s = np.where(scales_f32 > 0.0, scales_f32, np.float32(1.0))
        q_int = np.clip(np.round(wf_c / safe_s), -127, 127).astype(np.int8)
        del wf_c, scales_f32, safe_s

        sc_2d = scales_f16[..., 0]
        q_2d = q_int.reshape(-1, cols)
        for t in range(t0, t1):
            tile_offsets[t] = cur_off
            lr0 = (t - t0) * tile_rows
            lr1 = min(sc_2d.shape[0], lr0 + tile_rows)
            n_r = lr1 - lr0
            body = sc_2d[lr0:lr1].tobytes() + q_2d[lr0:lr1].tobytes()
            tcrc = zlib.crc32(body) & 0xFFFFFFFF
            thdr = struct.pack(QR_TILE_HDR_FMT, tcrc, n_r, 0, 0, 0)
            raw_t = thdr + body
            padded_t = raw_t + b"\x00" * _pad64(len(raw_t))
            tile_chunks.append(padded_t)
            cur_off += len(padded_t)

    tile_offsets[n_tiles] = cur_off
    return _assemble_qr_blob(
        MODE_QR_Q8, tile_rows, ecc_group_size, rows, cols, b"", tile_chunks, tile_offsets
    )


class QRTiledZeroDecodeWeight:
    """
    Zero-Copy Memory-Mapped AVX2 objekat za QR-Tiled režime (Mode 8 / 9 / 10 / 11).
    Pokazuje direktno na mmap stranice OS-a bez malloc kopiranja u RAM i bez indeksiranja redova!
    """

    def __init__(self, buf_view: np.ndarray, verify_crc: bool = True, auto_heal: bool = True):
        if _AVX2_LIB is None or not hasattr(_AVX2_LIB, "dpuf_matmul_qr_zd4_f32"):
            raise RuntimeError("Potreban je dpuf_avx2.dll sa QR-Tiled podrškom.")

        hdr_bytes = buf_view[:64].tobytes()
        (
            mode,
            magic,
            tile_rows,
            ecc_group_size,
            rows,
            cols,
            n_tiles,
            n_ecc,
            aux_off,
            tiles_base_off,
            ecc_base_off,
        ) = struct.unpack(QR_FINDER_FMT, hdr_bytes)

        if mode not in ALL_QR_MODES or magic != QR_FINDER_MAGIC:
            raise ValueError(f"Nevažeći QR Finder Header (mode={mode}, magic={magic!r})!")

        self.mode = int(mode)
        self.rows = int(rows)
        self.cols = int(cols)
        self.tile_rows = int(tile_rows)
        self.ecc_group_size = int(ecc_group_size)
        self.n_tiles = int(n_tiles)
        self.n_ecc = int(n_ecc)
        self.healed_tiles = 0

        pos = 64
        t_off_len = (self.n_tiles + 1) * 8
        self.tile_offsets = buf_view[pos : pos + t_off_len].view(np.uint64)
        pos += t_off_len
        e_off_len = (self.n_ecc + 1) * 8
        self.ecc_offsets = buf_view[pos : pos + e_off_len].view(np.uint64)

        self._buf = buf_view
        if self.mode in (MODE_QR_HADAMARD_Q4, MODE_QR_HAD4_PACKED, MODE_QR_HAD4_ASYM, MODE_QR_HAD6):
            self.col_scale = None
            self.sign_bits = buf_view[aux_off : aux_off + (cols // 8)]
        elif self.mode == MODE_QR_Q8:
            self.col_scale = None
            self.sign_bits = None
        else:
            self.col_scale = buf_view[aux_off : aux_off + cols * 4].view(np.float32)
            self.sign_bits = None

        self.tiles_base_off = int(tiles_base_off)
        self.tiles_view = buf_view[tiles_base_off:ecc_base_off]
        self.ecc_view = buf_view[ecc_base_off:]

        if verify_crc:
            tile_ok = np.empty(self.n_tiles, dtype=np.uint8)
            if hasattr(_AVX2_LIB, "dpuf_qr_verify_tiles_ex"):
                bad = _AVX2_LIB.dpuf_qr_verify_tiles_ex(
                    self.mode, self.n_tiles, self.cols, self.tiles_view, self.tile_offsets, tile_ok
                )
            else:
                bad = _AVX2_LIB.dpuf_qr_verify_tiles(
                    self.n_tiles, self.cols, self.tiles_view, self.tile_offsets, tile_ok
                )
            if bad > 0:
                if not auto_heal or self.ecc_group_size <= 0:
                    raise ValueError(f"Detektovano {bad} oštećenih QR pločica!")
                writable = buf_view.copy()
                w_tiles = writable[self.tiles_base_off : ecc_base_off]
                w_ecc = writable[ecc_base_off:]
                bad_indices = np.flatnonzero(tile_ok == 0)
                for g in np.unique(bad_indices // self.ecc_group_size):
                    g0 = int(g * self.ecc_group_size)
                    g1 = min(self.n_tiles, g0 + self.ecc_group_size)
                    group_bad = [int(idx) for idx in bad_indices if g0 <= idx < g1]
                    if len(group_bad) > 1:
                        raise ValueError(
                            f"Nepopravljivo oštećenje: {len(group_bad)} loših pločica u ECC grupi {g}!"
                        )
                    target_t = group_bad[0]
                    t_start = int(self.tile_offsets[target_t])
                    t_end = int(self.tile_offsets[target_t + 1])
                    t_len = t_end - t_start

                    e_start = int(self.ecc_offsets[g])
                    e_end = int(self.ecc_offsets[g + 1])
                    rec_u64 = w_ecc[e_start:e_end].view(np.uint64)[: t_len // 8].copy()
                    for peer in range(g0, g1):
                        if peer == target_t:
                            continue
                        ps = int(self.tile_offsets[peer])
                        pe = int(self.tile_offsets[peer + 1])
                        plen = min(t_len, pe - ps)
                        rec_u64[: plen // 8] ^= w_tiles[ps : ps + plen].view(np.uint64)

                    w_tiles[t_start:t_end] = rec_u64.view(np.uint8)
                    th_bytes = w_tiles[t_start : t_start + 16].tobytes()
                    tcrc, n_r, _, n_o8, n_s16 = struct.unpack(QR_TILE_HDR_FMT, th_bytes)
                    body_len = _qr_tile_body_len(self.mode, n_r, cols, n_o8, n_s16)
                    recalc = zlib.crc32(w_tiles[t_start + 16 : t_start + 16 + body_len].tobytes()) & 0xFFFFFFFF
                    if recalc != tcrc:
                        raise ValueError(f"ECC oporavak pločice {target_t} nije uspeo (CRC mismatch)!")
                    self.healed_tiles += 1

                self._buf = writable
                self.tile_offsets = writable[64 : 64 + t_off_len].view(np.uint64)
                self.ecc_offsets = writable[64 + t_off_len : 64 + t_off_len + e_off_len].view(np.uint64)
                if self.mode in (MODE_QR_HADAMARD_Q4, MODE_QR_HAD4_PACKED, MODE_QR_HAD4_ASYM, MODE_QR_HAD6):
                    self.sign_bits = writable[aux_off : aux_off + (cols // 8)]
                elif self.mode == MODE_QR_Q8:
                    self.col_scale = None
                    self.sign_bits = None
                else:
                    self.col_scale = writable[aux_off : aux_off + cols * 4].view(np.float32)
                self.tiles_view = writable[tiles_base_off:ecc_base_off]
                self.ecc_view = writable[ecc_base_off:]

    def matmul_input_x_wt(self, x: np.ndarray) -> np.ndarray:
        orig_shape = x.shape
        x_c = np.ascontiguousarray(x, dtype=np.float32)
        m = int(x_c.size // self.cols)
        out = np.empty((m, self.rows), dtype=np.float32)
        if self.mode == MODE_QR_TILED_ZD4:
            _AVX2_LIB.dpuf_matmul_qr_zd4_f32(
                m, self.rows, self.cols, self.n_tiles, x_c.ravel(),
                self.tiles_view, self.tile_offsets, self.col_scale, out.ravel(),
            )
        elif self.mode == MODE_QR_SPLIT_INT4:
            _AVX2_LIB.dpuf_matmul_qr_split4_f32(
                m, self.rows, self.cols, self.n_tiles, x_c.ravel(),
                self.tiles_view, self.tile_offsets, self.col_scale, out.ravel(),
            )
        elif self.mode == MODE_QR_HADAMARD_Q4:
            _AVX2_LIB.dpuf_matmul_qr_hadamard4_f32(
                m, self.rows, self.cols, self.n_tiles, x_c.ravel(),
                self.tiles_view, self.tile_offsets, self.sign_bits, out.ravel(),
            )
        elif self.mode == MODE_QR_Q8:
            _AVX2_LIB.dpuf_matmul_qr_q8_f32(
                m, self.rows, self.cols, self.n_tiles, x_c.ravel(),
                self.tiles_view, self.tile_offsets, out.ravel(),
            )
        elif self.mode == MODE_QR_HAD4_PACKED:
            _AVX2_LIB.dpuf_matmul_qr_hadamard4_packed_f32(
                m, self.rows, self.cols, self.n_tiles, x_c.ravel(),
                self.tiles_view, self.tile_offsets, self.sign_bits, out.ravel(),
            )
        elif self.mode == MODE_QR_HAD4_ASYM:
            _AVX2_LIB.dpuf_matmul_qr_had4_asym_f32(
                m, self.rows, self.cols, self.n_tiles, x_c.ravel(),
                self.tiles_view, self.tile_offsets, self.sign_bits, out.ravel(),
            )
        elif self.mode == MODE_QR_HAD6:
            _AVX2_LIB.dpuf_matmul_qr_had6_f32(
                m, self.rows, self.cols, self.n_tiles, x_c.ravel(),
                self.tiles_view, self.tile_offsets, self.sign_bits, out.ravel(),
            )
        return out.reshape(*orig_shape[:-1], self.rows)

    def lookup_embeddings(self, input_ids: np.ndarray) -> np.ndarray:
        ids_i32 = np.ascontiguousarray(input_ids, dtype=np.int32)
        out = np.empty((ids_i32.size, self.cols), dtype=np.float32)
        if self.mode == MODE_QR_TILED_ZD4:
            _AVX2_LIB.dpuf_embed_qr_zd4_f32(
                ids_i32.size, self.cols, ids_i32.ravel(),
                self.tiles_view, self.tile_offsets, self.col_scale, out.ravel(),
            )
        elif self.mode == MODE_QR_SPLIT_INT4:
            _AVX2_LIB.dpuf_embed_qr_split4_f32(
                ids_i32.size, self.cols, ids_i32.ravel(),
                self.tiles_view, self.tile_offsets, self.col_scale, out.ravel(),
            )
        elif self.mode == MODE_QR_HADAMARD_Q4:
            _AVX2_LIB.dpuf_embed_qr_hadamard4_f32(
                ids_i32.size, self.cols, ids_i32.ravel(),
                self.tiles_view, self.tile_offsets, self.sign_bits, out.ravel(),
            )
        elif self.mode == MODE_QR_Q8:
            _AVX2_LIB.dpuf_embed_qr_q8_f32(
                ids_i32.size, self.cols, ids_i32.ravel(),
                self.tiles_view, self.tile_offsets, out.ravel(),
            )
        elif self.mode == MODE_QR_HAD4_PACKED:
            _AVX2_LIB.dpuf_embed_qr_hadamard4_packed_f32(
                ids_i32.size, self.cols, ids_i32.ravel(),
                self.tiles_view, self.tile_offsets, self.sign_bits, out.ravel(),
            )
        elif self.mode == MODE_QR_HAD4_ASYM:
            _AVX2_LIB.dpuf_embed_qr_had4_asym_f32(
                ids_i32.size, self.cols, ids_i32.ravel(),
                self.tiles_view, self.tile_offsets, self.sign_bits, out.ravel(),
            )
        elif self.mode == MODE_QR_HAD6:
            _AVX2_LIB.dpuf_embed_qr_had6_f32(
                ids_i32.size, self.cols, ids_i32.ravel(),
                self.tiles_view, self.tile_offsets, self.sign_bits, out.ravel(),
            )
        return out.reshape(*input_ids.shape, self.cols)

    def release(self):
        self._buf = None
        self.tiles_view = None
        self.ecc_view = None
        self.tile_offsets = None
        self.ecc_offsets = None
        self.col_scale = None
        self.sign_bits = None

    def heap_bytes(self) -> int:
        b = self._buf
        while b is not None:
            if isinstance(b, mmap.mmap) or (isinstance(b, memoryview) and isinstance(getattr(b, "obj", None), mmap.mmap)):
                return 0
            b = getattr(b, "base", None)
        return int(self._buf.nbytes)

    def mapped_bytes(self) -> int:
        return int(self._buf.nbytes)


class QRMmapSession:
    """Drži otvoren zero-copy mmap za QR-Tiled .dpuf modele."""

    def __init__(self, fobj, mm: mmap.mmap, mm_u8: np.ndarray, weights: Dict[str, Any], meta: Dict[str, Any], stats: Dict[str, Any]):
        self._fobj = fobj
        self._mm = mm
        self._mm_u8 = mm_u8
        self.weights = weights
        self.meta = meta
        self.stats = stats

    def close(self):
        for w in self.weights.values():
            if hasattr(w, "release"):
                w.release()
        self.weights.clear()
        self._mm_u8 = None
        import gc
        gc.collect()
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fobj is not None:
            self._fobj.close()
            self._fobj = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class _DpufWeightDesc(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int32),
        ("rows", ctypes.c_int32),
        ("cols", ctypes.c_int32),
        ("n_tiles", ctypes.c_int32),
        ("tensor_base", ctypes.c_void_p),
        ("tile_offsets", ctypes.c_void_p),
        ("col_scale", ctypes.c_void_p),
        ("sign_bits", ctypes.c_void_p),
    ]


class _DpufModelConfig(ctypes.Structure):
    _fields_ = [
        ("hidden_size", ctypes.c_int32),
        ("intermediate_size", ctypes.c_int32),
        ("num_layers", ctypes.c_int32),
        ("num_q_heads", ctypes.c_int32),
        ("num_kv_heads", ctypes.c_int32),
        ("head_dim", ctypes.c_int32),
        ("vocab_size", ctypes.c_int32),
        ("max_seq_len", ctypes.c_int32),
        ("has_qk_norm", ctypes.c_int32),
        ("rms_eps", ctypes.c_float),
        ("rope_theta", ctypes.c_float),
        ("n_threads", ctypes.c_int32),
        ("use_llama3_rope", ctypes.c_int32),
        ("rope_factor", ctypes.c_float),
        ("rope_low_freq", ctypes.c_float),
        ("rope_high_freq", ctypes.c_float),
        ("rope_orig_max_pos", ctypes.c_int32),
        ("kv_type", ctypes.c_int32),
        ("has_mrope", ctypes.c_int32),
        ("mrope_section_0", ctypes.c_int32),
        ("mrope_section_1", ctypes.c_int32),
        ("mrope_section_2", ctypes.c_int32),
        ("mrope_interleaved", ctypes.c_int32),
    ]


class _DpufLayerWeights(ctypes.Structure):
    _fields_ = [
        ("attn_norm", ctypes.c_void_p),
        ("attn_q_norm", ctypes.c_void_p),
        ("attn_k_norm", ctypes.c_void_p),
        ("ffn_norm", ctypes.c_void_p),
        ("w_q", _DpufWeightDesc),
        ("w_k", _DpufWeightDesc),
        ("w_v", _DpufWeightDesc),
        ("w_o", _DpufWeightDesc),
        ("w_gate", _DpufWeightDesc),
        ("w_up", _DpufWeightDesc),
        ("w_down", _DpufWeightDesc),
    ]


def _make_weight_desc(w: QRTiledZeroDecodeWeight) -> _DpufWeightDesc:
    return _DpufWeightDesc(
        mode=int(w.mode),
        rows=int(w.rows),
        cols=int(w.cols),
        n_tiles=int(w.n_tiles),
        tensor_base=int(w.tiles_view.ctypes.data),
        tile_offsets=int(w.tile_offsets.ctypes.data),
        col_scale=int(w.col_scale.ctypes.data) if w.col_scale is not None else 0,
        sign_bits=int(w.sign_bits.ctypes.data) if w.sign_bits is not None else 0,
    )


class DPUFCTransformerEngine:
    """
    Pure C / AVX2 + Persistent Threadpool Transformer Engine (Steps 1 & 2):
    Runs end-to-end token generation & batch prompt prefill directly over mmap'd QR weights
    with persistent FP32 KV-cache and zero Python/ctypes per-layer overhead!
    """

    def __init__(
        self,
        session: QRMmapSession,
        model_config: Optional[Dict[str, Any]] = None,
        max_seq_len: int = 1024,
        n_threads: int = 8,
        kv_type: int = 0,
    ):
        if _AVX2_LIB is None or not hasattr(_AVX2_LIB, "dpuf_engine_create"):
            raise RuntimeError("Potreban je dpuf_avx2.dll kompajliran sa dpuf_engine.c podrškom.")

        self._session = session
        weights = session.weights
        meta_cfg = dict(session.meta.get("model_config", {}))
        if model_config:
            meta_cfg.update(model_config)

        tok_w = weights["token_embd.weight"]
        lm_w = weights.get("output.weight", tok_w)
        out_norm = np.ascontiguousarray(weights["output_norm.weight"], dtype=np.float32)

        num_layers = 0
        while f"blk.{num_layers}.attn_q.weight" in weights:
            num_layers += 1

        hidden_size = int(meta_cfg.get("hidden_size", tok_w.cols))
        vocab_size = int(meta_cfg.get("vocab_size", tok_w.rows))
        inter_size = int(meta_cfg.get("intermediate_size", weights["blk.0.ffn_gate.weight"].rows))
        has_qk_norm = 1 if ("blk.0.attn_q_norm.weight" in weights) else 0
        if has_qk_norm:
            head_dim = int(meta_cfg.get("head_dim", weights["blk.0.attn_q_norm.weight"].size))
        else:
            head_dim = int(meta_cfg.get("head_dim", 64 if hidden_size == 2048 else 128))
        num_q_heads = int(
            meta_cfg.get("num_q_heads", meta_cfg.get("num_attention_heads", weights["blk.0.attn_q.weight"].rows // head_dim))
        )
        num_kv_heads = int(
            meta_cfg.get("num_kv_heads", meta_cfg.get("num_key_value_heads", weights["blk.0.attn_k.weight"].rows // head_dim))
        )
        rms_eps = float(meta_cfg.get("rms_eps", meta_cfg.get("rms_norm_eps", 1e-6 if has_qk_norm else 1e-5)))
        rope_theta = float(meta_cfg.get("rope_theta", 1000000.0 if has_qk_norm else 500000.0))

        rope_sc = meta_cfg.get("rope_scaling") or {}
        use_llama3_rope = int(meta_cfg.get("use_llama3_rope", 1 if rope_sc.get("rope_type") == "llama3" else 0))

        self.vocab_size = vocab_size
        self.cfg = _DpufModelConfig(
            hidden_size=hidden_size,
            intermediate_size=inter_size,
            num_layers=num_layers,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            max_seq_len=int(max_seq_len),
            has_qk_norm=has_qk_norm,
            rms_eps=rms_eps,
            rope_theta=rope_theta,
            n_threads=int(n_threads),
            use_llama3_rope=use_llama3_rope,
            rope_factor=float(meta_cfg.get("rope_factor", rope_sc.get("factor", 32.0))),
            rope_low_freq=float(meta_cfg.get("rope_low_freq", rope_sc.get("low_freq_factor", 1.0))),
            rope_high_freq=float(meta_cfg.get("rope_high_freq", rope_sc.get("high_freq_factor", 4.0))),
            rope_orig_max_pos=int(meta_cfg.get("rope_orig_max_pos", rope_sc.get("original_max_position_embeddings", 8192))),
            kv_type=int(kv_type),
            has_mrope=1 if "mrope_section" in meta_cfg else 0,
            mrope_section_0=int(meta_cfg.get("mrope_section", [24, 20, 20])[0]) if "mrope_section" in meta_cfg else 0,
            mrope_section_1=int(meta_cfg.get("mrope_section", [24, 20, 20])[1]) if "mrope_section" in meta_cfg else 0,
            mrope_section_2=int(meta_cfg.get("mrope_section", [24, 20, 20])[2]) if "mrope_section" in meta_cfg else 0,
            mrope_interleaved=1 if meta_cfg.get("mrope_interleaved", False) else 0,
        )

        self._keepalive = [out_norm]
        layer_arr = (_DpufLayerWeights * num_layers)()
        for l in range(num_layers):
            p = f"blk.{l}"
            an = np.ascontiguousarray(weights[f"{p}.attn_norm.weight"], dtype=np.float32)
            fn = np.ascontiguousarray(weights[f"{p}.ffn_norm.weight"], dtype=np.float32)
            self._keepalive.extend([an, fn])
            qn_ptr, kn_ptr = 0, 0
            if has_qk_norm:
                qn = np.ascontiguousarray(weights[f"{p}.attn_q_norm.weight"], dtype=np.float32)
                kn = np.ascontiguousarray(weights[f"{p}.attn_k_norm.weight"], dtype=np.float32)
                self._keepalive.extend([qn, kn])
                qn_ptr = int(qn.ctypes.data)
                kn_ptr = int(kn.ctypes.data)

            layer_arr[l] = _DpufLayerWeights(
                attn_norm=int(an.ctypes.data),
                attn_q_norm=qn_ptr,
                attn_k_norm=kn_ptr,
                ffn_norm=int(fn.ctypes.data),
                w_q=_make_weight_desc(weights[f"{p}.attn_q.weight"]),
                w_k=_make_weight_desc(weights[f"{p}.attn_k.weight"]),
                w_v=_make_weight_desc(weights[f"{p}.attn_v.weight"]),
                w_o=_make_weight_desc(weights[f"{p}.attn_output.weight"]),
                w_gate=_make_weight_desc(weights[f"{p}.ffn_gate.weight"]),
                w_up=_make_weight_desc(weights[f"{p}.ffn_up.weight"]),
                w_down=_make_weight_desc(weights[f"{p}.ffn_down.weight"]),
            )

        self._layer_arr = layer_arr
        self._tok_desc = _make_weight_desc(tok_w)
        self._lm_desc = _make_weight_desc(lm_w)
        self._handle = _AVX2_LIB.dpuf_engine_create(
            ctypes.byref(self.cfg),
            self._layer_arr,
            ctypes.byref(self._tok_desc),
            ctypes.byref(self._lm_desc),
            int(out_norm.ctypes.data),
        )
        if not self._handle:
            raise RuntimeError("dpuf_engine_create failed (OOM or invalid config).")

    def reset_cache(self):
        if self._handle:
            _AVX2_LIB.dpuf_engine_reset_cache(self._handle)

    def forward_token(self, token_id: int, pos: int) -> np.ndarray:
        out = np.empty(self.vocab_size, dtype=np.float32)
        rc = _AVX2_LIB.dpuf_engine_forward_token(self._handle, int(token_id), int(pos), out)
        if rc != 0:
            raise RuntimeError(f"dpuf_engine_forward_token failed with rc={rc}")
        return out

    def forward_batch(self, token_ids: np.ndarray, pos0: int = 0, only_last_logits: bool = True) -> np.ndarray:
        ids = np.ascontiguousarray(token_ids, dtype=np.int32).ravel()
        m = int(ids.size)
        out_m = 1 if only_last_logits else m
        out = np.empty((out_m, self.vocab_size), dtype=np.float32)
        rc = _AVX2_LIB.dpuf_engine_forward_batch(
            self._handle, m, ids, int(pos0), 1 if only_last_logits else 0, out.ravel()
        )
        if rc != 0:
            raise RuntimeError(f"dpuf_engine_forward_batch failed with rc={rc}")
        return out

    def close(self):
        if getattr(self, "_handle", None):
            _AVX2_LIB.dpuf_engine_destroy(self._handle)
            self._handle = None
        self._keepalive = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        self.close()


def open_dpuf_qr_mmap(
    path: pathlib.Path,
    verify_tiles: bool = True,
    auto_heal: bool = True,
    verify_file_crc: bool = False,
) -> QRMmapSession:
    """
    Otvara QR-Tiled .dpuf fajl preko Zero-Copy mmap-a (< 5-20 ms bez čitanja celog fajla u heap RAM!).
    Podržava Mode 8 (QR-ZD4), Mode 9 (Varijanta A Split-Int4) i Mode 10 (Varijanta B Hadamard-Int4).
    """
    t0 = time.perf_counter()
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if len(mm) < 24:
        mm.close()
        f.close()
        raise ValueError("Prekratak fajl za DPUF zaglavlje!")

    magic, num_tensors, hdr_crc, pay_crc, header_len = struct.unpack("<8sIIII", mm[:24])
    if magic not in VALID_MAGICS:
        mm.close()
        f.close()
        raise ValueError(f"Nepoznat magic: {magic!r}")
    header_bytes = bytes(mm[24 : 24 + header_len])
    if (zlib.crc32(header_bytes) & 0xFFFFFFFF) != hdr_crc:
        mm.close()
        f.close()
        raise ValueError("CRC32 greška u Finder/JSON zaglavlju!")

    payload_base = 24 + header_len
    if verify_file_crc:
        if (zlib.crc32(mm[payload_base:]) & 0xFFFFFFFF) != pay_crc:
            mm.close()
            f.close()
            raise ValueError("Payload CRC32 mismatch!")

    header_obj = json.loads(header_bytes.decode("utf-8"))
    meta = header_obj.get("__metadata__", {})
    tensor_index = header_obj["__tensors__"]
    _check_reader_compatibility(meta)

    mm_u8 = np.frombuffer(mm, dtype=np.uint8)
    weights: Dict[str, Any] = {}
    healed_total = 0
    for name, (off, length) in tensor_index.items():
        abs_off = payload_base + int(off)
        view = mm_u8[abs_off : abs_off + int(length)]
        mode = int(view[0])
        if mode in ALL_QR_MODES:
            qw = QRTiledZeroDecodeWeight(view, verify_crc=verify_tiles, auto_heal=auto_heal)
            weights[name] = qw
            healed_total += qw.healed_tiles
        elif mode == MODE_RAW_TINY:
            _, ndim, d0, d1 = struct.unpack("<BBII", bytes(view[:10]))
            raw_slice = view[10:]
            if d1 == 2 and raw_slice.size == d0 * 4:
                arr = raw_slice.view(np.float32).copy()
            else:
                arr = bf16_u16_to_fp32(raw_slice.view(np.uint16))
            weights[name] = arr.reshape((d0,)) if (ndim == 1 or d1 in (1, 2)) else arr.reshape((d0, d1))
        else:
            weights[name] = unpack_zero_decode_tensor(bytes(view), prefer_packed_native=True)

    open_ms = (time.perf_counter() - t0) * 1000.0
    return QRMmapSession(
        f, mm, mm_u8, weights, meta, {"open_ms": open_ms, "healed_tiles": healed_total}
    )


# ==============================================================================
# 5. SAMOOPISUJUĆI KONTEJNER SA BOGATIM METAPODACIMA (__metadata__) I CRC32 ZAŠTITOM
# ==============================================================================
def save_dpuf_container(
    path: pathlib.Path,
    encoded_tensors: Dict[str, bytes],
    metadata: Optional[Dict[str, Any]] = None,
    use_dpuf_magic: bool = True,
    align64: Optional[bool] = None,
) -> int:
    uses_qr_mode = any(len(b) > 0 and b[0] in ALL_QR_MODES for b in encoded_tensors.values())
    uses_v11_modes = any(len(b) > 0 and b[0] == MODE_DIRECT_ALS_5BIT for b in encoded_tensors.values())
    do_align = uses_qr_mode if align64 is None else bool(align64)

    tensor_index = {}
    offset = 0
    pad_sizes = []
    pay_crc = 0
    for name, blob in encoded_tensors.items():
        pad_n = _pad64(offset) if do_align else 0
        pad_sizes.append(pad_n)
        if pad_n:
            pay_crc = zlib.crc32(b"\x00" * pad_n, pay_crc)
            offset += pad_n
        tensor_index[name] = [offset, len(blob)]
        pay_crc = zlib.crc32(blob, pay_crc)
        offset += len(blob)

    pay_crc &= 0xFFFFFFFF
    payload_bytes = offset

    min_ver = "1.2.0" if uses_qr_mode else ("1.1.0" if uses_v11_modes else "1.0.0")
    meta_block = {
        "format": f"DPUF / LR-FLAC v{DPUF_FORMAT_VERSION}",
        "format_version": DPUF_FORMAT_VERSION,
        "format_version_code": DPUF_FORMAT_VERSION_CODE,
        "min_reader_version": min_ver,
        "codec_version": DPUF_CODEC_VERSION,
        "model_revision": "1.0.0",
        "model_revision_code": 10000,
        "supersedes": None,
        "author": "D. P. (dule_pan)",
        "license": DEFAULT_LICENSE_NOTICE,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "num_tensors": len(encoded_tensors),
        "payload_bytes": payload_bytes,
    }
    if metadata:
        meta_block.update(metadata)
        if "format_version" in metadata and "format_version_code" not in metadata:
            meta_block["format_version_code"] = semver_to_code(metadata["format_version"])
        if "model_revision" in metadata and "model_revision_code" not in metadata:
            meta_block["model_revision_code"] = semver_to_code(metadata["model_revision"])

    full_header_obj = {
        "__metadata__": meta_block,
        "__tensors__": tensor_index,
    }
    header_bytes = json.dumps(full_header_obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if do_align:
        # JSON ignoriše završne razmake, čime payload počinje tačno na 64-bajtnom poravnanju fajla!
        pad_hdr = _pad64(24 + len(header_bytes))
        if pad_hdr:
            header_bytes += b" " * pad_hdr

    hdr_crc = zlib.crc32(header_bytes) & 0xFFFFFFFF

    magic = MAGIC_DPUF if use_dpuf_magic else MAGIC_LRFLAC
    top_hdr = struct.pack("<8sIIII", magic, len(encoded_tensors), hdr_crc, pay_crc, len(header_bytes))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(top_hdr)
        f.write(header_bytes)
        for blob, pad_n in zip(encoded_tensors.values(), pad_sizes):
            if pad_n:
                f.write(b"\x00" * pad_n)
            f.write(blob)
    return 24 + len(header_bytes) + payload_bytes


def inspect_dpuf_metadata(path: pathlib.Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        top = f.read(24)
        if len(top) < 24:
            raise ValueError("Prekratak fajl za DPUF/LRFLAC zaglavlje!")
        magic, num_tensors, hdr_crc, pay_crc, header_len = struct.unpack("<8sIIII", top)
        if magic not in VALID_MAGICS:
            raise ValueError(f"Nepoznat magic potpis: {magic!r}")
        header_bytes = f.read(header_len)

    if (zlib.crc32(header_bytes) & 0xFFFFFFFF) != hdr_crc:
        raise ValueError("CRC32 greška u zaglavlju fajla!")

    header_obj = json.loads(header_bytes.decode("utf-8"))
    meta = header_obj.get("__metadata__", {})
    meta["magic"] = magic.decode("ascii")
    meta["header_crc32_hex"] = f"0x{hdr_crc:08X}"
    meta["payload_crc32_hex"] = f"0x{pay_crc:08X}"
    return meta


def load_dpuf_container(
    path: pathlib.Path, verify_crc: bool = True
) -> Tuple[Dict[str, bytes], Dict[str, Any]]:
    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) < 24:
        raise ValueError("Bezbednosna greška: prekratak fajl!")

    magic, num_tensors, hdr_crc, pay_crc, header_len = struct.unpack("<8sIIII", raw[:24])
    if magic not in VALID_MAGICS:
        raise ValueError(f"Bezbednosna greška: nepoznat magic header {magic!r}!")
    if num_tensors > MAX_ALLOWED_TENSORS:
        raise ValueError(f"Bezbednosna greška: previše tenzora ({num_tensors})!")
    if 24 + header_len > len(raw):
        raise ValueError("Bezbednosna greška: oštećena dužina zaglavlja!")

    header_bytes = raw[24 : 24 + header_len]
    payload = raw[24 + header_len :]

    if verify_crc:
        if (zlib.crc32(header_bytes) & 0xFFFFFFFF) != hdr_crc:
            raise ValueError("Bezbednosna greška: CRC32 zaglavlja se ne poklapa!")
        if (zlib.crc32(payload) & 0xFFFFFFFF) != pay_crc:
            raise ValueError("Bezbednosna greška: CRC32 payload-a se ne poklapa!")

    header_obj = json.loads(header_bytes.decode("utf-8"))
    if "__tensors__" in header_obj:
        meta = header_obj.get("__metadata__", {})
        tensor_index = header_obj["__tensors__"]
    else:
        meta = {}
        tensor_index = header_obj

    _check_reader_compatibility(meta)

    out = {}
    for name, (off, length) in tensor_index.items():
        if off < 0 or length < 0 or off + length > len(payload):
            raise ValueError(f"Bezbednosna greška: tenzor {name} izlazi van granica fajla!")
        out[name] = payload[off : off + length]
    return out, meta


def compare_dpuf_versions(old_path: pathlib.Path, new_path: pathlib.Path) -> Dict[str, Any]:
    old_meta = inspect_dpuf_metadata(old_path)
    new_meta = inspect_dpuf_metadata(new_path)
    old_rev = int(old_meta.get("model_revision_code", semver_to_code(old_meta.get("model_revision", "1.0.0"))))
    new_rev = int(new_meta.get("model_revision_code", semver_to_code(new_meta.get("model_revision", "1.0.0"))))
    old_fmt = int(old_meta.get("format_version_code", semver_to_code(old_meta.get("format_version", "1.0.0"))))
    new_fmt = int(new_meta.get("format_version_code", semver_to_code(new_meta.get("format_version", "1.0.0"))))
    return {
        "is_newer_model": new_rev > old_rev,
        "is_newer_format": new_fmt > old_fmt,
        "old_model_revision": old_meta.get("model_revision", "1.0.0"),
        "new_model_revision": new_meta.get("model_revision", "1.0.0"),
        "old_format_version": old_meta.get("format_version", "1.0.0"),
        "new_format_version": new_meta.get("format_version", "1.0.0"),
        "supersedes": new_meta.get("supersedes"),
    }


def extract_hf_tokenizer_and_config(
    model_dir: pathlib.Path,
    max_seq_len: int = 2048,
    has_qk_norm: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Ekstraktuje `model_config` i kompletan byte-level BPE `tokenizer` (vocab, merges, added_tokens,
    BOS/EOS, chat_format) iz HuggingFace direktorijuma kako bi `.dpuf` fajl bio 100% samodovoljan (Korak 5).
    """
    model_dir = pathlib.Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tok_json = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tcfg_path = model_dir / "tokenizer_config.json"
    tcfg = json.loads(tcfg_path.read_text(encoding="utf-8")) if tcfg_path.exists() else {}
    gcfg_path = model_dir / "generation_config.json"
    gcfg = json.loads(gcfg_path.read_text(encoding="utf-8")) if gcfg_path.exists() else {}

    model_type = str(cfg.get("model_type", "unknown")).lower()
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    hidden_size = int(text_cfg["hidden_size"])
    num_q_heads = int(text_cfg["num_attention_heads"])
    head_dim = int(text_cfg.get("head_dim", hidden_size // num_q_heads))
    rope_scaling = text_cfg.get("rope_scaling") or cfg.get("rope_scaling") or {}
    use_llama3_rope = isinstance(rope_scaling, dict) and rope_scaling.get("rope_type") == "llama3"

    model_config = {
        "architectures": cfg.get("architectures", []),
        "model_type": model_type,
        "hidden_size": hidden_size,
        "intermediate_size": int(text_cfg["intermediate_size"]),
        "num_layers": int(text_cfg["num_hidden_layers"]),
        "num_q_heads": num_q_heads,
        "num_kv_heads": int(text_cfg.get("num_key_value_heads", num_q_heads)),
        "head_dim": head_dim,
        "vocab_size": int(text_cfg["vocab_size"]),
        "max_seq_len": int(max_seq_len),
        "has_qk_norm": bool("qwen3" in model_type) if has_qk_norm is None else bool(has_qk_norm),
        "rms_eps": float(text_cfg.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(text_cfg.get("rope_theta", 10000.0)),
        "use_llama3_rope": bool(use_llama3_rope),
        "rope_factor": float(rope_scaling.get("factor", 1.0)) if use_llama3_rope else 1.0,
        "rope_low_freq": float(rope_scaling.get("low_freq_factor", 1.0)) if use_llama3_rope else 1.0,
        "rope_high_freq": float(rope_scaling.get("high_freq_factor", 4.0)) if use_llama3_rope else 4.0,
        "rope_orig_max_pos": int(rope_scaling.get("original_max_position_embeddings", 8192)) if use_llama3_rope else 8192,
        "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", text_cfg.get("tie_word_embeddings", True))),
    }
    if isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling:
        model_config["mrope_section"] = [int(x) for x in rope_scaling["mrope_section"]]
        model_config["mrope_interleaved"] = bool(rope_scaling.get("mrope_interleaved", False))
    for tok_key in ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"):
        if tok_key in cfg and cfg[tok_key] is not None:
            model_config[tok_key] = int(cfg[tok_key])

    vcfg = cfg.get("vision_config")
    if isinstance(vcfg, dict):
        pp_path = model_dir / "preprocessor_config.json"
        pp_cfg = json.loads(pp_path.read_text(encoding="utf-8")) if pp_path.exists() else {}
        pp_size = pp_cfg.get("size") if isinstance(pp_cfg.get("size"), dict) else {}
        vrp = vcfg.get("rope_parameters") or {}
        v_heads = int(vcfg.get("num_heads", vcfg.get("num_attention_heads", 16)))
        v_hidden = int(vcfg["hidden_size"])
        model_config["vision_config"] = {
            "depth": int(vcfg["depth"]),
            "hidden_size": v_hidden,
            "intermediate_size": int(vcfg["intermediate_size"]),
            "num_heads": v_heads,
            "head_dim": int(vcfg.get("head_dim", v_hidden // v_heads)),
            "in_channels": int(vcfg.get("in_channels", 3)),
            "patch_size": int(vcfg.get("patch_size", 16)),
            "spatial_merge_size": int(vcfg.get("spatial_merge_size", 2)),
            "temporal_patch_size": int(vcfg.get("temporal_patch_size", 2)),
            "out_hidden_size": int(vcfg.get("out_hidden_size", hidden_size)),
            "num_position_embeddings": int(vcfg.get("num_position_embeddings", 2304)),
            "deepstack_visual_indexes": [int(x) for x in vcfg.get("deepstack_visual_indexes", [])],
            "hidden_act": str(vcfg.get("hidden_act", "gelu_pytorch_tanh")),
            "rope_theta": float(vrp.get("rope_theta", vcfg.get("rope_theta", 10000.0))),
            "layer_norm_eps": float(vcfg.get("layer_norm_eps", 1e-6)),
            "image_mean": [float(x) for x in pp_cfg.get("image_mean", [0.5, 0.5, 0.5])],
            "image_std": [float(x) for x in pp_cfg.get("image_std", [0.5, 0.5, 0.5])],
            "min_pixels": int(pp_size.get("shortest_edge", 65536)),
            "max_pixels": int(pp_size.get("longest_edge", 16777216)),
        }

    raw_vocab: Dict[str, int] = tok_json["model"]["vocab"]
    added_list = tok_json.get("added_tokens", [])
    max_tid = max(
        max((int(v) for v in raw_vocab.values()), default=0),
        max((int(a["id"]) for a in added_list), default=0),
    )
    vocab_arr = [""] * (max_tid + 1)
    for tok_s, tid in raw_vocab.items():
        vocab_arr[int(tid)] = tok_s
    added_tokens: Dict[str, int] = {}
    for item in added_list:
        tid = int(item["id"])
        content = str(item["content"])
        vocab_arr[tid] = content
        added_tokens[content] = tid

    merges_norm = []
    for m in tok_json["model"].get("merges", []):
        if isinstance(m, (list, tuple)):
            a, b = str(m[0]), str(m[1])
            if " " in a or " " in b:
                raise ValueError(f"Nevalidan space u BPE merge paru: {m!r}")
            merges_norm.append(f"{a} {b}")
        else:
            merges_norm.append(str(m))

    eos_ids = []
    def _push_eos(val):
        if isinstance(val, int) and val not in eos_ids:
            eos_ids.append(val)
        elif isinstance(val, list):
            for v in val:
                if isinstance(v, int) and v not in eos_ids:
                    eos_ids.append(v)

    eos_tok_name = tcfg.get("eos_token")
    if isinstance(eos_tok_name, str) and eos_tok_name in added_tokens:
        _push_eos(added_tokens[eos_tok_name])
    _push_eos(cfg.get("eos_token_id"))
    _push_eos(gcfg.get("eos_token_id"))

    add_bos = bool(tcfg.get("add_bos_token", False))
    bos_tok_name = tcfg.get("bos_token")
    bos_id = added_tokens.get(bos_tok_name) if isinstance(bos_tok_name, str) else None
    if bos_id is None and add_bos:
        bos_id = cfg.get("bos_token_id")

    pre_str = json.dumps(tok_json.get("pre_tokenizer", {}), ensure_ascii=False)
    number_group_len = 3 if "\\p{N}{1,3}" in pre_str else 1

    if "<|start_header_id|>" in added_tokens:
        chat_format = "llama3"
    elif "<|im_start|>" in added_tokens:
        chat_format = "chatml"
    else:
        chat_format = "raw"

    tokenizer_meta = {
        "type": "byte_level_bpe",
        "vocab": vocab_arr,
        "merges": merges_norm,
        "added_tokens": added_tokens,
        "bos_token_id": bos_id,
        "eos_token_ids": eos_ids,
        "add_bos": add_bos,
        "number_group_len": number_group_len,
        "chat_format": chat_format,
    }
    return {
        "model_name": model_dir.name,
        "model_config": model_config,
        "tokenizer": tokenizer_meta,
    }
