/*
 * dpuf_avx2.c — Native AVX2 + FMA + OpenMP Zero-Decode Kernel for DPUF v1.0/v1.1
 * Copyright (c) 2026 D. P. (dule_pan). Licensed under DPUF Community & Enterprise License.
 *
 * Computes Y = ((X * col_scale) @ Q^T) * row_scale directly from packed 4-bit nibble pairs
 * + escape overflow buffers WITHOUT keeping float32/int16 weight matrices in RAM!
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <intrin.h>
#include <omp.h>
#include "dpuf_kernels.h"

static __forceinline int32_t decode_outlier(
    const uint8_t* over8,
    const uint16_t* super16,
    uint32_t* o_idx,
    uint32_t* s_idx
) {
    uint32_t ov = (uint32_t)over8[(*o_idx)++];
    if (ov == 255u) {
        ov = (uint32_t)super16[(*s_idx)++];
    }
    uint32_t zz = ov + 15u;
    return (int32_t)(zz >> 1) ^ -(int32_t)(zz & 1u);
}

static __forceinline float hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

DPUF_EXPORT void dpuf_index_rows_zd4(
    int32_t rows,
    int32_t cols,
    const uint8_t* pairs,
    const uint8_t* over8,
    uint32_t* row_over_off,
    uint32_t* row_super_off
) {
    uint32_t o_cur = 0;
    uint32_t s_cur = 0;

    if ((cols & 15) == 0) {
        const __m128i mask0F = _mm_set1_epi8(0x0F);
        int32_t half_cols = cols >> 1;
        for (int32_t r = 0; r < rows; r++) {
            row_over_off[r] = o_cur;
            row_super_off[r] = s_cur;
            const uint8_t* rp = pairs + (size_t)r * (size_t)half_cols;
            for (int32_t c = 0; c < half_cols; c += 8) {
                __m128i raw8 = _mm_loadl_epi64((const __m128i*)(rp + c));
                __m128i hi = _mm_and_si128(_mm_srli_epi16(raw8, 4), mask0F);
                __m128i lo = _mm_and_si128(raw8, mask0F);
                __m128i nib16 = _mm_unpacklo_epi8(hi, lo);
                uint32_t esc = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(nib16, mask0F));
                while (esc) {
                    if (over8[o_cur++] == 255u) {
                        s_cur++;
                    }
                    esc &= esc - 1u;
                }
            }
        }
    } else {
        for (int32_t r = 0; r < rows; r++) {
            row_over_off[r] = o_cur;
            row_super_off[r] = s_cur;
            size_t base = (size_t)r * (size_t)cols;
            for (int32_t c = 0; c < cols; c++) {
                size_t idx = base + (size_t)c;
                uint8_t b = pairs[idx >> 1];
                uint8_t nib = (idx & 1u) ? (b & 0x0Fu) : (b >> 4);
                if (nib == 15u) {
                    if (over8[o_cur++] == 255u) {
                        s_cur++;
                    }
                }
            }
        }
    }
    row_over_off[rows] = o_cur;
    row_super_off[rows] = s_cur;
}

DPUF_EXPORT void dpuf_matmul_zd4_f32(
    int32_t M,
    int32_t rows,
    int32_t cols,
    const float* x_in,
    const uint8_t* pairs,
    const uint8_t* over8,
    const uint16_t* super16,
    const uint32_t* row_over_off,
    const uint32_t* row_super_off,
    const float* row_scale,
    const float* col_scale,
    float* out
) {
    const __m128i lut16 = _mm_setr_epi8(
        0, -1, 1, -2, 2, -3, 3, -4,
        4, -5, 5, -6, 6, -7, 7, 0
    );
    const __m128i mask0F = _mm_set1_epi8(0x0F);

    /* Fused input column scaling: x_scaled[m, c] = x_in[m, c] * col_scale[c] */
    float stack_x[4096];
    float* x_scaled = ((size_t)M * (size_t)cols <= 4096u)
        ? stack_x
        : (float*)malloc((size_t)M * (size_t)cols * sizeof(float));
    if (!x_scaled) return;

    if ((cols & 7) == 0) {
        for (int32_t m = 0; m < M; m++) {
            const float* src = x_in + (size_t)m * (size_t)cols;
            float* dst = x_scaled + (size_t)m * (size_t)cols;
            for (int32_t c = 0; c < cols; c += 8) {
                __m256 xv = _mm256_loadu_ps(src + c);
                __m256 cv = _mm256_loadu_ps(col_scale + c);
                _mm256_storeu_ps(dst + c, _mm256_mul_ps(xv, cv));
            }
        }
    } else {
        for (int32_t m = 0; m < M; m++) {
            const float* src = x_in + (size_t)m * (size_t)cols;
            float* dst = x_scaled + (size_t)m * (size_t)cols;
            for (int32_t c = 0; c < cols; c++) {
                dst[c] = src[c] * col_scale[c];
            }
        }
    }

    if ((cols & 15) != 0) {
        int32_t r;
        #pragma omp parallel for schedule(static) if(rows >= 64)
        for (r = 0; r < rows; r++) {
            float r_s = row_scale[r];
            size_t base = (size_t)r * (size_t)cols;
            for (int32_t m = 0; m < M; m++) {
                uint32_t o_idx = row_over_off[r];
                uint32_t s_idx = row_super_off[r];
                const float* xr = x_scaled + (size_t)m * (size_t)cols;
                double acc = 0.0;
                for (int32_t c = 0; c < cols; c++) {
                    size_t idx = base + (size_t)c;
                    uint8_t b = pairs[idx >> 1];
                    uint8_t nib = (idx & 1u) ? (b & 0x0Fu) : (b >> 4);
                    int32_t qi = (nib < 15u)
                        ? ((int32_t)(nib >> 1) ^ -(int32_t)(nib & 1u))
                        : decode_outlier(over8, super16, &o_idx, &s_idx);
                    acc += (double)xr[c] * (double)qi;
                }
                out[(size_t)m * (size_t)rows + (size_t)r] = (float)acc * r_s;
            }
        }
        if (x_scaled != stack_x) free(x_scaled);
        return;
    }

    enum { TILE_COLS = 1024 };
    int32_t half_cols = cols >> 1;

    if (M == 1) {
        /* Register-resident vector-matrix FMA (lowest latency for single-token decode) */
        int32_t r;
        #pragma omp parallel for schedule(static) if(rows >= 64)
        for (r = 0; r < rows; r++) {
            uint32_t o_idx = row_over_off[r];
            uint32_t s_idx = row_super_off[r];
            const uint8_t* rp = pairs + (size_t)r * (size_t)half_cols;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (int32_t c = 0; c < cols; c += 16) {
                __m128i raw8 = _mm_loadl_epi64((const __m128i*)(rp + (c >> 1)));
                __m128i hi = _mm_and_si128(_mm_srli_epi16(raw8, 4), mask0F);
                __m128i lo = _mm_and_si128(raw8, mask0F);
                __m128i nib16 = _mm_unpacklo_epi8(hi, lo);

                uint32_t esc = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(nib16, mask0F));
                __m128i q16_i8 = _mm_shuffle_epi8(lut16, nib16);

                __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q16_i8));
                __m256 wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q16_i8, 8)));

                if (esc) {
                    float tmp[16];
                    _mm256_storeu_ps(tmp, wf0);
                    _mm256_storeu_ps(tmp + 8, wf1);
                    while (esc) {
                        unsigned long bit;
                        _BitScanForward(&bit, esc);
                        tmp[bit] = (float)decode_outlier(over8, super16, &o_idx, &s_idx);
                        esc &= esc - 1u;
                    }
                    wf0 = _mm256_loadu_ps(tmp);
                    wf1 = _mm256_loadu_ps(tmp + 8);
                }

                __m256 x0 = _mm256_loadu_ps(x_scaled + c);
                __m256 x1 = _mm256_loadu_ps(x_scaled + c + 8);
                acc0 = _mm256_fmadd_ps(wf0, x0, acc0);
                acc1 = _mm256_fmadd_ps(wf1, x1, acc1);
            }
            out[r] = hsum256_ps(_mm256_add_ps(acc0, acc1)) * row_scale[r];
        }
    } else {
        int32_t r;
        #pragma omp parallel for schedule(static) if(rows >= 32)
        for (r = 0; r < rows; r++) {
            __declspec(align(32)) float w_tile[TILE_COLS];
            uint32_t o_idx = row_over_off[r];
            uint32_t s_idx = row_super_off[r];
            const uint8_t* rp = pairs + (size_t)r * (size_t)half_cols;
            float r_s = row_scale[r];

            for (int32_t m = 0; m < M; m++) {
                out[(size_t)m * (size_t)rows + (size_t)r] = 0.0f;
            }

            for (int32_t c0 = 0; c0 < cols; c0 += TILE_COLS) {
                int32_t c_end = c0 + TILE_COLS;
                if (c_end > cols) c_end = cols;
                int32_t tile_len = c_end - c0;

                for (int32_t tc = 0; tc < tile_len; tc += 16) {
                    int32_t c = c0 + tc;
                    __m128i raw8 = _mm_loadl_epi64((const __m128i*)(rp + (c >> 1)));
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(raw8, 4), mask0F);
                    __m128i lo = _mm_and_si128(raw8, mask0F);
                    __m128i nib16 = _mm_unpacklo_epi8(hi, lo);

                    uint32_t esc = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(nib16, mask0F));
                    __m128i q16_i8 = _mm_shuffle_epi8(lut16, nib16);

                    __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q16_i8));
                    __m256 wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q16_i8, 8)));

                    _mm256_store_ps(w_tile + tc, wf0);
                    _mm256_store_ps(w_tile + tc + 8, wf1);
                    while (esc) {
                        unsigned long bit;
                        _BitScanForward(&bit, esc);
                        w_tile[tc + (int32_t)bit] = (float)decode_outlier(over8, super16, &o_idx, &s_idx);
                        esc &= esc - 1u;
                    }
                }

                int32_t m = 0;
                for (; m + 3 < M; m += 4) {
                    const float* x0_p = x_scaled + (size_t)m * (size_t)cols + c0;
                    const float* x1_p = x0_p + cols;
                    const float* x2_p = x1_p + cols;
                    const float* x3_p = x2_p + cols;
                    __m256 a0 = _mm256_setzero_ps();
                    __m256 a1 = _mm256_setzero_ps();
                    __m256 a2 = _mm256_setzero_ps();
                    __m256 a3 = _mm256_setzero_ps();
                    for (int32_t tc = 0; tc < tile_len; tc += 8) {
                        __m256 wv = _mm256_load_ps(w_tile + tc);
                        a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0_p + tc), a0);
                        a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1_p + tc), a1);
                        a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x2_p + tc), a2);
                        a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x3_p + tc), a3);
                    }
                    out[(size_t)(m + 0) * (size_t)rows + (size_t)r] += hsum256_ps(a0);
                    out[(size_t)(m + 1) * (size_t)rows + (size_t)r] += hsum256_ps(a1);
                    out[(size_t)(m + 2) * (size_t)rows + (size_t)r] += hsum256_ps(a2);
                    out[(size_t)(m + 3) * (size_t)rows + (size_t)r] += hsum256_ps(a3);
                }
                for (; m < M; m++) {
                    const float* xp = x_scaled + (size_t)m * (size_t)cols + c0;
                    __m256 a0 = _mm256_setzero_ps();
                    __m256 a1 = _mm256_setzero_ps();
                    for (int32_t tc = 0; tc < tile_len; tc += 16) {
                        __m256 wv0 = _mm256_load_ps(w_tile + tc);
                        __m256 wv1 = _mm256_load_ps(w_tile + tc + 8);
                        a0 = _mm256_fmadd_ps(wv0, _mm256_loadu_ps(xp + tc), a0);
                        a1 = _mm256_fmadd_ps(wv1, _mm256_loadu_ps(xp + tc + 8), a1);
                    }
                    out[(size_t)m * (size_t)rows + (size_t)r] += hsum256_ps(_mm256_add_ps(a0, a1));
                }
            }

            for (int32_t m = 0; m < M; m++) {
                out[(size_t)m * (size_t)rows + (size_t)r] *= r_s;
            }
        }
    }

    if (x_scaled != stack_x) free(x_scaled);
}

DPUF_EXPORT void dpuf_embed_zd4_f32(
    int32_t n_tokens,
    int32_t cols,
    const int32_t* token_ids,
    const uint8_t* pairs,
    const uint8_t* over8,
    const uint16_t* super16,
    const uint32_t* row_over_off,
    const uint32_t* row_super_off,
    const float* row_scale,
    const float* col_scale,
    float* out
) {
    for (int32_t t = 0; t < n_tokens; t++) {
        int32_t r = token_ids[t];
        uint32_t o_idx = row_over_off[r];
        uint32_t s_idx = row_super_off[r];
        float r_s = row_scale[r];
        float* dst = out + (size_t)t * (size_t)cols;
        size_t base = (size_t)r * (size_t)cols;
        for (int32_t c = 0; c < cols; c++) {
            size_t idx = base + (size_t)c;
            uint8_t b = pairs[idx >> 1];
            uint8_t nib = (idx & 1u) ? (b & 0x0Fu) : (b >> 4);
            int32_t qi = (nib < 15u)
                ? ((int32_t)(nib >> 1) ^ -(int32_t)(nib & 1u))
                : decode_outlier(over8, super16, &o_idx, &s_idx);
            dst[c] = (float)qi * r_s * col_scale[c];
        }
    }
}

/* ============================================================================
 * QR-INSPIRED SELF-CONTAINED TILED FORMAT (MODE_QR_TILED_ZD4 = 8)
 * - Finder Directory: 64-byte aligned offsets & tile descriptors
 * - Self-Contained 2D Tiles (16 rows/tile): [Header][row_scale][pairs][over8][super16][pad64]
 * - Per-Tile IEEE-802.3 CRC32 + RAID-XOR Parity ECC Group Self-Healing
 * ============================================================================ */

/* DpufQrTileHdr is defined in dpuf_kernels.h */

static uint32_t g_crc32_tab[256];
static volatile int g_crc_init = 0;

static void init_crc32_table(void) {
    if (g_crc_init) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        }
        g_crc32_tab[i] = c;
    }
    g_crc_init = 1;
}

DPUF_EXPORT uint32_t dpuf_crc32_bytes(const uint8_t* data, uint32_t len) {
    init_crc32_table();
    uint32_t crc = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < len; i++) {
        crc = g_crc32_tab[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

/*
 * Fast verification of all QR tiles in a memory-mapped tensor.
 * Returns number of corrupted tiles found, and writes corrupted tile index (or -1) per ECC group.
 */
DPUF_EXPORT int32_t dpuf_qr_verify_tiles(
    int32_t n_tiles,
    int32_t cols,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    uint8_t* tile_ok_out
) {
    init_crc32_table();
    int32_t bad_count = 0;
    int32_t t;
    #pragma omp parallel for reduction(+:bad_count) schedule(dynamic, 4) if(n_tiles >= 16)
    for (t = 0; t < n_tiles; t++) {
        const uint8_t* tptr = tensor_base + tile_offsets[t];
        const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
        uint32_t body_bytes = (uint32_t)(
            (size_t)th->n_rows * 4u
            + ((size_t)th->n_rows * (size_t)cols + 1u) / 2u
            + (size_t)th->n_over8
            + (size_t)th->n_super16 * 2u
        );
        uint32_t calc = dpuf_crc32_bytes(tptr + sizeof(DpufQrTileHdr), body_bytes);
        uint8_t ok = (calc == th->tile_crc32) ? 1u : 0u;
        tile_ok_out[t] = ok;
        if (!ok) bad_count += 1;
    }
    return bad_count;
}

/*
 * Direct Zero-Copy AVX2+FMA MatMul over 64-byte aligned Self-Contained QR Tiles!
 * No pre-indexing, no malloc copies of pairs/over8 — runs directly on mmap'd file pages!
 */
DPUF_EXPORT void dpuf_matmul_qr_zd4_f32(
    int32_t M,
    int32_t rows,
    int32_t cols,
    int32_t n_tiles,
    const float* x_in,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    const float* col_scale,
    float* out
) {
    const __m128i lut16 = _mm_setr_epi8(
        0, -1, 1, -2, 2, -3, 3, -4,
        4, -5, 5, -6, 6, -7, 7, 0
    );
    const __m128i mask0F = _mm_set1_epi8(0x0F);

    float stack_x[4096];
    float* x_scaled = ((size_t)M * (size_t)cols <= 4096u)
        ? stack_x
        : (float*)malloc((size_t)M * (size_t)cols * sizeof(float));
    if (!x_scaled) return;

    if ((cols & 7) == 0) {
        for (int32_t m = 0; m < M; m++) {
            const float* src = x_in + (size_t)m * (size_t)cols;
            float* dst = x_scaled + (size_t)m * (size_t)cols;
            for (int32_t c = 0; c < cols; c += 8) {
                __m256 xv = _mm256_loadu_ps(src + c);
                __m256 cv = _mm256_loadu_ps(col_scale + c);
                _mm256_storeu_ps(dst + c, _mm256_mul_ps(xv, cv));
            }
        }
    } else {
        for (int32_t m = 0; m < M; m++) {
            const float* src = x_in + (size_t)m * (size_t)cols;
            float* dst = x_scaled + (size_t)m * (size_t)cols;
            for (int32_t c = 0; c < cols; c++) {
                dst[c] = src[c] * col_scale[c];
            }
        }
    }

    const int32_t tile_stride = 16;

    if ((cols & 15) != 0) {
        int32_t t;
        #pragma omp parallel for schedule(static) if(n_tiles >= 4)
        for (t = 0; t < n_tiles; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const float* r_scale = (const float*)(tptr + sizeof(DpufQrTileHdr));
            const uint8_t* pairs = (const uint8_t*)(r_scale + n_r);
            size_t pairs_bytes = ((size_t)n_r * (size_t)cols + 1u) / 2u;
            const uint8_t* over8 = pairs + pairs_bytes;
            const uint16_t* super16 = (const uint16_t*)(over8 + th->n_over8);

            for (int32_t m = 0; m < M; m++) {
                const float* xr = x_scaled + (size_t)m * (size_t)cols;
                uint32_t o_idx = 0;
                uint32_t s_idx = 0;
                for (int32_t lr = 0; lr < n_r; lr++) {
                    double acc = 0.0;
                    size_t base = (size_t)lr * (size_t)cols;
                    for (int32_t c = 0; c < cols; c++) {
                        size_t idx = base + (size_t)c;
                        uint8_t b = pairs[idx >> 1];
                        uint8_t nib = (idx & 1u) ? (b & 0x0Fu) : (b >> 4);
                        int32_t qi = (nib < 15u)
                            ? ((int32_t)(nib >> 1) ^ -(int32_t)(nib & 1u))
                            : decode_outlier(over8, super16, &o_idx, &s_idx);
                        acc += (double)xr[c] * (double)qi;
                    }
                    out[(size_t)m * (size_t)rows + (size_t)(r0 + lr)] = (float)acc * r_scale[lr];
                }
            }
        }
        if (x_scaled != stack_x) free(x_scaled);
        return;
    }

    int32_t half_cols = cols >> 1;

    if (M == 1) {
        int32_t t;
        #pragma omp parallel for schedule(static) if(n_tiles >= 4)
        for (t = 0; t < n_tiles; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const float* r_scale = (const float*)(tptr + sizeof(DpufQrTileHdr));
            const uint8_t* pairs = (const uint8_t*)(r_scale + n_r);
            const uint8_t* over8 = pairs + (size_t)n_r * (size_t)half_cols;
            const uint16_t* super16 = (const uint16_t*)(over8 + th->n_over8);

            uint32_t o_idx = 0;
            uint32_t s_idx = 0;

            for (int32_t lr = 0; lr < n_r; lr++) {
                const uint8_t* rp = pairs + (size_t)lr * (size_t)half_cols;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();

                for (int32_t c = 0; c < cols; c += 16) {
                    __m128i raw8 = _mm_loadl_epi64((const __m128i*)(rp + (c >> 1)));
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(raw8, 4), mask0F);
                    __m128i lo = _mm_and_si128(raw8, mask0F);
                    __m128i nib16 = _mm_unpacklo_epi8(hi, lo);

                    uint32_t esc = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(nib16, mask0F));
                    __m128i q16_i8 = _mm_shuffle_epi8(lut16, nib16);

                    __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q16_i8));
                    __m256 wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q16_i8, 8)));

                    if (esc) {
                        float tmp[16];
                        _mm256_storeu_ps(tmp, wf0);
                        _mm256_storeu_ps(tmp + 8, wf1);
                        while (esc) {
                            unsigned long bit;
                            _BitScanForward(&bit, esc);
                            tmp[bit] = (float)decode_outlier(over8, super16, &o_idx, &s_idx);
                            esc &= esc - 1u;
                        }
                        wf0 = _mm256_loadu_ps(tmp);
                        wf1 = _mm256_loadu_ps(tmp + 8);
                    }
                    acc0 = _mm256_fmadd_ps(wf0, _mm256_loadu_ps(x_scaled + c), acc0);
                    acc1 = _mm256_fmadd_ps(wf1, _mm256_loadu_ps(x_scaled + c + 8), acc1);
                }
                out[r0 + lr] = hsum256_ps(_mm256_add_ps(acc0, acc1)) * r_scale[lr];
            }
        }
    } else {
        enum { TILE_COLS = 1024 };
        int32_t t;
        #pragma omp parallel for schedule(static) if(n_tiles >= 4)
        for (t = 0; t < n_tiles; t++) {
            __declspec(align(32)) float w_tile[TILE_COLS];
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const float* r_scale = (const float*)(tptr + sizeof(DpufQrTileHdr));
            const uint8_t* pairs = (const uint8_t*)(r_scale + n_r);
            const uint8_t* over8 = pairs + (size_t)n_r * (size_t)half_cols;
            const uint16_t* super16 = (const uint16_t*)(over8 + th->n_over8);

            uint32_t o_idx = 0;
            uint32_t s_idx = 0;

            for (int32_t lr = 0; lr < n_r; lr++) {
                int32_t r = r0 + lr;
                const uint8_t* rp = pairs + (size_t)lr * (size_t)half_cols;
                float r_s = r_scale[lr];

                for (int32_t m = 0; m < M; m++) {
                    out[(size_t)m * (size_t)rows + (size_t)r] = 0.0f;
                }

                for (int32_t c0 = 0; c0 < cols; c0 += TILE_COLS) {
                    int32_t c_end = c0 + TILE_COLS;
                    if (c_end > cols) c_end = cols;
                    int32_t tile_len = c_end - c0;

                    for (int32_t tc = 0; tc < tile_len; tc += 16) {
                        int32_t c = c0 + tc;
                        __m128i raw8 = _mm_loadl_epi64((const __m128i*)(rp + (c >> 1)));
                        __m128i hi = _mm_and_si128(_mm_srli_epi16(raw8, 4), mask0F);
                        __m128i lo = _mm_and_si128(raw8, mask0F);
                        __m128i nib16 = _mm_unpacklo_epi8(hi, lo);

                        uint32_t esc = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(nib16, mask0F));
                        __m128i q16_i8 = _mm_shuffle_epi8(lut16, nib16);

                        __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q16_i8));
                        __m256 wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q16_i8, 8)));

                        _mm256_store_ps(w_tile + tc, wf0);
                        _mm256_store_ps(w_tile + tc + 8, wf1);
                        while (esc) {
                            unsigned long bit;
                            _BitScanForward(&bit, esc);
                            w_tile[tc + (int32_t)bit] = (float)decode_outlier(over8, super16, &o_idx, &s_idx);
                            esc &= esc - 1u;
                        }
                    }

                    int32_t m = 0;
                    for (; m + 3 < M; m += 4) {
                        const float* x0_p = x_scaled + (size_t)m * (size_t)cols + c0;
                        const float* x1_p = x0_p + cols;
                        const float* x2_p = x1_p + cols;
                        const float* x3_p = x2_p + cols;
                        __m256 a0 = _mm256_setzero_ps();
                        __m256 a1 = _mm256_setzero_ps();
                        __m256 a2 = _mm256_setzero_ps();
                        __m256 a3 = _mm256_setzero_ps();
                        for (int32_t tc = 0; tc < tile_len; tc += 8) {
                            __m256 wv = _mm256_load_ps(w_tile + tc);
                            a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0_p + tc), a0);
                            a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1_p + tc), a1);
                            a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x2_p + tc), a2);
                            a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x3_p + tc), a3);
                        }
                        out[(size_t)(m + 0) * (size_t)rows + (size_t)r] += hsum256_ps(a0);
                        out[(size_t)(m + 1) * (size_t)rows + (size_t)r] += hsum256_ps(a1);
                        out[(size_t)(m + 2) * (size_t)rows + (size_t)r] += hsum256_ps(a2);
                        out[(size_t)(m + 3) * (size_t)rows + (size_t)r] += hsum256_ps(a3);
                    }
                    for (; m < M; m++) {
                        const float* xp = x_scaled + (size_t)m * (size_t)cols + c0;
                        __m256 a0 = _mm256_setzero_ps();
                        __m256 a1 = _mm256_setzero_ps();
                        for (int32_t tc = 0; tc < tile_len; tc += 16) {
                            __m256 wv0 = _mm256_load_ps(w_tile + tc);
                            __m256 wv1 = _mm256_load_ps(w_tile + tc + 8);
                            a0 = _mm256_fmadd_ps(wv0, _mm256_loadu_ps(xp + tc), a0);
                            a1 = _mm256_fmadd_ps(wv1, _mm256_loadu_ps(xp + tc + 8), a1);
                        }
                        out[(size_t)m * (size_t)rows + (size_t)r] += hsum256_ps(_mm256_add_ps(a0, a1));
                    }
                }
                for (int32_t m = 0; m < M; m++) {
                    out[(size_t)m * (size_t)rows + (size_t)r] *= r_s;
                }
            }
        }
    }

    if (x_scaled != stack_x) free(x_scaled);
}

/* Direct Zero-Copy Embedding Lookup inside a 16-row QR Tile */
DPUF_EXPORT void dpuf_embed_qr_zd4_f32(
    int32_t n_tokens,
    int32_t cols,
    const int32_t* token_ids,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    const float* col_scale,
    float* out
) {
    const int32_t tile_stride = 16;
    for (int32_t tok = 0; tok < n_tokens; tok++) {
        int32_t r = token_ids[tok];
        int32_t t = r / tile_stride;
        int32_t lr = r % tile_stride;

        const uint8_t* tptr = tensor_base + tile_offsets[t];
        const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
        int32_t n_r = (int32_t)th->n_rows;
        const float* r_scale = (const float*)(tptr + sizeof(DpufQrTileHdr));
        const uint8_t* pairs = (const uint8_t*)(r_scale + n_r);
        size_t pairs_bytes = ((size_t)n_r * (size_t)cols + 1u) / 2u;
        const uint8_t* over8 = pairs + pairs_bytes;
        const uint16_t* super16 = (const uint16_t*)(over8 + th->n_over8);

        /* Count outliers in rows [0..lr-1] inside this tiny 16-row tile */
        uint32_t o_idx = 0;
        uint32_t s_idx = 0;
        for (int32_t prev_r = 0; prev_r < lr; prev_r++) {
            size_t base = (size_t)prev_r * (size_t)cols;
            for (int32_t c = 0; c < cols; c++) {
                size_t idx = base + (size_t)c;
                uint8_t b = pairs[idx >> 1];
                uint8_t nib = (idx & 1u) ? (b & 0x0Fu) : (b >> 4);
                if (nib == 15u) {
                    if (over8[o_idx++] == 255u) s_idx++;
                }
            }
        }

        float r_s = r_scale[lr];
        float* dst = out + (size_t)tok * (size_t)cols;
        size_t base = (size_t)lr * (size_t)cols;
        for (int32_t c = 0; c < cols; c++) {
            size_t idx = base + (size_t)c;
            uint8_t b = pairs[idx >> 1];
            uint8_t nib = (idx & 1u) ? (b & 0x0Fu) : (b >> 4);
            int32_t qi = (nib < 15u)
                ? ((int32_t)(nib >> 1) ^ -(int32_t)(nib & 1u))
                : decode_outlier(over8, super16, &o_idx, &s_idx);
            dst[c] = (float)qi * r_s * col_scale[c];
        }
    }
}


/* ============================================================================
 * FAST ENGINE KERNELS:
 *   - Unified Tile Verifier (Modes 8, 9, 10)
 *   - Mode 8 Tile-Range Kernel (eliminates per-matmul OpenMP fork/join)
 *   - Mode 9 (Varijanta A): Branchless Split-Dense int4 + Sparse Gap-Delta CSR + Q8_32 VPMADDUBSW
 *   - Mode 10 (Varijanta B): Walsh-Hadamard FWHT-64 + Pure Dense Blk32 int4 + Q8_32 VPMADDUBSW
 * ============================================================================ */

static __forceinline float fp16_bits_to_f32(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int32_t)h)));
}

static __forceinline void apply_sign_mask_f32(int32_t cols, const float* src, const uint8_t* sign_bits, float* dst) {
    for (int32_t b = 0; b < (cols >> 3); b++) {
        uint8_t m = sign_bits[b];
        int32_t c0 = b << 3;
        dst[c0 + 0] = (m & 1u)   ? -src[c0 + 0] : src[c0 + 0];
        dst[c0 + 1] = (m & 2u)   ? -src[c0 + 1] : src[c0 + 1];
        dst[c0 + 2] = (m & 4u)   ? -src[c0 + 2] : src[c0 + 2];
        dst[c0 + 3] = (m & 8u)   ? -src[c0 + 3] : src[c0 + 3];
        dst[c0 + 4] = (m & 16u)  ? -src[c0 + 4] : src[c0 + 4];
        dst[c0 + 5] = (m & 32u)  ? -src[c0 + 5] : src[c0 + 5];
        dst[c0 + 6] = (m & 64u)  ? -src[c0 + 6] : src[c0 + 6];
        dst[c0 + 7] = (m & 128u) ? -src[c0 + 7] : src[c0 + 7];
    }
}

static __forceinline void fwht64_block_f32(float* y) {
    for (int i = 0; i < 64; i += 2) {
        float a = y[i], b = y[i + 1];
        y[i] = a + b;
        y[i + 1] = a - b;
    }
    for (int i = 0; i < 64; i += 4) {
        for (int j = 0; j < 2; j++) {
            float a = y[i + j], b = y[i + j + 2];
            y[i + j] = a + b;
            y[i + j + 2] = a - b;
        }
    }
    for (int i = 0; i < 64; i += 8) {
        for (int j = 0; j < 4; j++) {
            float a = y[i + j], b = y[i + j + 4];
            y[i + j] = a + b;
            y[i + j + 4] = a - b;
        }
    }
    __m256 v0 = _mm256_loadu_ps(y + 0);
    __m256 v1 = _mm256_loadu_ps(y + 8);
    __m256 v2 = _mm256_loadu_ps(y + 16);
    __m256 v3 = _mm256_loadu_ps(y + 24);
    __m256 v4 = _mm256_loadu_ps(y + 32);
    __m256 v5 = _mm256_loadu_ps(y + 40);
    __m256 v6 = _mm256_loadu_ps(y + 48);
    __m256 v7 = _mm256_loadu_ps(y + 56);

    __m256 a0 = _mm256_add_ps(v0, v1), b0 = _mm256_sub_ps(v0, v1);
    __m256 a1 = _mm256_add_ps(v2, v3), b1 = _mm256_sub_ps(v2, v3);
    __m256 a2 = _mm256_add_ps(v4, v5), b2 = _mm256_sub_ps(v4, v5);
    __m256 a3 = _mm256_add_ps(v6, v7), b3 = _mm256_sub_ps(v6, v7);

    __m256 c0 = _mm256_add_ps(a0, a1), d0 = _mm256_sub_ps(a0, a1);
    __m256 c1 = _mm256_add_ps(b0, b1), d1 = _mm256_sub_ps(b0, b1);
    __m256 c2 = _mm256_add_ps(a2, a3), d2 = _mm256_sub_ps(a2, a3);
    __m256 c3 = _mm256_add_ps(b2, b3), d3 = _mm256_sub_ps(b2, b3);

    const __m256 inv8 = _mm256_set1_ps(0.125f);
    _mm256_storeu_ps(y + 0,  _mm256_mul_ps(_mm256_add_ps(c0, c2), inv8));
    _mm256_storeu_ps(y + 8,  _mm256_mul_ps(_mm256_add_ps(c1, c3), inv8));
    _mm256_storeu_ps(y + 16, _mm256_mul_ps(_mm256_add_ps(d0, d2), inv8));
    _mm256_storeu_ps(y + 24, _mm256_mul_ps(_mm256_add_ps(d1, d3), inv8));
    _mm256_storeu_ps(y + 32, _mm256_mul_ps(_mm256_sub_ps(c0, c2), inv8));
    _mm256_storeu_ps(y + 40, _mm256_mul_ps(_mm256_sub_ps(c1, c3), inv8));
    _mm256_storeu_ps(y + 48, _mm256_mul_ps(_mm256_sub_ps(d0, d2), inv8));
    _mm256_storeu_ps(y + 56, _mm256_mul_ps(_mm256_sub_ps(d1, d3), inv8));
}

DPUF_EXPORT void dpuf_fwht64_sign_f32(
    int32_t cols,
    const float* x_in,
    const uint8_t* sign_bits,
    float* x_out
) {
    if (sign_bits) {
        apply_sign_mask_f32(cols, x_in, sign_bits, x_out);
    } else if (x_out != x_in) {
        memcpy(x_out, x_in, (size_t)cols * sizeof(float));
    }
    for (int32_t c = 0; c < cols; c += 64) {
        fwht64_block_f32(x_out + c);
    }
}

DPUF_EXPORT void dpuf_scale_quantize_x_q8_32(
    int32_t cols,
    const float* x_in,
    const float* col_scale,
    float* x_scaled_out,
    int8_t* x_q8_out,
    float* x_blk_scale_out
) {
    const int32_t n_blk = cols >> 5;
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256i perm_idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i min_i8 = _mm256_set1_epi8(-127);

    for (int32_t b = 0; b < n_blk; b++) {
        int32_t c0 = b << 5;
        __m256 v0 = _mm256_loadu_ps(x_in + c0);
        __m256 v1 = _mm256_loadu_ps(x_in + c0 + 8);
        __m256 v2 = _mm256_loadu_ps(x_in + c0 + 16);
        __m256 v3 = _mm256_loadu_ps(x_in + c0 + 24);
        if (col_scale) {
            v0 = _mm256_mul_ps(v0, _mm256_loadu_ps(col_scale + c0));
            v1 = _mm256_mul_ps(v1, _mm256_loadu_ps(col_scale + c0 + 8));
            v2 = _mm256_mul_ps(v2, _mm256_loadu_ps(col_scale + c0 + 16));
            v3 = _mm256_mul_ps(v3, _mm256_loadu_ps(col_scale + c0 + 24));
        }
        if (x_scaled_out) {
            _mm256_storeu_ps(x_scaled_out + c0,      v0);
            _mm256_storeu_ps(x_scaled_out + c0 + 8,  v1);
            _mm256_storeu_ps(x_scaled_out + c0 + 16, v2);
            _mm256_storeu_ps(x_scaled_out + c0 + 24, v3);
        }
        __m256 m01 = _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask));
        __m256 m23 = _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask));
        __m256 mall = _mm256_max_ps(m01, m23);
        __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(mall), _mm256_extractf128_ps(mall, 1));
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
        float amax = _mm_cvtss_f32(m128);
        float d = amax / 127.0f;
        float inv_d = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
        x_blk_scale_out[b] = d;

        __m256 vinv = _mm256_set1_ps(inv_d);
        __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vinv));
        __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vinv));
        __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vinv));
        __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vinv));
        __m256i p01 = _mm256_packs_epi32(q0, q1);
        __m256i p23 = _mm256_packs_epi32(q2, q3);
        __m256i p8  = _mm256_packs_epi16(p01, p23);
        __m256i p_ord = _mm256_permutevar8x32_epi32(p8, perm_idx);
        p_ord = _mm256_max_epi8(p_ord, min_i8);
        _mm256_storeu_si256((__m256i*)(x_q8_out + c0), p_ord);
    }
}

DPUF_EXPORT int32_t dpuf_qr_verify_tiles_ex(
    int32_t mode,
    int32_t n_tiles,
    int32_t cols,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    uint8_t* tile_ok_out
) {
    init_crc32_table();
    int32_t bad_count = 0;
    int32_t t;
    #pragma omp parallel for reduction(+:bad_count) schedule(dynamic, 4) if(n_tiles >= 16)
    for (t = 0; t < n_tiles; t++) {
        const uint8_t* tptr = tensor_base + tile_offsets[t];
        const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
        uint32_t body_bytes = 0;
        if (mode == DPUF_MODE_QR_ZD4) {
            body_bytes = (uint32_t)(
                (size_t)th->n_rows * 4u
                + ((size_t)th->n_rows * (size_t)cols + 1u) / 2u
                + (size_t)th->n_over8
                + (size_t)th->n_super16 * 2u
            );
        } else if (mode == DPUF_MODE_QR_SPLIT4) {
            body_bytes = (uint32_t)(
                (size_t)th->n_rows * 4u
                + (size_t)th->n_rows * 2u
                + ((size_t)th->n_rows * (size_t)cols) / 2u
                + (size_t)th->n_over8 * 2u
                + (size_t)th->n_super16 * 2u
            );
        } else if (mode == DPUF_MODE_QR_HAD4) {
            int32_t n_blk = cols >> 5;
            body_bytes = (uint32_t)(
                (size_t)th->n_rows * (size_t)n_blk * 2u
                + ((size_t)th->n_rows * (size_t)cols) / 2u
            );
        } else if (mode == DPUF_MODE_QR_Q8) {
            int32_t n_blk = cols >> 5;
            body_bytes = (uint32_t)(
                (size_t)th->n_rows * (size_t)n_blk * 2u
                + (size_t)th->n_rows * (size_t)cols
            );
        } else if (mode == DPUF_MODE_QR_HAD4_PACKED) {
            int32_t n_blk = cols >> 5;
            int32_t n_8g = ((int32_t)th->n_rows + 7) >> 3;
            body_bytes = (uint32_t)((size_t)n_8g * (size_t)n_blk * 144u);
        } else if (mode == DPUF_MODE_QR_HAD4_ASYM) {
            int32_t n_blk64 = cols >> 6;
            int32_t n_8g = ((int32_t)th->n_rows + 7) >> 3;
            body_bytes = (uint32_t)((size_t)n_8g * (size_t)n_blk64 * 288u);
        } else if (mode == DPUF_MODE_QR_HAD6) {
            int32_t n_blk64 = cols >> 6;
            int32_t n_8g = ((int32_t)th->n_rows + 7) >> 3;
            body_bytes = (uint32_t)((size_t)n_8g * (size_t)n_blk64 * 416u);
        }
        uint32_t calc = dpuf_crc32_bytes(tptr + sizeof(DpufQrTileHdr), body_bytes);
        uint8_t ok = (calc == th->tile_crc32) ? 1u : 0u;
        tile_ok_out[t] = ok;
        if (!ok) bad_count += 1;
    }
    return bad_count;
}

void dpuf_matmul_qr_zd4_tile_range(
    int32_t M,
    int32_t rows,
    int32_t cols,
    int32_t t_start,
    int32_t t_end,
    const float* x_scaled,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    float* out
) {
    const __m128i lut16 = _mm_setr_epi8(
        0, -1, 1, -2, 2, -3, 3, -4,
        4, -5, 5, -6, 6, -7, 7, 0
    );
    const __m128i mask0F = _mm_set1_epi8(0x0F);
    const int32_t tile_stride = 16;
    const int32_t half_cols = cols >> 1;

    if (M == 1) {
        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const float* r_scale = (const float*)(tptr + sizeof(DpufQrTileHdr));
            const uint8_t* pairs = (const uint8_t*)(r_scale + n_r);
            const uint8_t* over8 = pairs + (size_t)n_r * (size_t)half_cols;
            const uint16_t* super16 = (const uint16_t*)(over8 + th->n_over8);

            uint32_t o_idx = 0;
            uint32_t s_idx = 0;
            for (int32_t lr = 0; lr < n_r; lr++) {
                const uint8_t* rp = pairs + (size_t)lr * (size_t)half_cols;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                for (int32_t c = 0; c < cols; c += 16) {
                    __m128i raw8 = _mm_loadl_epi64((const __m128i*)(rp + (c >> 1)));
                    __m128i hi = _mm_and_si128(_mm_srli_epi16(raw8, 4), mask0F);
                    __m128i lo = _mm_and_si128(raw8, mask0F);
                    __m128i nib16 = _mm_unpacklo_epi8(hi, lo);
                    uint32_t esc = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(nib16, mask0F));
                    __m128i q16_i8 = _mm_shuffle_epi8(lut16, nib16);
                    __m256 wf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q16_i8));
                    __m256 wf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q16_i8, 8)));
                    if (esc) {
                        float tmp[16];
                        _mm256_storeu_ps(tmp, wf0);
                        _mm256_storeu_ps(tmp + 8, wf1);
                        while (esc) {
                            unsigned long bit;
                            _BitScanForward(&bit, esc);
                            tmp[bit] = (float)decode_outlier(over8, super16, &o_idx, &s_idx);
                            esc &= esc - 1u;
                        }
                        wf0 = _mm256_loadu_ps(tmp);
                        wf1 = _mm256_loadu_ps(tmp + 8);
                    }
                    acc0 = _mm256_fmadd_ps(wf0, _mm256_loadu_ps(x_scaled + c), acc0);
                    acc1 = _mm256_fmadd_ps(wf1, _mm256_loadu_ps(x_scaled + c + 8), acc1);
                }
                out[r0 + lr] = hsum256_ps(_mm256_add_ps(acc0, acc1)) * r_scale[lr];
            }
        }
    } else {
        enum { TILE_COLS = 1024 };
        __declspec(align(32)) float w_tile[TILE_COLS];
        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const float* r_scale = (const float*)(tptr + sizeof(DpufQrTileHdr));
            const uint8_t* pairs = (const uint8_t*)(r_scale + n_r);
            const uint8_t* over8 = pairs + (size_t)n_r * (size_t)half_cols;
            const uint16_t* super16 = (const uint16_t*)(over8 + th->n_over8);

            uint32_t o_idx = 0;
            uint32_t s_idx = 0;
            for (int32_t lr = 0; lr < n_r; lr++) {
                int32_t r = r0 + lr;
                const uint8_t* rp = pairs + (size_t)lr * (size_t)half_cols;
                float r_s = r_scale[lr];
                for (int32_t m = 0; m < M; m++) {
                    out[(size_t)m * (size_t)rows + (size_t)r] = 0.0f;
                }
                for (int32_t c0 = 0; c0 < cols; c0 += TILE_COLS) {
                    int32_t c_end = (c0 + TILE_COLS < cols) ? (c0 + TILE_COLS) : cols;
                    int32_t tile_len = c_end - c0;
                    for (int32_t tc = 0; tc < tile_len; tc += 16) {
                        int32_t c = c0 + tc;
                        __m128i raw8 = _mm_loadl_epi64((const __m128i*)(rp + (c >> 1)));
                        __m128i hi = _mm_and_si128(_mm_srli_epi16(raw8, 4), mask0F);
                        __m128i lo = _mm_and_si128(raw8, mask0F);
                        __m128i nib16 = _mm_unpacklo_epi8(hi, lo);
                        uint32_t esc = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi8(nib16, mask0F));
                        __m128i q16_i8 = _mm_shuffle_epi8(lut16, nib16);
                        _mm256_store_ps(w_tile + tc, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q16_i8)));
                        _mm256_store_ps(w_tile + tc + 8, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q16_i8, 8))));
                        while (esc) {
                            unsigned long bit;
                            _BitScanForward(&bit, esc);
                            w_tile[tc + (int32_t)bit] = (float)decode_outlier(over8, super16, &o_idx, &s_idx);
                            esc &= esc - 1u;
                        }
                    }
                    int32_t m = 0;
                    for (; m + 3 < M; m += 4) {
                        const float* x0_p = x_scaled + (size_t)m * (size_t)cols + c0;
                        const float* x1_p = x0_p + cols;
                        const float* x2_p = x1_p + cols;
                        const float* x3_p = x2_p + cols;
                        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
                        for (int32_t tc = 0; tc < tile_len; tc += 8) {
                            __m256 wv = _mm256_load_ps(w_tile + tc);
                            a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0_p + tc), a0);
                            a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1_p + tc), a1);
                            a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x2_p + tc), a2);
                            a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x3_p + tc), a3);
                        }
                        out[(size_t)(m + 0) * (size_t)rows + (size_t)r] += hsum256_ps(a0);
                        out[(size_t)(m + 1) * (size_t)rows + (size_t)r] += hsum256_ps(a1);
                        out[(size_t)(m + 2) * (size_t)rows + (size_t)r] += hsum256_ps(a2);
                        out[(size_t)(m + 3) * (size_t)rows + (size_t)r] += hsum256_ps(a3);
                    }
                    for (; m < M; m++) {
                        const float* xp = x_scaled + (size_t)m * (size_t)cols + c0;
                        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                        for (int32_t tc = 0; tc < tile_len; tc += 16) {
                            a0 = _mm256_fmadd_ps(_mm256_load_ps(w_tile + tc), _mm256_loadu_ps(xp + tc), a0);
                            a1 = _mm256_fmadd_ps(_mm256_load_ps(w_tile + tc + 8), _mm256_loadu_ps(xp + tc + 8), a1);
                        }
                        out[(size_t)m * (size_t)rows + (size_t)r] += hsum256_ps(_mm256_add_ps(a0, a1));
                    }
                }
                for (int32_t m = 0; m < M; m++) {
                    out[(size_t)m * (size_t)rows + (size_t)r] *= r_s;
                }
            }
        }
    }
}

/* ============================================================================
 * MODE 9 (Varijanta A): Split-Dense [-8..+7] 4-bit + Sparse Gap-Delta CSR
 *   Tile body: [row_scale fp32 * n_r][row_nnz u16 * n_r][base_pairs (n_r*cols/2)][ov_pairs u16][super16 i16]
 * ============================================================================ */

static __forceinline __m256i unpack_q32_from_16B(const uint8_t* rp16, __m256i mask0F, __m256i eights) {
    __m128i raw16 = _mm_loadu_si128((const __m128i*)rp16);
    __m256i u32 = _mm256_and_si256(
        _mm256_inserti128_si256(_mm256_castsi128_si256(raw16), _mm_srli_epi16(raw16, 4), 1),
        mask0F
    );
    return _mm256_sub_epi8(u32, eights);
}

static __forceinline void unpack_32_to_f32(
    const uint8_t* rp16,
    float scale,
    float* dst32,
    __m128i mask0F_128,
    __m128i eights_128
) {
    __m128i raw16 = _mm_loadu_si128((const __m128i*)rp16);
    __m128i lo = _mm_sub_epi8(_mm_and_si128(raw16, mask0F_128), eights_128);
    __m128i hi = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(raw16, 4), mask0F_128), eights_128);
    __m256 vs = _mm256_set1_ps(scale);
    _mm256_store_ps(dst32 + 0,  _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), vs));
    _mm256_store_ps(dst32 + 8,  _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), vs));
    _mm256_store_ps(dst32 + 16, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), vs));
    _mm256_store_ps(dst32 + 24, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), vs));
}

static __forceinline void transpose_8x8_ps(__m256 row[8]) {
    __m256 t0 = _mm256_unpacklo_ps(row[0], row[1]);
    __m256 t1 = _mm256_unpackhi_ps(row[0], row[1]);
    __m256 t2 = _mm256_unpacklo_ps(row[2], row[3]);
    __m256 t3 = _mm256_unpackhi_ps(row[2], row[3]);
    __m256 t4 = _mm256_unpacklo_ps(row[4], row[5]);
    __m256 t5 = _mm256_unpackhi_ps(row[4], row[5]);
    __m256 t6 = _mm256_unpacklo_ps(row[6], row[7]);
    __m256 t7 = _mm256_unpackhi_ps(row[6], row[7]);
    __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    __m256 u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    row[0] = _mm256_permute2f128_ps(u0, u4, 0x20);
    row[1] = _mm256_permute2f128_ps(u1, u5, 0x20);
    row[2] = _mm256_permute2f128_ps(u2, u6, 0x20);
    row[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
    row[4] = _mm256_permute2f128_ps(u0, u4, 0x31);
    row[5] = _mm256_permute2f128_ps(u1, u5, 0x31);
    row[6] = _mm256_permute2f128_ps(u2, u6, 0x31);
    row[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
}

static void gemm_blk32_8x6_q8_acc(
    int32_t M_cur,
    int32_t cols,
    int32_t b,
    int32_t lr_off,
    const __m256i w_tr[8],
    __m256i wsum128,
    __m256 v_wscale,
    const int8_t* x_q8_base,
    const float* x_scale_base,
    int32_t n_blk,
    float tile_acc[256][16]
) {
    const __m256i mask80 = _mm256_set1_epi8((char)0x80);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const int32_t c0 = b << 5;
    int32_t m = 0;

    for (; m + 5 < M_cur; m += 6) {
        __declspec(align(32)) uint32_t xb[6][8];
        for (int j = 0; j < 6; j++) {
            __m256i xu = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(x_q8_base + (size_t)(m + j) * (size_t)cols + c0)), mask80);
            _mm256_store_si256((__m256i*)xb[j], xu);
        }
        __m256i a0 = _mm256_setzero_si256(), a1 = _mm256_setzero_si256(), a2 = _mm256_setzero_si256();
        __m256i a3 = _mm256_setzero_si256(), a4 = _mm256_setzero_si256(), a5 = _mm256_setzero_si256();
        for (int g = 0; g < 8; g++) {
            __m256i wt = w_tr[g];
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi32((int32_t)xb[0][g]), wt), ones16));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi32((int32_t)xb[1][g]), wt), ones16));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi32((int32_t)xb[2][g]), wt), ones16));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi32((int32_t)xb[3][g]), wt), ones16));
            a4 = _mm256_add_epi32(a4, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi32((int32_t)xb[4][g]), wt), ones16));
            a5 = _mm256_add_epi32(a5, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi32((int32_t)xb[5][g]), wt), ones16));
        }
        a0 = _mm256_sub_epi32(a0, wsum128);
        a1 = _mm256_sub_epi32(a1, wsum128);
        a2 = _mm256_sub_epi32(a2, wsum128);
        a3 = _mm256_sub_epi32(a3, wsum128);
        a4 = _mm256_sub_epi32(a4, wsum128);
        a5 = _mm256_sub_epi32(a5, wsum128);

        __m256 s0 = _mm256_mul_ps(v_wscale, _mm256_set1_ps(x_scale_base[(size_t)(m + 0) * (size_t)n_blk + b]));
        __m256 s1 = _mm256_mul_ps(v_wscale, _mm256_set1_ps(x_scale_base[(size_t)(m + 1) * (size_t)n_blk + b]));
        __m256 s2 = _mm256_mul_ps(v_wscale, _mm256_set1_ps(x_scale_base[(size_t)(m + 2) * (size_t)n_blk + b]));
        __m256 s3 = _mm256_mul_ps(v_wscale, _mm256_set1_ps(x_scale_base[(size_t)(m + 3) * (size_t)n_blk + b]));
        __m256 s4 = _mm256_mul_ps(v_wscale, _mm256_set1_ps(x_scale_base[(size_t)(m + 4) * (size_t)n_blk + b]));
        __m256 s5 = _mm256_mul_ps(v_wscale, _mm256_set1_ps(x_scale_base[(size_t)(m + 5) * (size_t)n_blk + b]));

        _mm256_store_ps(&tile_acc[m + 0][lr_off], _mm256_fmadd_ps(_mm256_cvtepi32_ps(a0), s0, _mm256_load_ps(&tile_acc[m + 0][lr_off])));
        _mm256_store_ps(&tile_acc[m + 1][lr_off], _mm256_fmadd_ps(_mm256_cvtepi32_ps(a1), s1, _mm256_load_ps(&tile_acc[m + 1][lr_off])));
        _mm256_store_ps(&tile_acc[m + 2][lr_off], _mm256_fmadd_ps(_mm256_cvtepi32_ps(a2), s2, _mm256_load_ps(&tile_acc[m + 2][lr_off])));
        _mm256_store_ps(&tile_acc[m + 3][lr_off], _mm256_fmadd_ps(_mm256_cvtepi32_ps(a3), s3, _mm256_load_ps(&tile_acc[m + 3][lr_off])));
        _mm256_store_ps(&tile_acc[m + 4][lr_off], _mm256_fmadd_ps(_mm256_cvtepi32_ps(a4), s4, _mm256_load_ps(&tile_acc[m + 4][lr_off])));
        _mm256_store_ps(&tile_acc[m + 5][lr_off], _mm256_fmadd_ps(_mm256_cvtepi32_ps(a5), s5, _mm256_load_ps(&tile_acc[m + 5][lr_off])));
    }
    for (; m < M_cur; m++) {
        __declspec(align(32)) uint32_t xb[8];
        __m256i xu = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(x_q8_base + (size_t)m * (size_t)cols + c0)), mask80);
        _mm256_store_si256((__m256i*)xb, xu);
        __m256i a = _mm256_setzero_si256();
        for (int g = 0; g < 8; g++) {
            a = _mm256_add_epi32(a, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi32((int32_t)xb[g]), w_tr[g]), ones16));
        }
        a = _mm256_sub_epi32(a, wsum128);
        __m256 sm = _mm256_mul_ps(v_wscale, _mm256_set1_ps(x_scale_base[(size_t)m * (size_t)n_blk + b]));
        _mm256_store_ps(&tile_acc[m][lr_off], _mm256_fmadd_ps(_mm256_cvtepi32_ps(a), sm, _mm256_load_ps(&tile_acc[m][lr_off])));
    }
}

static void transpose_16x64_to_wt(const float w_raw[16][64], float w_t[64][16], int32_t chunk_len) {
    for (int32_t kb = 0; kb < chunk_len; kb += 8) {
        __m256 lo[8], hi[8];
        for (int i = 0; i < 8; i++) {
            lo[i] = _mm256_load_ps(&w_raw[i][kb]);
            hi[i] = _mm256_load_ps(&w_raw[i + 8][kb]);
        }
        transpose_8x8_ps(lo);
        transpose_8x8_ps(hi);
        for (int k = 0; k < 8; k++) {
            _mm256_store_ps(&w_t[kb + k][0], lo[k]);
            _mm256_store_ps(&w_t[kb + k][8], hi[k]);
        }
    }
}

static void gemm_chunk_16x6_transposed_f32(
    int32_t M_cur,
    int32_t cols,
    int32_t c0,
    int32_t chunk_len,
    const float w_t[64][16],
    const float* x_mat_base,
    float tile_acc[256][16],
    int32_t init_zero
) {
    int32_t m = 0;
    for (; m + 5 < M_cur; m += 6) {
        const float* xp0 = x_mat_base + (size_t)(m + 0) * (size_t)cols + c0;
        const float* xp1 = xp0 + cols;
        const float* xp2 = xp1 + cols;
        const float* xp3 = xp2 + cols;
        const float* xp4 = xp3 + cols;
        const float* xp5 = xp4 + cols;

        __m256 a0_0, a0_1, a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, a4_0, a4_1, a5_0, a5_1;
        if (init_zero) {
            a0_0 = _mm256_setzero_ps(); a0_1 = _mm256_setzero_ps();
            a1_0 = _mm256_setzero_ps(); a1_1 = _mm256_setzero_ps();
            a2_0 = _mm256_setzero_ps(); a2_1 = _mm256_setzero_ps();
            a3_0 = _mm256_setzero_ps(); a3_1 = _mm256_setzero_ps();
            a4_0 = _mm256_setzero_ps(); a4_1 = _mm256_setzero_ps();
            a5_0 = _mm256_setzero_ps(); a5_1 = _mm256_setzero_ps();
        } else {
            a0_0 = _mm256_load_ps(&tile_acc[m + 0][0]); a0_1 = _mm256_load_ps(&tile_acc[m + 0][8]);
            a1_0 = _mm256_load_ps(&tile_acc[m + 1][0]); a1_1 = _mm256_load_ps(&tile_acc[m + 1][8]);
            a2_0 = _mm256_load_ps(&tile_acc[m + 2][0]); a2_1 = _mm256_load_ps(&tile_acc[m + 2][8]);
            a3_0 = _mm256_load_ps(&tile_acc[m + 3][0]); a3_1 = _mm256_load_ps(&tile_acc[m + 3][8]);
            a4_0 = _mm256_load_ps(&tile_acc[m + 4][0]); a4_1 = _mm256_load_ps(&tile_acc[m + 4][8]);
            a5_0 = _mm256_load_ps(&tile_acc[m + 5][0]); a5_1 = _mm256_load_ps(&tile_acc[m + 5][8]);
        }

        for (int32_t k = 0; k < chunk_len; k++) {
            __m256 w0 = _mm256_load_ps(&w_t[k][0]);
            __m256 w1 = _mm256_load_ps(&w_t[k][8]);
            __m256 x0 = _mm256_broadcast_ss(xp0 + k);
            a0_0 = _mm256_fmadd_ps(w0, x0, a0_0); a0_1 = _mm256_fmadd_ps(w1, x0, a0_1);
            __m256 x1 = _mm256_broadcast_ss(xp1 + k);
            a1_0 = _mm256_fmadd_ps(w0, x1, a1_0); a1_1 = _mm256_fmadd_ps(w1, x1, a1_1);
            __m256 x2 = _mm256_broadcast_ss(xp2 + k);
            a2_0 = _mm256_fmadd_ps(w0, x2, a2_0); a2_1 = _mm256_fmadd_ps(w1, x2, a2_1);
            __m256 x3 = _mm256_broadcast_ss(xp3 + k);
            a3_0 = _mm256_fmadd_ps(w0, x3, a3_0); a3_1 = _mm256_fmadd_ps(w1, x3, a3_1);
            __m256 x4 = _mm256_broadcast_ss(xp4 + k);
            a4_0 = _mm256_fmadd_ps(w0, x4, a4_0); a4_1 = _mm256_fmadd_ps(w1, x4, a4_1);
            __m256 x5 = _mm256_broadcast_ss(xp5 + k);
            a5_0 = _mm256_fmadd_ps(w0, x5, a5_0); a5_1 = _mm256_fmadd_ps(w1, x5, a5_1);
        }

        _mm256_store_ps(&tile_acc[m + 0][0], a0_0); _mm256_store_ps(&tile_acc[m + 0][8], a0_1);
        _mm256_store_ps(&tile_acc[m + 1][0], a1_0); _mm256_store_ps(&tile_acc[m + 1][8], a1_1);
        _mm256_store_ps(&tile_acc[m + 2][0], a2_0); _mm256_store_ps(&tile_acc[m + 2][8], a2_1);
        _mm256_store_ps(&tile_acc[m + 3][0], a3_0); _mm256_store_ps(&tile_acc[m + 3][8], a3_1);
        _mm256_store_ps(&tile_acc[m + 4][0], a4_0); _mm256_store_ps(&tile_acc[m + 4][8], a4_1);
        _mm256_store_ps(&tile_acc[m + 5][0], a5_0); _mm256_store_ps(&tile_acc[m + 5][8], a5_1);
    }
    for (; m < M_cur; m++) {
        const float* xpm = x_mat_base + (size_t)m * (size_t)cols + c0;
        __m256 a0 = init_zero ? _mm256_setzero_ps() : _mm256_load_ps(&tile_acc[m][0]);
        __m256 a1 = init_zero ? _mm256_setzero_ps() : _mm256_load_ps(&tile_acc[m][8]);
        for (int32_t k = 0; k < chunk_len; k++) {
            __m256 xm = _mm256_broadcast_ss(xpm + k);
            a0 = _mm256_fmadd_ps(_mm256_load_ps(&w_t[k][0]), xm, a0);
            a1 = _mm256_fmadd_ps(_mm256_load_ps(&w_t[k][8]), xm, a1);
        }
        _mm256_store_ps(&tile_acc[m][0], a0);
        _mm256_store_ps(&tile_acc[m][8], a1);
    }
}

static __forceinline void finalize_wtr_8x32(
    __m256 row_f[8],
    __m256i w_tr[8],
    __m256i* wsum128_out
) {
    transpose_8x8_ps(row_f);
    const __m256i ones_u8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i wsum = _mm256_setzero_si256();
    for (int g = 0; g < 8; g++) {
        w_tr[g] = _mm256_castps_si256(row_f[g]);
        wsum = _mm256_add_epi32(wsum, _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, w_tr[g]), ones16));
    }
    *wsum128_out = _mm256_slli_epi32(wsum, 7);
}

static __forceinline void prep_wtr_8x32(
    int32_t n_8,
    const uint8_t* pairs_base,
    int32_t half_cols,
    int32_t b,
    __m256i mask0F,
    __m256i eights,
    __m256i w_tr[8],
    __m256i* wsum128_out
) {
    __m256 row_f[8];
    for (int ri = 0; ri < n_8; ri++) {
        __m256i q = unpack_q32_from_16B(pairs_base + (size_t)ri * (size_t)half_cols + (b << 4), mask0F, eights);
        row_f[ri] = _mm256_castsi256_ps(q);
    }
    for (int ri = n_8; ri < 8; ri++) row_f[ri] = _mm256_setzero_ps();
    finalize_wtr_8x32(row_f, w_tr, wsum128_out);
}

static __forceinline void prep_wtr_8x32_q8(
    int32_t n_8,
    const int8_t* wq_base,
    int32_t cols,
    int32_t b,
    __m256i w_tr[8],
    __m256i* wsum128_out
) {
    __m256 row_f[8];
    for (int ri = 0; ri < n_8; ri++) {
        __m256i w = _mm256_loadu_si256((const __m256i*)(wq_base + (size_t)ri * (size_t)cols + (b << 5)));
        row_f[ri] = _mm256_castsi256_ps(w);
    }
    for (int ri = n_8; ri < 8; ri++) row_f[ri] = _mm256_setzero_ps();
    finalize_wtr_8x32(row_f, w_tr, wsum128_out);
}

static __forceinline float sparse_dot_row_f32(
    uint16_t nnz,
    const uint16_t* ov,
    const int16_t* super16,
    uint32_t* s_idx,
    const float* x_scaled,
    uint32_t n_super16
) {
    float acc = 0.0f;
    int32_t col = 0;
    if (n_super16 == 0u) {
        for (uint16_t k = 0; k < nnz; k++) {
            uint16_t pv = ov[k];
            col += (int32_t)(pv & 0xFFu);
            acc += (float)(int8_t)(pv >> 8) * x_scaled[col];
        }
    } else {
        for (uint16_t k = 0; k < nnz; k++) {
            uint16_t pv = ov[k];
            col += (int32_t)(pv & 0xFFu);
            int32_t dv = (int32_t)(int8_t)(pv >> 8);
            if (dv == -128) dv = (int32_t)super16[(*s_idx)++];
            acc += (float)dv * x_scaled[col];
        }
    }
    return acc;
}

void dpuf_matmul_qr_split4_tile_range(
    int32_t M,
    int32_t rows,
    int32_t cols,
    int32_t t_start,
    int32_t t_end,
    const float* x_scaled,
    const int8_t* x_q8,
    const float* x_blk_scale,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    float* out
) {
    const __m256i mask0F = _mm256_set1_epi8(0x0F);
    const __m256i eights = _mm256_set1_epi8(8);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const int32_t n_blk = cols >> 5;
    const int32_t half_cols = cols >> 1;
    const int32_t tile_stride = 16;

    if (M == 1) {
        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const float* r_scale = (const float*)(tptr + sizeof(DpufQrTileHdr));
            const uint16_t* row_nnz = (const uint16_t*)(r_scale + n_r);
            const uint8_t* pairs = (const uint8_t*)(row_nnz + n_r);
            const uint16_t* ov_pairs = (const uint16_t*)(pairs + (size_t)n_r * (size_t)half_cols);
            const int16_t* super16 = (const int16_t*)(ov_pairs + th->n_over8);

            uint32_t o_idx = 0;
            uint32_t s_idx = 0;
            int32_t lr = 0;

            for (; lr + 1 < n_r; lr += 2) {
                const uint8_t* rp0 = pairs + (size_t)lr * (size_t)half_cols;
                const uint8_t* rp1 = rp0 + half_cols;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();

                for (int32_t b = 0; b < n_blk; b++) {
                    __m256i xq32 = _mm256_loadu_si256((const __m256i*)(x_q8 + (b << 5)));
                    __m256 vxs = _mm256_set1_ps(x_blk_scale[b]);

                    __m256i q0 = unpack_q32_from_16B(rp0 + (b << 4), mask0F, eights);
                    __m256i d0 = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(_mm256_sign_epi8(q0, q0), _mm256_sign_epi8(xq32, q0)),
                        ones16
                    );
                    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), vxs, acc0);

                    __m256i q1 = unpack_q32_from_16B(rp1 + (b << 4), mask0F, eights);
                    __m256i d1 = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(_mm256_sign_epi8(q1, q1), _mm256_sign_epi8(xq32, q1)),
                        ones16
                    );
                    acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1), vxs, acc1);
                }

                float s0 = hsum256_ps(acc0);
                uint16_t nnz0 = row_nnz[lr];
                if (nnz0) {
                    s0 += sparse_dot_row_f32(nnz0, ov_pairs + o_idx, super16, &s_idx, x_scaled, th->n_super16);
                    o_idx += nnz0;
                }
                out[r0 + lr] = s0 * r_scale[lr];

                float s1 = hsum256_ps(acc1);
                uint16_t nnz1 = row_nnz[lr + 1];
                if (nnz1) {
                    s1 += sparse_dot_row_f32(nnz1, ov_pairs + o_idx, super16, &s_idx, x_scaled, th->n_super16);
                    o_idx += nnz1;
                }
                out[r0 + lr + 1] = s1 * r_scale[lr + 1];
            }

            for (; lr < n_r; lr++) {
                const uint8_t* rp = pairs + (size_t)lr * (size_t)half_cols;
                __m256 acc = _mm256_setzero_ps();
                for (int32_t b = 0; b < n_blk; b++) {
                    __m256i xq32 = _mm256_loadu_si256((const __m256i*)(x_q8 + (b << 5)));
                    __m256i q = unpack_q32_from_16B(rp + (b << 4), mask0F, eights);
                    __m256i d = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(_mm256_sign_epi8(q, q), _mm256_sign_epi8(xq32, q)),
                        ones16
                    );
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(x_blk_scale[b]), acc);
                }
                float s = hsum256_ps(acc);
                uint16_t nnz = row_nnz[lr];
                if (nnz) {
                    s += sparse_dot_row_f32(nnz, ov_pairs + o_idx, super16, &s_idx, x_scaled, th->n_super16);
                    o_idx += nnz;
                }
                out[r0 + lr] = s * r_scale[lr];
            }
        }
    } else if (M >= 8) {
        const __m128i mask0F_128 = _mm_set1_epi8(0x0F);
        const __m128i eights_128 = _mm_set1_epi8(8);
        __declspec(align(32)) float w_raw[16][64];
        __declspec(align(32)) float w_t[64][16];
        __declspec(align(32)) float tile_acc[256][16];
        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const float* r_scale = (const float*)(tptr + sizeof(DpufQrTileHdr));
            const uint16_t* row_nnz = (const uint16_t*)(r_scale + n_r);
            const uint8_t* pairs = (const uint8_t*)(row_nnz + n_r);
            const uint16_t* ov_pairs = (const uint16_t*)(pairs + (size_t)n_r * (size_t)half_cols);
            const int16_t* super16 = (const int16_t*)(ov_pairs + th->n_over8);

            uint32_t o_beg_init[16], s_beg_init[16];
            uint32_t oi = 0, si = 0;
            for (int32_t lr = 0; lr < n_r; lr++) {
                o_beg_init[lr] = oi;
                s_beg_init[lr] = si;
                uint16_t nnz = row_nnz[lr];
                for (uint16_t k = 0; k < nnz; k++) {
                    if ((int8_t)(ov_pairs[oi++] >> 8) == -128) si++;
                }
            }
            for (int32_t lr = n_r; lr < 16; lr++) {
                memset(w_raw[lr], 0, sizeof(w_raw[lr]));
            }

            for (int32_t m0 = 0; m0 < M; m0 += 256) {
                int32_t M_cur = (m0 + 256 <= M) ? 256 : (M - m0);
                const float* x_base = x_scaled + (size_t)m0 * (size_t)cols;
                uint32_t o_beg[16], s_beg[16], k_cur[16];
                int32_t col_cur[16];
                for (int32_t lr = 0; lr < n_r; lr++) {
                    o_beg[lr] = o_beg_init[lr];
                    s_beg[lr] = s_beg_init[lr];
                    k_cur[lr] = 0;
                    col_cur[lr] = 0;
                }
                for (int32_t c0 = 0; c0 < cols; c0 += 64) {
                    int32_t chunk_len = (c0 + 64 <= cols) ? 64 : (cols - c0);
                    int32_t b0 = c0 >> 5;
                    int32_t b_cnt = chunk_len >> 5;
                    for (int32_t lr = 0; lr < n_r; lr++) {
                        float rs = r_scale[lr];
                        const uint8_t* rp = pairs + (size_t)lr * (size_t)half_cols;
                        for (int32_t kb = 0; kb < b_cnt; kb++) {
                            unpack_32_to_f32(rp + ((b0 + kb) << 4), rs, &w_raw[lr][kb << 5], mask0F_128, eights_128);
                        }
                        uint16_t nnz = row_nnz[lr];
                        while (k_cur[lr] < nnz) {
                            uint16_t pv = ov_pairs[o_beg[lr] + k_cur[lr]];
                            int32_t ncol = col_cur[lr] + (int32_t)(pv & 0xFFu);
                            if (ncol >= c0 + chunk_len) break;
                            col_cur[lr] = ncol;
                            int32_t dv = (int32_t)(int8_t)(pv >> 8);
                            if (dv == -128) dv = (int32_t)super16[s_beg[lr]++];
                            w_raw[lr][ncol - c0] += (float)dv * rs;
                            k_cur[lr]++;
                        }
                    }
                    transpose_16x64_to_wt(w_raw, w_t, chunk_len);
                    gemm_chunk_16x6_transposed_f32(M_cur, cols, c0, chunk_len, w_t, x_base, tile_acc, (c0 == 0));
                }
                for (int32_t m = 0; m < M_cur; m++) {
                    memcpy(out + (size_t)(m0 + m) * (size_t)rows + (size_t)r0, tile_acc[m], (size_t)n_r * sizeof(float));
                }
            }
        }
    } else {
        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const float* r_scale = (const float*)(tptr + sizeof(DpufQrTileHdr));
            const uint16_t* row_nnz = (const uint16_t*)(r_scale + n_r);
            const uint8_t* pairs = (const uint8_t*)(row_nnz + n_r);
            const uint16_t* ov_pairs = (const uint16_t*)(pairs + (size_t)n_r * (size_t)half_cols);
            const int16_t* super16 = (const int16_t*)(ov_pairs + th->n_over8);

            uint32_t o_idx = 0;
            uint32_t s_idx = 0;
            for (int32_t lr = 0; lr < n_r; lr++) {
                int32_t r = r0 + lr;
                const uint8_t* rp = pairs + (size_t)lr * (size_t)half_cols;
                float r_s = r_scale[lr];
                uint16_t nnz = row_nnz[lr];
                const uint16_t* row_ov = ov_pairs + o_idx;
                uint32_t s_next = s_idx;

                int32_t m = 0;
                for (; m + 3 < M; m += 4) {
                    const int8_t* xq0 = x_q8 + (size_t)(m + 0) * (size_t)cols;
                    const int8_t* xq1 = xq0 + cols;
                    const int8_t* xq2 = xq1 + cols;
                    const int8_t* xq3 = xq2 + cols;
                    const float* xs0 = x_blk_scale + (size_t)(m + 0) * (size_t)n_blk;
                    const float* xs1 = xs0 + n_blk;
                    const float* xs2 = xs1 + n_blk;
                    const float* xs3 = xs2 + n_blk;

                    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();

                    for (int32_t b = 0; b < n_blk; b++) {
                        __m256i q = unpack_q32_from_16B(rp + (b << 4), mask0F, eights);
                        __m256i qabs = _mm256_sign_epi8(q, q);
                        int32_t c0 = b << 5;

                        __m256i d0 = _mm256_madd_epi16(_mm256_maddubs_epi16(qabs, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xq0 + c0)), q)), ones16);
                        __m256i d1 = _mm256_madd_epi16(_mm256_maddubs_epi16(qabs, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xq1 + c0)), q)), ones16);
                        __m256i d2 = _mm256_madd_epi16(_mm256_maddubs_epi16(qabs, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xq2 + c0)), q)), ones16);
                        __m256i d3 = _mm256_madd_epi16(_mm256_maddubs_epi16(qabs, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xq3 + c0)), q)), ones16);

                        a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_set1_ps(xs0[b]), a0);
                        a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1), _mm256_set1_ps(xs1[b]), a1);
                        a2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d2), _mm256_set1_ps(xs2[b]), a2);
                        a3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d3), _mm256_set1_ps(xs3[b]), a3);
                    }

                    float f0 = hsum256_ps(a0), f1 = hsum256_ps(a1);
                    float f2 = hsum256_ps(a2), f3 = hsum256_ps(a3);
                    if (nnz) {
                        const float* xf0 = x_scaled + (size_t)(m + 0) * (size_t)cols;
                        const float* xf1 = xf0 + cols;
                        const float* xf2 = xf1 + cols;
                        const float* xf3 = xf2 + cols;
                        int32_t col = 0;
                        uint32_t sc = s_idx;
                        for (uint16_t k = 0; k < nnz; k++) {
                            uint16_t pv = row_ov[k];
                            col += (int32_t)(pv & 0xFFu);
                            int32_t dv = (int32_t)(int8_t)(pv >> 8);
                            if (dv == -128) dv = (int32_t)super16[sc++];
                            float fv = (float)dv;
                            f0 += fv * xf0[col];
                            f1 += fv * xf1[col];
                            f2 += fv * xf2[col];
                            f3 += fv * xf3[col];
                        }
                        s_next = sc;
                    }
                    out[(size_t)(m + 0) * (size_t)rows + (size_t)r] = f0 * r_s;
                    out[(size_t)(m + 1) * (size_t)rows + (size_t)r] = f1 * r_s;
                    out[(size_t)(m + 2) * (size_t)rows + (size_t)r] = f2 * r_s;
                    out[(size_t)(m + 3) * (size_t)rows + (size_t)r] = f3 * r_s;
                }
                for (; m < M; m++) {
                    const int8_t* xqm = x_q8 + (size_t)m * (size_t)cols;
                    const float* xsm = x_blk_scale + (size_t)m * (size_t)n_blk;
                    const float* xfm = x_scaled + (size_t)m * (size_t)cols;
                    __m256 a = _mm256_setzero_ps();
                    for (int32_t b = 0; b < n_blk; b++) {
                        __m256i q = unpack_q32_from_16B(rp + (b << 4), mask0F, eights);
                        __m256i d = _mm256_madd_epi16(
                            _mm256_maddubs_epi16(_mm256_sign_epi8(q, q), _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xqm + (b << 5))), q)),
                            ones16
                        );
                        a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(xsm[b]), a);
                    }
                    float f = hsum256_ps(a);
                    if (nnz) {
                        int32_t col = 0;
                        uint32_t sc = s_idx;
                        for (uint16_t k = 0; k < nnz; k++) {
                            uint16_t pv = row_ov[k];
                            col += (int32_t)(pv & 0xFFu);
                            int32_t dv = (int32_t)(int8_t)(pv >> 8);
                            if (dv == -128) dv = (int32_t)super16[sc++];
                            f += (float)dv * xfm[col];
                        }
                        s_next = sc;
                    }
                    out[(size_t)m * (size_t)rows + (size_t)r] = f * r_s;
                }
                o_idx += nnz;
                s_idx = s_next;
            }
        }
    }
}

DPUF_EXPORT void dpuf_matmul_qr_split4_f32(
    int32_t M,
    int32_t rows,
    int32_t cols,
    int32_t n_tiles,
    const float* x_in,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    const float* col_scale,
    float* out
) {
    int32_t n_blk = cols >> 5;
    size_t x_cnt = (size_t)M * (size_t)cols;
    size_t b_cnt = (size_t)M * (size_t)n_blk;
    float* x_scaled = (float*)malloc(x_cnt * sizeof(float));
    int8_t* x_q8 = (int8_t*)malloc(x_cnt * sizeof(int8_t));
    float* x_blk_s = (float*)malloc(b_cnt * sizeof(float));
    if (!x_scaled || !x_q8 || !x_blk_s) {
        free(x_scaled); free(x_q8); free(x_blk_s);
        return;
    }

    for (int32_t m = 0; m < M; m++) {
        dpuf_scale_quantize_x_q8_32(
            cols,
            x_in + (size_t)m * (size_t)cols,
            col_scale,
            x_scaled + (size_t)m * (size_t)cols,
            x_q8 + (size_t)m * (size_t)cols,
            x_blk_s + (size_t)m * (size_t)n_blk
        );
    }

    int32_t nth = (n_tiles >= 4) ? 8 : 1;
    if (nth > omp_get_max_threads()) nth = omp_get_max_threads();
    #pragma omp parallel num_threads(nth) if(nth > 1)
    {
        int32_t tid = omp_get_thread_num();
        int32_t nt = omp_get_num_threads();
        int32_t t0 = (n_tiles * tid) / nt;
        int32_t t1 = (n_tiles * (tid + 1)) / nt;
        dpuf_matmul_qr_split4_tile_range(
            M, rows, cols, t0, t1, x_scaled, x_q8, x_blk_s, tensor_base, tile_offsets, out
        );
    }

    free(x_scaled);
    free(x_q8);
    free(x_blk_s);
}

DPUF_EXPORT void dpuf_embed_qr_split4_f32(
    int32_t n_tokens,
    int32_t cols,
    const int32_t* token_ids,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    const float* col_scale,
    float* out
) {
    const int32_t tile_stride = 16;
    const int32_t n_blk = cols >> 5;
    const int32_t half_cols = cols >> 1;

    for (int32_t tok = 0; tok < n_tokens; tok++) {
        int32_t r = token_ids[tok];
        int32_t t = r / tile_stride;
        int32_t lr = r % tile_stride;

        const uint8_t* tptr = tensor_base + tile_offsets[t];
        const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
        int32_t n_r = (int32_t)th->n_rows;
        const float* r_scale = (const float*)(tptr + sizeof(DpufQrTileHdr));
        const uint16_t* row_nnz = (const uint16_t*)(r_scale + n_r);
        const uint8_t* pairs = (const uint8_t*)(row_nnz + n_r);
        const uint16_t* ov_pairs = (const uint16_t*)(pairs + (size_t)n_r * (size_t)half_cols);
        const int16_t* super16 = (const int16_t*)(ov_pairs + th->n_over8);

        uint32_t o_idx = 0;
        uint32_t s_idx = 0;
        for (int32_t prev = 0; prev < lr; prev++) {
            uint16_t nnz_p = row_nnz[prev];
            for (uint16_t k = 0; k < nnz_p; k++) {
                if ((int8_t)(ov_pairs[o_idx++] >> 8) == -128) s_idx++;
            }
        }

        const uint8_t* rp = pairs + (size_t)lr * (size_t)half_cols;
        float r_s = r_scale[lr];
        float* dst = out + (size_t)tok * (size_t)cols;

        for (int32_t b = 0; b < n_blk; b++) {
            const uint8_t* bp = rp + (b << 4);
            int32_t c0 = b << 5;
            for (int32_t k = 0; k < 16; k++) {
                uint8_t byte_v = bp[k];
                int32_t q_lo = (int32_t)(byte_v & 0x0Fu) - 8;
                int32_t q_hi = (int32_t)(byte_v >> 4) - 8;
                dst[c0 + k]      = (float)q_lo * r_s * col_scale[c0 + k];
                dst[c0 + k + 16] = (float)q_hi * r_s * col_scale[c0 + k + 16];
            }
        }

        uint16_t nnz = row_nnz[lr];
        if (nnz) {
            int32_t col = 0;
            for (uint16_t k = 0; k < nnz; k++) {
                uint16_t pv = ov_pairs[o_idx++];
                col += (int32_t)(pv & 0xFFu);
                int32_t dv = (int32_t)(int8_t)(pv >> 8);
                if (dv == -128) dv = (int32_t)super16[s_idx++];
                dst[col] += (float)dv * r_s * col_scale[col];
            }
        }
    }
}

/* ============================================================================
 * MODE 10 (Varijanta B): Walsh-Hadamard FWHT-64 + Pure Dense Blk32 Symmetric int4
 *   Tile body: [blk_scales_f16 (n_r * cols/32 * 2B)][base_pairs (n_r * cols/2)]
 * ============================================================================ */

void dpuf_matmul_qr_hadamard4_tile_range(
    int32_t M,
    int32_t rows,
    int32_t cols,
    int32_t t_start,
    int32_t t_end,
    const float* x_rot,
    const int8_t* x_q8,
    const float* x_blk_scale,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    float* out
) {
    const __m256i mask0F = _mm256_set1_epi8(0x0F);
    const __m256i eights = _mm256_set1_epi8(8);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const int32_t n_blk = cols >> 5;
    const int32_t half_cols = cols >> 1;
    const int32_t tile_stride = 16;

    if (M == 1) {
        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const uint16_t* sc_all = (const uint16_t*)(tptr + sizeof(DpufQrTileHdr));
            const uint8_t* pairs = (const uint8_t*)(sc_all + (size_t)n_r * (size_t)n_blk);

            int32_t lr = 0;
            for (; lr + 1 < n_r; lr += 2) {
                const uint16_t* sc0 = sc_all + (size_t)lr * (size_t)n_blk;
                const uint16_t* sc1 = sc0 + n_blk;
                const uint8_t* rp0 = pairs + (size_t)lr * (size_t)half_cols;
                const uint8_t* rp1 = rp0 + half_cols;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();

                for (int32_t b = 0; b < n_blk; b++) {
                    __m256i xq32 = _mm256_loadu_si256((const __m256i*)(x_q8 + (b << 5)));
                    float xs = x_blk_scale[b];

                    __m256i q0 = unpack_q32_from_16B(rp0 + (b << 4), mask0F, eights);
                    __m256i d0 = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(_mm256_sign_epi8(q0, q0), _mm256_sign_epi8(xq32, q0)),
                        ones16
                    );
                    float wxs0 = fp16_bits_to_f32(sc0[b]) * xs;
                    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_set1_ps(wxs0), acc0);

                    __m256i q1 = unpack_q32_from_16B(rp1 + (b << 4), mask0F, eights);
                    __m256i d1 = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(_mm256_sign_epi8(q1, q1), _mm256_sign_epi8(xq32, q1)),
                        ones16
                    );
                    float wxs1 = fp16_bits_to_f32(sc1[b]) * xs;
                    acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1), _mm256_set1_ps(wxs1), acc1);
                }
                out[r0 + lr]     = hsum256_ps(acc0);
                out[r0 + lr + 1] = hsum256_ps(acc1);
            }
            for (; lr < n_r; lr++) {
                const uint16_t* sc = sc_all + (size_t)lr * (size_t)n_blk;
                const uint8_t* rp = pairs + (size_t)lr * (size_t)half_cols;
                __m256 acc = _mm256_setzero_ps();
                for (int32_t b = 0; b < n_blk; b++) {
                    __m256i xq32 = _mm256_loadu_si256((const __m256i*)(x_q8 + (b << 5)));
                    __m256i q = unpack_q32_from_16B(rp + (b << 4), mask0F, eights);
                    __m256i d = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(_mm256_sign_epi8(q, q), _mm256_sign_epi8(xq32, q)),
                        ones16
                    );
                    float wxs = fp16_bits_to_f32(sc[b]) * x_blk_scale[b];
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(wxs), acc);
                }
                out[r0 + lr] = hsum256_ps(acc);
            }
        }
    } else if (M >= 8) {
        (void)x_rot;
        __declspec(align(32)) float tile_acc[256][16];
        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const uint16_t* sc_all = (const uint16_t*)(tptr + sizeof(DpufQrTileHdr));
            const uint8_t* pairs = (const uint8_t*)(sc_all + (size_t)n_r * (size_t)n_blk);

            for (int32_t m0 = 0; m0 < M; m0 += 256) {
                int32_t M_cur = (m0 + 256 <= M) ? 256 : (M - m0);
                const int8_t* xq_base = x_q8 + (size_t)m0 * (size_t)cols;
                const float* xs_base = x_blk_scale + (size_t)m0 * (size_t)n_blk;
                memset(tile_acc, 0, (size_t)M_cur * 16u * sizeof(float));

                for (int32_t lr_base = 0; lr_base < n_r; lr_base += 8) {
                    int32_t n_8 = (lr_base + 8 <= n_r) ? 8 : (n_r - lr_base);
                    const uint8_t* pairs_b = pairs + (size_t)lr_base * (size_t)half_cols;
                    for (int32_t b = 0; b < n_blk; b++) {
                        __declspec(align(16)) uint16_t sc16[8];
                        for (int ri = 0; ri < n_8; ri++) sc16[ri] = sc_all[(size_t)(lr_base + ri) * (size_t)n_blk + b];
                        for (int ri = n_8; ri < 8; ri++) sc16[ri] = 0u;
                        __m256 v_ws = _mm256_cvtph_ps(_mm_load_si128((const __m128i*)sc16));

                        __m256i w_tr[8], wsum128;
                        prep_wtr_8x32(n_8, pairs_b, half_cols, b, mask0F, eights, w_tr, &wsum128);
                        gemm_blk32_8x6_q8_acc(M_cur, cols, b, lr_base, w_tr, wsum128, v_ws, xq_base, xs_base, n_blk, tile_acc);
                    }
                }
                for (int32_t m = 0; m < M_cur; m++) {
                    memcpy(out + (size_t)(m0 + m) * (size_t)rows + (size_t)r0, tile_acc[m], (size_t)n_r * sizeof(float));
                }
            }
        }
    } else {
        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const uint16_t* sc_all = (const uint16_t*)(tptr + sizeof(DpufQrTileHdr));
            const uint8_t* pairs = (const uint8_t*)(sc_all + (size_t)n_r * (size_t)n_blk);

            for (int32_t lr = 0; lr < n_r; lr++) {
                int32_t r = r0 + lr;
                const uint16_t* sc = sc_all + (size_t)lr * (size_t)n_blk;
                const uint8_t* rp = pairs + (size_t)lr * (size_t)half_cols;

                int32_t m = 0;
                for (; m + 3 < M; m += 4) {
                    const int8_t* xq0 = x_q8 + (size_t)(m + 0) * (size_t)cols;
                    const int8_t* xq1 = xq0 + cols;
                    const int8_t* xq2 = xq1 + cols;
                    const int8_t* xq3 = xq2 + cols;
                    const float* xs0 = x_blk_scale + (size_t)(m + 0) * (size_t)n_blk;
                    const float* xs1 = xs0 + n_blk;
                    const float* xs2 = xs1 + n_blk;
                    const float* xs3 = xs2 + n_blk;

                    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
                    __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();

                    for (int32_t b = 0; b < n_blk; b++) {
                        float ws = fp16_bits_to_f32(sc[b]);
                        __m256i q = unpack_q32_from_16B(rp + (b << 4), mask0F, eights);
                        __m256i qabs = _mm256_sign_epi8(q, q);
                        int32_t c0 = b << 5;

                        __m256i d0 = _mm256_madd_epi16(_mm256_maddubs_epi16(qabs, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xq0 + c0)), q)), ones16);
                        __m256i d1 = _mm256_madd_epi16(_mm256_maddubs_epi16(qabs, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xq1 + c0)), q)), ones16);
                        __m256i d2 = _mm256_madd_epi16(_mm256_maddubs_epi16(qabs, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xq2 + c0)), q)), ones16);
                        __m256i d3 = _mm256_madd_epi16(_mm256_maddubs_epi16(qabs, _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xq3 + c0)), q)), ones16);

                        a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_set1_ps(ws * xs0[b]), a0);
                        a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1), _mm256_set1_ps(ws * xs1[b]), a1);
                        a2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d2), _mm256_set1_ps(ws * xs2[b]), a2);
                        a3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d3), _mm256_set1_ps(ws * xs3[b]), a3);
                    }
                    out[(size_t)(m + 0) * (size_t)rows + (size_t)r] = hsum256_ps(a0);
                    out[(size_t)(m + 1) * (size_t)rows + (size_t)r] = hsum256_ps(a1);
                    out[(size_t)(m + 2) * (size_t)rows + (size_t)r] = hsum256_ps(a2);
                    out[(size_t)(m + 3) * (size_t)rows + (size_t)r] = hsum256_ps(a3);
                }
                for (; m < M; m++) {
                    const int8_t* xqm = x_q8 + (size_t)m * (size_t)cols;
                    const float* xsm = x_blk_scale + (size_t)m * (size_t)n_blk;
                    __m256 a = _mm256_setzero_ps();
                    for (int32_t b = 0; b < n_blk; b++) {
                        float ws = fp16_bits_to_f32(sc[b]);
                        __m256i q = unpack_q32_from_16B(rp + (b << 4), mask0F, eights);
                        __m256i d = _mm256_madd_epi16(
                            _mm256_maddubs_epi16(_mm256_sign_epi8(q, q), _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xqm + (b << 5))), q)),
                            ones16
                        );
                        a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(ws * xsm[b]), a);
                    }
                    out[(size_t)m * (size_t)rows + (size_t)r] = hsum256_ps(a);
                }
            }
        }
    }
}

DPUF_EXPORT void dpuf_matmul_qr_hadamard4_f32(
    int32_t M,
    int32_t rows,
    int32_t cols,
    int32_t n_tiles,
    const float* x_in,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    const uint8_t* sign_bits,
    float* out
) {
    int32_t n_blk = cols >> 5;
    size_t x_cnt = (size_t)M * (size_t)cols;
    size_t b_cnt = (size_t)M * (size_t)n_blk;
    float* x_rot = (float*)malloc(x_cnt * sizeof(float));
    int8_t* x_q8 = (int8_t*)malloc(x_cnt * sizeof(int8_t));
    float* x_blk_s = (float*)malloc(b_cnt * sizeof(float));
    if (!x_rot || !x_q8 || !x_blk_s) {
        free(x_rot); free(x_q8); free(x_blk_s);
        return;
    }

    for (int32_t m = 0; m < M; m++) {
        dpuf_fwht64_sign_f32(cols, x_in + (size_t)m * (size_t)cols, sign_bits, x_rot + (size_t)m * (size_t)cols);
        dpuf_scale_quantize_x_q8_32(
            cols,
            x_rot + (size_t)m * (size_t)cols,
            NULL,
            NULL,
            x_q8 + (size_t)m * (size_t)cols,
            x_blk_s + (size_t)m * (size_t)n_blk
        );
    }

    int32_t nth = (n_tiles >= 4) ? 8 : 1;
    if (nth > omp_get_max_threads()) nth = omp_get_max_threads();
    #pragma omp parallel num_threads(nth) if(nth > 1)
    {
        int32_t tid = omp_get_thread_num();
        int32_t nt = omp_get_num_threads();
        int32_t t0 = (n_tiles * tid) / nt;
        int32_t t1 = (n_tiles * (tid + 1)) / nt;
        dpuf_matmul_qr_hadamard4_tile_range(
            M, rows, cols, t0, t1, x_rot, x_q8, x_blk_s, tensor_base, tile_offsets, out
        );
    }

    free(x_rot);
    free(x_q8);
    free(x_blk_s);
}

DPUF_EXPORT void dpuf_embed_qr_hadamard4_f32(
    int32_t n_tokens,
    int32_t cols,
    const int32_t* token_ids,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    const uint8_t* sign_bits,
    float* out
) {
    const int32_t tile_stride = 16;
    const int32_t n_blk = cols >> 5;
    const int32_t half_cols = cols >> 1;

    for (int32_t tok = 0; tok < n_tokens; tok++) {
        int32_t r = token_ids[tok];
        int32_t t = r / tile_stride;
        int32_t lr = r % tile_stride;

        const uint8_t* tptr = tensor_base + tile_offsets[t];
        const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
        int32_t n_r = (int32_t)th->n_rows;
        const uint16_t* sc_all = (const uint16_t*)(tptr + sizeof(DpufQrTileHdr));
        const uint8_t* pairs = (const uint8_t*)(sc_all + (size_t)n_r * (size_t)n_blk);

        const uint16_t* sc = sc_all + (size_t)lr * (size_t)n_blk;
        const uint8_t* rp = pairs + (size_t)lr * (size_t)half_cols;
        float* dst = out + (size_t)tok * (size_t)cols;

        for (int32_t b = 0; b < n_blk; b++) {
            float ws = fp16_bits_to_f32(sc[b]);
            const uint8_t* bp = rp + (b << 4);
            int32_t c0 = b << 5;
            for (int32_t k = 0; k < 16; k++) {
                uint8_t byte_v = bp[k];
                int32_t q_lo = (int32_t)(byte_v & 0x0Fu) - 8;
                int32_t q_hi = (int32_t)(byte_v >> 4) - 8;
                dst[c0 + k]      = (float)q_lo * ws;
                dst[c0 + k + 16] = (float)q_hi * ws;
            }
        }
        /* Inverse transform: FWHT64 then sign flip */
        for (int32_t c = 0; c < cols; c += 64) {
            fwht64_block_f32(dst + c);
        }
        if (sign_bits) {
            apply_sign_mask_f32(cols, dst, sign_bits, dst);
        }
    }
}

void dpuf_embed_qr_any_f32(
    const DpufWeightDesc* w,
    int32_t n_tokens,
    const int32_t* token_ids,
    float* out,
    float* scratch_f32
) {
    (void)scratch_f32;
    if (w->mode == DPUF_MODE_QR_ZD4) {
        dpuf_embed_qr_zd4_f32(n_tokens, w->cols, token_ids, w->tensor_base, w->tile_offsets, w->col_scale, out);
    } else if (w->mode == DPUF_MODE_QR_SPLIT4) {
        dpuf_embed_qr_split4_f32(n_tokens, w->cols, token_ids, w->tensor_base, w->tile_offsets, w->col_scale, out);
    } else if (w->mode == DPUF_MODE_QR_HAD4) {
        dpuf_embed_qr_hadamard4_f32(n_tokens, w->cols, token_ids, w->tensor_base, w->tile_offsets, w->sign_bits, out);
    } else if (w->mode == DPUF_MODE_QR_Q8) {
        dpuf_embed_qr_q8_f32(n_tokens, w->cols, token_ids, w->tensor_base, w->tile_offsets, out);
    } else if (w->mode == DPUF_MODE_QR_HAD4_PACKED) {
        dpuf_embed_qr_hadamard4_packed_f32(n_tokens, w->cols, token_ids, w->tensor_base, w->tile_offsets, w->sign_bits, out);
    } else if (w->mode == DPUF_MODE_QR_HAD4_ASYM) {
        dpuf_embed_qr_had4_asym_f32(n_tokens, w->cols, token_ids, w->tensor_base, w->tile_offsets, w->sign_bits, out);
    } else if (w->mode == DPUF_MODE_QR_HAD6) {
        dpuf_embed_qr_had6_f32(n_tokens, w->cols, token_ids, w->tensor_base, w->tile_offsets, w->sign_bits, out);
    }
}

/* ============================================================================
 * MODE 11: QR-Tiled Pure Dense Blk32 Symmetric int8 [-127..127] (8.5 b/w, 0% outliers)
 *   Tile body: [blk_scales_f16 (n_r * cols/32 * 2B)][w_q8 (n_r * cols * 1B)]
 * ============================================================================ */

static void gemm_blk32_8x6_q8_exact_acc(
    int32_t M_cur,
    int32_t cols,
    int32_t b,
    int32_t lr_off,
    const __m256i w_tr[8],
    __m256 v_wscale,
    const int8_t* x_q8_base,
    const float* x_scale_base,
    int32_t n_blk,
    float tile_acc[256][16]
) {
    /* VPMADDWD (int16 * int16 -> int32) avoids INT16 saturation of VPMADDUBSW(u8, s8) at |q|=127 */
    __m256i w16_lo[8], w16_hi[8];
    for (int g = 0; g < 8; g++) {
        __m128i lo128 = _mm256_castsi256_si128(w_tr[g]);
        __m128i hi128 = _mm256_extracti128_si256(w_tr[g], 1);
        w16_lo[g] = _mm256_cvtepi8_epi16(lo128);
        w16_hi[g] = _mm256_cvtepi8_epi16(hi128);
    }

    const int32_t c0 = b << 5;
    int32_t m = 0;
    for (; m + 5 < M_cur; m += 6) {
        __m256i lo[6], hi[6];
        for (int j = 0; j < 6; j++) { lo[j] = _mm256_setzero_si256(); hi[j] = _mm256_setzero_si256(); }

        __declspec(align(32)) int32_t xb[6][8];
        for (int j = 0; j < 6; j++) {
            __m256i xs = _mm256_loadu_si256((const __m256i*)(x_q8_base + (size_t)(m + j) * (size_t)cols + c0));
            _mm256_store_si256((__m256i*)xb[j], xs);
        }

        for (int g = 0; g < 8; g++) {
            __m256i wl = w16_lo[g], wh = w16_hi[g];
            for (int j = 0; j < 6; j++) {
                __m256i x16 = _mm256_cvtepi8_epi16(_mm_set1_epi32(xb[j][g]));
                lo[j] = _mm256_add_epi32(lo[j], _mm256_madd_epi16(x16, wl));
                hi[j] = _mm256_add_epi32(hi[j], _mm256_madd_epi16(x16, wh));
            }
        }

        for (int j = 0; j < 6; j++) {
            __m256i sum8 = _mm256_permute4x64_epi64(_mm256_hadd_epi32(lo[j], hi[j]), 0xD8);
            __m256 s = _mm256_mul_ps(v_wscale, _mm256_set1_ps(x_scale_base[(size_t)(m + j) * (size_t)n_blk + b]));
            _mm256_store_ps(&tile_acc[m + j][lr_off], _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum8), s, _mm256_load_ps(&tile_acc[m + j][lr_off])));
        }
    }
    for (; m < M_cur; m++) {
        __declspec(align(32)) int32_t xb[8];
        _mm256_store_si256((__m256i*)xb, _mm256_loadu_si256((const __m256i*)(x_q8_base + (size_t)m * (size_t)cols + c0)));
        __m256i alo = _mm256_setzero_si256(), ahi = _mm256_setzero_si256();
        for (int g = 0; g < 8; g++) {
            __m256i x16 = _mm256_cvtepi8_epi16(_mm_set1_epi32(xb[g]));
            alo = _mm256_add_epi32(alo, _mm256_madd_epi16(x16, w16_lo[g]));
            ahi = _mm256_add_epi32(ahi, _mm256_madd_epi16(x16, w16_hi[g]));
        }
        __m256i sum8 = _mm256_permute4x64_epi64(_mm256_hadd_epi32(alo, ahi), 0xD8);
        __m256 sm = _mm256_mul_ps(v_wscale, _mm256_set1_ps(x_scale_base[(size_t)m * (size_t)n_blk + b]));
        _mm256_store_ps(&tile_acc[m][lr_off], _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum8), sm, _mm256_load_ps(&tile_acc[m][lr_off])));
    }
}

void dpuf_matmul_qr_q8_tile_range(
    int32_t M,
    int32_t rows,
    int32_t cols,
    int32_t t_start,
    int32_t t_end,
    const int8_t* x_q8,
    const float* x_blk_scale,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    float* out
) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    const int32_t n_blk = cols >> 5;
    const int32_t tile_stride = 16;

    if (M == 1) {
        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const uint16_t* sc_all = (const uint16_t*)(tptr + sizeof(DpufQrTileHdr));
            const int8_t* w_all = (const int8_t*)(sc_all + (size_t)n_r * (size_t)n_blk);

            int32_t lr = 0;
            for (; lr + 1 < n_r; lr += 2) {
                const uint16_t* sc0 = sc_all + (size_t)lr * (size_t)n_blk;
                const uint16_t* sc1 = sc0 + n_blk;
                const int8_t* wp0 = w_all + (size_t)lr * (size_t)cols;
                const int8_t* wp1 = wp0 + cols;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();

                for (int32_t b = 0; b < n_blk; b++) {
                    __m256i xq32 = _mm256_loadu_si256((const __m256i*)(x_q8 + (b << 5)));
                    float xs = x_blk_scale[b];

                    __m256i q0 = _mm256_loadu_si256((const __m256i*)(wp0 + (b << 5)));
                    __m256i d0 = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(_mm256_sign_epi8(q0, q0), _mm256_sign_epi8(xq32, q0)),
                        ones16
                    );
                    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_set1_ps(fp16_bits_to_f32(sc0[b]) * xs), acc0);

                    __m256i q1 = _mm256_loadu_si256((const __m256i*)(wp1 + (b << 5)));
                    __m256i d1 = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(_mm256_sign_epi8(q1, q1), _mm256_sign_epi8(xq32, q1)),
                        ones16
                    );
                    acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1), _mm256_set1_ps(fp16_bits_to_f32(sc1[b]) * xs), acc1);
                }
                out[r0 + lr]     = hsum256_ps(acc0);
                out[r0 + lr + 1] = hsum256_ps(acc1);
            }
            for (; lr < n_r; lr++) {
                const uint16_t* sc = sc_all + (size_t)lr * (size_t)n_blk;
                const int8_t* wp = w_all + (size_t)lr * (size_t)cols;
                __m256 acc = _mm256_setzero_ps();
                for (int32_t b = 0; b < n_blk; b++) {
                    __m256i xq32 = _mm256_loadu_si256((const __m256i*)(x_q8 + (b << 5)));
                    __m256i q = _mm256_loadu_si256((const __m256i*)(wp + (b << 5)));
                    __m256i d = _mm256_madd_epi16(
                        _mm256_maddubs_epi16(_mm256_sign_epi8(q, q), _mm256_sign_epi8(xq32, q)),
                        ones16
                    );
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(fp16_bits_to_f32(sc[b]) * x_blk_scale[b]), acc);
                }
                out[r0 + lr] = hsum256_ps(acc);
            }
        }
    } else if (M >= 8) {
        __declspec(align(32)) float tile_acc[256][16];
        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const uint16_t* sc_all = (const uint16_t*)(tptr + sizeof(DpufQrTileHdr));
            const int8_t* w_all = (const int8_t*)(sc_all + (size_t)n_r * (size_t)n_blk);

            for (int32_t m0 = 0; m0 < M; m0 += 256) {
                int32_t M_cur = (m0 + 256 <= M) ? 256 : (M - m0);
                const int8_t* xq_base = x_q8 + (size_t)m0 * (size_t)cols;
                const float* xs_base = x_blk_scale + (size_t)m0 * (size_t)n_blk;
                memset(tile_acc, 0, (size_t)M_cur * 16u * sizeof(float));

                for (int32_t lr_base = 0; lr_base < n_r; lr_base += 8) {
                    int32_t n_8 = (lr_base + 8 <= n_r) ? 8 : (n_r - lr_base);
                    const int8_t* wq_base = w_all + (size_t)lr_base * (size_t)cols;
                    for (int32_t b = 0; b < n_blk; b++) {
                        __declspec(align(16)) uint16_t sc16[8];
                        for (int ri = 0; ri < n_8; ri++) sc16[ri] = sc_all[(size_t)(lr_base + ri) * (size_t)n_blk + b];
                        for (int ri = n_8; ri < 8; ri++) sc16[ri] = 0u;
                        __m256 v_ws = _mm256_cvtph_ps(_mm_load_si128((const __m128i*)sc16));

                        __m256i w_tr[8], wsum128;
                        prep_wtr_8x32_q8(n_8, wq_base, cols, b, w_tr, &wsum128);
                        (void)wsum128;
                        gemm_blk32_8x6_q8_exact_acc(M_cur, cols, b, lr_base, w_tr, v_ws, xq_base, xs_base, n_blk, tile_acc);
                    }
                }
                for (int32_t m = 0; m < M_cur; m++) {
                    memcpy(out + (size_t)(m0 + m) * (size_t)rows + (size_t)r0, tile_acc[m], (size_t)n_r * sizeof(float));
                }
            }
        }
    } else {
        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            const uint16_t* sc_all = (const uint16_t*)(tptr + sizeof(DpufQrTileHdr));
            const int8_t* w_all = (const int8_t*)(sc_all + (size_t)n_r * (size_t)n_blk);

            for (int32_t lr = 0; lr < n_r; lr++) {
                int32_t r = r0 + lr;
                const uint16_t* sc = sc_all + (size_t)lr * (size_t)n_blk;
                const int8_t* wp = w_all + (size_t)lr * (size_t)cols;
                for (int32_t m = 0; m < M; m++) {
                    const int8_t* xqm = x_q8 + (size_t)m * (size_t)cols;
                    const float* xsm = x_blk_scale + (size_t)m * (size_t)n_blk;
                    __m256 a = _mm256_setzero_ps();
                    for (int32_t b = 0; b < n_blk; b++) {
                        __m256i q = _mm256_loadu_si256((const __m256i*)(wp + (b << 5)));
                        __m256i d = _mm256_madd_epi16(
                            _mm256_maddubs_epi16(_mm256_sign_epi8(q, q), _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xqm + (b << 5))), q)),
                            ones16
                        );
                        a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(fp16_bits_to_f32(sc[b]) * xsm[b]), a);
                    }
                    out[(size_t)m * (size_t)rows + (size_t)r] = hsum256_ps(a);
                }
            }
        }
    }
}

DPUF_EXPORT void dpuf_matmul_qr_q8_f32(
    int32_t M,
    int32_t rows,
    int32_t cols,
    int32_t n_tiles,
    const float* x_in,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    float* out
) {
    int32_t n_blk = cols >> 5;
    size_t x_cnt = (size_t)M * (size_t)cols;
    size_t b_cnt = (size_t)M * (size_t)n_blk;
    int8_t* x_q8 = (int8_t*)malloc(x_cnt * sizeof(int8_t));
    float* x_blk_s = (float*)malloc(b_cnt * sizeof(float));
    if (!x_q8 || !x_blk_s) {
        free(x_q8); free(x_blk_s);
        return;
    }
    for (int32_t m = 0; m < M; m++) {
        dpuf_scale_quantize_x_q8_32(
            cols, x_in + (size_t)m * (size_t)cols, NULL, NULL,
            x_q8 + (size_t)m * (size_t)cols, x_blk_s + (size_t)m * (size_t)n_blk
        );
    }
    int32_t nth = (n_tiles >= 4) ? 8 : 1;
    if (nth > omp_get_max_threads()) nth = omp_get_max_threads();
    #pragma omp parallel num_threads(nth) if(nth > 1)
    {
        int32_t tid = omp_get_thread_num();
        int32_t nt = omp_get_num_threads();
        int32_t t0 = (n_tiles * tid) / nt;
        int32_t t1 = (n_tiles * (tid + 1)) / nt;
        dpuf_matmul_qr_q8_tile_range(M, rows, cols, t0, t1, x_q8, x_blk_s, tensor_base, tile_offsets, out);
    }
    free(x_q8);
    free(x_blk_s);
}

DPUF_EXPORT void dpuf_embed_qr_q8_f32(
    int32_t n_tokens,
    int32_t cols,
    const int32_t* token_ids,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    float* out
) {
    const int32_t tile_stride = 16;
    const int32_t n_blk = cols >> 5;
    for (int32_t tok = 0; tok < n_tokens; tok++) {
        int32_t r = token_ids[tok];
        int32_t t = r / tile_stride;
        int32_t lr = r % tile_stride;
        const uint8_t* tptr = tensor_base + tile_offsets[t];
        const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
        int32_t n_r = (int32_t)th->n_rows;
        const uint16_t* sc_all = (const uint16_t*)(tptr + sizeof(DpufQrTileHdr));
        const int8_t* w_all = (const int8_t*)(sc_all + (size_t)n_r * (size_t)n_blk);
        const uint16_t* sc = sc_all + (size_t)lr * (size_t)n_blk;
        const int8_t* wp = w_all + (size_t)lr * (size_t)cols;
        float* dst = out + (size_t)tok * (size_t)cols;
        for (int32_t b = 0; b < n_blk; b++) {
            float ws = fp16_bits_to_f32(sc[b]);
            int32_t c0 = b << 5;
            for (int32_t k = 0; k < 32; k++) {
                dst[c0 + k] = (float)wp[c0 + k] * ws;
            }
        }
    }
}

/* ============================================================================
 * MODE 12: Pre-Packed 8x32 QR Hadamard4 (Zero Transpose, Zero wsum128, L2-Tiled)
 *   Per 8-row subgroup (n_8g = (n_r+7)>>3):
 *     [sc_grp: n_blk * 8 * uint16_t (n_blk * 16B)]
 *     [q_grp : n_blk * 4 * 32 * uint8_t (n_blk * 128B)]
 *     In chunk c (32B): byte[ri*4 + k] has lo4=col(8c+k), hi4=col(8c+4+k) for row ri
 * ============================================================================ */

static __forceinline int32_t hsum256_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s4 = _mm_add_epi32(lo, hi);
    __m128i s2 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
    __m128i s1 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0xB1));
    return _mm_cvtsi128_si32(s1);
}

void dpuf_prep_had4_packed_slab(
    int32_t mc,
    int32_t cols,
    const float* x_slab_f32,
    const uint8_t* sign_bits,
    float* f_scratch,
    int8_t* x_out,
    float* x_scale_out,
    int32_t* x_sum8_out,
    int32_t pack_slab
) {
    const int32_t n_blk = cols >> 5;
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256i perm_idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i min_i8 = _mm256_set1_epi8(-127);
    const __m256i ones_u8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);

    for (int32_t j = 0; j < mc; j++) {
        dpuf_fwht64_sign_f32(
            cols,
            x_slab_f32 + (size_t)j * (size_t)cols,
            sign_bits,
            f_scratch + (size_t)j * (size_t)cols
        );
    }

    for (int32_t b = 0; b < n_blk; b++) {
        int32_t c0 = b << 5;
        int32_t* xp_blk = (int32_t*)(x_out + (size_t)b * 192u);
        for (int32_t j = 0; j < mc; j++) {
            const float* f_ptr = f_scratch + (size_t)j * (size_t)cols + (size_t)c0;
            __m256 v0 = _mm256_loadu_ps(f_ptr);
            __m256 v1 = _mm256_loadu_ps(f_ptr + 8);
            __m256 v2 = _mm256_loadu_ps(f_ptr + 16);
            __m256 v3 = _mm256_loadu_ps(f_ptr + 24);

            __m256 m01 = _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask));
            __m256 m23 = _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask));
            __m256 mall = _mm256_max_ps(m01, m23);
            __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(mall), _mm256_extractf128_ps(mall, 1));
            m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
            m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
            float amax = _mm_cvtss_f32(m128);
            if (amax <= 1e-4f) {
                if (!pack_slab) {
                    x_scale_out[b] = 0.0f;
                    x_sum8_out[b] = 0;
                    memset(x_out + c0, 0, 32);
                } else {
                    x_scale_out[(size_t)b * 6u + (size_t)j] = 0.0f;
                    x_sum8_out[(size_t)b * 6u + (size_t)j] = 0;
                    for (int g = 0; g < 8; g++) xp_blk[g * 6 + j] = 0;
                }
                continue;
            }
            float d = amax / 127.0f;
            float inv_d = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;

            __m256 vinv = _mm256_set1_ps(inv_d);
            __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vinv));
            __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vinv));
            __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vinv));
            __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vinv));
            __m256i p01 = _mm256_packs_epi32(q0, q1);
            __m256i p23 = _mm256_packs_epi32(q2, q3);
            __m256i p8  = _mm256_packs_epi16(p01, p23);
            __m256i p_ord = _mm256_max_epi8(_mm256_permutevar8x32_epi32(p8, perm_idx), min_i8);

            __m256i s8 = _mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, p_ord), ones16);
            int32_t sum8 = hsum256_epi32(s8) << 3;

            if (!pack_slab) {
                x_scale_out[b] = d;
                x_sum8_out[b] = sum8;
                _mm256_storeu_si256((__m256i*)(x_out + c0), p_ord);
            } else {
                x_scale_out[(size_t)b * 6u + (size_t)j] = d;
                x_sum8_out[(size_t)b * 6u + (size_t)j] = sum8;
                __declspec(align(32)) int32_t g8[8];
                _mm256_store_si256((__m256i*)g8, p_ord);
                for (int g = 0; g < 8; g++) xp_blk[g * 6 + j] = g8[g];
            }
        }
    }
}

static __forceinline void store_8rows_out(float* dst, int32_t n_8, __m256 v) {
    if (n_8 == 8) {
        _mm256_storeu_ps(dst, v);
    } else {
        __declspec(align(32)) float tmp[8];
        _mm256_store_ps(tmp, v);
        memcpy(dst, tmp, (size_t)n_8 * sizeof(float));
    }
}

static void had4p_slab_8x6(
    int32_t n_blk,
    int32_t rows,
    int32_t row_off,
    int32_t n_8,
    int32_t slab_m,
    const uint16_t* sc_grp,
    const uint8_t* q_grp,
    const int32_t* xp_base,
    const float* xs_base,
    const int32_t* xsum_base,
    float* out
) {
    const __m256i mask0F = _mm256_set1_epi8(0x0F);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();

    for (int32_t b = 0; b < n_blk; b++) {
        const uint8_t* qp = q_grp + (size_t)b * 128u;
        const int32_t* xp = xp_base + (size_t)b * 48u;
        __m256i d0 = _mm256_setzero_si256(), d1 = _mm256_setzero_si256();
        __m256i d2 = _mm256_setzero_si256(), d3 = _mm256_setzero_si256();
        __m256i d4 = _mm256_setzero_si256(), d5 = _mm256_setzero_si256();

        /* Exact INT16 accumulation across all 8 nibbles (max |sum| = 8*2*15*127 = 30480 <= 32767) */
        for (int c = 0; c < 4; c++) {
            __m256i raw = _mm256_loadu_si256((const __m256i*)(qp + (c << 5)));
            __m256i w0 = _mm256_and_si256(raw, mask0F);
            const int32_t* xg0 = xp + (c * 2) * 6;
            d0 = _mm256_add_epi16(d0, _mm256_maddubs_epi16(w0, _mm256_set1_epi32(xg0[0])));
            d1 = _mm256_add_epi16(d1, _mm256_maddubs_epi16(w0, _mm256_set1_epi32(xg0[1])));
            d2 = _mm256_add_epi16(d2, _mm256_maddubs_epi16(w0, _mm256_set1_epi32(xg0[2])));
            d3 = _mm256_add_epi16(d3, _mm256_maddubs_epi16(w0, _mm256_set1_epi32(xg0[3])));
            d4 = _mm256_add_epi16(d4, _mm256_maddubs_epi16(w0, _mm256_set1_epi32(xg0[4])));
            d5 = _mm256_add_epi16(d5, _mm256_maddubs_epi16(w0, _mm256_set1_epi32(xg0[5])));

            __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask0F);
            const int32_t* xg1 = xp + (c * 2 + 1) * 6;
            d0 = _mm256_add_epi16(d0, _mm256_maddubs_epi16(w1, _mm256_set1_epi32(xg1[0])));
            d1 = _mm256_add_epi16(d1, _mm256_maddubs_epi16(w1, _mm256_set1_epi32(xg1[1])));
            d2 = _mm256_add_epi16(d2, _mm256_maddubs_epi16(w1, _mm256_set1_epi32(xg1[2])));
            d3 = _mm256_add_epi16(d3, _mm256_maddubs_epi16(w1, _mm256_set1_epi32(xg1[3])));
            d4 = _mm256_add_epi16(d4, _mm256_maddubs_epi16(w1, _mm256_set1_epi32(xg1[4])));
            d5 = _mm256_add_epi16(d5, _mm256_maddubs_epi16(w1, _mm256_set1_epi32(xg1[5])));
        }

        const __m256i ones16 = _mm256_set1_epi16(1);
        __m256 v_ws = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(sc_grp + (size_t)b * 8u)));
        const float* xs = xs_base + (size_t)b * 6u;
        const int32_t* xm = xsum_base + (size_t)b * 6u;

        d0 = _mm256_sub_epi32(_mm256_madd_epi16(d0, ones16), _mm256_set1_epi32(xm[0]));
        d1 = _mm256_sub_epi32(_mm256_madd_epi16(d1, ones16), _mm256_set1_epi32(xm[1]));
        d2 = _mm256_sub_epi32(_mm256_madd_epi16(d2, ones16), _mm256_set1_epi32(xm[2]));
        d3 = _mm256_sub_epi32(_mm256_madd_epi16(d3, ones16), _mm256_set1_epi32(xm[3]));
        d4 = _mm256_sub_epi32(_mm256_madd_epi16(d4, ones16), _mm256_set1_epi32(xm[4]));
        d5 = _mm256_sub_epi32(_mm256_madd_epi16(d5, ones16), _mm256_set1_epi32(xm[5]));

        acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_mul_ps(v_ws, _mm256_set1_ps(xs[0])), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1), _mm256_mul_ps(v_ws, _mm256_set1_ps(xs[1])), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d2), _mm256_mul_ps(v_ws, _mm256_set1_ps(xs[2])), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d3), _mm256_mul_ps(v_ws, _mm256_set1_ps(xs[3])), acc3);
        acc4 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d4), _mm256_mul_ps(v_ws, _mm256_set1_ps(xs[4])), acc4);
        acc5 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d5), _mm256_mul_ps(v_ws, _mm256_set1_ps(xs[5])), acc5);
    }

    store_8rows_out(out + (size_t)(slab_m + 0) * (size_t)rows + (size_t)row_off, n_8, acc0);
    store_8rows_out(out + (size_t)(slab_m + 1) * (size_t)rows + (size_t)row_off, n_8, acc1);
    store_8rows_out(out + (size_t)(slab_m + 2) * (size_t)rows + (size_t)row_off, n_8, acc2);
    store_8rows_out(out + (size_t)(slab_m + 3) * (size_t)rows + (size_t)row_off, n_8, acc3);
    store_8rows_out(out + (size_t)(slab_m + 4) * (size_t)rows + (size_t)row_off, n_8, acc4);
    store_8rows_out(out + (size_t)(slab_m + 5) * (size_t)rows + (size_t)row_off, n_8, acc5);
}

static void had4p_slab_8x_tail(
    int32_t mc,
    int32_t n_blk,
    int32_t rows,
    int32_t row_off,
    int32_t n_8,
    int32_t slab_m,
    const uint16_t* sc_grp,
    const uint8_t* q_grp,
    const int32_t* xp_base,
    const float* xs_base,
    const int32_t* xsum_base,
    float* out
) {
    const __m256i mask0F = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256 acc[6];
    for (int j = 0; j < mc; j++) acc[j] = _mm256_setzero_ps();

    for (int32_t b = 0; b < n_blk; b++) {
        const uint8_t* qp = q_grp + (size_t)b * 128u;
        const int32_t* xp = xp_base + (size_t)b * 48u;
        __m256i d[6];
        for (int j = 0; j < mc; j++) d[j] = _mm256_setzero_si256();

        for (int c = 0; c < 4; c++) {
            __m256i raw = _mm256_loadu_si256((const __m256i*)(qp + (c << 5)));
            __m256i w0 = _mm256_and_si256(raw, mask0F);
            __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask0F);
            const int32_t* xg0 = xp + (c * 2) * 6;
            const int32_t* xg1 = xp + (c * 2 + 1) * 6;
            for (int j = 0; j < mc; j++) {
                d[j] = _mm256_add_epi16(d[j], _mm256_maddubs_epi16(w0, _mm256_set1_epi32(xg0[j])));
                d[j] = _mm256_add_epi16(d[j], _mm256_maddubs_epi16(w1, _mm256_set1_epi32(xg1[j])));
            }
        }
        __m256 v_ws = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(sc_grp + (size_t)b * 8u)));
        const float* xs = xs_base + (size_t)b * 6u;
        const int32_t* xm = xsum_base + (size_t)b * 6u;
        for (int j = 0; j < mc; j++) {
            __m256i dc = _mm256_sub_epi32(_mm256_madd_epi16(d[j], ones16), _mm256_set1_epi32(xm[j]));
            acc[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dc), _mm256_mul_ps(v_ws, _mm256_set1_ps(xs[j])), acc[j]);
        }
    }
    for (int j = 0; j < mc; j++) {
        store_8rows_out(out + (size_t)(slab_m + j) * (size_t)rows + (size_t)row_off, n_8, acc[j]);
    }
}

void dpuf_matmul_qr_hadamard4_packed_tile_range(
    int32_t M,
    int32_t rows,
    int32_t cols,
    int32_t t_start,
    int32_t t_end,
    const int8_t* x_q8,
    const float* x_blk_scale,
    const int32_t* x_sum8,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    float* out
) {
    const int32_t n_blk = cols >> 5;
    const int32_t tile_stride = 16;

    if (M == 1) {
        const __m256i mask0F = _mm256_set1_epi8(0x0F);
        const __m256i ones16 = _mm256_set1_epi16(1);
        const int32_t* x_i32 = (const int32_t*)x_q8;

        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            int32_t n_8g = (n_r + 7) >> 3;

            for (int32_t rg = 0; rg < n_8g; rg++) {
                int32_t n_8 = (rg * 8 + 8 <= n_r) ? 8 : (n_r - rg * 8);
                const uint8_t* grp_base = tptr + sizeof(DpufQrTileHdr) + (size_t)rg * (size_t)n_blk * 144u;
                const uint16_t* sc_grp = (const uint16_t*)grp_base;
                const uint8_t*  q_grp  = grp_base + (size_t)n_blk * 16u;

                __m256 acc = _mm256_setzero_ps();
                for (int32_t b = 0; b < n_blk; b++) {
                    float bs = x_blk_scale[b];
                    if (bs <= 0.0f) continue;
                    const uint8_t* qp = q_grp + (size_t)b * 128u;
                    const int32_t* xb = x_i32 + (size_t)b * 8u;
                    __m256i d = _mm256_setzero_si256();
                    for (int c = 0; c < 4; c++) {
                        __m256i raw = _mm256_loadu_si256((const __m256i*)(qp + (c << 5)));
                        __m256i w0 = _mm256_and_si256(raw, mask0F);
                        __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask0F);
                        d = _mm256_add_epi16(d, _mm256_maddubs_epi16(w0, _mm256_set1_epi32(xb[c * 2])));
                        d = _mm256_add_epi16(d, _mm256_maddubs_epi16(w1, _mm256_set1_epi32(xb[c * 2 + 1])));
                    }
                    __m256i dc = _mm256_sub_epi32(_mm256_madd_epi16(d, ones16), _mm256_set1_epi32(x_sum8[b]));
                    __m256 v_ws = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(sc_grp + (size_t)b * 8u)));
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dc), _mm256_mul_ps(v_ws, _mm256_set1_ps(bs)), acc);
                }
                store_8rows_out(out + r0 + rg * 8, n_8, acc);
            }
        }
        return;
    }

    /* Multi-token slab path (L2-cache tiled Mc=48 around tile loop, register-resident acc[6]) */
    for (int32_t m0 = 0; m0 < M; m0 += 48) {
        int32_t m1 = (m0 + 48 < M) ? (m0 + 48) : M;
        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            int32_t n_8g = (n_r + 7) >> 3;

            for (int32_t rg = 0; rg < n_8g; rg++) {
                int32_t n_8 = (rg * 8 + 8 <= n_r) ? 8 : (n_r - rg * 8);
                int32_t row_off = r0 + rg * 8;
                const uint8_t* grp_base = tptr + sizeof(DpufQrTileHdr) + (size_t)rg * (size_t)n_blk * 144u;
                const uint16_t* sc_grp = (const uint16_t*)grp_base;
                const uint8_t*  q_grp  = grp_base + (size_t)n_blk * 16u;

                for (int32_t slab_m = m0; slab_m < m1; slab_m += 6) {
                    int32_t mc = (slab_m + 6 <= m1) ? 6 : (m1 - slab_m);
                    int32_t s_idx = slab_m / 6;
                    const int32_t* xp_base = (const int32_t*)(x_q8 + (size_t)s_idx * (size_t)n_blk * 192u);
                    const float* xs_base = x_blk_scale + (size_t)s_idx * (size_t)n_blk * 6u;
                    const int32_t* xm_base = x_sum8 + (size_t)s_idx * (size_t)n_blk * 6u;

                    if (mc == 6) {
                        had4p_slab_8x6(n_blk, rows, row_off, n_8, slab_m, sc_grp, q_grp, xp_base, xs_base, xm_base, out);
                    } else {
                        had4p_slab_8x_tail(mc, n_blk, rows, row_off, n_8, slab_m, sc_grp, q_grp, xp_base, xs_base, xm_base, out);
                    }
                }
            }
        }
    }
}

DPUF_EXPORT void dpuf_matmul_qr_hadamard4_packed_f32(
    int32_t M,
    int32_t rows,
    int32_t cols,
    int32_t n_tiles,
    const float* x_in,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    const uint8_t* sign_bits,
    float* out
) {
    int32_t n_blk = cols >> 5;
    int32_t n_slabs = (M == 1) ? 1 : ((M + 5) / 6);
    int32_t M_pad = (M == 1) ? 1 : (n_slabs * 6);
    float* f_scr = (float*)malloc((size_t)M_pad * (size_t)cols * sizeof(float));
    int8_t* x_q  = (int8_t*)malloc((size_t)M_pad * (size_t)cols * sizeof(int8_t));
    float* x_bs  = (float*)malloc((size_t)M_pad * (size_t)n_blk * sizeof(float));
    int32_t* x_sm = (int32_t*)malloc((size_t)M_pad * (size_t)n_blk * sizeof(int32_t));
    if (!f_scr || !x_q || !x_bs || !x_sm) {
        free(f_scr); free(x_q); free(x_bs); free(x_sm);
        return;
    }

    if (M == 1) {
        dpuf_prep_had4_packed_slab(1, cols, x_in, sign_bits, f_scr, x_q, x_bs, x_sm, 0);
    } else {
        int32_t s;
        #pragma omp parallel for schedule(static) if(n_slabs >= 4)
        for (s = 0; s < n_slabs; s++) {
            int32_t m0 = s * 6;
            int32_t mc = (m0 + 6 <= M) ? 6 : (M - m0);
            dpuf_prep_had4_packed_slab(
                mc, cols,
                x_in + (size_t)m0 * (size_t)cols,
                sign_bits,
                f_scr + (size_t)m0 * (size_t)cols,
                x_q + (size_t)s * (size_t)n_blk * 192u,
                x_bs + (size_t)s * (size_t)n_blk * 6u,
                x_sm + (size_t)s * (size_t)n_blk * 6u,
                1
            );
        }
    }

    int32_t nth = (n_tiles >= 4) ? 8 : 1;
    if (nth > omp_get_max_threads()) nth = omp_get_max_threads();
    #pragma omp parallel num_threads(nth) if(nth > 1)
    {
        int32_t tid = omp_get_thread_num();
        int32_t nt = omp_get_num_threads();
        int32_t t0 = (n_tiles * tid) / nt;
        int32_t t1 = (n_tiles * (tid + 1)) / nt;
        dpuf_matmul_qr_hadamard4_packed_tile_range(
            M, rows, cols, t0, t1, x_q, x_bs, x_sm, tensor_base, tile_offsets, out
        );
    }

    free(f_scr); free(x_q); free(x_bs); free(x_sm);
}

DPUF_EXPORT void dpuf_embed_qr_hadamard4_packed_f32(
    int32_t n_tokens,
    int32_t cols,
    const int32_t* token_ids,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    const uint8_t* sign_bits,
    float* out
) {
    const int32_t tile_stride = 16;
    const int32_t n_blk = cols >> 5;
    for (int32_t tok = 0; tok < n_tokens; tok++) {
        int32_t r = token_ids[tok];
        int32_t t = r / tile_stride;
        int32_t lr = r % tile_stride;
        int32_t rg = lr >> 3;
        int32_t ri = lr & 7;

        const uint8_t* tptr = tensor_base + tile_offsets[t];
        const uint8_t* grp_base = tptr + sizeof(DpufQrTileHdr) + (size_t)rg * (size_t)n_blk * 144u;
        const uint16_t* sc_grp = (const uint16_t*)grp_base;
        const uint8_t*  q_grp  = grp_base + (size_t)n_blk * 16u;
        float* dst = out + (size_t)tok * (size_t)cols;

        for (int32_t b = 0; b < n_blk; b++) {
            float ws = fp16_bits_to_f32(sc_grp[(size_t)b * 8u + (size_t)ri]);
            const uint8_t* qp = q_grp + (size_t)b * 128u;
            int32_t c0 = b << 5;
            for (int32_t c = 0; c < 4; c++) {
                const uint8_t* cp = qp + (size_t)c * 32u + (size_t)ri * 4u;
                int32_t col = c0 + (c << 3);
                for (int32_t k = 0; k < 4; k++) {
                    uint8_t byte_v = cp[k];
                    dst[col + k]     = (float)((int32_t)(byte_v & 0x0Fu) - 8) * ws;
                    dst[col + 4 + k] = (float)((int32_t)(byte_v >> 4) - 8) * ws;
                }
            }
        }
        for (int32_t c = 0; c < cols; c += 64) {
            fwht64_block_f32(dst + c);
        }
        if (sign_bits) {
            apply_sign_mask_f32(cols, dst, sign_bits, dst);
        }
    }
}

/* ============================================================================
 * MODE 13 (DPUF_MODE_QR_HAD4_ASYM, blk=64) & MODE 14 (DPUF_MODE_QR_HAD6, blk=64)
 *   8-row group layout per blk64:
 *     [dm_grp: 8*fp16 d + 8*fp16 m = 32B]
 *     [q4_grp: 2*128B = 256B low nibbles]
 *     [q2_grp (Mode 14 only): 2*64B = 128B hi-2bits]
 * ============================================================================ */

static __forceinline void had4_dot_blk32_mc(
    int32_t mc,
    const uint8_t* qp,
    const int32_t* xp,
    __m256i* d32_out
) {
    const __m256i mask0F = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i d[6];
    for (int j = 0; j < mc; j++) d[j] = _mm256_setzero_si256();
    for (int c = 0; c < 4; c++) {
        __m256i raw = _mm256_loadu_si256((const __m256i*)(qp + (c << 5)));
        __m256i w0 = _mm256_and_si256(raw, mask0F);
        __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask0F);
        const int32_t* xg0 = xp + (c * 2) * 6;
        const int32_t* xg1 = xp + (c * 2 + 1) * 6;
        for (int j = 0; j < mc; j++) {
            d[j] = _mm256_add_epi16(d[j], _mm256_maddubs_epi16(w0, _mm256_set1_epi32(xg0[j])));
            d[j] = _mm256_add_epi16(d[j], _mm256_maddubs_epi16(w1, _mm256_set1_epi32(xg1[j])));
        }
    }
    for (int j = 0; j < mc; j++) d32_out[j] = _mm256_madd_epi16(d[j], ones16);
}

static __forceinline void had6_dot_blk32_mc(
    int32_t mc,
    const uint8_t* qp,
    const uint8_t* h2p,
    const int32_t* xp,
    __m256i* d32_out
) {
    const __m256i mask0F = _mm256_set1_epi8(0x0F);
    const __m256i mask03 = _mm256_set1_epi8(0x03);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i srlv_hi = _mm256_set_epi64x(6, 4, 2, 0);
    for (int j = 0; j < mc; j++) d32_out[j] = _mm256_setzero_si256();
    for (int c = 0; c < 4; c++) {
        __m256i raw = _mm256_loadu_si256((const __m256i*)(qp + (c << 5)));
        __m128i hi0_64 = _mm_loadl_epi64((const __m128i*)(h2p + (c << 4)));
        __m128i hi1_64 = _mm_loadl_epi64((const __m128i*)(h2p + (c << 4) + 8));
        __m256i wh0 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srlv_epi64(_mm256_broadcastq_epi64(hi0_64), srlv_hi), mask03), 4);
        __m256i wh1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srlv_epi64(_mm256_broadcastq_epi64(hi1_64), srlv_hi), mask03), 4);
        __m256i w0 = _mm256_or_si256(_mm256_and_si256(raw, mask0F), wh0);
        __m256i w1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(raw, 4), mask0F), wh1);
        const int32_t* xg0 = xp + (c * 2) * 6;
        const int32_t* xg1 = xp + (c * 2 + 1) * 6;
        for (int j = 0; j < mc; j++) {
            __m256i p = _mm256_add_epi16(
                _mm256_maddubs_epi16(w0, _mm256_set1_epi32(xg0[j])),
                _mm256_maddubs_epi16(w1, _mm256_set1_epi32(xg1[j]))
            );
            d32_out[j] = _mm256_add_epi32(d32_out[j], _mm256_madd_epi16(p, ones16));
        }
    }
}

static void had4a_slab_8x6(
    int32_t n_blk64, int32_t rows, int32_t row_off, int32_t n_8, int32_t slab_m,
    const uint16_t* dm_grp, const uint8_t* q4_grp,
    const int32_t* xp_base, const float* xs_base, const int32_t* xsum_base, float* out
) {
    __m256 acc[6];
    for (int j = 0; j < 6; j++) acc[j] = _mm256_setzero_ps();
    for (int32_t g = 0; g < n_blk64; g++) {
        int32_t b0 = g * 2, b1 = b0 + 1;
        __m256i d0[6], d1[6];
        had4_dot_blk32_mc(6, q4_grp + (size_t)b0 * 128u, xp_base + (size_t)b0 * 48u, d0);
        had4_dot_blk32_mc(6, q4_grp + (size_t)b1 * 128u, xp_base + (size_t)b1 * 48u, d1);
        __m256 v_d = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(dm_grp + (size_t)g * 16u)));
        __m256 v_m = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(dm_grp + (size_t)g * 16u + 8u)));
        const float* xs0 = xs_base + (size_t)b0 * 6u;
        const float* xs1 = xs_base + (size_t)b1 * 6u;
        const int32_t* xm0 = xsum_base + (size_t)b0 * 6u;
        const int32_t* xm1 = xsum_base + (size_t)b1 * 6u;
        for (int j = 0; j < 6; j++) {
            __m256 fdot = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(d0[j]), _mm256_set1_ps(xs0[j]),
                _mm256_mul_ps(_mm256_cvtepi32_ps(d1[j]), _mm256_set1_ps(xs1[j]))
            );
            float xm_s = ((float)xm0[j] * xs0[j] + (float)xm1[j] * xs1[j]) * 0.125f;
            acc[j] = _mm256_fmadd_ps(fdot, v_d, acc[j]);
            acc[j] = _mm256_fmadd_ps(v_m, _mm256_set1_ps(xm_s), acc[j]);
        }
    }
    for (int j = 0; j < 6; j++) store_8rows_out(out + (size_t)(slab_m + j) * (size_t)rows + (size_t)row_off, n_8, acc[j]);
}

static void had6_slab_8x6(
    int32_t n_blk64, int32_t rows, int32_t row_off, int32_t n_8, int32_t slab_m,
    const uint16_t* dm_grp, const uint8_t* q4_grp, const uint8_t* q2_grp,
    const int32_t* xp_base, const float* xs_base, const int32_t* xsum_base, float* out
) {
    __m256 acc[6];
    for (int j = 0; j < 6; j++) acc[j] = _mm256_setzero_ps();
    for (int32_t g = 0; g < n_blk64; g++) {
        int32_t b0 = g * 2, b1 = b0 + 1;
        __m256i d0[6], d1[6];
        had6_dot_blk32_mc(6, q4_grp + (size_t)b0 * 128u, q2_grp + (size_t)b0 * 64u, xp_base + (size_t)b0 * 48u, d0);
        had6_dot_blk32_mc(6, q4_grp + (size_t)b1 * 128u, q2_grp + (size_t)b1 * 64u, xp_base + (size_t)b1 * 48u, d1);
        __m256 v_d = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(dm_grp + (size_t)g * 16u)));
        __m256 v_m = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(dm_grp + (size_t)g * 16u + 8u)));
        const float* xs0 = xs_base + (size_t)b0 * 6u;
        const float* xs1 = xs_base + (size_t)b1 * 6u;
        const int32_t* xm0 = xsum_base + (size_t)b0 * 6u;
        const int32_t* xm1 = xsum_base + (size_t)b1 * 6u;
        for (int j = 0; j < 6; j++) {
            __m256 fdot = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(d0[j]), _mm256_set1_ps(xs0[j]),
                _mm256_mul_ps(_mm256_cvtepi32_ps(d1[j]), _mm256_set1_ps(xs1[j]))
            );
            float xm_s = ((float)xm0[j] * xs0[j] + (float)xm1[j] * xs1[j]) * 0.125f;
            acc[j] = _mm256_fmadd_ps(fdot, v_d, acc[j]);
            acc[j] = _mm256_fmadd_ps(v_m, _mm256_set1_ps(xm_s), acc[j]);
        }
    }
    for (int j = 0; j < 6; j++) store_8rows_out(out + (size_t)(slab_m + j) * (size_t)rows + (size_t)row_off, n_8, acc[j]);
}

static void had_asym_slab_8x_tail(
    int32_t mc, int32_t is_had6, int32_t n_blk64, int32_t rows, int32_t row_off, int32_t n_8, int32_t slab_m,
    const uint16_t* dm_grp, const uint8_t* q4_grp, const uint8_t* q2_grp,
    const int32_t* xp_base, const float* xs_base, const int32_t* xsum_base, float* out
) {
    __m256 acc[6];
    for (int j = 0; j < mc; j++) acc[j] = _mm256_setzero_ps();
    for (int32_t g = 0; g < n_blk64; g++) {
        int32_t b0 = g * 2, b1 = b0 + 1;
        __m256i d0[6], d1[6];
        if (is_had6) {
            had6_dot_blk32_mc(mc, q4_grp + (size_t)b0 * 128u, q2_grp + (size_t)b0 * 64u, xp_base + (size_t)b0 * 48u, d0);
            had6_dot_blk32_mc(mc, q4_grp + (size_t)b1 * 128u, q2_grp + (size_t)b1 * 64u, xp_base + (size_t)b1 * 48u, d1);
        } else {
            had4_dot_blk32_mc(mc, q4_grp + (size_t)b0 * 128u, xp_base + (size_t)b0 * 48u, d0);
            had4_dot_blk32_mc(mc, q4_grp + (size_t)b1 * 128u, xp_base + (size_t)b1 * 48u, d1);
        }
        __m256 v_d = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(dm_grp + (size_t)g * 16u)));
        __m256 v_m = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(dm_grp + (size_t)g * 16u + 8u)));
        const float* xs0 = xs_base + (size_t)b0 * 6u;
        const float* xs1 = xs_base + (size_t)b1 * 6u;
        const int32_t* xm0 = xsum_base + (size_t)b0 * 6u;
        const int32_t* xm1 = xsum_base + (size_t)b1 * 6u;
        for (int j = 0; j < mc; j++) {
            __m256 fdot = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(d0[j]), _mm256_set1_ps(xs0[j]),
                _mm256_mul_ps(_mm256_cvtepi32_ps(d1[j]), _mm256_set1_ps(xs1[j]))
            );
            float xm_s = ((float)xm0[j] * xs0[j] + (float)xm1[j] * xs1[j]) * 0.125f;
            acc[j] = _mm256_fmadd_ps(fdot, v_d, acc[j]);
            acc[j] = _mm256_fmadd_ps(v_m, _mm256_set1_ps(xm_s), acc[j]);
        }
    }
    for (int j = 0; j < mc; j++) store_8rows_out(out + (size_t)(slab_m + j) * (size_t)rows + (size_t)row_off, n_8, acc[j]);
}

static void had_asym_tile_range_impl(
    int32_t is_had6, int32_t M, int32_t rows, int32_t cols, int32_t t_start, int32_t t_end,
    const int8_t* x_q8, const float* x_blk_scale, const int32_t* x_sum8,
    const uint8_t* tensor_base, const uint64_t* tile_offsets, float* out
) {
    const int32_t n_blk = cols >> 5;
    const int32_t n_64 = cols >> 6;
    const int32_t tile_stride = 16;
    const size_t grp_bytes = is_had6 ? 416u : 288u;

    if (M == 1) {
        const __m256i mask0F = _mm256_set1_epi8(0x0F);
        const __m256i mask03 = _mm256_set1_epi8(0x03);
        const __m256i ones16 = _mm256_set1_epi16(1);
        const __m256i srlv_hi = _mm256_set_epi64x(6, 4, 2, 0);
        const int32_t* x_i32 = (const int32_t*)x_q8;

        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            int32_t n_8g = (n_r + 7) >> 3;

            for (int32_t rg = 0; rg < n_8g; rg++) {
                int32_t n_8 = (rg * 8 + 8 <= n_r) ? 8 : (n_r - rg * 8);
                const uint8_t* grp_base = tptr + sizeof(DpufQrTileHdr) + (size_t)rg * (size_t)n_64 * grp_bytes;
                const uint16_t* dm_grp = (const uint16_t*)grp_base;
                const uint8_t* q4_grp = grp_base + (size_t)n_64 * 32u;
                const uint8_t* q2_grp = is_had6 ? (q4_grp + (size_t)n_64 * 256u) : NULL;

                __m256 acc = _mm256_setzero_ps();
                for (int32_t g = 0; g < n_64; g++) {
                    int32_t b[2] = { g * 2, g * 2 + 1 };
                    __m256i d_b[2];
                    for (int k = 0; k < 2; k++) {
                        const uint8_t* qp = q4_grp + (size_t)b[k] * 128u;
                        const int32_t* xb = x_i32 + (size_t)b[k] * 8u;
                        if (is_had6) {
                            const uint8_t* h2p = q2_grp + (size_t)b[k] * 64u;
                            __m256i d32 = _mm256_setzero_si256();
                            for (int c = 0; c < 4; c++) {
                                __m256i raw = _mm256_loadu_si256((const __m256i*)(qp + (c << 5)));
                                __m128i hi0_64 = _mm_loadl_epi64((const __m128i*)(h2p + (c << 4)));
                                __m128i hi1_64 = _mm_loadl_epi64((const __m128i*)(h2p + (c << 4) + 8));
                                __m256i wh0 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srlv_epi64(_mm256_broadcastq_epi64(hi0_64), srlv_hi), mask03), 4);
                                __m256i wh1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srlv_epi64(_mm256_broadcastq_epi64(hi1_64), srlv_hi), mask03), 4);
                                __m256i w0 = _mm256_or_si256(_mm256_and_si256(raw, mask0F), wh0);
                                __m256i w1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(raw, 4), mask0F), wh1);
                                __m256i p = _mm256_add_epi16(
                                    _mm256_maddubs_epi16(w0, _mm256_set1_epi32(xb[c * 2])),
                                    _mm256_maddubs_epi16(w1, _mm256_set1_epi32(xb[c * 2 + 1]))
                                );
                                d32 = _mm256_add_epi32(d32, _mm256_madd_epi16(p, ones16));
                            }
                            d_b[k] = d32;
                        } else {
                            __m256i d16 = _mm256_setzero_si256();
                            for (int c = 0; c < 4; c++) {
                                __m256i raw = _mm256_loadu_si256((const __m256i*)(qp + (c << 5)));
                                __m256i w0 = _mm256_and_si256(raw, mask0F);
                                __m256i w1 = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask0F);
                                d16 = _mm256_add_epi16(d16, _mm256_maddubs_epi16(w0, _mm256_set1_epi32(xb[c * 2])));
                                d16 = _mm256_add_epi16(d16, _mm256_maddubs_epi16(w1, _mm256_set1_epi32(xb[c * 2 + 1])));
                            }
                            d_b[k] = _mm256_madd_epi16(d16, ones16);
                        }
                    }
                    float xs0 = x_blk_scale[b[0]], xs1 = x_blk_scale[b[1]];
                    __m256 fdot = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(d_b[0]), _mm256_set1_ps(xs0),
                        _mm256_mul_ps(_mm256_cvtepi32_ps(d_b[1]), _mm256_set1_ps(xs1))
                    );
                    float xm_s = ((float)x_sum8[b[0]] * xs0 + (float)x_sum8[b[1]] * xs1) * 0.125f;
                    __m256 v_d = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(dm_grp + (size_t)g * 16u)));
                    __m256 v_m = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(dm_grp + (size_t)g * 16u + 8u)));
                    acc = _mm256_fmadd_ps(fdot, v_d, acc);
                    acc = _mm256_fmadd_ps(v_m, _mm256_set1_ps(xm_s), acc);
                }
                store_8rows_out(out + r0 + rg * 8, n_8, acc);
            }
        }
        return;
    }

    for (int32_t m0 = 0; m0 < M; m0 += 48) {
        int32_t m1 = (m0 + 48 < M) ? (m0 + 48) : M;
        for (int32_t t = t_start; t < t_end; t++) {
            const uint8_t* tptr = tensor_base + tile_offsets[t];
            const DpufQrTileHdr* th = (const DpufQrTileHdr*)tptr;
            int32_t n_r = (int32_t)th->n_rows;
            int32_t r0 = t * tile_stride;
            int32_t n_8g = (n_r + 7) >> 3;

            for (int32_t rg = 0; rg < n_8g; rg++) {
                int32_t n_8 = (rg * 8 + 8 <= n_r) ? 8 : (n_r - rg * 8);
                int32_t row_off = r0 + rg * 8;
                const uint8_t* grp_base = tptr + sizeof(DpufQrTileHdr) + (size_t)rg * (size_t)n_64 * grp_bytes;
                const uint16_t* dm_grp = (const uint16_t*)grp_base;
                const uint8_t* q4_grp = grp_base + (size_t)n_64 * 32u;
                const uint8_t* q2_grp = is_had6 ? (q4_grp + (size_t)n_64 * 256u) : NULL;

                for (int32_t slab_m = m0; slab_m < m1; slab_m += 6) {
                    int32_t mc = (slab_m + 6 <= m1) ? 6 : (m1 - slab_m);
                    int32_t s_idx = slab_m / 6;
                    const int32_t* xp_base = (const int32_t*)(x_q8 + (size_t)s_idx * (size_t)n_blk * 192u);
                    const float* xs_base = x_blk_scale + (size_t)s_idx * (size_t)n_blk * 6u;
                    const int32_t* xm_base = x_sum8 + (size_t)s_idx * (size_t)n_blk * 6u;

                    if (mc == 6) {
                        if (is_had6) had6_slab_8x6(n_64, rows, row_off, n_8, slab_m, dm_grp, q4_grp, q2_grp, xp_base, xs_base, xm_base, out);
                        else had4a_slab_8x6(n_64, rows, row_off, n_8, slab_m, dm_grp, q4_grp, xp_base, xs_base, xm_base, out);
                    } else {
                        had_asym_slab_8x_tail(mc, is_had6, n_64, rows, row_off, n_8, slab_m, dm_grp, q4_grp, q2_grp, xp_base, xs_base, xm_base, out);
                    }
                }
            }
        }
    }
}

void dpuf_matmul_qr_had4_asym_tile_range(
    int32_t M, int32_t rows, int32_t cols, int32_t t_start, int32_t t_end,
    const int8_t* x_q8, const float* x_blk_scale, const int32_t* x_sum8,
    const uint8_t* tensor_base, const uint64_t* tile_offsets, float* out
) {
    had_asym_tile_range_impl(0, M, rows, cols, t_start, t_end, x_q8, x_blk_scale, x_sum8, tensor_base, tile_offsets, out);
}

void dpuf_matmul_qr_had6_tile_range(
    int32_t M, int32_t rows, int32_t cols, int32_t t_start, int32_t t_end,
    const int8_t* x_q8, const float* x_blk_scale, const int32_t* x_sum8,
    const uint8_t* tensor_base, const uint64_t* tile_offsets, float* out
) {
    had_asym_tile_range_impl(1, M, rows, cols, t_start, t_end, x_q8, x_blk_scale, x_sum8, tensor_base, tile_offsets, out);
}

static void had_asym_matmul_f32_impl(
    int32_t is_had6, int32_t M, int32_t rows, int32_t cols, int32_t n_tiles,
    const float* x_in, const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
) {
    int32_t n_blk = cols >> 5;
    int32_t n_slabs = (M == 1) ? 1 : ((M + 5) / 6);
    int32_t M_pad = (M == 1) ? 1 : (n_slabs * 6);
    float* f_scr = (float*)malloc((size_t)M_pad * (size_t)cols * sizeof(float));
    int8_t* x_q  = (int8_t*)malloc((size_t)M_pad * (size_t)cols * sizeof(int8_t));
    float* x_bs  = (float*)malloc((size_t)M_pad * (size_t)n_blk * sizeof(float));
    int32_t* x_sm = (int32_t*)malloc((size_t)M_pad * (size_t)n_blk * sizeof(int32_t));
    if (!f_scr || !x_q || !x_bs || !x_sm) {
        free(f_scr); free(x_q); free(x_bs); free(x_sm);
        return;
    }
    if (M == 1) {
        dpuf_prep_had4_packed_slab(1, cols, x_in, sign_bits, f_scr, x_q, x_bs, x_sm, 0);
    } else {
        int32_t s;
        #pragma omp parallel for schedule(static) if(n_slabs >= 4)
        for (s = 0; s < n_slabs; s++) {
            int32_t m0 = s * 6;
            int32_t mc = (m0 + 6 <= M) ? 6 : (M - m0);
            dpuf_prep_had4_packed_slab(
                mc, cols, x_in + (size_t)m0 * (size_t)cols, sign_bits,
                f_scr + (size_t)m0 * (size_t)cols,
                x_q + (size_t)s * (size_t)n_blk * 192u,
                x_bs + (size_t)s * (size_t)n_blk * 6u,
                x_sm + (size_t)s * (size_t)n_blk * 6u, 1
            );
        }
    }
    int32_t nth = (n_tiles >= 4) ? 8 : 1;
    if (nth > omp_get_max_threads()) nth = omp_get_max_threads();
    #pragma omp parallel num_threads(nth) if(nth > 1)
    {
        int32_t tid = omp_get_thread_num();
        int32_t nt = omp_get_num_threads();
        int32_t t0 = (n_tiles * tid) / nt;
        int32_t t1 = (n_tiles * (tid + 1)) / nt;
        had_asym_tile_range_impl(is_had6, M, rows, cols, t0, t1, x_q, x_bs, x_sm, tensor_base, tile_offsets, out);
    }
    free(f_scr); free(x_q); free(x_bs); free(x_sm);
}

DPUF_EXPORT void dpuf_matmul_qr_had4_asym_f32(
    int32_t M, int32_t rows, int32_t cols, int32_t n_tiles,
    const float* x_in, const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
) {
    had_asym_matmul_f32_impl(0, M, rows, cols, n_tiles, x_in, tensor_base, tile_offsets, sign_bits, out);
}

DPUF_EXPORT void dpuf_matmul_qr_had6_f32(
    int32_t M, int32_t rows, int32_t cols, int32_t n_tiles,
    const float* x_in, const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
) {
    had_asym_matmul_f32_impl(1, M, rows, cols, n_tiles, x_in, tensor_base, tile_offsets, sign_bits, out);
}

static void embed_qr_had_asym_impl(
    int32_t is_had6, int32_t n_tokens, int32_t cols, const int32_t* token_ids,
    const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
) {
    const int32_t tile_stride = 16;
    const int32_t n_64 = cols >> 6;
    const size_t grp_bytes = is_had6 ? 416u : 288u;
    for (int32_t tok = 0; tok < n_tokens; tok++) {
        int32_t r = token_ids[tok];
        int32_t t = r / tile_stride;
        int32_t lr = r % tile_stride;
        int32_t rg = lr >> 3;
        int32_t ri = lr & 7;

        const uint8_t* tptr = tensor_base + tile_offsets[t];
        const uint8_t* grp_base = tptr + sizeof(DpufQrTileHdr) + (size_t)rg * (size_t)n_64 * grp_bytes;
        const uint16_t* dm_grp = (const uint16_t*)grp_base;
        const uint8_t* q4_grp = grp_base + (size_t)n_64 * 32u;
        const uint8_t* q2_grp = is_had6 ? (q4_grp + (size_t)n_64 * 256u) : NULL;
        float* dst = out + (size_t)tok * (size_t)cols;

        for (int32_t g = 0; g < n_64; g++) {
            float wd = fp16_bits_to_f32(dm_grp[(size_t)g * 16u + (size_t)ri]);
            float wm = fp16_bits_to_f32(dm_grp[(size_t)g * 16u + 8u + (size_t)ri]);
            for (int sub = 0; sub < 2; sub++) {
                int32_t b = g * 2 + sub;
                const uint8_t* qp = q4_grp + (size_t)b * 128u;
                const uint8_t* h2p = is_had6 ? (q2_grp + (size_t)b * 64u) : NULL;
                int32_t c0 = b << 5;
                for (int32_t c = 0; c < 4; c++) {
                    const uint8_t* cp = qp + (size_t)c * 32u + (size_t)ri * 4u;
                    const uint8_t* hp0 = is_had6 ? (h2p + (size_t)c * 16u) : NULL;
                    const uint8_t* hp1 = is_had6 ? (hp0 + 8u) : NULL;
                    int32_t p_idx = ri * 4;
                    int32_t s_shift = (p_idx >> 3) * 2;
                    int32_t p_byte0 = p_idx & 7;
                    int32_t col = c0 + (c << 3);
                    for (int32_t k = 0; k < 4; k++) {
                        uint8_t byte_v = cp[k];
                        int32_t q0 = (int32_t)(byte_v & 0x0Fu);
                        int32_t q1 = (int32_t)(byte_v >> 4);
                        if (is_had6) {
                            q0 |= ((int32_t)((hp0[p_byte0 + k] >> s_shift) & 0x03)) << 4;
                            q1 |= ((int32_t)((hp1[p_byte0 + k] >> s_shift) & 0x03)) << 4;
                        }
                        dst[col + k]     = (float)q0 * wd + wm;
                        dst[col + 4 + k] = (float)q1 * wd + wm;
                    }
                }
            }
        }
        for (int32_t c = 0; c < cols; c += 64) {
            fwht64_block_f32(dst + c);
        }
        if (sign_bits) {
            apply_sign_mask_f32(cols, dst, sign_bits, dst);
        }
    }
}

DPUF_EXPORT void dpuf_embed_qr_had4_asym_f32(
    int32_t n_tokens, int32_t cols, const int32_t* token_ids,
    const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
) {
    embed_qr_had_asym_impl(0, n_tokens, cols, token_ids, tensor_base, tile_offsets, sign_bits, out);
}

DPUF_EXPORT void dpuf_embed_qr_had6_f32(
    int32_t n_tokens, int32_t cols, const int32_t* token_ids,
    const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
) {
    embed_qr_had_asym_impl(1, n_tokens, cols, token_ids, tensor_base, tile_offsets, sign_bits, out);
}
