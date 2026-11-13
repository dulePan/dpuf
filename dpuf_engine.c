/*
 * dpuf_engine.c — Pure C / AVX2+FMA Persistent-Threadpool Transformer Engine for DPUF
 * Copyright (c) 2026 D. P. (dule_pan). Licensed under DPUF Community & Enterprise License.
 *
 * Implements Steps 1 & 2:
 *   - Complete Transformer forward pass (RMSNorm, QK-Norm, RoPE, KV-Cache, GQA, SwiGLU, LM-Head)
 *   - Supports Qwen3 & Llama-3 architectures natively over Zero-Copy QR-Tiled mmap weights
 *   - Single persistent OpenMP team per forward call with sub-microsecond atomic spin barriers
 *   - Supports Mode 8 (QR-ZD4), Mode 9 (Varijanta A Split-Dense Int4 + Sparse CSR),
 *     and Mode 10 (Varijanta B Walsh-Hadamard FWHT-64 + Pure Dense Blk32 Int4)
 */
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <intrin.h>
#include <omp.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "dpuf_kernels.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DPUF_ENGINE_MAX_BATCH 256

static __forceinline float eng_hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

typedef struct {
    __declspec(align(64)) volatile long arrived;
    __declspec(align(64)) volatile long sense;
} DpufSpinBarrier;

static __forceinline void engine_barrier(DpufSpinBarrier* bar, int32_t n_threads, int32_t* local_sense) {
    if (n_threads <= 1) return;
    long my_sense = 1 - *local_sense;
    *local_sense = (int32_t)my_sense;
#ifdef _WIN32
    MemoryBarrier();
    long prev = _InterlockedIncrement(&bar->arrived);
    if (prev == n_threads) {
        bar->arrived = 0;
        MemoryBarrier();
        bar->sense = my_sense;
    } else {
        uint32_t spins = 0;
        while (bar->sense != my_sense) {
            _mm_pause();
            if (++spins > 32768u) {
                SwitchToThread();
                spins = 0;
            }
        }
    }
#else
    (void)bar;
    #pragma omp barrier
#endif
}

typedef struct {
    DpufModelConfig cfg;
    DpufLayerWeights* layers;
    DpufWeightDesc w_tok_embd;
    DpufWeightDesc w_lm_head;
    const float* output_norm;

    /* Precomputed RoPE tables [max_seq_len, head_dim / 2] */
    float* cos_tab;
    float* sin_tab;

    /* Persistent KV Cache [num_layers, num_kv_heads, max_seq_len, head_dim] (contiguous over pos) */
    float*  k_cache;
    float*  v_cache;
    int8_t* k_cache_q8;
    int8_t* v_cache_q8;
    float*  k_scale_q8;
    float*  v_scale_q8;

    /* Scratch buffers for up to DPUF_ENGINE_MAX_BATCH tokens */
    float* h_buf;
    float* hn_buf;
    float* q_buf;
    float* k_buf;
    float* v_buf;
    float* attn_out;
    float* gate_buf;
    float* up_buf;
    float* ffn_hidden;

    /* 3 parallel activation prep slots (for Q/K/V and Gate/Up fusion) */
    float*   prep_f32[3];
    int8_t*  prep_q8[3];
    float*   prep_scale[3];
    int32_t* prep_xsum[3];

    /* Per-layer sign-sharing flags for Hadamard co-inputs (QKV & Gate/Up) */
    uint8_t* layer_share_kv;
    uint8_t* layer_share_gu;

    /* Per-thread attention score scratch [n_threads, max_seq_len] */
    float* attn_scores;

    /* Per-layer dynamic atomic counters (16 longs = 64B cache-line per stage) */
    volatile long* work_ctr;
    size_t work_ctr_bytes;

    /* Physical-core affinity masks to prevent SMT sibling collisions */
    uint64_t core_masks[32];
    int32_t  n_phys_cores;
    float    swiglu_deadzone;

    DpufSpinBarrier bar;
} DpufEngine;

static void precompute_rope(DpufEngine* eng) {
    int32_t hd = eng->cfg.head_dim;
    int32_t half = hd >> 1;
    int32_t max_seq = eng->cfg.max_seq_len;
    float theta = eng->cfg.rope_theta;

    float* inv_freq = (float*)malloc((size_t)half * sizeof(float));
    if (!inv_freq) return;

    for (int32_t j = 0; j < half; j++) {
        float freq = 1.0f / powf(theta, (float)(2 * j) / (float)hd);
        if (eng->cfg.use_llama3_rope) {
            float factor = eng->cfg.rope_factor;
            float low_f = eng->cfg.rope_low_freq;
            float high_f = eng->cfg.rope_high_freq;
            float orig_max = (float)eng->cfg.rope_orig_max_pos;
            float wavelen = (2.0f * (float)M_PI) / freq;
            float low_wavelen = orig_max / low_f;
            float high_wavelen = orig_max / high_f;
            if (wavelen > low_wavelen) {
                freq /= factor;
            } else if (wavelen >= high_wavelen) {
                float smooth = (orig_max / wavelen - low_f) / (high_f - low_f);
                freq = (1.0f - smooth) * (freq / factor) + smooth * freq;
            }
        }
        inv_freq[j] = freq;
    }

    for (int32_t p = 0; p < max_seq; p++) {
        float* cp = eng->cos_tab + (size_t)p * (size_t)half;
        float* sp = eng->sin_tab + (size_t)p * (size_t)half;
        float pf = (float)p;
        for (int32_t j = 0; j < half; j++) {
            float ang = pf * inv_freq[j];
            cp[j] = cosf(ang);
            sp[j] = sinf(ang);
        }
    }
    free(inv_freq);
}

static void rms_norm_f32(int32_t n, const float* x, const float* w, float eps, float* out) {
    __m256 acc = _mm256_setzero_ps();
    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    float ss = eng_hsum256_ps(acc);
    for (; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    __m256 vinv = _mm256_set1_ps(inv);
    for (i = 0; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw));
    }
    for (; i < n; i++) out[i] = (x[i] * inv) * w[i];
}

static void add_and_rms_norm_f32(
    int32_t n, float* x, const float* res, const float* w, float eps, float* out
) {
    if (!res) {
        rms_norm_f32(n, x, w, eps, out);
        return;
    }
    __m256 acc = _mm256_setzero_ps();
    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 s = _mm256_add_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(res + i));
        _mm256_storeu_ps(x + i, s);
        acc = _mm256_fmadd_ps(s, s, acc);
    }
    float ss = eng_hsum256_ps(acc);
    for (; i < n; i++) {
        float s = x[i] + res[i];
        x[i] = s;
        ss += s * s;
    }
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    __m256 vinv = _mm256_set1_ps(inv);
    for (i = 0; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(w + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw));
    }
    for (; i < n; i++) out[i] = (x[i] * inv) * w[i];
}

static void apply_rope_neox_f32(int32_t hd, float* vec, const float* cos_p, const float* sin_p) {
    int32_t half = hd >> 1;
    int32_t j = 0;
    for (; j + 7 < half; j += 8) {
        __m256 x1 = _mm256_loadu_ps(vec + j);
        __m256 x2 = _mm256_loadu_ps(vec + j + half);
        __m256 vc = _mm256_loadu_ps(cos_p + j);
        __m256 vs = _mm256_loadu_ps(sin_p + j);
        __m256 o1 = _mm256_fmsub_ps(x1, vc, _mm256_mul_ps(x2, vs));
        __m256 o2 = _mm256_fmadd_ps(x2, vc, _mm256_mul_ps(x1, vs));
        _mm256_storeu_ps(vec + j, o1);
        _mm256_storeu_ps(vec + j + half, o2);
    }
    for (; j < half; j++) {
        float x1 = vec[j];
        float x2 = vec[j + half];
        float c = cos_p[j];
        float s = sin_p[j];
        vec[j]        = x1 * c - x2 * s;
        vec[j + half] = x2 * c + x1 * s;
    }
}

static __forceinline __m256 eng_avx2_exp_ps(__m256 x) {
    const __m256 ln2_hi  = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo  = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 inv_ln2 = _mm256_set1_ps(1.44269504088896341f);
    x = _mm256_max_ps(_mm256_set1_ps(-45.0f), _mm256_min_ps(x, _mm256_set1_ps(45.0f)));
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, inv_ln2), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(fx, ln2_hi, x);
    r = _mm256_fnmadd_ps(fx, ln2_lo, r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
    __m256 r2 = _mm256_mul_ps(r, r);
    __m256 exp_r = _mm256_add_ps(_mm256_fmadd_ps(p, r2, r), _mm256_set1_ps(1.0f));
    __m256i n = _mm256_cvtps_epi32(fx);
    n = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(exp_r, _mm256_castsi256_ps(n));
}

static __forceinline __m128 hsum4_256_ps(__m256 a0, __m256 a1, __m256 a2, __m256 a3) {
    __m256 h01 = _mm256_hadd_ps(a0, a1);
    __m256 h23 = _mm256_hadd_ps(a2, a3);
    __m256 h   = _mm256_hadd_ps(h01, h23);
    return _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps(h, 1));
}

static __forceinline float dot_f32_avx2(int32_t n, const float* a, const float* b) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int32_t i = 0;
    for (; i + 15 < n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 7 < n; i += 8) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    }
    float s = eng_hsum256_ps(_mm256_add_ps(acc0, acc1));
    for (; i < n; i++) s += a[i] * b[i];
    return s;
}

static void gqa_attn_head_pair_f32(
    int32_t hd,
    int32_t kv_len,
    float attn_scale,
    const float* kc_head,
    const float* vc_head,
    const float* q0,
    const float* q1, /* NULL for single head */
    float* scores0,
    float* scores1,  /* NULL for single head */
    float* out0,
    float* out1      /* NULL for single head */
) {
    const __m256 vscale = _mm256_set1_ps(attn_scale);
    const __m128 vscale128 = _mm_set1_ps(attn_scale);
    int32_t t = 0;

    if (q1 && (hd & 7) == 0) {
        __m128 vmax0 = _mm_set1_ps(-1e30f);
        __m128 vmax1 = _mm_set1_ps(-1e30f);
        for (; t + 3 < kv_len; t += 4) {
            const float* kp = kc_head + (size_t)t * (size_t)hd;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
            __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
            for (int32_t i = 0; i < hd; i += 8) {
                __m256 vq0 = _mm256_loadu_ps(q0 + i);
                __m256 vq1 = _mm256_loadu_ps(q1 + i);
                __m256 vk0 = _mm256_loadu_ps(kp + i);
                a0 = _mm256_fmadd_ps(vq0, vk0, a0);
                b0 = _mm256_fmadd_ps(vq1, vk0, b0);
                __m256 vk1 = _mm256_loadu_ps(kp + (size_t)hd + i);
                a1 = _mm256_fmadd_ps(vq0, vk1, a1);
                b1 = _mm256_fmadd_ps(vq1, vk1, b1);
                __m256 vk2 = _mm256_loadu_ps(kp + (size_t)hd * 2u + i);
                a2 = _mm256_fmadd_ps(vq0, vk2, a2);
                b2 = _mm256_fmadd_ps(vq1, vk2, b2);
                __m256 vk3 = _mm256_loadu_ps(kp + (size_t)hd * 3u + i);
                a3 = _mm256_fmadd_ps(vq0, vk3, a3);
                b3 = _mm256_fmadd_ps(vq1, vk3, b3);
            }
            __m128 s0 = _mm_mul_ps(hsum4_256_ps(a0, a1, a2, a3), vscale128);
            __m128 s1 = _mm_mul_ps(hsum4_256_ps(b0, b1, b2, b3), vscale128);
            _mm_storeu_ps(scores0 + t, s0);
            _mm_storeu_ps(scores1 + t, s1);
            vmax0 = _mm_max_ps(vmax0, s0);
            vmax1 = _mm_max_ps(vmax1, s1);
        }
        vmax0 = _mm_max_ps(vmax0, _mm_shuffle_ps(vmax0, vmax0, 0x4E));
        vmax0 = _mm_max_ps(vmax0, _mm_shuffle_ps(vmax0, vmax0, 0xB1));
        float max0 = _mm_cvtss_f32(vmax0);
        vmax1 = _mm_max_ps(vmax1, _mm_shuffle_ps(vmax1, vmax1, 0x4E));
        vmax1 = _mm_max_ps(vmax1, _mm_shuffle_ps(vmax1, vmax1, 0xB1));
        float max1 = _mm_cvtss_f32(vmax1);

        for (; t < kv_len; t++) {
            const float* kp = kc_head + (size_t)t * (size_t)hd;
            __m256 a0 = _mm256_setzero_ps(), b0 = _mm256_setzero_ps();
            for (int32_t i = 0; i < hd; i += 8) {
                __m256 vk = _mm256_loadu_ps(kp + i);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q0 + i), vk, a0);
                b0 = _mm256_fmadd_ps(_mm256_loadu_ps(q1 + i), vk, b0);
            }
            float s0 = eng_hsum256_ps(a0) * attn_scale;
            float s1 = eng_hsum256_ps(b0) * attn_scale;
            scores0[t] = s0; if (s0 > max0) max0 = s0;
            scores1[t] = s1; if (s1 > max1) max1 = s1;
        }
        (void)vscale;
        /* Softmax both score arrays */
        {
            __m256 vm0 = _mm256_set1_ps(max0), vm1 = _mm256_set1_ps(max1);
            __m256 vs0 = _mm256_setzero_ps(), vs1 = _mm256_setzero_ps();
            int32_t k = 0;
            for (; k + 7 < kv_len; k += 8) {
                __m256 e0 = eng_avx2_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(scores0 + k), vm0));
                __m256 e1 = eng_avx2_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(scores1 + k), vm1));
                _mm256_storeu_ps(scores0 + k, e0);
                _mm256_storeu_ps(scores1 + k, e1);
                vs0 = _mm256_add_ps(vs0, e0);
                vs1 = _mm256_add_ps(vs1, e1);
            }
            float sp0 = eng_hsum256_ps(vs0), sp1 = eng_hsum256_ps(vs1);
            for (; k < kv_len; k++) {
                float e0 = expf(scores0[k] - max0); scores0[k] = e0; sp0 += e0;
                float e1 = expf(scores1[k] - max1); scores1[k] = e1; sp1 += e1;
            }
            float inv0 = 1.0f / (sp0 + 1e-12f), inv1 = 1.0f / (sp1 + 1e-12f);
            __m256 vi0 = _mm256_set1_ps(inv0), vi1 = _mm256_set1_ps(inv1);
            for (k = 0; k + 7 < kv_len; k += 8) {
                _mm256_storeu_ps(scores0 + k, _mm256_mul_ps(_mm256_loadu_ps(scores0 + k), vi0));
                _mm256_storeu_ps(scores1 + k, _mm256_mul_ps(_mm256_loadu_ps(scores1 + k), vi1));
            }
            for (; k < kv_len; k++) { scores0[k] *= inv0; scores1[k] *= inv1; }
        }

        /* Register-resident weighted sum of V sharing V loads across q0 & q1 (32 floats = 4 YMM per head) */
        for (int32_t d0 = 0; d0 < hd; d0 += 32) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
            __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
            int32_t k = 0;
            for (; k + 1 < kv_len; k += 2) {
                const float* vp0 = vc_head + (size_t)k * (size_t)hd + d0;
                const float* vp1 = vp0 + hd;
                __m256 pa0 = _mm256_set1_ps(scores0[k]),     pa1 = _mm256_set1_ps(scores0[k + 1]);
                __m256 pb0 = _mm256_set1_ps(scores1[k]),     pb1 = _mm256_set1_ps(scores1[k + 1]);
                __m256 v0 = _mm256_loadu_ps(vp0);
                a0 = _mm256_fmadd_ps(pa0, v0, a0); b0 = _mm256_fmadd_ps(pb0, v0, b0);
                __m256 v1 = _mm256_loadu_ps(vp0 + 8);
                a1 = _mm256_fmadd_ps(pa0, v1, a1); b1 = _mm256_fmadd_ps(pb0, v1, b1);
                __m256 v2 = _mm256_loadu_ps(vp0 + 16);
                a2 = _mm256_fmadd_ps(pa0, v2, a2); b2 = _mm256_fmadd_ps(pb0, v2, b2);
                __m256 v3 = _mm256_loadu_ps(vp0 + 24);
                a3 = _mm256_fmadd_ps(pa0, v3, a3); b3 = _mm256_fmadd_ps(pb0, v3, b3);
                v0 = _mm256_loadu_ps(vp1);
                a0 = _mm256_fmadd_ps(pa1, v0, a0); b0 = _mm256_fmadd_ps(pb1, v0, b0);
                v1 = _mm256_loadu_ps(vp1 + 8);
                a1 = _mm256_fmadd_ps(pa1, v1, a1); b1 = _mm256_fmadd_ps(pb1, v1, b1);
                v2 = _mm256_loadu_ps(vp1 + 16);
                a2 = _mm256_fmadd_ps(pa1, v2, a2); b2 = _mm256_fmadd_ps(pb1, v2, b2);
                v3 = _mm256_loadu_ps(vp1 + 24);
                a3 = _mm256_fmadd_ps(pa1, v3, a3); b3 = _mm256_fmadd_ps(pb1, v3, b3);
            }
            if (k < kv_len) {
                const float* vp = vc_head + (size_t)k * (size_t)hd + d0;
                __m256 pa = _mm256_set1_ps(scores0[k]), pb = _mm256_set1_ps(scores1[k]);
                __m256 v0 = _mm256_loadu_ps(vp);      a0 = _mm256_fmadd_ps(pa, v0, a0); b0 = _mm256_fmadd_ps(pb, v0, b0);
                __m256 v1 = _mm256_loadu_ps(vp + 8);  a1 = _mm256_fmadd_ps(pa, v1, a1); b1 = _mm256_fmadd_ps(pb, v1, b1);
                __m256 v2 = _mm256_loadu_ps(vp + 16); a2 = _mm256_fmadd_ps(pa, v2, a2); b2 = _mm256_fmadd_ps(pb, v2, b2);
                __m256 v3 = _mm256_loadu_ps(vp + 24); a3 = _mm256_fmadd_ps(pa, v3, a3); b3 = _mm256_fmadd_ps(pb, v3, b3);
            }
            _mm256_storeu_ps(out0 + d0,      a0); _mm256_storeu_ps(out1 + d0,      b0);
            _mm256_storeu_ps(out0 + d0 + 8,  a1); _mm256_storeu_ps(out1 + d0 + 8,  b1);
            _mm256_storeu_ps(out0 + d0 + 16, a2); _mm256_storeu_ps(out1 + d0 + 16, b2);
            _mm256_storeu_ps(out0 + d0 + 24, a3); _mm256_storeu_ps(out1 + d0 + 24, b3);
        }
        return;
    }

    /* Single head fast path */
    __m128 vmax = _mm_set1_ps(-1e30f);
    if ((hd & 7) == 0) {
        for (; t + 3 < kv_len; t += 4) {
            const float* kp = kc_head + (size_t)t * (size_t)hd;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            for (int32_t i = 0; i < hd; i += 8) {
                __m256 vq = _mm256_loadu_ps(q0 + i);
                a0 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(kp + i),                    a0);
                a1 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(kp + (size_t)hd + i),       a1);
                a2 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(kp + (size_t)hd * 2u + i),  a2);
                a3 = _mm256_fmadd_ps(vq, _mm256_loadu_ps(kp + (size_t)hd * 3u + i),  a3);
            }
            __m128 s4 = _mm_mul_ps(hsum4_256_ps(a0, a1, a2, a3), vscale128);
            _mm_storeu_ps(scores0 + t, s4);
            vmax = _mm_max_ps(vmax, s4);
        }
    }
    vmax = _mm_max_ps(vmax, _mm_shuffle_ps(vmax, vmax, 0x4E));
    vmax = _mm_max_ps(vmax, _mm_shuffle_ps(vmax, vmax, 0xB1));
    float max_s = _mm_cvtss_f32(vmax);
    for (; t < kv_len; t++) {
        float s = dot_f32_avx2(hd, q0, kc_head + (size_t)t * (size_t)hd) * attn_scale;
        scores0[t] = s;
        if (s > max_s) max_s = s;
    }

    /* Softmax */
    {
        __m256 vm = _mm256_set1_ps(max_s), vs = _mm256_setzero_ps();
        int32_t k = 0;
        for (; k + 7 < kv_len; k += 8) {
            __m256 e = eng_avx2_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(scores0 + k), vm));
            _mm256_storeu_ps(scores0 + k, e);
            vs = _mm256_add_ps(vs, e);
        }
        float sp = eng_hsum256_ps(vs);
        for (; k < kv_len; k++) { float e = expf(scores0[k] - max_s); scores0[k] = e; sp += e; }
        float inv = 1.0f / (sp + 1e-12f);
        __m256 vi = _mm256_set1_ps(inv);
        for (k = 0; k + 7 < kv_len; k += 8) {
            _mm256_storeu_ps(scores0 + k, _mm256_mul_ps(_mm256_loadu_ps(scores0 + k), vi));
        }
        for (; k < kv_len; k++) scores0[k] *= inv;
    }

    if ((hd & 63) == 0) {
        for (int32_t d0 = 0; d0 < hd; d0 += 64) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
            __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();
            int32_t k = 0;
            for (; k + 1 < kv_len; k += 2) {
                const float* vp0 = vc_head + (size_t)k * (size_t)hd + d0;
                const float* vp1 = vp0 + hd;
                __m256 p0 = _mm256_set1_ps(scores0[k]), p1 = _mm256_set1_ps(scores0[k + 1]);
                a0 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp0 + 0),  _mm256_fmadd_ps(p1, _mm256_loadu_ps(vp1 + 0),  a0));
                a1 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp0 + 8),  _mm256_fmadd_ps(p1, _mm256_loadu_ps(vp1 + 8),  a1));
                a2 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp0 + 16), _mm256_fmadd_ps(p1, _mm256_loadu_ps(vp1 + 16), a2));
                a3 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp0 + 24), _mm256_fmadd_ps(p1, _mm256_loadu_ps(vp1 + 24), a3));
                a4 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp0 + 32), _mm256_fmadd_ps(p1, _mm256_loadu_ps(vp1 + 32), a4));
                a5 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp0 + 40), _mm256_fmadd_ps(p1, _mm256_loadu_ps(vp1 + 40), a5));
                a6 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp0 + 48), _mm256_fmadd_ps(p1, _mm256_loadu_ps(vp1 + 48), a6));
                a7 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp0 + 56), _mm256_fmadd_ps(p1, _mm256_loadu_ps(vp1 + 56), a7));
            }
            if (k < kv_len) {
                const float* vp = vc_head + (size_t)k * (size_t)hd + d0;
                __m256 p0 = _mm256_set1_ps(scores0[k]);
                a0 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp + 0),  a0);
                a1 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp + 8),  a1);
                a2 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp + 16), a2);
                a3 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp + 24), a3);
                a4 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp + 32), a4);
                a5 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp + 40), a5);
                a6 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp + 48), a6);
                a7 = _mm256_fmadd_ps(p0, _mm256_loadu_ps(vp + 56), a7);
            }
            _mm256_storeu_ps(out0 + d0 + 0,  a0); _mm256_storeu_ps(out0 + d0 + 8,  a1);
            _mm256_storeu_ps(out0 + d0 + 16, a2); _mm256_storeu_ps(out0 + d0 + 24, a3);
            _mm256_storeu_ps(out0 + d0 + 32, a4); _mm256_storeu_ps(out0 + d0 + 40, a5);
            _mm256_storeu_ps(out0 + d0 + 48, a6); _mm256_storeu_ps(out0 + d0 + 56, a7);
        }
    } else {
        memset(out0, 0, (size_t)hd * sizeof(float));
        for (int32_t k = 0; k < kv_len; k++) {
            const float* vp = vc_head + (size_t)k * (size_t)hd;
            __m256 va = _mm256_set1_ps(scores0[k]);
            int32_t i = 0;
            for (; i + 7 < hd; i += 8) _mm256_storeu_ps(out0 + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(vp + i), _mm256_loadu_ps(out0 + i)));
            for (; i < hd; i++) out0[i] += scores0[k] * vp[i];
        }
    }
}

static __forceinline void vec_softmax_avx2(int32_t n, float* scores, float max_s) {
    __m256 vmax = _mm256_set1_ps(max_s);
    __m256 vsum = _mm256_setzero_ps();
    int32_t t = 0;
    for (; t + 7 < n; t += 8) {
        __m256 e = eng_avx2_exp_ps(_mm256_sub_ps(_mm256_loadu_ps(scores + t), vmax));
        _mm256_storeu_ps(scores + t, e);
        vsum = _mm256_add_ps(vsum, e);
    }
    float sum_p = eng_hsum256_ps(vsum);
    for (; t < n; t++) {
        float e = expf(scores[t] - max_s);
        scores[t] = e;
        sum_p += e;
    }
    float inv_sum = 1.0f / (sum_p + 1e-12f);
    __m256 vinv = _mm256_set1_ps(inv_sum);
    t = 0;
    for (; t + 7 < n; t += 8) {
        _mm256_storeu_ps(scores + t, _mm256_mul_ps(_mm256_loadu_ps(scores + t), vinv));
    }
    for (; t < n; t++) scores[t] *= inv_sum;
}

static __forceinline float dot_f32_q8_blk32_avx2(int32_t n, const float* q, const int8_t* kq, const float* ks) {
    int32_t n_blk = n >> 5;
    __m256 total_v = _mm256_setzero_ps();
    for (int32_t b = 0; b < n_blk; b++) {
        int32_t c0 = b << 5;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m128i k0 = _mm_loadu_si128((const __m128i*)(kq + c0));
        __m128i k1 = _mm_loadu_si128((const __m128i*)(kq + c0 + 16));
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + c0),      _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(k0)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + c0 + 8),  _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(k0, 8))), acc1);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + c0 + 16), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(k1)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + c0 + 24), _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(k1, 8))), acc1);
        total_v = _mm256_fmadd_ps(_mm256_add_ps(acc0, acc1), _mm256_set1_ps(ks[b]), total_v);
    }
    return eng_hsum256_ps(total_v);
}

static void gqa_attn_head_pair_q8(
    int32_t hd,
    int32_t kv_len,
    float attn_scale,
    const int8_t* kc_q8,
    const float* ks_head,
    const int8_t* vc_q8,
    const float* vs_head,
    const float* q0,
    const float* q1,
    float* scores0,
    float* scores1,
    float* out0,
    float* out1
) {
    const int32_t hd_blk = hd >> 5;
    if (q1) {
        __m128 vmax0 = _mm_set1_ps(-1e30f), vmax1 = _mm_set1_ps(-1e30f);
        __m128 vscale128 = _mm_set1_ps(attn_scale);
        int32_t t = 0;
        for (; t + 1 < kv_len; t += 2) {
            const int8_t* k0p = kc_q8 + (size_t)t * (size_t)hd;
            const int8_t* k1p = k0p + hd;
            const float* ks0p = ks_head + (size_t)t * (size_t)hd_blk;
            const float* ks1p = ks0p + hd_blk;
            __m256 sa0 = _mm256_setzero_ps(), sa1 = _mm256_setzero_ps();
            __m256 sb0 = _mm256_setzero_ps(), sb1 = _mm256_setzero_ps();
            for (int32_t b = 0; b < hd_blk; b++) {
                int32_t c0 = b << 5;
                __m256 vks0 = _mm256_set1_ps(ks0p[b]), vks1 = _mm256_set1_ps(ks1p[b]);
                __m256 t0a = _mm256_setzero_ps(), t0b = _mm256_setzero_ps();
                __m256 t1a = _mm256_setzero_ps(), t1b = _mm256_setzero_ps();
                for (int32_t j = 0; j < 32; j += 16) {
                    __m256 q0l = _mm256_loadu_ps(q0 + c0 + j);
                    __m256 q0h = _mm256_loadu_ps(q0 + c0 + j + 8);
                    __m256 q1l = _mm256_loadu_ps(q1 + c0 + j);
                    __m256 q1h = _mm256_loadu_ps(q1 + c0 + j + 8);
                    __m128i m0 = _mm_loadu_si128((const __m128i*)(k0p + c0 + j));
                    __m256 f0l = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(m0));
                    __m256 f0h = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(m0, 8)));
                    t0a = _mm256_fmadd_ps(q0l, f0l, _mm256_fmadd_ps(q0h, f0h, t0a));
                    t0b = _mm256_fmadd_ps(q1l, f0l, _mm256_fmadd_ps(q1h, f0h, t0b));
                    __m128i m1 = _mm_loadu_si128((const __m128i*)(k1p + c0 + j));
                    __m256 f1l = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(m1));
                    __m256 f1h = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(m1, 8)));
                    t1a = _mm256_fmadd_ps(q0l, f1l, _mm256_fmadd_ps(q0h, f1h, t1a));
                    t1b = _mm256_fmadd_ps(q1l, f1l, _mm256_fmadd_ps(q1h, f1h, t1b));
                }
                sa0 = _mm256_fmadd_ps(t0a, vks0, sa0); sb0 = _mm256_fmadd_ps(t0b, vks0, sb0);
                sa1 = _mm256_fmadd_ps(t1a, vks1, sa1); sb1 = _mm256_fmadd_ps(t1b, vks1, sb1);
            }
            __m128 s = _mm_mul_ps(hsum4_256_ps(sa0, sa1, sb0, sb1), vscale128);
            float d0 = _mm_cvtss_f32(s);
            float d1 = _mm_cvtss_f32(_mm_shuffle_ps(s, s, 0x55));
            float d2 = _mm_cvtss_f32(_mm_shuffle_ps(s, s, 0xAA));
            float d3 = _mm_cvtss_f32(_mm_shuffle_ps(s, s, 0xFF));
            scores0[t] = d0; scores0[t + 1] = d1;
            scores1[t] = d2; scores1[t + 1] = d3;
            vmax0 = _mm_max_ps(vmax0, _mm_set1_ps((d0 > d1) ? d0 : d1));
            vmax1 = _mm_max_ps(vmax1, _mm_set1_ps((d2 > d3) ? d2 : d3));
        }
        float max0 = _mm_cvtss_f32(vmax0), max1 = _mm_cvtss_f32(vmax1);
        for (; t < kv_len; t++) {
            float s0 = dot_f32_q8_blk32_avx2(hd, q0, kc_q8 + (size_t)t * (size_t)hd, ks_head + (size_t)t * (size_t)hd_blk) * attn_scale;
            float s1 = dot_f32_q8_blk32_avx2(hd, q1, kc_q8 + (size_t)t * (size_t)hd, ks_head + (size_t)t * (size_t)hd_blk) * attn_scale;
            scores0[t] = s0; if (s0 > max0) max0 = s0;
            scores1[t] = s1; if (s1 > max1) max1 = s1;
        }
        vec_softmax_avx2(kv_len, scores0, max0);
        vec_softmax_avx2(kv_len, scores1, max1);

        /* Register-resident blk32 V accumulation sharing Q8->FP32 unpack across q0 & q1 */
        for (int32_t b = 0; b < hd_blk; b++) {
            int32_t c0 = b << 5;
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
            __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
            for (int32_t k = 0; k < kv_len; k++) {
                float vs = vs_head[(size_t)k * (size_t)hd_blk + (size_t)b];
                __m256 wa = _mm256_set1_ps(scores0[k] * vs);
                __m256 wb = _mm256_set1_ps(scores1[k] * vs);
                const int8_t* vp = vc_q8 + (size_t)k * (size_t)hd + c0;
                __m128i v_lo = _mm_loadu_si128((const __m128i*)vp);
                __m128i v_hi = _mm_loadu_si128((const __m128i*)(vp + 16));
                __m256 vf0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo));
                a0 = _mm256_fmadd_ps(wa, vf0, a0); b0 = _mm256_fmadd_ps(wb, vf0, b0);
                __m256 vf1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_lo, 8)));
                a1 = _mm256_fmadd_ps(wa, vf1, a1); b1 = _mm256_fmadd_ps(wb, vf1, b1);
                __m256 vf2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi));
                a2 = _mm256_fmadd_ps(wa, vf2, a2); b2 = _mm256_fmadd_ps(wb, vf2, b2);
                __m256 vf3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_hi, 8)));
                a3 = _mm256_fmadd_ps(wa, vf3, a3); b3 = _mm256_fmadd_ps(wb, vf3, b3);
            }
            _mm256_storeu_ps(out0 + c0,      a0); _mm256_storeu_ps(out1 + c0,      b0);
            _mm256_storeu_ps(out0 + c0 + 8,  a1); _mm256_storeu_ps(out1 + c0 + 8,  b1);
            _mm256_storeu_ps(out0 + c0 + 16, a2); _mm256_storeu_ps(out1 + c0 + 16, b2);
            _mm256_storeu_ps(out0 + c0 + 24, a3); _mm256_storeu_ps(out1 + c0 + 24, b3);
        }
        return;
    }

    float max_s = -1e30f;
    for (int32_t t = 0; t < kv_len; t++) {
        float s = dot_f32_q8_blk32_avx2(hd, q0, kc_q8 + (size_t)t * (size_t)hd, ks_head + (size_t)t * (size_t)hd_blk) * attn_scale;
        scores0[t] = s;
        if (s > max_s) max_s = s;
    }
    vec_softmax_avx2(kv_len, scores0, max_s);
    for (int32_t b = 0; b < hd_blk; b++) {
        int32_t c0 = b << 5;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        for (int32_t k = 0; k < kv_len; k++) {
            __m256 wa = _mm256_set1_ps(scores0[k] * vs_head[(size_t)k * (size_t)hd_blk + (size_t)b]);
            const int8_t* vp = vc_q8 + (size_t)k * (size_t)hd + c0;
            __m128i v_lo = _mm_loadu_si128((const __m128i*)vp);
            __m128i v_hi = _mm_loadu_si128((const __m128i*)(vp + 16));
            a0 = _mm256_fmadd_ps(wa, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_lo)), a0);
            a1 = _mm256_fmadd_ps(wa, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_lo, 8))), a1);
            a2 = _mm256_fmadd_ps(wa, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v_hi)), a2);
            a3 = _mm256_fmadd_ps(wa, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v_hi, 8))), a3);
        }
        _mm256_storeu_ps(out0 + c0,      a0);
        _mm256_storeu_ps(out0 + c0 + 8,  a1);
        _mm256_storeu_ps(out0 + c0 + 16, a2);
        _mm256_storeu_ps(out0 + c0 + 24, a3);
    }
}

static __forceinline void quantize_vec_q8_blk32(int32_t n, const float* src, int8_t* dst, float* out_scales) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256i perm_idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    const __m256i min_i8 = _mm256_set1_epi8(-127);
    int32_t n_blk = n >> 5;
    for (int32_t b = 0; b < n_blk; b++) {
        int32_t c0 = b << 5;
        __m256 v0 = _mm256_loadu_ps(src + c0);
        __m256 v1 = _mm256_loadu_ps(src + c0 + 8);
        __m256 v2 = _mm256_loadu_ps(src + c0 + 16);
        __m256 v3 = _mm256_loadu_ps(src + c0 + 24);
        __m256 mall = _mm256_max_ps(
            _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask)),
            _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask))
        );
        __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(mall), _mm256_extractf128_ps(mall, 1));
        m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
        m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
        float amax = _mm_cvtss_f32(m128);
        out_scales[b] = amax / 127.0f;
        float inv_d = (amax > 1e-12f) ? (127.0f / amax) : 0.0f;
        __m256 vinv = _mm256_set1_ps(inv_d);
        __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, vinv));
        __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, vinv));
        __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, vinv));
        __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, vinv));
        __m256i p8 = _mm256_packs_epi16(_mm256_packs_epi32(q0, q1), _mm256_packs_epi32(q2, q3));
        p8 = _mm256_max_epi8(_mm256_permutevar8x32_epi32(p8, perm_idx), min_i8);
        _mm256_storeu_si256((__m256i*)(dst + c0), p8);
    }
}

static __inline int dpuf_mode_uses_had_slab(int32_t m) {
    return (m == DPUF_MODE_QR_HAD4_PACKED || m == DPUF_MODE_QR_HAD4_ASYM || m == DPUF_MODE_QR_HAD6);
}

static void prepare_input_vec(
    const DpufWeightDesc* w,
    const float* x_m,
    float* f_m,
    int8_t* q_m,
    float* bs_m,
    int32_t* xsum_m
) {
    int32_t cols = w->cols;
    if (w->mode == DPUF_MODE_QR_ZD4) {
        const float* cs = w->col_scale;
        int32_t c = 0;
        for (; c + 7 < cols; c += 8) {
            _mm256_storeu_ps(f_m + c, _mm256_mul_ps(_mm256_loadu_ps(x_m + c), _mm256_loadu_ps(cs + c)));
        }
        for (; c < cols; c++) f_m[c] = x_m[c] * cs[c];
    } else if (w->mode == DPUF_MODE_QR_SPLIT4) {
        dpuf_scale_quantize_x_q8_32(cols, x_m, w->col_scale, f_m, q_m, bs_m);
    } else if (w->mode == DPUF_MODE_QR_HAD4) {
        dpuf_fwht64_sign_f32(cols, x_m, w->sign_bits, f_m);
        dpuf_scale_quantize_x_q8_32(cols, f_m, NULL, NULL, q_m, bs_m);
    } else if (w->mode == DPUF_MODE_QR_Q8) {
        dpuf_scale_quantize_x_q8_32(cols, x_m, NULL, NULL, q_m, bs_m);
    } else if (dpuf_mode_uses_had_slab(w->mode)) {
        dpuf_prep_had4_packed_slab(1, cols, x_m, w->sign_bits, f_m, q_m, bs_m, xsum_m, 0);
    }
}

static void prepare_input_m_or_parallel(
    const DpufWeightDesc* w,
    int32_t M,
    const float* x_mat,
    float* f_buf,
    int8_t* q_buf,
    float* bs_buf,
    int32_t* xsum_buf,
    int32_t tid,
    int32_t nt
) {
    int32_t cols = w->cols;
    if (dpuf_mode_uses_had_slab(w->mode) && M >= 2) {
        int32_t n_blk = cols >> 5;
        int32_t n_slabs = (M + 5) / 6;
        for (int32_t s = tid; s < n_slabs; s += nt) {
            int32_t m0 = s * 6;
            int32_t mc = (m0 + 6 <= M) ? 6 : (M - m0);
            dpuf_prep_had4_packed_slab(
                mc, cols,
                x_mat + (size_t)m0 * (size_t)cols,
                w->sign_bits,
                f_buf + (size_t)m0 * (size_t)cols,
                q_buf + (size_t)s * (size_t)n_blk * 192u,
                bs_buf + (size_t)s * (size_t)n_blk * 6u,
                xsum_buf + (size_t)s * (size_t)n_blk * 6u,
                1
            );
        }
        return;
    }

    if (M == 1 && (cols & 63) == 0 && nt > 1) {
        int32_t n64 = cols >> 6;
        int32_t c0 = ((n64 * tid) / nt) << 6;
        int32_t c1 = ((n64 * (tid + 1)) / nt) << 6;
        int32_t len = c1 - c0;
        if (len <= 0) return;
        if (w->mode == DPUF_MODE_QR_ZD4) {
            const float* cs = w->col_scale + c0;
            const float* xm = x_mat + c0;
            float* fm = f_buf + c0;
            for (int32_t c = 0; c < len; c += 8) {
                _mm256_storeu_ps(fm + c, _mm256_mul_ps(_mm256_loadu_ps(xm + c), _mm256_loadu_ps(cs + c)));
            }
        } else if (w->mode == DPUF_MODE_QR_SPLIT4) {
            dpuf_scale_quantize_x_q8_32(len, x_mat + c0, w->col_scale + c0, f_buf + c0, q_buf + c0, bs_buf + (c0 >> 5));
        } else if (w->mode == DPUF_MODE_QR_HAD4) {
            dpuf_fwht64_sign_f32(len, x_mat + c0, w->sign_bits + (c0 >> 3), f_buf + c0);
            dpuf_scale_quantize_x_q8_32(len, f_buf + c0, NULL, NULL, q_buf + c0, bs_buf + (c0 >> 5));
        } else if (w->mode == DPUF_MODE_QR_Q8) {
            dpuf_scale_quantize_x_q8_32(len, x_mat + c0, NULL, NULL, q_buf + c0, bs_buf + (c0 >> 5));
        } else if (dpuf_mode_uses_had_slab(w->mode)) {
            dpuf_prep_had4_packed_slab(
                1, len, x_mat + c0, w->sign_bits + (c0 >> 3), f_buf + c0,
                q_buf + c0, bs_buf + (c0 >> 5), xsum_buf + (c0 >> 5), 0
            );
        }
    } else {
        for (int32_t m = tid; m < M; m += nt) {
            prepare_input_vec(
                w,
                x_mat + (size_t)m * (size_t)cols,
                f_buf + (size_t)m * (size_t)cols,
                q_buf + (size_t)m * (size_t)cols,
                bs_buf + (size_t)m * (size_t)(cols >> 5),
                xsum_buf + (size_t)m * (size_t)(cols >> 5)
            );
        }
    }
}

static int weights_share_prep(const DpufWeightDesc* a, const DpufWeightDesc* b) {
    if (a->cols != b->cols) return 0;
    if (a->mode == DPUF_MODE_QR_Q8 && b->mode == DPUF_MODE_QR_Q8) return 1;
    if (((dpuf_mode_uses_had_slab(a->mode) && dpuf_mode_uses_had_slab(b->mode)) ||
         (a->mode == DPUF_MODE_QR_HAD4 && b->mode == DPUF_MODE_QR_HAD4)) &&
        a->sign_bits && b->sign_bits) {
        size_t sb_len = (size_t)(a->cols >> 3);
        return memcmp(a->sign_bits, b->sign_bits, sb_len) == 0;
    }
    return 0;
}

static void prepare_multi_weights_group(
    int32_t n_items,
    const DpufWeightDesc* const* ws,
    const int32_t* slots,
    int32_t M,
    const float* x_mat,
    DpufEngine* eng,
    int32_t tid,
    int32_t nt
) {
    if (n_items <= 0) return;
    if (n_items == 1) {
        int32_t slot = slots[0];
        prepare_input_m_or_parallel(
            ws[0], M, x_mat,
            eng->prep_f32[slot], eng->prep_q8[slot],
            eng->prep_scale[slot], eng->prep_xsum[slot],
            tid, nt
        );
        return;
    }

    int32_t is_slab[4] = {0};
    int32_t cnt[4] = {0};
    int32_t total_jobs = 0;
    const int32_t n_slabs = (M + 5) / 6;
    for (int32_t k = 0; k < n_items; k++) {
        is_slab[k] = (dpuf_mode_uses_had_slab(ws[k]->mode) && M >= 2) ? 1 : 0;
        cnt[k] = is_slab[k] ? n_slabs : M;
        total_jobs += cnt[k];
    }

    for (int32_t job = tid; job < total_jobs; job += nt) {
        int32_t rem = job;
        int32_t k = 0;
        while (k + 1 < n_items && rem >= cnt[k]) {
            rem -= cnt[k];
            k++;
        }
        const DpufWeightDesc* w = ws[k];
        const int32_t slot = slots[k];
        if (is_slab[k]) {
            int32_t m0 = rem * 6;
            int32_t mc = (m0 + 6 <= M) ? 6 : (M - m0);
            int32_t n_blk = w->cols >> 5;
            dpuf_prep_had4_packed_slab(
                mc, w->cols,
                x_mat + (size_t)m0 * (size_t)w->cols,
                w->sign_bits,
                eng->prep_f32[slot] + (size_t)m0 * (size_t)w->cols,
                eng->prep_q8[slot] + (size_t)rem * (size_t)n_blk * 192u,
                eng->prep_scale[slot] + (size_t)rem * (size_t)n_blk * 6u,
                eng->prep_xsum[slot] + (size_t)rem * (size_t)n_blk * 6u,
                1
            );
        } else {
            prepare_input_vec(
                w,
                x_mat + (size_t)rem * (size_t)w->cols,
                eng->prep_f32[slot] + (size_t)rem * (size_t)w->cols,
                eng->prep_q8[slot]  + (size_t)rem * (size_t)w->cols,
                eng->prep_scale[slot] + (size_t)rem * (size_t)(w->cols >> 5),
                eng->prep_xsum[slot]  + (size_t)rem * (size_t)(w->cols >> 5)
            );
        }
    }
}

static __forceinline void swiglu_range_avx2(
    size_t s0, size_t s1, const float* gate, const float* up, float* out, float deadzone
) {
    size_t idx = s0;
    const __m256 vc_neg40 = _mm256_set1_ps(-40.0f);
    const __m256 vc_pos40 = _mm256_set1_ps(40.0f);
    const __m256 vc_one   = _mm256_set1_ps(1.0f);
    const __m256 v_dz     = _mm256_set1_ps(deadzone);
    const int do_dz       = (deadzone > 0.0f);

    for (; idx + 7 < s1; idx += 8) {
        __m256 vg = _mm256_loadu_ps(gate + idx);
        __m256 vu = _mm256_loadu_ps(up + idx);
        __m256 neg_g = _mm256_max_ps(vc_neg40, _mm256_min_ps(_mm256_sub_ps(_mm256_setzero_ps(), vg), vc_pos40));
        __m256 silu = _mm256_div_ps(vg, _mm256_add_ps(vc_one, eng_avx2_exp_ps(neg_g)));
        __m256 res = _mm256_mul_ps(silu, vu);
        if (do_dz) {
            __m256 abs_res = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), res);
            __m256 mask = _mm256_cmp_ps(abs_res, v_dz, _CMP_GE_OQ);
            res = _mm256_and_ps(res, mask);
        }
        _mm256_storeu_ps(out + idx, res);
    }
    for (; idx < s1; idx++) {
        float g = gate[idx];
        float silu = (g >= 40.0f) ? g : ((g <= -40.0f) ? 0.0f : (g / (1.0f + expf(-g))));
        float val = silu * up[idx];
        if (do_dz && fabsf(val) < deadzone) val = 0.0f;
        out[idx] = val;
    }
}

static void fused_res_norm_prep_had4p_slab(
    const DpufWeightDesc* w,
    int32_t M,
    float* h_buf,
    const float* res_src,
    const float* norm_w,
    float eps,
    float* norm_out,
    float* f_buf,
    int8_t* q_buf,
    float* bs_buf,
    int32_t* xsum_buf,
    int32_t tid,
    int32_t nt
) {
    int32_t cols = w->cols;
    int32_t n_blk = cols >> 5;
    int32_t n_slabs = (M + 5) / 6;
    for (int32_t s = tid; s < n_slabs; s += nt) {
        int32_t m0 = s * 6;
        int32_t mc = (m0 + 6 <= M) ? 6 : (M - m0);
        for (int32_t k = 0; k < mc; k++) {
            int32_t m = m0 + k;
            add_and_rms_norm_f32(
                cols,
                h_buf + (size_t)m * (size_t)cols,
                res_src ? (res_src + (size_t)m * (size_t)cols) : NULL,
                norm_w,
                eps,
                norm_out + (size_t)m * (size_t)cols
            );
        }
        dpuf_prep_had4_packed_slab(
            mc, cols,
            norm_out + (size_t)m0 * (size_t)cols,
            w->sign_bits,
            f_buf + (size_t)m0 * (size_t)cols,
            q_buf + (size_t)s * (size_t)n_blk * 192u,
            bs_buf + (size_t)s * (size_t)n_blk * 6u,
            xsum_buf + (size_t)s * (size_t)n_blk * 6u,
            1
        );
    }
}

static void run_weight_tile_range(
    const DpufWeightDesc* w,
    int32_t M,
    const DpufEngine* eng,
    int32_t slot,
    int32_t t0,
    int32_t t1,
    float* out
) {
    if (t0 >= t1) return;

    if (w->mode == DPUF_MODE_QR_ZD4) {
        dpuf_matmul_qr_zd4_tile_range(
            M, w->rows, w->cols, t0, t1, eng->prep_f32[slot], w->tensor_base, w->tile_offsets, out
        );
    } else if (w->mode == DPUF_MODE_QR_SPLIT4) {
        dpuf_matmul_qr_split4_tile_range(
            M, w->rows, w->cols, t0, t1, eng->prep_f32[slot], eng->prep_q8[slot], eng->prep_scale[slot],
            w->tensor_base, w->tile_offsets, out
        );
    } else if (w->mode == DPUF_MODE_QR_HAD4) {
        dpuf_matmul_qr_hadamard4_tile_range(
            M, w->rows, w->cols, t0, t1, eng->prep_f32[slot], eng->prep_q8[slot], eng->prep_scale[slot],
            w->tensor_base, w->tile_offsets, out
        );
    } else if (w->mode == DPUF_MODE_QR_Q8) {
        dpuf_matmul_qr_q8_tile_range(
            M, w->rows, w->cols, t0, t1, eng->prep_q8[slot], eng->prep_scale[slot],
            w->tensor_base, w->tile_offsets, out
        );
    } else if (w->mode == DPUF_MODE_QR_HAD4_PACKED) {
        dpuf_matmul_qr_hadamard4_packed_tile_range(
            M, w->rows, w->cols, t0, t1, eng->prep_q8[slot], eng->prep_scale[slot], eng->prep_xsum[slot],
            w->tensor_base, w->tile_offsets, out
        );
    } else if (w->mode == DPUF_MODE_QR_HAD4_ASYM) {
        dpuf_matmul_qr_had4_asym_tile_range(
            M, w->rows, w->cols, t0, t1, eng->prep_q8[slot], eng->prep_scale[slot], eng->prep_xsum[slot],
            w->tensor_base, w->tile_offsets, out
        );
    } else if (w->mode == DPUF_MODE_QR_HAD6) {
        dpuf_matmul_qr_had6_tile_range(
            M, w->rows, w->cols, t0, t1, eng->prep_q8[slot], eng->prep_scale[slot], eng->prep_xsum[slot],
            w->tensor_base, w->tile_offsets, out
        );
    }
}

static void run_weight_tile_slice(
    const DpufWeightDesc* w,
    int32_t M,
    const DpufEngine* eng,
    int32_t slot,
    float* out,
    int32_t tid,
    int32_t nt
) {
    int32_t t0 = (w->n_tiles * tid) / nt;
    int32_t t1 = (w->n_tiles * (tid + 1)) / nt;
    run_weight_tile_range(w, M, eng, slot, t0, t1, out);
}

static void run_weight_dynamic(
    const DpufWeightDesc* w,
    int32_t M,
    const DpufEngine* eng,
    int32_t slot,
    float* out,
    volatile long* ctr,
    int32_t chunk,
    int32_t tid,
    int32_t nt
) {
#ifdef _WIN32
    if (nt <= 1 || M < 4 || w->n_tiles < nt * 2) {
        run_weight_tile_slice(w, M, eng, slot, out, tid, nt);
        return;
    }
    const int32_t total = w->n_tiles;
    while (1) {
        int32_t t0 = (int32_t)_InterlockedExchangeAdd(ctr, (long)chunk);
        if (t0 >= total) break;
        int32_t t1 = t0 + chunk;
        if (t1 > total) t1 = total;
        run_weight_tile_range(w, M, eng, slot, t0, t1, out);
    }
#else
    (void)ctr; (void)chunk;
    run_weight_tile_slice(w, M, eng, slot, out, tid, nt);
#endif
}

static void run_multi_weights_dynamic(
    int32_t nw,
    const DpufWeightDesc* const* ws,
    const int32_t* slots,
    float* const* outs,
    int32_t M,
    const DpufEngine* eng,
    volatile long* ctr,
    int32_t chunk,
    int32_t tid,
    int32_t nt
) {
#ifdef _WIN32
    if (nt <= 1 || M < 4) {
        for (int32_t k = 0; k < nw; k++) {
            run_weight_tile_slice(ws[k], M, eng, slots[k], outs[k], tid, nt);
        }
        return;
    }
    int32_t total = 0;
    for (int32_t k = 0; k < nw; k++) total += ws[k]->n_tiles;
    while (1) {
        int32_t g0 = (int32_t)_InterlockedExchangeAdd(ctr, (long)chunk);
        if (g0 >= total) break;
        int32_t g1 = g0 + chunk;
        if (g1 > total) g1 = total;
        int32_t base = 0;
        for (int32_t k = 0; k < nw; k++) {
            int32_t ntk = ws[k]->n_tiles;
            int32_t end = base + ntk;
            int32_t a = (g0 > base) ? g0 : base;
            int32_t b = (g1 < end)  ? g1 : end;
            if (a < b) {
                run_weight_tile_range(ws[k], M, eng, slots[k], a - base, b - base, outs[k]);
            }
            base = end;
        }
    }
#else
    (void)ctr; (void)chunk;
    for (int32_t k = 0; k < nw; k++) {
        run_weight_tile_slice(ws[k], M, eng, slots[k], outs[k], tid, nt);
    }
#endif
}

DPUF_EXPORT void* dpuf_engine_create(
    const DpufModelConfig* cfg,
    const DpufLayerWeights* layers,
    const DpufWeightDesc* w_tok_embd,
    const DpufWeightDesc* w_lm_head,
    const float* output_norm
) {
    if (!cfg || !layers || !w_tok_embd || !w_lm_head || !output_norm) return NULL;
    DpufEngine* eng = (DpufEngine*)calloc(1, sizeof(DpufEngine));
    if (!eng) return NULL;

    eng->cfg = *cfg;
    if (eng->cfg.n_threads <= 0) eng->cfg.n_threads = 8;
    if (eng->cfg.n_threads > 32) eng->cfg.n_threads = 32;
    int32_t max_th = omp_get_max_threads();
    if (eng->cfg.n_threads > max_th && max_th > 0) eng->cfg.n_threads = max_th;

    eng->layers = (DpufLayerWeights*)malloc((size_t)cfg->num_layers * sizeof(DpufLayerWeights));
    if (!eng->layers) { free(eng); return NULL; }
    memcpy(eng->layers, layers, (size_t)cfg->num_layers * sizeof(DpufLayerWeights));
    eng->w_tok_embd = *w_tok_embd;
    eng->w_lm_head = *w_lm_head;
    eng->output_norm = output_norm;
    const char* env_dz = getenv("DPUF_SWIGLU_DEADZONE");
    eng->swiglu_deadzone = env_dz ? (float)atof(env_dz) : 0.005f;

    int32_t h = cfg->hidden_size;
    int32_t inter = cfg->intermediate_size;
    int32_t q_dim = cfg->num_q_heads * cfg->head_dim;
    int32_t kv_dim = cfg->num_kv_heads * cfg->head_dim;
    int32_t max_seq = cfg->max_seq_len;
    int32_t B = DPUF_ENGINE_MAX_BATCH;
    int32_t B_pad = ((B + 5) / 6) * 6;
    int32_t max_dim = h;
    if (inter > max_dim) max_dim = inter;
    if (q_dim > max_dim) max_dim = q_dim;

    size_t kv_elems = (size_t)cfg->num_layers * (size_t)max_seq * (size_t)kv_dim;
    size_t kv_scale_elems = (size_t)cfg->num_layers * (size_t)max_seq * (size_t)(kv_dim >> 5);
    eng->cos_tab = (float*)malloc((size_t)max_seq * (size_t)(cfg->head_dim >> 1) * sizeof(float));
    eng->sin_tab = (float*)malloc((size_t)max_seq * (size_t)(cfg->head_dim >> 1) * sizeof(float));

    int kv_ok = 1;
    if (cfg->kv_type == DPUF_KV_Q8) {
        eng->k_cache_q8 = (int8_t*)calloc(kv_elems, sizeof(int8_t));
        eng->v_cache_q8 = (int8_t*)calloc(kv_elems, sizeof(int8_t));
        eng->k_scale_q8 = (float*)calloc(kv_scale_elems, sizeof(float));
        eng->v_scale_q8 = (float*)calloc(kv_scale_elems, sizeof(float));
        kv_ok = eng->k_cache_q8 && eng->v_cache_q8 && eng->k_scale_q8 && eng->v_scale_q8;
    } else {
        eng->k_cache = (float*)calloc(kv_elems, sizeof(float));
        eng->v_cache = (float*)calloc(kv_elems, sizeof(float));
        kv_ok = eng->k_cache && eng->v_cache;
    }

    eng->h_buf      = (float*)malloc((size_t)B * (size_t)h * sizeof(float));
    eng->hn_buf     = (float*)malloc((size_t)B * (size_t)h * sizeof(float));
    eng->q_buf      = (float*)malloc((size_t)B * (size_t)q_dim * sizeof(float));
    eng->k_buf      = (float*)malloc((size_t)B * (size_t)kv_dim * sizeof(float));
    eng->v_buf      = (float*)malloc((size_t)B * (size_t)kv_dim * sizeof(float));
    int32_t attn_dim = (q_dim > h) ? q_dim : h;
    eng->attn_out   = (float*)malloc((size_t)B * (size_t)attn_dim * sizeof(float));
    eng->gate_buf   = (float*)malloc((size_t)B * (size_t)inter * sizeof(float));
    eng->up_buf     = (float*)malloc((size_t)B * (size_t)inter * sizeof(float));
    eng->ffn_hidden = (float*)malloc((size_t)B * (size_t)inter * sizeof(float));
    eng->attn_scores = (float*)malloc(2u * (size_t)eng->cfg.n_threads * (size_t)max_seq * sizeof(float));
    eng->layer_share_kv = (uint8_t*)calloc((size_t)cfg->num_layers, 1);
    eng->layer_share_gu = (uint8_t*)calloc((size_t)cfg->num_layers, 1);
    eng->work_ctr_bytes = ((size_t)cfg->num_layers * 8u + 4u) * 16u * sizeof(long);
    eng->work_ctr = (volatile long*)calloc(1, eng->work_ctr_bytes);

    int ok = eng->cos_tab && eng->sin_tab && kv_ok &&
             eng->h_buf && eng->hn_buf && eng->q_buf && eng->k_buf && eng->v_buf &&
             eng->attn_out && eng->gate_buf && eng->up_buf && eng->ffn_hidden &&
             eng->attn_scores && eng->layer_share_kv && eng->layer_share_gu && eng->work_ctr;

    for (int k = 0; k < 3; k++) {
        eng->prep_f32[k]   = (float*)malloc((size_t)B_pad * (size_t)max_dim * sizeof(float));
        eng->prep_q8[k]    = (int8_t*)malloc((size_t)B_pad * (size_t)max_dim * sizeof(int8_t));
        eng->prep_scale[k] = (float*)malloc((size_t)B_pad * (size_t)(max_dim >> 5) * sizeof(float));
        eng->prep_xsum[k]  = (int32_t*)malloc((size_t)B_pad * (size_t)(max_dim >> 5) * sizeof(int32_t));
        if (!eng->prep_f32[k] || !eng->prep_q8[k] || !eng->prep_scale[k] || !eng->prep_xsum[k]) ok = 0;
    }

    if (!ok) {
        free(eng->cos_tab); free(eng->sin_tab);
        free(eng->k_cache); free(eng->v_cache);
        free(eng->k_cache_q8); free(eng->v_cache_q8);
        free(eng->k_scale_q8); free(eng->v_scale_q8);
        free(eng->h_buf); free(eng->hn_buf); free(eng->q_buf); free(eng->k_buf); free(eng->v_buf);
        free(eng->attn_out); free(eng->gate_buf); free(eng->up_buf); free(eng->ffn_hidden);
        free(eng->attn_scores); free(eng->layer_share_kv); free(eng->layer_share_gu);
        free((void*)eng->work_ctr);
        for (int k = 0; k < 3; k++) {
            free(eng->prep_f32[k]); free(eng->prep_q8[k]); free(eng->prep_scale[k]); free(eng->prep_xsum[k]);
        }
        free(eng->layers);
        free(eng);
        return NULL;
    }

    for (int32_t l = 0; l < cfg->num_layers; l++) {
        const DpufLayerWeights* lw = &layers[l];
        int share_qk = weights_share_prep(&lw->w_q, &lw->w_k);
        int share_qv = weights_share_prep(&lw->w_q, &lw->w_v);
        eng->layer_share_kv[l] = (uint8_t)((share_qk ? 1 : 0) | ((share_qk && share_qv) ? 2 : 0));
        eng->layer_share_gu[l] = (uint8_t)weights_share_prep(&lw->w_gate, &lw->w_up);
    }

    precompute_rope(eng);
#ifdef _WIN32
    eng->n_phys_cores = 0;
    {
        DWORD len = 0;
        DWORD_PTR proc_mask = 0, sys_mask = 0;
        if (GetProcessAffinityMask(GetCurrentProcess(), &proc_mask, &sys_mask) && proc_mask) {
            GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
            if (len > 0) {
                uint8_t* info_buf = (uint8_t*)malloc(len);
                if (info_buf && GetLogicalProcessorInformationEx(
                        RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)info_buf, &len)) {
                    DWORD off = 0;
                    while (off < len && eng->n_phys_cores < 32) {
                        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(info_buf + off);
                        if (p->Relationship == RelationProcessorCore && p->Processor.GroupCount > 0) {
                            if (p->Processor.GroupMask[0].Group == 0) {
                                KAFFINITY m = p->Processor.GroupMask[0].Mask & proc_mask;
                                if (m != 0) eng->core_masks[eng->n_phys_cores++] = (uint64_t)m;
                            }
                        }
                        if (p->Size == 0) break;
                        off += p->Size;
                    }
                }
                free(info_buf);
            }
        }
    }
#endif
    eng->bar.arrived = 0;
    eng->bar.sense = 0;
    return eng;
}

DPUF_EXPORT void dpuf_engine_destroy(void* handle) {
    DpufEngine* eng = (DpufEngine*)handle;
    if (!eng) return;
    free((void*)eng->work_ctr);
    free(eng->cos_tab);
    free(eng->sin_tab);
    free(eng->k_cache);
    free(eng->v_cache);
    free(eng->k_cache_q8);
    free(eng->v_cache_q8);
    free(eng->k_scale_q8);
    free(eng->v_scale_q8);
    free(eng->h_buf);
    free(eng->hn_buf);
    free(eng->q_buf);
    free(eng->k_buf);
    free(eng->v_buf);
    free(eng->attn_out);
    free(eng->gate_buf);
    free(eng->up_buf);
    free(eng->ffn_hidden);
    free(eng->attn_scores);
    free(eng->layer_share_kv);
    free(eng->layer_share_gu);
    for (int k = 0; k < 3; k++) {
        free(eng->prep_f32[k]);
        free(eng->prep_q8[k]);
        free(eng->prep_scale[k]);
        free(eng->prep_xsum[k]);
    }
    free(eng->layers);
    free(eng);
}

DPUF_EXPORT void dpuf_engine_reset_cache(void* handle) {
    DpufEngine* eng = (DpufEngine*)handle;
    if (!eng) return;
    size_t kv_elems = (size_t)eng->cfg.num_layers * (size_t)eng->cfg.max_seq_len *
                      (size_t)(eng->cfg.num_kv_heads * eng->cfg.head_dim);
    if (eng->cfg.kv_type == DPUF_KV_Q8) {
        size_t kv_scale_elems = (size_t)eng->cfg.num_layers * (size_t)eng->cfg.max_seq_len *
                                (size_t)((eng->cfg.num_kv_heads * eng->cfg.head_dim) >> 5);
        if (eng->k_cache_q8) memset(eng->k_cache_q8, 0, kv_elems);
        if (eng->v_cache_q8) memset(eng->v_cache_q8, 0, kv_elems);
        if (eng->k_scale_q8) memset(eng->k_scale_q8, 0, kv_scale_elems * sizeof(float));
        if (eng->v_scale_q8) memset(eng->v_scale_q8, 0, kv_scale_elems * sizeof(float));
    } else {
        if (eng->k_cache) memset(eng->k_cache, 0, kv_elems * sizeof(float));
        if (eng->v_cache) memset(eng->v_cache, 0, kv_elems * sizeof(float));
    }
}

static void eng_compute_mrope_cos_sin(
    int32_t hd, float theta, int32_t pt, int32_t ph, int32_t pw,
    const int32_t* mrope_sec, float* out_cos, float* out_sin
) {
    int32_t half = hd >> 1;
    int32_t sec_h = mrope_sec ? mrope_sec[1] : 20;
    int32_t len_hw = sec_h * 3;

    for (int32_t j = 0; j < half; j++) {
        float freq = 1.0f / powf(theta, (2.0f * (float)j) / (float)hd);
        float pos_val = (float)pt;
        if (j < len_hw) {
            int32_t rem = j % 3;
            if (rem == 1) pos_val = (float)ph;
            else if (rem == 2) pos_val = (float)pw;
            else pos_val = (float)pt;
        }
        float ang = pos_val * freq;
        out_cos[j] = cosf(ang);
        out_sin[j] = sinf(ang);
    }
}

DPUF_EXPORT int32_t dpuf_engine_forward_batch_ex(
    void* handle,
    int32_t n_tokens,
    const int32_t* token_ids,
    int32_t pos0,
    const float* embeds_in,
    const int32_t* pos_t,
    const int32_t* pos_h,
    const int32_t* pos_w,
    const float* const* deepstack_in,
    int32_t only_last_logits,
    float* logits_out
) {
    DpufEngine* eng = (DpufEngine*)handle;
    if (!eng || (!token_ids && !embeds_in) || (!logits_out && only_last_logits != 2)) return -1;
    if (n_tokens <= 0 || n_tokens > DPUF_ENGINE_MAX_BATCH) return -2;
    int32_t effective_pos0 = pos_t ? pos_t[0] : pos0;
    if (effective_pos0 < 0 || effective_pos0 + n_tokens > eng->cfg.max_seq_len) return -3;

    const int32_t M = n_tokens;
    const int32_t h_dim = eng->cfg.hidden_size;
    const int32_t inter = eng->cfg.intermediate_size;
    const int32_t n_layers = eng->cfg.num_layers;
    const int32_t n_q = eng->cfg.num_q_heads;
    const int32_t n_kv = eng->cfg.num_kv_heads;
    const int32_t hd = eng->cfg.head_dim;
    const int32_t q_dim = n_q * hd;
    const int32_t kv_dim = n_kv * hd;
    const int32_t groups = n_q / n_kv;
    const int32_t half_hd = hd >> 1;
    const float eps = eng->cfg.rms_eps;
    const float attn_scale = 1.0f / sqrtf((float)hd);
    const size_t layer_kv_stride = (size_t)eng->cfg.max_seq_len * (size_t)kv_dim;

    if (embeds_in) {
        if (token_ids) {
            for (int32_t m = 0; m < M; m++) {
                if (token_ids[m] >= 0) {
                    dpuf_embed_qr_any_f32(&eng->w_tok_embd, 1, token_ids + m, eng->h_buf + (size_t)m * (size_t)h_dim, eng->prep_f32[0]);
                } else {
                    memcpy(eng->h_buf + (size_t)m * (size_t)h_dim, embeds_in + (size_t)m * (size_t)h_dim, (size_t)h_dim * sizeof(float));
                }
            }
        } else {
            memcpy(eng->h_buf, embeds_in, (size_t)M * (size_t)h_dim * sizeof(float));
        }
    } else {
        dpuf_embed_qr_any_f32(&eng->w_tok_embd, M, token_ids, eng->h_buf, eng->prep_f32[0]);
    }

    /* Precompute per-token RoPE cos/sin pointers (handling 3D M-RoPE if enabled) */
    const float* token_cos_p[DPUF_ENGINE_MAX_BATCH];
    const float* token_sin_p[DPUF_ENGINE_MAX_BATCH];
    float mrope_cos_buf[DPUF_ENGINE_MAX_BATCH * 64];
    float mrope_sin_buf[DPUF_ENGINE_MAX_BATCH * 64];

    for (int32_t m = 0; m < M; m++) {
        int32_t pt = pos_t ? pos_t[m] : (pos0 + m);
        int32_t ph = pos_h ? pos_h[m] : (pos0 + m);
        int32_t pw = pos_w ? pos_w[m] : (pos0 + m);
        if (!eng->cfg.has_mrope || (pt == ph && pt == pw)) {
            token_cos_p[m] = eng->cos_tab + (size_t)pt * (size_t)half_hd;
            token_sin_p[m] = eng->sin_tab + (size_t)pt * (size_t)half_hd;
        } else {
            float* cp = mrope_cos_buf + (size_t)m * (size_t)half_hd;
            float* sp = mrope_sin_buf + (size_t)m * (size_t)half_hd;
            eng_compute_mrope_cos_sin(hd, eng->cfg.rope_theta, pt, ph, pw, eng->cfg.mrope_section, cp, sp);
            token_cos_p[m] = cp;
            token_sin_p[m] = sp;
        }
    }

    eng->bar.arrived = 0;
    eng->bar.sense = 0;
    if (M >= 4) memset((void*)eng->work_ctr, 0, eng->work_ctr_bytes);

    int32_t req_th = eng->cfg.n_threads;

    #pragma omp parallel num_threads(req_th)
    {
        int32_t tid = omp_get_thread_num();
        int32_t nt = omp_get_num_threads();
        int32_t sense = 0;
        float* scores  = eng->attn_scores + (size_t)tid * 2u * (size_t)eng->cfg.max_seq_len;
        float* scores1 = scores + (size_t)eng->cfg.max_seq_len;
#ifdef _WIN32
        DWORD_PTR prev_aff = 0;
        if (M >= 4 && nt <= eng->n_phys_cores && tid < eng->n_phys_cores) {
            prev_aff = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)eng->core_masks[tid]);
        }
#endif

        const int32_t can_slab_fuse = (M >= 2 && ((M + 5) / 6) >= nt);

        for (int32_t l = 0; l < n_layers; l++) {
            const DpufLayerWeights* lw = &eng->layers[l];
            const float* prev_res = (l > 0) ? eng->hn_buf : NULL;

            /* 1 & 2. Fused [prev w_down residual +] Input RMSNorm + Prep Q/K/V */
            int32_t k_slot = 1, v_slot = 2;
            if (can_slab_fuse && eng->layer_share_kv[l] == 3 && dpuf_mode_uses_had_slab(lw->w_q.mode)) {
                fused_res_norm_prep_had4p_slab(
                    &lw->w_q, M, eng->h_buf, prev_res, lw->attn_norm, eps, eng->hn_buf,
                    eng->prep_f32[0], eng->prep_q8[0], eng->prep_scale[0], eng->prep_xsum[0], tid, nt
                );
                k_slot = 0;
                v_slot = 0;
                engine_barrier(&eng->bar, nt, &sense);
            } else {
                for (int32_t m = tid; m < M; m += nt) {
                    add_and_rms_norm_f32(
                        h_dim,
                        eng->h_buf + (size_t)m * (size_t)h_dim,
                        prev_res ? (prev_res + (size_t)m * (size_t)h_dim) : NULL,
                        lw->attn_norm, eps,
                        eng->hn_buf + (size_t)m * (size_t)h_dim
                    );
                }
                engine_barrier(&eng->bar, nt, &sense);

                if (eng->layer_share_kv[l] == 3) {
                    const DpufWeightDesc* ws[1] = { &lw->w_q };
                    const int32_t slots[1] = { 0 };
                    prepare_multi_weights_group(1, ws, slots, M, eng->hn_buf, eng, tid, nt);
                    k_slot = 0;
                    v_slot = 0;
                } else if (eng->layer_share_kv[l] == 1) {
                    const DpufWeightDesc* ws[2] = { &lw->w_q, &lw->w_v };
                    const int32_t slots[2] = { 0, 2 };
                    prepare_multi_weights_group(2, ws, slots, M, eng->hn_buf, eng, tid, nt);
                    k_slot = 0;
                    v_slot = 2;
                } else {
                    const DpufWeightDesc* ws[3] = { &lw->w_q, &lw->w_k, &lw->w_v };
                    const int32_t slots[3] = { 0, 1, 2 };
                    prepare_multi_weights_group(3, ws, slots, M, eng->hn_buf, eng, tid, nt);
                }
                engine_barrier(&eng->bar, nt, &sense);
            }

            /* 3. Execute Q, K, V matmuls without intermediate barriers */
            {
                const DpufWeightDesc* qkv_m[3] = { &lw->w_q, &lw->w_k, &lw->w_v };
                const int32_t qkv_slots[3] = { 0, k_slot, v_slot };
                float* qkv_outs[3] = { eng->q_buf, eng->k_buf, eng->v_buf };
                run_multi_weights_dynamic(3, qkv_m, qkv_slots, qkv_outs, M, eng, eng->work_ctr + (size_t)l * 128u + 0u, 2, tid, nt);
            }
            engine_barrier(&eng->bar, nt, &sense);

            /* 4. Per-KV-head QK-Norm + RoPE + Contiguous KV Cache + Vectorized GQA Attention */
            const int32_t hd_blk = hd >> 5;
            const size_t head_stride = (size_t)eng->cfg.max_seq_len * (size_t)hd;
            const size_t scale_head_stride = (size_t)eng->cfg.max_seq_len * (size_t)hd_blk;
            const int is_q8_kv = (eng->cfg.kv_type == DPUF_KV_Q8);

            if (M < 8) {
                for (int32_t kvh = tid; kvh < n_kv; kvh += nt) {
                    float* kc_head   = is_q8_kv ? NULL : (eng->k_cache + (size_t)l * layer_kv_stride + (size_t)kvh * head_stride);
                    float* vc_head   = is_q8_kv ? NULL : (eng->v_cache + (size_t)l * layer_kv_stride + (size_t)kvh * head_stride);
                    int8_t* kc_q8    = is_q8_kv ? (eng->k_cache_q8 + (size_t)l * layer_kv_stride + (size_t)kvh * head_stride) : NULL;
                    int8_t* vc_q8    = is_q8_kv ? (eng->v_cache_q8 + (size_t)l * layer_kv_stride + (size_t)kvh * head_stride) : NULL;
                    float* ks_head   = is_q8_kv ? (eng->k_scale_q8 + ((size_t)l * (size_t)n_kv + (size_t)kvh) * scale_head_stride) : NULL;
                    float* vs_head   = is_q8_kv ? (eng->v_scale_q8 + ((size_t)l * (size_t)n_kv + (size_t)kvh) * scale_head_stride) : NULL;

                    for (int32_t m = 0; m < M; m++) {
                        int32_t pos = pos_t ? pos_t[m] : (pos0 + m);
                        const float* cos_p = token_cos_p[m];
                        const float* sin_p = token_sin_p[m];
                        float* k_src = eng->k_buf + (size_t)m * (size_t)kv_dim + (size_t)kvh * (size_t)hd;
                        float* v_src = eng->v_buf + (size_t)m * (size_t)kv_dim + (size_t)kvh * (size_t)hd;

                        if (eng->cfg.has_qk_norm && lw->attn_k_norm) {
                            rms_norm_f32(hd, k_src, lw->attn_k_norm, eps, k_src);
                        }
                        apply_rope_neox_f32(hd, k_src, cos_p, sin_p);

                        if (is_q8_kv) {
                            quantize_vec_q8_blk32(hd, k_src, kc_q8 + (size_t)pos * (size_t)hd, ks_head + (size_t)pos * (size_t)hd_blk);
                            quantize_vec_q8_blk32(hd, v_src, vc_q8 + (size_t)pos * (size_t)hd, vs_head + (size_t)pos * (size_t)hd_blk);
                        } else {
                            memcpy(kc_head + (size_t)pos * (size_t)hd, k_src, (size_t)hd * sizeof(float));
                            memcpy(vc_head + (size_t)pos * (size_t)hd, v_src, (size_t)hd * sizeof(float));
                        }
                    }

                    for (int32_t m = 0; m < M; m++) {
                        int32_t pos = pos_t ? pos_t[m] : (pos0 + m);
                        int32_t kv_len = pos + 1;
                        const float* cos_p = token_cos_p[m];
                        const float* sin_p = token_sin_p[m];

                        for (int32_t g = 0; g < groups; g++) {
                            int32_t qh = kvh * groups + g;
                            float* q_vec = eng->q_buf + (size_t)m * (size_t)q_dim + (size_t)qh * (size_t)hd;
                            if (eng->cfg.has_qk_norm && lw->attn_q_norm) {
                                rms_norm_f32(hd, q_vec, lw->attn_q_norm, eps, q_vec);
                            }
                            apply_rope_neox_f32(hd, q_vec, cos_p, sin_p);
                        }

                        for (int32_t g = 0; g < groups; g += 2) {
                            int32_t qh0 = kvh * groups + g;
                            const float* q0 = eng->q_buf + (size_t)m * (size_t)q_dim + (size_t)qh0 * (size_t)hd;
                            float* a0 = eng->attn_out + (size_t)m * (size_t)q_dim + (size_t)qh0 * (size_t)hd;
                            int has_p = (g + 1 < groups);
                            const float* q1 = has_p ? (q0 + hd) : NULL;
                            float* a1 = has_p ? (a0 + hd) : NULL;
                            if (is_q8_kv) {
                                gqa_attn_head_pair_q8(hd, kv_len, attn_scale, kc_q8, ks_head, vc_q8, vs_head, q0, q1, scores, has_p ? scores1 : NULL, a0, a1);
                            } else {
                                gqa_attn_head_pair_f32(hd, kv_len, attn_scale, kc_head, vc_head, q0, q1, scores, has_p ? scores1 : NULL, a0, a1);
                            }
                        }
                    }
                }
            } else {
                /* 4a. Parallel K/V norm+RoPE+store & Q norm+RoPE across all threads */
                const int32_t tot_kv = M * n_kv;
                for (int32_t idx = tid; idx < tot_kv; idx += nt) {
                    int32_t m = idx / n_kv;
                    int32_t kvh = idx % n_kv;
                    int32_t pos = pos_t ? pos_t[m] : (pos0 + m);
                    const float* cos_p = token_cos_p[m];
                    const float* sin_p = token_sin_p[m];
                    float* k_src = eng->k_buf + (size_t)m * (size_t)kv_dim + (size_t)kvh * (size_t)hd;
                    float* v_src = eng->v_buf + (size_t)m * (size_t)kv_dim + (size_t)kvh * (size_t)hd;
                    if (eng->cfg.has_qk_norm && lw->attn_k_norm) {
                        rms_norm_f32(hd, k_src, lw->attn_k_norm, eps, k_src);
                    }
                    apply_rope_neox_f32(hd, k_src, cos_p, sin_p);
                    if (is_q8_kv) {
                        int8_t* kc_q8 = eng->k_cache_q8 + (size_t)l * layer_kv_stride + (size_t)kvh * head_stride;
                        int8_t* vc_q8 = eng->v_cache_q8 + (size_t)l * layer_kv_stride + (size_t)kvh * head_stride;
                        float* ks_head = eng->k_scale_q8 + ((size_t)l * (size_t)n_kv + (size_t)kvh) * scale_head_stride;
                        float* vs_head = eng->v_scale_q8 + ((size_t)l * (size_t)n_kv + (size_t)kvh) * scale_head_stride;
                        quantize_vec_q8_blk32(hd, k_src, kc_q8 + (size_t)pos * (size_t)hd, ks_head + (size_t)pos * (size_t)hd_blk);
                        quantize_vec_q8_blk32(hd, v_src, vc_q8 + (size_t)pos * (size_t)hd, vs_head + (size_t)pos * (size_t)hd_blk);
                    } else {
                        float* kc_head = eng->k_cache + (size_t)l * layer_kv_stride + (size_t)kvh * head_stride;
                        float* vc_head = eng->v_cache + (size_t)l * layer_kv_stride + (size_t)kvh * head_stride;
                        memcpy(kc_head + (size_t)pos * (size_t)hd, k_src, (size_t)hd * sizeof(float));
                        memcpy(vc_head + (size_t)pos * (size_t)hd, v_src, (size_t)hd * sizeof(float));
                    }
                }
                const int32_t tot_q = M * n_q;
                for (int32_t idx = tid; idx < tot_q; idx += nt) {
                    int32_t m = idx / n_q;
                    int32_t qh = idx % n_q;
                    int32_t pos = pos_t ? pos_t[m] : (pos0 + m);
                    const float* cos_p = token_cos_p[m];
                    const float* sin_p = token_sin_p[m];
                    float* q_vec = eng->q_buf + (size_t)m * (size_t)q_dim + (size_t)qh * (size_t)hd;
                    if (eng->cfg.has_qk_norm && lw->attn_q_norm) {
                        rms_norm_f32(hd, q_vec, lw->attn_q_norm, eps, q_vec);
                    }
                    apply_rope_neox_f32(hd, q_vec, cos_p, sin_p);
                }
                engine_barrier(&eng->bar, nt, &sense);

                /* 4b. Dynamic LPT work-stealing over (mb_rev, kvh), heaviest tokens first */
                const int32_t mb_sz = 8;
                const int32_t n_mb = (M + mb_sz - 1) / mb_sz;
                const int32_t tot_tasks = n_mb * n_kv;
                volatile long* a_ctr = eng->work_ctr + (size_t)l * 128u + 16u;
                while (1) {
#ifdef _WIN32
                    int32_t task = (int32_t)_InterlockedExchangeAdd(a_ctr, 1L);
#else
                    int32_t task;
                    #pragma omp atomic capture
                    { task = (int32_t)(*a_ctr); (*a_ctr)++; }
#endif
                    if (task >= tot_tasks) break;
                    int32_t mb_rev = task / n_kv;
                    int32_t mb = n_mb - 1 - mb_rev;
                    int32_t kvh = task % n_kv;
                    int32_t m0 = mb * mb_sz;
                    int32_t m1 = (m0 + mb_sz <= M) ? (m0 + mb_sz) : M;

                    const float* kc_head = is_q8_kv ? NULL : (eng->k_cache + (size_t)l * layer_kv_stride + (size_t)kvh * head_stride);
                    const float* vc_head = is_q8_kv ? NULL : (eng->v_cache + (size_t)l * layer_kv_stride + (size_t)kvh * head_stride);
                    const int8_t* kc_q8  = is_q8_kv ? (eng->k_cache_q8 + (size_t)l * layer_kv_stride + (size_t)kvh * head_stride) : NULL;
                    const int8_t* vc_q8  = is_q8_kv ? (eng->v_cache_q8 + (size_t)l * layer_kv_stride + (size_t)kvh * head_stride) : NULL;
                    const float* ks_head = is_q8_kv ? (eng->k_scale_q8 + ((size_t)l * (size_t)n_kv + (size_t)kvh) * scale_head_stride) : NULL;
                    const float* vs_head = is_q8_kv ? (eng->v_scale_q8 + ((size_t)l * (size_t)n_kv + (size_t)kvh) * scale_head_stride) : NULL;

                    for (int32_t m = m0; m < m1; m++) {
                        int32_t kv_len = pos0 + m + 1;
                        for (int32_t g = 0; g < groups; g += 2) {
                            int32_t qh0 = kvh * groups + g;
                            const float* q0 = eng->q_buf + (size_t)m * (size_t)q_dim + (size_t)qh0 * (size_t)hd;
                            float* a0 = eng->attn_out + (size_t)m * (size_t)q_dim + (size_t)qh0 * (size_t)hd;
                            int has_p = (g + 1 < groups);
                            const float* q1 = has_p ? (q0 + hd) : NULL;
                            float* a1 = has_p ? (a0 + hd) : NULL;
                            if (is_q8_kv) {
                                gqa_attn_head_pair_q8(hd, kv_len, attn_scale, kc_q8, ks_head, vc_q8, vs_head, q0, q1, scores, has_p ? scores1 : NULL, a0, a1);
                            } else {
                                gqa_attn_head_pair_f32(hd, kv_len, attn_scale, kc_head, vc_head, q0, q1, scores, has_p ? scores1 : NULL, a0, a1);
                            }
                        }
                    }
                }
            }
            engine_barrier(&eng->bar, nt, &sense);

            /* 5. Prep O_proj (slot 0) from attn_out */
            prepare_input_m_or_parallel(
                &lw->w_o, M, eng->attn_out, eng->prep_f32[0], eng->prep_q8[0], eng->prep_scale[0], eng->prep_xsum[0], tid, nt
            );
            engine_barrier(&eng->bar, nt, &sense);

            /* Reuse hn_buf [M, h_dim] for O_proj output */
            run_weight_dynamic(&lw->w_o, M, eng, 0, eng->hn_buf, eng->work_ctr + (size_t)l * 128u + 32u, 1, tid, nt);
            engine_barrier(&eng->bar, nt, &sense);

            /* 6 & 7. Fused Residual h += O_proj + FFN RMSNorm + Prep Gate/Up */
            int32_t up_slot = 1;
            if (can_slab_fuse && eng->layer_share_gu[l] && dpuf_mode_uses_had_slab(lw->w_gate.mode)) {
                fused_res_norm_prep_had4p_slab(
                    &lw->w_gate, M, eng->h_buf, eng->hn_buf, lw->ffn_norm, eps, eng->attn_out,
                    eng->prep_f32[0], eng->prep_q8[0], eng->prep_scale[0], eng->prep_xsum[0], tid, nt
                );
                up_slot = 0;
                engine_barrier(&eng->bar, nt, &sense);
            } else {
                for (int32_t m = tid; m < M; m += nt) {
                    add_and_rms_norm_f32(
                        h_dim,
                        eng->h_buf + (size_t)m * (size_t)h_dim,
                        eng->hn_buf + (size_t)m * (size_t)h_dim,
                        lw->ffn_norm, eps,
                        eng->attn_out + (size_t)m * (size_t)h_dim
                    );
                }
                engine_barrier(&eng->bar, nt, &sense);

                if (eng->layer_share_gu[l]) {
                    const DpufWeightDesc* ws[1] = { &lw->w_gate };
                    const int32_t slots[1] = { 0 };
                    prepare_multi_weights_group(1, ws, slots, M, eng->attn_out, eng, tid, nt);
                    up_slot = 0;
                } else {
                    const DpufWeightDesc* ws[2] = { &lw->w_gate, &lw->w_up };
                    const int32_t slots[2] = { 0, 1 };
                    prepare_multi_weights_group(2, ws, slots, M, eng->attn_out, eng, tid, nt);
                }
                engine_barrier(&eng->bar, nt, &sense);
            }

            /* 8. Execute Gate & Up matmuls together */
            {
                const DpufWeightDesc* gu_m[2] = { &lw->w_gate, &lw->w_up };
                const int32_t gu_slots[2] = { 0, up_slot };
                float* gu_outs[2] = { eng->gate_buf, eng->up_buf };
                run_multi_weights_dynamic(2, gu_m, gu_slots, gu_outs, M, eng, eng->work_ctr + (size_t)l * 128u + 48u, 2, tid, nt);
            }
            engine_barrier(&eng->bar, nt, &sense);

            /* 9 & 10. Fused Vectorized AVX2 SwiGLU + Prep w_down (slot 0) */
            if (can_slab_fuse && dpuf_mode_uses_had_slab(lw->w_down.mode)) {
                int32_t n_blk = inter >> 5;
                int32_t n_slabs = (M + 5) / 6;
                for (int32_t s = tid; s < n_slabs; s += nt) {
                    int32_t m0 = s * 6;
                    int32_t mc = (m0 + 6 <= M) ? 6 : (M - m0);
                    swiglu_range_avx2(
                        (size_t)m0 * (size_t)inter, (size_t)(m0 + mc) * (size_t)inter,
                        eng->gate_buf, eng->up_buf, eng->ffn_hidden, eng->swiglu_deadzone
                    );
                    dpuf_prep_had4_packed_slab(
                        mc, inter,
                        eng->ffn_hidden + (size_t)m0 * (size_t)inter,
                        lw->w_down.sign_bits,
                        eng->prep_f32[0]   + (size_t)m0 * (size_t)inter,
                        eng->prep_q8[0]    + (size_t)s * (size_t)n_blk * 192u,
                        eng->prep_scale[0] + (size_t)s * (size_t)n_blk * 6u,
                        eng->prep_xsum[0]  + (size_t)s * (size_t)n_blk * 6u,
                        1
                    );
                }
                engine_barrier(&eng->bar, nt, &sense);
            } else {
                size_t total_inter = (size_t)M * (size_t)inter;
                size_t s0 = (total_inter * (size_t)tid) / (size_t)nt;
                size_t s1 = (total_inter * (size_t)(tid + 1)) / (size_t)nt;
                swiglu_range_avx2(s0, s1, eng->gate_buf, eng->up_buf, eng->ffn_hidden, eng->swiglu_deadzone);
                engine_barrier(&eng->bar, nt, &sense);

                prepare_input_m_or_parallel(
                    &lw->w_down, M, eng->ffn_hidden, eng->prep_f32[0], eng->prep_q8[0], eng->prep_scale[0], eng->prep_xsum[0], tid, nt
                );
                engine_barrier(&eng->bar, nt, &sense);
            }

            /* 11. Execute w_down into hn_buf (Dual-Regime Dynamic Architecture)
             * - M > 1 (Prompt Prefill): Draft Pruning (56/64 tiles = 87.5%) for max throughput (750+ tok/s)
             * - M == 1 (Generation): Full tiles (64/64) + Early-Exit Vocab + DeadZone for max speed & 100% stability
             */
            int32_t down_tiles = lw->w_down.n_tiles;
            const char* env_kt = (M > 1) ? getenv("DPUF_PREFILL_KEEP_TILES") : getenv("DPUF_GEN_KEEP_TILES");
            if (env_kt) {
                int32_t kt = atoi(env_kt);
                if (kt > 0 && kt < down_tiles) down_tiles = kt;
            }
            if (down_tiles < lw->w_down.n_tiles) {
                int32_t t0 = (down_tiles * tid) / nt;
                int32_t t1 = (down_tiles * (tid + 1)) / nt;
                run_weight_tile_range(&lw->w_down, M, eng, 0, t0, t1, eng->hn_buf);
                int32_t r_done = down_tiles * 16;
                int32_t rem_r = lw->w_down.rows - r_done;
                if (rem_r > 0) {
                    for (int32_t m = 0; m < M; m++) {
                        float* hn_m = eng->hn_buf + (size_t)m * (size_t)h_dim;
                        int32_t r0 = r_done + (rem_r * tid) / nt;
                        int32_t r1 = r_done + (rem_r * (tid + 1)) / nt;
                        memset(hn_m + r0, 0, (size_t)(r1 - r0) * sizeof(float));
                    }
                }
            } else {
                run_weight_dynamic(&lw->w_down, M, eng, 0, eng->hn_buf, eng->work_ctr + (size_t)l * 128u + 64u, 1, tid, nt);
            }

            /* DeepStack additive injection after layers 0, 1, 2 */
            if (deepstack_in && l < 3 && deepstack_in[l]) {
                const float* ds_l = deepstack_in[l];
                for (int32_t m = tid; m < M; m += nt) {
                    float* hn_m = eng->hn_buf + (size_t)m * (size_t)h_dim;
                    const float* ds_m = ds_l + (size_t)m * (size_t)h_dim;
                    for (int32_t d = 0; d < h_dim; d++) hn_m[d] += ds_m[d];
                }
            }
            engine_barrier(&eng->bar, nt, &sense);
        }

        /* Final residual + output RMSNorm + LM-Head (skip when only_last_logits == 2 for KV-only chunks) */
        if (only_last_logits != 2) {
            int32_t lm_start = only_last_logits ? (M - 1) : 0;
            int32_t lm_M = M - lm_start;
            const float* last_res = (n_layers > 0) ? (eng->hn_buf + (size_t)lm_start * (size_t)h_dim) : NULL;
            if (dpuf_mode_uses_had_slab(eng->w_lm_head.mode) && lm_M >= 2 && ((lm_M + 5) / 6) >= nt) {
                fused_res_norm_prep_had4p_slab(
                    &eng->w_lm_head, lm_M,
                    eng->h_buf + (size_t)lm_start * (size_t)h_dim,
                    last_res,
                    eng->output_norm, eps,
                    eng->attn_out,
                    eng->prep_f32[0], eng->prep_q8[0], eng->prep_scale[0], eng->prep_xsum[0], tid, nt
                );
                engine_barrier(&eng->bar, nt, &sense);
            } else {
                for (int32_t lm_m = tid; lm_m < lm_M; lm_m += nt) {
                    int32_t m = lm_start + lm_m;
                    float* out_n = eng->hn_buf + (size_t)lm_m * (size_t)h_dim;
                    add_and_rms_norm_f32(
                        h_dim,
                        eng->h_buf + (size_t)m * (size_t)h_dim,
                        last_res ? (eng->hn_buf + (size_t)m * (size_t)h_dim) : NULL,
                        eng->output_norm, eps, out_n
                    );
                }
                engine_barrier(&eng->bar, nt, &sense);
                prepare_input_m_or_parallel(
                    &eng->w_lm_head, lm_M, eng->hn_buf, eng->prep_f32[0], eng->prep_q8[0], eng->prep_scale[0], eng->prep_xsum[0], tid, nt
                );
                engine_barrier(&eng->bar, nt, &sense);
            }

            const char* env_t = getenv("DPUF_EARLY_TILES");
            int32_t use_early = (env_t != NULL && lm_M == 1 && eng->w_lm_head.n_tiles > 2048);
            if (use_early) {
                int32_t za_tiles = atoi(env_t);
                if (za_tiles < 256) za_tiles = 256;
                if (za_tiles > eng->w_lm_head.n_tiles) za_tiles = eng->w_lm_head.n_tiles;
                int32_t za0 = (za_tiles * tid) / nt;
                int32_t za1 = (za_tiles * (tid + 1)) / nt;
                run_weight_tile_range(&eng->w_lm_head, 1, eng, 0, za0, za1, logits_out);
                engine_barrier(&eng->bar, nt, &sense);

                volatile long* exit_flag = eng->work_ctr + (size_t)n_layers * 128u + 32u;
                if (tid == 0) {
                    *exit_flag = 0;
                    int32_t n_za_tok = za_tiles * 16;
                    if (n_za_tok > eng->cfg.vocab_size) n_za_tok = eng->cfg.vocab_size;
                    float b1 = -1e30f, b2 = -1e30f;
                    for (int32_t i = 0; i < n_za_tok; i++) {
                        float v = logits_out[i];
                        if (v > b1) { b2 = b1; b1 = v; }
                        else if (v > b2) { b2 = v; }
                    }
                    const char* env_m = getenv("DPUF_EARLY_MARGIN");
                    float m_thresh = env_m ? (float)atof(env_m) : 1.25f;
                    if (m_thresh > 0.0f && (b1 - b2) >= m_thresh) {
                        *exit_flag = 1;
                    }
                }
                engine_barrier(&eng->bar, nt, &sense);

                if (*exit_flag != 0) {
                    int32_t n_za_tok = za_tiles * 16;
                    int32_t V = eng->cfg.vocab_size;
                    int32_t rem_tok = V - n_za_tok;
                    if (rem_tok > 0) {
                        int32_t t_tok0 = n_za_tok + (rem_tok * tid) / nt;
                        int32_t t_tok1 = n_za_tok + (rem_tok * (tid + 1)) / nt;
                        for (int32_t i = t_tok0; i < t_tok1; i++) logits_out[i] = -1e9f;
                    }
                } else {
                    int32_t rem_tiles = eng->w_lm_head.n_tiles - za_tiles;
                    int32_t zr0 = za_tiles + (rem_tiles * tid) / nt;
                    int32_t zr1 = za_tiles + (rem_tiles * (tid + 1)) / nt;
                    run_weight_tile_range(&eng->w_lm_head, 1, eng, 0, zr0, zr1, logits_out);
                }
                engine_barrier(&eng->bar, nt, &sense);
            } else {
                run_weight_dynamic(&eng->w_lm_head, lm_M, eng, 0, logits_out, eng->work_ctr + (size_t)n_layers * 128u, 64, tid, nt);
            }
        }
#ifdef _WIN32
        if (prev_aff != 0) SetThreadAffinityMask(GetCurrentThread(), prev_aff);
#endif
    }

    return 0;
}

DPUF_EXPORT int32_t dpuf_engine_forward_batch(
    void* handle,
    int32_t n_tokens,
    const int32_t* token_ids,
    int32_t pos0,
    int32_t only_last_logits,
    float* logits_out
) {
    return dpuf_engine_forward_batch_ex(
        handle, n_tokens, token_ids, pos0, NULL, NULL, NULL, NULL, NULL, only_last_logits, logits_out
    );
}

DPUF_EXPORT int32_t dpuf_engine_forward_token(
    void* handle,
    int32_t token_id,
    int32_t pos,
    float* logits_out
) {
    int32_t tid = token_id;
    return dpuf_engine_forward_batch(handle, 1, &tid, pos, 1, logits_out);
}
