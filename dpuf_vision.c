/*
 * dpuf_vision.c — Pure C / AVX2 ViT Encoder & Patch Merger Implementation for DPUF
 * Copyright (c) 2026 D. P. (dule_pan). Licensed under DPUF Community & Enterprise License.
 */
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "stb_image.h"

#include "dpuf_vision.h"
#include "dpuf_kernels.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static __forceinline float vis_hsum256_ps(__m256 v) {
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

static void vis_layer_norm_f32(
    int32_t n, const float* x, const float* w, const float* b, float eps, float* out
) {
    __m256 sum_v = _mm256_setzero_ps();
    __m256 sq_v  = _mm256_setzero_ps();
    int32_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        sum_v = _mm256_add_ps(sum_v, v);
        sq_v  = _mm256_fmadd_ps(v, v, sq_v);
    }
    float s = vis_hsum256_ps(sum_v);
    float ss = vis_hsum256_ps(sq_v);
    for (; i < n; i++) {
        s += x[i];
        ss += x[i] * x[i];
    }
    float fn = (float)n;
    float mean = s / fn;
    float var = ss / fn - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float inv_std = 1.0f / sqrtf(var + eps);

    __m256 v_mean = _mm256_set1_ps(mean);
    __m256 v_inv  = _mm256_set1_ps(inv_std);
    for (i = 0; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = w ? _mm256_loadu_ps(w + i) : _mm256_set1_ps(1.0f);
        __m256 vb = b ? _mm256_loadu_ps(b + i) : _mm256_setzero_ps();
        __m256 xn = _mm256_mul_ps(_mm256_sub_ps(vx, v_mean), v_inv);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(xn, vw, vb));
    }
    for (; i < n; i++) {
        float xn = (x[i] - mean) * inv_std;
        out[i] = xn * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
    }
}

static __forceinline float vis_gelu_pytorch_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
}

static __forceinline float vis_gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

int32_t dpuf_vision_smart_resize(
    int32_t in_w, int32_t in_h, const DpufVisionConfig* cfg,
    int32_t* out_w, int32_t* out_h
) {
    int32_t factor = cfg->patch_size * cfg->spatial_merge_size; /* 16 * 2 = 32 */
    if (factor <= 0) factor = 32;
    int32_t min_p = cfg->min_pixels > 0 ? cfg->min_pixels : 65536;
    int32_t max_p = cfg->max_pixels > 0 ? cfg->max_pixels : 16777216;

    int32_t h_bar = (int32_t)(roundf((float)in_h / (float)factor)) * factor;
    int32_t w_bar = (int32_t)(roundf((float)in_w / (float)factor)) * factor;
    if (h_bar < factor) h_bar = factor;
    if (w_bar < factor) w_bar = factor;

    int64_t prod = (int64_t)h_bar * (int64_t)w_bar;
    if (prod > (int64_t)max_p) {
        float beta = sqrtf((float)((int64_t)in_h * (int64_t)in_w) / (float)max_p);
        h_bar = (int32_t)floorf((float)in_h / beta / (float)factor) * factor;
        w_bar = (int32_t)floorf((float)in_w / beta / (float)factor) * factor;
        if (h_bar < factor) h_bar = factor;
        if (w_bar < factor) w_bar = factor;
    } else if (prod < (int64_t)min_p) {
        float beta = sqrtf((float)min_p / (float)((int64_t)in_h * (int64_t)in_w));
        h_bar = (int32_t)ceilf((float)in_h * beta / (float)factor) * factor;
        w_bar = (int32_t)ceilf((float)in_w * beta / (float)factor) * factor;
    }
    *out_w = w_bar;
    *out_h = h_bar;
    return 0;
}

float* dpuf_vision_load_image(
    const char* image_path, const DpufVisionConfig* cfg,
    int32_t* out_w, int32_t* out_h
) {
    int raw_w = 0, raw_h = 0, channels = 0;
    stbi_uc* raw_pixels = stbi_load(image_path, &raw_w, &raw_h, &channels, 3);
    if (!raw_pixels) return NULL;

    int32_t target_w = 0, target_h = 0;
    dpuf_vision_smart_resize(raw_w, raw_h, cfg, &target_w, &target_h);

    float* norm_rgb = (float*)malloc((size_t)target_w * (size_t)target_h * 3u * sizeof(float));
    if (!norm_rgb) {
        stbi_image_free(raw_pixels);
        return NULL;
    }

    float m[3] = { cfg->image_mean[0], cfg->image_mean[1], cfg->image_mean[2] };
    float s[3] = { cfg->image_std[0],  cfg->image_std[1],  cfg->image_std[2] };
    if (s[0] == 0.0f) s[0] = 0.5f;
    if (s[1] == 0.0f) s[1] = 0.5f;
    if (s[2] == 0.0f) s[2] = 0.5f;

    /* Bilinear resize + normalization into [target_h, target_w, 3] */
    float x_ratio = (float)raw_w / (float)target_w;
    float y_ratio = (float)raw_h / (float)target_h;

    int32_t y;
    #pragma omp parallel for schedule(static)
    for (y = 0; y < target_h; y++) {
        float src_y = ((float)y + 0.5f) * y_ratio - 0.5f;
        int32_t y0 = (int32_t)floorf(src_y);
        int32_t y1 = y0 + 1;
        float dy = src_y - (float)y0;
        if (y0 < 0) { y0 = 0; dy = 0.0f; }
        if (y1 >= raw_h) { y1 = raw_h - 1; dy = 0.0f; }

        for (int32_t x = 0; x < target_w; x++) {
            float src_x = ((float)x + 0.5f) * x_ratio - 0.5f;
            int32_t x0 = (int32_t)floorf(src_x);
            int32_t x1 = x0 + 1;
            float dx = src_x - (float)x0;
            if (x0 < 0) { x0 = 0; dx = 0.0f; }
            if (x1 >= raw_w) { x1 = raw_w - 1; dx = 0.0f; }

            float w00 = (1.0f - dx) * (1.0f - dy);
            float w01 = dx * (1.0f - dy);
            float w10 = (1.0f - dx) * dy;
            float w11 = dx * dy;

            const stbi_uc* p00 = raw_pixels + (y0 * raw_w + x0) * 3;
            const stbi_uc* p01 = raw_pixels + (y0 * raw_w + x1) * 3;
            const stbi_uc* p10 = raw_pixels + (y1 * raw_w + x0) * 3;
            const stbi_uc* p11 = raw_pixels + (y1 * raw_w + x1) * 3;

            size_t dst_idx = ((size_t)y * (size_t)target_w + (size_t)x) * 3u;
            for (int c = 0; c < 3; c++) {
                float val = w00 * (float)p00[c] + w01 * (float)p01[c] +
                            w10 * (float)p10[c] + w11 * (float)p11[c];
                norm_rgb[dst_idx + c] = (val / 255.0f - m[c]) / s[c];
            }
        }
    }

    stbi_image_free(raw_pixels);
    *out_w = target_w;
    *out_h = target_h;
    return norm_rgb;
}

void dpuf_vision_free_image(float* img_rgb_norm) {
    if (img_rgb_norm) free(img_rgb_norm);
}

void dpuf_vision_free_output(DpufVisionOutput* out) {
    if (!out) return;
    if (out->main_embeds) { free(out->main_embeds); out->main_embeds = NULL; }
    for (int k = 0; k < 3; k++) {
        if (out->deepstack_embeds[k]) {
            free(out->deepstack_embeds[k]);
            out->deepstack_embeds[k] = NULL;
        }
    }
    if (out->pos_t) { free(out->pos_t); out->pos_t = NULL; }
    if (out->pos_h) { free(out->pos_h); out->pos_h = NULL; }
    if (out->pos_w) { free(out->pos_w); out->pos_w = NULL; }
    out->llm_tokens = 0;
}

static void vis_matmul_vec(const DpufWeightDesc* w, const float* x, const float* b, float* out) {
    if (!w || !w->tensor_base) return;
    if (w->mode == DPUF_MODE_QR_HAD4_PACKED) {
        dpuf_matmul_qr_hadamard4_packed_f32(1, w->rows, w->cols, w->n_tiles, x, w->tensor_base, w->tile_offsets, w->sign_bits, out);
    } else if (w->mode == DPUF_MODE_QR_HAD4_ASYM) {
        dpuf_matmul_qr_had4_asym_f32(1, w->rows, w->cols, w->n_tiles, x, w->tensor_base, w->tile_offsets, w->sign_bits, out);
    } else if (w->mode == DPUF_MODE_QR_HAD6) {
        dpuf_matmul_qr_had6_f32(1, w->rows, w->cols, w->n_tiles, x, w->tensor_base, w->tile_offsets, w->sign_bits, out);
    } else if (w->mode == DPUF_MODE_QR_HAD4) {
        dpuf_matmul_qr_hadamard4_f32(1, w->rows, w->cols, w->n_tiles, x, w->tensor_base, w->tile_offsets, w->sign_bits, out);
    } else if (w->mode == DPUF_MODE_QR_SPLIT4) {
        dpuf_matmul_qr_split4_f32(1, w->rows, w->cols, w->n_tiles, x, w->tensor_base, w->tile_offsets, w->col_scale, out);
    } else if (w->mode == DPUF_MODE_QR_ZD4) {
        dpuf_matmul_qr_zd4_f32(1, w->rows, w->cols, w->n_tiles, x, w->tensor_base, w->tile_offsets, w->col_scale, out);
    } else if (w->mode == DPUF_MODE_QR_Q8) {
        dpuf_matmul_qr_q8_f32(1, w->rows, w->cols, w->n_tiles, x, w->tensor_base, w->tile_offsets, out);
    }
    if (b) {
        int32_t rows = w->rows;
        int32_t i = 0;
        for (; i + 7 < rows; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(out + i), _mm256_loadu_ps(b + i)));
        }
        for (; i < rows; i++) out[i] += b[i];
    }
}

int32_t dpuf_vision_forward(
    const DpufVisionModel* vm,
    const float* img_rgb_norm,
    int32_t w, int32_t h,
    int32_t start_pos,
    DpufVisionOutput* out
) {
    if (!vm || !img_rgb_norm || !out) return -1;
    const DpufVisionConfig* cfg = &vm->cfg;
    int32_t patch_sz = cfg->patch_size > 0 ? cfg->patch_size : 16;
    int32_t merge_sz = cfg->spatial_merge_size > 0 ? cfg->spatial_merge_size : 2;
    int32_t temp_sz  = cfg->temporal_patch_size > 0 ? cfg->temporal_patch_size : 2;
    int32_t v_dim    = cfg->hidden_size;       /* 1152 */
    int32_t v_depth  = cfg->depth;             /* 27 */
    int32_t n_heads  = cfg->num_heads;         /* 16 */
    int32_t hd       = cfg->head_dim;          /* 72 */
    int32_t out_dim  = cfg->out_hidden_size;   /* 4096 */
    int32_t merge_dim = v_dim * (merge_sz * merge_sz); /* 4608 */
    float eps = cfg->layer_norm_eps > 0.0f ? cfg->layer_norm_eps : 1e-6f;

    int32_t grid_h = h / patch_sz;
    int32_t grid_w = w / patch_sz;
    int32_t n_patches = grid_h * grid_w;
    int32_t llm_tokens = (grid_h / merge_sz) * (grid_w / merge_sz);
    int32_t patch_dim = 3 * temp_sz * patch_sz * patch_sz; /* 3 * 2 * 16 * 16 = 1536 */

    out->llm_tokens = llm_tokens;
    out->grid_h = grid_h;
    out->grid_w = grid_w;
    out->main_embeds = (float*)calloc((size_t)llm_tokens * (size_t)out_dim, sizeof(float));
    for (int k = 0; k < 3; k++) {
        out->deepstack_embeds[k] = (float*)calloc((size_t)llm_tokens * (size_t)out_dim, sizeof(float));
    }
    out->pos_t = (int32_t*)malloc((size_t)llm_tokens * sizeof(int32_t));
    out->pos_h = (int32_t*)malloc((size_t)llm_tokens * sizeof(int32_t));
    out->pos_w = (int32_t*)malloc((size_t)llm_tokens * sizeof(int32_t));

    /* Initialize 3D M-RoPE positions for merged visual tokens */
    int32_t blocks_w = grid_w / merge_sz;
    for (int32_t bh = 0; bh < grid_h / merge_sz; bh++) {
        for (int32_t bw = 0; bw < grid_w / merge_sz; bw++) {
            int32_t tid = bh * blocks_w + bw;
            out->pos_t[tid] = start_pos;
            out->pos_h[tid] = start_pos + bh;
            out->pos_w[tid] = start_pos + bw;
        }
    }

    /* Allocate scratch buffers */
    float* hidden = (float*)malloc((size_t)n_patches * (size_t)v_dim * sizeof(float));
    float* qkv_buf = (float*)malloc((size_t)n_patches * (size_t)(v_dim * 3) * sizeof(float));
    float* attn_buf = (float*)malloc((size_t)n_patches * (size_t)v_dim * sizeof(float));
    float* norm_buf = (float*)malloc((size_t)n_patches * (size_t)v_dim * sizeof(float));
    float* ffn1_buf = (float*)malloc((size_t)n_patches * 4352u * sizeof(float)); /* zero-padded to 4352 */

    /* 1. Patchify + Patch Embed Projection */
    int32_t p, t;
    #pragma omp parallel for schedule(static)
    for (p = 0; p < n_patches; p++) {
        /* Spatial merge block ordering */
        int32_t block_idx = p / (merge_sz * merge_sz);
        int32_t in_block   = p % (merge_sz * merge_sz);
        int32_t bh = block_idx / blocks_w;
        int32_t bw = block_idx % blocks_w;
        int32_t ih = in_block / merge_sz;
        int32_t iw = in_block % merge_sz;
        int32_t py = (bh * merge_sz + ih) * patch_sz;
        int32_t px = (bw * merge_sz + iw) * patch_sz;

        float pvec[1536];
        int32_t idx = 0;
        for (int c = 0; c < 3; c++) {
            for (int t = 0; t < temp_sz; t++) {
                for (int dy = 0; dy < patch_sz; dy++) {
                    for (int dx = 0; dx < patch_sz; dx++) {
                        pvec[idx++] = img_rgb_norm[((py + dy) * w + (px + dx)) * 3 + c];
                    }
                }
            }
        }
        float* h_p = hidden + (size_t)p * (size_t)v_dim;
        vis_matmul_vec(&vm->w_patch_embed, pvec, vm->patch_embed_b, h_p);
    }

    /* 2. Add learned pos_embed with bilinear interpolation (align_corners = True) */
    if (vm->pos_embed) {
        int32_t side = 48; /* sqrt(2304) */
        float gh_denom = grid_h > 1 ? (float)(grid_h - 1) : 1.0f;
        float gw_denom = grid_w > 1 ? (float)(grid_w - 1) : 1.0f;

        #pragma omp parallel for schedule(static)
        for (p = 0; p < n_patches; p++) {
            int32_t block_idx = p / (merge_sz * merge_sz);
            int32_t in_block   = p % (merge_sz * merge_sz);
            int32_t bh = block_idx / blocks_w;
            int32_t bw = block_idx % blocks_w;
            int32_t ih = in_block / merge_sz;
            int32_t iw = in_block % merge_sz;
            int32_t r = bh * merge_sz + ih;
            int32_t c = bw * merge_sz + iw;

            float src_h = (float)r * (float)(side - 1) / gh_denom;
            float src_w = (float)c * (float)(side - 1) / gw_denom;
            int32_t h0 = (int32_t)floorf(src_h);
            int32_t h1 = h0 + 1 < side ? h0 + 1 : h0;
            int32_t w0 = (int32_t)floorf(src_w);
            int32_t w1 = w0 + 1 < side ? w0 + 1 : w0;
            float dh = src_h - (float)h0;
            float dw = src_w - (float)w0;

            float w00 = (1.0f - dh) * (1.0f - dw);
            float w01 = (1.0f - dh) * dw;
            float w10 = dh * (1.0f - dw);
            float w11 = dh * dw;

            const float* pe00 = vm->pos_embed + (size_t)(h0 * side + w0) * (size_t)v_dim;
            const float* pe01 = vm->pos_embed + (size_t)(h0 * side + w1) * (size_t)v_dim;
            const float* pe10 = vm->pos_embed + (size_t)(h1 * side + w0) * (size_t)v_dim;
            const float* pe11 = vm->pos_embed + (size_t)(h1 * side + w1) * (size_t)v_dim;

            float* h_p = hidden + (size_t)p * (size_t)v_dim;
            for (int32_t d = 0; d < v_dim; d++) {
                h_p[d] += w00 * pe00[d] + w01 * pe01[d] + w10 * pe10[d] + w11 * pe11[d];
            }
        }
    }

    /* Precompute axial 2D RoPE frequencies for all patches */
    float* cos_tab = (float*)malloc((size_t)n_patches * (size_t)hd * sizeof(float));
    float* sin_tab = (float*)malloc((size_t)n_patches * (size_t)hd * sizeof(float));
    float v_theta = cfg->rope_theta > 0.0f ? cfg->rope_theta : 10000.0f;
    float inv_freq[18];
    for (int k = 0; k < 18; k++) inv_freq[k] = 1.0f / powf(v_theta, 2.0f * (float)k / 36.0f);

    for (int32_t p = 0; p < n_patches; p++) {
        int32_t block_idx = p / (merge_sz * merge_sz);
        int32_t in_block   = p % (merge_sz * merge_sz);
        int32_t bh = block_idx / blocks_w;
        int32_t bw = block_idx % blocks_w;
        int32_t ih = in_block / merge_sz;
        int32_t iw = in_block % merge_sz;
        float r = (float)(bh * merge_sz + ih);
        float c = (float)(bw * merge_sz + iw);

        float* cp = cos_tab + (size_t)p * (size_t)hd;
        float* sp = sin_tab + (size_t)p * (size_t)hd;
        for (int k = 0; k < 18; k++) {
            float ah = r * inv_freq[k];
            float aw = c * inv_freq[k];
            cp[k]      = cosf(ah); sp[k]      = sinf(ah);
            cp[k + 18] = cosf(aw); sp[k + 18] = sinf(aw);
            cp[k + 36] = cosf(ah); sp[k + 36] = sinf(ah);
            cp[k + 54] = cosf(aw); sp[k + 54] = sinf(aw);
        }
    }

    float attn_scale = 1.0f / sqrtf((float)hd);

    /* 3. Run ViT Transformer Layers */
    for (int32_t l = 0; l < v_depth; l++) {
        const DpufVisionBlock* vb = &vm->blocks[l];

        /* Norm1 + QKV projection */
        #pragma omp parallel for schedule(static)
        for (p = 0; p < n_patches; p++) {
            float* h_p = hidden + (size_t)p * (size_t)v_dim;
            float* norm_p = norm_buf + (size_t)p * (size_t)v_dim;
            float* qkv_p = qkv_buf + (size_t)p * (size_t)(v_dim * 3);

            vis_layer_norm_f32(v_dim, h_p, vb->norm1_w, vb->norm1_b, eps, norm_p);
            vis_matmul_vec(&vb->w_qkv, norm_p, vb->qkv_b, qkv_p);

            /* Apply 2D axial RoPE to Q and K */
            float* q_p = qkv_p;
            float* k_p = qkv_p + v_dim;
            const float* cp = cos_tab + (size_t)p * (size_t)hd;
            const float* sp = sin_tab + (size_t)p * (size_t)hd;
            for (int h_idx = 0; h_idx < n_heads; h_idx++) {
                float* qh = q_p + h_idx * hd;
                float* kh = k_p + h_idx * hd;
                for (int j = 0; j < 36; j++) {
                    float q1 = qh[j], q2 = qh[j + 36];
                    float k1 = kh[j], k2 = kh[j + 36];
                    float c = cp[j], s = sp[j];
                    qh[j]      = q1 * c - q2 * s;
                    qh[j + 36] = q2 * c + q1 * s;
                    kh[j]      = k1 * c - k2 * s;
                    kh[j + 36] = k2 * c + k1 * s;
                }
            }
        }

        /* Non-causal bidirectional multi-head attention */
        int32_t h_idx;
        #pragma omp parallel for schedule(dynamic, 1)
        for (h_idx = 0; h_idx < n_heads; h_idx++) {
            float* scores = (float*)malloc((size_t)n_patches * sizeof(float));
            for (int32_t i = 0; i < n_patches; i++) {
                const float* qi = qkv_buf + (size_t)i * (size_t)(v_dim * 3) + h_idx * hd;
                float max_sc = -1e30f;
                for (int32_t j = 0; j < n_patches; j++) {
                    const float* kj = qkv_buf + (size_t)j * (size_t)(v_dim * 3) + v_dim + h_idx * hd;
                    float dot = 0.0f;
                    for (int d = 0; d < hd; d++) dot += qi[d] * kj[d];
                    float sc = dot * attn_scale;
                    scores[j] = sc;
                    if (sc > max_sc) max_sc = sc;
                }
                float sum_exp = 0.0f;
                for (int32_t j = 0; j < n_patches; j++) {
                    float ev = expf(scores[j] - max_sc);
                    scores[j] = ev;
                    sum_exp += ev;
                }
                float inv_sum = 1.0f / (sum_exp + 1e-12f);
                float* out_i = attn_buf + (size_t)i * (size_t)v_dim + h_idx * hd;
                for (int d = 0; d < hd; d++) out_i[d] = 0.0f;
                for (int32_t j = 0; j < n_patches; j++) {
                    float w = scores[j] * inv_sum;
                    const float* vj = qkv_buf + (size_t)j * (size_t)(v_dim * 3) + 2 * v_dim + h_idx * hd;
                    for (int d = 0; d < hd; d++) out_i[d] += w * vj[d];
                }
            }
            free(scores);
        }

        /* Proj + residual add + Norm2 + MLP */
        #pragma omp parallel for schedule(static)
        for (p = 0; p < n_patches; p++) {
            float* h_p = hidden + (size_t)p * (size_t)v_dim;
            float* attn_p = attn_buf + (size_t)p * (size_t)v_dim;
            float proj_out[1152];
            vis_matmul_vec(&vb->w_proj, attn_p, vb->proj_b, proj_out);
            for (int d = 0; d < v_dim; d++) h_p[d] += proj_out[d];

            float* norm2_p = norm_buf + (size_t)p * (size_t)v_dim;
            vis_layer_norm_f32(v_dim, h_p, vb->norm2_w, vb->norm2_b, eps, norm2_p);

            float* ffn1_p = ffn1_buf + (size_t)p * 4352u;
            vis_matmul_vec(&vb->w_fc1, norm2_p, vb->fc1_b, ffn1_p);
            for (int d = 0; d < 4304; d++) ffn1_p[d] = vis_gelu_pytorch_tanh(ffn1_p[d]);
            for (int d = 4304; d < 4352; d++) ffn1_p[d] = 0.0f; /* zero-pad */

            float fc2_out[1152];
            vis_matmul_vec(&vb->w_fc2, ffn1_p, vb->fc2_b, fc2_out);
            for (int d = 0; d < v_dim; d++) h_p[d] += fc2_out[d];
        }

        /* DeepStack Merger check for layers 8, 16, 24 */
        for (int ds = 0; ds < 3; ds++) {
            if (cfg->deepstack_visual_indexes[ds] == l) {
                const DpufPatchMerger* pm = &vm->deepstack[ds];
                #pragma omp parallel for schedule(static)
                for (t = 0; t < llm_tokens; t++) {
                    const float* blk_in = hidden + (size_t)t * (size_t)merge_dim;
                    float blk_norm[4608];
                    /* DeepStack uses postshuffle norm */
                    vis_layer_norm_f32(merge_dim, blk_in, pm->norm_w, pm->norm_b, eps, blk_norm);
                    float fc1_out[4608];
                    vis_matmul_vec(&pm->w_fc1, blk_norm, pm->fc1_b, fc1_out);
                    for (int d = 0; d < merge_dim; d++) fc1_out[d] = vis_gelu_exact(fc1_out[d]);
                    float* ds_dst = out->deepstack_embeds[ds] + (size_t)t * (size_t)out_dim;
                    vis_matmul_vec(&pm->w_fc2, fc1_out, pm->fc2_b, ds_dst);
                }
            }
        }
    }

    /* 4. Final PatchMerger (use_postshuffle_norm = False) */
    const DpufPatchMerger* mm = &vm->merger;
    #pragma omp parallel for schedule(static)
    for (p = 0; p < n_patches; p++) {
        float* h_p = hidden + (size_t)p * (size_t)v_dim;
        float* n_p = norm_buf + (size_t)p * (size_t)v_dim;
        vis_layer_norm_f32(v_dim, h_p, mm->norm_w, mm->norm_b, eps, n_p);
    }

    #pragma omp parallel for schedule(static)
    for (t = 0; t < llm_tokens; t++) {
        const float* blk_in = norm_buf + (size_t)t * (size_t)merge_dim;
        float fc1_out[4608];
        vis_matmul_vec(&mm->w_fc1, blk_in, mm->fc1_b, fc1_out);
        for (int d = 0; d < merge_dim; d++) fc1_out[d] = vis_gelu_exact(fc1_out[d]);
        float* main_dst = out->main_embeds + (size_t)t * (size_t)out_dim;
        vis_matmul_vec(&mm->w_fc2, fc1_out, mm->fc2_b, main_dst);
    }

    /* Cleanup */
    free(cos_tab);
    free(sin_tab);
    free(hidden);
    free(qkv_buf);
    free(attn_buf);
    free(norm_buf);
    free(ffn1_buf);
    return 0;
}
