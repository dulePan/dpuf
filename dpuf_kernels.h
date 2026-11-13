#ifndef DPUF_KERNELS_H
#define DPUF_KERNELS_H

#include <stdint.h>

#ifdef _WIN32
#define DPUF_EXPORT __declspec(dllexport)
#else
#define DPUF_EXPORT __attribute__((visibility("default")))
#endif

enum {
    DPUF_MODE_QR_ZD4         = 8,
    DPUF_MODE_QR_SPLIT4      = 9,
    DPUF_MODE_QR_HAD4        = 10,
    DPUF_MODE_QR_Q8          = 11,
    DPUF_MODE_QR_HAD4_PACKED = 12,
    DPUF_MODE_QR_HAD4_ASYM   = 13,
    DPUF_MODE_QR_HAD6        = 14
};

enum {
    DPUF_KV_F32 = 0,
    DPUF_KV_Q8  = 1
};

#pragma pack(push, 1)
typedef struct {
    uint32_t tile_crc32;
    uint16_t n_rows;
    uint16_t flags;
    uint32_t n_over8;     /* Mode 8: over8 count | Mode 9: sparse outlier count | Mode 10: reserved */
    uint32_t n_super16;   /* Mode 8 & 9: int16/uint16 super-outlier count | Mode 10: reserved */
} DpufQrTileHdr;          /* 16 bytes */
#pragma pack(pop)

typedef struct {
    int32_t mode;               /* 8, 9, or 10 */
    int32_t rows;
    int32_t cols;
    int32_t n_tiles;
    const uint8_t* tensor_base;
    const uint64_t* tile_offsets;
    const float* col_scale;     /* Mode 8 & 9: float[cols], else NULL */
    const uint8_t* sign_bits;   /* Mode 10: uint8[cols/8], else NULL */
} DpufWeightDesc;               /* 48 bytes */

typedef struct {
    int32_t hidden_size;
    int32_t intermediate_size;
    int32_t num_layers;
    int32_t num_q_heads;
    int32_t num_kv_heads;
    int32_t head_dim;
    int32_t vocab_size;
    int32_t max_seq_len;
    int32_t has_qk_norm;
    float rms_eps;
    float rope_theta;
    int32_t n_threads;
    int32_t use_llama3_rope;
    float rope_factor;
    float rope_low_freq;
    float rope_high_freq;
    int32_t rope_orig_max_pos;
    int32_t kv_type;            /* 0 = FP32 (default), 1 = Q8_HEAD (4x smaller RAM & faster bandwidth) */
    int32_t has_mrope;          /* 1 = Qwen3-VL Interleaved 3D M-RoPE */
    int32_t mrope_section[3];   /* e.g. [24, 20, 20] */
    int32_t mrope_interleaved;  /* 1 = interleaved */
} DpufModelConfig;              /* 92 bytes */

typedef struct {
    const float* attn_norm;
    const float* attn_q_norm;
    const float* attn_k_norm;
    const float* ffn_norm;
    DpufWeightDesc w_q;
    DpufWeightDesc w_k;
    DpufWeightDesc w_v;
    DpufWeightDesc w_o;
    DpufWeightDesc w_gate;
    DpufWeightDesc w_up;
    DpufWeightDesc w_down;
} DpufLayerWeights;             /* 368 bytes */

#ifdef __cplusplus
extern "C" {
#endif

DPUF_EXPORT uint32_t dpuf_crc32_bytes(const uint8_t* data, uint32_t len);

DPUF_EXPORT int32_t dpuf_qr_verify_tiles(
    int32_t n_tiles,
    int32_t cols,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    uint8_t* tile_ok_out
);

DPUF_EXPORT int32_t dpuf_qr_verify_tiles_ex(
    int32_t mode,
    int32_t n_tiles,
    int32_t cols,
    const uint8_t* tensor_base,
    const uint64_t* tile_offsets,
    uint8_t* tile_ok_out
);

/* Fast AVX2 activation prep & transform helpers */
DPUF_EXPORT void dpuf_scale_quantize_x_q8_32(
    int32_t cols,
    const float* x_in,
    const float* col_scale,   /* optional NULL -> no col_scale */
    float* x_scaled_out,      /* optional NULL if not needed */
    int8_t* x_q8_out,
    float* x_blk_scale_out
);

DPUF_EXPORT void dpuf_fwht64_sign_f32(
    int32_t cols,
    const float* x_in,
    const uint8_t* sign_bits,
    float* x_out
);

/* Tile-range kernels (safe to invoke inside an existing persistent parallel region!) */
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
);

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
);

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
);

/* Unified Mode 8/9/10 embedding lookup for arbitrary number of tokens */
void dpuf_embed_qr_any_f32(
    const DpufWeightDesc* w,
    int32_t n_tokens,
    const int32_t* token_ids,
    float* out,
    float* scratch_f32  /* at least w->cols floats for Mode 10, else can be NULL */
);

/* Standalone exported Mode 8, 9, 10 wrappers */
DPUF_EXPORT void dpuf_matmul_qr_zd4_f32(
    int32_t M, int32_t rows, int32_t cols, int32_t n_tiles,
    const float* x_in, const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const float* col_scale, float* out
);

DPUF_EXPORT void dpuf_matmul_qr_split4_f32(
    int32_t M, int32_t rows, int32_t cols, int32_t n_tiles,
    const float* x_in, const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const float* col_scale, float* out
);

DPUF_EXPORT void dpuf_embed_qr_split4_f32(
    int32_t n_tokens, int32_t cols, const int32_t* token_ids,
    const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const float* col_scale, float* out
);

DPUF_EXPORT void dpuf_matmul_qr_hadamard4_f32(
    int32_t M, int32_t rows, int32_t cols, int32_t n_tiles,
    const float* x_in, const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
);

DPUF_EXPORT void dpuf_embed_qr_hadamard4_f32(
    int32_t n_tokens, int32_t cols, const int32_t* token_ids,
    const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
);

void dpuf_matmul_qr_q8_tile_range(
    int32_t M, int32_t rows, int32_t cols, int32_t t_start, int32_t t_end,
    const int8_t* x_q8, const float* x_blk_scale,
    const uint8_t* tensor_base, const uint64_t* tile_offsets, float* out
);

DPUF_EXPORT void dpuf_matmul_qr_q8_f32(
    int32_t M, int32_t rows, int32_t cols, int32_t n_tiles,
    const float* x_in, const uint8_t* tensor_base, const uint64_t* tile_offsets, float* out
);

DPUF_EXPORT void dpuf_embed_qr_q8_f32(
    int32_t n_tokens, int32_t cols, const int32_t* token_ids,
    const uint8_t* tensor_base, const uint64_t* tile_offsets, float* out
);

/* MODE 12: Pre-packed 8x32 QR Hadamard4 + zero-wsum128 + slab X-packing */
void dpuf_prep_had4_packed_slab(
    int32_t mc,
    int32_t cols,
    const float* x_slab_f32,
    const uint8_t* sign_bits,
    float* f_scratch,           /* mc * cols floats */
    int8_t* x_out,              /* M==1: cols int8; slab(mc>=1): n_blk * 8 * 6 * 4 bytes */
    float* x_scale_out,         /* M==1: n_blk floats; slab: n_blk * 6 floats */
    int32_t* x_sum8_out,        /* M==1: n_blk int32; slab: n_blk * 6 int32 */
    int32_t pack_slab           /* 0 = plain row-major (for M==1), 1 = packed [n_blk][8][6] */
);

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
);

DPUF_EXPORT void dpuf_matmul_qr_hadamard4_packed_f32(
    int32_t M, int32_t rows, int32_t cols, int32_t n_tiles,
    const float* x_in, const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
);

DPUF_EXPORT void dpuf_embed_qr_hadamard4_packed_f32(
    int32_t n_tokens, int32_t cols, const int32_t* token_ids,
    const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
);

/* MODE 13 (QR_HAD4_ASYM, blk=64) & MODE 14 (QR_HAD6, blk=64) */
void dpuf_matmul_qr_had4_asym_tile_range(
    int32_t M, int32_t rows, int32_t cols, int32_t t_start, int32_t t_end,
    const int8_t* x_q8, const float* x_blk_scale, const int32_t* x_sum8,
    const uint8_t* tensor_base, const uint64_t* tile_offsets, float* out
);

DPUF_EXPORT void dpuf_matmul_qr_had4_asym_f32(
    int32_t M, int32_t rows, int32_t cols, int32_t n_tiles,
    const float* x_in, const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
);

DPUF_EXPORT void dpuf_embed_qr_had4_asym_f32(
    int32_t n_tokens, int32_t cols, const int32_t* token_ids,
    const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
);

void dpuf_matmul_qr_had6_tile_range(
    int32_t M, int32_t rows, int32_t cols, int32_t t_start, int32_t t_end,
    const int8_t* x_q8, const float* x_blk_scale, const int32_t* x_sum8,
    const uint8_t* tensor_base, const uint64_t* tile_offsets, float* out
);

DPUF_EXPORT void dpuf_matmul_qr_had6_f32(
    int32_t M, int32_t rows, int32_t cols, int32_t n_tiles,
    const float* x_in, const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
);

DPUF_EXPORT void dpuf_embed_qr_had6_f32(
    int32_t n_tokens, int32_t cols, const int32_t* token_ids,
    const uint8_t* tensor_base, const uint64_t* tile_offsets,
    const uint8_t* sign_bits, float* out
);

/* Persistent C Transformer Engine (Steps 1 & 2) */
DPUF_EXPORT void* dpuf_engine_create(
    const DpufModelConfig* cfg,
    const DpufLayerWeights* layers,
    const DpufWeightDesc* w_tok_embd,
    const DpufWeightDesc* w_lm_head,
    const float* output_norm
);
DPUF_EXPORT void dpuf_engine_destroy(void* handle);
DPUF_EXPORT void dpuf_engine_reset_cache(void* handle);
DPUF_EXPORT int32_t dpuf_engine_forward_batch(
    void* handle,
    int32_t n_tokens,
    const int32_t* token_ids,
    int32_t pos0,
    int32_t only_last_logits,
    float* logits_out
);
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
);
DPUF_EXPORT int32_t dpuf_engine_forward_token(
    void* handle,
    int32_t token_id,
    int32_t pos,
    float* logits_out
);

#ifdef __cplusplus
}
#endif

#endif /* DPUF_KERNELS_H */
