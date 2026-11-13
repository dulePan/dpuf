/*
 * dpuf_vision.h — Vision-Language ViT Encoder, DeepStack & M-RoPE for DPUF
 * Copyright (c) 2026 D. P. (dule_pan). Licensed under DPUF Community & Enterprise License.
 */
#ifndef DPUF_VISION_H
#define DPUF_VISION_H

#include <stdint.h>
#include "dpuf_kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int32_t depth;
    int32_t hidden_size;
    int32_t intermediate_size;
    int32_t num_heads;
    int32_t head_dim;
    int32_t in_channels;
    int32_t patch_size;
    int32_t spatial_merge_size;
    int32_t temporal_patch_size;
    int32_t out_hidden_size;
    int32_t num_position_embeddings;
    int32_t deepstack_visual_indexes[3];
    float rope_theta;
    float layer_norm_eps;
    float image_mean[3];
    float image_std[3];
    int32_t min_pixels;
    int32_t max_pixels;
} DpufVisionConfig;

typedef struct {
    const float* norm1_w;
    const float* norm1_b;
    const float* norm2_w;
    const float* norm2_b;
    DpufWeightDesc w_qkv;
    const float* qkv_b;
    DpufWeightDesc w_proj;
    const float* proj_b;
    DpufWeightDesc w_fc1;
    const float* fc1_b;
    DpufWeightDesc w_fc2;
    const float* fc2_b;
} DpufVisionBlock;

typedef struct {
    const float* norm_w;
    const float* norm_b;
    DpufWeightDesc w_fc1;
    const float* fc1_b;
    DpufWeightDesc w_fc2;
    const float* fc2_b;
} DpufPatchMerger;

typedef struct {
    DpufVisionConfig cfg;
    DpufWeightDesc w_patch_embed;
    const float* patch_embed_b;
    const float* pos_embed;
    DpufVisionBlock* blocks;
    DpufPatchMerger merger;
    DpufPatchMerger deepstack[3];
} DpufVisionModel;

typedef struct {
    int32_t llm_tokens;
    int32_t grid_h;
    int32_t grid_w;
    float* main_embeds;         /* [llm_tokens, out_hidden_size] */
    float* deepstack_embeds[3]; /* [llm_tokens, out_hidden_size] each */
    int32_t* pos_t;             /* [llm_tokens] */
    int32_t* pos_h;             /* [llm_tokens] */
    int32_t* pos_w;             /* [llm_tokens] */
} DpufVisionOutput;

DPUF_EXPORT int32_t dpuf_vision_smart_resize(
    int32_t in_w, int32_t in_h, const DpufVisionConfig* cfg,
    int32_t* out_w, int32_t* out_h
);

DPUF_EXPORT float* dpuf_vision_load_image(
    const char* image_path, const DpufVisionConfig* cfg,
    int32_t* out_w, int32_t* out_h
);

DPUF_EXPORT void dpuf_vision_free_image(float* img_rgb_norm);

DPUF_EXPORT int32_t dpuf_vision_forward(
    const DpufVisionModel* vm,
    const float* img_rgb_norm,
    int32_t w, int32_t h,
    int32_t start_pos,
    DpufVisionOutput* out
);

DPUF_EXPORT void dpuf_vision_free_output(DpufVisionOutput* out);

#ifdef __cplusplus
}
#endif

#endif /* DPUF_VISION_H */
