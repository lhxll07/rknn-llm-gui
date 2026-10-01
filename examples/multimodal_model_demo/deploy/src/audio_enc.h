#include "rknn_api.h"

#ifndef _RKNN_AUDIO_ENC_H_
#define _RKNN_AUDIO_ENC_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_max_frames;
    int model_mel_bins;
    int model_audio_token;
    int model_embed_size;
} rknn_audio_context_t;

int init_audioenc(const char* model_path, rknn_audio_context_t* app_ctx, const int core_num);

int release_audioenc(rknn_audio_context_t* app_ctx);

int run_audioenc(rknn_audio_context_t* app_ctx, void* feature_data, void* mask_data, float* out_result);

#ifdef __cplusplus
}
#endif

#endif //_RKNN_AUDIO_ENC_H_