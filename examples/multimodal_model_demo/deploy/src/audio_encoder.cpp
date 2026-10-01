// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "audio_enc.h"
#include "audio_utils.h"

// IEEE-754 binary16 (half) conversion for the RKNN audio encoder inputs.
// The RKNN model io tensors are float16; feeding float32 bytes while declaring
// RKNN_TENSOR_FLOAT16 causes the runtime to misinterpret the data.
static inline uint16_t fp32_to_fp16(float f)
{
    union { float f; uint32_t u; } v = { f };
    uint32_t u = v.u;
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t exp = (int32_t)((u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = u & 0x7fffff;
    if (exp >= 31) {
        // overflow -> Inf
        return (uint16_t)(sign | 0x7c00);
    }
    if (exp <= 0) {
        if (exp < -10) {
            return (uint16_t)sign;                       // underflow -> 0
        }
        mant |= 0x800000;                                // implicit leading 1
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1);
        uint32_t round = (rem > (1u << (shift - 1))) ||
                         ((rem == (1u << (shift - 1))) && (half_mant & 1));
        return (uint16_t)(sign | (half_mant + round));
    }
    uint32_t half_mant = mant >> 13;
    uint32_t rem = mant & 0x1fff;
    uint32_t round = (rem > 0x1000) || ((rem == 0x1000) && (half_mant & 1));
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (half_mant + round));
}

static inline void vec_fp32_to_fp16(const std::vector<float>& src, std::vector<uint16_t>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = fp32_to_fp16(src[i]);
    }
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage:\n"
                << "  " << argv[0]
                << " <model_path> <wav_path> <core_num>\n\n"
                << "Arguments:\n"
                << "  model_path       Path to the audio encoder model (e.g., ./gemma4_audio_rk3588.rknn)\n"
                << "  wav_path         Path to the input wav file (decoded to 16kHz mono automatically)\n"
                << "  core_num         Number of NPU cores to use (e.g., 1/2/3)\n";
        return -1;
    }

    const char* model_path = argv[1];
    const char* wav_path   = argv[2];
    const int   core_num   = atoi(argv[3]);

    int ret;
    rknn_audio_context_t rknn_audio_ctx;
    memset(&rknn_audio_ctx, 0, sizeof(rknn_audio_context_t));

    std::chrono::high_resolution_clock::time_point t_start_us = std::chrono::high_resolution_clock::now();

    ret = init_audioenc(model_path, &rknn_audio_ctx, core_num);
    if (ret != 0) {
        printf("init_audioenc fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }
    std::chrono::high_resolution_clock::time_point t_load_end_us = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
    printf("%s: Model loaded in %8.2f ms\n", __func__, load_time.count() / 1000.0);

    // Read audio (miniaudio decodes and converts to mono 16 kHz float32)
    audio_buffer_t audio;
    memset(&audio, 0, sizeof(audio_buffer_t));
    ret = read_audio(wav_path, &audio);
    if (ret != 0) {
        printf("read_audio fail! ret=%d wav_path=%s\n", ret, wav_path);
        release_audioenc(&rknn_audio_ctx);
        return -1;
    }
    printf("audio: num_frames=%d, num_channels=%d, sample_rate=%d\n",
           audio.num_frames, audio.num_channels, audio.sample_rate);

    // Preprocess: mel spectrogram -> model input
    int n_mels  = rknn_audio_ctx.model_mel_bins;
    int n_frame = rknn_audio_ctx.model_max_frames;
    const int n_freqs = GEMMA4_FFT_LENGTH / 2 + 1;

    std::vector<float> mel_filters((size_t)n_freqs * (size_t)n_mels);
    build_gemma4_mel_filters(mel_filters.data(), n_freqs, n_mels,
                             GEMMA4_F_MIN, GEMMA4_F_MAX,
                             GEMMA4_SAMPLE_RATE, GEMMA4_FFT_LENGTH);

    std::vector<float> padded_feature((size_t)n_frame * (size_t)n_mels, 0.0f);
    int actual_len = 0;
    gemma4_audio_preprocess(&audio, mel_filters.data(), n_mels, n_frame,
                            padded_feature, &actual_len);

    // Build mask: 1 for valid frames, 0 for silence-padded frames
    std::vector<float> padded_mask((size_t)n_frame, 0.0f);
    for (int i = 0; i < std::min(actual_len, n_frame); i++) {
        padded_mask[i] = 1.0f;
    }

    // Build output buffer
    int n_audio_tokens = rknn_audio_ctx.model_audio_token;
    int embed_size     = rknn_audio_ctx.model_embed_size;
    int n_output       = rknn_audio_ctx.io_num.n_output;
    int audio_vec_len  = n_audio_tokens * embed_size * n_output;
    float* audio_vec = (float*)malloc(audio_vec_len * sizeof(float));
    if (!audio_vec) {
        printf("malloc audio_vec fail!\n");
        free(audio.data);
        release_audioenc(&rknn_audio_ctx);
        return -1;
    }
    memset(audio_vec, 0, audio_vec_len * sizeof(float));

    std::chrono::high_resolution_clock::time_point t_every_begin_us = std::chrono::high_resolution_clock::now();
    // RKNN audio model inputs are float16; convert float32 mel/mask to half.
    std::vector<uint16_t> padded_feature_fp16, padded_mask_fp16;
    vec_fp32_to_fp16(padded_feature, padded_feature_fp16);
    vec_fp32_to_fp16(padded_mask, padded_mask_fp16);
    ret = run_audioenc(&rknn_audio_ctx, padded_feature_fp16.data(), padded_mask_fp16.data(), audio_vec);
    if (ret != 0) {
        printf("run_audioenc fail! ret=%d\n", ret);
    }
    std::chrono::high_resolution_clock::time_point t_every_end_us = std::chrono::high_resolution_clock::now();
    auto encoder_time = std::chrono::duration_cast<std::chrono::microseconds>(t_every_end_us - t_every_begin_us);
    printf("%s: Encoder the audio cost %8.2f ms\n", __func__, encoder_time.count() / 1000.0);

    // Writes the array audio_vec to the file
    std::ofstream file("./audio_vec.bin", std::ios::binary);
    file.write(reinterpret_cast<char*>(audio_vec), audio_vec_len * sizeof(float));
    file.close();

    free(audio_vec);
    free(audio.data);

    ret = release_audioenc(&rknn_audio_ctx);
    if (ret != 0) {
        printf("release_audioenc fail! ret=%d\n", ret);
    }

    return 0;
}