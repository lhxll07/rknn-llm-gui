// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "image_enc.h"
#include "audio_enc.h"
#include "audio_utils.h"
#include "rkllm.h"

using namespace std;
LLMHandle llmHandle = nullptr;

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

void exit_handler(int signal)
{
    if (llmHandle != nullptr)
    {
        {
            cout << "程序即将退出" << endl;
            LLMHandle _tmp = llmHandle;
            llmHandle = nullptr;
            rkllm_destroy(_tmp);
        }
    }
    exit(signal);
}

int callback(RKLLMResult *result, void *userdata, LLMCallState state)
{

    if (state == RKLLM_RUN_FINISH)
    {
        printf("\n");
    }
    else if (state == RKLLM_RUN_ERROR)
    {
        printf("\\run error\n");
    }
    else if (state == RKLLM_RUN_NORMAL)
    {
        printf("%s", result->text);
        // for(int i=0; i<result->num; i++)
        // {
        //     printf("%d token_id: %d logprob: %f\n", i, result->tokens[i].id, result->tokens[i].logprob);
        // }
    }
    return 0;
}

// Expand the image into a square and fill it with the specified background color
cv::Mat expand2square(const cv::Mat& img, const cv::Scalar& background_color) {
    int width = img.cols;
    int height = img.rows;

    // If the width and height are equal, return to the original image directly
    if (width == height) {
        return img.clone();
    }

    // Calculate the new size and create a new image
    int size = std::max(width, height);
    cv::Mat result(size, size, img.type(), background_color);

    // Calculate the image paste position
    int x_offset = (size - width) / 2;
    int y_offset = (size - height) / 2;

    // Paste the original image into the center of the new image
    cv::Rect roi(x_offset, y_offset, width, height);
    img.copyTo(result(roi));

    return result;
}

static bool is_empty_arg(const char* arg)
{
    return arg == NULL || arg[0] == '\0';
}

int main(int argc, char** argv)
{
    if (argc < 10) {
        std::cerr << "Usage: " << argv[0]
                << " image_path img_encoder_model_path audio_path aud_encoder_model_path llm_model_path max_new_tokens max_context_len rknn_core_num platform "
                << "  platform: rk3588 | rk3576 | rk3562 | rv1126b\n"
                << "  Note: pass \"\" for image_path/img_encoder_model_path or audio_path/aud_encoder_model_path to disable that modality\n";
        return -1;
    }

    const char * image_path = argv[1];
    const char * img_encoder_model_path = argv[2];
    const char * audio_path = argv[3];
    const char * aud_encoder_model_path = argv[4];

    bool enable_image = !is_empty_arg(image_path) && !is_empty_arg(img_encoder_model_path);
    bool enable_audio = !is_empty_arg(audio_path) && !is_empty_arg(aud_encoder_model_path);
    if (is_empty_arg(image_path) != is_empty_arg(img_encoder_model_path)) {
        std::cerr << "[Warning] image_path and img_encoder_model_path must be both provided or both empty; image disabled\n";
        enable_image = false;
    }
    if (is_empty_arg(audio_path) != is_empty_arg(aud_encoder_model_path)) {
        std::cerr << "[Warning] audio_path and aud_encoder_model_path must be both provided or both empty; audio disabled\n";
        enable_audio = false;
    }
    printf("enable_image=%d, enable_audio=%d\n", enable_image ? 1 : 0, enable_audio ? 1 : 0);

    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[5];
    param.top_k = 1;
    param.max_new_tokens = std::atoi(argv[6]);
    param.max_context_len = std::atoi(argv[7]);
    param.skip_special_token = true;
    param.extend_param.base_domain_id = 0;

    const char* platform = argv[9];
    if (strcmp(platform, "rv1126b") == 0 || strcmp(platform, "rk3562") == 0) {
        param.extend_param.base_domain_id = 0;
    } else if (strcmp(platform, "rk3588") == 0 || strcmp(platform, "rk3576") == 0) {
        param.extend_param.base_domain_id = 1;
    } else {
        std::cerr << "Error: Unknown platform '" << platform
                  << "'. Supported: rk3588, rk3576, rk3562, rv1126b\n";
        return -1;
    }

    const char* img_start   = "<|vision_start|>";
    const char* img_end     = "<|vision_end|>";
    const char* img_content = "<|image_pad|>";
    const char* audio_start   = "<|audio>";
    const char* audio_end     = "<audio|>";
    const char* audio_content = "<|audio|>";

    // Gemma4
    // const char* img_start   = "<|image>";
    // const char* img_end     = "<image|>";
    // const char* img_content = "<|image|>";

    //DeepSeekOCR
    // img_start   = "";
    // img_end     = "";
    // img_content = "<｜▁pad▁｜>";

    if (argc > 10) img_start   = argv[10];
    if (argc > 11) img_end     = argv[11];
    if (argc > 12) img_content = argv[12];
    if (argc > 13) audio_start   = argv[13];
    if (argc > 14) audio_end     = argv[14];
    if (argc > 15) audio_content = argv[15];

    std::cerr << "[Warning] Using img_start/img_end/img_content: "
                << img_start << " , "
                << img_end << " , "
                << img_content
                << ". Please customize these values according to your model, "
                << "otherwise the output may be incorrect.\n";
    std::cerr << "[Warning] Using audio_start/audio_end/audio_content: "
            << audio_start << " , "
            << audio_end << " , "
            << audio_content
            << ". Please customize these values according to your model, "
            << "otherwise the output may be incorrect.\n";

    int ret;
    int exit_code = 0;
    std::chrono::high_resolution_clock::time_point t_start_us = std::chrono::high_resolution_clock::now();
    RKLLMCallback rkllm_callback = {};
    rkllm_callback.result_callback = callback;
    ret = rkllm_init(&llmHandle, &param, &rkllm_callback);
    if (ret == 0){
        printf("rkllm init success\n");
    } else {
        printf("rkllm init failed\n");
        exit_handler(-1);
    }
    std::chrono::high_resolution_clock::time_point t_load_end_us = std::chrono::high_resolution_clock::now();

    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
    printf("%s: LLM Model loaded in %8.2f ms\n", __func__, load_time.count() / 1000.0);

    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    const int core_num = atoi(argv[8]);

    float* img_vec = NULL;
    size_t n_image_tokens = 0;
    size_t image_width = 0;
    size_t image_height = 0;

    float* audio_vec = NULL;
    size_t n_audio_tokens = 0;
    vector<string> pre_input;

    if (enable_image) {
        t_start_us = std::chrono::high_resolution_clock::now();
        ret = init_imgenc(img_encoder_model_path, &rknn_app_ctx, core_num);
        if (ret != 0) {
            printf("init_imgenc fail! ret=%d model_path=%s\n", ret, img_encoder_model_path);
            exit_code = -1;
            goto exit_cleanup;
        }
        t_load_end_us = std::chrono::high_resolution_clock::now();

        load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
        printf("%s: ImgEnc Model loaded in %8.2f ms\n", __func__, load_time.count() / 1000.0);
            
        // The image is read in BGR format
        cv::Mat img = cv::imread(image_path);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // Expand the image into a square and fill it with the specified background color (According the modeling_minicpmv.py)
        cv::Scalar background_color(127.5, 127.5, 127.5);
        cv::Mat square_img = expand2square(img, background_color);

        // Resize the image
        image_width = rknn_app_ctx.model_width;
        image_height = rknn_app_ctx.model_height;
        cv::Mat resized_img;
        cv::Size new_size(image_width, image_height);
        cv::resize(square_img, resized_img, new_size, 0, 0, cv::INTER_LINEAR);

        n_image_tokens = rknn_app_ctx.model_image_token;
        size_t image_embed_len = rknn_app_ctx.model_embed_size;
        size_t n_embed_output = rknn_app_ctx.io_num.n_output;
        int rkllm_image_embed_len = n_image_tokens * image_embed_len * n_embed_output;
        img_vec = (float*)malloc(rkllm_image_embed_len * sizeof(float));
        if (!img_vec) {
            printf("malloc img_vec fail!\n");
            exit_code = -1;
            goto exit_cleanup;
        }
        memset(img_vec, 0, rkllm_image_embed_len * sizeof(float));

        t_start_us = std::chrono::high_resolution_clock::now();
        ret = run_imgenc(&rknn_app_ctx, resized_img.data, img_vec);
        if (ret != 0) {
            printf("run_imgenc fail! ret=%d\n", ret);
            exit_code = -1;
            goto exit_cleanup;
        }
        t_load_end_us = std::chrono::high_resolution_clock::now();
        load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
        printf("%s: ImgEnc Model inference took %8.2f ms\n", __func__, load_time.count() / 1000.0);
    }

    // ---- Audio encoder ----
    rknn_audio_context_t rknn_audio_ctx;
    memset(&rknn_audio_ctx, 0, sizeof(rknn_audio_context_t));
    audio_buffer_t audio;
    memset(&audio, 0, sizeof(audio_buffer_t));
    if (enable_audio) {
        t_start_us = std::chrono::high_resolution_clock::now();
        ret = init_audioenc(aud_encoder_model_path, &rknn_audio_ctx, core_num);
        if (ret != 0) {
            printf("init_audioenc fail! ret=%d model_path=%s\n", ret, aud_encoder_model_path);
            exit_code = -1;
            goto exit_cleanup;
        }
        t_load_end_us = std::chrono::high_resolution_clock::now();
        load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
        printf("%s: AudioEnc Model loaded in %8.2f ms\n", __func__, load_time.count() / 1000.0);

        // Read audio (miniaudio decodes and converts to mono 16 kHz float32)
        ret = read_audio(audio_path, &audio);
        if (ret != 0) {
            printf("read_audio fail! ret=%d wav_path=%s\n", ret, audio_path);
            exit_code = -1;
            goto exit_cleanup;
        }
        printf("audio: num_frames=%d, num_channels=%d, sample_rate=%d\n",
            audio.num_frames, audio.num_channels, audio.sample_rate);

        // Mel spectrogram preprocessing
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

        // Audio encoder inference
        size_t n_audio_tokens_total = rknn_audio_ctx.model_audio_token;
        size_t audio_embed_size     = rknn_audio_ctx.model_embed_size;
        size_t n_audio_output       = rknn_audio_ctx.io_num.n_output;
        printf("audio_embed_size=%zu, n_audio_output=%zu, total_embed_dim=%zu\n",
            audio_embed_size, n_audio_output, audio_embed_size * n_audio_output);

        int rkllm_audio_embed_len = n_audio_tokens_total * audio_embed_size * n_audio_output;
        audio_vec = (float*)malloc(rkllm_audio_embed_len * sizeof(float));
        if (!audio_vec) {
            printf("malloc audio_vec fail!\n");
            exit_code = -1;
            goto exit_cleanup;
        }
        memset(audio_vec, 0, rkllm_audio_embed_len * sizeof(float));

        t_start_us = std::chrono::high_resolution_clock::now();
        // RKNN audio model inputs are float16; convert float32 mel/mask to half.
        std::vector<uint16_t> padded_feature_fp16, padded_mask_fp16;
        vec_fp32_to_fp16(padded_feature, padded_feature_fp16);
        vec_fp32_to_fp16(padded_mask, padded_mask_fp16);
        ret = run_audioenc(&rknn_audio_ctx, padded_feature_fp16.data(), padded_mask_fp16.data(), audio_vec);
        if (ret != 0) {
            printf("run_audioenc fail! ret=%d\n", ret);
            exit_code = -1;
            goto exit_cleanup;
        }
        t_load_end_us = std::chrono::high_resolution_clock::now();
        load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
        printf("%s: AudioEnc Model inference took %8.2f ms\n", __func__, load_time.count() / 1000.0);

        // Number of valid audio tokens after the encoder's 4x temporal subsampling
        n_audio_tokens = ((actual_len + 1) / 2 + 1) / 2;
        if (n_audio_tokens > n_audio_tokens_total) {
            n_audio_tokens = n_audio_tokens_total;
        }
        if (n_audio_tokens == 0) {
            printf("[Warning]: audio too short/empty: %d samples -> 0 valid audio tokens; disabling audio input\n", audio.num_frames);
        }
        printf("audio: valid mel frames=%d / %d, valid audio tokens=%zu / %zu\n",
            actual_len, n_frame, n_audio_tokens, n_audio_tokens_total);
    }
    
    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));

    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));

    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = 0;
    // rkllm_set_chat_template(llmHandle, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n");

    if (enable_image) {
        pre_input.push_back("<image>这张图片中有什么？");
        pre_input.push_back("<image>What is in the image?");
    }
    if (enable_audio && n_audio_tokens != 0) {
        pre_input.push_back("<audio>这段音频说了什么？");
        pre_input.push_back("<audio>What did the audio say?");
    }
    cout << "\n**********************可输入以下问题对应序号获取回答/或自定义输入********************\n"
         << endl;
    for (int i = 0; i < (int)pre_input.size(); i++)
    {
        cout << "[" << i << "] " << pre_input[i] << endl;
    }
    cout << "\n*************************************************************************\n"
         << endl;

    while(true) {
        std::string input_str;
        printf("\n");
        printf("user: ");
        std::getline(std::cin, input_str);
        if (input_str == "exit")
        {
            break;
        }
        if (input_str == "clear")
        {
            ret = rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
            if (ret != 0)
            {
                printf("clear kv cache failed!\n");
            }
            continue;
        }
        for (int i = 0; i < (int)pre_input.size(); i++)
        {
            if (input_str == to_string(i))
            {
                input_str = pre_input[i];
                cout << input_str << endl;
            }
        }

        bool has_image = enable_image && (input_str.find("<image>") != std::string::npos);
        bool has_audio = enable_audio && (input_str.find("<audio>") != std::string::npos);
        if (!has_audio && !has_image)
        {
            rkllm_input.input_type = RKLLM_INPUT_PROMPT;
            rkllm_input.role = "user";
            rkllm_input.prompt_input = (char*)input_str.c_str();
        } else {
            rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
            rkllm_input.role = "user";
            rkllm_input.multimodal_input.prompt = (char*)input_str.c_str();
            if (has_image && enable_image) {
                rkllm_input.multimodal_input.image.image_embed = img_vec;
                rkllm_input.multimodal_input.image.n_image_tokens = n_image_tokens;
                rkllm_input.multimodal_input.image.n_image = 1;
                rkllm_input.multimodal_input.image.image_start = img_start;
                rkllm_input.multimodal_input.image.image_end = img_end;
                rkllm_input.multimodal_input.image.image_content = img_content;
                rkllm_input.multimodal_input.image.image_height = image_height;
                rkllm_input.multimodal_input.image.image_width = image_width;
            }
            if (has_audio && enable_audio) {
                rkllm_input.multimodal_input.audio.audio_embed = audio_vec;
                rkllm_input.multimodal_input.audio.n_audio_tokens = n_audio_tokens;
                rkllm_input.multimodal_input.audio.n_audio = 1;
                rkllm_input.multimodal_input.audio.audio_start = audio_start;
                rkllm_input.multimodal_input.audio.audio_end = audio_end;
                rkllm_input.multimodal_input.audio.audio_content = audio_content;
            }
        }
        printf("robot: ");
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
    }

exit_cleanup:
    if (enable_image) {
        ret = release_imgenc(&rknn_app_ctx);
        if (ret != 0) {
            printf("release_imgenc fail! ret=%d\n", ret);
        }
        if (img_vec) {
            free(img_vec);
            img_vec = NULL;
        }
    }
    if (enable_audio) {
        ret = release_audioenc(&rknn_audio_ctx);
        if (ret != 0) {
            printf("release_audioenc fail! ret=%d\n", ret);
        }
        if (audio_vec) {
            free(audio_vec);
            audio_vec = NULL;
        }
        if (audio.data) {
            free(audio.data);
            audio.data = NULL;
        }
    }
    rkllm_destroy(llmHandle);

    return exit_code;
}