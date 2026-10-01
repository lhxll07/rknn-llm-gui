#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#include "pocketfft_hdronly.h"

#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "audio_utils.h"

int read_audio(const char *path, audio_buffer_t *audio)
{
    const int channels = 1;
    const int sampler_rate = 16000;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, channels, sampler_rate);

    ma_decoder decoder;
    ma_result result = ma_decoder_init_file(path, &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        fprintf(stderr, "Error: failed to open audio file (%s)\n", ma_result_description(result));
        return -1;
    }

    ma_uint64 frame_count;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        fprintf(stderr, "Error: failed to retrieve the length of the audio data (%s)\n", ma_result_description(result));
        return -1;
    }

    audio->data = (float*)malloc(frame_count * channels * sizeof(float));
    if (audio->data == NULL) {
        ma_decoder_uninit(&decoder);
        fprintf(stderr, "Error: failed to allocate memory for audio data\n");
        return -1;
    }

    ma_uint64 frames_read;
    result = ma_decoder_read_pcm_frames(&decoder, audio->data, frame_count, &frames_read);
    if (result != MA_SUCCESS) {
        free(audio->data);
        audio->data = NULL;
        ma_decoder_uninit(&decoder);
        fprintf(stderr, "Error: failed to read the frames of the audio data (%s)\n", ma_result_description(result));
        return -1;
    }

    audio->num_frames = (int)frames_read;
    audio->num_channels = channels;
    audio->sample_rate = sampler_rate;

    ma_decoder_uninit(&decoder);

    return 0;
}

// Build a USM-style mel filterbank matrix in row-major [n_freqs, n_mels] layout.
// Identical math to HF's create_fb_matrix (HTK mel scale, no Slaney normalisation).
void build_gemma4_mel_filters(float* fb,
                              int n_freqs,
                              int n_mels,
                              float f_min,
                              float f_max,
                              int sample_rate,
                              int fft_length)
{
    if (!fb) {
        printf("Error: build_gemma4_mel_filters: fb is NULL!\n");
        return;
    }
    if (n_freqs <= 0 || n_mels <= 0 || sample_rate <= 0 || fft_length <= 0) {
        printf("Error: build_gemma4_mel_filters: invalid params!\n");
        return;
    }

    std::vector<float> all_freqs(n_freqs);
    const float bin_hz = (float)sample_rate / (float)fft_length;
    for (int i = 0; i < n_freqs; i++) {
        all_freqs[i] = i * bin_hz;
    }

    const float m_min = 2595.0f * log10f(1.0f + f_min / 700.0f);
    const float m_max = 2595.0f * log10f(1.0f + f_max / 700.0f);

    std::vector<float> f_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        float m = m_min + (m_max - m_min) * (float)i / (float)(n_mels + 1);
        f_pts[i] = 700.0f * (powf(10.0f, m / 2595.0f) - 1.0f);
    }

    std::vector<float> f_diff(n_mels + 1);
    for (int i = 0; i < n_mels + 1; i++) {
        f_diff[i] = f_pts[i + 1] - f_pts[i];
    }

    for (int k = 0; k < n_freqs; k++) {
        for (int m = 0; m < n_mels; m++) {
            float down = (all_freqs[k] - f_pts[m])     / f_diff[m];
            float up   = (f_pts[m + 2] - all_freqs[k]) / f_diff[m + 1];
            float val  = std::min(down, up);
            fb[k * n_mels + m] = val < 0.0f ? 0.0f : val;
        }
    }
}

// Gemma-4 log-mel preprocessing (Gemma4AudioFeatureExtractor._extract_spectrogram
// + __call__). 
//
// Output layout: padded_feature[t * n_mels + m] (time-major, n_mels contiguous),
// matching the RKNN model input shape [1, n_frame, n_mels].
void gemma4_audio_preprocess(audio_buffer_t* audio,
                             const float* mel_filters,
                             int n_mels,
                             int n_frame,
                             std::vector<float>& padded_feature,
                             int* actual_len)
{
    const int frame_length = GEMMA4_FRAME_LENGTH;
    const int fft_length   = GEMMA4_FFT_LENGTH;
    const int hop_length   = GEMMA4_HOP_LENGTH;
    const int n_freqs      = fft_length / 2 + 1;
    const int pad_left     = GEMMA4_SEMI_CAUSAL_LEFT;
    const int pad_multiple = GEMMA4_PAD_TO_MULTIPLE;
    const float mel_floor  = GEMMA4_MEL_FLOOR;

    const int audio_length = (audio && audio->data) ? audio->num_frames : 0;

    if (!mel_filters || !actual_len) {
        printf("Error: gemma4_audio_preprocess: invalid args!\n");
        if (actual_len) *actual_len = 0;
        return;
    }

    const int L_pad128 = ((audio_length + pad_multiple - 1) / pad_multiple) * pad_multiple;
    const int L_total  = pad_left + L_pad128;

    const int min_buf  = (n_frame > 0) ? ((n_frame - 1) * hop_length + frame_length + 1) : 0;
    const int buf_size = std::max(L_total, min_buf) + 16;

    std::vector<float> padded_audio(buf_size, 0.0f);
    if (audio_length > 0) {
        std::copy(audio->data, audio->data + audio_length,
                  padded_audio.begin() + pad_left);
    }

    int n_valid = 0;
    {
        long rhs = (long)audio_length + pad_left - frame_length;
        if (rhs > 0) {
            n_valid = (int)((rhs + hop_length - 1) / hop_length);
        }
    }
    *actual_len = std::min(n_frame, n_valid);

    std::vector<float> window(frame_length);
    for (int i = 0; i < frame_length; i++) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)i / (float)frame_length));
    }

    std::vector<float> fft_in(fft_length, 0.0f);
    std::vector<std::complex<float>> fft_out(n_freqs);
    pocketfft::shape_t  fft_shape   = {static_cast<size_t>(fft_length)};
    pocketfft::stride_t stride_in   = {sizeof(float)};
    pocketfft::stride_t stride_out  = {sizeof(std::complex<float>)};

    std::vector<float> magnitude(n_freqs);

    std::fill(padded_feature.begin(), padded_feature.end(), 0.0f);

    for (int i = 0; i < n_frame; i++) {
        const int start = i * hop_length;

        for (int k = 0; k < frame_length; k++) {
            fft_in[k] = padded_audio[start + k] * window[k];
        }
        for (int k = frame_length; k < fft_length; k++) {
            fft_in[k] = 0.0f;
        }

        pocketfft::r2c(fft_shape, stride_in, stride_out, 0, true,
                       fft_in.data(), fft_out.data(), 1.0f);

        for (int k = 0; k < n_freqs; k++) {
            const float re = fft_out[k].real();
            const float im = fft_out[k].imag();
            magnitude[k] = sqrtf(re * re + im * im);
        }

        const size_t row_off = (size_t)i * (size_t)n_mels;
        const bool   valid   = (i < *actual_len);
        if (!valid) continue;
        for (int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for (int k = 0; k < n_freqs; k++) {
                sum += magnitude[k] * mel_filters[k * n_mels + m];
            }
            padded_feature[row_off + (size_t)m] = logf(sum + mel_floor);
        }
    }
}