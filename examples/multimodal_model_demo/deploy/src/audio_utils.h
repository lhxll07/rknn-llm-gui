#ifndef _AUDIO_UTILS_H_
#define _AUDIO_UTILS_H_

#include <vector>

/**
 * @brief Audio buffer
 */
typedef struct
{
    float *data;
    int num_frames;
    int num_channels;
    int sample_rate;
} audio_buffer_t;

// Gemma-4 audio preprocessing parameters, mirroring HuggingFace
// Gemma4AudioFeatureExtractor (transformers/models/gemma4/feature_extraction_gemma4.py).
#define GEMMA4_SAMPLE_RATE       16000
#define GEMMA4_HOP_LENGTH        160     // 10 ms @ 16 kHz
#define GEMMA4_FRAME_LENGTH      320     // 20 ms @ 16 kHz
#define GEMMA4_FFT_LENGTH        512     // 2^ceil(log2(320)) = 512
#define GEMMA4_F_MIN             0.0f
#define GEMMA4_F_MAX             8000.0f
#define GEMMA4_MEL_FLOOR         1e-3f
#define GEMMA4_PAD_TO_MULTIPLE   128
#define GEMMA4_SEMI_CAUSAL_LEFT  (GEMMA4_FRAME_LENGTH / 2)  // = 160

/**
 * @brief Reads an audio file into a buffer.
 *
 * Decodes the file with miniaudio and converts it to mono 16 kHz float32.
 *
 * @param path [in] Path to the audio file.
 * @param audio [out] Pointer to the audio buffer structure that will store the read data.
 * @return int 0 on success, -1 on error.
 */
int read_audio(const char *path, audio_buffer_t *audio);

/**
 * @brief Builds a USM-style mel filterbank matrix in row-major [n_freqs, n_mels] layout.
 */
void build_gemma4_mel_filters(float* fb,
                              int n_freqs,
                              int n_mels,
                              float f_min,
                              float f_max,
                              int sample_rate,
                              int fft_length);

/**
 * @brief Gemma-4 log-mel preprocessing.
 *
 * Output layout: padded_feature[t * n_mels + m] (time-major, n_mels contiguous),
 * matching the RKNN model input shape [1, n_frame, n_mels].
 */
void gemma4_audio_preprocess(audio_buffer_t* audio,
                             const float* mel_filters,
                             int n_mels,
                             int n_frame,
                             std::vector<float>& padded_feature,
                             int* actual_len);

#endif // _AUDIO_UTILS_H_