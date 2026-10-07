// Compatibility check for the repository's RKLLM Runtime 1.3.1 API.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "rkllm.h"

struct Capture {
    std::vector<int32_t> tokens;
    bool finished = false;
    bool error = false;
};

static int callback(RKLLMResult* result, void* userdata, LLMCallState state) {
    auto& capture = *static_cast<Capture*>(userdata);
    if (state == RKLLM_RUN_FINISH) capture.finished = true;
    if (state == RKLLM_RUN_ERROR) capture.error = true;
    // Byte-level tokenizers can generate partial UTF-8 characters. Those
    // tokens are returned with WAITING and still count toward generation.
    if ((state == RKLLM_RUN_NORMAL || state == RKLLM_RUN_WAITING)
            && result && result->token_id >= 0) {
        capture.tokens.push_back(result->token_id);
    }
    return 0;
}

static bool run(const char* path, int context, int vocab_size, Capture& capture) {
    constexpr int generated_tokens = 8;
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = path;
    param.max_context_len = context;
    param.max_new_tokens = generated_tokens;
    param.top_k = 1;
    param.top_p = 1.0f;
    param.temperature = 1.0f;
    param.repeat_penalty = 1.0f;
    param.frequency_penalty = 0.0f;
    param.presence_penalty = 0.0f;
    param.mirostat = 0;
    param.skip_special_token = false;
    param.ignore_eos_token = true;
    param.is_async = false;
    param.extend_param.n_batch = 1;

    RKLLMCallback callbacks = {};
    callbacks.result_callback = callback;
    callbacks.result_userdata = &capture;
    LLMHandle handle = nullptr;
    const int init_result = rkllm_init(&handle, &param, &callbacks);
    if (init_result != 0 || !handle) {
        std::cerr << "rkllm_init failed: " << init_result << " (" << path << ")\n";
        if (handle) rkllm_destroy(handle);
        return false;
    }

    int32_t ids[] = {4, 5};
    RKLLMInput input = {};
    input.role = "user";
    input.input_type = RKLLM_INPUT_TOKEN;
    input.token_input.input_ids = ids;
    input.token_input.n_tokens = 2;
    RKLLMInferParam infer = {};
    infer.mode = RKLLM_INFER_GENERATE;
    infer.keep_history = 0;
    infer.max_new_tokens = generated_tokens;
    const int run_result = rkllm_run(handle, &input, &infer, &capture);
    const int destroy_result = rkllm_destroy(handle);
    bool valid = run_result == 0 && destroy_result == 0 && capture.finished && !capture.error
                 && capture.tokens.size() == generated_tokens;
    for (const int32_t token : capture.tokens) {
        if (token < 0 || token >= vocab_size) valid = false;
    }
    if (!valid) {
        std::cerr << "Incomplete or invalid inference: run=" << run_result
                  << " destroy=" << destroy_result << " finished=" << capture.finished
                  << " error=" << capture.error << " tokens=" << capture.tokens.size() << '\n';
    }
    return valid;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " reference.rkllm native.rkllm max_context vocab_size\n";
        return 2;
    }
    const int context = std::atoi(argv[3]);
    const int vocab_size = std::atoi(argv[4]);
    if (context < 32 || context > 16384 || vocab_size <= 5) return 2;
    Capture reference, native;
    if (!run(argv[1], context, vocab_size, reference)) return 1;
    if (!run(argv[2], context, vocab_size, native)) return 1;
    if (reference.tokens != native.tokens) {
        std::cerr << "Reference and native token sequences differ\n";
        return 1;
    }
    std::cout << "\nNATIVE_SMOKE_RESULT={\"runtime_verified\":true,\"input_ids\":[4,5],\"tokens\":[";
    for (std::size_t i = 0; i < native.tokens.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << native.tokens[i];
    }
    std::cout << "],\"reference_tokens_identical\":true}\n";
    return 0;
}
