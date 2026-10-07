// Greedy text inference on RKLLM Runtime 1.3.1, with all token events captured.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
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
    if ((state == RKLLM_RUN_NORMAL || state == RKLLM_RUN_WAITING) && result) {
        if (result->token_id >= 0) capture.tokens.push_back(result->token_id);
        if (result->text) std::cout << result->text << std::flush;
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " model.rkllm context vocab_size prompt-or-tokens:ID,ID\n";
        return 2;
    }
    const int context = std::atoi(argv[2]);
    const int vocab = std::atoi(argv[3]);
    if (context < 128 || context > 16384 || vocab < 1) return 2;
    const std::string argument = argv[4];
    const bool token_mode = argument.rfind("tokens:", 0) == 0;
    std::vector<int32_t> input_ids;
    if (token_mode) {
        std::istringstream source(argument.substr(7));
        std::string part;
        while (std::getline(source, part, ',')) {
            char* end = nullptr;
            long id = std::strtol(part.c_str(), &end, 10);
            if (part.empty() || *end || id < 0 || id >= vocab) return 2;
            input_ids.push_back(static_cast<int32_t>(id));
        }
        if (input_ids.empty() || input_ids.size() + 64 > static_cast<std::size_t>(context)) return 2;
    }
    Capture capture;
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1];
    param.max_context_len = context;
    param.max_new_tokens = 64;
    param.top_k = 1;
    param.top_p = 1;
    param.temperature = 1;
    param.repeat_penalty = 1;
    param.frequency_penalty = 0;
    param.presence_penalty = 0;
    param.mirostat = 0;
    param.skip_special_token = true;
    param.ignore_eos_token = false;
    param.is_async = false;
    param.extend_param.n_batch = 1;
    RKLLMCallback callbacks = {};
    callbacks.result_callback = callback;
    callbacks.result_userdata = &capture;
    LLMHandle handle = nullptr;
    int init = rkllm_init(&handle, &param, &callbacks);
    if (init != 0 || !handle) {
        std::cerr << "rkllm_init failed: " << init << '\n';
        if (handle) rkllm_destroy(handle);
        return 1;
    }
    RKLLMInput input = {};
    if (token_mode) {
        // IDs already contain the complete serialized chat template.
        if (rkllm_set_chat_template(handle, "", "", "") != 0) {
            rkllm_destroy(handle);
            return 1;
        }
        input.input_type = RKLLM_INPUT_TOKEN;
        input.token_input.input_ids = input_ids.data();
        input.token_input.n_tokens = static_cast<int>(input_ids.size());
    } else {
        input.input_type = RKLLM_INPUT_PROMPT;
        input.prompt_input = argv[4];
    }
    input.role = "user";
    RKLLMInferParam infer = {};
    infer.mode = RKLLM_INFER_GENERATE;
    infer.keep_history = 0;
    infer.max_new_tokens = 64;
    int status = rkllm_run(handle, &input, &infer, &capture);
    int destroy = rkllm_destroy(handle);
    bool valid = status == 0 && destroy == 0 && capture.finished && !capture.error
                 && !capture.tokens.empty() && capture.tokens.size() <= 64;
    for (int32_t id : capture.tokens) if (id < 0 || id >= vocab) valid = false;
    std::cout << "\nTEXT_PROBE_RESULT={\"valid\":" << (valid ? "true" : "false")
              << ",\"token_input\":" << (token_mode ? "true" : "false")
              << ",\"run\":" << status << ",\"destroy\":" << destroy
              << ",\"finished\":" << (capture.finished ? "true" : "false")
              << ",\"tokens\":[";
    for (std::size_t i = 0; i < capture.tokens.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << capture.tokens[i];
    }
    std::cout << "]}\n";
    return valid ? 0 : 1;
}
