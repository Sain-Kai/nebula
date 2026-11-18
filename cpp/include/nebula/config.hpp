#pragma once

#include <string>
#include <vector>
#include <cstddef>

namespace nebula {

struct ModelConfig {
    size_t vocab_size = 320;
    size_t d_model = 128;
    size_t n_layers = 4;
    size_t n_heads = 4;
    size_t d_ff = 512;
    size_t max_seq_len = 128;
    float layer_norm_eps = 1e-5f;

    // Mixture of Experts parameters
    bool is_moe = true;
    size_t num_experts = 4;
    size_t top_k_experts = 2;
    std::vector<size_t> moe_layers = {1, 3};

    std::string activation_fn = "gelu"; // "gelu", "relu", "swiglu"
    bool use_bias = true;
};

struct SamplingConfig {
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repetition_penalty = 1.05f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    size_t max_new_tokens = 16;
    unsigned int seed = 42;
};

struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8765;
    int stream_delay_ms = 40;
};

struct NebulaConfig {
    ModelConfig model;
    SamplingConfig sampling;
    ServerConfig server;
};

} // namespace nebula
