#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include "nebula/tensor.hpp"

namespace nebula {

struct AttentionInspection {
    Tensor query;
    Tensor key;
    Tensor value;
    Tensor attention_weights; // (batch, n_heads, seq_len, total_kv_len)
};

struct NeuronInspection {
    Tensor pre_activation;
    Tensor neuron_activations;
    float sparsity = 0.0f;
    std::vector<int> top_neuron_indices;
    std::vector<float> top_neuron_values;
};

class LayerNorm {
public:
    size_t d_model;
    float eps;
    std::vector<float> gamma;
    std::vector<float> beta;

    LayerNorm(size_t dim, float epsilon = 1e-5f)
        : d_model(dim), eps(epsilon), gamma(dim, 1.0f), beta(dim, 0.0f) {}

    Tensor forward(const Tensor& x) const {
        Tensor out = x;
        size_t batch = x.shape[0];
        size_t seq = x.shape[1];

        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq; ++s) {
                double mean = 0.0;
                for (size_t d = 0; d < d_model; ++d) {
                    mean += x.at3(b, s, d);
                }
                mean /= d_model;

                double var = 0.0;
                for (size_t d = 0; d < d_model; ++d) {
                    double diff = x.at3(b, s, d) - mean;
                    var += diff * diff;
                }
                var /= d_model;

                float inv_std = static_cast<float>(1.0 / std::sqrt(var + eps));
                for (size_t d = 0; d < d_model; ++d) {
                    float norm_val = static_cast<float>((x.at3(b, s, d) - mean) * inv_std);
                    out.at3(b, s, d) = gamma[d] * norm_val + beta[d];
                }
            }
        }
        return out;
    }
};

class MultiHeadAttention {
public:
    size_t d_model;
    size_t n_heads;
    size_t d_head;
    Tensor w_q, w_k, w_v, w_o;

    MultiHeadAttention(size_t dim, size_t heads, std::mt19937& rng)
        : d_model(dim), n_heads(heads), d_head(dim / heads),
          w_q({dim, dim}), w_k({dim, dim}), w_v({dim, dim}), w_o({dim, dim}) {
        float scale = 1.0f / std::sqrt(static_cast<float>(dim));
        std::normal_distribution<float> dist(0.0f, scale);
        for (float& val : w_q.data) val = dist(rng);
        for (float& val : w_k.data) val = dist(rng);
        for (float& val : w_v.data) val = dist(rng);
        for (float& val : w_o.data) val = dist(rng);
    }

    Tensor forward(
        const Tensor& x,
        Tensor& past_k,
        Tensor& past_v,
        AttentionInspection& inspection,
        bool causal_mask = true
    ) {
        size_t batch = x.shape[0];
        size_t seq = x.shape[1];

        // Q, K, V Projections
        Tensor q_proj = Tensor::matmul3d_proj(x, w_q);
        Tensor k_proj = Tensor::matmul3d_proj(x, w_k);
        Tensor v_proj = Tensor::matmul3d_proj(x, w_v);

        // Reshape to (batch, n_heads, seq, d_head)
        Tensor q_head({batch, n_heads, seq, d_head});
        Tensor k_new({batch, n_heads, seq, d_head});
        Tensor v_new({batch, n_heads, seq, d_head});

        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq; ++s) {
                for (size_t h = 0; h < n_heads; ++h) {
                    for (size_t d = 0; d < d_head; ++d) {
                        size_t flat_idx = h * d_head + d;
                        q_head.at4(b, h, s, d) = q_proj.at3(b, s, flat_idx);
                        k_new.at4(b, h, s, d) = k_proj.at3(b, s, flat_idx);
                        v_new.at4(b, h, s, d) = v_proj.at3(b, s, flat_idx);
                    }
                }
            }
        }

        // KV Cache concatenation
        Tensor current_k, current_v;
        if (past_k.size() > 0 && past_k.shape[2] > 0) {
            size_t past_len = past_k.shape[2];
            size_t total_len = past_len + seq;
            current_k = Tensor({batch, n_heads, total_len, d_head});
            current_v = Tensor({batch, n_heads, total_len, d_head});

            for (size_t b = 0; b < batch; ++b) {
                for (size_t h = 0; h < n_heads; ++h) {
                    // Copy past
                    for (size_t s = 0; s < past_len; ++s) {
                        for (size_t d = 0; d < d_head; ++d) {
                            current_k.at4(b, h, s, d) = past_k.at4(b, h, s, d);
                            current_v.at4(b, h, s, d) = past_v.at4(b, h, s, d);
                        }
                    }
                    // Append new
                    for (size_t s = 0; s < seq; ++s) {
                        for (size_t d = 0; d < d_head; ++d) {
                            current_k.at4(b, h, past_len + s, d) = k_new.at4(b, h, s, d);
                            current_v.at4(b, h, past_len + s, d) = v_new.at4(b, h, s, d);
                        }
                    }
                }
            }
        } else {
            current_k = k_new;
            current_v = v_new;
        }

        past_k = current_k;
        past_v = current_v;
        size_t total_kv_len = current_k.shape[2];

        // Scaled dot-product attention
        float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d_head));
        Tensor attn_weights({batch, n_heads, seq, total_kv_len});

        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < n_heads; ++h) {
                for (size_t i = 0; i < seq; ++i) {
                    size_t past_offset = total_kv_len - seq;
                    float max_score = -1e9f;

                    for (size_t j = 0; j < total_kv_len; ++j) {
                        if (causal_mask && j > past_offset + i) {
                            attn_weights.at4(b, h, i, j) = -1e9f;
                            continue;
                        }
                        double dot = 0.0;
                        for (size_t d = 0; d < d_head; ++d) {
                            dot += q_head.at4(b, h, i, d) * current_k.at4(b, h, j, d);
                        }
                        float score = static_cast<float>(dot * inv_sqrt_d);
                        attn_weights.at4(b, h, i, j) = score;
                        if (score > max_score) max_score = score;
                    }

                    // Softmax across j
                    double exp_sum = 0.0;
                    for (size_t j = 0; j < total_kv_len; ++j) {
                        if (causal_mask && j > past_offset + i) {
                            attn_weights.at4(b, h, i, j) = 0.0f;
                        } else {
                            float exp_s = std::exp(attn_weights.at4(b, h, i, j) - max_score);
                            attn_weights.at4(b, h, i, j) = exp_s;
                            exp_sum += exp_s;
                        }
                    }
                    float inv_exp_sum = exp_sum > 0.0 ? static_cast<float>(1.0 / exp_sum) : 0.0f;
                    for (size_t j = 0; j < total_kv_len; ++j) {
                        attn_weights.at4(b, h, i, j) *= inv_exp_sum;
                    }
                }
            }
        }

        // Weighted combination with V -> (batch, seq, d_model)
        Tensor head_out_concat({batch, seq, d_model}, 0.0f);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq; ++s) {
                for (size_t h = 0; h < n_heads; ++h) {
                    for (size_t d = 0; d < d_head; ++d) {
                        double acc = 0.0;
                        for (size_t j = 0; j < total_kv_len; ++j) {
                            acc += attn_weights.at4(b, h, s, j) * current_v.at4(b, h, j, d);
                        }
                        size_t flat_idx = h * d_head + d;
                        head_out_concat.at3(b, s, flat_idx) = static_cast<float>(acc);
                    }
                }
            }
        }

        // Final output projection W_O
        Tensor out = Tensor::matmul3d_proj(head_out_concat, w_o);

        inspection.query = q_head;
        inspection.key = current_k;
        inspection.value = current_v;
        inspection.attention_weights = attn_weights;

        return out;
    }
};

class FeedForward {
public:
    size_t d_model;
    size_t d_ff;
    Tensor w_gate;
    Tensor w_down;

    FeedForward(size_t dim, size_t ff_dim, std::mt19937& rng)
        : d_model(dim), d_ff(ff_dim), w_gate({dim, ff_dim}), w_down({ff_dim, dim}) {
        float scale1 = 1.0f / std::sqrt(static_cast<float>(dim));
        float scale2 = 1.0f / std::sqrt(static_cast<float>(ff_dim));
        std::normal_distribution<float> dist1(0.0f, scale1);
        std::normal_distribution<float> dist2(0.0f, scale2);
        for (float& val : w_gate.data) val = dist1(rng);
        for (float& val : w_down.data) val = dist2(rng);
    }

    Tensor forward(const Tensor& x, NeuronInspection& inspection) const {
        Tensor pre_act = Tensor::matmul3d_proj(x, w_gate);
        Tensor activated = pre_act;
        activated.gelu_();

        Tensor out = Tensor::matmul3d_proj(activated, w_down);

        // Analyze neuron activations for the last token in sequence
        size_t b = 0, s = x.shape[1] - 1;
        std::vector<std::pair<float, int>> neuron_pairs;
        neuron_pairs.reserve(d_ff);
        size_t inactive_count = 0;

        for (size_t d = 0; d < d_ff; ++d) {
            float val = activated.at3(b, s, d);
            if (std::abs(val) < 0.05f) inactive_count++;
            neuron_pairs.push_back({std::abs(val), static_cast<int>(d)});
        }

        std::sort(neuron_pairs.rbegin(), neuron_pairs.rend());

        inspection.pre_activation = pre_act;
        inspection.neuron_activations = activated;
        inspection.sparsity = (static_cast<float>(inactive_count) / d_ff) * 100.0f;
        inspection.top_neuron_indices.clear();
        inspection.top_neuron_values.clear();

        for (size_t i = 0; i < std::min<size_t>(10, neuron_pairs.size()); ++i) {
            int n_idx = neuron_pairs[i].second;
            inspection.top_neuron_indices.push_back(n_idx);
            inspection.top_neuron_values.push_back(activated.at3(b, s, n_idx));
        }

        return out;
    }
};

} // namespace nebula
