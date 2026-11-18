#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "nebula/tensor.hpp"
#include "nebula/tokenizer.hpp"

namespace nebula {

struct CacheSnapshot {
    int step = 0;
    size_t total_cached_tokens = 0;
    size_t memory_bytes = 0;
    float memory_kb = 0.0f;
    std::vector<float> k_norms;
    std::vector<float> v_norms;
};

class KVCacheTracker {
public:
    size_t n_layers;
    size_t n_heads;
    size_t d_head;
    size_t max_seq_len;

    KVCacheTracker(size_t layers, size_t heads, size_t head_dim, size_t max_seq = 128)
        : n_layers(layers), n_heads(heads), d_head(head_dim), max_seq_len(max_seq) {}

    CacheSnapshot record_step(int step, const std::vector<std::pair<Tensor, Tensor>>& past_kv) const {
        CacheSnapshot snap;
        snap.step = step;

        if (past_kv.empty() || past_kv[0].first.size() == 0) {
            return snap;
        }

        snap.total_cached_tokens = past_kv[0].first.shape[2];
        size_t total_bytes = 0;

        for (const auto& [k, v] : past_kv) {
            size_t layer_bytes = (k.size() + v.size()) * sizeof(float);
            total_bytes += layer_bytes;
            snap.k_norms.push_back(k.norm());
            snap.v_norms.push_back(v.norm());
        }

        snap.memory_bytes = total_bytes;
        snap.memory_kb = static_cast<float>(total_bytes) / 1024.0f;
        return snap;
    }
};

struct TokenCandidate {
    int token_id;
    std::string token_str;
    float probability;
    float logit;
};

struct LayerPrediction {
    size_t layer_idx;
    std::string layer_name;
    float entropy;
    TokenCandidate top_prediction;
    std::vector<TokenCandidate> top_k_candidates;
    float norm;
};

class LogitLens {
public:
    Tensor w_unembed;
    NebulaTokenizer tokenizer;

    explicit LogitLens(const Tensor& unembed) : w_unembed(unembed) {}

    std::vector<LayerPrediction> analyze_residual_stream(
        const std::vector<Tensor>& residual_streams,
        size_t top_k = 5
    ) const {
        std::vector<LayerPrediction> predictions;
        size_t vocab_size = w_unembed.shape[1];

        for (size_t l = 0; l < residual_streams.size(); ++l) {
            const Tensor& h = residual_streams[l];
            size_t b = 0, s = h.shape[1] - 1;

            // Project last token: (d_model) * (d_model x vocab_size)
            std::vector<float> logits(vocab_size, 0.0f);
            float max_l = -1e9f;

            for (size_t v = 0; v < vocab_size; ++v) {
                double dot = 0.0;
                for (size_t d = 0; d < h.shape[2]; ++d) {
                    dot += h.at3(b, s, d) * w_unembed.at2(d, v);
                }
                logits[v] = static_cast<float>(dot);
                if (logits[v] > max_l) max_l = logits[v];
            }

            // Softmax
            double exp_sum = 0.0;
            std::vector<float> probs(vocab_size, 0.0f);
            for (size_t v = 0; v < vocab_size; ++v) {
                float e = std::exp(logits[v] - max_l);
                probs[v] = e;
                exp_sum += e;
            }

            float inv_sum = exp_sum > 0.0 ? static_cast<float>(1.0 / exp_sum) : 0.0f;
            double entropy = 0.0;
            std::vector<std::pair<float, int>> ranked;
            ranked.reserve(vocab_size);

            for (size_t v = 0; v < vocab_size; ++v) {
                probs[v] *= inv_sum;
                if (probs[v] > 1e-10f) {
                    entropy -= probs[v] * std::log(probs[v]);
                }
                ranked.push_back({probs[v], static_cast<int>(v)});
            }

            std::sort(ranked.rbegin(), ranked.rend());

            LayerPrediction lp;
            lp.layer_idx = l;
            lp.layer_name = (l == 0) ? "Embedding" : ("Layer " + std::to_string(l));
            lp.entropy = static_cast<float>(entropy);

            for (size_t k = 0; k < std::min(top_k, ranked.size()); ++k) {
                int tid = ranked[k].second;
                TokenCandidate cand;
                cand.token_id = tid;
                cand.token_str = tokenizer.decode({tid});
                cand.probability = ranked[k].first;
                cand.logit = logits[tid];
                lp.top_k_candidates.push_back(cand);
            }

            if (!lp.top_k_candidates.empty()) {
                lp.top_prediction = lp.top_k_candidates[0];
            }

            // Compute L2 norm of the vector
            double norm_acc = 0.0;
            for (size_t d = 0; d < h.shape[2]; ++d) {
                float val = h.at3(b, s, d);
                norm_acc += val * val;
            }
            lp.norm = static_cast<float>(std::sqrt(norm_acc));

            predictions.push_back(lp);
        }

        return predictions;
    }
};

struct HeadAnalysis {
    size_t head_idx;
    float mean_entropy;
    float bos_attention;
    float diagonal_attention;
    std::string classification;
};

class AttentionAnalyzer {
public:
    size_t n_heads;

    explicit AttentionAnalyzer(size_t heads) : n_heads(heads) {}

    std::vector<HeadAnalysis> analyze_layer_attention(const Tensor& attn_weights) const {
        // attn_weights: (batch, n_heads, seq, total_kv_len)
        std::vector<HeadAnalysis> analyses;
        if (attn_weights.size() == 0) return analyses;

        size_t b = 0;
        size_t seq = attn_weights.shape[2];
        size_t total_kv = attn_weights.shape[3];

        for (size_t h = 0; h < n_heads; ++h) {
            double total_entropy = 0.0;
            double bos_acc = 0.0;
            double diag_acc = 0.0;

            for (size_t i = 0; i < seq; ++i) {
                bos_acc += attn_weights.at4(b, h, i, 0);

                size_t past_len = total_kv - seq;
                size_t diag_col = past_len + i;
                if (diag_col < total_kv) {
                    diag_acc += attn_weights.at4(b, h, i, diag_col);
                }

                double row_entropy = 0.0;
                for (size_t j = 0; j < total_kv; ++j) {
                    float w = attn_weights.at4(b, h, i, j);
                    if (w > 1e-10f) {
                        row_entropy -= w * std::log(w);
                    }
                }
                total_entropy += row_entropy;
            }

            float mean_ent = static_cast<float>(total_entropy / seq);
            float bos_avg = static_cast<float>(bos_acc / seq);
            float diag_avg = static_cast<float>(diag_acc / seq);

            std::string cls;
            if (mean_ent < 0.5f) {
                cls = "Highly Focused";
            } else if (bos_avg > 0.4f) {
                cls = "BOS Anchor Head";
            } else if (diag_avg > 0.4f) {
                cls = "Previous Token Tracker";
            } else {
                cls = "Broad Context Aggregator";
            }

            analyses.push_back({h, mean_ent, bos_avg, diag_avg, cls});
        }

        return analyses;
    }
};

} // namespace nebula
