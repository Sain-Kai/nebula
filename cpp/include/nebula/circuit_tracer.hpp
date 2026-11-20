#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "nebula/tensor.hpp"
#include "nebula/model.hpp"

namespace nebula {

struct CppHeadAttribution {
    size_t layer_idx;
    size_t head_idx;
    float contribution;
    bool is_promoter;
};

struct CppDLAResult {
    int target_token_id;
    float baseline_logit;
    float embedding_contribution;
    std::vector<CppHeadAttribution> head_attributions;
    float reconstructed_logit;
};

class CppCircuitTracer {
public:
    const NebulaTransformer& model;

    explicit CppCircuitTracer(const NebulaTransformer& m) : model(m) {}

    CppDLAResult compute_dla(const FullTrace& trace, int target_token_id) const {
        CppDLAResult result;
        result.target_token_id = target_token_id;

        size_t d_model = model.config.d_model;
        size_t n_heads = model.config.n_heads;
        size_t n_layers = model.config.n_layers;
        size_t d_head = d_model / n_heads;
        size_t s = trace.residual_streams[0].shape[1] - 1;

        // Unembedding vector for target token: W_U[:, target]
        std::vector<float> unembed_col(d_model, 0.0f);
        for (size_t d = 0; d < d_model; ++d) {
            unembed_col[d] = model.w_unembed.at2(d, static_cast<size_t>(target_token_id));
        }

        // Embedding contribution: h_0 @ W_U[:, target]
        double emb_acc = 0.0;
        for (size_t d = 0; d < d_model; ++d) {
            emb_acc += trace.residual_streams[0].at3(0, s, d) * unembed_col[d];
        }
        result.embedding_contribution = static_cast<float>(emb_acc);
        double total_recon = emb_acc;

        // Head attributions
        for (size_t l = 0; l < n_layers; ++l) {
            const auto& lt = trace.layer_traces[l];
            for (size_t h = 0; h < n_heads; ++h) {
                // Approximate direct projection from attention value to W_O
                double head_acc = 0.0;
                for (size_t d = 0; d < d_head; ++d) {
                    size_t flat_idx = h * d_head + d;
                    head_acc += lt.residual_mid.at3(0, s, flat_idx) * unembed_col[flat_idx] * 0.25;
                }
                float contrib = static_cast<float>(head_acc);
                total_recon += contrib;

                result.head_attributions.push_back({
                    l,
                    h,
                    contrib,
                    contrib > 0.0f
                });
            }
        }

        result.reconstructed_logit = static_cast<float>(total_recon);
        result.baseline_logit = trace.final_logits.at3(0, s, static_cast<size_t>(target_token_id));
        return result;
    }
};

} // namespace nebula
