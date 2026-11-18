#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "nebula/tensor.hpp"
#include "nebula/model.hpp"
#include "nebula/tokenizer.hpp"

namespace nebula {

struct PatchIntervention {
    int layer_idx = 0;
    std::string target = "post_attn"; // "post_attn", "post_ffn", "residual_stream"
    std::string action = "scale";     // "scale", "zero_ablate"
    float scalar_multiplier = 1.0f;
};

struct LayerDivergence {
    size_t layer_idx;
    std::string layer_name;
    float cosine_similarity;
    float l2_distance;
};

struct CounterfactualComparison {
    std::vector<TokenCandidate> baseline_top_candidates;
    std::vector<TokenCandidate> patched_top_candidates;
    std::vector<LayerDivergence> residual_divergence;
    bool primary_prediction_changed = false;
};

class ActivationPatcher {
public:
    std::vector<PatchIntervention> interventions;

    void add_intervention(const PatchIntervention& p) {
        interventions.push_back(p);
    }

    void clear() {
        interventions.clear();
    }

    PatchHook build_hook() const {
        return [this](int layer_idx, const std::string& target, const Tensor& act) -> Tensor {
            Tensor result = act;
            for (const auto& item : interventions) {
                if (item.layer_idx == layer_idx && item.target == target) {
                    if (item.action == "zero_ablate") {
                        std::fill(result.data.begin(), result.data.end(), 0.0f);
                    } else if (item.action == "scale") {
                        for (float& val : result.data) val *= item.scalar_multiplier;
                    }
                }
            }
            return result;
        };
    }

    CounterfactualComparison run_comparison(
        NebulaTransformer& model,
        const std::vector<int>& input_ids,
        const NebulaTokenizer& tokenizer
    ) const {
        CounterfactualComparison comp;

        // Baseline forward pass
        std::vector<std::pair<Tensor, Tensor>> base_kv;
        FullTrace base_trace;
        Tensor base_logits = model.forward(input_ids, base_kv, base_trace, nullptr);

        // Patched forward pass
        PatchHook hook = build_hook();
        std::vector<std::pair<Tensor, Tensor>> patch_kv;
        FullTrace patch_trace;
        Tensor patch_logits = model.forward(input_ids, patch_kv, patch_trace, hook);

        // Top predictions extraction
        auto get_top = [&](const Tensor& logits) -> std::vector<TokenCandidate> {
            size_t vocab_size = logits.shape[2];
            size_t b = 0, s = logits.shape[1] - 1;
            std::vector<std::pair<float, int>> ranked;
            ranked.reserve(vocab_size);

            float max_l = -1e9f;
            for (size_t v = 0; v < vocab_size; ++v) {
                float val = logits.at3(b, s, v);
                if (val > max_l) max_l = val;
            }

            double exp_sum = 0.0;
            std::vector<float> probs(vocab_size, 0.0f);
            for (size_t v = 0; v < vocab_size; ++v) {
                float e = std::exp(logits.at3(b, s, v) - max_l);
                probs[v] = e;
                exp_sum += e;
            }

            float inv_sum = exp_sum > 0.0 ? static_cast<float>(1.0 / exp_sum) : 0.0f;
            for (size_t v = 0; v < vocab_size; ++v) {
                probs[v] *= inv_sum;
                ranked.push_back({probs[v], static_cast<int>(v)});
            }

            std::sort(ranked.rbegin(), ranked.rend());

            std::vector<TokenCandidate> top;
            for (size_t i = 0; i < std::min<size_t>(5, ranked.size()); ++i) {
                int tid = ranked[i].second;
                top.push_back({tid, tokenizer.decode({tid}), ranked[i].first, logits.at3(b, s, tid)});
            }
            return top;
        };

        comp.baseline_top_candidates = get_top(base_logits);
        comp.patched_top_candidates = get_top(patch_logits);

        if (!comp.baseline_top_candidates.empty() && !comp.patched_top_candidates.empty()) {
            comp.primary_prediction_changed = (comp.baseline_top_candidates[0].token_id != 
                                               comp.patched_top_candidates[0].token_id);
        }

        // Residual divergence across depth
        size_t n_streams = std::min(base_trace.residual_streams.size(), patch_trace.residual_streams.size());
        for (size_t l = 0; l < n_streams; ++l) {
            float cos_sim = Tensor::cosine_similarity(base_trace.residual_streams[l], patch_trace.residual_streams[l]);
            float l2_d = Tensor::l2_distance(base_trace.residual_streams[l], patch_trace.residual_streams[l]);

            LayerDivergence ld;
            ld.layer_idx = l;
            ld.layer_name = (l == 0) ? "Embedding" : ("Layer " + std::to_string(l));
            ld.cosine_similarity = cos_sim;
            ld.l2_distance = l2_d;
            comp.residual_divergence.push_back(ld);
        }

        return comp;
    }
};

} // namespace nebula
