#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "nebula/tensor.hpp"
#include "nebula/layers.hpp"

namespace nebula {

struct MoERouterInspection {
    Tensor router_logits;
    Tensor all_expert_probs;
    std::vector<int> top_k_indices;
    std::vector<float> top_k_weights;
    float routing_entropy = 0.0f;
};

class MoERouter {
public:
    size_t d_model;
    size_t num_experts;
    size_t top_k;
    Tensor w_gate;

    MoERouter(size_t dim, size_t experts, size_t k, std::mt19937& rng)
        : d_model(dim), num_experts(experts), top_k(k), w_gate({dim, experts}) {
        float scale = 1.0f / std::sqrt(static_cast<float>(dim));
        std::normal_distribution<float> dist(0.0f, scale);
        for (float& val : w_gate.data) val = dist(rng);
    }

    void route(const Tensor& x, MoERouterInspection& inspection) const {
        Tensor logits = Tensor::matmul3d_proj(x, w_gate);
        Tensor probs = logits;
        probs.softmax_last_dim();

        // Extract for the last token in sequence
        size_t b = 0, s = x.shape[1] - 1;
        std::vector<std::pair<float, int>> expert_scores;
        expert_scores.reserve(num_experts);

        double entropy = 0.0;
        for (size_t e = 0; e < num_experts; ++e) {
            float p = probs.at3(b, s, e);
            expert_scores.push_back({p, static_cast<int>(e)});
            if (p > 1e-10f) {
                entropy -= p * std::log(p);
            }
        }

        std::sort(expert_scores.rbegin(), expert_scores.rend());

        inspection.top_k_indices.clear();
        inspection.top_k_weights.clear();

        float weight_sum = 0.0f;
        for (size_t i = 0; i < top_k; ++i) {
            inspection.top_k_indices.push_back(expert_scores[i].second);
            weight_sum += expert_scores[i].first;
        }

        float inv_sum = weight_sum > 0.0f ? 1.0f / weight_sum : 0.0f;
        for (size_t i = 0; i < top_k; ++i) {
            inspection.top_k_weights.push_back(expert_scores[i].first * inv_sum);
        }

        inspection.router_logits = logits;
        inspection.all_expert_probs = probs;
        inspection.routing_entropy = static_cast<float>(entropy);
    }
};

class MoELayer {
public:
    size_t d_model;
    size_t d_ff;
    size_t num_experts;
    size_t top_k;
    MoERouter router;
    std::vector<FeedForward> experts;

    MoELayer(size_t dim, size_t ff_dim, size_t n_experts, size_t k, std::mt19937& rng)
        : d_model(dim), d_ff(ff_dim), num_experts(n_experts), top_k(k),
          router(dim, n_experts, k, rng) {
        experts.reserve(n_experts);
        for (size_t i = 0; i < n_experts; ++i) {
            experts.emplace_back(dim, ff_dim, rng);
        }
    }

    Tensor forward(const Tensor& x, MoERouterInspection& router_insp, std::vector<NeuronInspection>& expert_insps) {
        router.route(x, router_insp);

        Tensor combined_output(x.shape, 0.0f);
        expert_insps.resize(num_experts);

        size_t batch = x.shape[0];
        size_t seq = x.shape[1];

        // Evaluate experts and accumulate top-k outputs
        for (size_t i = 0; i < top_k; ++i) {
            int exp_idx = router_insp.top_k_indices[i];
            float weight = router_insp.top_k_weights[i];

            Tensor exp_out = experts[exp_idx].forward(x, expert_insps[exp_idx]);

            for (size_t b = 0; b < batch; ++b) {
                for (size_t s = 0; s < seq; ++s) {
                    for (size_t d = 0; d < d_model; ++d) {
                        combined_output.at3(b, s, d) += weight * exp_out.at3(b, s, d);
                    }
                }
            }
        }

        return combined_output;
    }
};

} // namespace nebula
