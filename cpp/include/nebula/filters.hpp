#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

namespace nebula {

struct SamplingTelemetry {
    float temperature;
    int top_k;
    float top_p;
    int sampled_token;
    float sampled_prob;
};

class LogitFilter {
public:
    float temperature;
    int top_k;
    float top_p;
    float repetition_penalty;

    LogitFilter(float temp = 0.7f, int k = 40, float p = 0.9f, float rep_pen = 1.05f)
        : temperature(std::max(temp, 1e-5f)),
          top_k(k),
          top_p(std::clamp(p, 0.0f, 1.0f)),
          repetition_penalty(rep_pen) {}

    int sample(
        const std::vector<float>& logits,
        const std::vector<int>& generated_tokens,
        std::mt19937& rng,
        SamplingTelemetry& telem
    ) const {
        std::vector<float> processed = logits;
        size_t vocab_size = processed.size();

        // 1. Repetition penalty
        for (int tid : generated_tokens) {
            if (tid >= 0 && static_cast<size_t>(tid) < vocab_size) {
                if (processed[tid] > 0.0f) {
                    processed[tid] /= repetition_penalty;
                } else {
                    processed[tid] *= repetition_penalty;
                }
            }
        }

        // 2. Temperature scaling
        float inv_temp = 1.0f / temperature;
        for (float& val : processed) {
            val *= inv_temp;
        }

        // 3. Top-K filtering
        std::vector<std::pair<float, int>> indexed;
        indexed.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            indexed.push_back({processed[i], static_cast<int>(i)});
        }
        std::sort(indexed.rbegin(), indexed.rend());

        if (top_k > 0 && static_cast<size_t>(top_k) < vocab_size) {
            for (size_t i = top_k; i < vocab_size; ++i) {
                indexed[i].first = -1e9f;
            }
        }

        // 4. Top-P (Nucleus) filtering
        float max_l = indexed[0].first;
        double exp_sum = 0.0;
        std::vector<float> probs(indexed.size(), 0.0f);

        for (size_t i = 0; i < indexed.size(); ++i) {
            if (indexed[i].first > -1e8f) {
                float e = std::exp(indexed[i].first - max_l);
                probs[i] = e;
                exp_sum += e;
            }
        }

        float inv_exp_sum = exp_sum > 0.0 ? static_cast<float>(1.0 / exp_sum) : 0.0f;
        double cum_p = 0.0;
        size_t cutoff_idx = indexed.size();

        for (size_t i = 0; i < indexed.size(); ++i) {
            probs[i] *= inv_exp_sum;
            cum_p += probs[i];
            if (cum_p > top_p && i > 0) {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Renormalize kept probabilities
        double final_sum = 0.0;
        for (size_t i = 0; i < cutoff_idx; ++i) {
            final_sum += probs[i];
        }

        float inv_final_sum = final_sum > 0.0 ? static_cast<float>(1.0 / final_sum) : 0.0f;
        for (size_t i = 0; i < cutoff_idx; ++i) {
            probs[i] *= inv_final_sum;
        }

        // Random categorical sample
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        float acc = 0.0f;
        int chosen_token = indexed[0].second;
        float chosen_prob = probs[0];

        for (size_t i = 0; i < cutoff_idx; ++i) {
            acc += probs[i];
            if (r <= acc) {
                chosen_token = indexed[i].second;
                chosen_prob = probs[i];
                break;
            }
        }

        telem.temperature = temperature;
        telem.top_k = top_k;
        telem.top_p = top_p;
        telem.sampled_token = chosen_token;
        telem.sampled_prob = chosen_prob;

        return chosen_token;
    }
};

} // namespace nebula
