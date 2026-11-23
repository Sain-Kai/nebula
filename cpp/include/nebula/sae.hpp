#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include "nebula/tensor.hpp"

namespace nebula {

struct CppActiveFeature {
    size_t feature_idx;
    float activation;
    float decoder_norm;
};

struct CppSAEResult {
    size_t l0_sparsity;
    float reconstruction_mse;
    std::vector<CppActiveFeature> active_features;
};

class CppSparseAutoencoder {
public:
    size_t d_model;
    size_t n_features;
    size_t k_sparsity;
    Tensor w_enc;
    Tensor w_dec;

    CppSparseAutoencoder(size_t dim = 128, size_t expansion = 4, size_t k = 16, unsigned int seed = 42)
        : d_model(dim), n_features(dim * expansion), k_sparsity(k),
          w_enc({dim, dim * expansion}), w_dec({dim * expansion, dim}) {
        std::mt19937 rng(seed);
        float scale = 1.0f / std::sqrt(static_cast<float>(dim));
        std::normal_distribution<float> dist(0.0f, scale);

        for (float& val : w_enc.data) val = dist(rng);
        for (float& val : w_dec.data) val = dist(rng);

        // Normalize decoder rows to unit L2 norm
        for (size_t f = 0; f < n_features; ++f) {
            double sum_sq = 0.0;
            for (size_t d = 0; d < d_model; ++d) {
                float v = w_dec.at2(f, d);
                sum_sq += v * v;
            }
            float inv_norm = sum_sq > 0.0 ? 1.0f / static_cast<float>(std::sqrt(sum_sq)) : 1.0f;
            for (size_t d = 0; d < d_model; ++d) {
                w_dec.at2(f, d) *= inv_norm;
            }
        }
    }

    CppSAEResult decompose(const Tensor& x_vec, size_t top_n = 8) const {
        CppSAEResult result;
        size_t d = x_vec.shape.back();
        if (d != d_model) return result;

        // 1. Encode: f = ReLU(x @ W_enc)
        std::vector<float> pre_act(n_features, 0.0f);
        for (size_t f = 0; f < n_features; ++f) {
            double dot = 0.0;
            for (size_t i = 0; i < d_model; ++i) {
                dot += x_vec.data[i] * w_enc.at2(i, f);
            }
            pre_act[f] = std::max(0.0f, static_cast<float>(dot));
        }

        // 2. Top-K Sparsity enforcement
        std::vector<std::pair<float, size_t>> ranked;
        ranked.reserve(n_features);
        for (size_t f = 0; f < n_features; ++f) {
            ranked.push_back({pre_act[f], f});
        }
        std::sort(ranked.rbegin(), ranked.rend());

        std::vector<float> sparse_f(n_features, 0.0f);
        size_t active_count = 0;
        for (size_t i = 0; i < std::min(k_sparsity, n_features); ++i) {
            if (ranked[i].first > 0.0f) {
                sparse_f[ranked[i].second] = ranked[i].first;
                active_count++;
            }
        }
        result.l0_sparsity = active_count;

        // 3. Decode: x_hat = sparse_f @ W_dec
        std::vector<float> x_hat(d_model, 0.0f);
        for (size_t i = 0; i < d_model; ++i) {
            double acc = 0.0;
            for (size_t f = 0; f < n_features; ++f) {
                if (sparse_f[f] > 0.0f) {
                    acc += sparse_f[f] * w_dec.at2(f, i);
                }
            }
            x_hat[i] = static_cast<float>(acc);
        }

        // 4. Reconstruction MSE
        double mse_acc = 0.0;
        for (size_t i = 0; i < d_model; ++i) {
            double diff = x_vec.data[i] - x_hat[i];
            mse_acc += diff * diff;
        }
        result.reconstruction_mse = static_cast<float>(mse_acc / d_model);

        // Top active features list
        for (size_t i = 0; i < std::min(top_n, ranked.size()); ++i) {
            if (ranked[i].first > 0.0f) {
                result.active_features.push_back({
                    ranked[i].second,
                    ranked[i].first,
                    1.0f // unit norm
                });
            }
        }

        return result;
    }
};

} // namespace nebula
