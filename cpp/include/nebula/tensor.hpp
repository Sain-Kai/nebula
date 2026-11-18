#pragma once

#include <vector>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <string>

namespace nebula {

class Tensor {
public:
    std::vector<size_t> shape;
    std::vector<float> data;

    Tensor() : shape{0} {}

    explicit Tensor(const std::vector<size_t>& shp, float val = 0.0f) : shape(shp) {
        size_t total = 1;
        for (auto s : shape) total *= s;
        data.assign(total, val);
    }

    size_t size() const {
        return data.size();
    }

    size_t ndim() const {
        return shape.size();
    }

    // Flat indexing
    float& operator[](size_t idx) {
        return data[idx];
    }

    const float& operator[](size_t idx) const {
        return data[idx];
    }

    // 2D access: [row, col]
    float& at2(size_t r, size_t c) {
        return data[r * shape[1] + c];
    }

    const float& at2(size_t r, size_t c) const {
        return data[r * shape[1] + c];
    }

    // 3D access: [b, s, d]
    float& at3(size_t b, size_t s, size_t d) {
        return data[(b * shape[1] + s) * shape[2] + d];
    }

    const float& at3(size_t b, size_t s, size_t d) const {
        return data[(b * shape[1] + s) * shape[2] + d];
    }

    // 4D access: [b, h, s, d]
    float& at4(size_t b, size_t h, size_t s, size_t d) {
        return data[((b * shape[1] + h) * shape[2] + s) * shape[3] + d];
    }

    const float& at4(size_t b, size_t h, size_t s, size_t d) const {
        return data[((b * shape[1] + h) * shape[2] + s) * shape[3] + d];
    }

    void reshape(const std::vector<size_t>& new_shape) {
        size_t total = 1;
        for (auto s : new_shape) total *= s;
        if (total != data.size()) {
            throw std::runtime_error("Cannot reshape tensor of size " + 
                                     std::to_string(data.size()) + " to target size " + 
                                     std::to_string(total));
        }
        shape = new_shape;
    }

    float norm() const {
        double sum = 0.0;
        for (float v : data) sum += v * v;
        return static_cast<float>(std::sqrt(sum));
    }

    static float cosine_similarity(const Tensor& a, const Tensor& b) {
        if (a.size() != b.size() || a.size() == 0) return 0.0f;
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a.data[i] * b.data[i];
            na += a.data[i] * a.data[i];
            nb += b.data[i] * b.data[i];
        }
        double denom = std::sqrt(na) * std::sqrt(nb);
        if (denom < 1e-12) return 0.0f;
        return static_cast<float>(dot / denom);
    }

    static float l2_distance(const Tensor& a, const Tensor& b) {
        if (a.size() != b.size()) return 0.0f;
        double sum = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            double diff = a.data[i] - b.data[i];
            sum += diff * diff;
        }
        return static_cast<float>(std::sqrt(sum));
    }

    // 2D Matrix multiplication: C = A (M x K) * B (K x N) -> (M x N)
    static Tensor matmul2d(const Tensor& a, const Tensor& b) {
        if (a.ndim() != 2 || b.ndim() != 2 || a.shape[1] != b.shape[0]) {
            throw std::runtime_error("Invalid dimensions for 2D matmul");
        }
        size_t M = a.shape[0];
        size_t K = a.shape[1];
        size_t N = b.shape[1];

        Tensor out({M, N}, 0.0f);
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float a_ik = a.at2(i, k);
                for (size_t j = 0; j < N; ++j) {
                    out.at2(i, j) += a_ik * b.at2(k, j);
                }
            }
        }
        return out;
    }

    // 3D Matrix multiplication: A (B x S x K) * B (K x N) -> (B x S x N)
    static Tensor matmul3d_proj(const Tensor& a, const Tensor& w) {
        if (a.ndim() != 3 || w.ndim() != 2 || a.shape[2] != w.shape[0]) {
            throw std::runtime_error("Invalid dimensions for 3D projection");
        }
        size_t batch = a.shape[0];
        size_t seq = a.shape[1];
        size_t K = a.shape[2];
        size_t N = w.shape[1];

        Tensor out({batch, seq, N}, 0.0f);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t s = 0; s < seq; ++s) {
                for (size_t k = 0; k < K; ++k) {
                    float a_val = a.at3(b, s, k);
                    for (size_t j = 0; j < N; ++j) {
                        out.at3(b, s, j) += a_val * w.at2(k, j);
                    }
                }
            }
        }
        return out;
    }

    // In-place Softmax over the last dimension
    void softmax_last_dim() {
        if (data.empty()) return;
        size_t last_dim = shape.back();
        size_t outer_size = data.size() / last_dim;

        for (size_t i = 0; i < outer_size; ++i) {
            size_t offset = i * last_dim;
            float max_val = data[offset];
            for (size_t j = 1; j < last_dim; ++j) {
                if (data[offset + j] > max_val) max_val = data[offset + j];
            }

            double sum = 0.0;
            for (size_t j = 0; j < last_dim; ++j) {
                float exp_val = std::exp(data[offset + j] - max_val);
                data[offset + j] = exp_val;
                sum += exp_val;
            }

            float inv_sum = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
            for (size_t j = 0; j < last_dim; ++j) {
                data[offset + j] *= inv_sum;
            }
        }
    }

    // In-place GELU activation
    void gelu_() {
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        for (float& x : data) {
            x = 0.5f * x * (1.0f + std::tanh(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
        }
    }
};

} // namespace nebula
