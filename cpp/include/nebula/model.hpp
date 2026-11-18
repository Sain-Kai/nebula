#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include "nebula/tensor.hpp"
#include "nebula/config.hpp"
#include "nebula/layers.hpp"
#include "nebula/moe.hpp"

namespace nebula {

struct LayerTrace {
    size_t layer_idx;
    bool is_moe;
    Tensor residual_in;
    AttentionInspection attention;
    Tensor residual_mid;
    bool has_dense_neuron;
    NeuronInspection dense_neuron;
    MoERouterInspection moe_router;
    std::vector<NeuronInspection> moe_expert_neurons;
    Tensor residual_out;
};

struct FullTrace {
    std::vector<int> input_ids;
    std::vector<Tensor> residual_streams;
    std::vector<LayerTrace> layer_traces;
    Tensor final_logits;
};

using PatchHook = std::function<Tensor(int layer_idx, const std::string& target, const Tensor& activation)>;

class TransformerBlock {
public:
    size_t layer_idx;
    bool is_moe;
    LayerNorm ln1;
    MultiHeadAttention attn;
    LayerNorm ln2;
    std::unique_ptr<FeedForward> dense_ffn;
    std::unique_ptr<MoELayer> moe_layer;

    TransformerBlock(size_t idx, const ModelConfig& cfg, std::mt19937& rng)
        : layer_idx(idx),
          is_moe(cfg.is_moe && std::find(cfg.moe_layers.begin(), cfg.moe_layers.end(), idx) != cfg.moe_layers.end()),
          ln1(cfg.d_model, cfg.layer_norm_eps),
          attn(cfg.d_model, cfg.n_heads, rng),
          ln2(cfg.d_model, cfg.layer_norm_eps) {
        if (is_moe) {
            moe_layer = std::make_unique<MoELayer>(cfg.d_model, cfg.d_ff, cfg.num_experts, cfg.top_k_experts, rng);
        } else {
            dense_ffn = std::make_unique<FeedForward>(cfg.d_model, cfg.d_ff, rng);
        }
    }

    Tensor forward(
        const Tensor& x,
        Tensor& past_k,
        Tensor& past_v,
        LayerTrace& trace,
        const PatchHook& patch_hook = nullptr
    ) {
        trace.layer_idx = layer_idx;
        trace.is_moe = is_moe;
        trace.residual_in = x;

        // 1. Pre-attention norm
        Tensor norm1 = ln1.forward(x);

        // 2. Attention
        Tensor attn_out = attn.forward(norm1, past_k, past_v, trace.attention);

        if (patch_hook) {
            attn_out = patch_hook(static_cast<int>(layer_idx), "post_attn", attn_out);
        }

        // 3. First Residual Addition
        Tensor post_attn(x.shape);
        for (size_t i = 0; i < x.size(); ++i) post_attn[i] = x[i] + attn_out[i];
        trace.residual_mid = post_attn;

        // 4. Pre-FFN norm
        Tensor norm2 = ln2.forward(post_attn);

        // 5. MLP or MoE computation
        Tensor ffn_out;
        if (is_moe) {
            trace.has_dense_neuron = false;
            ffn_out = moe_layer->forward(norm2, trace.moe_router, trace.moe_expert_neurons);
        } else {
            trace.has_dense_neuron = true;
            ffn_out = dense_ffn->forward(norm2, trace.dense_neuron);
        }

        if (patch_hook) {
            ffn_out = patch_hook(static_cast<int>(layer_idx), "post_ffn", ffn_out);
        }

        // 6. Second Residual Addition
        Tensor post_ffn(post_attn.shape);
        for (size_t i = 0; i < post_attn.size(); ++i) post_ffn[i] = post_attn[i] + ffn_out[i];

        if (patch_hook) {
            post_ffn = patch_hook(static_cast<int>(layer_idx), "residual_stream", post_ffn);
        }

        trace.residual_out = post_ffn;
        return post_ffn;
    }
};

class NebulaTransformer {
public:
    ModelConfig config;
    std::mt19937 rng;
    Tensor w_token_emb;
    Tensor w_pos_emb;
    std::vector<TransformerBlock> blocks;
    LayerNorm ln_final;
    Tensor w_unembed;

    explicit NebulaTransformer(const ModelConfig& cfg, unsigned int seed = 42)
        : config(cfg), rng(seed),
          w_token_emb({cfg.vocab_size, cfg.d_model}),
          w_pos_emb({cfg.max_seq_len, cfg.d_model}),
          ln_final(cfg.d_model, cfg.layer_norm_eps),
          w_unembed({cfg.d_model, cfg.vocab_size}) {
        float scale = 1.0f / std::sqrt(static_cast<float>(cfg.d_model));
        std::normal_distribution<float> dist(0.0f, scale);

        for (float& val : w_token_emb.data) val = dist(rng);
        for (float& val : w_pos_emb.data) val = dist(rng);
        for (float& val : w_unembed.data) val = dist(rng);

        blocks.reserve(cfg.n_layers);
        for (size_t i = 0; i < cfg.n_layers; ++i) {
            blocks.emplace_back(i, cfg, rng);
        }
    }

    Tensor forward(
        const std::vector<int>& input_ids,
        std::vector<std::pair<Tensor, Tensor>>& past_kv,
        FullTrace& trace,
        const PatchHook& patch_hook = nullptr
    ) {
        size_t batch = 1;
        size_t seq = input_ids.size();
        size_t start_pos = (past_kv.size() > 0 && past_kv[0].first.size() > 0) ? past_kv[0].first.shape[2] : 0;

        trace.input_ids = input_ids;
        trace.residual_streams.clear();
        trace.layer_traces.resize(config.n_layers);

        // Token + Positional Embedding
        Tensor h({batch, seq, config.d_model}, 0.0f);
        for (size_t s = 0; s < seq; ++s) {
            int tid = input_ids[s];
            size_t pos = start_pos + s;
            if (pos >= config.max_seq_len) pos = config.max_seq_len - 1;

            for (size_t d = 0; d < config.d_model; ++d) {
                float tok_e = w_token_emb.at2(static_cast<size_t>(tid), d);
                float pos_e = w_pos_emb.at2(pos, d);
                h.at3(0, s, d) = tok_e + pos_e;
            }
        }

        if (patch_hook) {
            h = patch_hook(-1, "embedding", h);
        }

        trace.residual_streams.push_back(h);

        if (past_kv.size() < config.n_layers) {
            past_kv.resize(config.n_layers);
        }

        // Pass through blocks
        for (size_t l = 0; l < config.n_layers; ++l) {
            h = blocks[l].forward(h, past_kv[l].first, past_kv[l].second, trace.layer_traces[l], patch_hook);
            trace.residual_streams.push_back(h);
        }

        // Final Norm & Unembedding
        Tensor h_norm = ln_final.forward(h);
        Tensor logits = Tensor::matmul3d_proj(h_norm, w_unembed);
        trace.final_logits = logits;

        return logits;
    }
};

} // namespace nebula
