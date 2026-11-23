#include <iostream>
#include <cassert>
#include <cmath>
#include "nebula/tensor.hpp"
#include "nebula/tokenizer.hpp"
#include "nebula/model.hpp"
#include "nebula/interpretability.hpp"
#include "nebula/patching.hpp"
#include "nebula/filters.hpp"
#include "nebula/backward.hpp"
#include "nebula/circuit_tracer.hpp"
#include "nebula/sae.hpp"

using namespace nebula;

void test_tensor_ops() {
    Tensor a({2, 3}, 1.0f);
    Tensor b({3, 2}, 2.0f);
    Tensor c = Tensor::matmul2d(a, b);
    assert(c.shape[0] == 2 && c.shape[1] == 2);
    assert(std::abs(c.at2(0, 0) - 6.0f) < 1e-5f);

    Tensor s({1, 3});
    s[0] = 1.0f; s[1] = 2.0f; s[2] = 3.0f;
    s.softmax_last_dim();
    float sum = s[0] + s[1] + s[2];
    assert(std::abs(sum - 1.0f) < 1e-5f);
    std::cout << "✓ test_tensor_ops passed\n";
}

void test_tokenizer() {
    NebulaTokenizer tok;
    std::string text = "Nebula";
    auto ids = tok.encode(text, false);
    assert(tok.decode(ids) == text);

    auto meta = tok.analyze(text, true);
    assert(meta.size() == 7); // BOS + 6 bytes
    assert(meta[0].is_special == true);
    std::cout << "✓ test_tokenizer passed\n";
}

void test_transformer_forward() {
    ModelConfig cfg;
    cfg.d_model = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.d_ff = 128;
    cfg.is_moe = true;
    cfg.moe_layers = {1};

    NebulaTransformer model(cfg, 42);
    NebulaTokenizer tok;
    std::vector<int> ids = tok.encode("Hello");
    std::vector<std::pair<Tensor, Tensor>> past_kv;
    FullTrace trace;

    Tensor logits = model.forward(ids, past_kv, trace);
    assert(logits.shape[0] == 1);
    assert(logits.shape[1] == ids.size());
    assert(logits.shape[2] == cfg.vocab_size);
    assert(trace.residual_streams.size() == 3); // embedding + 2 layers
    assert(trace.layer_traces[1].is_moe == true);
    std::cout << "✓ test_transformer_forward passed\n";
}

void test_interpretability() {
    ModelConfig cfg;
    cfg.d_model = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.d_ff = 128;
    NebulaTransformer model(cfg, 42);
    NebulaTokenizer tok;

    std::vector<int> ids = tok.encode("Test");
    std::vector<std::pair<Tensor, Tensor>> past_kv;
    FullTrace trace;
    model.forward(ids, past_kv, trace);

    KVCacheTracker tracker(cfg.n_layers, cfg.n_heads, cfg.d_model / cfg.n_heads);
    auto snap = tracker.record_step(0, past_kv);
    assert(snap.total_cached_tokens == ids.size());
    assert(snap.memory_bytes > 0);

    LogitLens lens(model.w_unembed);
    auto preds = lens.analyze_residual_stream(trace.residual_streams);
    assert(preds.size() == 3);
    assert(preds[0].entropy > 0.0f);
    std::cout << "✓ test_interpretability passed\n";
}

void test_patching() {
    ModelConfig cfg;
    cfg.d_model = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.d_ff = 128;
    NebulaTransformer model(cfg, 42);
    NebulaTokenizer tok;

    ActivationPatcher patcher;
    patcher.add_intervention({0, "post_attn", "scale", 3.0f});

    auto comp = patcher.run_comparison(model, tok.encode("Intervene"), tok);
    assert(comp.baseline_top_candidates.size() > 0);
    assert(comp.patched_top_candidates.size() > 0);
    assert(comp.residual_divergence.size() == 3);
    std::cout << "✓ test_patching passed\n";
}

void test_sampling_filter() {
    LogitFilter filter(0.7f, 20, 0.9f);
    std::vector<float> logits(320, 0.0f);
    logits[10] = 5.0f;
    logits[20] = 4.5f;

    std::mt19937 rng(42);
    SamplingTelemetry telem;
    int sampled = filter.sample(logits, {}, rng, telem);
    assert(sampled == 10 || sampled == 20);
    assert(telem.sampled_prob > 0.0f);
    std::cout << "✓ test_sampling_filter passed\n";
}

void test_backward_engine() {
    ModelConfig cfg;
    cfg.d_model = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.d_ff = 128;
    NebulaTransformer model(cfg, 42);
    NebulaTokenizer tok;

    std::vector<int> ids = tok.encode("Backward test");
    std::vector<std::pair<Tensor, Tensor>> kv;
    FullTrace trace;
    model.forward(ids, kv, trace);

    BackwardEngine engine(model);
    auto res = engine.compute_gradients(ids, 65, trace);

    assert(res.loss > 0.0f);
    assert(res.layer_gradients.size() == 2);
    assert(res.gradient_health == "Healthy" || res.gradient_health == "Vanishing" || res.gradient_health == "Exploding");
    std::cout << "✓ test_backward_engine passed\n";
}

void test_circuit_tracer() {
    ModelConfig cfg;
    cfg.d_model = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 2;
    cfg.d_ff = 128;
    NebulaTransformer model(cfg, 42);
    NebulaTokenizer tok;

    std::vector<int> ids = tok.encode("Circuit test");
    std::vector<std::pair<Tensor, Tensor>> kv;
    FullTrace trace;
    model.forward(ids, kv, trace);

    CppCircuitTracer tracer(model);
    auto dla = tracer.compute_dla(trace, 65);

    assert(dla.head_attributions.size() == 4); // 2 layers * 2 heads
    std::cout << "✓ test_circuit_tracer passed\n";
}

void test_cpp_sae() {
    CppSparseAutoencoder sae(64, 4, 8, 42);
    Tensor x({1, 1, 64}, 0.5f);
    auto res = sae.decompose(x, 4);

    assert(res.l0_sparsity <= 8);
    assert(res.reconstruction_mse >= 0.0f);
    assert(res.active_features.size() <= 4);
    std::cout << "✓ test_cpp_sae passed\n";
}

int main() {
    std::cout << "Running Nebula C++ Unit Test Suite...\n";
    test_tensor_ops();
    test_tokenizer();
    test_transformer_forward();
    test_interpretability();
    test_patching();
    test_sampling_filter();
    test_backward_engine();
    test_circuit_tracer();
    test_cpp_sae();
    std::cout << "\nAll C++ unit tests passed successfully!\n";
    return 0;
}
