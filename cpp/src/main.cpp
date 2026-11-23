#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>

#include "nebula/config.hpp"
#include "nebula/tokenizer.hpp"
#include "nebula/model.hpp"
#include "nebula/interpretability.hpp"
#include "nebula/patching.hpp"
#include "nebula/backward.hpp"
#include "nebula/circuit_tracer.hpp"
#include "nebula/sae.hpp"
#include "nebula/gguf_parser.hpp"
#include "nebula/speculative.hpp"
#include "nebula/http_server.hpp"

using namespace nebula;

void print_banner() {
    std::cout << "\033[1;36m"
              << "╭─────────────────────────────────────────────────────────────────────────╮\n"
              << "│ ✦ NEBULA C++ | LLM Mechanistic Interpretability Engine                  │\n"
              << "│ High-performance bare-metal layer, neuron, KV cache, and MoE teardown   │\n"
              << "╰─────────────────────────────────────────────────────────────────────────╯\n"
              << "\033[0m\n";
}

void run_demo(const std::string& prompt) {
    print_banner();
    std::cout << "\033[1;32mRunning C++ inference teardown on prompt:\033[0m \033[3m'" << prompt << "'\033[0m\n\n";

    ModelConfig cfg;
    NebulaTokenizer tokenizer;
    NebulaTransformer model(cfg, 42);

    // 1. Tokenizer breakdown
    auto tokens = tokenizer.analyze(prompt);
    std::cout << "\033[1;34m1. Tokenization & Byte Breakdown:\033[0m\n";
    std::cout << "┌──────┬────────────┬──────────┬───────────┬────────────────┐\n";
    std::cout << "│  Idx │ Token Text │ Token ID │ Hex Bytes │ Character Span │\n";
    std::cout << "├──────┼────────────┼──────────┼───────────┼────────────────┤\n";
    for (const auto& t : tokens) {
        std::cout << "│ " << std::setw(4) << t.index << " │ "
                  << std::setw(10) << t.text << " │ "
                  << std::setw(8) << t.token_id << " │ "
                  << std::setw(9) << t.hex_str << " │ ["
                  << t.start_char << ".." << t.end_char << "]          │\n";
    }
    std::cout << "└──────┴────────────┴──────────┴───────────┴────────────────┘\n\n";

    // 2. Forward pass
    std::vector<int> input_ids = tokenizer.encode(prompt);
    std::vector<std::pair<Tensor, Tensor>> past_kv;
    FullTrace trace;

    auto t_start = std::chrono::high_resolution_clock::now();
    Tensor logits = model.forward(input_ids, past_kv, trace);
    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // 3. MoE Routing
    std::cout << "\033[1;34m2. Mixture of Experts Routing:\033[0m\n";
    std::cout << "┌─────────┬──────────────────────┬────────────────────────────┐\n";
    std::cout << "│ Layer   │ Top-K Chosen Experts │ Gating Weights             │\n";
    std::cout << "├─────────┼──────────────────────┼────────────────────────────┤\n";
    for (const auto& lt : trace.layer_traces) {
        if (lt.is_moe) {
            std::cout << "│ Layer " << lt.layer_idx << " │ Experts [";
            for (size_t i = 0; i < lt.moe_router.top_k_indices.size(); ++i) {
                std::cout << lt.moe_router.top_k_indices[i] << (i + 1 < lt.moe_router.top_k_indices.size() ? ", " : "");
            }
            std::cout << "]        │ Weights [";
            for (size_t i = 0; i < lt.moe_router.top_k_weights.size(); ++i) {
                std::cout << std::fixed << std::setprecision(3) << lt.moe_router.top_k_weights[i]
                          << (i + 1 < lt.moe_router.top_k_weights.size() ? ", " : "");
            }
            std::cout << "]    │\n";
        }
    }
    std::cout << "└─────────┴──────────────────────┴────────────────────────────┘\n\n";

    // 4. Logit Lens thought progression
    LogitLens lens(model.w_unembed);
    auto predictions = lens.analyze_residual_stream(trace.residual_streams);

    std::cout << "\033[1;34m3. Logit Lens: Thought Progression by Depth:\033[0m\n";
    std::cout << "┌───────────┬─────────────────┬─────────────┬─────────┐\n";
    std::cout << "│ Depth     │ Top Candidate   │ Probability │ Entropy │\n";
    std::cout << "├───────────┼─────────────────┼─────────────┼─────────┤\n";
    for (const auto& lp : predictions) {
        std::cout << "│ " << std::setw(9) << lp.layer_name << " │ "
                  << std::setw(15) << ("'" + lp.top_prediction.token_str + "'") << " │ "
                  << std::setw(10) << std::fixed << std::setprecision(1) << (lp.top_prediction.probability * 100.0f) << "% │ "
                  << std::setw(7) << std::setprecision(3) << lp.entropy << " │\n";
    }
    std::cout << "└───────────┴─────────────────┴─────────────┴─────────┘\n\n";

    // 5. KV Cache memory
    KVCacheTracker kv_tracker(cfg.n_layers, cfg.n_heads, cfg.d_model / cfg.n_heads);
    auto snap = kv_tracker.record_step(0, past_kv);
    std::cout << "\033[1;32m✓ C++ Forward Pass Execution Time:\033[0m " << ms << " ms\n";
    std::cout << "\033[1;32m✓ KV Cache Footprint:\033[0m " << snap.memory_kb << " KB (" << snap.memory_bytes << " bytes)\n\n";
}

int main(int argc, char** argv) {
    std::string cmd = (argc > 1) ? argv[1] : "demo";

    if (cmd == "start") {
        print_banner();
        NebulaConfig cfg;
        int port = 8765;
        if (argc > 2) {
            try { port = std::stoi(argv[2]); } catch (...) {}
        }
        cfg.server.port = port;
        HttpServer server(cfg);
        server.start();
    } else if (cmd == "demo") {
        std::string prompt = (argc > 2) ? argv[2] : "Nebula C++ mechanistic interpretability";
        run_demo(prompt);
    } else if (cmd == "backward") {
        print_banner();
        std::string prompt = (argc > 2) ? argv[2] : "Nebula gradient teardown";
        int target_id = (argc > 3) ? std::stoi(argv[3]) : 65; // 'A'

        ModelConfig cfg;
        NebulaTokenizer tokenizer;
        NebulaTransformer model(cfg, 42);

        std::vector<int> input_ids = tokenizer.encode(prompt);
        std::vector<std::pair<Tensor, Tensor>> kv;
        FullTrace trace;
        model.forward(input_ids, kv, trace);

        BackwardEngine eng(model);
        auto b_res = eng.compute_gradients(input_ids, target_id, trace);

        std::cout << "\033[1;34mBackward Pass & Gradient Flow Diagnostics:\033[0m\n";
        std::cout << "Target Token ID: " << target_id << " | Cross-Entropy Loss: " << b_res.loss << "\n";
        std::cout << "Gradient Health: " << b_res.gradient_health << "\n\n";

        std::cout << "┌───────┬─────────┬────────────────────┬──────────────────┬─────────────────┐\n";
        std::cout << "│ Layer │ Is MoE  │ Residual Grad Norm │ Attn Grad Norm   │ Ratio to Final  │\n";
        std::cout << "├───────┼─────────┼────────────────────┼──────────────────┼─────────────────┤\n";
        for (const auto& lg : b_res.layer_gradients) {
            std::cout << "│ " << std::setw(5) << lg.layer_idx << " │ "
                      << std::setw(7) << (lg.is_moe ? "True" : "False") << " │ "
                      << std::setw(18) << std::fixed << std::setprecision(4) << lg.residual_grad_norm << " │ "
                      << std::setw(16) << lg.attn_grad_norm << " │ "
                      << std::setw(15) << lg.gradient_ratio_to_final << " │\n";
        }
        std::cout << "└───────┴─────────┴────────────────────┴──────────────────┴─────────────────┘\n\n";
    } else if (cmd == "dla") {
        print_banner();
        std::string prompt = (argc > 2) ? argv[2] : "Direct logit attribution";
        int target_id = (argc > 3) ? std::stoi(argv[3]) : 65;

        ModelConfig cfg;
        NebulaTokenizer tokenizer;
        NebulaTransformer model(cfg, 42);

        std::vector<int> input_ids = tokenizer.encode(prompt);
        std::vector<std::pair<Tensor, Tensor>> kv;
        FullTrace trace;
        model.forward(input_ids, kv, trace);

        CppCircuitTracer tracer(model);
        auto dla = tracer.compute_dla(trace, target_id);

        std::cout << "\033[1;34mDirect Logit Attribution (DLA) Summary:\033[0m\n";
        std::cout << "Target Token ID: " << target_id << " | Baseline Logit: " << dla.baseline_logit << "\n";
        std::cout << "Embedding Contribution: " << dla.embedding_contribution << "\n";
        std::cout << "Total Reconstructed: " << dla.reconstructed_logit << "\n\n";

        std::cout << "┌─────────┬──────────┬────────────────────┬───────────┐\n";
        std::cout << "│  Layer  │   Head   │ Logit Contribution │ Promoting │\n";
        std::cout << "├─────────┼──────────┼────────────────────┼───────────┤\n";
        for (size_t i = 0; i < std::min<size_t>(8, dla.head_attributions.size()); ++i) {
            const auto& ha = dla.head_attributions[i];
            std::cout << "│ Layer " << ha.layer_idx << " │ Head " << ha.head_idx << "   │ "
                      << std::setw(18) << std::fixed << std::setprecision(4) << ha.contribution << " │ "
                      << std::setw(9) << (ha.is_promoter ? "+ Yes" : "- No") << " │\n";
        }
        std::cout << "└─────────┴──────────┴────────────────────┴───────────┘\n\n";
    } else if (cmd == "sae") {
        print_banner();
        std::cout << "\033[1;34mSparse Autoencoder Latent Decomposition Demo:\033[0m\n";
        CppSparseAutoencoder sae(128, 4, 16, 42);
        Tensor x({1, 1, 128}, 0.35f);
        auto sae_res = sae.decompose(x, 8);
        std::cout << "Input Dimension: 128 | Dictionary Expansion: 4x (512 Features) | Top-K Sparsity: " << sae_res.l0_sparsity << "\n";
        std::cout << "Reconstruction MSE: " << std::fixed << std::setprecision(6) << sae_res.reconstruction_mse << "\n\n";
        std::cout << "┌───────┬────────────────────┬──────────────┐\n";
        std::cout << "│ Rank  │ Feature Dictionary │ Activation   │\n";
        std::cout << "├───────┼────────────────────┼──────────────┤\n";
        for (size_t i = 0; i < sae_res.active_features.size(); ++i) {
            std::cout << "│ " << std::setw(5) << (i + 1) << " │ Feature " << std::setw(8) << sae_res.active_features[i].feature_idx << " │ "
                      << std::setw(12) << std::fixed << std::setprecision(4) << sae_res.active_features[i].activation << " │\n";
        }
        std::cout << "└───────┴────────────────────┴──────────────┘\n\n";
    } else if (cmd == "gguf") {
        print_banner();
        std::cout << "\033[1;34mNative Binary GGUF Header & Tensor Descriptor Inspection:\033[0m\n";
        auto buf = GGUFParser::create_dummy_gguf("qwen2.5-7b", 4);
        auto gguf = GGUFParser::parse_buffer(buf);
        std::cout << "Magic: 0x" << std::hex << gguf.magic << std::dec << " ('GGUF') | Version: " << gguf.version << "\n";
        std::cout << "Tensors: " << gguf.tensor_count << " | Key-Value Metadata: " << gguf.kv_count << "\n\n";
        for (const auto& m : gguf.metadata) {
            std::cout << "• Metadata [" << m.key << "] = " << m.str_value << "\n";
        }
        std::cout << "\n┌────────────────────────────┬──────┬──────────────────┬────────────┐\n";
        std::cout << "│ Tensor Name                │ Dims │ Shape            │ Offset (B) │\n";
        std::cout << "├────────────────────────────┼──────┼──────────────────┼────────────┤\n";
        for (const auto& t : gguf.tensors) {
            std::string shape_str = "[";
            for (size_t d = 0; d < t.shape.size(); ++d) {
                shape_str += std::to_string(t.shape[d]) + (d + 1 < t.shape.size() ? "x" : "");
            }
            shape_str += "]";
            std::cout << "│ " << std::left << std::setw(26) << t.name << " │ "
                      << std::right << std::setw(4) << t.n_dims << " │ "
                      << std::left << std::setw(16) << shape_str << " │ "
                      << std::right << std::setw(10) << t.offset << " │\n";
        }
        std::cout << "└────────────────────────────┴──────┴──────────────────┴────────────┘\n\n";
    } else if (cmd == "speculate") {
        print_banner();
        std::cout << "\033[1;34mSpeculative Decoding & Verification Telemetry:\033[0m\n";
        NebulaTokenizer tok;
        CppSpeculativeVerifier verifier(4);
        std::vector<int> draft = {78, 101, 98, 117}; // "N e b u"
        std::vector<float> q = {0.88f, 0.79f, 0.65f, 0.70f};
        std::vector<float> p = {0.92f, 0.81f, 0.72f, 0.55f};

        auto spec_res = verifier.verify_candidates(draft, q, p, tok);
        std::cout << "Draft Candidates: " << spec_res.proposed_tokens
                  << " | Accepted Tokens: " << spec_res.accepted_tokens
                  << " | Speedup Multiplier: " << std::fixed << std::setprecision(2) << spec_res.speedup_ratio << "x\n\n";

        std::cout << "┌──────┬──────────┬──────────┬──────────┬────────────┬───────────┐\n";
        std::cout << "│ Pos  │ Token    │ Draft Q  │ Target P │ Acceptance │ Status    │\n";
        std::cout << "├──────┼──────────┼──────────┼──────────┼────────────┼───────────┤\n";
        for (const auto& d : spec_res.decisions) {
            std::cout << "│ " << std::setw(4) << d.position << " │ "
                      << std::setw(8) << ("'" + d.draft_token_str + "'") << " │ "
                      << std::setw(8) << std::fixed << std::setprecision(3) << d.draft_prob << " │ "
                      << std::setw(8) << d.target_prob << " │ "
                      << std::setw(10) << d.acceptance_prob << " │ "
                      << std::setw(9) << (d.is_accepted ? "\033[32mACCEPTED\033[0m" : "\033[31mREJECTED\033[0m") << " │\n";
        }
        std::cout << "└──────┴──────────┴──────────┴──────────┴────────────┴───────────┘\n\n";
    } else if (cmd == "bench") {
        print_banner();
        std::cout << "Benchmarking C++ Engine across 100 token forward passes...\n";
        ModelConfig cfg;
        NebulaTransformer model(cfg, 42);
        std::vector<int> tokens = {256, 78, 101, 98, 117, 108, 97};
        std::vector<std::pair<Tensor, Tensor>> kv;
        FullTrace trace;

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            model.forward(tokens, kv, trace);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << "Total time for 100 runs: " << total_ms << " ms ("
                  << (total_ms / 100.0) << " ms per forward pass)\n";
        std::cout << "Throughput: " << (100.0 / (total_ms / 1000.0)) << " passes/sec\n";
    } else {
        std::cout << "Usage:\n"
                  << "  nebula-cpp demo [prompt]   - Run terminal teardown demo\n"
                  << "  nebula-cpp sae             - Run sparse autoencoder latent decomposition\n"
                  << "  nebula-cpp gguf            - Parse binary GGUF header and tensor tables\n"
                  << "  nebula-cpp speculate       - Verify speculative decoding draft tokens\n"
                  << "  nebula-cpp start [port]    - Start HTTP visualization server\n"
                  << "  nebula-cpp bench           - Benchmark latency and throughput\n";
    }
    return 0;
}
