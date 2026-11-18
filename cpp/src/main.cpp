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
                  << "  nebula-cpp start [port]     - Start HTTP visualization server\n"
                  << "  nebula-cpp bench            - Benchmark latency and throughput\n";
    }
    return 0;
}
