#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <functional>
#include <thread>
#include <atomic>
#include <iostream>
#include <fstream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

#include "nebula/config.hpp"
#include "nebula/model.hpp"
#include "nebula/tokenizer.hpp"
#include "nebula/interpretability.hpp"
#include "nebula/patching.hpp"
#include "nebula/filters.hpp"
#include "nebula/json.hpp"

namespace nebula {

class HttpServer {
public:
    NebulaConfig config;
    NebulaTokenizer tokenizer;
    std::unique_ptr<NebulaTransformer> model;
    std::string web_dir;
    std::atomic<bool> running{false};
    int server_fd = -1;

    HttpServer(const NebulaConfig& cfg, const std::string& static_path = "./nebula/web")
        : config(cfg), web_dir(static_path) {
        model = std::make_unique<NebulaTransformer>(cfg.model, 42);
    }

    ~HttpServer() {
        stop();
    }

    void start() {
        server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            std::cerr << "Failed to create socket\n";
            return;
        }

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = inet_addr(config.server.host.c_str());
        address.sin_port = htons(config.server.port);

        if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
            std::cerr << "Bind failed on port " << config.server.port << "\n";
            close(server_fd);
            return;
        }

        if (listen(server_fd, 10) < 0) {
            std::cerr << "Listen failed\n";
            close(server_fd);
            return;
        }

        running = true;
        std::cout << "✦ NEBULA C++ Server running at http://" 
                  << config.server.host << ":" << config.server.port << "\n";

        while (running) {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
            if (client_fd < 0) {
                if (!running) break;
                continue;
            }

            handle_client(client_fd);
        }
    }

    void stop() {
        if (running) {
            running = false;
            if (server_fd >= 0) {
                close(server_fd);
                server_fd = -1;
            }
        }
    }

private:
    void handle_client(int client_fd) {
        char buffer[16384] = {0};
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        if (bytes_read <= 0) {
            close(client_fd);
            return;
        }

        std::string request(buffer, bytes_read);
        std::istringstream req_stream(request);
        std::string method, path, proto;
        req_stream >> method >> path >> proto;

        // Extract body if POST
        std::string body;
        size_t body_pos = request.find("\r\n\r\n");
        if (body_pos != std::string::npos) {
            body = request.substr(body_pos + 4);
        }

        std::string response;
        if (method == "GET") {
            if (path == "/" || path == "/index.html") {
                response = serve_file(web_dir + "/index.html", "text/html");
            } else if (path == "/static/styles.css" || path == "/styles.css") {
                response = serve_file(web_dir + "/styles.css", "text/css");
            } else if (path == "/static/app.js" || path == "/app.js") {
                response = serve_file(web_dir + "/app.js", "application/javascript");
            } else if (path == "/api/health") {
                JsonValue res = JsonValue::object();
                res.set("status", "ok");
                res.set("engine", "nebula-cpp");
                res.set("version", "0.1.0");
                response = build_json_response(res.serialize());
            } else if (path == "/api/model/info") {
                JsonValue res = JsonValue::object();
                res.set("model_name", "Nebula-Transformer-MoE-CPP");
                res.set("vocab_size", config.model.vocab_size);
                res.set("d_model", config.model.d_model);
                res.set("n_layers", config.model.n_layers);
                res.set("n_heads", config.model.n_heads);
                res.set("d_ff", config.model.d_ff);
                res.set("is_moe", config.model.is_moe);
                res.set("num_experts", config.model.num_experts);
                res.set("top_k_experts", config.model.top_k_experts);
                response = build_json_response(res.serialize());
            } else {
                response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
            }
        } else if (method == "POST") {
            if (path == "/api/tokenize") {
                std::string text = parse_json_string_field(body, "text");
                if (text.empty()) text = "Nebula";
                auto meta_list = tokenizer.analyze(text);
                auto ids = tokenizer.encode(text);

                JsonValue res = JsonValue::object();
                JsonValue arr = JsonValue::array();
                for (const auto& m : meta_list) {
                    JsonValue item = JsonValue::object();
                    item.set("index", m.index);
                    item.set("token_id", m.token_id);
                    item.set("text", m.text);
                    item.set("hex", m.hex_str);
                    item.set("is_special", m.is_special);
                    item.set("start_char", m.start_char);
                    item.set("end_char", m.end_char);
                    arr.push(item);
                }
                res.set("tokens", arr);
                res.set("total_tokens", ids.size());
                response = build_json_response(res.serialize());
            } else if (path == "/api/inference") {
                std::string prompt = parse_json_string_field(body, "prompt");
                if (prompt.empty()) prompt = "Why is the sky blue?";
                int max_tokens = parse_json_int_field(body, "max_new_tokens", 8);

                response = build_json_response(execute_inference_json(prompt, max_tokens));
            } else if (path == "/api/patch-compare") {
                std::string prompt = parse_json_string_field(body, "prompt");
                if (prompt.empty()) prompt = "Nebula";
                ActivationPatcher patcher;
                patcher.add_intervention({0, "post_attn", "scale", 2.0f});
                auto comp = patcher.run_comparison(*model, tokenizer.encode(prompt), tokenizer);

                JsonValue res = JsonValue::object();
                res.set("primary_prediction_changed", comp.primary_prediction_changed);
                JsonValue base_arr = JsonValue::array();
                for (const auto& c : comp.baseline_top_candidates) {
                    JsonValue item = JsonValue::object();
                    item.set("token_id", c.token_id);
                    item.set("token_str", c.token_str);
                    item.set("prob", c.probability);
                    base_arr.push(item);
                }
                res.set("baseline_top_candidates", base_arr);

                JsonValue patch_arr = JsonValue::array();
                for (const auto& c : comp.patched_top_candidates) {
                    JsonValue item = JsonValue::object();
                    item.set("token_id", c.token_id);
                    item.set("token_str", c.token_str);
                    item.set("prob", c.probability);
                    patch_arr.push(item);
                }
                res.set("patched_top_candidates", patch_arr);
                response = build_json_response(res.serialize());
            } else {
                response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
            }
        } else {
            response = "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n";
        }

        ssize_t nw = write(client_fd, response.c_str(), response.size());
        (void)nw;
        close(client_fd);
    }

    std::string serve_file(const std::string& filepath, const std::string& mime) {
        std::ifstream f(filepath, std::ios::binary);
        if (!f.is_open()) {
            return "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
        }
        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::stringstream ss;
        ss << "HTTP/1.1 200 OK\r\n"
           << "Content-Type: " << mime << "\r\n"
           << "Content-Length: " << content.size() << "\r\n"
           << "Access-Control-Allow-Origin: *\r\n"
           << "Connection: close\r\n\r\n"
           << content;
        return ss.str();
    }

    std::string build_json_response(const std::string& json_str) {
        std::stringstream ss;
        ss << "HTTP/1.1 200 OK\r\n"
           << "Content-Type: application/json\r\n"
           << "Content-Length: " << json_str.size() << "\r\n"
           << "Access-Control-Allow-Origin: *\r\n"
           << "Connection: close\r\n\r\n"
           << json_str;
        return ss.str();
    }

    std::string execute_inference_json(const std::string& prompt, int max_new_tokens) {
        std::vector<int> prompt_tokens = tokenizer.encode(prompt);
        std::vector<int> current_tokens = prompt_tokens;

        KVCacheTracker kv_tracker(config.model.n_layers, config.model.n_heads, config.model.d_model / config.model.n_heads);
        LogitLens lens(model->w_unembed);
        AttentionAnalyzer attn_analyzer(config.model.n_heads);
        LogitFilter filter(0.7f, 40, 0.9f);
        std::mt19937 step_rng(42);

        std::vector<std::pair<Tensor, Tensor>> past_kv;
        FullTrace trace;

        // Prefill
        Tensor logits = model->forward(prompt_tokens, past_kv, trace);
        CacheSnapshot snap = kv_tracker.record_step(0, past_kv);

        std::vector<float> last_logits(config.model.vocab_size);
        size_t last_s = logits.shape[1] - 1;
        for (size_t v = 0; v < config.model.vocab_size; ++v) {
            last_logits[v] = logits.at3(0, last_s, v);
        }

        SamplingTelemetry telem;
        int next_token = filter.sample(last_logits, current_tokens, step_rng, telem);

        JsonValue res = JsonValue::object();
        res.set("prompt", prompt);
        res.set("total_steps", max_new_tokens + 1);

        JsonValue frames = JsonValue::array();

        auto make_frame = [&](int step, bool is_prefill, int cur_t, int nxt_t) {
            JsonValue f = JsonValue::object();
            f.set("step", step);
            f.set("is_prefill", is_prefill);
            f.set("token_id", cur_t);
            f.set("token_str", tokenizer.decode({cur_t}));
            f.set("next_token_id", nxt_t);
            f.set("next_token_str", tokenizer.decode({nxt_t}));

            JsonValue kv_obj = JsonValue::object();
            kv_obj.set("step", snap.step);
            kv_obj.set("total_cached_tokens", snap.total_cached_tokens);
            kv_obj.set("memory_kb", snap.memory_kb);
            f.set("kv_cache", kv_obj);

            // Logit lens
            auto preds = lens.analyze_residual_stream(trace.residual_streams);
            JsonValue lens_arr = JsonValue::array();
            for (const auto& p : preds) {
                JsonValue lp = JsonValue::object();
                lp.set("layer_name", p.layer_name);
                lp.set("entropy", p.entropy);
                JsonValue top = JsonValue::object();
                top.set("token_str", p.top_prediction.token_str);
                top.set("probability", p.top_prediction.probability);
                lp.set("top_prediction", top);
                lens_arr.push(lp);
            }
            f.set("logit_lens", lens_arr);

            // MoE routing
            JsonValue moe_arr = JsonValue::array();
            for (const auto& lt : trace.layer_traces) {
                if (lt.is_moe) {
                    JsonValue m = JsonValue::object();
                    m.set("layer_idx", lt.layer_idx);
                    JsonValue topk_idx = JsonValue::array();
                    JsonValue topk_w = JsonValue::array();
                    for (int idx : lt.moe_router.top_k_indices) topk_idx.push(idx);
                    for (float w : lt.moe_router.top_k_weights) topk_w.push(w);
                    m.set("top_k_indices", topk_idx);
                    m.set("top_k_weights", topk_w);
                    moe_arr.push(m);
                }
            }
            f.set("moe_routing", moe_arr);

            return f;
        };

        frames.push(make_frame(0, true, prompt_tokens.back(), next_token));

        // Autoregressive steps
        std::vector<int> generated_ids;
        for (int s = 1; s <= max_new_tokens; ++s) {
            current_tokens.push_back(next_token);
            generated_ids.push_back(next_token);

            logits = model->forward({next_token}, past_kv, trace);
            snap = kv_tracker.record_step(s, past_kv);

            for (size_t v = 0; v < config.model.vocab_size; ++v) {
                last_logits[v] = logits.at3(0, 0, v);
            }
            int sampled = filter.sample(last_logits, current_tokens, step_rng, telem);
            frames.push(make_frame(s, false, next_token, sampled));
            next_token = sampled;
        }

        res.set("frames", frames);
        res.set("generated_text", tokenizer.decode(generated_ids));
        return res.serialize();
    }

    std::string parse_json_string_field(const std::string& json, const std::string& field) {
        std::string search = "\"" + field + "\":";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return "";
        pos += search.size();
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\"')) pos++;
        size_t end = json.find("\"", pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    }

    int parse_json_int_field(const std::string& json, const std::string& field, int default_val) {
        std::string search = "\"" + field + "\":";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return default_val;
        pos += search.size();
        while (pos < json.size() && json[pos] == ' ') pos++;
        try {
            return std::stoi(json.substr(pos));
        } catch (...) {
            return default_val;
        }
    }
};

} // namespace nebula
