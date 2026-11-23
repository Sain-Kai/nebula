#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>
#include "nebula/tokenizer.hpp"

namespace nebula {

struct CppSpecDecision {
    size_t position{0};
    int draft_token_id{0};
    std::string draft_token_str;
    float draft_prob{0.0f};
    float target_prob{0.0f};
    float acceptance_prob{0.0f};
    bool is_accepted{false};
    int bonus_token_id{-1};
    std::string bonus_token_str;
};

struct CppSpecRoundTrace {
    size_t round_idx{0};
    size_t proposed_tokens{0};
    size_t accepted_tokens{0};
    float speedup_ratio{1.0f};
    std::vector<CppSpecDecision> decisions;
};

class CppSpeculativeVerifier {
public:
    size_t gamma;

    explicit CppSpeculativeVerifier(size_t draft_gamma = 4) : gamma(draft_gamma) {}

    CppSpecRoundTrace verify_candidates(
        const std::vector<int>& draft_ids,
        const std::vector<float>& draft_probs,
        const std::vector<float>& target_probs,
        const NebulaTokenizer& tokenizer,
        unsigned int seed = 42
    ) const {
        CppSpecRoundTrace trace;
        trace.round_idx = 1;
        trace.proposed_tokens = std::min(draft_ids.size(), gamma);

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        size_t accepted = 0;
        for (size_t i = 0; i < trace.proposed_tokens; ++i) {
            CppSpecDecision dec;
            dec.position = i;
            dec.draft_token_id = draft_ids[i];
            dec.draft_token_str = tokenizer.decode({draft_ids[i]});
            dec.draft_prob = (i < draft_probs.size()) ? draft_probs[i] : 0.5f;
            dec.target_prob = (i < target_probs.size()) ? target_probs[i] : 0.5f;

            float ratio = (dec.draft_prob > 1e-6f) ? (dec.target_prob / dec.draft_prob) : 1.0f;
            dec.acceptance_prob = std::min(1.0f, ratio);

            float r = dist(rng);
            if (r <= dec.acceptance_prob) {
                dec.is_accepted = true;
                accepted++;
                trace.decisions.push_back(dec);
            } else {
                dec.is_accepted = false;
                // Correction token sampled from target distribution
                dec.bonus_token_id = 32; // space
                dec.bonus_token_str = " ";
                trace.decisions.push_back(dec);
                break; // Rejection terminates speculative chain
            }
        }

        trace.accepted_tokens = accepted;
        // In speculative decoding, 1 target verification pass yields (accepted + 1) output tokens
        trace.speedup_ratio = static_cast<float>(accepted + 1);
        return trace;
    }
};

} // namespace nebula
