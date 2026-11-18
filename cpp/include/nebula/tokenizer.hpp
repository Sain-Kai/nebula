#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <cstdint>

namespace nebula {

struct TokenMetadata {
    int index = 0;
    int token_id = 0;
    std::string text;
    std::vector<uint8_t> byte_values;
    std::string hex_str;
    bool is_special = false;
    int start_char = 0;
    int end_char = 0;
};

class NebulaTokenizer {
public:
    static constexpr int BOS_TOKEN_ID = 256;
    static constexpr int EOS_TOKEN_ID = 257;
    static constexpr int PAD_TOKEN_ID = 258;
    static constexpr int MASK_TOKEN_ID = 259;
    static constexpr int VOCAB_SIZE = 320;

    std::vector<int> encode(const std::string& text, bool add_bos = true, bool add_eos = false) const {
        std::vector<int> token_ids;
        if (add_bos) token_ids.push_back(BOS_TOKEN_ID);

        for (unsigned char c : text) {
            token_ids.push_back(static_cast<int>(c));
        }

        if (add_eos) token_ids.push_back(EOS_TOKEN_ID);
        return token_ids;
    }

    std::string decode(const std::vector<int>& token_ids) const {
        std::string result;
        for (int tid : token_ids) {
            if (tid == BOS_TOKEN_ID) {
                result += "<|bos|>";
            } else if (tid == EOS_TOKEN_ID) {
                result += "<|eos|>";
            } else if (tid == PAD_TOKEN_ID) {
                result += "<|pad|>";
            } else if (tid == MASK_TOKEN_ID) {
                result += "<|mask|>";
            } else if (tid >= 0 && tid < 256) {
                result += static_cast<char>(tid);
            } else {
                result += "<|token_" + std::to_string(tid) + "|>";
            }
        }
        return result;
    }

    std::vector<TokenMetadata> analyze(const std::string& text, bool add_bos = true) const {
        std::vector<TokenMetadata> metadata;
        int current_idx = 0;

        if (add_bos) {
            TokenMetadata bos_meta;
            bos_meta.index = current_idx++;
            bos_meta.token_id = BOS_TOKEN_ID;
            bos_meta.text = "<|bos|>";
            bos_meta.is_special = true;
            bos_meta.start_char = 0;
            bos_meta.end_char = 0;
            metadata.push_back(bos_meta);
        }

        int char_offset = 0;
        for (size_t i = 0; i < text.size(); ++i) {
            uint8_t byte_val = static_cast<uint8_t>(text[i]);
            TokenMetadata meta;
            meta.index = current_idx++;
            meta.token_id = byte_val;
            meta.byte_values.push_back(byte_val);

            std::stringstream ss;
            ss << "0x" << std::uppercase << std::setfill('0') << std::setw(2) << std::hex << static_cast<int>(byte_val);
            meta.hex_str = ss.str();

            if (byte_val >= 32 && byte_val <= 126 && byte_val != ' ') {
                meta.text = std::string(1, static_cast<char>(byte_val));
            } else if (byte_val == ' ') {
                meta.text = "␣";
            } else if (byte_val == '\n') {
                meta.text = "↵";
            } else if (byte_val == '\t') {
                meta.text = "⇥";
            } else {
                meta.text = meta.hex_str;
            }

            meta.is_special = false;
            meta.start_char = char_offset;
            meta.end_char = char_offset + 1;
            char_offset++;

            metadata.push_back(meta);
        }

        return metadata;
    }
};

} // namespace nebula
