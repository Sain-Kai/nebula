#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace nebula {

enum class GGUFType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims{0};
    std::vector<uint64_t> shape;
    uint32_t ggml_type{0};
    uint64_t offset{0};
};

struct GGUFMetadata {
    std::string key;
    uint32_t value_type{0};
    std::string str_value;
    int64_t int_value{0};
    double float_value{0.0};
};

struct GGUFHeader {
    uint32_t magic{0};
    uint32_t version{0};
    uint64_t tensor_count{0};
    uint64_t kv_count{0};
    std::vector<GGUFMetadata> metadata;
    std::vector<GGUFTensorInfo> tensors;
    bool is_valid{false};
};

class GGUFParser {
public:
    static constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" in little-endian

    static GGUFHeader parse_buffer(const std::vector<uint8_t>& buf) {
        GGUFHeader header;
        if (buf.size() < 24) return header;

        size_t offset = 0;
        std::memcpy(&header.magic, buf.data() + offset, 4); offset += 4;
        if (header.magic != GGUF_MAGIC) {
            return header;
        }

        std::memcpy(&header.version, buf.data() + offset, 4); offset += 4;
        std::memcpy(&header.tensor_count, buf.data() + offset, 8); offset += 8;
        std::memcpy(&header.kv_count, buf.data() + offset, 8); offset += 8;

        // Parse key-value metadata pairs
        for (uint64_t i = 0; i < header.kv_count && offset < buf.size(); ++i) {
            if (offset + 8 > buf.size()) break;
            uint64_t key_len = 0;
            std::memcpy(&key_len, buf.data() + offset, 8); offset += 8;
            if (offset + key_len > buf.size()) break;

            std::string key(reinterpret_cast<const char*>(buf.data() + offset), key_len);
            offset += key_len;

            if (offset + 4 > buf.size()) break;
            uint32_t val_type = 0;
            std::memcpy(&val_type, buf.data() + offset, 4); offset += 4;

            GGUFMetadata meta;
            meta.key = key;
            meta.value_type = val_type;

            if (val_type == static_cast<uint32_t>(GGUFType::STRING)) {
                if (offset + 8 <= buf.size()) {
                    uint64_t str_len = 0;
                    std::memcpy(&str_len, buf.data() + offset, 8); offset += 8;
                    if (offset + str_len <= buf.size()) {
                        meta.str_value = std::string(reinterpret_cast<const char*>(buf.data() + offset), str_len);
                        offset += str_len;
                    }
                }
            } else if (val_type == static_cast<uint32_t>(GGUFType::UINT32) || val_type == static_cast<uint32_t>(GGUFType::INT32)) {
                if (offset + 4 <= buf.size()) {
                    int32_t val = 0;
                    std::memcpy(&val, buf.data() + offset, 4); offset += 4;
                    meta.int_value = val;
                }
            } else if (val_type == static_cast<uint32_t>(GGUFType::UINT64) || val_type == static_cast<uint32_t>(GGUFType::INT64)) {
                if (offset + 8 <= buf.size()) {
                    int64_t val = 0;
                    std::memcpy(&val, buf.data() + offset, 8); offset += 8;
                    meta.int_value = val;
                }
            } else if (val_type == static_cast<uint32_t>(GGUFType::FLOAT32)) {
                if (offset + 4 <= buf.size()) {
                    float val = 0.0f;
                    std::memcpy(&val, buf.data() + offset, 4); offset += 4;
                    meta.float_value = val;
                }
            }
            header.metadata.push_back(meta);
        }

        // Parse tensor descriptors
        for (uint64_t i = 0; i < header.tensor_count && offset < buf.size(); ++i) {
            if (offset + 8 > buf.size()) break;
            uint64_t name_len = 0;
            std::memcpy(&name_len, buf.data() + offset, 8); offset += 8;
            if (offset + name_len > buf.size()) break;

            std::string name(reinterpret_cast<const char*>(buf.data() + offset), name_len);
            offset += name_len;

            if (offset + 4 > buf.size()) break;
            uint32_t n_dims = 0;
            std::memcpy(&n_dims, buf.data() + offset, 4); offset += 4;

            std::vector<uint64_t> dims(n_dims, 0);
            for (uint32_t d = 0; d < n_dims && offset + 8 <= buf.size(); ++d) {
                std::memcpy(&dims[d], buf.data() + offset, 8); offset += 8;
            }

            if (offset + 12 > buf.size()) break;
            uint32_t type = 0;
            uint64_t t_offset = 0;
            std::memcpy(&type, buf.data() + offset, 4); offset += 4;
            std::memcpy(&t_offset, buf.data() + offset, 8); offset += 8;

            header.tensors.push_back({name, n_dims, dims, type, t_offset});
        }

        header.is_valid = true;
        return header;
    }

    // Helper to generate a valid binary GGUF buffer for tests and mocking
    static std::vector<uint8_t> create_dummy_gguf(const std::string& arch = "llama", uint64_t num_tensors = 2) {
        std::vector<uint8_t> buf;
        auto append_u32 = [&](uint32_t val) {
            uint8_t b[4];
            std::memcpy(b, &val, 4);
            buf.insert(buf.end(), b, b + 4);
        };
        auto append_u64 = [&](uint64_t val) {
            uint8_t b[8];
            std::memcpy(b, &val, 8);
            buf.insert(buf.end(), b, b + 8);
        };
        auto append_str = [&](const std::string& s) {
            append_u64(s.size());
            buf.insert(buf.end(), s.begin(), s.end());
        };

        append_u32(GGUF_MAGIC);
        append_u32(3); // version 3
        append_u64(num_tensors);
        append_u64(1); // 1 kv pair: general.architecture

        // KV pair: general.architecture = arch
        append_str("general.architecture");
        append_u32(static_cast<uint32_t>(GGUFType::STRING));
        append_str(arch);

        // Tensors
        for (uint64_t i = 0; i < num_tensors; ++i) {
            std::string t_name = "blk." + std::to_string(i) + ".attn_q.weight";
            append_str(t_name);
            append_u32(2); // 2 dimensions
            append_u64(128); // dim 0
            append_u64(128); // dim 1
            append_u32(0); // F32
            append_u64(i * 1024); // mock data offset
        }

        return buf;
    }
};

} // namespace nebula
