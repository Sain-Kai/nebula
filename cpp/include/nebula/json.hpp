#pragma once

#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <iomanip>

namespace nebula {

class JsonValue {
public:
    enum Type { Null, Bool, Number, String, Array, Object };

    Type type = Null;
    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val;
    std::vector<JsonValue> arr_val;
    std::vector<std::pair<std::string, JsonValue>> obj_val;

    JsonValue() : type(Null) {}
    JsonValue(bool b) : type(Bool), bool_val(b) {}
    JsonValue(int n) : type(Number), num_val(n) {}
    JsonValue(size_t n) : type(Number), num_val(static_cast<double>(n)) {}
    JsonValue(float n) : type(Number), num_val(n) {}
    JsonValue(double n) : type(Number), num_val(n) {}
    JsonValue(const char* s) : type(String), str_val(s) {}
    JsonValue(const std::string& s) : type(String), str_val(s) {}

    static JsonValue array() {
        JsonValue j;
        j.type = Array;
        return j;
    }

    static JsonValue object() {
        JsonValue j;
        j.type = Object;
        return j;
    }

    void push(const JsonValue& v) {
        if (type == Array) arr_val.push_back(v);
    }

    void set(const std::string& key, const JsonValue& v) {
        if (type == Object) {
            for (auto& pair : obj_val) {
                if (pair.first == key) {
                    pair.second = v;
                    return;
                }
            }
            obj_val.push_back({key, v});
        }
    }

    std::string serialize() const {
        std::stringstream ss;
        serialize_to(ss);
        return ss.str();
    }

private:
    void serialize_to(std::stringstream& ss) const {
        switch (type) {
            case Null: ss << "null"; break;
            case Bool: ss << (bool_val ? "true" : "false"); break;
            case Number:
                if (num_val == static_cast<long long>(num_val)) {
                    ss << static_cast<long long>(num_val);
                } else {
                    ss << std::fixed << std::setprecision(4) << num_val;
                }
                break;
            case String:
                ss << "\"";
                for (char c : str_val) {
                    if (c == '"') ss << "\\\"";
                    else if (c == '\\') ss << "\\\\";
                    else if (c == '\n') ss << "\\n";
                    else if (c == '\r') ss << "\\r";
                    else if (c == '\t') ss << "\\t";
                    else ss << c;
                }
                ss << "\"";
                break;
            case Array:
                ss << "[";
                for (size_t i = 0; i < arr_val.size(); ++i) {
                    arr_val[i].serialize_to(ss);
                    if (i + 1 < arr_val.size()) ss << ",";
                }
                ss << "]";
                break;
            case Object:
                ss << "{";
                for (size_t i = 0; i < obj_val.size(); ++i) {
                    ss << "\"" << obj_val[i].first << "\":";
                    obj_val[i].second.serialize_to(ss);
                    if (i + 1 < obj_val.size()) ss << ",";
                }
                ss << "}";
                break;
        }
    }
};

} // namespace nebula
