#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

struct NodeDataBlock;

class NodeGraphHash {
public:
    static uint64_t start();
    static void combine(uint64_t& hash, uint64_t value);
    static void combineFloat(uint64_t& hash, float value);
    static void combineString(uint64_t& hash, const std::string& value);
    static void combineInputHash(uint64_t& hash, const NodeDataBlock* input);
    static void combineBytes(uint64_t& hash, const void* data, size_t size);

    template <typename T>
    static void combinePod(uint64_t& hash, const T& value) {
        static_assert(std::is_trivially_copyable_v<T>, "combinePod requires trivially copyable type");
        combineBytes(hash, &value, sizeof(T));
    }

    template <typename T>
    static void combinePodVector(uint64_t& hash, const std::vector<T>& values) {
        static_assert(std::is_trivially_copyable_v<T>, "combinePodVector requires trivially copyable type");
        combine(hash, static_cast<uint64_t>(values.size()));
        if (!values.empty()) {
            combineBytes(hash, values.data(), sizeof(T) * values.size());
        }
    }

private:
    static constexpr uint64_t hashOffset = 1469598103934665603ull;
    static constexpr uint64_t hashPrime = 1099511628211ull;
};
