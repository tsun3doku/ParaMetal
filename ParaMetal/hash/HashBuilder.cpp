#include "HashBuilder.hpp"

#include <cstring>

uint64_t HashBuilder::start() {
    return hashOffset;
}

void HashBuilder::combine(uint64_t& hash, uint64_t value) {
    hash ^= value;
    hash *= hashPrime;
}

void HashBuilder::combineFloat(uint64_t& hash, float value) {
    static_assert(sizeof(float) == sizeof(uint32_t), "float size mismatch");
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    combine(hash, bits);
}

void HashBuilder::combineString(uint64_t& hash, const std::string& value) {
    for (unsigned char ch : value) {
        combine(hash, ch);
    }
}

void HashBuilder::combineBytes(uint64_t& hash, const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        combine(hash, static_cast<uint64_t>(bytes[i]));
    }
}
