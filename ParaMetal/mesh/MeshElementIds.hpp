#pragma once

#include <algorithm>
#include <cstdint>

inline uint64_t packMeshEdge(uint32_t a, uint32_t b) {
    const uint32_t lo = std::min(a, b);
    const uint32_t hi = std::max(a, b);
    return (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi);
}

inline void unpackMeshEdge(uint64_t key, uint32_t& a, uint32_t& b) {
    a = static_cast<uint32_t>(key >> 32);
    b = static_cast<uint32_t>(key & 0xffffffffu);
}
