#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <glm/glm.hpp>

inline uint32_t spread3(uint32_t x) {
    x &= 0x3FF;
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8))  & 0x0300F00F;
    x = (x | (x << 4))  & 0x030C30C3;
    x = (x | (x << 2))  & 0x09249249;
    return x;
}

inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    return spread3(x) | (spread3(y) << 1) | (spread3(z) << 2);
}

inline std::vector<uint32_t> computeMortonPermutation(const std::vector<glm::vec4>& positions) {
    if (positions.empty()) {
        return {};
    }

    glm::vec3 minPos(std::numeric_limits<float>::max());
    glm::vec3 maxPos(std::numeric_limits<float>::lowest());
    for (const auto& p : positions) {
        minPos = glm::min(minPos, glm::vec3(p));
        maxPos = glm::max(maxPos, glm::vec3(p));
    }

    glm::vec3 extent = maxPos - minPos;
    const float invExtent = 1.0f / std::max({extent.x, extent.y, extent.z, 1e-12f});

    const size_t count = positions.size();
    std::vector<uint32_t> mortonCodes(count);
    for (size_t i = 0; i < count; ++i) {
        glm::vec3 t = (glm::vec3(positions[i]) - minPos) * invExtent;
        uint32_t ix = static_cast<uint32_t>(t.x * 1023.0f) & 0x3FF;
        uint32_t iy = static_cast<uint32_t>(t.y * 1023.0f) & 0x3FF;
        uint32_t iz = static_cast<uint32_t>(t.z * 1023.0f) & 0x3FF;
        mortonCodes[i] = morton3D(ix, iy, iz);
    }

    std::vector<uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return mortonCodes[a] < mortonCodes[b];
    });

    std::vector<uint32_t> oldToNew(count);
    for (uint32_t newIdx = 0; newIdx < static_cast<uint32_t>(count); ++newIdx) {
        oldToNew[order[newIdx]] = newIdx;
    }

    return oldToNew;
}
