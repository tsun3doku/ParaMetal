#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

class TriangleHashGrid;

class VoxelGrid {
public:
    struct VoxelGridParams {
        alignas(16) glm::vec3 gridMin;
        float cellSize;
        alignas(16) glm::ivec3 gridDim;
        uint32_t totalCells;
    };

    VoxelGrid();
    ~VoxelGrid();

    void build(
        const std::vector<glm::vec3>& positions,
        const std::vector<uint32_t>& indices,
        const TriangleHashGrid& triangleGrid,
        int gridSize);
    glm::vec3 toCanonical(const glm::vec3& worldPos) const;

    const VoxelGridParams& getParams() const { return params; }

    const std::vector<int32_t>& getTrianglesList() const { return trianglesList; }
    const std::vector<int32_t>& getOffsets() const { return offsets; }

private:
    static constexpr float CANONICAL_DOMAIN_SIZE = 1000.0f;
    static constexpr float CANONICAL_SCALE = 990.222f;
    static constexpr float CANONICAL_BIAS = 4.998f;
    static int clampInt(int v, int lo, int hi) { return std::max(lo, std::min(v, hi)); }
    
    static bool triBoxOverlap(const float boxcenter[3], const float boxhalfsize[3], const float triverts[3][3]);
    void buildTriangleLists(
        const std::vector<glm::vec3>& positions,
        const std::vector<uint32_t>& indices,
        const TriangleHashGrid& triangleGrid);

    VoxelGridParams params;  
    
    std::vector<int32_t> trianglesList;
    std::vector<int32_t> offsets;
};
