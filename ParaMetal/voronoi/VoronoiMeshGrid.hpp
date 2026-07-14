#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "spatial/TriangleHashGrid.hpp"
#include "spatial/VoxelGrid.hpp"
#include "util/GeometryUtils.hpp"
#include "util/Structs.hpp"

class VoronoiMeshGrid {
public:
    VoronoiMeshGrid();
    ~VoronoiMeshGrid();

    void buildGrids(
        const std::vector<glm::vec3>& positions,
        const std::vector<uint32_t>& indices,
        float targetCellSize,
        VoxelGrid& voxelGrid,
        int voxelResolution);

    glm::ivec3 getDimensions() const { return gridDim; }
    float getCellSize() const { return cellSize; }
    glm::vec3 getGridMin() const { return gridMin; }
    glm::vec3 getGridMax() const { return gridMax; }
    float sampleSDFGrid(const glm::vec3& pos) const;

private:
    TriangleHashGrid triHashGrid;
    std::vector<float> sdfGrid;
    glm::ivec3 sdfGridDim;
    float sdfCellSize;

    glm::ivec3 gridDim;
    glm::vec3 gridMin;
    glm::vec3 gridMax;
    float cellSize;

    void buildSDFGrid(
        const std::vector<glm::vec3>& positions,
        const std::vector<uint32_t>& indices,
        const TriangleHashGrid& triangleGrid);
    float computePointToMeshDistance(
        const glm::vec3& point,
        const std::vector<glm::vec3>& positions,
        const std::vector<uint32_t>& indices) const;
};
