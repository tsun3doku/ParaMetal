#include "VoronoiMeshGrid.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>

VoronoiMeshGrid::VoronoiMeshGrid() {
}

VoronoiMeshGrid::~VoronoiMeshGrid() {
}

void VoronoiMeshGrid::buildGrids(
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>& indices,
    float targetCellSize,
    VoxelGrid& voxelGrid,
    int voxelResolution) {
    cellSize = targetCellSize;

    if (positions.empty() || indices.empty()) {
        std::cerr << "[VoronoiMeshGrid] Empty mesh" << std::endl;
        return;
    }

    glm::vec3 minBounds(FLT_MAX);
    glm::vec3 maxBounds(-FLT_MAX);

    for (const glm::vec3& position : positions) {
        minBounds = glm::min(minBounds, position);
        maxBounds = glm::max(maxBounds, position);
    }
    float padding = cellSize * 3.0f;

    gridMin = minBounds - glm::vec3(padding);
    gridMax = maxBounds + glm::vec3(padding);

    glm::vec3 gridSize = gridMax - gridMin;
    gridDim.x = std::max(2, static_cast<int>(std::ceil(gridSize.x / cellSize)));
    gridDim.y = std::max(2, static_cast<int>(std::ceil(gridSize.y / cellSize)));
    gridDim.z = std::max(2, static_cast<int>(std::ceil(gridSize.z / cellSize)));

    gridMax = gridMin + glm::vec3(gridDim) * cellSize;

    TriangleHashGrid sharedTriGrid;
    sharedTriGrid.build(positions, indices, gridMin, gridMax, cellSize);

    buildSDFGrid(positions, indices, sharedTriGrid);

    voxelGrid.build(positions, indices, sharedTriGrid, voxelResolution);
}

float VoronoiMeshGrid::computePointToMeshDistance(
    const glm::vec3& point,
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>& indices) const {
    float minDistSq = FLT_MAX;

    std::vector<size_t> nearbyTriangles;
    triHashGrid.getNearbyTriangles(point, nearbyTriangles);
    if (nearbyTriangles.empty()) {
        const int fallbackRadius = 4;
        triHashGrid.getNearbyTriangles(point, fallbackRadius, nearbyTriangles);
    }

    for (size_t triIdx : nearbyTriangles) {
        const size_t indexBase = triIdx * 3;
        if (indexBase + 2 >= indices.size()) {
            continue;
        }
        if (indices[indexBase] >= positions.size() ||
            indices[indexBase + 1] >= positions.size() ||
            indices[indexBase + 2] >= positions.size()) {
            continue;
        }

        const glm::vec3& v0 = positions[indices[indexBase]];
        const glm::vec3& v1 = positions[indices[indexBase + 1]];
        const glm::vec3& v2 = positions[indices[indexBase + 2]];

        glm::vec3 closestPoint = closestPointOnTriangle(point, v0, v1, v2);
        float distSq = glm::distance2(point, closestPoint);

        minDistSq = std::min(minDistSq, distSq);
    }

    return std::sqrt(minDistSq);
}

void VoronoiMeshGrid::buildSDFGrid(
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>& indices,
    const TriangleHashGrid& triangleGrid) {
    sdfGridDim = gridDim;
    sdfCellSize = cellSize;

    size_t totalCells = static_cast<size_t>(sdfGridDim.x) * sdfGridDim.y * sdfGridDim.z;
    sdfGrid.resize(totalCells);

    triHashGrid = triangleGrid;

    for (int z = 0; z < sdfGridDim.z; z++) {
        for (int y = 0; y < sdfGridDim.y; y++) {
            for (int x = 0; x < sdfGridDim.x; x++) {
                glm::vec3 gridPoint = gridMin + glm::vec3(x, y, z) * sdfCellSize;

                float unsignedDist = computePointToMeshDistance(gridPoint, positions, indices);

                size_t idx = z * sdfGridDim.y * sdfGridDim.x + y * sdfGridDim.x + x;
                sdfGrid[idx] = unsignedDist;
            }
        }
    }
}

float VoronoiMeshGrid::sampleSDFGrid(const glm::vec3& pos) const {
    glm::vec3 gridPos = (pos - gridMin) / sdfCellSize;

    gridPos.x = glm::clamp(gridPos.x, 0.0f, static_cast<float>(sdfGridDim.x - 1));
    gridPos.y = glm::clamp(gridPos.y, 0.0f, static_cast<float>(sdfGridDim.y - 1));
    gridPos.z = glm::clamp(gridPos.z, 0.0f, static_cast<float>(sdfGridDim.z - 1));

    int x0 = static_cast<int>(std::floor(gridPos.x));
    int y0 = static_cast<int>(std::floor(gridPos.y));
    int z0 = static_cast<int>(std::floor(gridPos.z));
    int x1 = std::min(x0 + 1, sdfGridDim.x - 1);
    int y1 = std::min(y0 + 1, sdfGridDim.y - 1);
    int z1 = std::min(z0 + 1, sdfGridDim.z - 1);

    float fx = gridPos.x - x0;
    float fy = gridPos.y - y0;
    float fz = gridPos.z - z0;

    float c000 = sdfGrid[z0 * sdfGridDim.y * sdfGridDim.x + y0 * sdfGridDim.x + x0];
    float c100 = sdfGrid[z0 * sdfGridDim.y * sdfGridDim.x + y0 * sdfGridDim.x + x1];
    float c010 = sdfGrid[z0 * sdfGridDim.y * sdfGridDim.x + y1 * sdfGridDim.x + x0];
    float c110 = sdfGrid[z0 * sdfGridDim.y * sdfGridDim.x + y1 * sdfGridDim.x + x1];
    float c001 = sdfGrid[z1 * sdfGridDim.y * sdfGridDim.x + y0 * sdfGridDim.x + x0];
    float c101 = sdfGrid[z1 * sdfGridDim.y * sdfGridDim.x + y0 * sdfGridDim.x + x1];
    float c011 = sdfGrid[z1 * sdfGridDim.y * sdfGridDim.x + y1 * sdfGridDim.x + x0];
    float c111 = sdfGrid[z1 * sdfGridDim.y * sdfGridDim.x + y1 * sdfGridDim.x + x1];

    float c00 = c000 * (1 - fx) + c100 * fx;
    float c01 = c001 * (1 - fx) + c101 * fx;
    float c10 = c010 * (1 - fx) + c110 * fx;
    float c11 = c011 * (1 - fx) + c111 * fx;

    float c0 = c00 * (1 - fy) + c10 * fy;
    float c1 = c01 * (1 - fy) + c11 * fy;

    return c0 * (1 - fz) + c1 * fz;
}
