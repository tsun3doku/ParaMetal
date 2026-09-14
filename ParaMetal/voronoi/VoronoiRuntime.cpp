#include "VoronoiRuntime.hpp"

void VoronoiRuntime::invalidateMaterialization() {
    voronoiReady = false;
    seedPositions.clear();
    globalRemeshRuntimeModelIds.clear();
    globalRemeshPositions.clear();
    globalRemeshTriangleIndices.clear();
    globalRemeshSurfacePositions.clear();
    globalRemeshSurfaceTriangleIndices.clear();
    globalSdfPadding = 0.0f;
}

void VoronoiRuntime::setSeedPositions(
    const std::vector<glm::vec4>& positions,
    const std::array<glm::vec3, 8>& domainCorners) {
    seedPositions = positions;
    pointDomainCorners = domainCorners;
}

void VoronoiRuntime::setGlobalGeometry(
    const std::vector<uint32_t>& runtimeModelIds,
    const std::vector<std::vector<glm::vec3>>& positions,
    const std::vector<std::vector<uint32_t>>& triangleIndices,
    const std::vector<std::vector<glm::vec3>>& surfacePositions,
    const std::vector<std::vector<uint32_t>>& surfaceTriangleIndices,
    float sdfPadding) {
    globalRemeshRuntimeModelIds = runtimeModelIds;
    globalRemeshPositions = positions;
    globalRemeshTriangleIndices = triangleIndices;
    globalRemeshSurfacePositions = surfacePositions;
    globalRemeshSurfaceTriangleIndices = surfaceTriangleIndices;
    globalSdfPadding = sdfPadding;
    voronoiReady = false;
}

void VoronoiRuntime::clearGeometry() {
    invalidateMaterialization();
}

void VoronoiRuntime::setParams(float updatedCellSize, int updatedVoxelResolution) {
    if (cellSize == updatedCellSize && voxelResolution == updatedVoxelResolution) {
        return;
    }

    cellSize = updatedCellSize;
    voxelResolution = updatedVoxelResolution;
    invalidateMaterialization();
}

void VoronoiRuntime::markReady() {
    voronoiReady = true;
}

void VoronoiRuntime::cleanup() {
    invalidateMaterialization();
}
