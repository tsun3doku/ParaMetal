#include "VoronoiRuntime.hpp"

#include <cmath>

void VoronoiRuntime::invalidateMaterialization() {
    voronoiReady = false;
    seedPositions.clear();
    globalRemeshRuntimeModelIds.clear();
    globalRemeshPositions.clear();
    globalRemeshTriangleIndices.clear();
    globalRemeshSurfacePositions.clear();
    globalRemeshSurfaceTriangleIndices.clear();
    globalVolumePadding = 0.0f;
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
    const std::vector<std::array<float, 16>>& worldToModel,
    float volumePadding) {
    globalRemeshRuntimeModelIds = runtimeModelIds;
    globalRemeshPositions = positions;
    globalRemeshTriangleIndices = triangleIndices;
    globalRemeshSurfacePositions = surfacePositions;
    globalRemeshSurfaceTriangleIndices = surfaceTriangleIndices;
    globalRemeshWorldToModel = worldToModel;
    globalVolumePadding = volumePadding;
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

void VoronoiRuntime::setLloydParams(
    int iterations,
    float volumeThreshold,
    float damping) {
    const int nextIterations = iterations < 0 ? 0 : iterations;
    const float nextThreshold =
        volumeThreshold > 0.0f && std::isfinite(volumeThreshold)
            ? volumeThreshold
            : voronoi::DefaultLloydVolumeThresholdFraction;
    const float nextDamping =
        damping > 0.0f && damping <= 1.0f && std::isfinite(damping)
            ? damping
            : voronoi::DefaultLloydDamping;
    if (lloydIterations == nextIterations &&
        lloydVolumeThreshold == nextThreshold &&
        lloydDamping == nextDamping) {
        return;
    }

    lloydIterations = nextIterations;
    lloydVolumeThreshold = nextThreshold;
    lloydDamping = nextDamping;
    invalidateMaterialization();
}

void VoronoiRuntime::setContactRange(float range) {
    const float nextRange =
        range > 0.0f && std::isfinite(range)
            ? range
            : voronoi::DefaultContactRange;
    if (contactRange == nextRange) {
        return;
    }

    contactRange = nextRange;
    invalidateMaterialization();
}

void VoronoiRuntime::markReady() {
    voronoiReady = true;
}

void VoronoiRuntime::cleanup() {
    invalidateMaterialization();
}
