#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>
#include "VoronoiConfig.hpp"
#include "VoronoiGpuStructs.hpp"

class VoronoiRuntime {
public:
    bool isReady() const { return voronoiReady; }

    const std::vector<uint32_t>& getGlobalRemeshRuntimeModelIds() const { return globalRemeshRuntimeModelIds; }
    const std::vector<std::vector<glm::vec3>>& getGlobalRemeshPositions() const { return globalRemeshPositions; }
    const std::vector<std::vector<uint32_t>>& getGlobalRemeshTriangleIndices() const { return globalRemeshTriangleIndices; }
    const std::vector<std::vector<glm::vec3>>& getGlobalRemeshSurfacePositions() const { return globalRemeshSurfacePositions; }
    const std::vector<std::vector<uint32_t>>& getGlobalRemeshSurfaceTriangleIndices() const { return globalRemeshSurfaceTriangleIndices; }
    float getGlobalVolumePadding() const { return globalVolumePadding; }

    std::vector<glm::vec4>& getSeedPositions() { return seedPositions; }
    const std::vector<glm::vec4>& getSeedPositions() const { return seedPositions; }
    const std::array<glm::vec3, 8>& getPointDomainCorners() const { return pointDomainCorners; }

    void clearGeometry();
    void setParams(int voxelResolution);
    int getVoxelResolution() const { return voxelResolution; }
    int getLloydIterations() const { return lloydIterations; }
    float getLloydVolumeThreshold() const { return lloydVolumeThreshold; }
    float getLloydDamping() const { return lloydDamping; }
    float getContactRange() const { return contactRange; }
    void setLloydParams(int iterations, float volumeThreshold, float damping);
    void setContactRange(float range);
    void markReady();

    void setSeedPositions(
        const std::vector<glm::vec4>& positions,
        const std::array<glm::vec3, 8>& domainCorners);
    void setGlobalGeometry(
        const std::vector<uint32_t>& runtimeModelIds,
        const std::vector<std::vector<glm::vec3>>& positions,
        const std::vector<std::vector<uint32_t>>& triangleIndices,
        const std::vector<std::vector<glm::vec3>>& surfacePositions,
        const std::vector<std::vector<uint32_t>>& surfaceTriangleIndices,
        float volumePadding);

    void cleanup();

private:
    void invalidateMaterialization();

    int voxelResolution = 128;
    int lloydIterations = int(voronoi::DefaultLloydIterations);
    float lloydVolumeThreshold = voronoi::DefaultLloydVolumeThresholdFraction;
    float lloydDamping = voronoi::DefaultLloydDamping;
    float contactRange = voronoi::DefaultContactRange;

    std::vector<glm::vec4> seedPositions;
    std::array<glm::vec3, 8> pointDomainCorners{};
    std::vector<uint32_t> globalRemeshRuntimeModelIds;
    std::vector<std::vector<glm::vec3>> globalRemeshPositions;
    std::vector<std::vector<uint32_t>> globalRemeshTriangleIndices;
    std::vector<std::vector<glm::vec3>> globalRemeshSurfacePositions;
    std::vector<std::vector<uint32_t>> globalRemeshSurfaceTriangleIndices;
    float globalVolumePadding = 0.0f;

    bool voronoiReady = false;
};
