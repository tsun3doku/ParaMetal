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
    float getGlobalSdfPadding() const { return globalSdfPadding; }

    std::vector<glm::vec4>& getSeedPositions() { return seedPositions; }
    const std::vector<glm::vec4>& getSeedPositions() const { return seedPositions; }
    const std::array<glm::vec3, 8>& getPointDomainCorners() const { return pointDomainCorners; }

    void clearGeometry();
    void setParams(float cellSize, int voxelResolution);
    float getCellSize() const { return cellSize; }
    int getVoxelResolution() const { return voxelResolution; }
    float getContactRange() const { return contactRange; }
    void setContactRange(float range) { contactRange = range; }
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
        float sdfPadding);

    void cleanup();

private:
    void invalidateMaterialization();

    float cellSize = 0.005f;
    int voxelResolution = 128;
    float contactRange = voronoi::DefaultContactRange;

    std::vector<glm::vec4> seedPositions;
    std::array<glm::vec3, 8> pointDomainCorners{};
    std::vector<uint32_t> globalRemeshRuntimeModelIds;
    std::vector<std::vector<glm::vec3>> globalRemeshPositions;
    std::vector<std::vector<uint32_t>> globalRemeshTriangleIndices;
    std::vector<std::vector<glm::vec3>> globalRemeshSurfacePositions;
    std::vector<std::vector<uint32_t>> globalRemeshSurfaceTriangleIndices;
    float globalSdfPadding = 0.0f;

    bool voronoiReady = false;
};
