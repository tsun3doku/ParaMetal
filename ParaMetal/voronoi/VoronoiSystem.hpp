#pragma once

#include "voronoi/VoronoiRuntime.hpp"
#include "voronoi/VoronoiSystemBuildStage.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

class MemoryAllocator;
class VulkanDevice;
class CommandPool;
class VoronoiCandidateCompute;

class VoronoiSystem {
public:
    VoronoiSystem(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& commandPool);
    ~VoronoiSystem();

    bool isReady() const { return runtime.isReady(); }

    void setSeedPositions(const std::vector<glm::vec4>& positions, const std::array<glm::vec3, 8>& domainCorners);
    void setGlobalGeometry(
        const std::vector<uint32_t>& runtimeModelIds,
        const std::vector<std::vector<glm::vec3>>& positions,
        const std::vector<std::vector<uint32_t>>& triangleIndices,
        const std::vector<std::vector<glm::vec3>>& surfacePositions,
        const std::vector<std::vector<uint32_t>>& surfaceTriangleIndices,
        const std::vector<std::array<float, 16>>& worldToModel,
        float volumePadding);
    void clearGeometry();
    void setParams(float cellSize, int voxelResolution);
    void setLloydParams(int iterations, float volumeThreshold, float damping);
    void setContactRange(float range);
    bool ensureConfigured();

    VoronoiSystemBuildStage& getBuildStage() { return *voronoiSystemBuildStage; }

private:
    bool rebuildVoronoiRuntime();
    void dispatchVoronoiCandidateUpdates();

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;
    VoronoiRuntime runtime;

    std::unique_ptr<VoronoiSystemBuildStage> voronoiSystemBuildStage;
    std::unique_ptr<VoronoiCandidateCompute> voronoiCandidateCompute;
};
