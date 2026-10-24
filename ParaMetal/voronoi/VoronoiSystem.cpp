#include "VoronoiSystem.hpp"

#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "voronoi/VoronoiCandidateCompute.hpp"

#include <iostream>

VoronoiSystem::VoronoiSystem(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& commandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      commandPool(commandPool) {
    voronoiSystemBuildStage = std::make_unique<VoronoiSystemBuildStage>(vulkanDevice, memoryAllocator, commandPool);
    voronoiCandidateCompute = std::make_unique<VoronoiCandidateCompute>(vulkanDevice, commandPool);
    if (!voronoiCandidateCompute->initialize()) {
        voronoiCandidateCompute.reset();
    }
}

VoronoiSystem::~VoronoiSystem() = default;

void VoronoiSystem::setSeedPositions(
    const std::vector<glm::vec4>& positions,
    const std::array<glm::vec3, 8>& domainCorners) {
    runtime.setSeedPositions(positions, domainCorners);
}

void VoronoiSystem::setGlobalGeometry(
    const std::vector<uint32_t>& runtimeModelIds,
    const std::vector<std::vector<glm::vec3>>& positions,
    const std::vector<std::vector<uint32_t>>& triangleIndices,
    const std::vector<std::vector<glm::vec3>>& surfacePositions,
    const std::vector<std::vector<uint32_t>>& surfaceTriangleIndices,
    float volumePadding) {
    runtime.setGlobalGeometry(runtimeModelIds, positions, triangleIndices, surfacePositions, surfaceTriangleIndices, volumePadding);
}

void VoronoiSystem::clearGeometry() {
    runtime.clearGeometry();
}

void VoronoiSystem::setParams(float cellSize, int voxelResolution) {
    runtime.setParams(cellSize, voxelResolution);
}

void VoronoiSystem::setContactRange(float range) {
    runtime.setContactRange(range);
}

bool VoronoiSystem::ensureConfigured() {
    if (!runtime.isReady()) {
        if (!rebuildVoronoiRuntime()) {
            std::cerr << "[VoronoiSystem] rebuildVoronoiRuntime returned false" << std::endl;
            return false;
        }
    }

    dispatchVoronoiCandidateUpdates();
    return true;
}

bool VoronoiSystem::rebuildVoronoiRuntime() {
    if (!voronoiSystemBuildStage->buildGlobalDomain(runtime)) {
        std::cerr << "[VoronoiSystem] buildGlobalDomain returned false" << std::endl;
        return false;
    }
    if (!voronoiSystemBuildStage->buildGlobalDisplayCandidates(runtime)) {
        std::cerr << "[VoronoiSystem] buildGlobalDisplayCandidates returned false\n";
        return false;
    }
    runtime.markReady();
    return true;
}

void VoronoiSystem::dispatchVoronoiCandidateUpdates() {
    if (!voronoiCandidateCompute || !voronoiSystemBuildStage || voronoiSystemBuildStage->getCandidateNodeCount() == 0) {
        return;
    }

    const auto& faceCounts = voronoiSystemBuildStage->getGlobalDisplayFaceCounts();
    const auto& vertexBuffers = voronoiSystemBuildStage->getGlobalDisplayVertexBuffers();
    const auto& vertexBufferOffsets = voronoiSystemBuildStage->getGlobalDisplayVertexBufferOffsets();
    const auto& faceIndexBuffers = voronoiSystemBuildStage->getGlobalDisplayFaceIndexBuffers();
    const auto& faceIndexBufferOffsets = voronoiSystemBuildStage->getGlobalDisplayFaceIndexBufferOffsets();
    const auto& candidateBuffers = voronoiSystemBuildStage->getGlobalDisplayCandidateBuffers();
    const auto& candidateBufferOffsets = voronoiSystemBuildStage->getGlobalDisplayCandidateBufferOffsets();

    const VkBuffer seedPositionBuffer = voronoiSystemBuildStage->getSeedPositionBuffer();
    const VkDeviceSize seedPositionBufferOffset = voronoiSystemBuildStage->getSeedPositionBufferOffset();
    const uint32_t seedCount = voronoiSystemBuildStage->getCandidateNodeCount();

    for (size_t i = 0; i < faceCounts.size(); ++i) {
        if (faceCounts[i] == 0 || candidateBuffers[i] == VK_NULL_HANDLE) {
            continue;
        }

        voronoiCandidateCompute->updateDescriptors(
            vertexBuffers[i],
            vertexBufferOffsets[i],
            faceIndexBuffers[i],
            faceIndexBufferOffsets[i],
            seedPositionBuffer,
            seedPositionBufferOffset,
            candidateBuffers[i],
            candidateBufferOffsets[i]);

        voronoiCandidateCompute->dispatch(faceCounts[i], seedCount);
    }
}
