#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include "vulkan/VulkanExternalBuffer.hpp"
#include "cuda/CudaExternalBuffer.hpp"

class MemoryAllocator;
class VulkanDevice;
class CommandPool;
class VoronoiRuntime;
struct VoronoiProduct;

namespace voronoi {
class KNNNeighbors;
class RVD;
class VoronoiCandidate;
}

class VoronoiSystemBuildStage {
  public:
    VoronoiSystemBuildStage(VulkanDevice &vulkanDevice, MemoryAllocator &memoryAllocator, CommandPool &commandPool);
    ~VoronoiSystemBuildStage();

    bool buildGlobalDomain(VoronoiRuntime &runtime);
    bool buildGlobalDisplayCandidates(VoronoiRuntime &runtime);
    // Non-detaching copy of this generation's handles into a product record
    void copyGpu(VoronoiProduct &product) const;

    uint32_t getCandidateNodeCount() const { return candidateNodeCount; }
    VkBuffer getSeedPositionBuffer() const { return seedPositionBuffer; }
    VkDeviceSize getSeedPositionBufferOffset() const { return seedPositionBufferOffset; }

    const std::vector<uint32_t>& getGlobalFragmentInstanceIds() const { return globalFragmentInstanceIds; }
    const std::vector<uint32_t>& getGlobalFragmentSeedIds() const { return globalFragmentSeedIds; }
    const std::vector<uint32_t>& getGlobalSeedToFragment() const { return globalSeedToFragment; }
    const std::vector<glm::dvec3>& getGlobalFragmentCentroids() const { return globalFragmentCentroids; }
    const std::vector<double>& getGlobalFragmentVolumes() const { return globalFragmentVolumes; }
    const std::vector<uint32_t>& getGlobalFaceFragmentA() const { return globalFaceFragmentA; }
    const std::vector<uint32_t>& getGlobalFaceFragmentB() const { return globalFaceFragmentB; }
    const std::vector<double>& getGlobalFaceAreasA() const { return globalFaceAreasA; }
    const std::vector<glm::dvec3>& getGlobalFaceAreaVectorsA() const { return globalFaceAreaVectorsA; }
    const std::vector<glm::dvec3>& getGlobalFaceMomentsA() const { return globalFaceMomentsA; }
    const std::vector<double>& getGlobalFaceAreasB() const { return globalFaceAreasB; }
    const std::vector<glm::dvec3>& getGlobalFaceAreaVectorsB() const { return globalFaceAreaVectorsB; }
    const std::vector<glm::dvec3>& getGlobalFaceMomentsB() const { return globalFaceMomentsB; }
    const std::vector<uint8_t>& getGlobalFaceTypes() const { return globalFaceTypes; }
    const std::vector<uint32_t>& getGlobalFaceDomainFaces() const { return globalFaceDomainFaces; }
    const std::vector<uint32_t>& getGlobalContactCouplingFaceA() const { return globalContactCouplingFaceA; }
    const std::vector<uint32_t>& getGlobalContactCouplingFaceB() const { return globalContactCouplingFaceB; }
    const std::vector<double>& getGlobalContactCouplingAreas() const { return globalContactCouplingAreas; }
    const std::vector<uint32_t>& getGlobalBoundaryFaceIndices() const { return globalBoundaryFaceIndices; }
    const std::vector<uint32_t>& getGlobalBoundaryFaceFragments() const { return globalBoundaryFaceFragments; }
    const std::vector<uint32_t>& getGlobalBoundaryFaceDomainFaces() const { return globalBoundaryFaceDomainFaces; }
    const std::vector<uint32_t>& getGlobalInstanceFragmentCounts() const { return globalInstanceFragmentCounts; }
    glm::vec3 getGlobalVolumeGridMin() const { return globalVolumeGridMin; }
    glm::ivec3 getGlobalVolumeGridDim() const { return globalVolumeGridDim; }
    float getGlobalVolumeGridCellSize() const { return globalVolumeGridCellSize; }
    float getGlobalNominalCellSize() const { return globalNominalCellSize; }
    const std::vector<glm::vec4>& getGlobalSeedPositions() const { return globalSeedPositions; }
    const std::array<glm::vec3, 8>& getGlobalDomainCorners() const { return globalDomainCorners; }

    const std::vector<uint32_t>& getGlobalDisplayRuntimeModelIds() const;
    const std::vector<uint32_t>& getGlobalDisplayFaceCounts() const;
    const std::vector<VkBuffer>& getGlobalDisplayVertexBuffers() const;
    const std::vector<VkDeviceSize>& getGlobalDisplayVertexBufferOffsets() const;
    const std::vector<VkBuffer>& getGlobalDisplayFaceIndexBuffers() const;
    const std::vector<VkDeviceSize>& getGlobalDisplayFaceIndexBufferOffsets() const;
    const std::vector<VkBuffer>& getGlobalDisplayCandidateBuffers() const;
    const std::vector<VkDeviceSize>& getGlobalDisplayCandidateBufferOffsets() const;

    void cleanup();
 
  private:
    bool uploadCandidateInputs(const std::vector<glm::vec4> &positions, const std::vector<uint32_t> &neighborIndices);
    void extractMeshTriangles(const std::vector<glm::vec3> &positions, const std::vector<uint32_t> &indices,
                              std::vector<std::array<glm::vec4, 3>> &outMeshTriangles) const;

    VulkanDevice &vulkanDevice;
    MemoryAllocator &memoryAllocator;
    CommandPool &commandPool;

    uint32_t candidateNodeCount = 0;
    VkBuffer candidateNeighborIndicesBuffer = VK_NULL_HANDLE;
    VkDeviceSize candidateNeighborIndicesBufferOffset = 0;
    VkBuffer seedPositionBuffer = VK_NULL_HANDLE;
    VkDeviceSize seedPositionBufferOffset = 0;
    VkBuffer candidatePointVertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize candidatePointVertexBufferOffset = 0;
    VkBuffer contactVertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize contactVertexBufferOffset = 0;
    uint32_t contactVertexCount = 0;

    std::vector<uint32_t> globalFragmentInstanceIds;
    std::vector<uint32_t> globalFragmentSeedIds;
    std::vector<uint32_t> globalSeedToFragment;
    std::vector<glm::dvec3> globalFragmentCentroids;
    std::vector<double> globalFragmentVolumes;
    float globalNominalCellSize = 0.0f;
    std::vector<uint32_t> globalFaceFragmentA;
    std::vector<uint32_t> globalFaceFragmentB;
    std::vector<double> globalFaceAreasA;
    std::vector<glm::dvec3> globalFaceAreaVectorsA;
    std::vector<glm::dvec3> globalFaceMomentsA;
    std::vector<double> globalFaceAreasB;
    std::vector<glm::dvec3> globalFaceAreaVectorsB;
    std::vector<glm::dvec3> globalFaceMomentsB;
    std::vector<uint8_t> globalFaceTypes;
    std::vector<uint32_t> globalFaceDomainFaces;
    std::vector<uint32_t> globalContactCouplingFaceA;
    std::vector<uint32_t> globalContactCouplingFaceB;
    std::vector<double> globalContactCouplingAreas;
    std::vector<uint32_t> globalBoundaryFaceIndices;
    std::vector<uint32_t> globalBoundaryFaceFragments;
    std::vector<uint32_t> globalBoundaryFaceDomainFaces;
    std::vector<uint32_t> globalInstanceFragmentCounts;
    glm::vec3 globalVolumeGridMin{};
    glm::ivec3 globalVolumeGridDim{};
    float globalVolumeGridCellSize = 0.0f;
    std::vector<glm::vec4> globalSeedPositions;
    std::array<glm::vec3, 8> globalDomainCorners{};

    VkBuffer globalSeedIdGridBuffer = VK_NULL_HANDLE;
    VkDeviceSize globalSeedIdGridBufferOffset = 0;
    uint32_t globalSeedIdGridCount = 0;
    VulkanExternalBuffer globalSeedIdGridExternal;
    CudaExternalBuffer globalSeedIdGridCuda;

    std::unique_ptr<voronoi::KNNNeighbors> knn;
    std::unique_ptr<voronoi::RVD> rvd;
    std::unique_ptr<voronoi::VoronoiCandidate> candidate;
};
