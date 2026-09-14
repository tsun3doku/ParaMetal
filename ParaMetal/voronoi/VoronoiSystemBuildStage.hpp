#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

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
    void releaseGpuResources(VoronoiProduct &product);

    uint32_t getCandidateNodeCount() const { return candidateNodeCount; }
    VkBuffer getCandidateNeighborIndicesBuffer() const { return candidateNeighborIndicesBuffer; }
    VkDeviceSize getCandidateNeighborIndicesBufferOffset() const { return candidateNeighborIndicesBufferOffset; }
    VkBuffer getSeedPositionBuffer() const { return seedPositionBuffer; }
    VkDeviceSize getSeedPositionBufferOffset() const { return seedPositionBufferOffset; }
    VkBuffer getCandidatePointVertexBuffer() const { return candidatePointVertexBuffer; }
    VkDeviceSize getCandidatePointVertexBufferOffset() const { return candidatePointVertexBufferOffset; }
    VkBuffer getContactVertexBuffer() const { return contactVertexBuffer; }
    VkDeviceSize getContactVertexBufferOffset() const { return contactVertexBufferOffset; }
    uint32_t getContactVertexCount() const { return contactVertexCount; }

    const std::vector<uint32_t>& getGlobalFragmentInstanceIds() const { return globalFragmentInstanceIds; }
    const std::vector<uint32_t>& getGlobalFragmentSeedIds() const { return globalFragmentSeedIds; }
    const std::vector<glm::vec3>& getGlobalFragmentCentroids() const { return globalFragmentCentroids; }
    const std::vector<float>& getGlobalFragmentVolumes() const { return globalFragmentVolumes; }
    const std::vector<uint32_t>& getGlobalFaceFragmentA() const { return globalFaceFragmentA; }
    const std::vector<uint32_t>& getGlobalFaceFragmentB() const { return globalFaceFragmentB; }
    const std::vector<float>& getGlobalFaceAreas() const { return globalFaceAreas; }
    const std::vector<float>& getGlobalFaceOrthogonalDistancesA() const { return globalFaceOrthogonalDistancesA; }
    const std::vector<float>& getGlobalFaceOrthogonalDistancesB() const { return globalFaceOrthogonalDistancesB; }
    const std::vector<glm::vec3>& getGlobalFaceNormals() const { return globalFaceNormals; }
    const std::vector<uint32_t>& getGlobalContactFaceFragmentA() const { return globalContactFaceFragmentA; }
    const std::vector<uint32_t>& getGlobalContactFaceFragmentB() const { return globalContactFaceFragmentB; }
    const std::vector<float>& getGlobalContactFaceAreas() const { return globalContactFaceAreas; }
    const std::vector<float>& getGlobalContactFaceOrthogonalDistancesA() const { return globalContactFaceOrthogonalDistancesA; }
    const std::vector<float>& getGlobalContactFaceOrthogonalDistancesB() const { return globalContactFaceOrthogonalDistancesB; }
    const std::vector<float>& getGlobalContactFaceGaps() const { return globalContactFaceGaps; }
    const std::vector<uint32_t>& getGlobalBoundaryFaceFragments() const { return globalBoundaryFaceFragments; }
    const std::vector<uint32_t>& getGlobalBoundaryFaceDomainFaces() const { return globalBoundaryFaceDomainFaces; }
    const std::vector<float>& getGlobalBoundaryFaceAreas() const { return globalBoundaryFaceAreas; }
    const std::vector<float>& getGlobalBoundaryFaceOrthogonalDistances() const { return globalBoundaryFaceOrthogonalDistances; }
    const std::vector<uint32_t>& getGlobalInstanceFragmentCounts() const { return globalInstanceFragmentCounts; }
    const std::vector<float>& getGlobalSdfValues() const { return globalSdfValues; }
    const std::vector<uint32_t>& getGlobalSdfRuntimeModelIds() const { return globalSdfRuntimeModelIds; }
    glm::vec3 getGlobalSdfGridMin() const { return globalSdfGridMin; }
    glm::ivec3 getGlobalSdfGridDim() const { return globalSdfGridDim; }
    float getGlobalSdfCellSize() const { return globalSdfCellSize; }
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

    VkBuffer getGlobalSeedIdGridBuffer() const { return globalSeedIdGridBuffer; }
    VkDeviceSize getGlobalSeedIdGridBufferOffset() const { return globalSeedIdGridBufferOffset; }
    uint32_t getGlobalSeedIdGridCount() const { return globalSeedIdGridCount; }

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
    std::vector<glm::vec3> globalFragmentCentroids;
    std::vector<float> globalFragmentVolumes;
    float globalNominalCellSize = 0.0f;
    std::vector<uint32_t> globalFaceFragmentA;
    std::vector<uint32_t> globalFaceFragmentB;
    std::vector<float> globalFaceAreas;
    std::vector<float> globalFaceOrthogonalDistancesA;
    std::vector<float> globalFaceOrthogonalDistancesB;
    std::vector<glm::vec3> globalFaceNormals;
    std::vector<uint32_t> globalContactFaceFragmentA;
    std::vector<uint32_t> globalContactFaceFragmentB;
    std::vector<float> globalContactFaceAreas;
    std::vector<float> globalContactFaceOrthogonalDistancesA;
    std::vector<float> globalContactFaceOrthogonalDistancesB;
    std::vector<float> globalContactFaceGaps;
    std::vector<uint32_t> globalBoundaryFaceFragments;
    std::vector<uint32_t> globalBoundaryFaceDomainFaces;
    std::vector<float> globalBoundaryFaceAreas;
    std::vector<float> globalBoundaryFaceOrthogonalDistances;
    std::vector<uint32_t> globalInstanceFragmentCounts;
    std::vector<float> globalSdfValues;
    std::vector<uint32_t> globalSdfRuntimeModelIds;
    glm::vec3 globalSdfGridMin{};
    glm::ivec3 globalSdfGridDim{};
    float globalSdfCellSize = 0.0f;
    std::vector<glm::vec4> globalSeedPositions;
    std::array<glm::vec3, 8> globalDomainCorners{};

    VkBuffer globalSeedIdGridBuffer = VK_NULL_HANDLE;
    VkDeviceSize globalSeedIdGridBufferOffset = 0;
    uint32_t globalSeedIdGridCount = 0;

    std::unique_ptr<voronoi::KNNNeighbors> knn;
    std::unique_ptr<voronoi::RVD> rvd;
    std::unique_ptr<voronoi::VoronoiCandidate> candidate;
};
