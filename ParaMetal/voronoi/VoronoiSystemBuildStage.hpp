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
    void copyGpu(VoronoiProduct &product) const;

    uint32_t getCandidateNodeCount() const { return outCandidateNodeCount; }
    VkBuffer getSeedPositionBuffer() const { return outSeedPositionBuffer; }
    VkDeviceSize getSeedPositionBufferOffset() const { return outSeedPositionBufferOffset; }


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

    VulkanDevice &vulkanDevice;
    MemoryAllocator &memoryAllocator;
    CommandPool &commandPool;

    uint32_t outCandidateNodeCount = 0;
    VkBuffer outCandidateNeighborIndicesBuffer = VK_NULL_HANDLE;
    VkDeviceSize outCandidateNeighborIndicesBufferOffset = 0;
    VkBuffer outSeedPositionBuffer = VK_NULL_HANDLE;
    VkDeviceSize outSeedPositionBufferOffset = 0;
    VkBuffer outCandidatePointVertexBuffer = VK_NULL_HANDLE;
    VkDeviceSize outCandidatePointVertexBufferOffset = 0;



    VkBuffer outGlobalSeedIdGridBuffer = VK_NULL_HANDLE;
    VkDeviceSize outGlobalSeedIdGridBufferOffset = 0;
    VulkanExternalBuffer outGlobalSeedIdGridExternal;
    CudaExternalBuffer outGlobalSeedIdGridCuda;

    std::unique_ptr<voronoi::KNNNeighbors> knn;
    std::unique_ptr<voronoi::RVD> rvd;
    std::unique_ptr<voronoi::VoronoiCandidate> candidate;
};

