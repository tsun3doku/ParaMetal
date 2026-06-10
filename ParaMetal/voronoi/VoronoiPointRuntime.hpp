#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "voronoi/VoronoiDomainRuntime.hpp"

class CommandPool;
class MemoryAllocator;
class VulkanDevice;

class VoronoiPointRuntime : public VoronoiDomainRuntime {
public:
    VoronoiPointRuntime(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        uint64_t domainKey,
        std::vector<glm::vec4> positions,
        std::vector<uint32_t> boundaryConditions,
        std::vector<float> fixedTemperatures,
        CommandPool& renderCommandPool);
    ~VoronoiPointRuntime();

    bool isPointDomain() const override { return true; }
    bool createVoronoiBuffers() override;
    void cleanup() override;

    uint64_t getDomainKey() const { return domainKey; }
    const std::vector<glm::vec4>& getPositions() const { return positions; }
    const std::vector<uint32_t>& getBoundaryConditions() const { return boundaryConditions; }
    const std::vector<float>& getFixedTemperatures() const { return fixedTemperatures; }
    glm::vec3 getAabbMin() const { return aabbMin; }
    glm::vec3 getAabbMax() const { return aabbMax; }
    VkBuffer getCandidateBuffer() const override { return candidateBuffer; }
    VkDeviceSize getCandidateBufferOffset() const override { return candidateBufferOffset; }

private:
    void computeAabb();

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    uint64_t domainKey = 0;
    std::vector<glm::vec4> positions;
    std::vector<uint32_t> boundaryConditions;
    std::vector<float> fixedTemperatures;
    CommandPool& renderCommandPool;
    glm::vec3 aabbMin{0.0f};
    glm::vec3 aabbMax{1.0f};
    VkBuffer candidateBuffer = VK_NULL_HANDLE;
    VkDeviceSize candidateBufferOffset = 0;
};
