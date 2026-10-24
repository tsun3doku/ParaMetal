#include "CFDRuntime.hpp"
#include "vulkan/VulkanDevice.hpp"

namespace cfd {

CFDRuntime::CFDRuntime() = default;

CFDRuntime::~CFDRuntime() {
    cleanup();
}

bool CFDRuntime::initialize(
    VulkanDevice& device,
    int cudaDevice,
    uint32_t inFluidNodeCount,
    uint32_t inGlobalFaceCount,
    uint32_t inGlobalBoundaryFaceCount) {
    cleanup();

    fluidNodeCount = inFluidNodeCount;
    globalFaceCount = inGlobalFaceCount;
    globalBoundaryFaceCount = inGlobalBoundaryFaceCount;

    constexpr VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    if (fluidNodeCount > 0) {
        const VkDeviceSize size = static_cast<VkDeviceSize>(fluidNodeCount) * sizeof(glm::vec4);
        if (!velocity.initialize(device, size, usage)) {
            cleanup();
            return false;
        }
        if (!velocityMapping.ensureMapped(velocity, cudaDevice)) {
            cleanup();
            return false;
        }
    }

    if (globalFaceCount > 0) {
        const VkDeviceSize size = static_cast<VkDeviceSize>(globalFaceCount) * sizeof(float);
        if (!faceFlux.initialize(device, size, usage)) {
            cleanup();
            return false;
        }
        if (!faceFluxMapping.ensureMapped(faceFlux, cudaDevice)) {
            cleanup();
            return false;
        }
    }

    if (globalBoundaryFaceCount > 0) {
        const VkDeviceSize size = static_cast<VkDeviceSize>(globalBoundaryFaceCount) * sizeof(float);
        if (!boundaryFlux.initialize(device, size, usage)) {
            cleanup();
            return false;
        }
        if (!boundaryFluxMapping.ensureMapped(boundaryFlux, cudaDevice)) {
            cleanup();
            return false;
        }
    }

    return true;
}

void CFDRuntime::cleanup() {
    velocityMapping.cleanup();
    faceFluxMapping.cleanup();
    boundaryFluxMapping.cleanup();

    velocity.cleanup();
    faceFlux.cleanup();
    boundaryFlux.cleanup();

    fluidNodeCount = 0;
    globalFaceCount = 0;
    globalBoundaryFaceCount = 0;
}

} // namespace cfd
