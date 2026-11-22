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
    uint32_t fluidNodeCount,
    uint32_t globalFaceCount,
    uint32_t globalBoundaryFaceCount) {
    cleanup();

    inFluidNodeCount = fluidNodeCount;
    inGlobalFaceCount = globalFaceCount;
    inGlobalBoundaryFaceCount = globalBoundaryFaceCount;

    constexpr VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    if (fluidNodeCount > 0) {
        const VkDeviceSize size = static_cast<VkDeviceSize>(fluidNodeCount) * sizeof(glm::vec4);
        if (!outVelocity.initialize(device, size, usage)) {
            cleanup();
            return false;
        }
        if (!outVelocityMapping.ensureMapped(outVelocity, cudaDevice)) {
            cleanup();
            return false;
        }

        const VkDeviceSize gradSize = static_cast<VkDeviceSize>(fluidNodeCount) * 3 * sizeof(glm::vec4);
        if (!outVelocityGradients.initialize(device, gradSize, usage)) {
            cleanup();
            return false;
        }
        if (!outVelocityGradientsMapping.ensureMapped(outVelocityGradients, cudaDevice)) {
            cleanup();
            return false;
        }
    }

    if (globalFaceCount > 0) {
        const VkDeviceSize size = static_cast<VkDeviceSize>(globalFaceCount) * sizeof(float);
        if (!outFaceFlux.initialize(device, size, usage)) {
            cleanup();
            return false;
        }
        if (!outFaceFluxMapping.ensureMapped(outFaceFlux, cudaDevice)) {
            cleanup();
            return false;
        }
    }

    if (globalBoundaryFaceCount > 0) {
        const VkDeviceSize size = static_cast<VkDeviceSize>(globalBoundaryFaceCount) * sizeof(float);
        if (!outBoundaryFlux.initialize(device, size, usage)) {
            cleanup();
            return false;
        }
        if (!outBoundaryFluxMapping.ensureMapped(outBoundaryFlux, cudaDevice)) {
            cleanup();
            return false;
        }
    }

    return true;
}

void CFDRuntime::cleanup() {
    clearTopology();
    outVelocityMapping.cleanup();
    outVelocityGradientsMapping.cleanup();
    outFaceFluxMapping.cleanup();
    outBoundaryFluxMapping.cleanup();

    outVelocity.cleanup();
    outVelocityGradients.cleanup();
    outFaceFlux.cleanup();
    outBoundaryFlux.cleanup();

    inFluidNodeCount = 0;
    inGlobalFaceCount = 0;
    inGlobalBoundaryFaceCount = 0;
}

} // namespace cfd
