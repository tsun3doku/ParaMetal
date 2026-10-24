#pragma once

#include "vulkan/VulkanExternalBuffer.hpp"
#include "cuda/CudaExternalBuffer.hpp"
#include <glm/glm.hpp>
#include <cstdint>
#include <vulkan/vulkan.h>

class VulkanDevice;

namespace cfd {

class CFDRuntime {
public:
    CFDRuntime();
    ~CFDRuntime();

    bool initialize(
        VulkanDevice& device,
        int cudaDevice,
        uint32_t fluidNodeCount,
        uint32_t globalFaceCount,
        uint32_t globalBoundaryFaceCount);

    void cleanup();

    glm::vec4* velocityCuda() {
        return reinterpret_cast<glm::vec4*>(velocityMapping.getPointer());
    }

    float* faceFluxCuda() {
        return faceFluxMapping.getPointer();
    }

    float* boundaryFluxCuda() {
        return boundaryFluxMapping.getPointer();
    }

    VkBuffer velocityBuffer() const { return velocity.getBuffer(); }
    VkBuffer faceFluxBuffer() const { return faceFlux.getBuffer(); }
    VkBuffer boundaryFluxBuffer() const { return boundaryFlux.getBuffer(); }

    const VulkanExternalBuffer& getVelocityExternalBuffer() const { return velocity; }
    const VulkanExternalBuffer& getFaceFluxExternalBuffer() const { return faceFlux; }
    const VulkanExternalBuffer& getBoundaryFluxExternalBuffer() const { return boundaryFlux; }

    uint32_t getFluidNodeCount() const { return fluidNodeCount; }
    uint32_t getGlobalFaceCount() const { return globalFaceCount; }
    uint32_t getGlobalBoundaryFaceCount() const { return globalBoundaryFaceCount; }

private:
    VulkanExternalBuffer velocity;
    CudaExternalBuffer velocityMapping;

    VulkanExternalBuffer faceFlux;
    CudaExternalBuffer faceFluxMapping;

    VulkanExternalBuffer boundaryFlux;
    CudaExternalBuffer boundaryFluxMapping;

    uint32_t fluidNodeCount = 0;
    uint32_t globalFaceCount = 0;
    uint32_t globalBoundaryFaceCount = 0;
};

} // namespace cfd
