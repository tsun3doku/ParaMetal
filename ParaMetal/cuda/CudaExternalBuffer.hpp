#pragma once

#include <memory>
#include <vulkan/vulkan.h>

class VulkanExternalBuffer;

class CudaExternalBuffer {
public:
    CudaExternalBuffer();
    ~CudaExternalBuffer();

    CudaExternalBuffer(const CudaExternalBuffer&) = delete;
    CudaExternalBuffer& operator=(const CudaExternalBuffer&) = delete;

    bool ensureMapped(const VulkanExternalBuffer& buffer, int cudaDevice);
    bool ensureMapped(
        VkDevice device,
        VkDeviceMemory memory,
        VkDeviceSize allocationSize,
        VkDeviceSize bufferSize,
        int cudaDevice);
    void cleanup();
    float* getPointer() const;

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation;
};
