#define NOMINMAX
#include <Windows.h>

#include "CudaExternalBuffer.hpp"
#include "../vulkan/VulkanExternalBuffer.hpp"
#include <vulkan/vulkan_win32.h>

#include <cuda_runtime.h>
#include <iostream>

class CudaExternalBuffer::Implementation {
public:
    bool ensureMapped(
        VkDevice device,
        VkDeviceMemory mem,
        VkDeviceSize allocationSize,
        VkDeviceSize bufferSize,
        int requestedDevice) {
        if (pointer && sourceMemory == mem) return true;
        cleanup();

        if (device == VK_NULL_HANDLE || mem == VK_NULL_HANDLE || allocationSize == 0 || bufferSize == 0) {
            return false;
        }

        cudaDevice = requestedDevice;
        if (cudaSetDevice(cudaDevice) != cudaSuccess) return false;

        const auto getMemoryHandle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
            vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
        if (!getMemoryHandle) return false;

        VkMemoryGetWin32HandleInfoKHR handleInfo{};
        handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        handleInfo.memory = mem;
        handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        HANDLE handle = nullptr;
        if (getMemoryHandle(device, &handleInfo, &handle) != VK_SUCCESS || !handle) {
            return false;
        }

        cudaExternalMemoryHandleDesc memoryDescription{};
        memoryDescription.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        memoryDescription.handle.win32.handle = handle;
        memoryDescription.size = allocationSize;
        const cudaError_t importResult = cudaImportExternalMemory(&memory, &memoryDescription);
        DWORD handleFlags = 0;
        if (GetHandleInformation(handle, &handleFlags)) CloseHandle(handle);
        if (importResult != cudaSuccess) {
            std::cerr << "[CudaExternalBuffer] import failed: " << cudaGetErrorString(importResult) << std::endl;
            return false;
        }

        cudaExternalMemoryBufferDesc bufferDescription{};
        bufferDescription.size = bufferSize;
        const cudaError_t mapResult = cudaExternalMemoryGetMappedBuffer(
            reinterpret_cast<void**>(&pointer), memory, &bufferDescription);
        if (mapResult != cudaSuccess) {
            std::cerr << "[CudaExternalBuffer] map failed: " << cudaGetErrorString(mapResult) << std::endl;
            cleanup();
            return false;
        }
        sourceMemory = mem;
        return true;
    }

    bool ensureMapped(const VulkanExternalBuffer& buffer, int requestedDevice) {
        if (pointer && sourceBuffer == buffer.getBuffer()) return true;
        if (ensureMapped(
                buffer.getDevice(),
                buffer.getMemory(),
                buffer.getAllocationSize(),
                buffer.getBufferSize(),
                requestedDevice)) {
            sourceBuffer = buffer.getBuffer();
            return true;
        }
        return false;
    }

    void cleanup() {
        if (cudaDevice >= 0) cudaSetDevice(cudaDevice);
        if (pointer) cudaFree(pointer);
        if (memory) cudaDestroyExternalMemory(memory);
        pointer = nullptr;
        memory = nullptr;
        sourceBuffer = VK_NULL_HANDLE;
        sourceMemory = VK_NULL_HANDLE;
        cudaDevice = -1;
    }

    float* pointer = nullptr;
    cudaExternalMemory_t memory = nullptr;
    VkBuffer sourceBuffer = VK_NULL_HANDLE;
    VkDeviceMemory sourceMemory = VK_NULL_HANDLE;
    int cudaDevice = -1;
};

CudaExternalBuffer::CudaExternalBuffer() : implementation(std::make_unique<Implementation>()) {}
CudaExternalBuffer::~CudaExternalBuffer() = default;
bool CudaExternalBuffer::ensureMapped(const VulkanExternalBuffer& buffer, int cudaDevice) { return implementation->ensureMapped(buffer, cudaDevice); }
bool CudaExternalBuffer::ensureMapped(
    VkDevice device,
    VkDeviceMemory memory,
    VkDeviceSize allocationSize,
    VkDeviceSize bufferSize,
    int cudaDevice) {
    return implementation->ensureMapped(device, memory, allocationSize, bufferSize, cudaDevice);
}
void CudaExternalBuffer::cleanup() { implementation->cleanup(); }
float* CudaExternalBuffer::getPointer() const { return implementation->pointer; }
