#include "HeatFluidRuntime.hpp"

#include "vulkan/VulkanDevice.hpp"
#include <iostream>

HeatFluidRuntime::HeatFluidRuntime() = default;

HeatFluidRuntime::~HeatFluidRuntime() {
    cleanup();
}

void HeatFluidRuntime::setMaterialProperties(float density, float specificHeat, float conductivity, float initialTemperatureC) {
    this->density = density;
    this->specificHeat = specificHeat;
    this->conductivity = conductivity;
    this->initialTemperatureC = initialTemperatureC;
}

bool HeatFluidRuntime::ensureSimulationBuffers(uint32_t count, VulkanDevice& device) {
    if (this->nodeCount == count && tempBufferA.isValid() && tempBufferB.isValid()) {
        return true;
    }

    cleanupSimulationBuffers();
    this->nodeCount = count;

    if (count == 0) {
        return true;
    }

    const VkDeviceSize byteSize = static_cast<VkDeviceSize>(count) * sizeof(float);
    const VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    if (!tempBufferA.initialize(device, byteSize, usage)) {
        std::cerr << "[HeatFluidRuntime-Diag] ensureSimulationBuffers FAILED: tempBufferA.initialize"
                  << " nodeCount=" << count << " byteSize=" << byteSize << std::endl;
        cleanupSimulationBuffers();
        this->nodeCount = 0;
        return false;
    }
    if (!tempBufferB.initialize(device, byteSize, usage)) {
        std::cerr << "[HeatFluidRuntime-Diag] ensureSimulationBuffers FAILED: tempBufferB.initialize"
                  << " nodeCount=" << count << " byteSize=" << byteSize << std::endl;
        cleanupSimulationBuffers();
        this->nodeCount = 0;
        return false;
    }

    return true;
}

void HeatFluidRuntime::initializePlayback(VulkanDevice& device, MemoryAllocator& allocator, uint32_t frameCapacity) {
    if (nodeCount == 0 || frameCapacity == 0) return;
    ensureSimulationBuffers(nodeCount, device);
    if (!playback) {
        playback = std::make_unique<HeatSystemPlayback>();
    }
    if (!playback->isValid() || playback->getFrameCapacity() != frameCapacity || playback->getNodeCount() != nodeCount) {
        playback->initialize(device, allocator, nodeCount, frameCapacity);
    }
}

void HeatFluidRuntime::cleanupSimulationBuffers() {
    cudaTempBufferA.cleanup();
    cudaTempBufferB.cleanup();
    tempBufferA.cleanup();
    tempBufferB.cleanup();
}

void HeatFluidRuntime::cleanup() {
    if (playback) {
        playback->cleanup();
        playback.reset();
    }
    cleanupSimulationBuffers();
    nodeCount = 0;
}
