#include "PointComputeRuntime.hpp"

#include "renderers/PointRenderer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/MemoryAllocator.hpp"

#include <cstring>

PointComputeRuntime::PointComputeRuntime(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& commandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      commandPool(commandPool) {}

PointComputeRuntime::~PointComputeRuntime() {
    disableAll();
}

void PointComputeRuntime::configure(const Config& config) {
    if (config.socketKey == 0) {
        return;
    }

    auto it = activeSystems.find(config.socketKey);
    if (it != activeSystems.end() && it->second.computeHash == config.computeHash) {
        return;
    }

    disable(config.socketKey);

    if (config.positions.empty()) {
        return;
    }

    std::vector<PointRenderer::PointVertex> vertices;
    vertices.reserve(config.positions.size());
    for (const glm::vec4& pos : config.positions) {
        vertices.push_back({ glm::vec3(pos), glm::vec3(1.0f) });
    }

    const VkDeviceSize bufferSize = sizeof(PointRenderer::PointVertex) * vertices.size();
    auto [buffer, offset] = memoryAllocator.allocate(
        bufferSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        alignof(PointRenderer::PointVertex));
    if (buffer == VK_NULL_HANDLE) {
        return;
    }

    void* mappedPtr = memoryAllocator.getMappedPointer(buffer, offset);
    if (!mappedPtr) {
        memoryAllocator.free(buffer, offset);
        return;
    }

    std::memcpy(mappedPtr, vertices.data(), static_cast<size_t>(bufferSize));

    SystemInstance instance{};
    instance.buffer = buffer;
    instance.offset = offset;
    instance.pointCount = static_cast<uint32_t>(vertices.size());
    instance.modelMatrix = config.modelMatrix;
    instance.computeHash = config.computeHash;
    activeSystems[config.socketKey] = instance;
}

void PointComputeRuntime::disable(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }
    auto it = activeSystems.find(socketKey);
    if (it == activeSystems.end()) {
        return;
    }
    if (it->second.buffer != VK_NULL_HANDLE) {
        memoryAllocator.free(it->second.buffer, it->second.offset);
    }
    activeSystems.erase(it);
}

void PointComputeRuntime::disableAll() {
    for (auto& [key, inst] : activeSystems) {
        (void)key;
        if (inst.buffer != VK_NULL_HANDLE) {
            memoryAllocator.free(inst.buffer, inst.offset);
        }
    }
    activeSystems.clear();
}

bool PointComputeRuntime::exportProduct(uint64_t socketKey, PointProduct& outProduct) const {
    outProduct = {};
    auto it = activeSystems.find(socketKey);
    if (it == activeSystems.end()) {
        return false;
    }
    const SystemInstance& inst = it->second;
    outProduct.positionBuffer = inst.buffer;
    outProduct.positionBufferOffset = inst.offset;
    outProduct.pointCount = inst.pointCount;
    outProduct.modelMatrix = inst.modelMatrix;
    outProduct.productHash = buildProductHash(outProduct);
    return outProduct.isValid();
}
