#include "PointComputeRuntime.hpp"

#include "hash/HashProduct.hpp"
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

bool PointComputeRuntime::buildProduct(const Config& config, PointProduct& product) {
    product = {};

    if (config.socketKey == 0 || config.positions.empty()) {
        return false;
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
        return false;
    }

    void* mappedPtr = memoryAllocator.getMappedPointer(buffer, offset);
    if (!mappedPtr) {
        memoryAllocator.free(buffer, offset);
        return false;
    }

    std::memcpy(mappedPtr, vertices.data(), static_cast<size_t>(bufferSize));

    // Ownership transfers to product
    product.positionBuffer = buffer;
    product.positionBufferOffset = offset;
    product.pointCount = static_cast<uint32_t>(vertices.size());
    product.modelMatrix = config.modelMatrix;
    HashProduct::seal(product);

    activeHashes[config.socketKey] = config.computeHash;
    return product.isValid();
}

void PointComputeRuntime::disable(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }
    activeHashes.erase(socketKey);
}

void PointComputeRuntime::disableAll() {
    activeHashes.clear();
}
