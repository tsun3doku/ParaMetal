#include "PointComputeRuntime.hpp"

#include "hash/HashProduct.hpp"
#include "renderers/PointRenderer.hpp"
#include "vulkan/MemoryAllocator.hpp"

PointComputeRuntime::PointComputeRuntime(MemoryAllocator& memoryAllocator)
    : memoryAllocator(memoryAllocator) {}

PointComputeRuntime::~PointComputeRuntime() {
    quiescedClear();
}

void PointComputeRuntime::releaseGeneration(PointBufferGeneration& generation) {
    if (generation.buffer != VK_NULL_HANDLE) {
        memoryAllocator.free(generation.buffer, generation.offset);
    }
    generation = {};
}

void PointComputeRuntime::releaseSlot(PointBufferSlot& slot) {
    releaseGeneration(slot.current);
    releaseGeneration(slot.retiring);
    releaseGeneration(slot.candidate);
}

void PointComputeRuntime::cancelCandidate(uint64_t socketKey) {
    const auto it = slots.find(socketKey);
    if (it == slots.end()) {
        return;
    }
    releaseGeneration(it->second.candidate);
}

bool PointComputeRuntime::prepareCandidate(const Config& config) {
    if (config.socketKey == 0 || config.positions == nullptr || config.positionCount == 0) {
        return false;
    }

    PointBufferSlot& slot = slots[config.socketKey];

    if (slot.candidate.buffer == VK_NULL_HANDLE &&
        slot.current.buffer != VK_NULL_HANDLE &&
        slot.current.hash == config.computeHash) {
        return true;
    }

    cancelCandidate(config.socketKey);

    const VkDeviceSize bufferSize =
        sizeof(PointRenderer::PointVertex) * config.positionCount;
    auto [buffer, offset] = memoryAllocator.allocate(
        bufferSize,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        alignof(PointRenderer::PointVertex));
    if (buffer == VK_NULL_HANDLE) {
        return false;
    }

    auto* mappedVertices = static_cast<PointRenderer::PointVertex*>(
        memoryAllocator.getMappedPointer(buffer, offset));
    if (!mappedVertices) {
        memoryAllocator.free(buffer, offset);
        return false;
    }

    for (std::size_t index = 0; index < config.positionCount; ++index) {
        mappedVertices[index] = {
            glm::vec3(config.positions[index]),
            glm::vec3(1.0f)
        };
    }

    slot.candidate.buffer = buffer;
    slot.candidate.offset = offset;
    slot.candidate.hash = config.computeHash;
    slot.candidate.pointCount = static_cast<uint32_t>(config.positionCount);
    slot.candidate.modelMatrix = config.modelMatrix;
    return true;
}

bool PointComputeRuntime::buildProduct(uint64_t socketKey, PointProduct& product) const {
    product = {};
    if (socketKey == 0) {
        return false;
    }

    const auto it = slots.find(socketKey);
    if (it == slots.end()) {
        return false;
    }
    const PointBufferSlot& slot = it->second;

    const PointBufferGeneration* source = nullptr;
    if (slot.candidate.buffer != VK_NULL_HANDLE) {
        source = &slot.candidate;
    } else if (slot.current.buffer != VK_NULL_HANDLE) {
        source = &slot.current;
    } else {
        return false;
    }

    if (!source || source->buffer == VK_NULL_HANDLE) {
        return false;
    }

    product.positionBuffer = source->buffer;
    product.positionBufferOffset = source->offset;
    product.pointCount = source->pointCount;
    product.modelMatrix = source->modelMatrix;
    HashProduct::seal(product);
    return product.isValid();
}

void PointComputeRuntime::commitPending(uint64_t socketKey) noexcept {
    const auto it = slots.find(socketKey);
    if (it == slots.end()) {
        return;
    }
    PointBufferSlot& slot = it->second;

    if (slot.candidate.buffer == VK_NULL_HANDLE) {
        return;
    }

    assert(slot.retiring.buffer == VK_NULL_HANDLE);
    if (slot.retiring.buffer != VK_NULL_HANDLE) {
        releaseGeneration(slot.candidate);
        return;
    }

    if (slot.current.buffer != VK_NULL_HANDLE) {
        slot.retiring = slot.current;
    }
    slot.current = slot.candidate;
    slot.candidate = {};
}

void PointComputeRuntime::clearRetiring(uint64_t socketKey) {
    const auto it = slots.find(socketKey);
    if (it == slots.end()) {
        return;
    }
    releaseGeneration(it->second.retiring);
}

void PointComputeRuntime::retireCurrent(uint64_t socketKey) {
    const auto it = slots.find(socketKey);
    if (it == slots.end()) {
        return;
    }
    PointBufferSlot& slot = it->second;
    assert(slot.retiring.buffer == VK_NULL_HANDLE);
    if (slot.retiring.buffer != VK_NULL_HANDLE) {
        return;
    }
    slot.retiring = slot.current;
    slot.current = {};
    cancelCandidate(socketKey);
}

bool PointComputeRuntime::currentHashMatches(uint64_t socketKey, uint64_t computeHash) const {
    const auto it = slots.find(socketKey);
    return it != slots.end() &&
           it->second.current.buffer != VK_NULL_HANDLE &&
           it->second.current.hash == computeHash;
}

void PointComputeRuntime::disable(uint64_t socketKey) {
    retireCurrent(socketKey);
}

bool PointComputeRuntime::hasAnyRetiring() const {
    for (const auto& [socketKey, slot] : slots) {
        (void)socketKey;
        if (slot.retiring.buffer != VK_NULL_HANDLE) return true;
    }
    return false;
}

void PointComputeRuntime::clearAllRetiring() {
    for (auto& [socketKey, slot] : slots) {
        (void)socketKey;
        if (slot.retiring.buffer != VK_NULL_HANDLE) {
            clearRetiring(socketKey);
        }
    }
}

void PointComputeRuntime::quiescedClear() noexcept {
    for (auto& [socketKey, slot] : slots) {
        (void)socketKey;
        releaseSlot(slot);
    }
    slots.clear();
}
