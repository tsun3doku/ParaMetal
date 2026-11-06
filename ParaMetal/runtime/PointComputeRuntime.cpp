#include "PointComputeRuntime.hpp"

#include "hash/HashProduct.hpp"
#include "renderers/PointRenderer.hpp"
#include "vulkan/MemoryAllocator.hpp"

#include <cstring>

PointComputeRuntime::PointComputeRuntime(MemoryAllocator& memoryAllocator)
    : memoryAllocator(memoryAllocator) {}

PointComputeRuntime::~PointComputeRuntime() {
    quiescedClear();
}

void PointComputeRuntime::releaseBuffer(std::pair<VkBuffer, VkDeviceSize>& buffer) {
    if (buffer.first != VK_NULL_HANDLE) {
        memoryAllocator.free(buffer.first, buffer.second);
    }
    buffer = {VK_NULL_HANDLE, 0};
}

void PointComputeRuntime::releaseSlot(PointBufferSlot& slot) {
    releaseBuffer(slot.current);
    releaseBuffer(slot.retiring);
    releaseBuffer(slot.candidate);
    slot.currentHash = 0;
    slot.candidateHash = 0;
    slot.currentPointCount = 0;
    slot.candidatePointCount = 0;
    slot.currentModelMatrix = glm::mat4(1.0f);
    slot.candidateModelMatrix = glm::mat4(1.0f);
    slot.candidateValid = false;
}

void PointComputeRuntime::cancelCandidate(uint64_t socketKey) {
    const auto it = slots.find(socketKey);
    if (it == slots.end()) {
        return;
    }
    releaseBuffer(it->second.candidate);
    it->second.candidateValid = false;
    it->second.candidateHash = 0;
    it->second.candidatePointCount = 0;
}

bool PointComputeRuntime::prepareCandidate(const Config& config) {
    if (config.socketKey == 0 || config.positions.empty()) {
        return false;
    }

    PointBufferSlot& slot = slots[config.socketKey];

    if (!slot.candidateValid &&
        slot.current.first != VK_NULL_HANDLE &&
        slot.currentHash == config.computeHash) {
        return true;
    }

    cancelCandidate(config.socketKey);

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

    slot.candidate = {buffer, offset};
    slot.candidateHash = config.computeHash;
    slot.candidatePointCount = static_cast<uint32_t>(vertices.size());
    slot.candidateModelMatrix = config.modelMatrix;
    slot.candidateValid = true;
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

    const std::pair<VkBuffer, VkDeviceSize>* source = nullptr;
    uint32_t pointCount = 0;
    glm::mat4 modelMatrix = glm::mat4(1.0f);
    if (slot.candidateValid) {
        source = &slot.candidate;
        pointCount = slot.candidatePointCount;
        modelMatrix = slot.candidateModelMatrix;
    } else if (slot.current.first != VK_NULL_HANDLE) {
        source = &slot.current;
        pointCount = slot.currentPointCount;
        modelMatrix = slot.currentModelMatrix;
    } else {
        return false;
    }

    if (source == nullptr || source->first == VK_NULL_HANDLE) {
        return false;
    }

    product.positionBuffer = source->first;
    product.positionBufferOffset = source->second;
    product.pointCount = pointCount;
    product.modelMatrix = modelMatrix;
    HashProduct::seal(product);
    return product.isValid();
}

void PointComputeRuntime::commitPending(uint64_t socketKey) noexcept {
    const auto it = slots.find(socketKey);
    if (it == slots.end()) {
        return;
    }
    PointBufferSlot& slot = it->second;

    if (!slot.candidateValid) {
        return;
    }

    assert(slot.retiring.first == VK_NULL_HANDLE);
    if (slot.retiring.first != VK_NULL_HANDLE) {
        releaseBuffer(slot.candidate);
        slot.candidateValid = false;
        return;
    }

    if (slot.current.first != VK_NULL_HANDLE) {
        slot.retiring = slot.current;
    }
    slot.current = slot.candidate;
    slot.currentHash = slot.candidateHash;
    slot.currentPointCount = slot.candidatePointCount;
    slot.currentModelMatrix = slot.candidateModelMatrix;
    slot.candidate = {VK_NULL_HANDLE, 0};
    slot.candidateValid = false;
    slot.candidateHash = 0;
}

void PointComputeRuntime::clearRetiring(uint64_t socketKey) {
    const auto it = slots.find(socketKey);
    if (it == slots.end()) {
        return;
    }
    releaseBuffer(it->second.retiring);
}

void PointComputeRuntime::retireCurrent(uint64_t socketKey) {
    const auto it = slots.find(socketKey);
    if (it == slots.end()) {
        return;
    }
    PointBufferSlot& slot = it->second;
    assert(slot.retiring.first == VK_NULL_HANDLE);
    if (slot.retiring.first != VK_NULL_HANDLE) {
        return;
    }
    slot.retiring = slot.current;
    slot.current = {VK_NULL_HANDLE, 0};
    slot.currentHash = 0;
    slot.currentPointCount = 0;
    slot.currentModelMatrix = glm::mat4(1.0f);
    cancelCandidate(socketKey);
}

bool PointComputeRuntime::currentHashMatches(uint64_t socketKey, uint64_t computeHash) const {
    const auto it = slots.find(socketKey);
    return it != slots.end() &&
           it->second.current.first != VK_NULL_HANDLE &&
           it->second.currentHash == computeHash;
}

void PointComputeRuntime::disable(uint64_t socketKey) {
    retireCurrent(socketKey);
}

bool PointComputeRuntime::hasAnyRetiring() const {
    for (const auto& [socketKey, slot] : slots) {
        (void)socketKey;
        if (slot.retiring.first != VK_NULL_HANDLE) return true;
    }
    return false;
}

void PointComputeRuntime::clearAllRetiring() {
    for (auto& [socketKey, slot] : slots) {
        (void)socketKey;
        if (slot.retiring.first != VK_NULL_HANDLE) {
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
