#include "RemeshController.hpp"

#include "hash/HashProduct.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "util/GeometryUtils.hpp"
#include "vulkan/MemoryAllocator.hpp"

#include <cassert>

RemeshController::OperatingScope::OperatingScope(std::atomic<bool>& isOperating)
    : isOperating(isOperating) {
    previousState = isOperating.exchange(true, std::memory_order_acq_rel);
}

RemeshController::OperatingScope::~OperatingScope() {
    isOperating.store(previousState, std::memory_order_release);
}

RemeshController::RemeshController(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    std::atomic<bool>& isOperating)
    : isOperating(isOperating),
      remesher(vulkanDevice, memoryAllocator) {
}

bool RemeshController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return false;
    }

    const auto configIt = configuredHashes.find(socketKey);
    if (configIt != configuredHashes.end() && configIt->second == config.computeHash) {
        cancelCandidate(socketKey);
        return true;
    }

    const auto failedIt = failedComputeHashes.find(socketKey);
    if (failedIt != failedComputeHashes.end() && failedIt->second == config.computeHash) {
        return false;
    }
    failedComputeHashes.erase(socketKey);

    // A generation cannot be admitted while the socket still owns a pending or retiring one.
    const auto slotIt = generationSlots.find(socketKey);
    if (slotIt != generationSlots.end() &&
        (slotIt->second.retiring || slotIt->second.candidate)) {
        return false;
    }

    // Build replacement generation off to the side while active generation remains live
    auto candidate = std::make_unique<RemeshSystem>(remesher);
    candidate->setSourceGeometry(config.pointPositions, config.triangleIndices);
    candidate->setParams(config.iterations, config.minAngleDegrees, config.maxEdgeLength, config.stepSize);
    candidate->setRuntimeModelId(config.runtimeModelId);

    OperatingScope operatingScope(isOperating);
    if (!candidate->remesh()) {
        // The failed candidate holds GPU allocations; the boundary releases it after quiescence.
        generationSlots[socketKey].retiring = std::move(candidate);
        failedComputeHashes[socketKey] = config.computeHash;
        return false;
    }

    GenerationSlot& slot = generationSlots[socketKey];
    // Reserve the committed-hash entry so commit performs no allocation.
    configuredHashes.try_emplace(socketKey, 0);
    pendingHashes[socketKey] = config.computeHash;
    slot.candidate = std::move(candidate);
    return true;
}

bool RemeshController::buildProduct(uint64_t socketKey, RemeshProduct& product) {
    if (socketKey == 0) {
        return false;
    }

    const auto it = generationSlots.find(socketKey);
    if (it == generationSlots.end()) {
        return false;
    }
    RemeshSystem* systemPtr = it->second.candidate
        ? it->second.candidate.get()
        : it->second.current.get();
    if (!systemPtr) {
        return false;
    }
    RemeshSystem& system = *systemPtr;

    product = {};

    // CPU data
    product.runtimeModelId = system.runtimeModelId();
    product.geometryPositions = toVec3Array(system.sourcePositions());
    product.geometryTriangleIndices = system.sourceTriangles();
    const SupportingHalfedge::IntrinsicMesh& mesh = system.intrinsicMesh();
    product.surfacePositions.reserve(mesh.vertices.size());
    product.surfaceNormals.reserve(mesh.vertices.size());
    for (const SupportingHalfedge::IntrinsicVertex& vertex : mesh.vertices) {
        product.surfacePositions.push_back(vertex.position);
        product.surfaceNormals.push_back(vertex.normal);
    }
    product.surfaceTriangleIndices = mesh.indices;

    // GPU resources (non-detaching copy, ownership transfers on commit)
    system.copyGpu(product);

    HashProduct::seal(product);
    return product.isValid();
}



void RemeshController::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    failedComputeHashes.erase(socketKey);
    retireCurrent(socketKey);
}

bool RemeshController::hasAnyRetiring() const {
    for (const auto& [socketKey, slot] : generationSlots) {
        (void)socketKey;
        if (slot.retiring) return true;
    }
    return false;
}

void RemeshController::clearAllRetiring() {
    for (auto it = generationSlots.begin(); it != generationSlots.end();) {
        GenerationSlot& slot = it->second;
        slot.retiring.reset();
        // A socket with no committed configuration owns no generation; release the rest.
        if (configuredHashes.find(it->first) == configuredHashes.end()) {
            slot.current.reset();
            slot.candidate.reset();
        }
        if (!slot.current && !slot.candidate && !slot.retiring) {
            it = generationSlots.erase(it);
        } else {
            ++it;
        }
    }
}

void RemeshController::commitPending(uint64_t socketKey) noexcept {
    const auto it = generationSlots.find(socketKey);
    if (it == generationSlots.end()) {
        return;
    }
    GenerationSlot& slot = it->second;
    assert(slot.retiring == nullptr);
    if (slot.retiring != nullptr) {
        return;
    }
    if (slot.candidate) {
        slot.retiring = std::move(slot.current);
        slot.current = std::move(slot.candidate);
    }

    const auto pending = pendingHashes.find(socketKey);
    if (pending != pendingHashes.end()) {
        const auto committed = configuredHashes.find(socketKey);
        if (committed != configuredHashes.end()) {
            committed->second = pending->second;
        }
        pendingHashes.erase(pending);
    }
    failedComputeHashes.erase(socketKey);
}

void RemeshController::cancelCandidate(uint64_t socketKey) {
    const auto it = generationSlots.find(socketKey);
    if (it != generationSlots.end() && it->second.candidate) {
        // A candidate only exists after admission, which requires an empty retiring slot.
        assert(it->second.retiring == nullptr);
        if (!it->second.retiring) {
            it->second.retiring = std::move(it->second.candidate);
        }
        // A socket that never committed a generation must not keep its reserved entry.
        if (!it->second.current) {
            const auto committed = configuredHashes.find(socketKey);
            if (committed != configuredHashes.end() && committed->second == 0) {
                configuredHashes.erase(committed);
            }
        }
    }
    pendingHashes.erase(socketKey);
}

void RemeshController::retireCurrent(uint64_t socketKey) {
    configuredHashes.erase(socketKey);
    pendingHashes.erase(socketKey);

    const auto it = generationSlots.find(socketKey);
    if (it == generationSlots.end()) {
        return;
    }
    GenerationSlot& slot = it->second;
    if (slot.candidate && !slot.retiring) {
        slot.retiring = std::move(slot.candidate);
    }
    if (slot.current && !slot.retiring) {
        slot.retiring = std::move(slot.current);
    }
    // A system that cannot enter retiring stays as the ownerless current and is
    // released by clearAllRetiring after the next successful quiesce.
}

void RemeshController::quiescedClear() noexcept {
    generationSlots.clear();
    configuredHashes.clear();
    pendingHashes.clear();
    failedComputeHashes.clear();
}
