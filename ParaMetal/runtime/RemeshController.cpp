#include "RemeshController.hpp"

#include "hash/HashProduct.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "util/GeometryUtils.hpp"
#include "vulkan/MemoryAllocator.hpp"

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

    const auto failedIt = failedComputeHashes.find(socketKey);
    if (failedIt != failedComputeHashes.end() && failedIt->second == config.computeHash) {
        return false;
    }
    failedComputeHashes.erase(socketKey);

    // Build replacement generation off to the side while active generation remains live
    auto candidate = std::make_unique<RemeshSystem>(remesher);
    candidate->setSourceGeometry(config.pointPositions, config.triangleIndices);
    candidate->setParams(config.iterations, config.minAngleDegrees, config.maxEdgeLength, config.stepSize);
    candidate->setRuntimeModelId(config.runtimeModelId);

    OperatingScope operatingScope(isOperating);
    if (!candidate->remesh()) {
        // Retain failed candidate to avoid destroying resources with in-flight GPU work
        retainedSystems.push_back(std::move(candidate));
        failedComputeHashes[socketKey] = config.computeHash;
        return false;
    }

    auto it = activeSystems.find(socketKey);
    if (it != activeSystems.end()) {
        if (it->second) {
            retainedSystems.push_back(std::move(it->second));
        }
        it->second = std::move(candidate);
    } else {
        activeSystems.emplace(socketKey, std::move(candidate));
    }
    return true;
}

bool RemeshController::buildProduct(uint64_t socketKey, RemeshProduct& product) {
    if (socketKey == 0) {
        return false;
    }

    const auto it = activeSystems.find(socketKey);
    if (it == activeSystems.end() || !it->second) {
        return false;
    }
    RemeshSystem& system = *it->second;

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
    auto it = activeSystems.find(socketKey);
    if (it != activeSystems.end()) {
        if (it->second) {
            retainedSystems.push_back(std::move(it->second));
        }
        activeSystems.erase(it);
    }
}

void RemeshController::disableAll() {
    failedComputeHashes.clear();
    for (auto& [socketKey, system] : activeSystems) {
        (void)socketKey;
        if (system) {
            retainedSystems.push_back(std::move(system));
        }
    }
    activeSystems.clear();
}
