#include "VoronoiSystem.hpp"

#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "voronoi/VoronoiCandidateCompute.hpp"
#include "voronoi/VoronoiModelRuntime.hpp"

#include <glm/mat4x4.hpp>
#include <iostream>

VoronoiSystem::VoronoiSystem(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    ModelRegistry& resourceManager,
    uint32_t maxFramesInFlight,
    CommandPool& renderCommandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      resourceManager(resourceManager),
      renderCommandPool(renderCommandPool),
      maxFramesInFlight(maxFramesInFlight) {

    voronoiSystemBuildStage = std::make_unique<VoronoiSystemBuildStage>(vulkanDevice, memoryAllocator, runtime.resourcesRef(), renderCommandPool);
    initializeVoronoiCandidateCompute();

    initialized = true;
}

VoronoiSystem::~VoronoiSystem() {
}

void VoronoiSystem::failInitialization(const char* stage) {
    std::cerr << "[VoronoiSystem] Initialization failed at stage: " << stage << std::endl;
    cleanupResources();
    cleanup();
}

void VoronoiSystem::setReceiverGeometry(
    const std::vector<uint32_t>& receiverNodeModelIds,
    const std::vector<std::vector<glm::vec3>>& receiverGeometryPositions,
    const std::vector<std::vector<uint32_t>>& receiverGeometryTriangleIndices,
    const std::vector<SupportingHalfedge::IntrinsicMesh>& receiverIntrinsicMeshes,
    const std::vector<std::vector<VoronoiModelRuntime::SurfaceVertex>>& receiverSurfaceVertices,
    const std::vector<std::vector<uint32_t>>& receiverIntrinsicTriangleIndices,
    const std::vector<uint32_t>& receiverModelIds,
    const std::vector<glm::mat4>& meshModelMatrices) {
    runtime.setReceiverGeometry(
        vulkanDevice,
        memoryAllocator,
        renderCommandPool,
        receiverNodeModelIds,
        receiverGeometryPositions,
        receiverGeometryTriangleIndices,
        receiverIntrinsicMeshes,
        receiverSurfaceVertices,
        receiverIntrinsicTriangleIndices,
        receiverModelIds,
        meshModelMatrices);

}

void VoronoiSystem::clearReceiverGeometry() {
    runtime.clearReceiverGeometry();
}

void VoronoiSystem::setParams(float cellSize, int voxelResolution) {
    runtime.setParams(cellSize, voxelResolution);
}

bool VoronoiSystem::ensureConfigured() {
    if (runtime.isReady()) {
        executeBufferTransfers();
        return true;
    }

    if (!rebuildVoronoiRuntime()) {
        return false;
    }

    executeBufferTransfers();
    return true;
}

void VoronoiSystem::initializeVoronoiCandidateCompute() {
    voronoiCandidateCompute = std::make_unique<VoronoiCandidateCompute>(vulkanDevice, renderCommandPool);
    if (voronoiCandidateCompute) {
        voronoiCandidateCompute->initialize();
    }
}

bool VoronoiSystem::rebuildVoronoiRuntime() {
    if (!voronoiSystemBuildStage->buildVoronoiDiagram(
            runtime,
            runtime.getCellSize(),
            runtime.getVoxelResolution(),
            K_NEIGHBORS)) {
        std::cerr << "[VoronoiSystem] Failed to build Voronoi diagram" << std::endl;
        return false;
    }

    voronoiSystemBuildStage->setGhostFromVoxelGrid(runtime);
    runtime.reorderNodes();

    runtime.markSeederReady();

    if (!voronoiSystemBuildStage->dispatchVoronoiCompute(runtime, debugEnable, K_NEIGHBORS)) {
        return false;
    }

    if (runtime.resourcesRef().voronoiNodeCount == 0) {
        return false;
    }

    if (!voronoiSystemBuildStage->stageSurfaceMappings(runtime)) {
        return false;
    }

    runtime.markReady();
    return true;
}

void VoronoiSystem::executeBufferTransfers() {
    dispatchVoronoiCandidateUpdates();
}

void VoronoiSystem::dispatchVoronoiCandidateUpdates() {
    if (!voronoiCandidateCompute || runtime.resourcesRef().voronoiNodeCount == 0) {
        return;
    }

    const VoronoiResources& resources = runtime.resourcesRef();
    size_t dispatchedCount = 0;
    size_t skippedMissingGeometryRuntime = 0;
    size_t skippedZeroFaces = 0;
    size_t skippedMissingCandidateBuffer = 0;

    for (const auto& modelRuntime : runtime.getModelRuntimes()) {
        const uint32_t receiverModelId = modelRuntime->getRuntimeModelId();
        uint32_t faceCount = static_cast<uint32_t>(modelRuntime->getIntrinsicTriangleCount());
        if (modelRuntime->getSurfaceBuffer() == VK_NULL_HANDLE) {
            ++skippedMissingGeometryRuntime;
            continue;
        }
        if (faceCount == 0) {
            ++skippedZeroFaces;
            continue;
        }
        if (modelRuntime->getVoronoiCandidateBuffer() == VK_NULL_HANDLE) {
            ++skippedMissingCandidateBuffer;
            continue;
        }

        VoronoiCandidateCompute::Bindings bindings{};
        bindings.vertexBuffer = modelRuntime->getSurfaceBuffer();
        bindings.vertexBufferOffset = modelRuntime->getSurfaceBufferOffset();
        bindings.faceIndexBuffer = modelRuntime->getTriangleIndicesBuffer();
        bindings.faceIndexBufferOffset = modelRuntime->getTriangleIndicesBufferOffset();
        bindings.seedPositionBuffer = resources.seedPositionBuffer;
        bindings.seedPositionBufferOffset = resources.seedPositionBufferOffset;
        bindings.candidateBuffer = modelRuntime->getVoronoiCandidateBuffer();
        bindings.candidateBufferOffset = modelRuntime->getVoronoiCandidateBufferOffset();

        voronoiCandidateCompute->updateDescriptors(bindings);
        voronoiCandidateCompute->dispatch(faceCount, runtime.getVoronoiNodeCount());
        ++dispatchedCount;
    }

}

void VoronoiSystem::cleanupResources() {
    runtime.cleanupResources(vulkanDevice);

    if (voronoiSystemBuildStage) {
        voronoiSystemBuildStage->cleanupResources();
    }
    if (voronoiCandidateCompute) {
        voronoiCandidateCompute->cleanupResources();
    }
}

void VoronoiSystem::cleanup() {
    runtime.cleanup(memoryAllocator);
}
