#include "VoronoiSystem.hpp"

#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "voronoi/VoronoiCandidateCompute.hpp"
#include "voronoi/VoronoiModelRuntime.hpp"

#include <glm/mat4x4.hpp>

VoronoiSystem::VoronoiSystem(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    ModelRegistry& resourceManager,
    uint32_t maxFramesInFlight,
    CommandPool& commandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      resourceManager(resourceManager),
      commandPool(commandPool),
      maxFramesInFlight(maxFramesInFlight) {

    voronoiSystemBuildStage = std::make_unique<VoronoiSystemBuildStage>(vulkanDevice, memoryAllocator, runtime.resourcesRef(), commandPool);
    initializeVoronoiCandidateCompute();

    initialized = true;
}

VoronoiSystem::~VoronoiSystem() {
}

void VoronoiSystem::failInitialization(const char* stage) {
    (void)stage;
    cleanupResources();
    cleanup();
}

void VoronoiSystem::setReceiverGeometry(
    uint32_t receiverNodeModelId,
    const std::vector<glm::vec3>& receiverGeometryPositions,
    const std::vector<uint32_t>& receiverGeometryTriangleIndices,
    const SupportingHalfedge::IntrinsicMesh& receiverIntrinsicMesh,
    const std::vector<VoronoiModelRuntime::SurfaceVertex>& receiverSurfaceVertices,
    const std::vector<uint32_t>& receiverIntrinsicTriangleIndices,
    uint32_t receiverModelId,
    const glm::mat4& meshModelMatrix) {
    runtime.setReceiverGeometry(
        vulkanDevice,
        memoryAllocator,
        commandPool,
        receiverNodeModelId,
        receiverGeometryPositions,
        receiverGeometryTriangleIndices,
        receiverIntrinsicMesh,
        receiverSurfaceVertices,
        receiverIntrinsicTriangleIndices,
        receiverModelId,
        meshModelMatrix);

}

void VoronoiSystem::setPointGeometry(const std::vector<glm::vec4>& positions) {
    runtime.setPointGeometry(
        vulkanDevice,
        memoryAllocator,
        commandPool,
        0,  
        positions);
}

void VoronoiSystem::setSeedPositions(const std::vector<glm::vec4>& positions) {
    runtime.setSeedPositions(positions);
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
    voronoiCandidateCompute = std::make_unique<VoronoiCandidateCompute>(vulkanDevice, commandPool);
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
        return false;
    }

    voronoiSystemBuildStage->setGhostFromVoxelGrid(runtime);
    runtime.reorderNodes();

    runtime.markMeshGridReady();

    if (!voronoiSystemBuildStage->dispatchVoronoiCompute(runtime, debugEnable, K_NEIGHBORS)) {
        return false;
    }

    if (runtime.resourcesRef().voronoiNodeCount == 0) {
        return false;
    }

    if (VoronoiDomainRuntime* domainRuntime = runtime.getDomainRuntime()) {
        if (!domainRuntime->isPointDomain()) {
            if (!voronoiSystemBuildStage->stageSurfaceMappings(runtime)) {
                return false;
            }
        }
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

    VoronoiDomainRuntime* domainRuntime = runtime.getDomainRuntime();
    if (!domainRuntime) {
        return;
    }

    // Point domains have no triangle faces 
    if (domainRuntime->isPointDomain()) {
        return;
    }

    VoronoiModelRuntime* modelRuntime = static_cast<VoronoiModelRuntime*>(domainRuntime);
    const uint32_t receiverModelId = modelRuntime->getRuntimeModelId();
    (void)receiverModelId;
    uint32_t faceCount = static_cast<uint32_t>(modelRuntime->getIntrinsicTriangleCount());
    if (modelRuntime->getSurfaceBuffer() == VK_NULL_HANDLE) {
        ++skippedMissingGeometryRuntime;
        return;
    }
    if (faceCount == 0) {
        ++skippedZeroFaces;
        return;
    }
    if (modelRuntime->getCandidateBuffer() == VK_NULL_HANDLE) {
        ++skippedMissingCandidateBuffer;
        return;
    }

    VoronoiCandidateCompute::Bindings bindings{};
    bindings.vertexBuffer = modelRuntime->getSurfaceBuffer();
    bindings.vertexBufferOffset = modelRuntime->getSurfaceBufferOffset();
    bindings.faceIndexBuffer = modelRuntime->getTriangleIndicesBuffer();
    bindings.faceIndexBufferOffset = modelRuntime->getTriangleIndicesBufferOffset();
    bindings.seedPositionBuffer = resources.seedPositionBuffer;
    bindings.seedPositionBufferOffset = resources.seedPositionBufferOffset;
    bindings.candidateBuffer = modelRuntime->getCandidateBuffer();
    bindings.candidateBufferOffset = modelRuntime->getCandidateBufferOffset();

    voronoiCandidateCompute->updateDescriptors(bindings);
    voronoiCandidateCompute->dispatch(faceCount, runtime.getVoronoiNodeCount());
    ++dispatchedCount;

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
