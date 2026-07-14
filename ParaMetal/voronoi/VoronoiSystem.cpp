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

    voronoiSystemBuildStage = std::make_unique<VoronoiSystemBuildStage>(vulkanDevice, memoryAllocator, commandPool);
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

void VoronoiSystem::setMeshGeometry(
    const std::vector<glm::vec3>& geometryPositions,
    const std::vector<uint32_t>& geometryTriangleIndices,
    const std::vector<voronoi::SurfaceVertex>& surfaceVertices,
    const std::vector<uint32_t>& surfaceTriangleIndices,
    uint32_t runtimeModelId,
    const glm::mat4& meshModelMatrix) {
    runtime.setMeshGeometry(
        vulkanDevice,
        memoryAllocator,
        commandPool,
        geometryPositions,
        geometryTriangleIndices,
        surfaceVertices,
        surfaceTriangleIndices,
        runtimeModelId,
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

void VoronoiSystem::clearGeometry() {
    runtime.clearGeometry();
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

    if (voronoiSystemBuildStage->getCandidateNodeCount() == 0) {
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
    if (!voronoiCandidateCompute || voronoiSystemBuildStage->getCandidateNodeCount() == 0) {
        return;
    }

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
    uint32_t faceCount = static_cast<uint32_t>(modelRuntime->getSurfaceTriangleCount());
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
    bindings.seedPositionBuffer = voronoiSystemBuildStage->getSeedPositionBuffer();
    bindings.seedPositionBufferOffset = voronoiSystemBuildStage->getSeedPositionBufferOffset();
    bindings.candidateBuffer = modelRuntime->getCandidateBuffer();
    bindings.candidateBufferOffset = modelRuntime->getCandidateBufferOffset();

    voronoiCandidateCompute->updateDescriptors(bindings);
    voronoiCandidateCompute->dispatch(faceCount, voronoiSystemBuildStage->getCandidateNodeCount());
    ++dispatchedCount;

}

void VoronoiSystem::cleanupResources() {
    if (voronoiSystemBuildStage) {
        voronoiSystemBuildStage->cleanupResources();
    }
    if (voronoiCandidateCompute) {
        voronoiCandidateCompute->cleanupResources();
    }
}

void VoronoiSystem::cleanup() {
    if (voronoiSystemBuildStage) {
        voronoiSystemBuildStage->cleanup();
    }
    runtime.cleanup(memoryAllocator);
}
