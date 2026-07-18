#include "HeatSystem.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

#include "heat/HeatModelRuntime.hpp"
#include "heat/HeatSystemPlayback.hpp"
#include "HeatSystemSimStage.hpp"
#include "HeatSystemSurfaceStage.hpp"
#include "HeatSystemDiffusionStage.hpp"
#include "heat/HeatGpuStructs.hpp"
#include "heat/HeatContactRuntime.hpp"
#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"

HeatSystem::HeatSystem(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    ModelRegistry& resourceManager,
    uint32_t maxFramesInFlight,
    CommandPool& renderCommandPool,
    CommandPool& transferCommandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      resourceManager(resourceManager),
      renderCommandPool(renderCommandPool),
      transferCommandPool(transferCommandPool),
      runtime(),
      maxFramesInFlight(maxFramesInFlight) {

    simStage = std::make_unique<HeatSystemSimStage>();
    surfaceStage = std::make_unique<HeatSystemSurfaceStage>(vulkanDevice);
    diffusionStage = std::make_unique<HeatSystemDiffusionStage>(vulkanDevice);

    if (!surfaceStage->createDescriptorPool(0) ||  
        !surfaceStage->createDescriptorSetLayout() ||
        !surfaceStage->createPipeline() ||
        !diffusionStage->createDescriptorPool(32) ||
        !diffusionStage->createDescriptorSetLayout() ||
        !diffusionStage->createPipeline()) {
        failInitialization("create compute resources");
        return;
    }

    if (!createComputeCommandBuffers(maxFramesInFlight)) {
        failInitialization("allocate compute command buffers");
        return;
    }

    initialized = true;
}

HeatSystem::~HeatSystem() {
    cleanup();
}

void HeatSystem::failInitialization(const char* stage) {
    std::cerr << "[HeatSystem] Initialization failed at stage: " << stage << std::endl;
    initialized = false;
    cleanup();
}

void HeatSystem::setHeatModels(
    const std::vector<std::vector<glm::vec3>>& modelSurfacePositions,
    const std::vector<std::vector<glm::vec3>>& modelSurfaceNormals,
    const std::vector<std::vector<uint32_t>>& modelSurfaceTriangleIndices,
    const std::vector<uint32_t>& modelRuntimeModelIds,
    const std::unordered_map<uint32_t, float>& modelInitialTemperaturesCByRuntimeId,
    const std::unordered_map<uint32_t, uint32_t>& modelBoundaryConditionTypesByRuntimeId,
    const std::unordered_map<uint32_t, float>& modelBoundaryTemperaturesCByRuntimeId,
    const std::unordered_map<uint32_t, float>& modelBoundaryHeatFluxesByRuntimeId,
    const std::unordered_map<uint32_t, float>& modelBoundaryHeatTransferCoefficientsByRuntimeId,
    const std::unordered_map<uint32_t, float>& modelVolumetricPowerDensitiesByRuntimeId,
    const std::unordered_map<uint32_t, float>& modelDensity,
    const std::unordered_map<uint32_t, float>& modelSpecificHeat,
    const std::unordered_map<uint32_t, float>& modelConductivity) {
    runtime.setHeatModels(
        modelSurfacePositions,
        modelSurfaceNormals,
        modelSurfaceTriangleIndices,
        modelRuntimeModelIds,
        modelInitialTemperaturesCByRuntimeId,
        modelBoundaryConditionTypesByRuntimeId,
        modelBoundaryTemperaturesCByRuntimeId,
        modelBoundaryHeatFluxesByRuntimeId,
        modelBoundaryHeatTransferCoefficientsByRuntimeId,
        modelVolumetricPowerDensitiesByRuntimeId,
        modelDensity,
        modelSpecificHeat,
        modelConductivity);
    this->modelRuntimeModelIds = modelRuntimeModelIds;
}


void HeatSystem::setParams(float updatedContactThermalConductance, float updatedSimulationDuration) {
    if (contactThermalConductance != updatedContactThermalConductance) {
        contactThermalConductance = updatedContactThermalConductance;
        heatParamsDirty = true;
    }
    if (timeline.getDuration() != updatedSimulationDuration) {
        timeline.setDuration(updatedSimulationDuration);
        heatParamsDirty = true;
    }
}

uint32_t HeatSystem::computeTimelineFrameCount() const {
    if (timeline.getDuration() <= 0.0f) {
        return 0;
    }
    return std::max(1u, static_cast<uint32_t>(std::ceil(timeline.getDuration() * TimelineFPS)));
}

void HeatSystem::setContactCouplings(const std::vector<ContactCoupling>& contactCouplings) {
    this->contactCouplings = contactCouplings;
    contactCouplingsDirty = true;
}

void HeatSystem::clearVoronoiInputs() {
    modelSimNodeBufferByModelId.clear();
    modelSimNodeBufferOffsetByModelId.clear();
    modelSimNodeCouplingBufferByModelId.clear();
    modelSimNodeCouplingBufferOffsetByModelId.clear();
    simNodeCounts.clear();
    simNodeCouplingCounts.clear();
    modelNodePositionsByModelId.clear();
    modelNodesByModelId.clear();
    modelNodeCouplingsByModelId.clear();
    modelSurfaceNodeIdsByModelId.clear();
    modelSurfacePatchAreasByModelId.clear();
    modelGMLSSurfaceStencilBufferByModelId.clear();
    modelGMLSSurfaceStencilBufferOffsetByModelId.clear();
    modelGMLSSurfaceWeightBufferByModelId.clear();
    modelGMLSSurfaceWeightBufferOffsetByModelId.clear();
    modelGMLSSurfaceWeightCountByModelId.clear();
    modelGMLSSurfaceGradientWeightBufferByModelId.clear();
    modelGMLSSurfaceGradientWeightBufferOffsetByModelId.clear();
    voronoiConfigDirty = true;
}

void HeatSystem::addVoronoiModelInput(
    uint32_t runtimeModelId,
    uint32_t simNodeCount,
    VkBuffer simNodeBuffer,
    VkDeviceSize simNodeBufferOffset,
    VkBuffer simNodeCouplingBuffer,
    VkDeviceSize simNodeCouplingBufferOffset,
    uint32_t simNodeCouplingCount,
    VkBuffer gmlsSurfaceStencilBuffer,
    VkDeviceSize gmlsSurfaceStencilBufferOffset,
    VkBuffer gmlsSurfaceWeightBuffer,
    VkDeviceSize gmlsSurfaceWeightBufferOffset,
    size_t gmlsSurfaceWeightCount,
    VkBuffer gmlsSurfaceGradientWeightBuffer,
    VkDeviceSize gmlsSurfaceGradientWeightBufferOffset,
    size_t gmlsSurfaceGradientWeightCount,
    const std::vector<glm::vec3>& nodePositions,
    const std::vector<voronoi::Node>& nodes,
    const std::vector<voronoi::NodeCoupling>& nodeCouplings,
    const std::vector<uint32_t>& surfaceNodeIds,
    const std::vector<float>& surfacePatchAreas) {
    if (runtimeModelId == 0 || simNodeCount == 0) {
        return;
    }
    
    modelSimNodeBufferByModelId[runtimeModelId] = simNodeBuffer;
    modelSimNodeBufferOffsetByModelId[runtimeModelId] = simNodeBufferOffset;
    modelSimNodeCouplingBufferByModelId[runtimeModelId] = simNodeCouplingBuffer;
    modelSimNodeCouplingBufferOffsetByModelId[runtimeModelId] = simNodeCouplingBufferOffset;
    simNodeCounts[runtimeModelId] = simNodeCount;
    simNodeCouplingCounts[runtimeModelId] = simNodeCouplingCount;
    modelGMLSSurfaceStencilBufferByModelId[runtimeModelId] = gmlsSurfaceStencilBuffer;
    modelGMLSSurfaceStencilBufferOffsetByModelId[runtimeModelId] = gmlsSurfaceStencilBufferOffset;
    modelGMLSSurfaceWeightBufferByModelId[runtimeModelId] = gmlsSurfaceWeightBuffer;
    modelGMLSSurfaceWeightBufferOffsetByModelId[runtimeModelId] = gmlsSurfaceWeightBufferOffset;
    modelGMLSSurfaceWeightCountByModelId[runtimeModelId] = gmlsSurfaceWeightCount;
    modelGMLSSurfaceGradientWeightBufferByModelId[runtimeModelId] = gmlsSurfaceGradientWeightBuffer;
    modelGMLSSurfaceGradientWeightBufferOffsetByModelId[runtimeModelId] = gmlsSurfaceGradientWeightBufferOffset;
    modelGMLSSurfaceGradientWeightCountByModelId[runtimeModelId] = gmlsSurfaceGradientWeightCount;
    modelNodePositionsByModelId[runtimeModelId] = nodePositions;
    modelNodesByModelId[runtimeModelId] = nodes;
    modelNodeCouplingsByModelId[runtimeModelId] = nodeCouplings;
    modelSurfaceNodeIdsByModelId[runtimeModelId] = surfaceNodeIds;
    modelSurfacePatchAreasByModelId[runtimeModelId] = surfacePatchAreas;
    voronoiConfigDirty = true;
}

void HeatSystem::update() {
    const auto currentTime = std::chrono::steady_clock::now();
    float deltaTime = std::chrono::duration<float>(currentTime - lastUpdateTime).count();
    lastUpdateTime = currentTime;

    const float maxFrameDelta = 1.0f / TimelineFPS;
    if (deltaTime > maxFrameDelta) {
        deltaTime = maxFrameDelta;
    }

    processResetTrigger();
    forwardSim(deltaTime);

    if (syntheticDirichletTestEnabled) {
        const float syntheticTemperatureC = 20.0f + 30.0f * (0.5f + 0.5f * std::sin(simulatedTime * 0.5f));
        for (const auto& [runtimeModelId, _] : runtime.getActiveModels()) {
            setRuntimeDirichletTemperatureC(runtimeModelId, 0u, syntheticTemperatureC);
        }
    }

    if (contactRuntime) {
        contactRuntime->clearSynchronization();
        if (shouldStepPhysics && !contactRuntime->solve(temperatureBufferAIsCurrent)) {
            std::cerr << "[HeatSystem] AmgX contact solve failed" << std::endl;
            shouldStepPhysics = false;
        }
    }
}

ComputePass::Synchronization HeatSystem::getSynchronization() const {
    return contactRuntime ? contactRuntime->getSynchronization() : ComputePass::Synchronization{};
}

bool HeatSystem::setRuntimeDirichletTemperatureC(uint32_t runtimeModelId, uint32_t regionId, float temperatureC) {
    HeatModelRuntime* heatModel = runtime.getModelByRuntimeId(runtimeModelId);
    if (!heatModel) {
        return false;
    }
    return heatModel->setRuntimeDirichletTemperatureC(regionId, temperatureC);
}

bool HeatSystem::setRuntimeNeumannHeatFlux(uint32_t runtimeModelId, uint32_t regionId, float heatFlux) {
    HeatModelRuntime* model = runtime.getModelByRuntimeId(runtimeModelId);
    return model && model->setNeumannHeatFlux(regionId, heatFlux);
}

bool HeatSystem::setRuntimeRobinState(uint32_t runtimeModelId, uint32_t regionId,
    float ambientTemperatureC, float heatTransferCoefficient) {
    HeatModelRuntime* model = runtime.getModelByRuntimeId(runtimeModelId);
    return model && model->setRobinState(regionId, ambientTemperatureC, heatTransferCoefficient);
}

bool HeatSystem::setRuntimeRobinTemperatureC(uint32_t runtimeModelId, uint32_t regionId, float ambientTemperatureC) {
    HeatModelRuntime* model = runtime.getModelByRuntimeId(runtimeModelId);
    return model && model->setRuntimeRobinTemperatureC(regionId, ambientTemperatureC);
}

bool HeatSystem::setRuntimeVolumetricPowerDensity(uint32_t runtimeModelId, float powerDensity) {
    HeatModelRuntime* model = runtime.getModelByRuntimeId(runtimeModelId);
    return model && model->setVolumetricPowerDensity(powerDensity);
}

void HeatSystem::processResetTrigger() {
    if (timeline.getResetCounter() != processedResetCounter) {
        processedResetCounter = timeline.getResetCounter();
        resetSimulationState();
    }
}

void HeatSystem::forwardSim(float deltaTime) {
    auto* playbackData = simRuntime.getMappedPlaybackData();
    if (!playbackData) {
        timeline.setPosition(0.0f);
        shouldStepPhysics = false;
        return;
    }

    const bool isPlaying = timeline.isPlaying();
    const bool isPaused = timeline.isPaused();
    const bool isScrubbing = timeline.isScrubbing();
    const float duration = timeline.getDuration();

    playbackData->resetCounter = timeline.getResetCounter();
    playbackData->recordedTimelineFrames = getRecordedTimelineFrames();
    playbackData->timelineFrameCount = computeTimelineFrameCount();

    const bool canAdvance = isPlaying && !isPaused && !isScrubbing && simulatedTime < duration;
    if (canAdvance) {
        physicsAccumulator = std::min(physicsAccumulator + deltaTime, HeatContactRuntime::FixedTimeStep);
    } else {
        physicsAccumulator = 0.0f;
    }
    shouldStepPhysics = canAdvance && physicsAccumulator >= HeatContactRuntime::FixedTimeStep;

    if (shouldStepPhysics) {
        const float remaining = std::max(0.0f, duration - simulatedTime);
        const float simDelta = std::min(HeatContactRuntime::FixedTimeStep, remaining);

        playbackData->deltaTime = simDelta / static_cast<float>(DefaultSubsteps);
        simulatedTime = std::min(duration, simulatedTime + simDelta);
        physicsAccumulator = std::max(0.0f, physicsAccumulator - HeatContactRuntime::FixedTimeStep);
    } else {
        playbackData->deltaTime = 0.0f;
    }

    // Advance display timeline
    if (isScrubbing) {
        timeline.setPosition(static_cast<float>(timeline.getScrubFrame()) / TimelineFPS);
    } else if (isPlaying && !isPaused) {
        timeline.advancePosition(deltaTime);
        timeline.setPosition(std::min(timeline.getCurrentPosition(), simulatedTime));
    }

}

bool HeatSystem::ensureConfigured() {
    const bool needsHardRebuild = runtime.needsRebuild() || contactCouplingsDirty || voronoiConfigDirty || heatParamsDirty;
    if (!needsHardRebuild) return true;


    const VkResult idleResult = vkDeviceWaitIdle(vulkanDevice.getDevice());
    if (idleResult != VK_SUCCESS) {
        std::cerr << "[HEAT-CONFIG] waitIdle failed"
                  << " result=" << static_cast<int>(idleResult)
                  << std::endl;
        return false;
    }

    if (contactRuntime) {
        contactRuntime->cleanup();
        contactRuntime.reset();
    }

    const bool modelsReady = runtime.ensureModelBindings(vulkanDevice, memoryAllocator, transferCommandPool);
    if (!modelsReady) {
        return false;
    }

    configureModelSimResources();

    const bool heatVoronoiReady = rebuildVoronoiRuntime();

    configureGMLSSurfaceWeights(heatVoronoiReady);

    if (!heatVoronoiReady) {
        if (contactRuntime) {
            contactRuntime->cleanup();
            contactRuntime.reset();
        }
        simRuntime.cleanup(memoryAllocator);
        return false;
    }

    // Initialize simulation buffers and playback on each model
    const uint32_t frameCapacity = computeHistoryFrameCapacity();
    for (const auto& [runtimeModelId, heatModel] : runtime.getActiveModels()) {
        if (!heatModel) continue;
        auto countIt = simNodeCounts.find(runtimeModelId);
        uint32_t nodeCount = (countIt != simNodeCounts.end()) ? countIt->second : 0;
        if (!heatModel->ensureSimulationBuffers(nodeCount)) {
            std::cerr << "[HEAT-CONFIG] ensureSimulationBuffers failed"
                      << " model=" << runtimeModelId
                      << " nodeCount=" << nodeCount
                      << std::endl;
            return false;
        }
        heatModel->initializePlayback(vulkanDevice, memoryAllocator, frameCapacity);
    }

    if (!rebuildContactRuntimes() || !resolveModelBoundaryAreas()) {
        return false;
    }
    resetSimulationState();

    voronoiConfigDirty = false;
    heatParamsDirty = false;
    return true;
}

bool HeatSystem::setupDescriptors(
    const std::vector<VkBuffer>& surfaceBuffers,
    const std::vector<VkDeviceSize>& surfaceOffsets,
    const std::vector<VkBuffer>& gradientBuffers,
    const std::vector<VkDeviceSize>& gradientOffsets) {
    if (!recreateDescriptorPools()) {
        return false;
    }

    const bool simReady = simRuntime.initialize(vulkanDevice, memoryAllocator) && diffusionStage;
    if (!simReady) return false;

    for (size_t i = 0; i < modelRuntimeModelIds.size() && i < surfaceBuffers.size(); ++i) {
        const uint32_t runtimeModelId = modelRuntimeModelIds[i];
        HeatModelRuntime* heatModel = runtime.getModelByRuntimeId(runtimeModelId);
        if (!heatModel) continue;

        auto itCount = simNodeCounts.find(runtimeModelId);
        uint32_t nodeCount = (itCount != simNodeCounts.end()) ? itCount->second : 0;
        if (nodeCount == 0) continue;

        auto* playback = heatModel->getPlayback();
        VkBuffer historyBuf = VK_NULL_HANDLE;
        VkDeviceSize historyOff = 0;
        uint32_t historyFrameCap = 0;
        if (playback) {
            historyBuf = playback->getHistoryBuffer();
            historyOff = playback->getHistoryBufferOffset();
            historyFrameCap = playback->getFrameCapacity();
        }
        heatModel->setHistoryBuffer(historyBuf, historyOff, historyFrameCap);
        if (!heatModel->updateAllDescriptors(
            surfaceBuffers[i],
            surfaceOffsets[i],
            gradientBuffers[i],
            gradientOffsets[i],
            surfaceStage->getDescriptorSetLayout(),
            surfaceStage->getGradientDescriptorSetLayout(),
            surfaceStage->getDescriptorPool(),
            diffusionStage->getDescriptorSetLayout(),
            diffusionStage->getDescriptorPool(),
            simRuntime.getPlaybackBuffer(),
            simRuntime.getPlaybackBufferOffset(),
            true)) {
            return false;
        }
    }

    return true;
}

void HeatSystem::configureGMLSSurfaceWeights(bool heatVoronoiReady) {
    for (const auto& [runtimeModelId, heatModel] : runtime.getActiveModels()) {
        if (!heatModel) continue;

        if (!heatVoronoiReady) {
            heatModel->setGMLSSurfaceWeights(VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 0, 0, VK_NULL_HANDLE, 0, 0);
            continue;
        }

        const auto stencilIt = modelGMLSSurfaceStencilBufferByModelId.find(runtimeModelId);
        const auto stencilOffsetIt = modelGMLSSurfaceStencilBufferOffsetByModelId.find(runtimeModelId);
        const auto valueWeightIt = modelGMLSSurfaceWeightBufferByModelId.find(runtimeModelId);
        const auto valueWeightOffsetIt = modelGMLSSurfaceWeightBufferOffsetByModelId.find(runtimeModelId);
        const auto valueWeightCountIt = modelGMLSSurfaceWeightCountByModelId.find(runtimeModelId);
        const auto gradientWeightIt = modelGMLSSurfaceGradientWeightBufferByModelId.find(runtimeModelId);
        const auto gradientWeightOffsetIt = modelGMLSSurfaceGradientWeightBufferOffsetByModelId.find(runtimeModelId);
        const auto gradientWeightCountIt = modelGMLSSurfaceGradientWeightCountByModelId.find(runtimeModelId);

        heatModel->setGMLSSurfaceWeights(
            stencilIt != modelGMLSSurfaceStencilBufferByModelId.end() ? stencilIt->second : VK_NULL_HANDLE,
            stencilOffsetIt != modelGMLSSurfaceStencilBufferOffsetByModelId.end() ? stencilOffsetIt->second : 0,
            valueWeightIt != modelGMLSSurfaceWeightBufferByModelId.end() ? valueWeightIt->second : VK_NULL_HANDLE,
            valueWeightOffsetIt != modelGMLSSurfaceWeightBufferOffsetByModelId.end() ? valueWeightOffsetIt->second : 0,
            valueWeightCountIt != modelGMLSSurfaceWeightCountByModelId.end() ? valueWeightCountIt->second : 0,
            gradientWeightIt != modelGMLSSurfaceGradientWeightBufferByModelId.end() ? gradientWeightIt->second : VK_NULL_HANDLE,
            gradientWeightOffsetIt != modelGMLSSurfaceGradientWeightBufferOffsetByModelId.end() ? gradientWeightOffsetIt->second : 0,
            gradientWeightCountIt != modelGMLSSurfaceGradientWeightCountByModelId.end() ? gradientWeightCountIt->second : 0);
    }
}

bool HeatSystem::recreateDescriptorPools() {
    uint32_t numModels = static_cast<uint32_t>(modelRuntimeModelIds.size());

    if (!surfaceStage->createDescriptorPool(numModels)) {
        return false;
    }

    if (!diffusionStage->createDescriptorPool(numModels)) {
        return false;
    }

    return true;
}

void HeatSystem::configureModelSimResources() {
    for (const auto& [runtimeModelId, heatModel] : runtime.getActiveModels()) {
        if (!heatModel) continue;

        const auto countIt = simNodeCounts.find(runtimeModelId);
        const auto nodeBufferIt = modelSimNodeBufferByModelId.find(runtimeModelId);
        const auto nodeBufferOffsetIt = modelSimNodeBufferOffsetByModelId.find(runtimeModelId);
        const auto couplingIt = modelSimNodeCouplingBufferByModelId.find(runtimeModelId);
        const auto couplingOffsetIt = modelSimNodeCouplingBufferOffsetByModelId.find(runtimeModelId);
        const auto couplingCountIt = simNodeCouplingCounts.find(runtimeModelId);
        const auto nodePositionsIt = modelNodePositionsByModelId.find(runtimeModelId);
        const auto nodesIt = modelNodesByModelId.find(runtimeModelId);
        const auto nodeCouplingsIt = modelNodeCouplingsByModelId.find(runtimeModelId);

        if (countIt == simNodeCounts.end() ||
            nodeBufferIt == modelSimNodeBufferByModelId.end() ||
            nodeBufferOffsetIt == modelSimNodeBufferOffsetByModelId.end() ||
            couplingIt == modelSimNodeCouplingBufferByModelId.end() ||
            couplingOffsetIt == modelSimNodeCouplingBufferOffsetByModelId.end() ||
            couplingCountIt == simNodeCouplingCounts.end()) {
            continue;
        }

        heatModel->setSimResources(
            nodeBufferIt->second,
            nodeBufferOffsetIt->second,
            countIt->second,
            couplingIt->second,
            couplingOffsetIt->second,
            couplingCountIt->second);

        if (nodePositionsIt != modelNodePositionsByModelId.end()) {
            heatModel->setNodePositions(nodePositionsIt->second);
        }
        if (nodesIt != modelNodesByModelId.end() &&
            nodeCouplingsIt != modelNodeCouplingsByModelId.end() &&
            nodesIt->second.size() == countIt->second &&
            nodeCouplingsIt->second.size() == couplingCountIt->second) {
            heatModel->setNodeTopology(nodesIt->second, nodeCouplingsIt->second);
        }
    }
}

bool HeatSystem::rebuildContactRuntimes() {
    if (contactRuntime) {
        contactRuntime->cleanup();
        contactRuntime.reset();
    }
    if (contactCouplings.empty()) {
        contactCouplingsDirty = false;
        return true;
    }

    auto runtimePtr = std::make_unique<HeatContactRuntime>();
    if (runtimePtr->build(
            vulkanDevice, runtime.getActiveModels(),
            contactCouplings, contactThermalConductance)) {
        contactRuntime = std::move(runtimePtr);
    } else {
        runtimePtr->cleanup();
        return false;
    }

    contactCouplingsDirty = false;
    return true;
}

bool HeatSystem::resolveModelBoundaryAreas() {
    for (const auto& [runtimeModelId, model] : runtime.getActiveModels()) {
        if (!model) {
            return false;
        }
        const std::vector<float>* coveredAreas = contactRuntime
            ? contactRuntime->findCoveredAreas(runtimeModelId)
            : nullptr;
        if (!model->resolveBoundaryContactAreas(
                coveredAreas ? *coveredAreas : std::vector<float>{})) {
            return false;
        }
    }
    return true;
}


void HeatSystem::setActive(bool active) {
    isActive = active;
}

void HeatSystem::setPlaybackState(bool paused, uint32_t resetCounter) {
    timeline.setPlaying(!paused);
    timeline.setResetCounter(resetCounter);
}

void HeatSystem::setRewindFrame(uint32_t frame) {
    timeline.setScrubFrame(frame);
    if (frame != heat::NoRewindFrame) {
        timeline.setPosition(static_cast<float>(frame) / TimelineFPS);
    }
}

uint32_t HeatSystem::getRewindFrame() const {
    return timeline.getScrubFrame();
}

uint32_t HeatSystem::getRecordedTimelineFrames() const {
    uint32_t maxRecorded = 0;
    for (const auto& [id, heatModel] : runtime.getActiveModels()) {
        if (heatModel) {
            auto* pb = heatModel->getPlayback();
            if (pb) maxRecorded = std::max(maxRecorded, pb->getRecordedFrameCount());
        }
    }
    return maxRecorded;
}

uint32_t HeatSystem::getTimelineFrameCount() const {
    return computeTimelineFrameCount();
}

void HeatSystem::resetSimulationState() {
    timeline.setPosition(0.0f);
    timeline.setScrubFrame(heat::NoRewindFrame);
    simulatedTime = 0.0f;
    shouldStepPhysics = false;
    needsInitialCapture = true;
    physicsAccumulator = 0.0f;
    temperatureBufferAIsCurrent = true;
    simRuntime.reset();
    resetVoronoiTemperatures();

    for (const auto& [id, heatModel] : runtime.getActiveModels()) {
        if (!heatModel) {
            continue;
        }

        auto* playback = heatModel->getPlayback();
        if (playback) {
            playback->reset();
        }
    }
}

void HeatSystem::resetVoronoiTemperatures() {
    for (const auto& [runtimeModelId, heatModel] : runtime.getActiveModels()) {
        if (!heatModel) continue;

        const float temperature = heatModel->getInitialTemperatureC();
        uint32_t nodeCount = heatModel->getSimNodeCount();
        if (nodeCount == 0) continue;

        // Upload temp values via staging buffer
        VkDeviceSize bufferSize = nodeCount * sizeof(float);
        std::vector<float> temps(nodeCount, temperature);

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceSize stagingOffset = 0;
        void* stagingMapped = nullptr;
        if (createStagingBuffer(memoryAllocator, bufferSize, stagingBuffer, stagingOffset, &stagingMapped) != VK_SUCCESS || !stagingMapped) {
            continue;
        }

        std::memcpy(stagingMapped, temps.data(), static_cast<size_t>(bufferSize));

        VkCommandBuffer cmd = transferCommandPool.beginCommands();
        if (cmd != VK_NULL_HANDLE) {
            // Copy to tempBufferA
            VkBufferCopy regionA{};
            regionA.srcOffset = stagingOffset;
            regionA.dstOffset = 0;
            regionA.size = bufferSize;
            vkCmdCopyBuffer(cmd, stagingBuffer, heatModel->getTempBufferA(), 1, &regionA);

            // Copy to tempBufferB
            VkBufferCopy regionB{};
            regionB.srcOffset = stagingOffset;
            regionB.dstOffset = 0;
            regionB.size = bufferSize;
            vkCmdCopyBuffer(cmd, stagingBuffer, heatModel->getTempBufferB(), 1, &regionB);

            const bool copyOk = transferCommandPool.endCommands(cmd);
            if (!copyOk) {
                std::cerr << "[HEAT-UPLOAD] resetTemperature failed"
                          << " model=" << runtimeModelId
                          << " nodeCount=" << nodeCount
                          << std::endl;
            }
        }

        memoryAllocator.free(stagingBuffer, stagingOffset);
    }
}

bool HeatSystem::createComputeCommandBuffers(uint32_t maxFramesInFlight) {
    computeCommandBuffers.resize(maxFramesInFlight);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = renderCommandPool.getHandle();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(computeCommandBuffers.size());

    if (vkAllocateCommandBuffers(vulkanDevice.getDevice(), &allocInfo, computeCommandBuffers.data()) != VK_SUCCESS) {
        computeCommandBuffers.clear();
        return false;
    }
    return true;
}

bool HeatSystem::hasDispatchableComputeWork() const {
    bool voronoiIsReady = voronoiReady();
    bool hasBuffers = !computeCommandBuffers.empty();
    return isActive && voronoiIsReady && hasBuffers;
}

bool HeatSystem::voronoiReady() const {
    if (!simRuntime.isInitialized() || modelRuntimeModelIds.empty()) {
        return false;
    }
    for (uint32_t runtimeModelId : modelRuntimeModelIds) {
        if (runtimeModelId == 0) {
            continue;
        }
        const auto countIt = simNodeCounts.find(runtimeModelId);
        const auto simNodeBufferIt = modelSimNodeBufferByModelId.find(runtimeModelId);
        const auto simCouplingIt = modelSimNodeCouplingBufferByModelId.find(runtimeModelId);
        const auto gmlsStencilIt = modelGMLSSurfaceStencilBufferByModelId.find(runtimeModelId);
        const auto gmlsWeightIt = modelGMLSSurfaceWeightBufferByModelId.find(runtimeModelId);
        const auto gmlsGradientIt = modelGMLSSurfaceGradientWeightBufferByModelId.find(runtimeModelId);
        if (countIt == simNodeCounts.end() ||
            countIt->second == 0 ||
            simNodeBufferIt == modelSimNodeBufferByModelId.end() ||
            !simNodeBufferIt->second ||
            simCouplingIt == modelSimNodeCouplingBufferByModelId.end() ||
            !simCouplingIt->second ||
            gmlsStencilIt == modelGMLSSurfaceStencilBufferByModelId.end() ||
            !gmlsStencilIt->second ||
            gmlsWeightIt == modelGMLSSurfaceWeightBufferByModelId.end() ||
            !gmlsWeightIt->second ||
            gmlsGradientIt == modelGMLSSurfaceGradientWeightBufferByModelId.end() ||
            !gmlsGradientIt->second) {
            return false;
        }
    }
    return true;
}

void HeatSystem::recordComputeCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        return;
    }

    bool hasWork = hasDispatchableComputeWork();
    if (hasWork &&
        simStage &&
        surfaceStage &&
        diffusionStage) {
        const auto* pbData = simRuntime.getMappedPlaybackData();
        const uint32_t recordedFrames = getRecordedTimelineFrames();

        // Capture the initial state on the first frame after a reset
        if (needsInitialCapture) {
            for (const auto& [id, heatModel] : runtime.getActiveModels()) {
                if (!heatModel || heatModel->getSimNodeCount() == 0) continue;

                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.buffer = heatModel->getTempBufferA();
                barrier.offset = 0;
                barrier.size = heatModel->getSimNodeCount() * sizeof(float);
                vkCmdPipelineBarrier(commandBuffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, nullptr, 1, &barrier, 0, nullptr);

                auto* pb = heatModel->getPlayback();
                if (pb) {
                    pb->recordFrame(commandBuffer,
                        heatModel->getTempBufferA(),
                        0);
                }
            }
            needsInitialCapture = false;
        }

        // Physics step
        const bool captureFrame = shouldStepPhysics && recordedFrames < computeHistoryFrameCapacity();

        const bool resultingAIsCurrent = simStage->recordComputeCommands(
            commandBuffer,
            runtime.getActiveModels(),
            *diffusionStage,
            shouldStepPhysics,
            captureFrame,
            temperatureBufferAIsCurrent,
            DefaultSubsteps);

        if (shouldStepPhysics) {
            temperatureBufferAIsCurrent = resultingAIsCurrent;
        }

        // Display pass
        const float displayTime = timeline.getCurrentPosition();
        const uint32_t displayFrame = static_cast<uint32_t>(displayTime * TimelineFPS);
        const bool replayFromHistory = displayFrame < recordedFrames;

        if (replayFromHistory) {
            for (const auto& [id, heatModel] : runtime.getActiveModels()) {
                if (!heatModel) continue;
                auto* pb = heatModel->getPlayback();
                if (!pb) continue;
                heatModel->updateHistoryDescriptorOffset(displayFrame, pb->getFrameStride(), currentFrame);
            }
        }

        const bool finalWritesBufferB = !temperatureBufferAIsCurrent;

        if (shouldStepPhysics) {
            std::vector<VkBufferMemoryBarrier> surfaceReadBarriers;
            for (const auto& [id, heatModel] : runtime.getActiveModels()) {
                if (!heatModel || heatModel->getSimNodeCount() == 0) continue;

                VkBuffer finalBuf = finalWritesBufferB ? heatModel->getTempBufferB() : heatModel->getTempBufferA();

                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barrier.buffer = finalBuf;
                barrier.offset = 0;
                barrier.size = heatModel->getSimNodeCount() * sizeof(float);
                surfaceReadBarriers.push_back(barrier);
            }
            if (!surfaceReadBarriers.empty()) {
                vkCmdPipelineBarrier(commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, 0, nullptr,
                    static_cast<uint32_t>(surfaceReadBarriers.size()), surfaceReadBarriers.data(),
                    0, nullptr);
            }
        }

        surfaceStage->dispatchSurfaceTemperatureUpdates(
            commandBuffer, runtime.getActiveModels(),
            replayFromHistory, finalWritesBufferB, currentFrame);

        if (currentFrame % 4 == 0) {
            surfaceStage->dispatchSurfaceGradientUpdates(
                commandBuffer, runtime.getActiveModels(),
                replayFromHistory, finalWritesBufferB, currentFrame);
        }
    }

    vkEndCommandBuffer(commandBuffer);
}

bool HeatSystem::configureMaterialNodes() {
    if (modelNodesByModelId.empty()) {
        return false;
    }

    for (const auto& [id, nodes] : modelNodesByModelId) {
        HeatModelRuntime* heatModelPtr = runtime.getModelByRuntimeId(id);
        if (!heatModelPtr) return false;

        auto countIt = simNodeCounts.find(id);
        if (countIt == simNodeCounts.end()) return false;

        float d = heatModelPtr->getDensity();
        float s = heatModelPtr->getSpecificHeat();
        float c = heatModelPtr->getConductivity();
        uint32_t modelNodeCount = countIt->second;
        if (modelNodeCount == 0 || nodes.size() != modelNodeCount) {
            return false;
        }

        std::vector<heat::MaterialNode> materialNodes(modelNodeCount);
        std::vector<float> nodalThermalMasses(modelNodeCount);
        for (uint32_t localNodeIndex = 0; localNodeIndex < modelNodeCount; ++localNodeIndex) {
            heat::MaterialNode& materialNode = materialNodes[localNodeIndex];
            materialNode.density = d;
            materialNode.specificHeat = s;
            materialNode.conductivity = c;
            const float volume = nodes[localNodeIndex].volume;
            if (!std::isfinite(volume) || volume <= 0.0f) {
                return false;
            }
            materialNode.thermalMass = d * s * volume;
            if (materialNode.thermalMass > 1e-20f) materialNode.conductivityPerMass = c / materialNode.thermalMass;
            nodalThermalMasses[localNodeIndex] = materialNode.thermalMass;
        }

        if (!heatModelPtr->createMaterialBuffer(materialNodes)) {
            return false;
        }

        heatModelPtr->setNodalThermalMasses(std::move(nodalThermalMasses));

    }

    return true;
}

bool HeatSystem::configureModelBoundaries() {
    for (const auto& [id, heatModelPtr] : runtime.getActiveModels()) {
        if (!heatModelPtr) {
            return false;
        }

        const auto surfaceNodesIt = modelSurfaceNodeIdsByModelId.find(id);
        const auto surfaceAreasIt = modelSurfacePatchAreasByModelId.find(id);
        if (surfaceNodesIt == modelSurfaceNodeIdsByModelId.end() ||
            surfaceAreasIt == modelSurfacePatchAreasByModelId.end()) {
            return false;
        }
        if (!heatModelPtr->configureBoundary(surfaceNodesIt->second, surfaceAreasIt->second)) {
            return false;
        }
    }

    return true;
}

bool HeatSystem::rebuildVoronoiRuntime() {
    if (modelRuntimeModelIds.empty() || modelSimNodeBufferByModelId.empty()) return false;
    return configureMaterialNodes() && configureModelBoundaries();
}

void HeatSystem::cleanup() {
    if (contactRuntime) {
        contactRuntime->cleanup();
        contactRuntime.reset();
    }
    simRuntime.cleanup(memoryAllocator);
    runtime.cleanup();
}
