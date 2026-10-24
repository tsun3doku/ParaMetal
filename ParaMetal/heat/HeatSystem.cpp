#include "HeatSystem.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_set>

#include "heat/HeatModelRuntime.hpp"
#include "heat/HeatSystemPlayback.hpp"
#include "HeatSystemSurfaceStage.hpp"
#include "heat/HeatFluidVolumeStage.hpp"
#include "heat/HeatGpuStructs.hpp"
#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "voronoi/VoronoiNodeIndex.hpp"
#include "util/GMLS.hpp"
#include "util/GeometryUtils.hpp"
#include "spatial/TriangleHashGrid.hpp"

HeatSystem::HeatSystem(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    uint32_t maxFramesInFlight,
    CommandPool& renderCommandPool,
    CommandPool& transferCommandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      renderCommandPool(renderCommandPool),
      transferCommandPool(transferCommandPool),
      maxFramesInFlight(maxFramesInFlight) {

    surfaceStage = std::make_unique<HeatSystemSurfaceStage>(vulkanDevice);
    fluidVolumeStage = std::make_unique<HeatFluidVolumeStage>(vulkanDevice, memoryAllocator, renderCommandPool, maxFramesInFlight);

    if (!surfaceStage->createDescriptorPool(0) ||  
        !surfaceStage->createDescriptorSetLayout() ||
        !surfaceStage->createPipeline()) {
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
    const std::unordered_map<uint32_t, float>& modelVolumetricPowerDensitiesByRuntimeId,
    const std::unordered_map<uint32_t, bool>& modelPrescribedTemperatureEnabledByRuntimeId,
    const std::unordered_map<uint32_t, float>& modelPrescribedTemperaturesCByRuntimeId,
    const std::unordered_map<uint32_t, float>& modelDensity,
    const std::unordered_map<uint32_t, float>& modelSpecificHeat,
    const std::unordered_map<uint32_t, float>& modelConductivity,
    units::LengthUnit worldUnit) {
    this->modelSurfacePositions = modelSurfacePositions;
    this->modelSurfaceNormals = modelSurfaceNormals;
    this->modelSurfaceTriangleIndices = modelSurfaceTriangleIndices;
    this->modelRuntimeModelIds = modelRuntimeModelIds;
    this->modelInitialTemperaturesCByRuntimeId = modelInitialTemperaturesCByRuntimeId;
    this->modelVolumetricPowerDensitiesByRuntimeId = modelVolumetricPowerDensitiesByRuntimeId;
    this->modelPrescribedTemperatureEnabledByRuntimeId = modelPrescribedTemperatureEnabledByRuntimeId;
    this->modelPrescribedTemperaturesCByRuntimeId = modelPrescribedTemperaturesCByRuntimeId;
    this->modelDensity = modelDensity;
    this->modelSpecificHeat = modelSpecificHeat;
    this->modelConductivity = modelConductivity;
    this->worldUnit = worldUnit;
    domainDirty = true;
}

float HeatSystem::getInitialTemperatureC(uint32_t runtimeModelId) const {
    const auto it = modelInitialTemperaturesCByRuntimeId.find(runtimeModelId);
    return it != modelInitialTemperaturesCByRuntimeId.end()
        ? it->second
        : HeatSimDefaults::ambientTemperatureC;
}

void HeatSystem::configureModelProperties(
    HeatModelRuntime& model,
    uint32_t runtimeModelId) const {
    const auto densityIt = modelDensity.find(runtimeModelId);
    const float density = densityIt != modelDensity.end()
        ? densityIt->second
        : HeatSimDefaults::density;
    const auto specificHeatIt = modelSpecificHeat.find(runtimeModelId);
    const float specificHeat = specificHeatIt != modelSpecificHeat.end()
        ? specificHeatIt->second
        : HeatSimDefaults::specificHeat;
    const auto conductivityIt = modelConductivity.find(runtimeModelId);
    const float conductivity = conductivityIt != modelConductivity.end()
        ? conductivityIt->second
        : HeatSimDefaults::conductivity;
    const auto powerDensityIt = modelVolumetricPowerDensitiesByRuntimeId.find(runtimeModelId);
    const float volumetricPowerDensity = powerDensityIt != modelVolumetricPowerDensitiesByRuntimeId.end()
        ? powerDensityIt->second
        : 0.0f;
    const auto prescribedEnabledIt = modelPrescribedTemperatureEnabledByRuntimeId.find(runtimeModelId);
    const bool prescribedEnabled = prescribedEnabledIt != modelPrescribedTemperatureEnabledByRuntimeId.end()
        ? prescribedEnabledIt->second : false;
    const auto prescribedTempIt = modelPrescribedTemperaturesCByRuntimeId.find(runtimeModelId);
    const float prescribedTemp = prescribedTempIt != modelPrescribedTemperaturesCByRuntimeId.end()
        ? prescribedTempIt->second : HeatSimDefaults::ambientTemperatureC;

    model.setMaterialProperties(density, specificHeat, conductivity);
    model.setWorldUnit(worldUnit);
    model.setInitialTemperatureC(getInitialTemperatureC(runtimeModelId));
    model.setVolumetricPowerDensity(volumetricPowerDensity);
    model.setPrescribedTemperature(prescribedEnabled, prescribedTemp);
}

bool HeatSystem::rebuildDomainRuntime() {
    if (!domainDirty) {
        return true;
    }

    const std::size_t modelCount = std::min({
        modelRuntimeModelIds.size(),
        modelSurfacePositions.size(),
        modelSurfaceNormals.size(),
        modelSurfaceTriangleIndices.size()});
    std::unordered_set<uint32_t> incomingModelIds;
    for (std::size_t index = 0; index < modelCount; ++index) {
        if (modelRuntimeModelIds[index] != 0) {
            incomingModelIds.insert(modelRuntimeModelIds[index]);
        }
    }
    domainRuntime.removeModelsNotIn(incomingModelIds);

    std::unordered_set<uint32_t> configuredModelIds;
    for (std::size_t index = 0; index < modelCount; ++index) {
        const uint32_t runtimeModelId = modelRuntimeModelIds[index];
        if (runtimeModelId == 0 || !configuredModelIds.insert(runtimeModelId).second) {
            continue;
        }

        const auto& positions = modelSurfacePositions[index];
        const auto& normals = modelSurfaceNormals[index];
        const auto& triangleIndices = modelSurfaceTriangleIndices[index];
        HeatModelRuntime* model = domainRuntime.getModelByRuntimeId(runtimeModelId);
        const bool geometryChanged = model &&
            (model->getSurfacePositions() != positions ||
             model->getSurfaceNormals() != normals ||
             model->getSurfaceTriangleIndices() != triangleIndices);

        if (!model || geometryChanged) {
            auto replacement = std::make_unique<HeatModelRuntime>(
                vulkanDevice,
                memoryAllocator,
                positions,
                normals,
                triangleIndices,
                getInitialTemperatureC(runtimeModelId));
            configureModelProperties(*replacement, runtimeModelId);
            if (replacement->isInitialized()) {
                domainRuntime.setModel(runtimeModelId, std::move(replacement));
            } else {
                std::cerr << "[HeatSystem] Failed to initialize heat model for model "
                          << runtimeModelId << std::endl;
                domainRuntime.removeModel(runtimeModelId);
            }
        } else {
            configureModelProperties(*model, runtimeModelId);
        }
    }

    domainDirty = false;
    return true;
}


void HeatSystem::setParams(float updatedSimulationDuration) {
    if (timeline.getDuration() != updatedSimulationDuration) {
        timeline.setDuration(updatedSimulationDuration);
        heatParamsDirty = true;
    }
}

void HeatSystem::setDomainBoundaryConditions(const heat::DomainBoundaryConditions& boundaries) {
    bool formulationChanged = false;
    bool valuesChanged = false;
    for (size_t i = 0; i < 6; ++i) {
        if (domainBoundaries[i].type != boundaries[i].type) {
            formulationChanged = true;
        }
        if (domainBoundaries[i].temperatureC != boundaries[i].temperatureC ||
            domainBoundaries[i].heatFlux != boundaries[i].heatFlux ||
            domainBoundaries[i].htc != boundaries[i].htc) {
            valuesChanged = true;
        }
    }
    domainBoundaries = boundaries;
    if (formulationChanged || valuesChanged) {
        heatParamsDirty = true;
        domainRuntime.setDomainBoundaryConditions(domainBoundaries);
        if (domainRuntime.isGlobalDomain()) {
            if (formulationChanged) {
                domainRuntime.markGlobalTopologyDirty();
            } else if (valuesChanged) {
                domainRuntime.markGlobalValuesDirty();
            }
        }
    }
}

void HeatSystem::setDomainBoundaryConditions(
    const uint32_t types[6],
    const float temperaturesC[6],
    const float heatFluxes[6],
    const float heatTransferCoefficients[6]) {
    setDomainBoundaryConditions(heat::fromFlatArrays(types, temperaturesC, heatFluxes, heatTransferCoefficients));
}

uint32_t HeatSystem::computeTimelineFrameCount() const {
    if (timeline.getDuration() <= 0.0f) {
        return 0;
    }
    return std::max(1u, static_cast<uint32_t>(std::ceil(timeline.getDuration() * TimelineFPS)));
}

void HeatSystem::releaseGmlsBuffers() {
    for (auto& [runtimeModelId, buffer] : modelGMLSSurfaceStencilBufferByModelId) {
        if (buffer != VK_NULL_HANDLE) {
            freeBuffer(memoryAllocator, buffer, modelGMLSSurfaceStencilBufferOffsetByModelId[runtimeModelId]);
        }
    }
    for (auto& [runtimeModelId, buffer] : modelGMLSSurfaceWeightBufferByModelId) {
        if (buffer != VK_NULL_HANDLE) {
            freeBuffer(memoryAllocator, buffer, modelGMLSSurfaceWeightBufferOffsetByModelId[runtimeModelId]);
        }
    }
    for (auto& [runtimeModelId, buffer] : modelGMLSSurfaceGradientWeightBufferByModelId) {
        if (buffer != VK_NULL_HANDLE) {
            freeBuffer(memoryAllocator, buffer, modelGMLSSurfaceGradientWeightBufferOffsetByModelId[runtimeModelId]);
        }
    }
    modelGMLSSurfaceStencilBufferByModelId.clear();
    modelGMLSSurfaceStencilBufferOffsetByModelId.clear();
    modelGMLSSurfaceWeightBufferByModelId.clear();
    modelGMLSSurfaceWeightBufferOffsetByModelId.clear();
    modelGMLSSurfaceWeightCountByModelId.clear();
    modelGMLSSurfaceGradientWeightBufferByModelId.clear();
    modelGMLSSurfaceGradientWeightBufferOffsetByModelId.clear();
    modelGMLSSurfaceGradientWeightCountByModelId.clear();
}

void HeatSystem::clearGlobalVoronoiInput() {
    releaseGmlsBuffers();
    simNodeCounts.clear();
    modelLocalToWorldByModelId.clear();
    domainRuntime.clearGlobalDomain();
    voronoiConfigDirty = true;
}

void HeatSystem::setHeatFluid(
    float density,
    float specificHeat,
    float conductivity,
    float initialTemperatureC) {
    if (!fluidRuntime) {
        fluidRuntime = std::make_unique<HeatFluidRuntime>();
    }
    fluidRuntime->setMaterialProperties(density, specificHeat, conductivity, initialTemperatureC);
    domainDirty = true;
}

void HeatSystem::clearHeatFluid() {
    if (fluidRuntime) {
        fluidRuntime->cleanup();
        fluidRuntime.reset();
    }
    domainDirty = true;
}

void HeatSystem::setAdvectionInput(
    VkBuffer faceFluxBuffer,
    VkDeviceSize faceFluxBufferSize,
    VkBuffer boundaryFluxBuffer,
    VkDeviceSize boundaryFluxBufferSize,
    uint32_t faceCount,
    uint32_t boundaryFaceCount) {
    HeatDomainRuntime::AdvectionInput input{};
    input.enabled = true;
    input.faceFluxBuffer = faceFluxBuffer;
    input.faceFluxBufferSize = faceFluxBufferSize;
    input.boundaryFluxBuffer = boundaryFluxBuffer;
    input.boundaryFluxBufferSize = boundaryFluxBufferSize;
    input.faceCount = faceCount;
    input.boundaryFaceCount = boundaryFaceCount;
    domainRuntime.setAdvectionInput(input);
}

void HeatSystem::clearAdvectionInput() {
    domainRuntime.clearAdvectionInput();
}

void HeatSystem::setGlobalVoronoiInput(
    const topology::GlobalDomain& domain,
    const std::unordered_map<uint32_t, std::array<float, 16>>& modelLocalToWorld) {
    if (!domain.isValid()) {
        clearGlobalVoronoiInput();
        return;
    }
    modelLocalToWorldByModelId = modelLocalToWorld;
    domainRuntime.setGlobalDomain(domain);
    domainDirty = true;
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
    physicsStepCompletedThisFrame = false;
    domainRuntime.clearGlobalSynchronization();
    domainRuntime.resetAdvectionStagingCopySchedule();
    domainRuntime.setAdvectionConnected(boundInputConnectionId != 0);

    if (domainRuntime.isGlobalDomain()) {
        bool completedDestinationIsA = false;
        bool newSolveCompleted = false;
        if (domainRuntime.pollGlobalSolve(completedDestinationIsA)) {
            if (domainRuntime.hasSolveFailed()) {
                // Terminal failure: halt heat physics until reset
                if (!physicsFailed) {
                    std::cerr << "[HeatSystem] PCG solve failed; heat physics halted until reset." << std::endl;
                }
                physicsFailed = true;
            } else {
                temperatureBufferAIsCurrent = completedDestinationIsA;
                physicsStepCompletedThisFrame = true;
                newSolveCompleted = true;
                ++completedSteps;
            }
        }

        if (!physicsFailed) {
            if (domainRuntime.globalValuesDirty()) {
                if (!domainRuntime.updateGlobalValues(domainRuntime.getActiveModels(), fluidRuntime.get(), FixedTimeStep)) {
                    std::cerr << "[HeatSystem] global value update failed; heat physics halted until reset." << std::endl;
                    physicsFailed = true;
                }
            }

            if (!physicsFailed && domainRuntime.globalAdvectionDirty()) {
                if (!domainRuntime.updateGlobalAdvection(fluidRuntime.get(), FixedTimeStep)) {
                    std::cerr << "[HeatSystem] global advection update failed; heat physics halted until reset." << std::endl;
                    physicsFailed = true;
                }
            }
        }

        forwardSim(deltaTime);

        if (!physicsFailed && shouldStepPhysics && !domainRuntime.isGlobalSolveInFlight()) {
            if (domainRuntime.launchGlobalSolve(temperatureBufferAIsCurrent)) {
                physicsAccumulator = std::max(0.0f, physicsAccumulator - FixedTimeStep);
            }
            shouldStepPhysics = false;
        }

        domainRuntime.prepareGlobalRenderSynchronization(newSolveCompleted);
    } else {
        forwardSim(deltaTime);

        physicsStepCompletedThisFrame = shouldStepPhysics;
    }
}

ComputePass::Synchronization HeatSystem::getSynchronization() const {
    if (!domainRuntime.isGlobalDomain()) {
        return ComputePass::Synchronization{};
    }
    return domainRuntime.getGlobalSynchronization();
}

bool HeatSystem::setRuntimeVolumetricPowerDensity(uint32_t runtimeModelId, float powerDensity) {
    HeatModelRuntime* model = domainRuntime.getModelByRuntimeId(runtimeModelId);
    if (!model || !model->setVolumetricPowerDensity(powerDensity)) {
        return false;
    }
    // Keep the authored store in sync so a later reconfiguration cannot revert this
    modelVolumetricPowerDensitiesByRuntimeId[runtimeModelId] = powerDensity;
    domainRuntime.markGlobalValuesDirty();
    return true;
}

bool HeatSystem::setRuntimePrescribedTemperature(uint32_t runtimeModelId, float tempC) {
    HeatModelRuntime* model = domainRuntime.getModelByRuntimeId(runtimeModelId);
    if (!model) {
        return false;
    }
    model->setPrescribedTemperature(model->hasPrescribedTemperature(), tempC);
    modelPrescribedTemperaturesCByRuntimeId[runtimeModelId] = tempC;
    domainRuntime.markGlobalValuesDirty();
    return true;
}

void HeatSystem::setInitialTemperatures(
    const std::unordered_map<uint32_t, float>& temperatures) {
    modelInitialTemperaturesCByRuntimeId = temperatures;
    for (const auto& [runtimeModelId, temperatureC] : temperatures) {
        HeatModelRuntime* model = domainRuntime.getModelByRuntimeId(runtimeModelId);
        if (model) {
            model->setInitialTemperatureC(temperatureC);
        }
    }
}

bool HeatSystem::setRuntimeFluidInitialTemperature(float temperatureC) {
    if (!fluidRuntime) {
        return false;
    }
    fluidRuntime->setInitialTemperatureC(temperatureC);
    return true;
}

void HeatSystem::processResetTrigger() {
    if (timeline.getResetCounter() != processedResetCounter) {
        processedResetCounter = timeline.getResetCounter();
        resetSimulationState();
    }
}

void HeatSystem::forwardSim(float deltaTime) {
    auto* playbackData = playbackRuntime.getMappedPlaybackData();
    if (!playbackData) {
        timeline.setPosition(0.0f);
        shouldStepPhysics = false;
        return;
    }

    const bool isPlaying = timeline.isPlaying();
    const bool isScrubbing = timeline.isScrubbing();
    const float duration = timeline.getDuration();

    playbackData->resetCounter = timeline.getResetCounter();
    playbackData->recordedTimelineFrames = getRecordedTimelineFrames();
    playbackData->timelineFrameCount = computeTimelineFrameCount();

    const bool canAdvance = isPlaying && !isScrubbing && simulatedTime < duration;
    if (canAdvance) {
        physicsAccumulator = std::min(physicsAccumulator + deltaTime, FixedTimeStep);
    } else {
        physicsAccumulator = 0.0f;
    }
    shouldStepPhysics = canAdvance && physicsAccumulator >= FixedTimeStep;

    if (shouldStepPhysics) {
        const float remaining = std::max(0.0f, duration - simulatedTime);
        const float simDelta = std::min(FixedTimeStep, remaining);

        playbackData->deltaTime = simDelta;
        if (!domainRuntime.isGlobalDomain()) {
            simulatedTime = std::min(duration, simulatedTime + simDelta);
            physicsAccumulator = std::max(0.0f, physicsAccumulator - FixedTimeStep);
        }
    } else {
        playbackData->deltaTime = 0.0f;
    }

    if (domainRuntime.isGlobalDomain() && physicsStepCompletedThisFrame) {
        const float remaining = std::max(0.0f, duration - simulatedTime);
        const float simDelta = std::min(FixedTimeStep, remaining);
        simulatedTime = std::min(duration, simulatedTime + simDelta);
    }

    // Advance display timeline
    if (isScrubbing) {
        timeline.setPosition(static_cast<float>(timeline.getScrubFrame()) / TimelineFPS);
    } else if (isPlaying) {
        timeline.advancePosition(deltaTime);
        timeline.setPosition(std::min(timeline.getCurrentPosition(), simulatedTime));
    }

}

bool HeatSystem::ensureConfigured() {
    const bool needsHardRebuild = domainDirty || voronoiConfigDirty || heatParamsDirty;
    if (!needsHardRebuild) return true;

    const bool modelsReady = rebuildDomainRuntime();
    if (!modelsReady) {
        std::cerr << "[HeatSystem-Diag] ensureConfigured FAILED: rebuildDomainRuntime"
                  << " domainDirty=" << domainDirty
                  << " activeModels=" << domainRuntime.getActiveModels().size() << std::endl;
        return false;
    }

    configureModelSimResources();

    for (const auto& [runtimeModelId, heatModel] : domainRuntime.getActiveModels()) {
        if (!heatModel) continue;
        const auto countIt = simNodeCounts.find(runtimeModelId);
        const uint32_t nodeCount = (countIt != simNodeCounts.end()) ? countIt->second : 0;
        if (!heatModel->ensureSimulationBuffers(nodeCount)) {
            std::cerr << "[HeatSystem-Diag] ensureConfigured FAILED: ensureSimulationBuffers"
                      << " runtimeModelId=" << runtimeModelId
                      << " nodeCount=" << nodeCount << std::endl;
            return false;
        }
    }

    const bool heatVoronoiReady = domainRuntime.isGlobalDomain() && rebuildGlobalThermalDomain();

    configureGMLSSurfaceWeights(heatVoronoiReady);

    if (!heatVoronoiReady) {
        std::cerr << "[HeatSystem-Diag] ensureConfigured FAILED: rebuildVoronoiRuntime"
                  << " modelRuntimeModelIds=" << modelRuntimeModelIds.size()
                  << " activeModels=" << domainRuntime.getActiveModels().size() << std::endl;
        playbackRuntime.cleanup(memoryAllocator);
        return false;
    }

    // Initialize playback after the rebuilt simulation resources are valid
    const uint32_t frameCapacity = computeHistoryFrameCapacity();
    for (const auto& [runtimeModelId, heatModel] : domainRuntime.getActiveModels()) {
        (void)runtimeModelId;
        if (!heatModel) continue;
        heatModel->initializePlayback(vulkanDevice, memoryAllocator, frameCapacity);
    }
    if (fluidRuntime) {
        fluidRuntime->initializePlayback(vulkanDevice, memoryAllocator, frameCapacity);
    }

    if (domainRuntime.isGlobalDomain()) {
        const auto& models = domainRuntime.getActiveModels();
        if (domainRuntime.globalTopologyDirty() || !domainRuntime.hasGlobalSolver()) {
            if (!domainRuntime.rebuildGlobalSolver(models, fluidRuntime.get(), vulkanDevice, FixedTimeStep)) {
                std::cerr << "[HeatSystem-Diag] ensureConfigured FAILED: rebuildGlobalSolver"
                          << " activeModels=" << models.size() << std::endl;
                playbackRuntime.cleanup(memoryAllocator);
                return false;
            }
            if (fluidVolumeStage) {
                const auto& domain = domainRuntime.getGlobalDomain();
                if (!fluidVolumeStage->initialize(domain.volumeGridDim) ||
                    !fluidVolumeStage->updateSeedToFluidNode(domain.seedPositions, domain, domainRuntime.getGlobalFragmentToFluidNode())) {
                    std::cerr << "[HeatSystem-Diag] fluidVolumeStage configuration failed" << std::endl;
                }
            }
        } else if (domainRuntime.globalValuesDirty()) {
            if (!domainRuntime.updateGlobalValues(models, fluidRuntime.get(), FixedTimeStep)) {
                std::cerr << "[HeatSystem-Diag] ensureConfigured FAILED: updateGlobalValues"
                          << " activeModels=" << models.size() << std::endl;
                playbackRuntime.cleanup(memoryAllocator);
                return false;
            }
        }
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

    if (!playbackRuntime.initialize(vulkanDevice, memoryAllocator)) return false;

    for (size_t i = 0; i < modelRuntimeModelIds.size() && i < surfaceBuffers.size(); ++i) {
        const uint32_t runtimeModelId = modelRuntimeModelIds[i];
        HeatModelRuntime* heatModel = domainRuntime.getModelByRuntimeId(runtimeModelId);
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
            playbackRuntime.getPlaybackBuffer(),
            playbackRuntime.getPlaybackBufferOffset(),
            true)) {
            return false;
        }
    }

    // Models retain surface buffers only after descriptors update, partial failure leaves ownership with caller
    for (size_t i = 0; i < modelRuntimeModelIds.size() && i < surfaceBuffers.size(); ++i) {
        HeatModelRuntime* heatModel = domainRuntime.getModelByRuntimeId(modelRuntimeModelIds[i]);
        if (!heatModel) continue;
        heatModel->retainSurfaceBuffers(
            surfaceBuffers[i],
            surfaceOffsets[i],
            gradientBuffers[i],
            gradientOffsets[i]);
    }

    return true;
}

void HeatSystem::configureGMLSSurfaceWeights(bool heatVoronoiReady) {
    for (const auto& [runtimeModelId, heatModel] : domainRuntime.getActiveModels()) {
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

    return true;
}

void HeatSystem::configureModelSimResources() {
    if (domainRuntime.isGlobalDomain()) {
        const auto& domain = domainRuntime.getGlobalDomain();
        const auto& models = domainRuntime.getActiveModels();
        std::unordered_map<uint32_t, uint32_t> fragmentCounts;
        for (const auto& fragment : domain.fragments) {
            fragmentCounts[fragment.instanceId] = fragmentCounts[fragment.instanceId] + 1;
        }

        for (const auto& [runtimeModelId, heatModel] : models) {
            if (!heatModel) continue;
            const auto countIt = fragmentCounts.find(runtimeModelId);
            if (countIt == fragmentCounts.end() || countIt->second == 0) {
                continue;
            }
            const uint32_t nodeCount = countIt->second;
            simNodeCounts[runtimeModelId] = nodeCount;
        }
    }
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
    for (const auto& [id, heatModel] : domainRuntime.getActiveModels()) {
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
    physicsFailed = false;
    needsInitialCapture = true;
    physicsAccumulator = 0.0f;
    temperatureBufferAIsCurrent = true;
    playbackRuntime.reset();
    resetVoronoiTemperatures();

    for (const auto& [id, heatModel] : domainRuntime.getActiveModels()) {
        if (!heatModel) {
            continue;
        }

        auto* playback = heatModel->getPlayback();
        if (playback) {
            playback->reset();
        }
    }

    if (fluidRuntime && fluidRuntime->getPlayback()) {
        fluidRuntime->getPlayback()->reset();
    }
    fluidVolumeDirty = true;
    surfaceDataDirty = true;
}

void HeatSystem::resetVoronoiTemperatures() {
    for (const auto& [runtimeModelId, heatModel] : domainRuntime.getActiveModels()) {
        if (!heatModel) continue;

        const float resetTemperatureC = heatModel->hasPrescribedTemperature()
            ? heatModel->getPrescribedTemperatureC()
            : heatModel->getInitialTemperatureC();
        uint32_t nodeCount = heatModel->getSimNodeCount();
        if (nodeCount == 0) continue;

        // Upload temp values via staging buffer
        VkDeviceSize bufferSize = nodeCount * sizeof(float);
        std::vector<float> temps(nodeCount, resetTemperatureC);

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

            transferCommandPool.endCommands(cmd);
        }

        memoryAllocator.free(stagingBuffer, stagingOffset);
    }

    if (fluidRuntime && fluidRuntime->getNodeCount() > 0) {
        const float resetTemperatureC = fluidRuntime->getInitialTemperatureC();
        const uint32_t nodeCount = fluidRuntime->getNodeCount();
        VkDeviceSize bufferSize = nodeCount * sizeof(float);
        std::vector<float> temps(nodeCount, resetTemperatureC);

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceSize stagingOffset = 0;
        void* stagingMapped = nullptr;
        if (createStagingBuffer(memoryAllocator, bufferSize, stagingBuffer, stagingOffset, &stagingMapped) == VK_SUCCESS && stagingMapped) {
            std::memcpy(stagingMapped, temps.data(), static_cast<size_t>(bufferSize));

            VkCommandBuffer cmd = transferCommandPool.beginCommands();
            if (cmd != VK_NULL_HANDLE) {
                VkBufferCopy regionA{};
                regionA.srcOffset = stagingOffset;
                regionA.dstOffset = 0;
                regionA.size = bufferSize;
                vkCmdCopyBuffer(cmd, stagingBuffer, fluidRuntime->getTempBufferA(), 1, &regionA);

                VkBufferCopy regionB{};
                regionB.srcOffset = stagingOffset;
                regionB.dstOffset = 0;
                regionB.size = bufferSize;
                vkCmdCopyBuffer(cmd, stagingBuffer, fluidRuntime->getTempBufferB(), 1, &regionB);

                transferCommandPool.endCommands(cmd);
            }

            memoryAllocator.free(stagingBuffer, stagingOffset);
        }
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
    if (!playbackRuntime.isInitialized() || modelRuntimeModelIds.empty()) {
        return false;
    }
    if (domainRuntime.isGlobalDomain()) {
        return domainRuntime.hasGlobalSolver();
    }
    return false;
}

void HeatSystem::recordComputeCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        return;
    }

    // Monotonic rendered-frame counter, frame slot index cannot drive periodic cadence
    ++renderedFrameCounter;

    if (domainRuntime.advectionStagingCopyScheduled()) {
        const auto& adv = domainRuntime.getAdvectionInput();
        std::vector<VkBufferMemoryBarrier> barriers;
        if (adv.faceFluxBuffer != VK_NULL_HANDLE && domainRuntime.getAdvectionFaceFluxBuffer() != VK_NULL_HANDLE && adv.faceFluxBufferSize > 0) {
            VkBufferCopy copyRegion{};
            copyRegion.size = adv.faceFluxBufferSize;
            vkCmdCopyBuffer(commandBuffer, adv.faceFluxBuffer, domainRuntime.getAdvectionFaceFluxBuffer(), 1, &copyRegion);

            VkBufferMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
            barrier.buffer = domainRuntime.getAdvectionFaceFluxBuffer();
            barrier.offset = 0;
            barrier.size = adv.faceFluxBufferSize;
            barriers.push_back(barrier);
        }
        if (adv.boundaryFluxBuffer != VK_NULL_HANDLE && domainRuntime.getAdvectionBoundaryFluxBuffer() != VK_NULL_HANDLE && adv.boundaryFluxBufferSize > 0) {
            VkBufferCopy copyRegion{};
            copyRegion.size = adv.boundaryFluxBufferSize;
            vkCmdCopyBuffer(commandBuffer, adv.boundaryFluxBuffer, domainRuntime.getAdvectionBoundaryFluxBuffer(), 1, &copyRegion);

            VkBufferMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
            barrier.buffer = domainRuntime.getAdvectionBoundaryFluxBuffer();
            barrier.offset = 0;
            barrier.size = adv.boundaryFluxBufferSize;
            barriers.push_back(barrier);
        }
        if (!barriers.empty()) {
            vkCmdPipelineBarrier(
                commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0, 0, nullptr,
                static_cast<uint32_t>(barriers.size()), barriers.data(),
                0, nullptr);
        }
    }

    bool hasWork = hasDispatchableComputeWork();
    if (hasWork) {
        const auto* pbData = playbackRuntime.getMappedPlaybackData();
        const uint32_t recordedFrames = getRecordedTimelineFrames();

        // Capture the initial state on the first frame after a reset
        if (needsInitialCapture) {
            for (const auto& [id, heatModel] : domainRuntime.getActiveModels()) {
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

            if (fluidRuntime && fluidRuntime->getNodeCount() > 0 && fluidRuntime->getPlayback()) {
                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.buffer = fluidRuntime->getTempBufferA();
                barrier.offset = 0;
                barrier.size = fluidRuntime->getNodeCount() * sizeof(float);
                vkCmdPipelineBarrier(commandBuffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, nullptr, 1, &barrier, 0, nullptr);

                fluidRuntime->getPlayback()->recordFrame(commandBuffer, fluidRuntime->getTempBufferA(), 0);
            }
            needsInitialCapture = false;
        }

        // Physics step
        const bool captureFrame = physicsStepCompletedThisFrame && recordedFrames < computeHistoryFrameCapacity();

        if (captureFrame) {
            std::vector<VkBufferMemoryBarrier> captureBarriers;
            for (const auto& [id, heatModel] : domainRuntime.getActiveModels()) {
                if (!heatModel || heatModel->getSimNodeCount() == 0) continue;

                VkBuffer finalBuf = temperatureBufferAIsCurrent
                    ? heatModel->getTempBufferA()
                    : heatModel->getTempBufferB();

                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.buffer = finalBuf;
                barrier.offset = 0;
                barrier.size = heatModel->getSimNodeCount() * sizeof(float);
                captureBarriers.push_back(barrier);
            }

            if (fluidRuntime && fluidRuntime->getNodeCount() > 0 && fluidRuntime->getPlayback()) {
                VkBuffer finalBuf = temperatureBufferAIsCurrent
                    ? fluidRuntime->getTempBufferA()
                    : fluidRuntime->getTempBufferB();

                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.buffer = finalBuf;
                barrier.offset = 0;
                barrier.size = fluidRuntime->getNodeCount() * sizeof(float);
                captureBarriers.push_back(barrier);
            }

            if (!captureBarriers.empty()) {
                vkCmdPipelineBarrier(commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, nullptr,
                    static_cast<uint32_t>(captureBarriers.size()), captureBarriers.data(),
                    0, nullptr);
            }
            for (const auto& [id, heatModel] : domainRuntime.getActiveModels()) {
                if (!heatModel || heatModel->getSimNodeCount() == 0) continue;
                auto* pb = heatModel->getPlayback();
                if (!pb) continue;
                VkBuffer finalBuf = temperatureBufferAIsCurrent
                    ? heatModel->getTempBufferA()
                    : heatModel->getTempBufferB();
                pb->recordFrame(commandBuffer, finalBuf, 0);
            }
            if (fluidRuntime && fluidRuntime->getNodeCount() > 0 && fluidRuntime->getPlayback()) {
                VkBuffer finalBuf = temperatureBufferAIsCurrent
                    ? fluidRuntime->getTempBufferA()
                    : fluidRuntime->getTempBufferB();
                fluidRuntime->getPlayback()->recordFrame(commandBuffer, finalBuf, 0);
            }
            domainRuntime.noteTemperatureReader(temperatureBufferAIsCurrent);
        }

        // Display pass
        const float displayTime = timeline.getCurrentPosition();
        const uint32_t displayFrame = static_cast<uint32_t>(displayTime * TimelineFPS);
        const bool replayFromHistory = displayFrame < recordedFrames;

        const bool displayDataChanged = physicsStepCompletedThisFrame ||
            replayFromHistory != lastFluidReplayFromHistory ||
            (replayFromHistory && displayFrame != lastFluidDisplayFrame);
        if (displayDataChanged) {
            fluidVolumeDirty = true;
            surfaceDataDirty = true;
        }
        lastFluidReplayFromHistory = replayFromHistory;
        lastFluidDisplayFrame = displayFrame;

        if (replayFromHistory) {
            for (const auto& [id, heatModel] : domainRuntime.getActiveModels()) {
                if (!heatModel) continue;
                auto* pb = heatModel->getPlayback();
                if (!pb) continue;
                heatModel->updateHistoryDescriptorOffset(displayFrame, pb->getFrameStride(), currentFrame);
            }
        }

        const bool currentIsB = !temperatureBufferAIsCurrent;

        if (physicsStepCompletedThisFrame) {
            std::vector<VkBufferMemoryBarrier> surfaceReadBarriers;
            for (const auto& [id, heatModel] : domainRuntime.getActiveModels()) {
                if (!heatModel || heatModel->getSimNodeCount() == 0) continue;

                VkBuffer finalBuf = currentIsB ? heatModel->getTempBufferB() : heatModel->getTempBufferA();

                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barrier.buffer = finalBuf;
                barrier.offset = 0;
                barrier.size = heatModel->getSimNodeCount() * sizeof(float);
                surfaceReadBarriers.push_back(barrier);
            }
            if (fluidRuntime && fluidRuntime->getNodeCount() > 0) {
                VkBuffer finalBuf = currentIsB ? fluidRuntime->getTempBufferB() : fluidRuntime->getTempBufferA();

                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                barrier.buffer = finalBuf;
                barrier.offset = 0;
                barrier.size = fluidRuntime->getNodeCount() * sizeof(float);
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

        if (surfaceDataDirty) {
            if (!replayFromHistory) {
                domainRuntime.noteTemperatureReader(temperatureBufferAIsCurrent);
            }
            surfaceStage->dispatchSurfaceTemperatureUpdates(
                commandBuffer, domainRuntime.getActiveModels(),
                replayFromHistory, currentIsB, currentFrame);

            if (renderedFrameCounter % 4 == 0) {
                surfaceStage->dispatchSurfaceGradientUpdates(
                    commandBuffer, domainRuntime.getActiveModels(),
                    replayFromHistory, currentIsB, currentFrame);
            }
            surfaceDataDirty = false;
        }

        if (fluidVolumeDirty && fluidVolumeStage && fluidVolumeStage->isValid() && domainRuntime.isGlobalDomain() && fluidRuntime && fluidRuntime->getNodeCount() > 0) {
            const auto& domain = domainRuntime.getGlobalDomain();
            if (domain.seedIdGridBuffer != VK_NULL_HANDLE) {
                VkBuffer sourceBuf = VK_NULL_HANDLE;
                VkDeviceSize sourceOff = 0;
                if (replayFromHistory && fluidRuntime->getPlayback() && fluidRuntime->getPlayback()->getHistoryBuffer() != VK_NULL_HANDLE) {
                    auto* playback = fluidRuntime->getPlayback();
                    sourceBuf = playback->getHistoryBuffer();
                    sourceOff = playback->getHistoryBufferOffset() + static_cast<VkDeviceSize>(displayFrame) * playback->getFrameStride();
                } else {
                    sourceBuf = currentIsB ? fluidRuntime->getTempBufferB() : fluidRuntime->getTempBufferA();
                    sourceOff = 0;
                    domainRuntime.noteTemperatureReader(temperatureBufferAIsCurrent);
                }

                if (sourceBuf != VK_NULL_HANDLE) {
                    fluidVolumeStage->recordCompute(
                        commandBuffer,
                        currentFrame,
                        domain.seedIdGridBuffer,
                        domain.seedIdGridBufferOffset,
                        sourceBuf,
                        sourceOff);
                    fluidVolumeDirty = false;
                }
            }
        }
    }

    vkEndCommandBuffer(commandBuffer);
}

bool HeatSystem::rebuildGlobalThermalDomain() {
    if (!domainRuntime.isGlobalDomain()) {
        return false;
    }
    const topology::GlobalDomain& domain = domainRuntime.getGlobalDomain();
    if (domain.fragments.empty()) {
        std::cerr << "[HeatSystem-Diag] rebuildGlobalThermalDomain FAILED: empty fragments" << std::endl;
        return false;
    }

    releaseGmlsBuffers();

    const VkDeviceSize alignment = vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;
    const auto& models = domainRuntime.getActiveModels();
    std::vector<uint32_t> modelIds;
    modelIds.reserve(models.size());
    for (const auto& [runtimeModelId, model] : models) {
        if (model) modelIds.push_back(runtimeModelId);
    }
    std::sort(modelIds.begin(), modelIds.end());

    std::unordered_map<uint32_t, std::vector<uint32_t>> fragmentIdsByModelId;
    uint32_t unownedFragmentCount = 0;
    for (uint32_t fragmentId = 0; fragmentId < domain.fragments.size(); ++fragmentId) {
        const auto& frag = domain.fragments[fragmentId];
        if (frag.instanceId == topology::InvalidInstanceId) {
            ++unownedFragmentCount;
        } else {
            fragmentIdsByModelId[frag.instanceId].push_back(fragmentId);
        }
    }

    if (fluidRuntime) {
        if (!fluidRuntime->ensureSimulationBuffers(unownedFragmentCount, vulkanDevice)) {
            std::cerr << "[HeatSystem-Diag] rebuildGlobalThermalDomain FAILED: fluidRuntime ensureSimulationBuffers"
                      << " unownedFragmentCount=" << unownedFragmentCount << std::endl;
            return false;
        }
    }

    for (uint32_t runtimeModelId : modelIds) {
        HeatModelRuntime* heatModelPtr = domainRuntime.getModelByRuntimeId(runtimeModelId);
        const auto fragmentIdsIt = fragmentIdsByModelId.find(runtimeModelId);
        if (!heatModelPtr) {
            std::cerr << "[HeatSystem-Diag] rebuildGlobalThermalDomain FAILED: no HeatModelRuntime for id=" << runtimeModelId << std::endl;
            return false;
        }
        if (fragmentIdsIt == fragmentIdsByModelId.end() || fragmentIdsIt->second.empty()) {
            std::cerr << "[HeatSystem-Diag] rebuildGlobalThermalDomain FAILED: no fragments for id=" << runtimeModelId << std::endl;
            return false;
        }
        const auto countIt = simNodeCounts.find(runtimeModelId);
        if (countIt == simNodeCounts.end() || countIt->second != fragmentIdsIt->second.size()) {
            std::cerr << "[HeatSystem-Diag] rebuildGlobalThermalDomain FAILED: simNodeCount mismatch id=" << runtimeModelId << std::endl;
            return false;
        }
    }

    for (uint32_t runtimeModelId : modelIds) {
        HeatModelRuntime* heatModelPtr = domainRuntime.getModelByRuntimeId(runtimeModelId);
        const auto fragmentIdsIt = fragmentIdsByModelId.find(runtimeModelId);
        const auto placementIt = modelLocalToWorldByModelId.find(runtimeModelId);
        if (!heatModelPtr || fragmentIdsIt == fragmentIdsByModelId.end() ||
            placementIt == modelLocalToWorldByModelId.end()) {
            return false;
        }
        const std::array<float, 16>& placement = placementIt->second;
        const glm::mat4 localToWorld(
            placement[0], placement[1], placement[2], placement[3],
            placement[4], placement[5], placement[6], placement[7],
            placement[8], placement[9], placement[10], placement[11],
            placement[12], placement[13], placement[14], placement[15]);

        const auto& fragmentIds = fragmentIdsIt->second;
        std::vector<glm::vec3> fragmentPositions;
        fragmentPositions.reserve(fragmentIds.size());
        for (uint32_t fragmentId : fragmentIds) {
            fragmentPositions.push_back(domain.fragments[fragmentId].centroid);
        }

        const auto& surfacePositions = heatModelPtr->getSurfacePositions();
        const size_t surfaceVertexCount = surfacePositions.size();

        VoronoiNodeIndex nodeIndex;
        nodeIndex.rebuild(fragmentPositions);

        constexpr uint32_t InvalidLocal = UINT32_MAX;
        constexpr uint32_t SupportCount = 32;

        const uint32_t fragmentCount = static_cast<uint32_t>(fragmentIds.size());
        const uint32_t targetSupportCount = std::min(SupportCount, fragmentCount);

        std::vector<uint32_t> globalToLocal(domain.fragments.size(), InvalidLocal);
        for (uint32_t local = 0; local < fragmentCount; ++local) {
            globalToLocal[fragmentIds[local]] = local;
        }

        std::vector<std::vector<uint32_t>> adjacency(fragmentCount);
        for (const auto& face : domain.faces) {
            if (face.fragmentA >= globalToLocal.size() ||
                face.fragmentB >= globalToLocal.size()) {
                continue;
            }

            const uint32_t a = globalToLocal[face.fragmentA];
            const uint32_t b = globalToLocal[face.fragmentB];

            // Both ends must belong to this model
            if (a == InvalidLocal || b == InvalidLocal) {
                continue;
            }

            adjacency[a].push_back(b);
            adjacency[b].push_back(a);
        }

        std::vector<uint32_t> visitStamp(fragmentCount, 0);
        uint32_t currentStamp = 0;

        std::vector<heat::GMLSSurfaceStencil> stencils(surfaceVertexCount);
        std::vector<heat::GMLSSurfaceWeight> valueWeights;
        std::vector<heat::GMLSSurfaceGradientWeight> gradientWeights;
        valueWeights.reserve(surfaceVertexCount * SupportCount);
        gradientWeights.reserve(surfaceVertexCount * SupportCount);

        uint32_t modelIndex = UINT32_MAX;
        for (size_t c = 0; c < modelRuntimeModelIds.size(); ++c) {
            if (modelRuntimeModelIds[c] == runtimeModelId) {
                modelIndex = static_cast<uint32_t>(c);
                break;
            }
        }

        const std::vector<uint32_t>* modelTriangleIndices =
            (modelIndex < modelSurfaceTriangleIndices.size())
                ? &modelSurfaceTriangleIndices[modelIndex]
                : nullptr;

        std::vector<glm::vec3> worldSurfacePositions;
        TriangleHashGrid surfaceGrid;
        if (modelTriangleIndices != nullptr &&
            !modelTriangleIndices->empty() &&
            !surfacePositions.empty()) {
            worldSurfacePositions.resize(surfacePositions.size());
            glm::vec3 boundsMin(std::numeric_limits<float>::max());
            glm::vec3 boundsMax(-std::numeric_limits<float>::max());
            for (size_t i = 0; i < surfacePositions.size(); ++i) {
                const glm::vec3 world = glm::vec3(
                    localToWorld * glm::vec4(surfacePositions[i], 1.0f));
                worldSurfacePositions[i] = world;
                boundsMin = glm::min(boundsMin, world);
                boundsMax = glm::max(boundsMax, world);
            }
            const glm::vec3 extent = boundsMax - boundsMin;
            const float maxExtent = std::max(std::max(extent.x, extent.y), extent.z);
            const float cellSize = std::max(1e-6f, maxExtent / 64.0f);
            const glm::vec3 padding(cellSize);
            surfaceGrid.build(
                worldSurfacePositions,
                *modelTriangleIndices,
                boundsMin - padding,
                boundsMax + padding,
                cellSize);
        }

        uint32_t failedStencilCount = 0;
        uint32_t reportedStencilFailures = 0;

        for (size_t vertexId = 0; vertexId < surfaceVertexCount; ++vertexId) {
            const glm::vec3 worldPos = glm::vec3(localToWorld * glm::vec4(surfacePositions[vertexId], 1.0f));

            std::vector<uint32_t> nearestNodeIds;
            std::vector<float> distanceSquared;

            // KNN is used only to identify which local piece of the model this surface point belongs to
            uint32_t anchor = InvalidLocal;
            uint32_t candidateCount = std::min(64u, fragmentCount);
            uint32_t testedCount = 0;

            while (anchor == InvalidLocal) {
                nodeIndex.findKNearest(
                    worldPos,
                    candidateCount,
                    nearestNodeIds,
                    distanceSquared);

                for (uint32_t k = testedCount; k < nearestNodeIds.size(); ++k) {
                    const uint32_t localFrag = nearestNodeIds[k];
                    const glm::vec3 nodePos = fragmentPositions[localFrag];

                    if (modelTriangleIndices != nullptr &&
                        !worldSurfacePositions.empty() &&
                        surfaceGrid.segmentIntersects(
                            worldPos,
                            nodePos,
                            worldSurfacePositions,
                            *modelTriangleIndices)) {
                        continue;
                    }

                    anchor = localFrag;
                    break;
                }

                if (anchor != InvalidLocal || candidateCount == fragmentCount) {
                    break;
                }

                testedCount = static_cast<uint32_t>(nearestNodeIds.size());
                candidateCount = std::min(candidateCount * 2u, fragmentCount);
            }

            std::vector<uint32_t> supportLocalIds;

            if (anchor != InvalidLocal) {
                using QueueEntry = std::pair<float, uint32_t>;
                std::priority_queue<
                    QueueEntry,
                    std::vector<QueueEntry>,
                    std::greater<QueueEntry>> frontier;

                ++currentStamp;
                if (currentStamp == 0) {
                    std::fill(visitStamp.begin(), visitStamp.end(), 0u);
                    currentStamp = 1;
                }

                const auto pushFragment = [&](uint32_t localFrag) {
                    if (visitStamp[localFrag] == currentStamp) {
                        return;
                    }

                    visitStamp[localFrag] = currentStamp;
                    const glm::vec3 delta = fragmentPositions[localFrag] - worldPos;
                    const float distSq = glm::dot(delta, delta);
                    frontier.emplace(distSq, localFrag);
                };

                pushFragment(anchor);

                while (!frontier.empty() && supportLocalIds.size() < targetSupportCount) {
                    const auto [distSq, localFrag] = frontier.top();
                    frontier.pop();

                    supportLocalIds.push_back(localFrag);

                    for (uint32_t neighbor : adjacency[localFrag]) {
                        pushFragment(neighbor);
                    }
                }
            }

            std::vector<uint32_t> sourceLocalIds;
            std::vector<glm::dvec3> sourcePositions;
            std::vector<double> computedValueWeights;
            std::vector<glm::dvec3> computedGradientWeights;
            double maxDistSq = 0.0;

            const uint32_t count = static_cast<uint32_t>(supportLocalIds.size());
            sourceLocalIds.reserve(count);
            sourcePositions.reserve(count);

            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t localFrag = supportLocalIds[i];
                const glm::vec3 nodePos = fragmentPositions[localFrag];

                sourceLocalIds.push_back(localFrag);
                sourcePositions.push_back(glm::dvec3(nodePos));

                const glm::dvec3 delta = glm::dvec3(nodePos) - glm::dvec3(worldPos);
                maxDistSq = std::max(maxDistSq, glm::dot(delta, delta));
            }

            const double kernelRadius = std::max(std::sqrt(maxDistSq) * 2.0, 1e-6);

            const bool gmlsSuccess = !sourcePositions.empty() &&
                GMLS::computeSurfaceWeights(
                    glm::dvec3(worldPos),
                    sourcePositions,
                    kernelRadius,
                    computedValueWeights,
                    computedGradientWeights);

            bool weightsFinite = true;
            double weightSum = 0.0;
            for (double w : computedValueWeights) {
                if (!std::isfinite(w)) {
                    weightsFinite = false;
                }
                weightSum += w;
            }

            const bool valid = gmlsSuccess &&
                computedValueWeights.size() == count &&
                computedGradientWeights.size() == count &&
                weightsFinite && std::isfinite(weightSum);

            heat::GMLSSurfaceStencil stencil{};
            if (valid) {
                stencil.valueWeightOffset = static_cast<uint32_t>(valueWeights.size());
                stencil.gradientWeightOffset = static_cast<uint32_t>(gradientWeights.size());
                stencil.valueWeightCount = count;
                stencil.gradientWeightCount = count;

                for (uint32_t k = 0; k < count; ++k) {
                    valueWeights.push_back({sourceLocalIds[k], static_cast<float>(computedValueWeights[k])});
                    gradientWeights.push_back({
                        sourceLocalIds[k],
                        static_cast<float>(computedGradientWeights[k].x),
                        static_cast<float>(computedGradientWeights[k].y),
                        static_cast<float>(computedGradientWeights[k].z)
                    });
                }
            } else {
                ++failedStencilCount;
                if (reportedStencilFailures < 10) {
                    ++reportedStencilFailures;
                    std::cerr << "[HeatSystem-Diag] GMLS stencil failed: model=" << runtimeModelId
                              << " vertex=" << vertexId
                              << " position=(" << worldPos.x << ", " << worldPos.y << ", " << worldPos.z << ")"
                              << " support=" << count
                              << " solve=" << (gmlsSuccess ? "ok" : "failed")
                              << " valueWeights=" << computedValueWeights.size()
                              << " gradientWeights=" << computedGradientWeights.size()
                              << " weightSum=" << weightSum
                              << " finite=" << (weightsFinite ? "yes" : "no")
                              << std::endl;
                }
                stencil.valueWeightOffset = 0;
                stencil.gradientWeightOffset = 0;
                stencil.valueWeightCount = 0;
                stencil.gradientWeightCount = 0;
            }
            stencils[vertexId] = stencil;
        }

        if (failedStencilCount > 0) {
            std::cerr << "[HeatSystem-Diag] GMLS stencil failures: model=" << runtimeModelId
                      << " failed=" << failedStencilCount
                      << " total=" << surfaceVertexCount << std::endl;
        }

        VkBuffer stencilBuffer = VK_NULL_HANDLE;
        VkDeviceSize stencilBufferOffset = 0;
        VkBuffer valueWeightBuffer = VK_NULL_HANDLE;
        VkDeviceSize valueWeightBufferOffset = 0;
        VkBuffer gradientWeightBuffer = VK_NULL_HANDLE;
        VkDeviceSize gradientWeightBufferOffset = 0;
        if (uploadDeviceBuffer(memoryAllocator, transferCommandPool, stencils.data(),
                stencils.size() * sizeof(heat::GMLSSurfaceStencil), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                alignment, stencilBuffer, stencilBufferOffset) != VK_SUCCESS ||
            uploadDeviceBuffer(memoryAllocator, transferCommandPool, valueWeights.data(),
                valueWeights.size() * sizeof(heat::GMLSSurfaceWeight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                alignment, valueWeightBuffer, valueWeightBufferOffset) != VK_SUCCESS ||
            uploadDeviceBuffer(memoryAllocator, transferCommandPool, gradientWeights.data(),
                gradientWeights.size() * sizeof(heat::GMLSSurfaceGradientWeight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                alignment, gradientWeightBuffer, gradientWeightBufferOffset) != VK_SUCCESS) {
            std::cerr << "[HeatSystem-Diag] rebuildGlobalThermalDomain FAILED: gmls upload id=" << runtimeModelId << std::endl;
            freeBuffer(memoryAllocator, stencilBuffer, stencilBufferOffset);
            freeBuffer(memoryAllocator, valueWeightBuffer, valueWeightBufferOffset);
            freeBuffer(memoryAllocator, gradientWeightBuffer, gradientWeightBufferOffset);
            return false;
        }

        modelGMLSSurfaceStencilBufferByModelId[runtimeModelId] = stencilBuffer;
        modelGMLSSurfaceStencilBufferOffsetByModelId[runtimeModelId] = stencilBufferOffset;
        modelGMLSSurfaceWeightBufferByModelId[runtimeModelId] = valueWeightBuffer;
        modelGMLSSurfaceWeightBufferOffsetByModelId[runtimeModelId] = valueWeightBufferOffset;
        modelGMLSSurfaceWeightCountByModelId[runtimeModelId] = valueWeights.size();
        modelGMLSSurfaceGradientWeightBufferByModelId[runtimeModelId] = gradientWeightBuffer;
        modelGMLSSurfaceGradientWeightBufferOffsetByModelId[runtimeModelId] = gradientWeightBufferOffset;
        modelGMLSSurfaceGradientWeightCountByModelId[runtimeModelId] = gradientWeights.size();
    }

    return true;
}

void HeatSystem::cleanup() {
    releaseGmlsBuffers();
    if (fluidVolumeStage) {
        fluidVolumeStage->cleanup();
        fluidVolumeStage.reset();
    }
    if (fluidRuntime) {
        fluidRuntime->cleanup();
        fluidRuntime.reset();
    }
    playbackRuntime.cleanup(memoryAllocator);
    domainRuntime.cleanup();
}
