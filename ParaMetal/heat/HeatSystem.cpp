#include "HeatSystem.hpp"

#include <cstring>

#include "heat/HeatModelRuntime.hpp"
#include "HeatSystemPlayback.hpp"
#include "HeatSystemResources.hpp"
#include "HeatSystemSimStage.hpp"
#include "HeatSystemSurfaceStage.hpp"
#include "HeatSystemStageContext.hpp"
#include "HeatSystemVoronoiStage.hpp"
#include "contact/ContactSystemComputeStage.hpp"
#include "heat/HeatGpuStructs.hpp"
#include "heat/HeatContactRuntime.hpp"
#include "vulkan/CommandBufferManager.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"
#include "voronoi/VoronoiAdapters.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <iostream>

HeatSystem::HeatSystem(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    ModelRegistry& resourceManager,
    HeatSystemResources& resources,
    uint32_t maxFramesInFlight,
    CommandPool& renderCommandPool)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      resourceManager(resourceManager),
      resources(resources),
      renderCommandPool(renderCommandPool),
      runtime(),
      maxFramesInFlight(maxFramesInFlight) {

    HeatSystemStageContext stageContext{
        vulkanDevice,
        memoryAllocator,
        resourceManager,
        renderCommandPool,
        resources
    };

    simStage = std::make_unique<HeatSystemSimStage>(stageContext);
    surfaceStage = std::make_unique<HeatSystemSurfaceStage>(stageContext);
    voronoiStage = std::make_unique<HeatSystemVoronoiStage>(stageContext);
    contactStage = std::make_unique<ContactSystemComputeStage>(vulkanDevice);

    if (!surfaceStage->createDescriptorPool(0) ||  
        !surfaceStage->createDescriptorSetLayout() ||
        !surfaceStage->createPipeline() ||
        !voronoiStage->createDescriptorPool(32) ||
        !voronoiStage->createDescriptorSetLayout() ||
        !voronoiStage->createPipeline() ||
        !contactStage->createDescriptorPool(32) ||
        !contactStage->createDescriptorSetLayout() ||
        !contactStage->createPipeline()) {
        failInitialization("create contact compute resources");
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
    const std::vector<SupportingHalfedge::IntrinsicMesh>& modelIntrinsicMeshes,
    const std::vector<uint32_t>& modelRuntimeModelIds,
    const std::unordered_map<uint32_t, float>& modelTemperatureByRuntimeId,
    const std::unordered_map<uint32_t, uint32_t>& modelBoundaryConditions,
    const std::unordered_map<uint32_t, float>& modelFixedTemperatureValues,
    const std::unordered_map<uint32_t, float>& modelDensity,
    const std::unordered_map<uint32_t, float>& modelSpecificHeat,
    const std::unordered_map<uint32_t, float>& modelConductivity) {
    runtime.setHeatModels(
        modelIntrinsicMeshes,
        modelRuntimeModelIds,
        modelTemperatureByRuntimeId,
        modelBoundaryConditions,
        modelFixedTemperatureValues,
        modelDensity,
        modelSpecificHeat,
        modelConductivity);
    this->modelRuntimeModelIds = modelRuntimeModelIds;
    this->initialTemps = modelTemperatureByRuntimeId;
}


void HeatSystem::setParams(float updatedContactThermalConductance, float updatedSimulationDuration) {
    if (contactThermalConductance != updatedContactThermalConductance) {
        contactThermalConductance = updatedContactThermalConductance;
        heatParamsDirty = true;
    }
    if (simulationDuration != updatedSimulationDuration) {
        simulationDuration = updatedSimulationDuration;
        heatParamsDirty = true;
    }
}

void HeatSystem::setContactCouplings(const std::vector<ContactCoupling>& contactCouplings) {
    this->contactCouplings = contactCouplings;
    contactCouplingsDirty = true;
}

void HeatSystem::clearVoronoiInputs() {
    modelVoronoiNodesByModelId.clear();
    modelVoronoiNodeBufferByModelId.clear();
    modelVoronoiNodeBufferOffsetByModelId.clear();
    modelSimNodeBufferByModelId.clear();
    modelSimNodeBufferOffsetByModelId.clear();
    modelSimGMLSInterfaceBufferByModelId.clear();
    modelSimGMLSInterfaceBufferOffsetByModelId.clear();
    voronoiNodeCounts.clear();
    simNodeCounts.clear();
    simGMLSInterfaceCounts.clear();
    modelVoronoiToSimByModelId.clear();
    modelGMLSSurfaceStencilBufferByModelId.clear();
    modelGMLSSurfaceStencilBufferOffsetByModelId.clear();
    modelGMLSSurfaceWeightBufferByModelId.clear();
    modelGMLSSurfaceWeightBufferOffsetByModelId.clear();
    modelGMLSSurfaceGradientWeightBufferByModelId.clear();
    modelGMLSSurfaceGradientWeightBufferOffsetByModelId.clear();
    modelVoronoiSeedFlagsByModelId.clear();
    modelVoronoiSeedPositionsByModelId.clear();
    voronoiConfigDirty = true;
}

void HeatSystem::addVoronoiModelInput(
    uint32_t runtimeModelId,
    const voronoi::Node* nodes,
    uint32_t voronoiNodeCount,
    VkBuffer voronoiNodeBuffer,
    VkDeviceSize voronoiNodeBufferOffset,
    uint32_t simNodeCount,
    VkBuffer simNodeBuffer,
    VkDeviceSize simNodeBufferOffset,
    VkBuffer simGMLSInterfaceBuffer,
    VkDeviceSize simGMLSInterfaceBufferOffset,
    uint32_t simGMLSInterfaceCount,
    VkBuffer gmlsSurfaceStencilBuffer,
    VkDeviceSize gmlsSurfaceStencilBufferOffset,
    VkBuffer gmlsSurfaceWeightBuffer,
    VkDeviceSize gmlsSurfaceWeightBufferOffset,
    size_t gmlsSurfaceWeightCount,
    VkBuffer gmlsSurfaceGradientWeightBuffer,
    VkDeviceSize gmlsSurfaceGradientWeightBufferOffset,
    size_t gmlsSurfaceGradientWeightCount,
    const std::vector<uint32_t>& seedFlags,
    const std::vector<glm::vec3>& seedPositions,
    const std::vector<uint32_t>& voronoiToSim) {
    if (runtimeModelId == 0 || voronoiNodeCount == 0 || simNodeCount == 0) {
        return;
    }
    
    modelVoronoiNodesByModelId[runtimeModelId] = nodes;
    modelVoronoiNodeBufferByModelId[runtimeModelId] = voronoiNodeBuffer;
    modelVoronoiNodeBufferOffsetByModelId[runtimeModelId] = voronoiNodeBufferOffset;
    modelSimNodeBufferByModelId[runtimeModelId] = simNodeBuffer;
    modelSimNodeBufferOffsetByModelId[runtimeModelId] = simNodeBufferOffset;
    modelSimGMLSInterfaceBufferByModelId[runtimeModelId] = simGMLSInterfaceBuffer;
    modelSimGMLSInterfaceBufferOffsetByModelId[runtimeModelId] = simGMLSInterfaceBufferOffset;
    voronoiNodeCounts[runtimeModelId] = voronoiNodeCount;
    simNodeCounts[runtimeModelId] = simNodeCount;
    simGMLSInterfaceCounts[runtimeModelId] = simGMLSInterfaceCount;
    modelGMLSSurfaceStencilBufferByModelId[runtimeModelId] = gmlsSurfaceStencilBuffer;
    modelGMLSSurfaceStencilBufferOffsetByModelId[runtimeModelId] = gmlsSurfaceStencilBufferOffset;
    modelGMLSSurfaceWeightBufferByModelId[runtimeModelId] = gmlsSurfaceWeightBuffer;
    modelGMLSSurfaceWeightBufferOffsetByModelId[runtimeModelId] = gmlsSurfaceWeightBufferOffset;
    modelGMLSSurfaceWeightCountByModelId[runtimeModelId] = gmlsSurfaceWeightCount;
    modelGMLSSurfaceGradientWeightBufferByModelId[runtimeModelId] = gmlsSurfaceGradientWeightBuffer;
    modelGMLSSurfaceGradientWeightBufferOffsetByModelId[runtimeModelId] = gmlsSurfaceGradientWeightBufferOffset;
    modelGMLSSurfaceGradientWeightCountByModelId[runtimeModelId] = gmlsSurfaceGradientWeightCount;
    modelVoronoiSeedFlagsByModelId[runtimeModelId] = seedFlags;
    modelVoronoiSeedPositionsByModelId[runtimeModelId] = seedPositions;
    modelVoronoiToSimByModelId[runtimeModelId] = voronoiToSim;
    voronoiConfigDirty = true;
}

void HeatSystem::update() {
    static auto lastTime = std::chrono::steady_clock::now();
    const auto currentTime = std::chrono::steady_clock::now();
    float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
    lastTime = currentTime;
    if (deltaTime > (1.0f / 30.0f)) {
        deltaTime = 1.0f / 30.0f;
    }

    if (playbackControls.resetCounter != processedResetCounter) {
        resetHeatState();
        processedResetCounter = playbackControls.resetCounter;
        playbackControls.rewindFrame = 0;
        for (auto& [id, playback] : playbacks) {
            if (playback) playback->reset();
        }
    }

    // Write playback uniform 
    auto* playbackData = simRuntime.getMappedPlaybackData();
    if (playbackData) {
        playbackData->paused = playbackControls.paused ? 1u : 0u;
        playbackData->resetCounter = playbackControls.resetCounter;
        playbackData->rewindFrame = playbackControls.rewindFrame;

        uint32_t recorded = 0;
        for (const auto& [id, pb] : playbacks) {
            if (pb) {
                recorded = std::max(recorded, pb->getRecordedFrames());
            }
        }
        playbackData->recordedFrames = recorded;
        playbackData->maxFrames = static_cast<uint32_t>(simulationDuration * 30.0f * static_cast<float>(NUM_SUBSTEPS));
        playbackData->deltaTime = deltaTime / static_cast<float>(NUM_SUBSTEPS);

        if (!playbackControls.paused && playbackControls.rewindFrame == 0) {
            playbackData->totalTime += deltaTime;
        }
    }

    (void)deltaTime;
}

bool HeatSystem::ensureConfigured() {
    const bool needsHardRebuild = runtime.needsRebuild() || contactCouplingsDirty || voronoiConfigDirty || heatParamsDirty;
    if (!needsHardRebuild) return true;

    vkDeviceWaitIdle(vulkanDevice.getDevice());
    return rebuildRuntimeResources(true);
}

bool HeatSystem::rebuildRuntimeResources(bool forceDescriptorReallocate) {
    const bool modelsReady = runtime.ensureModelBindings(
        vulkanDevice,
        memoryAllocator,
        renderCommandPool);
    if (!modelsReady) {
        return false;
    }

    std::unordered_map<uint32_t, HeatModelRuntime*> activeModels;
    for (const auto& [runtimeModelId, heatModel] : runtime.getActiveModels()) {
        if (heatModel) {
            activeModels[runtimeModelId] = heatModel.get();

            // Cache KDtree for spatial searches 
            const auto seedFlagsIt = modelVoronoiSeedFlagsByModelId.find(runtimeModelId);
            const auto seedPositionsIt = modelVoronoiSeedPositionsByModelId.find(runtimeModelId);
            if (seedFlagsIt != modelVoronoiSeedFlagsByModelId.end() && seedPositionsIt != modelVoronoiSeedPositionsByModelId.end()) {
                heatModel->setStencilKDTree(std::make_unique<StencilKDTree>(seedFlagsIt->second, seedPositionsIt->second));
            }
        }
    }

    const bool heatVoronoiReady = rebuildVoronoiRuntime();

    // Update GMLS surface weights
    for (const auto& [runtimeModelId, heatModel] : runtime.getActiveModels()) {
        if (!heatModel || runtimeModelId == 0) {
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
        if (!heatVoronoiReady) {
            heatModel->setGMLSSurfaceWeights(VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 0, 0, VK_NULL_HANDLE, 0, 0);
            continue;
        }

        heatModel->setGMLSSurfaceWeights(
            stencilIt != modelGMLSSurfaceStencilBufferByModelId.end() ? stencilIt->second : VK_NULL_HANDLE,
            stencilOffsetIt != modelGMLSSurfaceStencilBufferOffsetByModelId.end() ? stencilOffsetIt->second : 0,
            valueWeightIt != modelGMLSSurfaceWeightBufferByModelId.end() ? valueWeightIt->second : VK_NULL_HANDLE,
            valueWeightOffsetIt != modelGMLSSurfaceWeightBufferOffsetByModelId.end() ? valueWeightOffsetIt->second : 0,
            valueWeightCountIt != modelGMLSSurfaceWeightCountByModelId.end() ? valueWeightCountIt->second : 0,
            gradientWeightIt != modelGMLSSurfaceGradientWeightBufferByModelId.end() ? gradientWeightIt->second : VK_NULL_HANDLE,
            gradientWeightOffsetIt != modelGMLSSurfaceGradientWeightBufferOffsetByModelId.end() ? gradientWeightOffsetIt->second : 0,
            gradientWeightCountIt != modelGMLSSurfaceGradientWeightCountByModelId.end() ? gradientWeightCountIt->second : 0);

        auto countIt = simNodeCounts.find(runtimeModelId);
        uint32_t nodeCount = (countIt != simNodeCounts.end()) ? countIt->second : 0;
        heatModel->ensureSimulationBuffers(nodeCount);

        // Initialize or reinitialize playback history for this model
        uint32_t maxFrames = static_cast<uint32_t>(simulationDuration * 30.0f * static_cast<float>(NUM_SUBSTEPS));
        auto& playback = playbacks[runtimeModelId];
        if (!playback) {
            playback = std::make_unique<HeatSystemPlayback>();
        }
        if (!playback->isValid() || playback->getMaxFrames() != maxFrames || playback->getNodeCount() != nodeCount) {
            playback->initialize(vulkanDevice, memoryAllocator, nodeCount, maxFrames);
        }
    }

    if (!heatVoronoiReady) {
        for (auto& r : contactRuntimes) {
            if (r) {
                r->cleanup(memoryAllocator);
            }
        }
        contactRuntimes.clear();
        resources.hasContact = false;
        simRuntime.cleanup(memoryAllocator);

        return false;
    }

    uint32_t numModels = static_cast<uint32_t>(modelRuntimeModelIds.size());

    if (resources.surfaceDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vulkanDevice.getDevice(), resources.surfaceDescriptorPool, nullptr);
        resources.surfaceDescriptorPool = VK_NULL_HANDLE;
    }
    if (!surfaceStage->createDescriptorPool(numModels)) {
        return false;
    }

    // Recreate Voronoi descriptor pool
    if (resources.voronoiDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vulkanDevice.getDevice(), resources.voronoiDescriptorPool, nullptr);
        resources.voronoiDescriptorPool = VK_NULL_HANDLE;
    }
    if (!voronoiStage->createDescriptorPool(numModels)) {
        return false;
    }

    // Initialize sim runtime 
    bool simInitResult = simRuntime.initialize(vulkanDevice, memoryAllocator);

    const bool simReady = simInitResult && voronoiStage;
    if (!simReady) return false;

    for (const auto& [runtimeModelId, heatModel] : runtime.getActiveModels()) {
        if (!heatModel) {
            continue;
        }

        const auto countIt = simNodeCounts.find(runtimeModelId);
        const auto nodeBufferIt = modelSimNodeBufferByModelId.find(runtimeModelId);
        const auto nodeBufferOffsetIt = modelSimNodeBufferOffsetByModelId.find(runtimeModelId);
        const auto gmlsInterfaceIt = modelSimGMLSInterfaceBufferByModelId.find(runtimeModelId);
        const auto gmlsInterfaceOffsetIt = modelSimGMLSInterfaceBufferOffsetByModelId.find(runtimeModelId);
        const auto gmlsInterfaceCountIt = simGMLSInterfaceCounts.find(runtimeModelId);
        const auto voronoiToSimIt = modelVoronoiToSimByModelId.find(runtimeModelId);
        if (countIt == simNodeCounts.end() ||
            nodeBufferIt == modelSimNodeBufferByModelId.end() ||
            nodeBufferOffsetIt == modelSimNodeBufferOffsetByModelId.end() ||
            gmlsInterfaceIt == modelSimGMLSInterfaceBufferByModelId.end() ||
            gmlsInterfaceOffsetIt == modelSimGMLSInterfaceBufferOffsetByModelId.end() ||
            gmlsInterfaceCountIt == simGMLSInterfaceCounts.end()) {
            continue;
        }

        heatModel->setSimResources(
            nodeBufferIt->second,
            nodeBufferOffsetIt->second,
            countIt->second,
            gmlsInterfaceIt->second,
            gmlsInterfaceOffsetIt->second,
            gmlsInterfaceCountIt->second);

        if (voronoiToSimIt != modelVoronoiToSimByModelId.end()) {
            heatModel->setVoronoiToSimNodeId(voronoiToSimIt->second);
        }
    }

    if (contactStage) {
        const bool rebuildContacts = contactCouplingsDirty || heatParamsDirty || forceDescriptorReallocate;
        if (rebuildContacts) {
            for (auto& r : contactRuntimes) {
                if (r) {
                    r->cleanup(memoryAllocator);
                }
            }
            contactRuntimes.clear();

            for (const ContactCoupling& coupling : contactCouplings) {
                if (!coupling.isValid()) {
                    continue;
                }

                auto itA = activeModels.find(coupling.modelARuntimeModelId);
                auto itB = activeModels.find(coupling.modelBRuntimeModelId);
                if (itA == activeModels.end() || itB == activeModels.end() || !itA->second || !itB->second) {
                    continue;
                }

                auto runtimePtr = std::make_unique<HeatContactRuntime>();
                if (!runtimePtr->build(
                        vulkanDevice,
                        memoryAllocator,
                        coupling,
                        *itA->second,
                        *itB->second,
                        contactThermalConductance)) {
                    runtimePtr->cleanup(memoryAllocator);
                    continue;
                }

                if (!runtimePtr->createDescriptorSets(*contactStage, *itA->second, *itB->second)) {
                    runtimePtr->cleanup(memoryAllocator);
                    continue;
                }

                contactRuntimes.push_back(std::move(runtimePtr));
            }

            contactCouplingsDirty = false;
        }
    }

    resources.hasContact = !contactRuntimes.empty();

    // Update descriptors on each HeatModelRuntime using its own buffers
    for (const auto& [runtimeModelId, heatModel] : runtime.getActiveModels()) {
        if (!heatModel || heatModel->getIntrinsicVertexCount() == 0) continue;

        auto itCount = simNodeCounts.find(runtimeModelId);
        uint32_t nodeCount = (itCount != simNodeCounts.end()) ? itCount->second : 0;
        if (nodeCount == 0) continue;

        const auto nodeBufferIt = modelSimNodeBufferByModelId.find(runtimeModelId);
        const auto nodeBufferOffsetIt = modelSimNodeBufferOffsetByModelId.find(runtimeModelId);
        const auto gmlsInterfaceIt = modelSimGMLSInterfaceBufferByModelId.find(runtimeModelId);
        const auto gmlsInterfaceOffsetIt = modelSimGMLSInterfaceBufferOffsetByModelId.find(runtimeModelId);
        const auto gmlsInterfaceCountIt = simGMLSInterfaceCounts.find(runtimeModelId);
        const auto voronoiToSimIt = modelVoronoiToSimByModelId.find(runtimeModelId);
        if (nodeBufferIt != modelSimNodeBufferByModelId.end() &&
            nodeBufferOffsetIt != modelSimNodeBufferOffsetByModelId.end() &&
            gmlsInterfaceIt != modelSimGMLSInterfaceBufferByModelId.end() &&
            gmlsInterfaceOffsetIt != modelSimGMLSInterfaceBufferOffsetByModelId.end() &&
            gmlsInterfaceCountIt != simGMLSInterfaceCounts.end()) {
            heatModel->setSimResources(
                nodeBufferIt->second, nodeBufferOffsetIt->second, nodeCount,
                gmlsInterfaceIt->second, gmlsInterfaceOffsetIt->second,
                gmlsInterfaceCountIt->second);
            if (voronoiToSimIt != modelVoronoiToSimByModelId.end()) {
                heatModel->setVoronoiToSimNodeId(voronoiToSimIt->second);
            }
        }

        auto playbackIt = playbacks.find(runtimeModelId);
        VkBuffer historyBuf = VK_NULL_HANDLE;
        VkDeviceSize historyOff = 0;
        if (playbackIt != playbacks.end() && playbackIt->second) {
            historyBuf = playbackIt->second->getHistoryBuffer();
            historyOff = playbackIt->second->getHistoryBufferOffset();
        }
        heatModel->setHistoryBuffer(historyBuf, historyOff);
        heatModel->updateAllDescriptors(
            resources.surfaceDescriptorSetLayout,
            resources.surfaceGradientDescriptorSetLayout,
            resources.surfaceDescriptorPool,
            resources.voronoiDescriptorSetLayout,
            resources.voronoiDescriptorPool,
            simRuntime.getPlaybackBuffer(),
            simRuntime.getPlaybackBufferOffset(),
            historyBuf,
            historyOff,
            true); 
    }

    voronoiConfigDirty = false;
    heatParamsDirty = false;
    return true;
}

void HeatSystem::setActive(bool active) {
    isActive = active;
}

void HeatSystem::setPlaybackState(bool paused, uint32_t resetCounter) {
    playbackControls.paused = paused;
    playbackControls.resetCounter = resetCounter;
}

void HeatSystem::resetHeatState() {
    simRuntime.reset();
    resetVoronoiTemperatures();
}

void HeatSystem::resetVoronoiTemperatures() {
    for (const auto& [runtimeModelId, heatModel] : runtime.getActiveModels()) {
        if (!heatModel) continue;

        auto tempIt = initialTemps.find(runtimeModelId);
        if (tempIt == initialTemps.end()) continue;

        float temperature = tempIt->second;
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

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        if (cmd != VK_NULL_HANDLE) {
            // Copy to tempBufferA
            VkBufferCopy regionA{};
            regionA.srcOffset = stagingOffset;
            regionA.dstOffset = heatModel->getTempBufferAOffset();
            regionA.size = bufferSize;
            vkCmdCopyBuffer(cmd, stagingBuffer, heatModel->getTempBufferA(), 1, &regionA);

            // Copy to tempBufferB
            VkBufferCopy regionB{};
            regionB.srcOffset = stagingOffset;
            regionB.dstOffset = heatModel->getTempBufferBOffset();
            regionB.size = bufferSize;
            vkCmdCopyBuffer(cmd, stagingBuffer, heatModel->getTempBufferB(), 1, &regionB);

            renderCommandPool.endCommands(cmd);
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
        const auto countIt = voronoiNodeCounts.find(runtimeModelId);
        const auto bufferIt = modelVoronoiNodeBufferByModelId.find(runtimeModelId);
        const auto simNodeBufferIt = modelSimNodeBufferByModelId.find(runtimeModelId);
        const auto simGmlsIt = modelSimGMLSInterfaceBufferByModelId.find(runtimeModelId);
        const auto gmlsStencilIt = modelGMLSSurfaceStencilBufferByModelId.find(runtimeModelId);
        const auto gmlsWeightIt = modelGMLSSurfaceWeightBufferByModelId.find(runtimeModelId);
        const auto gmlsGradientIt = modelGMLSSurfaceGradientWeightBufferByModelId.find(runtimeModelId);
        if (countIt == voronoiNodeCounts.end() ||
            countIt->second == 0 ||
            bufferIt == modelVoronoiNodeBufferByModelId.end() ||
            !bufferIt->second ||
            simNodeBufferIt == modelSimNodeBufferByModelId.end() ||
            !simNodeBufferIt->second ||
            simGmlsIt == modelSimGMLSInterfaceBufferByModelId.end() ||
            !simGmlsIt->second ||
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
        voronoiStage) {
        simStage->recordComputeCommands(
            commandBuffer,
            currentFrame,
            simRuntime,
            runtime.getActiveModels(),
            *voronoiStage,
            *surfaceStage,
            *contactStage,
            contactRuntimes,
            MAX_NODE_NEIGHBORS,
            NUM_SUBSTEPS);

        // Snapshot final temperatures into playback history (only when running forward)
        if (!playbackControls.paused && playbackControls.rewindFrame == 0) {
            const bool finalWritesBufferB = voronoiStage->finalSubstepWritesBufferB(NUM_SUBSTEPS);

            std::vector<VkBufferMemoryBarrier> snapshotBarriers;
            for (const auto& [modelId, heatModel] : runtime.getActiveModels()) {
                if (!heatModel || heatModel->getSimNodeCount() == 0) continue;
                auto pbIt = playbacks.find(modelId);
                if (pbIt == playbacks.end() || !pbIt->second) continue;

                VkBuffer finalBuf = finalWritesBufferB ? heatModel->getTempBufferB() : heatModel->getTempBufferA();
                VkDeviceSize finalOff = finalWritesBufferB ? heatModel->getTempBufferBOffset() : heatModel->getTempBufferAOffset();

                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                barrier.buffer = finalBuf;
                barrier.offset = finalOff;
                barrier.size = heatModel->getSimNodeCount() * sizeof(float);
                snapshotBarriers.push_back(barrier);
            }
            if (!snapshotBarriers.empty()) {
                vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0,
                    0, nullptr,
                    static_cast<uint32_t>(snapshotBarriers.size()), snapshotBarriers.data(),
                    0, nullptr);
            }

            for (const auto& [modelId, heatModel] : runtime.getActiveModels()) {
                if (!heatModel || heatModel->getSimNodeCount() == 0) continue;
                auto pbIt = playbacks.find(modelId);
                if (pbIt == playbacks.end() || !pbIt->second) continue;

                VkBuffer finalBuf = finalWritesBufferB ? heatModel->getTempBufferB() : heatModel->getTempBufferA();
                VkDeviceSize finalOff = finalWritesBufferB ? heatModel->getTempBufferBOffset() : heatModel->getTempBufferAOffset();
                pbIt->second->recordFrame(commandBuffer, finalBuf, finalOff);
            }
        }
    }

    vkEndCommandBuffer(commandBuffer);
}

void HeatSystem::cleanupResources() {
    if (resources.surfacePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), resources.surfacePipeline, nullptr);
        resources.surfacePipeline = VK_NULL_HANDLE;
    }
    if (resources.surfacePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), resources.surfacePipelineLayout, nullptr);
        resources.surfacePipelineLayout = VK_NULL_HANDLE;
    }
    if (resources.surfaceDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vulkanDevice.getDevice(), resources.surfaceDescriptorPool, nullptr);
        resources.surfaceDescriptorPool = VK_NULL_HANDLE;
    }
    if (resources.surfaceDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vulkanDevice.getDevice(), resources.surfaceDescriptorSetLayout, nullptr);
        resources.surfaceDescriptorSetLayout = VK_NULL_HANDLE;
    }
    if (resources.voronoiPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(vulkanDevice.getDevice(), resources.voronoiPipeline, nullptr);
        resources.voronoiPipeline = VK_NULL_HANDLE;
    }
    if (resources.voronoiPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(vulkanDevice.getDevice(), resources.voronoiPipelineLayout, nullptr);
        resources.voronoiPipelineLayout = VK_NULL_HANDLE;
    }
    if (resources.voronoiDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(vulkanDevice.getDevice(), resources.voronoiDescriptorPool, nullptr);
        resources.voronoiDescriptorPool = VK_NULL_HANDLE;
    }
    if (resources.voronoiDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(vulkanDevice.getDevice(), resources.voronoiDescriptorSetLayout, nullptr);
        resources.voronoiDescriptorSetLayout = VK_NULL_HANDLE;
    }

    resources.hasContact = false;
}
void HeatSystem::cleanup() {
    cleanupResources();
    simRuntime.cleanup(memoryAllocator);
    runtime.cleanup();
    for (auto& [id, playback] : playbacks) {
        if (playback) playback->cleanup();
    }
    playbacks.clear();
}

bool HeatSystem::initializeVoronoiMaterialNodes() {
    if (modelVoronoiNodeBufferByModelId.empty()) {
        return false;
    }

    for (const auto& [id, voronoiNodeBuffer] : modelVoronoiNodeBufferByModelId) {
        if (!voronoiNodeBuffer) continue;
        HeatModelRuntime* heatModelPtr = runtime.getModelByRuntimeId(id);
        if (!heatModelPtr) continue;

        auto countIt = simNodeCounts.find(id);
        auto nodeBufferIt = modelSimNodeBufferByModelId.find(id);
        auto offsetIt = modelSimNodeBufferOffsetByModelId.find(id);
        if (countIt == simNodeCounts.end() ||
            nodeBufferIt == modelSimNodeBufferByModelId.end() ||
            offsetIt == modelSimNodeBufferOffsetByModelId.end()) continue;

        float d = heatModelPtr->getDensity();
        float s = heatModelPtr->getSpecificHeat();
        float c = heatModelPtr->getConductivity();
        uint32_t b = heatModelPtr->getBoundaryCondition();
        float f = heatModelPtr->getFixedTemperatureValue();

        uint32_t modelNodeCount = countIt->second;
        VkBuffer simNodeBuffer = nodeBufferIt->second;
        VkDeviceSize nodeBufferSize = modelNodeCount * sizeof(voronoi::Node);
        VkDeviceSize nodeBufferOffset = offsetIt->second;

        // Read back from GPU via staging buffer
        VkBuffer stagingBuf = VK_NULL_HANDLE;
        VkDeviceSize stagingOff = 0;
        void* stagingMapped = nullptr;
        if (createStagingBuffer(memoryAllocator, nodeBufferSize, stagingBuf, stagingOff, &stagingMapped) != VK_SUCCESS || !stagingMapped) {
            continue;
        }

        VkCommandBuffer cmd = renderCommandPool.beginCommands();
        VkBufferCopy region{nodeBufferOffset, stagingOff, nodeBufferSize};
        vkCmdCopyBuffer(cmd, simNodeBuffer, stagingBuf, 1, &region);
        renderCommandPool.endCommands(cmd);

        const voronoi::Node* nodes = static_cast<const voronoi::Node*>(stagingMapped);

        std::vector<heat::MaterialNode> materialNodes(modelNodeCount);
        for (uint32_t localNodeIndex = 0; localNodeIndex < modelNodeCount; ++localNodeIndex) {
            heat::MaterialNode& materialNode = materialNodes[localNodeIndex];
            materialNode.density = d;
            materialNode.specificHeat = s;
            materialNode.conductivity = c;
            materialNode.boundaryCondition = b;
            materialNode.fixedTemperatureValue = f;

            const float volume = std::abs(nodes[localNodeIndex].volume);
            materialNode.thermalMass = d * s * volume;
            if (materialNode.thermalMass > 1e-20f) materialNode.conductivityPerMass = c / materialNode.thermalMass;
        }

        memoryAllocator.free(stagingBuf, stagingOff);

        heatModelPtr->createMaterialBuffer(materialNodes);
    }

    return true;
}

bool HeatSystem::rebuildVoronoiRuntime() {
    if (modelRuntimeModelIds.empty() || modelVoronoiNodeBufferByModelId.empty()) return false;
    return initializeVoronoiMaterialNodes();
}
