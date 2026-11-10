#include "CFDSystem.hpp"
#include "CFDStreamlineStage.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/CommandBufferManager.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif

#include <iostream>
#include <algorithm>
#include <cmath>

namespace cfd {

static bool createTimelineSemaphore(VkDevice device, VkSemaphore& outSemaphore) {
#ifdef _WIN32
    VkExportSemaphoreCreateInfo exportInfo{};
    exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkSemaphoreTypeCreateInfo typeInfo{};
    typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    typeInfo.pNext = &exportInfo;
    typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    typeInfo.initialValue = 0;

    VkSemaphoreCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    createInfo.pNext = &typeInfo;

    return vkCreateSemaphore(device, &createInfo, nullptr, &outSemaphore) == VK_SUCCESS;
#else
    return false;
#endif
}

static bool importTimelineSemaphoreToCuda(VkDevice device, VkSemaphore vkSem, cudaExternalSemaphore_t& outCudaSem) {
#ifdef _WIN32
    const auto getHandle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
    if (!getHandle) {
        std::cerr << "[CFDSystem] vkGetSemaphoreWin32HandleKHR proc not found\n";
        return false;
    }

    VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.semaphore = vkSem;
    handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    HANDLE handle = nullptr;
    if (getHandle(device, &handleInfo, &handle) != VK_SUCCESS || !handle) {
        std::cerr << "[CFDSystem] vkGetSemaphoreWin32HandleKHR failed\n";
        return false;
    }

    cudaExternalSemaphoreHandleDesc desc{};
    desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    desc.handle.win32.handle = handle;

    const cudaError_t err = cudaImportExternalSemaphore(&outCudaSem, &desc);
    CloseHandle(handle);
    if (err != cudaSuccess) {
        std::cerr << "[CFDSystem] cudaImportExternalSemaphore failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }
    return true;
#else
    return false;
#endif
}

CFDSystem::CFDSystem(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& commandPool,
    uint32_t maxFramesInFlight)
    : vulkanDevice(vulkanDevice),
      memoryAllocator(memoryAllocator),
      commandPool(commandPool),
      maxFramesInFlight(maxFramesInFlight) {
    streamlineStage = std::make_unique<CFDStreamlineStage>(
        vulkanDevice, memoryAllocator, commandPool, maxFramesInFlight);
}

CFDSystem::~CFDSystem() {
    cleanup();
}

bool CFDSystem::configure(
    const topology::GlobalDomain& domain,
    float kinematicViscosity,
    const glm::vec3& initialVelocity,
    const uint32_t* domainBoundaryTypes,
    const glm::vec3* domainBoundaryVelocities,
    float fixedTimeStep,
    float targetCourant,
    int pressureIterations,
    float pressureTolerance,
    float simulationDuration) {
    this->kinematicViscosity = kinematicViscosity;
    this->initialVelocity = initialVelocity;
    this->fixedTimeStep = fixedTimeStep;
    this->targetCourant = targetCourant;
    this->pressureMaxIterations = pressureIterations;
    this->pressureTolerance = pressureTolerance;
    this->simulationDuration = simulationDuration;

    if (domainBoundaryTypes && domainBoundaryVelocities) {
        boundaryConditions = fromFlatArrays(domainBoundaryTypes, domainBoundaryVelocities);
    } else {
        boundaryConditions = createDefaultDomainBoundaries(initialVelocity);
    }

    return rebuildTopology(domain);
}

void CFDSystem::updateParameters(
    float kinematicViscosity,
    const glm::vec3& initialVelocity,
    const uint32_t* domainBoundaryTypes,
    const glm::vec3* domainBoundaryVelocities,
    float fixedTimeStep,
    float targetCourant) {
    const bool velocityChanged =
        glm::length(this->initialVelocity - initialVelocity) > 1e-6f;

    this->kinematicViscosity = kinematicViscosity;
    this->fixedTimeStep = fixedTimeStep;
    this->targetCourant = targetCourant;
    this->initialVelocity = initialVelocity;

    if (domainBoundaryTypes && domainBoundaryVelocities) {
        boundaryConditions = fromFlatArrays(domainBoundaryTypes, domainBoundaryVelocities);
        solver.setBoundaryConditions(boundaryConditions);
    }

    if (velocityChanged) {
        solver.initializeVelocity(initialVelocity);
    }

    if (initialized && !solver.buildMomentumSubcycleGraph(targetCourant)) {
        std::cerr << "[CFDSystem] CFDSolver::buildMomentumSubcycleGraph failed\n";
    }
}

void CFDSystem::setPressureSolverParameters(int maxIterations, float tolerance) {
    pressureMaxIterations = maxIterations;
    pressureTolerance = tolerance;
    if (initialized) {
        solver.setPressureSolverParameters(maxIterations, tolerance);
    }
}

bool CFDSystem::rebuildTopology(const topology::GlobalDomain& domain) {
    fluidNodeToGlobalFragment.clear();
    globalFragmentToFluidNode.assign(domain.fragments.size(), UINT32_MAX);

    for (uint32_t fragIdx = 0; fragIdx < domain.fragments.size(); ++fragIdx) {
        if (domain.fragments[fragIdx].instanceId == topology::InvalidInstanceId) {
            const uint32_t fluidIdx = static_cast<uint32_t>(fluidNodeToGlobalFragment.size());
            fluidNodeToGlobalFragment.push_back(fragIdx);
            globalFragmentToFluidNode[fragIdx] = fluidIdx;
        }
    }

    const uint32_t fluidCount = static_cast<uint32_t>(fluidNodeToGlobalFragment.size());
    if (fluidCount == 0) {
        initialized = false;
        return false;
    }

    if (!solver.initialize(vulkanDevice)) {
        std::cerr << "[CFDSystem] Failed to initialize CFDSolver\n";
        initialized = false;
        return false;
    }

    solver.setBoundaryConditions(boundaryConditions);
    solver.setPressureSolverParameters(pressureMaxIterations, pressureTolerance);

    if (!runtime.initialize(vulkanDevice, solver.getCudaDevice(), fluidCount,
                           static_cast<uint32_t>(domain.faces.size()),
                           static_cast<uint32_t>(domain.boundaryFaces.size()))) {
        std::cerr << "[CFDSystem] Failed to initialize CFDRuntime\n";
        initialized = false;
        return false;
    }

    if (!solver.buildTopology(
            domain.fragments,
            domain.faces,
            domain.boundaryFaces,
            globalFragmentToFluidNode,
            fluidNodeToGlobalFragment)) {
        std::cerr << "[CFDSystem] CFDSolver::buildTopology failed\n";
        initialized = false;
        return false;
    }

    if (!solver.bindOutput(
            runtime.velocityCuda(),
            runtime.faceFluxCuda(),
            static_cast<uint32_t>(domain.faces.size()),
            runtime.boundaryFluxCuda(),
            static_cast<uint32_t>(domain.boundaryFaces.size()))) {
        std::cerr << "[CFDSystem] CFDSolver::bindOutput failed\n";
        initialized = false;
        return false;
    }

    if (!createSyncObjects()) {
        std::cerr << "[CFDSystem] createSyncObjects failed\n";
        initialized = false;
        return false;
    }

    solver.bindCompletionSemaphore(cfdReadyCudaSemaphore);
    solver.bindReleaseSemaphore(cfdVulkanReleaseCudaSemaphore);

    if (!solver.initializeVelocity(initialVelocity)) {
        std::cerr << "[CFDSystem] CFDSolver::initializeVelocity failed\n";
        initialized = false;
        return false;
    }

    if (!solver.buildMomentumSubcycleGraph(targetCourant)) {
        std::cerr << "[CFDSystem] CFDSolver::buildMomentumSubcycleGraph failed\n";
        initialized = false;
        return false;
    }

    seedIdGridBuffer = domain.seedIdGridBuffer;
    seedIdGridBufferOffset = domain.seedIdGridBufferOffset;

    // One streamline seed per VelocityInlet boundary face, pushed just inside
    // the domain along the face's inward normal. The integrator validates each
    // start position against the field lookup and collapses invalid lines.
    std::vector<glm::vec4> streamlineSeeds;
    streamlineSeeds.reserve(domain.boundaryFaces.size());
    const float seedInset = 0.5f * domain.nominalCellSize;
    for (const topology::BoundaryFace& bf : domain.boundaryFaces) {
        if (bf.domainFace >= DomainBoundaryCount ||
            bf.fragment >= domain.fragments.size() ||
            bf.faceIndex >= domain.faces.size() ||
            globalFragmentToFluidNode[bf.fragment] == UINT32_MAX) {
            continue;
        }
        if (boundaryConditions[bf.domainFace].type != BoundaryType::VelocityInlet) {
            continue;
        }
        const topology::Face& face = domain.faces[bf.faceIndex];
        if (!std::isfinite(face.areaA) || !(face.areaA > 0.0)) {
            continue;
        }
        const glm::dvec3 centroid = face.momentA / face.areaA;
        if (!std::isfinite(centroid.x) || !std::isfinite(centroid.y) || !std::isfinite(centroid.z)) {
            continue;
        }
        const glm::vec3 inward = -getDomainBoxNormal(bf.domainFace);
        streamlineSeeds.push_back(glm::vec4(glm::vec3(centroid) + inward * seedInset, 1.0f));
    }

    streamlineStepLength = 0.5f * domain.nominalCellSize;
    if (!streamlineStage) {
        streamlineStage = std::make_unique<CFDStreamlineStage>(
            vulkanDevice, memoryAllocator, commandPool, maxFramesInFlight);
    }
    if (!streamlineStage->initialize(domain, streamlineSeeds, globalFragmentToFluidNode)) {
        std::cerr << "[CFDSystem] Streamline stage configuration failed\n";
    }

    const VkDeviceSize positionAlignment =
        vulkanDevice.getPhysicalDeviceProperties().limits.minStorageBufferOffsetAlignment;

    if (displayVelocityBufferHandle != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, displayVelocityBufferHandle, displayVelocityOffset);
        displayVelocityBufferHandle = VK_NULL_HANDLE;
        displayVelocityOffset = 0;
    }
    auto [displayBuffer, displayOffset] = memoryAllocator.allocate(
        static_cast<VkDeviceSize>(fluidCount) * sizeof(glm::vec4),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        positionAlignment);
    if (displayBuffer != VK_NULL_HANDLE) {
        displayVelocityBufferHandle = displayBuffer;
        displayVelocityOffset = displayOffset;
    } else {
        std::cerr << "[CFDSystem] Failed to allocate display velocity buffer\n";
    }

    if (!playback.initialize(vulkanDevice, memoryAllocator, fluidCount, computeHistoryFrameCapacity())) {
        std::cerr << "[CFDSystem] Failed to allocate velocity history; playback disabled\n";
    }

    simulatedTime = 0.0f;
    physicsAccumulator = 0.0f;
    shouldStepPhysics = false;
    currentDisplayFrame = 0;
    currentReplayFromHistory = false;
    lastDisplayFrame = 0;
    lastReplayFromHistory = false;
    needsInitialCapture = true;
    displayDirty = true;

    if (computeCommandBuffers.empty()) {
        if (!createComputeCommandBuffers(maxFramesInFlight)) {
            std::cerr << "[CFDSystem] createComputeCommandBuffers failed\n";
            initialized = false;
            return false;
        }
    }

    initialized = true;
    return true;
}

bool CFDSystem::createSyncObjects() {
    if (cfdReadyVkSemaphore != VK_NULL_HANDLE && cfdVulkanReleaseVkSemaphore != VK_NULL_HANDLE) {
            return true;
    }

    cleanupSyncObjects();

    VkDevice device = vulkanDevice.getDevice();
    if (!createTimelineSemaphore(device, cfdReadyVkSemaphore)) {
        std::cerr << "[CFDSystem] Failed to create cfdReadyVkSemaphore\n";
        return false;
    }

    if (!importTimelineSemaphoreToCuda(device, cfdReadyVkSemaphore, cfdReadyCudaSemaphore)) {
        std::cerr << "[CFDSystem] Failed to import cfdReadyVkSemaphore to CUDA\n";
        return false;
    }

    if (!createTimelineSemaphore(device, cfdVulkanReleaseVkSemaphore)) {
        std::cerr << "[CFDSystem] Failed to create cfdVulkanReleaseVkSemaphore\n";
        return false;
    }

    if (!importTimelineSemaphoreToCuda(device, cfdVulkanReleaseVkSemaphore, cfdVulkanReleaseCudaSemaphore)) {
        std::cerr << "[CFDSystem] Failed to import cfdVulkanReleaseVkSemaphore to CUDA\n";
        return false;
    }

    cfdReadyCounter = 0;
    pendingCfdReadyValue = 0;
    cfdVulkanReleaseCounter = 0;
    cfdReleaseWaitValue = 0;
    return true;
}

void CFDSystem::cleanupSyncObjects() {
    if (cfdReadyCudaSemaphore) {
        cudaDestroyExternalSemaphore(cfdReadyCudaSemaphore);
        cfdReadyCudaSemaphore = nullptr;
    }
    if (cfdVulkanReleaseCudaSemaphore) {
        cudaDestroyExternalSemaphore(cfdVulkanReleaseCudaSemaphore);
        cfdVulkanReleaseCudaSemaphore = nullptr;
    }
    cfdReleaseWaitValue = 0;

    VkDevice device = vulkanDevice.getDevice();
    if (device != VK_NULL_HANDLE) {
        if (cfdReadyVkSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, cfdReadyVkSemaphore, nullptr);
            cfdReadyVkSemaphore = VK_NULL_HANDLE;
        }
        if (cfdVulkanReleaseVkSemaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, cfdVulkanReleaseVkSemaphore, nullptr);
            cfdVulkanReleaseVkSemaphore = VK_NULL_HANDLE;
        }
    }
}

bool CFDSystem::createComputeCommandBuffers(uint32_t count) {
    computeCommandBuffers.resize(count);

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool.getHandle();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = count;

    if (vkAllocateCommandBuffers(vulkanDevice.getDevice(), &allocInfo, computeCommandBuffers.data()) != VK_SUCCESS) {
        computeCommandBuffers.clear();
        return false;
    }
    return true;
}

void CFDSystem::cleanup() {
    cleanupSyncObjects();

    if (!computeCommandBuffers.empty()) {
        vkFreeCommandBuffers(
            vulkanDevice.getDevice(),
            commandPool.getHandle(),
            static_cast<uint32_t>(computeCommandBuffers.size()),
            computeCommandBuffers.data());
        computeCommandBuffers.clear();
    }

    if (displayVelocityBufferHandle != VK_NULL_HANDLE) {
        freeBuffer(memoryAllocator, displayVelocityBufferHandle, displayVelocityOffset);
        displayVelocityBufferHandle = VK_NULL_HANDLE;
        displayVelocityOffset = 0;
    }
    playback.cleanup();
    if (streamlineStage) {
        streamlineStage->cleanup();
        streamlineStage.reset();
    }
    streamlineStepLength = 0.0f;
    seedIdGridBuffer = VK_NULL_HANDLE;
    seedIdGridBufferOffset = 0;

    solver.cleanup();
    runtime.cleanup();
    initialized = false;
}

void CFDSystem::releaseWaits() {
    if (!initialized || cfdVulkanReleaseVkSemaphore == VK_NULL_HANDLE || cfdReleaseWaitValue == 0) {
        return;
    }

    uint64_t completed = 0;
    if (vkGetSemaphoreCounterValue(vulkanDevice.getDevice(), cfdVulkanReleaseVkSemaphore, &completed) != VK_SUCCESS) {
        return;
    }
    if (completed >= cfdReleaseWaitValue) {
        cfdReleaseWaitValue = 0;
        return;
    }

    if (cfdVulkanReleaseCounter >= cfdReleaseWaitValue) {
        uint64_t readyCompleted = 0;
        const bool queueCanDeliver =
            pendingCfdReadyValue == 0 ||
            (cfdReadyVkSemaphore != VK_NULL_HANDLE &&
             vkGetSemaphoreCounterValue(vulkanDevice.getDevice(), cfdReadyVkSemaphore, &readyCompleted) == VK_SUCCESS &&
             readyCompleted >= pendingCfdReadyValue);
        if (queueCanDeliver) {
            cfdReleaseWaitValue = 0;
            return;
        }
    }

    VkSemaphoreSignalInfo signalInfo{};
    signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signalInfo.semaphore = cfdVulkanReleaseVkSemaphore;
    signalInfo.value = cfdReleaseWaitValue;
    vkSignalSemaphore(vulkanDevice.getDevice(), &signalInfo);
    cfdReleaseWaitValue = 0;
}

void CFDSystem::resetSimulationState() {
    pendingSubmission = false;
    shouldStepPhysics = false;
    simulatedTime = 0.0f;
    physicsAccumulator = 0.0f;
    needsInitialCapture = true;
    displayDirty = true;
    playback.reset();
    if (initialized) {
        solver.setBoundaryConditions(boundaryConditions);
        solver.initializeVelocity(initialVelocity);
    }
}

uint32_t CFDSystem::computeTimelineFrameCount() const {
    if (simulationDuration <= 0.0f || fixedTimeStep <= 0.0f) {
        return 0;
    }
    return std::max(1u, static_cast<uint32_t>(std::ceil(simulationDuration / fixedTimeStep)));
}

bool CFDSystem::hasTimelineMetadata() const {
    return getSimNodeState() != SimNodeState::Inactive &&
        simulationDuration > 0.0f &&
        playback.isValid();
}

uint32_t CFDSystem::getTimelineFrameCount() const {
    return computeTimelineFrameCount();
}

uint32_t CFDSystem::getPlaybackAvailableFrame() const {
    const uint32_t recordedFrames = playback.getRecordedFrameCount();
    if (recordedFrames == 0) {
        return 0;
    }
    return std::min(recordedFrames - 1u, computeTimelineFrameCount());
}

void CFDSystem::updateDisplayFrameState(const SimTickContext& context) {
    currentDisplayFrame = context.currentFrame;
    currentReplayFromHistory = currentDisplayFrame < playback.getRecordedFrameCount();

    const bool replayChanged = currentReplayFromHistory != lastReplayFromHistory;
    const bool frameChanged = currentReplayFromHistory && currentDisplayFrame != lastDisplayFrame;
    if (replayChanged || frameChanged) {
        displayDirty = true;
    }

    lastReplayFromHistory = currentReplayFromHistory;
    lastDisplayFrame = currentDisplayFrame;
}

void CFDSystem::update(const SimTickContext& context) {
    if (context.resetEpoch != processedResetEpoch) {
        processedResetEpoch = context.resetEpoch;
        resetLifecycleState();
        resetSimulationState();
    }

    const bool inputsValid = initialized &&
        runtime.getFluidNodeCount() > 0 &&
        !solver.hasFailed();
    const bool completed = inputsValid &&
        simulationDuration > 0.0f &&
        simulatedTime >= simulationDuration;
    updateLifecycleState(inputsValid, completed, context.playing);

    updateDisplayFrameState(context);

    pendingSubmission = false;

    const bool canAdvance = getSimNodeState() == SimNodeState::Running &&
        context.playing &&
        simulatedTime < simulationDuration;
    if (canAdvance) {
        physicsAccumulator = std::min(physicsAccumulator + context.deltaSeconds, fixedTimeStep);
    } else {
        physicsAccumulator = 0.0f;
    }
    shouldStepPhysics = canAdvance && physicsAccumulator >= fixedTimeStep;

    if (!shouldStepPhysics) {
        return;
    }

    const uint64_t completionVal = cfdReadyCounter + 1;
    const CFDSolver::StepOutcome stepOutcome =
        solver.stepCfd(kinematicViscosity, fixedTimeStep, targetCourant, completionVal, cfdReleaseWaitValue);
    switch (stepOutcome) {
    case CFDSolver::StepOutcome::Enqueued:
        cfdReadyCounter = completionVal;
        pendingCfdReadyValue = completionVal;
        cfdReleaseWaitValue = 0;
        pendingSubmission = true;
        physicsAccumulator = std::max(0.0f, physicsAccumulator - fixedTimeStep);
        simulatedTime = std::min(simulationDuration, simulatedTime + fixedTimeStep);
        break;
    case CFDSolver::StepOutcome::Deferred:
            break;
    case CFDSolver::StepOutcome::Failed:
            pendingCfdReadyValue = 0;
        cfdReleaseWaitValue = 0;
        updateLifecycleState(false, false, context.playing);
        break;
    }
}

bool CFDSystem::isSubmissionReady() const {
    return initialized &&
        runtime.getFluidNodeCount() > 0 &&
        !solver.hasFailed() &&
        !computeCommandBuffers.empty();
}

uint64_t CFDSystem::completedStepCount() const {
    if (cfdReadyVkSemaphore == VK_NULL_HANDLE) {
        return 0;
    }
    uint64_t completedValue = 0;
    if (vkGetSemaphoreCounterValue(vulkanDevice.getDevice(), cfdReadyVkSemaphore, &completedValue) != VK_SUCCESS) {
        return 0;
    }
    return std::min(completedValue, cfdReadyCounter);
}

void CFDSystem::recordComputeCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        return;
    }

    VkBufferMemoryBarrier barriers[3]{};
    uint32_t barrierCount = 0;

    if (runtime.velocityBuffer() != VK_NULL_HANDLE) {
        auto& b = barriers[barrierCount++];
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        b.buffer = runtime.velocityBuffer();
        b.offset = 0;
        b.size = runtime.getFluidNodeCount() * sizeof(glm::vec4);
    }
    if (runtime.faceFluxBuffer() != VK_NULL_HANDLE && runtime.getGlobalFaceCount() > 0) {
        auto& b = barriers[barrierCount++];
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        b.buffer = runtime.faceFluxBuffer();
        b.offset = 0;
        b.size = runtime.getGlobalFaceCount() * sizeof(float);
    }
    if (runtime.boundaryFluxBuffer() != VK_NULL_HANDLE && runtime.getGlobalBoundaryFaceCount() > 0) {
        auto& b = barriers[barrierCount++];
        b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        b.buffer = runtime.boundaryFluxBuffer();
        b.offset = 0;
        b.size = runtime.getGlobalBoundaryFaceCount() * sizeof(float);
    }

    if (barrierCount > 0) {
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            barrierCount, barriers,
            0, nullptr);
    }

    if (hasDispatchableComputeWork()) {
        const VkBuffer liveVelocity = runtime.velocityBuffer();
        const VkDeviceSize velocityBytes =
            static_cast<VkDeviceSize>(runtime.getFluidNodeCount()) * sizeof(glm::vec4);

        if (needsInitialCapture || pendingSubmission) {
            displayDirty = true;
            if (liveVelocity != VK_NULL_HANDLE &&
                playback.isValid() &&
                playback.getRecordedFrameCount() < playback.getFrameCapacity()) {
                playback.recordFrame(commandBuffer, liveVelocity, 0);
            }
            needsInitialCapture = false;
        }

        if (displayDirty && displayVelocityBufferHandle != VK_NULL_HANDLE) {
            VkBuffer sourceBuffer = liveVelocity;
            VkDeviceSize sourceOffset = 0;
            if (currentReplayFromHistory && playback.isValid()) {
                sourceBuffer = playback.getHistoryBuffer();
                sourceOffset = playback.getHistoryBufferOffset() +
                    static_cast<VkDeviceSize>(currentDisplayFrame) * playback.getFrameStride();
            }

            if (sourceBuffer != VK_NULL_HANDLE) {
                VkBufferCopy displayRegion{};
                displayRegion.srcOffset = sourceOffset;
                displayRegion.dstOffset = displayVelocityOffset;
                displayRegion.size = velocityBytes;
                vkCmdCopyBuffer(commandBuffer, sourceBuffer, displayVelocityBufferHandle, 1, &displayRegion);

                VkBufferMemoryBarrier displayBarrier{};
                displayBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                displayBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                displayBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                displayBarrier.buffer = displayVelocityBufferHandle;
                displayBarrier.offset = displayVelocityOffset;
                displayBarrier.size = velocityBytes;
                vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                    0,
                    0, nullptr,
                    1, &displayBarrier,
                    0, nullptr);

                if (streamlineStage &&
                    streamlineStage->isValid() &&
                    seedIdGridBuffer != VK_NULL_HANDLE) {
                    streamlineStage->recordCompute(
                        commandBuffer,
                        currentFrame,
                        seedIdGridBuffer,
                        seedIdGridBufferOffset,
                        displayVelocityBufferHandle,
                        displayVelocityOffset,
                        runtime.getFluidNodeCount(),
                        streamlineStepLength);
                }
                displayDirty = false;
            }
        }
    }

    vkEndCommandBuffer(commandBuffer);
}

VkBuffer CFDSystem::streamlineVertexBuffer() const {
    return streamlineStage ? streamlineStage->vertexBuffer() : VK_NULL_HANDLE;
}

VkDeviceSize CFDSystem::streamlineVertexBufferOffset() const {
    return streamlineStage ? streamlineStage->vertexBufferOffset() : 0;
}

uint32_t CFDSystem::streamlineVertexCount() const {
    return streamlineStage ? streamlineStage->vertexCount() : 0;
}

bool CFDSystem::readStreamlinePolylines(
    std::vector<mesh::Polyline>& polylines,
    std::string& outError) {
    polylines.clear();
    outError.clear();
    if (!streamlineStage || !streamlineStage->isValid()) {
        outError = "CFD streamline stage is unavailable";
        return false;
    }
    const uint32_t vertexCount = streamlineVertexCount();
    if (vertexCount == 0) {
        outError = "CFD streamline buffer is empty";
        return false;
    }

    std::vector<glm::vec4> vertices(vertexCount);
    if (downloadDeviceBuffer(
            memoryAllocator,
            commandPool,
            streamlineStage->vertexBuffer(),
            streamlineStage->vertexBufferOffset(),
            sizeof(glm::vec4) * vertexCount,
            vertices.data()) != VK_SUCCESS) {
        outError = "Failed to read back CFD streamlines";
        return false;
    }

    const uint32_t segmentCount = CFDStreamlineStage::MaxSegmentsPerLine;
    const uint32_t lineCount = vertexCount / (segmentCount * 2u);
    polylines.reserve(lineCount);
    for (uint32_t line = 0; line < lineCount; ++line) {
        const size_t base = static_cast<size_t>(line) * segmentCount * 2u;
        mesh::Polyline polyline;
        polyline.points.push_back(glm::vec3(vertices[base]));
        for (uint32_t segment = 0; segment < segmentCount; ++segment) {
            const glm::vec4& start = vertices[base + 2u * segment];
            const glm::vec4& end = vertices[base + 2u * segment + 1u];
            if (!(start.w > 0.0f) && !(end.w > 0.0f)) break;
            const glm::vec3 position(end);
            if (position != polyline.points.back()) polyline.points.push_back(position);
        }
        if (polyline.points.size() >= 2) polylines.push_back(std::move(polyline));
    }
    if (polylines.empty()) {
        outError = "CFD streamlines have no geometry";
        return false;
    }
    return true;
}

ComputePass::Synchronization CFDSystem::getSynchronization() const {
    ComputePass::Synchronization sync{};
    if (pendingSubmission && pendingCfdReadyValue > 0 && cfdReadyVkSemaphore != VK_NULL_HANDLE) {
        sync.addWait(cfdReadyVkSemaphore, pendingCfdReadyValue, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    }
    return sync;
}

} // namespace cfd
