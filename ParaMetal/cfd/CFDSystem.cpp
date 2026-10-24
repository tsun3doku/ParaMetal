#include "CFDSystem.hpp"
#include "vulkan/VulkanDevice.hpp"
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
    CommandPool& commandPool,
    uint32_t maxFramesInFlight)
    : vulkanDevice(vulkanDevice),
      commandPool(commandPool),
      maxFramesInFlight(maxFramesInFlight) {
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
    float pressureTolerance) {
    this->kinematicViscosity = kinematicViscosity;
    this->initialVelocity = initialVelocity;
    this->fixedTimeStep = fixedTimeStep;
    this->targetCourant = targetCourant;
    this->pressureMaxIterations = pressureIterations;
    this->pressureTolerance = pressureTolerance;

    if (domainBoundaryTypes && domainBoundaryVelocities) {
        for (size_t i = 0; i < 6; ++i) {
            boundaryConditions[i].type = static_cast<BoundaryType>(domainBoundaryTypes[i]);
            boundaryConditions[i].velocity = domainBoundaryVelocities[i];
        }
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
        DomainBoundaryConditions newConditions{};
        for (size_t i = 0; i < 6; ++i) {
            newConditions[i].type = static_cast<BoundaryType>(domainBoundaryTypes[i]);
            newConditions[i].velocity = domainBoundaryVelocities[i];
        }
        boundaryConditions = newConditions;
        solver.setBoundaryConditions(boundaryConditions);
    }

    if (velocityChanged) {
        solver.initializeVelocity(initialVelocity);
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
    // Fresh replacement candidates have no in-flight work, quiescence not required
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
        // Timeline semaphores persist across CFDSystem lifetime to preserve timeline ordering
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
    // Destruction occurs only after global shutdown quiesce, per-system waits unnecessary
    cleanupSyncObjects();

    if (!computeCommandBuffers.empty()) {
        vkFreeCommandBuffers(
            vulkanDevice.getDevice(),
            commandPool.getHandle(),
            static_cast<uint32_t>(computeCommandBuffers.size()),
            computeCommandBuffers.data());
        computeCommandBuffers.clear();
    }

    solver.cleanup();
    runtime.cleanup();
    initialized = false;
}

void CFDSystem::releaseWaits() {
    if (!initialized || cfdVulkanReleaseVkSemaphore == VK_NULL_HANDLE || cfdReleaseWaitValue == 0) {
        return;
    }

    // Semaphore indicates actual completion, release counter only tracks accepted submissions
    uint64_t completed = 0;
    if (vkGetSemaphoreCounterValue(vulkanDevice.getDevice(), cfdVulkanReleaseVkSemaphore, &completed) != VK_SUCCESS) {
        return;
    }
    if (completed >= cfdReleaseWaitValue) {
        cfdReleaseWaitValue = 0;
        return;
    }

    // Queued non-blocked submissions are owned by queue, host signal would double-signal
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
    if (initialized) {
        solver.setBoundaryConditions(boundaryConditions);
        solver.initializeVelocity(initialVelocity);
    }
}

void CFDSystem::update() {
    if (!initialized || !isActive || isPaused || runtime.getFluidNodeCount() == 0) {
        return;
    }

    const uint64_t completionVal = cfdReadyCounter + 1;
    switch (solver.stepCfd(kinematicViscosity, fixedTimeStep, targetCourant, completionVal, cfdReleaseWaitValue)) {
    case CFDSolver::StepOutcome::Enqueued:
        cfdReadyCounter = completionVal;
        pendingCfdReadyValue = completionVal;
        cfdReleaseWaitValue = 0;
        ++completedSteps;
        break;
    case CFDSolver::StepOutcome::Deferred:
        // Previous solve still in flight: no new publication this tick
        break;
    case CFDSolver::StepOutcome::Failed:
        // Terminal failure: unbind consumer publication
        pendingCfdReadyValue = 0;
        cfdReleaseWaitValue = 0;
        break;
    }
}

bool CFDSystem::hasDispatchableComputeWork() const {
    return initialized && isActive && !isPaused && !solver.hasFailed() &&
        (runtime.getFluidNodeCount() > 0) && !computeCommandBuffers.empty();
}

void CFDSystem::recordComputeCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame) {
    (void)currentFrame;
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        return;
    }

    // Pipeline barrier to make external buffer writes visible to subsequent compute/transfer
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

    vkEndCommandBuffer(commandBuffer);
}

ComputePass::Synchronization CFDSystem::getSynchronization() const {
    ComputePass::Synchronization sync{};
    if (pendingCfdReadyValue > 0 && cfdReadyVkSemaphore != VK_NULL_HANDLE) {
        sync.addWait(cfdReadyVkSemaphore, pendingCfdReadyValue, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    }
    return sync;
}

} // namespace cfd
