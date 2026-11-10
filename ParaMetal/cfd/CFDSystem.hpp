#pragma once

#include "simgraph/SimNode.hpp"
#include "cfd/CFDRuntime.hpp"
#include "cfd/CFDSolver.hpp"
#include "cfd/CFDSystemPlayback.hpp"
#include "cfd/CFDBoundaryConditions.hpp"
#include "mesh/Polyline.hpp"
#include "voronoi/GlobalDomainTopology.hpp"
#include <cuda_runtime.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <cstdint>
#include <string>

class VulkanDevice;
class CommandPool;
class MemoryAllocator;
class CFDStreamlineStage;

namespace cfd {

class CFDSystem : public SimNode {
public:
    CFDSystem(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& commandPool,
        uint32_t maxFramesInFlight);
    ~CFDSystem() override;

    bool configure(
        const topology::GlobalDomain& domain,
        float kinematicViscosity,
        const glm::vec3& initialVelocity,
        const uint32_t* domainBoundaryTypes = nullptr,
        const glm::vec3* domainBoundaryVelocities = nullptr,
        float fixedTimeStep = 1.0f / 60.0f,
        float targetCourant = 0.5f,
        int pressureIterations = 0,
        float pressureTolerance = 0.0f,
        float simulationDuration = 10.0f);

    void updateParameters(
        float kinematicViscosity,
        const glm::vec3& initialVelocity,
        const uint32_t* domainBoundaryTypes,
        const glm::vec3* domainBoundaryVelocities,
        float fixedTimeStep,
        float targetCourant);

    void setPressureSolverParameters(int maxIterations, float tolerance);

    void update(const SimTickContext& context) override;
    bool isSubmissionReady() const override;
    bool hasPendingSubmission() const override { return pendingSubmission || displayDirty; }
    float computeGpuMs() const override { return solver.getLastStepGpuMs(); }
    uint64_t completedStepCount() const override;
    const std::vector<VkCommandBuffer>& getComputeCommandBuffers() const override { return computeCommandBuffers; }
    void recordComputeCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame) override;
    ComputePass::Synchronization getSynchronization() const override;

    bool hasTimelineMetadata() const override;
    float getSimulationDurationSeconds() const override { return simulationDuration; }
    float getSimulationTimeSeconds() const override { return simulatedTime; }
    uint32_t getTimelineFrameCount() const override;
    uint32_t getRecordedTimelineFrames() const override { return playback.getRecordedFrameCount(); }
    uint32_t getPlaybackAvailableFrame() const override;

    const CFDRuntime& getRuntime() const { return runtime; }
    CFDRuntime& getRuntime() { return runtime; }

    // Line geometry integrated from the displayed velocity frame. The buffer is
    // drawn as-is by the overlay streamline renderer.
    VkBuffer streamlineVertexBuffer() const;
    VkDeviceSize streamlineVertexBufferOffset() const;
    uint32_t streamlineVertexCount() const;

    bool readStreamlinePolylines(
        std::vector<mesh::Polyline>& polylines,
        std::string& outError);

    VkSemaphore publicationReadySemaphore() const override { return cfdReadyVkSemaphore; }
    uint64_t publicationReadyValue() const override { return pendingCfdReadyValue; }
    VkSemaphore publicationReleaseSemaphore() const override { return cfdVulkanReleaseVkSemaphore; }
    uint64_t peekPublicationRelease() const override { return cfdVulkanReleaseCounter + 1; }
    void commitPublicationRelease(uint64_t value) override { cfdVulkanReleaseCounter = value; }
    void setPublicationReleaseWait(uint64_t value) override { cfdReleaseWaitValue = value; }

    bool isInitialized() const { return initialized; }
    void resetSimulationState();

    void cleanup();

    void releaseWaits();

private:
    void onEnteredInactive() override { pendingCfdReadyValue = 0; }
    bool rebuildTopology(const topology::GlobalDomain& domain);
    bool createSyncObjects();
    bool createComputeCommandBuffers(uint32_t maxFramesInFlight);
    void cleanupSyncObjects();
    uint32_t computeTimelineFrameCount() const;
    uint32_t computeHistoryFrameCapacity() const { return computeTimelineFrameCount() + 1u; }
    void updateDisplayFrameState(const SimTickContext& context);

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;
    uint32_t maxFramesInFlight = 2;

    CFDRuntime runtime;
    CFDSolver solver;
    CFDSystemPlayback playback;

    std::unique_ptr<CFDStreamlineStage> streamlineStage;
    VkBuffer seedIdGridBuffer = VK_NULL_HANDLE;
    VkDeviceSize seedIdGridBufferOffset = 0;
    VkBuffer displayVelocityBufferHandle = VK_NULL_HANDLE;
    VkDeviceSize displayVelocityOffset = 0;
    float streamlineStepLength = 0.0f;

    float kinematicViscosity = 1.516e-5f;
    glm::vec3 initialVelocity{0.0f};
    DomainBoundaryConditions boundaryConditions = createDefaultDomainBoundaries();
    float fixedTimeStep = 1.0f / 60.0f;
    float targetCourant = 0.5f;
    int pressureMaxIterations = 0;
    float pressureTolerance = 0.0f;
    float simulationDuration = 10.0f;

    float simulatedTime = 0.0f;
    float physicsAccumulator = 0.0f;
    bool shouldStepPhysics = false;
    bool needsInitialCapture = false;
    uint32_t currentDisplayFrame = 0;
    bool currentReplayFromHistory = false;
    uint32_t lastDisplayFrame = 0;
    bool lastReplayFromHistory = false;
    bool displayDirty = false;

    std::vector<uint32_t> fluidNodeToGlobalFragment;
    std::vector<uint32_t> globalFragmentToFluidNode;

    std::vector<VkCommandBuffer> computeCommandBuffers;

    VkSemaphore cfdReadyVkSemaphore = VK_NULL_HANDLE;
    cudaExternalSemaphore_t cfdReadyCudaSemaphore = nullptr;
    uint64_t cfdReadyCounter = 0;
    uint64_t pendingCfdReadyValue = 0;

    VkSemaphore cfdVulkanReleaseVkSemaphore = VK_NULL_HANDLE;
    cudaExternalSemaphore_t cfdVulkanReleaseCudaSemaphore = nullptr;
    uint64_t cfdVulkanReleaseCounter = 0;
    uint64_t cfdReleaseWaitValue = 0;

    bool initialized = false;
    bool pendingSubmission = false;
    uint32_t processedResetEpoch = 0;
};

} // namespace cfd
