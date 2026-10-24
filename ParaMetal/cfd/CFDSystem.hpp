#pragma once

#include "simgraph/SimNode.hpp"
#include "cfd/CFDRuntime.hpp"
#include "cfd/CFDSolver.hpp"
#include "cfd/CFDBoundaryConditions.hpp"
#include "voronoi/GlobalDomainTopology.hpp"
#include <cuda_runtime.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>
#include <cstdint>

class VulkanDevice;
class CommandPool;

namespace cfd {

class CFDSystem : public SimNode {
public:
    CFDSystem(
        VulkanDevice& vulkanDevice,
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
        float pressureTolerance = 0.0f);

    void updateParameters(
        float kinematicViscosity,
        const glm::vec3& initialVelocity,
        const uint32_t* domainBoundaryTypes,
        const glm::vec3* domainBoundaryVelocities,
        float fixedTimeStep,
        float targetCourant);

    void setPressureSolverParameters(int maxIterations, float tolerance);

    void update() override;
    bool hasDispatchableComputeWork() const override;
    float computeGpuMs() const override { return solver.getLastStepGpuMs(); }
    uint64_t completedStepCount() const override { return completedSteps; }
    const std::vector<VkCommandBuffer>& getComputeCommandBuffers() const override { return computeCommandBuffers; }
    void recordComputeCommands(VkCommandBuffer commandBuffer, uint32_t currentFrame) override;
    ComputePass::Synchronization getSynchronization() const override;

    const CFDRuntime& getRuntime() const { return runtime; }
    CFDRuntime& getRuntime() { return runtime; }

    // SimNode timeline publication interface
    VkSemaphore publicationReadySemaphore() const override { return cfdReadyVkSemaphore; }
    uint64_t publicationReadyValue() const override { return pendingCfdReadyValue; }
    VkSemaphore publicationReleaseSemaphore() const override { return cfdVulkanReleaseVkSemaphore; }
    uint64_t peekPublicationRelease() const override { return cfdVulkanReleaseCounter + 1; }
    void commitPublicationRelease(uint64_t value) override { cfdVulkanReleaseCounter = value; }
    void setPublicationReleaseWait(uint64_t value) override { cfdReleaseWaitValue = value; }

    bool isInitialized() const { return initialized; }
    void setActive(bool active) { isActive = active; }

    void setPaused(bool paused) { isPaused = paused; }
    void resetSimulationState();

    void cleanup();

    // Host-signals pending release wait when queue cannot deliver to prevent deadlock
    void releaseWaits();

private:
    bool rebuildTopology(const topology::GlobalDomain& domain);
    bool createSyncObjects();
    bool createComputeCommandBuffers(uint32_t maxFramesInFlight);
    void cleanupSyncObjects();

    VulkanDevice& vulkanDevice;
    CommandPool& commandPool;
    uint32_t maxFramesInFlight = 2;

    CFDRuntime runtime;
    CFDSolver solver;

    float kinematicViscosity = 1.516e-5f;
    glm::vec3 initialVelocity{0.0f};
    DomainBoundaryConditions boundaryConditions = createDefaultDomainBoundaries();
    float fixedTimeStep = 1.0f / 60.0f;
    float targetCourant = 0.5f;
    int pressureMaxIterations = 0;
    float pressureTolerance = 0.0f;

    std::vector<uint32_t> fluidNodeToGlobalFragment;
    std::vector<uint32_t> globalFragmentToFluidNode;

    std::vector<VkCommandBuffer> computeCommandBuffers;
    uint64_t completedSteps = 0;

    // Timeline synchronization
    VkSemaphore cfdReadyVkSemaphore = VK_NULL_HANDLE;
    cudaExternalSemaphore_t cfdReadyCudaSemaphore = nullptr;
    uint64_t cfdReadyCounter = 0;
    uint64_t pendingCfdReadyValue = 0;

    VkSemaphore cfdVulkanReleaseVkSemaphore = VK_NULL_HANDLE;
    cudaExternalSemaphore_t cfdVulkanReleaseCudaSemaphore = nullptr;
    uint64_t cfdVulkanReleaseCounter = 0;
    uint64_t cfdReleaseWaitValue = 0;

    bool initialized = false;
    bool isActive = true;
    bool isPaused = true;
};

} // namespace cfd
