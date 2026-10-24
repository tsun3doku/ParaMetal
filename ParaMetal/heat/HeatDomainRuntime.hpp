#pragma once

#include "framegraph/ComputePass.hpp"
#include "voronoi/GlobalDomainTopology.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <vulkan/vulkan.h>

#include "heat/HeatGlobalSolver.hpp"
#include "heat/HeatSystemPresets.hpp"
#include "heat/HeatBoundaryConditions.hpp"
#include "vem/VEMAgglomerator.hpp"
#include "cuda/CudaExternalBuffer.hpp"
#include "vulkan/VulkanExternalBuffer.hpp"

class HeatModelRuntime;
class HeatFluidRuntime;
class VulkanDevice;

class HeatDomainRuntime {
public:
    static constexpr uint32_t InvalidSolverNode = 0xFFFFFFFFu;

    ~HeatDomainRuntime();

    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& getActiveModels() const { return activeModels; }
    HeatModelRuntime* getModelByRuntimeId(uint32_t runtimeModelId) const;
    void removeModelsNotIn(const std::unordered_set<uint32_t>& runtimeModelIds);
    void setModel(uint32_t runtimeModelId, std::unique_ptr<HeatModelRuntime> model);
    void removeModel(uint32_t runtimeModelId);

    void cleanup();

    const topology::GlobalDomain& getGlobalDomain() const { return globalDomain; }
    bool isGlobalDomain() const { return globalDomain.isValid(); }
    void setGlobalDomain(topology::GlobalDomain domain);
    void clearGlobalDomain();

    bool hasGlobalSolver() const { return globalSolver != nullptr; }
    bool globalTopologyDirty() const { return globalTopologyDirtyFlag; }
    bool globalValuesDirty() const { return globalValuesDirtyFlag; }
    bool globalAdvectionDirty() const { return globalAdvectionDirtyFlag; }
    void markGlobalTopologyDirty() { globalTopologyDirtyFlag = true; }
    void markGlobalValuesDirty() { globalValuesDirtyFlag = true; }
    void markAdvectionDirty() { globalAdvectionDirtyFlag = true; }

    void setDomainBoundaryConditions(
        const uint32_t types[6],
        const float temperaturesC[6],
        const float heatFluxes[6],
        const float heatTransferCoefficients[6]) {
        domainBoundaries = heat::fromFlatArrays(types, temperaturesC, heatFluxes, heatTransferCoefficients);
    }

    void setDomainBoundaryConditions(const heat::DomainBoundaryConditions& boundaries) {
        domainBoundaries = boundaries;
    }

    const heat::DomainBoundaryConditions& getDomainBoundaryConditions() const {
        return domainBoundaries;
    }

    struct AdvectionInput {
        bool enabled = false;
        // Canonical per-face fluxes on shared fluid faces
        VkBuffer faceFluxBuffer = VK_NULL_HANDLE;
        VkDeviceSize faceFluxBufferSize = 0;
        VkBuffer boundaryFluxBuffer = VK_NULL_HANDLE;
        VkDeviceSize boundaryFluxBufferSize = 0;
        uint32_t faceCount = 0;
        uint32_t boundaryFaceCount = 0;
    };

    void setAdvectionInput(const AdvectionInput& input) {
        advectionInput = input;
        globalAdvectionDirtyFlag = true;
    }

    void clearAdvectionInput() {
        advectionInput = {};
        globalAdvectionDirtyFlag = true;
    }

    // Update advection connection availability
    void setAdvectionConnected(bool connected) {
        if (advectionConnected == connected) {
            return;
        }
        advectionConnected = connected;
        globalAdvectionDirtyFlag = true;
    }

    bool advectionStagingCopyScheduled() const { return stagingCopyScheduled; }
    void resetAdvectionStagingCopySchedule() { stagingCopyScheduled = false; }

    bool rebuildGlobalSolver(
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& models,
        HeatFluidRuntime* fluidRuntime,
        VulkanDevice& vulkanDevice,
        float fixedTimeStep);
    bool updateGlobalValues(
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& models,
        HeatFluidRuntime* fluidRuntime,
        float fixedTimeStep);
    bool updateGlobalAdvection(HeatFluidRuntime* fluidRuntime, float fixedTimeStep);
    bool launchGlobalSolve(bool sourceIsA);
    bool pollGlobalSolve(bool& completedDestinationIsA);
    bool isGlobalSolveInFlight() const { return globalSolveInFlight; }
    void prepareGlobalRenderSynchronization(bool newSolveCompleted);
    // Records current frame release value for the active read buffer to sync next solve overwrite
    void noteTemperatureReader(bool currentIsA);
    void waitGlobalSolveIdle();
    // Release pending solve wait on host during terminal recovery
    void releasePendingSolveWaitOnHost();
    bool hasSolveFailed() const { return globalSolver ? globalSolver->hasSolveFailed() : false; }

    void setGlobalSolverParameters(uint32_t maxIterations, float convergenceTolerance) {
        solverMaxIterations = maxIterations;
        solverConvergenceTolerance = convergenceTolerance;
        if (globalSolver) globalSolver->setSolverParameters(maxIterations, convergenceTolerance);
    }

    float getLastSolveMs() const {
        return globalSolver ? globalSolver->getLastSolveMs() : 0.0f;
    }

    ComputePass::Synchronization getGlobalSynchronization() const { return globalSynchronization; }
    void clearGlobalSynchronization() { globalSynchronization = ComputePass::Synchronization{}; }

    const std::vector<uint32_t>& getGlobalFragmentToFluidNode() const { return globalFragmentToFluidNode; }

    VkBuffer getAdvectionFaceFluxBuffer() const { return advectionFaceFlux.getBuffer(); }
    VkBuffer getAdvectionBoundaryFluxBuffer() const { return advectionBoundaryFlux.getBuffer(); }
    const AdvectionInput& getAdvectionInput() const { return advectionInput; }

private:
    bool extractVEMInputs(
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& models,
        HeatFluidRuntime* fluidRuntime,
        float fixedTimeStep,
        std::vector<float>& cellConductivity,
        std::vector<float>& cellCapacityOverDt,
        std::vector<float>& cellSourceAdditions,
        std::vector<float>& fixedCellTemperatures,
        std::array<vem::BoundaryCondition, 6>& vemBcs) const;

    std::vector<vem::DirichletTrace> buildDirichletTraces(
        const vem::VEMMesh& vemMesh,
        const std::vector<float>& fixedCellTemperatures) const;

    std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>> activeModels;

    topology::GlobalDomain globalDomain;
    // Cached VEM mesh from last topology build
    vem::VEMMesh vemMeshCache;
    vem::VEMCellRemap vemCellRemapCache;
    heat::DomainBoundaryConditions domainBoundaries = heat::createDefaultDomainBoundaries(HeatSimDefaults::ambientTemperatureC);

    bool globalTopologyDirtyFlag = false;
    bool globalValuesDirtyFlag = false;
    bool globalAdvectionDirtyFlag = false;

    std::unique_ptr<HeatGlobalSolver> globalSolver;
    uint32_t solverMaxIterations = 0;
    float solverConvergenceTolerance = 0.0f;
    ComputePass::Synchronization globalSynchronization;
    bool globalSolveInFlight = false;
    bool pendingDestinationIsA = false;
    uint64_t cudaReadyCounter = 0;
    uint64_t vulkanReleaseCounter = 0;
    uint64_t pendingCudaReadyValue = 0;
    uint64_t completedCudaReadyValue = 0;
    uint64_t pendingSolveReleaseValue = 0;
    // Last reader release values per solution buffer (0 = A, 1 = B)
    uint64_t lastReaderReleaseValue[2] = {0, 0};
    std::vector<uint32_t> fluidNodeToGlobalFragment;
    std::vector<uint32_t> globalFragmentToFluidNode;

    AdvectionInput advectionInput{};
    bool advectionConnected = false;
    bool stagingCopyScheduled = false;
    VulkanExternalBuffer advectionFaceFlux;
    VulkanExternalBuffer advectionBoundaryFlux;
    CudaExternalBuffer advectionFaceFluxCuda;
    CudaExternalBuffer advectionBoundaryFluxCuda;
    VkDevice vulkanDeviceHandle = VK_NULL_HANDLE;
};
