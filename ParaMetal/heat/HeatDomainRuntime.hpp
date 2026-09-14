#pragma once

#include "framegraph/ComputePass.hpp"
#include "voronoi/GlobalDomainTopology.hpp"

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "heat/HeatGlobalSolver.hpp"
#include "heat/HeatSystemPresets.hpp"
#include "heat/HeatFluidFlow.hpp"

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
        for (int i = 0; i < 6; ++i) {
            domainBoundaryTypes[i] = types[i];
            domainBoundaryTemperaturesC[i] = temperaturesC[i];
            domainBoundaryHeatFluxes[i] = heatFluxes[i];
            domainBoundaryHeatTransferCoefficients[i] = heatTransferCoefficients[i];
        }
    }

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
    void prepareGlobalRenderSynchronization(bool activeIsA, bool newSolveCompleted);
    void waitGlobalSolveIdle();
    bool hasSolveFailed() const { return globalSolver ? globalSolver->hasSolveFailed() : false; }

    ComputePass::Synchronization getGlobalSynchronization() const { return globalSynchronization; }
    void clearGlobalSynchronization() { globalSynchronization = ComputePass::Synchronization{}; }

    const std::vector<uint32_t>& getGlobalFragmentToFluidNode() const { return globalFragmentToFluidNode; }

private:
    bool assembleGlobalMatrix(
        const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& models,
        HeatFluidRuntime* fluidRuntime,
        float fixedTimeStep,
        std::vector<int>& rowOffsets,
        std::vector<int>& columnIndices,
        std::vector<float>& values,
        std::vector<float>& thermalMasses,
        std::vector<HeatGlobalSolver::TemperatureNodes>& temperatureNodes,
        std::vector<float>& rhsAdditions);

    bool updateFluidFlow(HeatFluidRuntime* fluidRuntime, float fixedTimeStep);

    std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>> activeModels;

    topology::GlobalDomain globalDomain;
    uint32_t domainBoundaryTypes[6] = {};
    float domainBoundaryTemperaturesC[6] = {
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC,
        HeatSimDefaults::ambientTemperatureC
    };
    float domainBoundaryHeatFluxes[6] = {};
    float domainBoundaryHeatTransferCoefficients[6] = {};

    bool globalTopologyDirtyFlag = false;
    bool globalValuesDirtyFlag = false;
    bool globalAdvectionDirtyFlag = false;

    std::unique_ptr<HeatGlobalSolver> globalSolver;
    ComputePass::Synchronization globalSynchronization;
    bool globalSolveInFlight = false;
    bool pendingDestinationIsA = false;
    uint64_t cudaReadyCounter = 0;
    uint64_t vulkanReleaseCounter = 0;
    uint64_t pendingCudaReadyValue = 0;
    uint64_t lastVulkanReleaseValue[2]{0, 0};
    std::vector<uint32_t> fragmentSolverNodes;
    std::vector<uint32_t> fluidNodeToGlobalFragment;
    std::vector<uint32_t> globalFragmentToFluidNode;
    heat::HeatFluidFlow heatFluidFlow;
};