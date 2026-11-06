#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

class FrameSync;
class HeatSystemComputeController;
class ModelComputeController;
class PointComputeRuntime;
class RemeshController;
class SceneContext;
class SimGraphRuntime;
class VulkanCoreContext;
class VoronoiSystemComputeController;

namespace cfd {
class CFDSystemComputeController;
}

// Owns simulation compute services while RuntimeSystems drives execution and lifecycle
class ComputeContext {
public:
    ComputeContext();
    ~ComputeContext();

    bool initialize(
        VulkanCoreContext& core,
        SceneContext& scene,
        FrameSync& frameSync,
        std::atomic<bool>& runtimeBusy);
    void shutdown();
        void quiescedClear();
    bool isInitialized() const;
    void tickDomainServices();

    SimGraphRuntime* simGraphRuntime() { return simGraphRuntimeState.get(); }
    const SimGraphRuntime* simGraphRuntime() const { return simGraphRuntimeState.get(); }
    ModelComputeController* modelComputeController() { return modelComputeControllerState.get(); }
    const ModelComputeController* modelComputeController() const { return modelComputeControllerState.get(); }
    PointComputeRuntime* pointComputeRuntime() { return pointComputeRuntimeState.get(); }
    const PointComputeRuntime* pointComputeRuntime() const { return pointComputeRuntimeState.get(); }
    RemeshController* remeshController() { return remeshControllerState.get(); }
    const RemeshController* remeshController() const { return remeshControllerState.get(); }
    VoronoiSystemComputeController* voronoiSystemComputeController() { return voronoiSystemComputeControllerState.get(); }
    const VoronoiSystemComputeController* voronoiSystemComputeController() const { return voronoiSystemComputeControllerState.get(); }
    HeatSystemComputeController* heatSystemComputeController() { return heatSystemComputeControllerState.get(); }
    const HeatSystemComputeController* heatSystemComputeController() const { return heatSystemComputeControllerState.get(); }
    cfd::CFDSystemComputeController* cfdSystemComputeController() { return cfdSystemComputeControllerState.get(); }
    const cfd::CFDSystemComputeController* cfdSystemComputeController() const { return cfdSystemComputeControllerState.get(); }

private:
    std::unique_ptr<SimGraphRuntime> simGraphRuntimeState;
    std::unique_ptr<ModelComputeController> modelComputeControllerState;
    std::unique_ptr<PointComputeRuntime> pointComputeRuntimeState;
    std::unique_ptr<RemeshController> remeshControllerState;
    std::unique_ptr<VoronoiSystemComputeController> voronoiSystemComputeControllerState;
    std::unique_ptr<HeatSystemComputeController> heatSystemComputeControllerState;
    std::unique_ptr<cfd::CFDSystemComputeController> cfdSystemComputeControllerState;
    bool initialized = false;
};
