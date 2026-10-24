#pragma once

#include <atomic>
#include <memory>

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
    bool isInitialized() const;

    SimGraphRuntime* simGraphRuntime();
    const SimGraphRuntime* simGraphRuntime() const;
    ModelComputeController* modelComputeController();
    const ModelComputeController* modelComputeController() const;
    PointComputeRuntime* pointComputeRuntime();
    const PointComputeRuntime* pointComputeRuntime() const;
    RemeshController* remeshController();
    const RemeshController* remeshController() const;
    VoronoiSystemComputeController* voronoiSystemComputeController();
    const VoronoiSystemComputeController* voronoiSystemComputeController() const;
    HeatSystemComputeController* heatSystemComputeController();
    const HeatSystemComputeController* heatSystemComputeController() const;
    cfd::CFDSystemComputeController* cfdSystemComputeController();
    const cfd::CFDSystemComputeController* cfdSystemComputeController() const;

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
