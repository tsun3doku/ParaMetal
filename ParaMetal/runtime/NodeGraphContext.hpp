#pragma once

#include <memory>

class ComputeContext;
class HeatDisplayController;
class ModelDisplayController;
class NodeGraphController;
class NodePayloadRegistry;
class PointDisplayController;
class RemeshDisplayController;
class RuntimeCFDComputeTransport;
class RuntimeHeatComputeTransport;
class RuntimeHeatDisplayTransport;
class RuntimeModelComputeTransport;
class RuntimeModelDisplayTransport;
class RuntimePointComputeTransport;
class RuntimePointDisplayTransport;
class RuntimeRemeshComputeTransport;
class RuntimeRemeshDisplayTransport;
class RuntimeVoronoiComputeTransport;
class RuntimeVoronoiDisplayTransport;
class VulkanCoreContext;
class VoronoiDisplayController;

struct NodeGraphDisplayControllers {
    ModelDisplayController& model;
    PointDisplayController& point;
    RemeshDisplayController& remesh;
    HeatDisplayController& heat;
    VoronoiDisplayController& voronoi;
};

// Owns node-graph payloads, package-to-product compilation, and compute/display transports
class NodeGraphContext {
public:
    NodeGraphContext();
    ~NodeGraphContext();

    bool initialize(
        VulkanCoreContext& core,
        ComputeContext& compute,
        const NodeGraphDisplayControllers& displayControllers);
    void shutdown();
    bool isInitialized() const;

    NodeGraphController* nodeGraphController();
    const NodeGraphController* nodeGraphController() const;

private:
    std::unique_ptr<NodePayloadRegistry> payloadRegistryState;
    std::unique_ptr<RuntimeModelComputeTransport> runtimeModelComputeTransportState;
    std::unique_ptr<RuntimeModelDisplayTransport> runtimeModelDisplayTransportState;
    std::unique_ptr<RuntimePointComputeTransport> runtimePointComputeTransportState;
    std::unique_ptr<RuntimePointDisplayTransport> runtimePointDisplayTransportState;
    std::unique_ptr<RuntimeRemeshDisplayTransport> runtimeRemeshDisplayTransportState;
    std::unique_ptr<RuntimeRemeshComputeTransport> runtimeRemeshTransportState;
    std::unique_ptr<RuntimeVoronoiComputeTransport> runtimeVoronoiComputeTransportState;
    std::unique_ptr<RuntimeVoronoiDisplayTransport> runtimeVoronoiDisplayTransportState;
    std::unique_ptr<RuntimeHeatComputeTransport> runtimeHeatComputeTransportState;
    std::unique_ptr<RuntimeHeatDisplayTransport> runtimeHeatDisplayTransportState;
    std::unique_ptr<RuntimeCFDComputeTransport> runtimeCFDComputeTransportState;
    std::unique_ptr<NodeGraphController> nodeGraphControllerState;
    bool initialized = false;
};
