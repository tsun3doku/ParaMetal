#include "NodeGraphContext.hpp"

#include "ComputeContext.hpp"
#include "VulkanCoreContext.hpp"

#include "nodegraph/NodeGraphController.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/RuntimeCFDComputeTransport.hpp"
#include "runtime/RuntimeCFDDisplayTransport.hpp"
#include "runtime/RuntimeHeatComputeTransport.hpp"
#include "runtime/RuntimeHeatDisplayTransport.hpp"
#include "runtime/RuntimeModelComputeTransport.hpp"
#include "runtime/RuntimeModelDisplayTransport.hpp"
#include "runtime/RuntimePointComputeTransport.hpp"
#include "runtime/RuntimePointDisplayTransport.hpp"
#include "runtime/RuntimeProductManager.hpp"
#include "runtime/RuntimeRemeshComputeTransport.hpp"
#include "runtime/RuntimeRemeshDisplayTransport.hpp"
#include "runtime/RuntimeTransports.hpp"
#include "runtime/RuntimeVoronoiComputeTransport.hpp"
#include "runtime/RuntimeVoronoiDisplayTransport.hpp"

NodeGraphContext::NodeGraphContext() = default;

NodeGraphContext::~NodeGraphContext() {
    shutdown();
}

bool NodeGraphContext::initialize(
    VulkanCoreContext& core,
    ComputeContext& compute,
    const NodeGraphDisplayControllers& displayControllers) {
    if (initialized) {
        return true;
    }

    auto* allocator = core.allocator();
    if (!allocator ||
        !compute.simGraphRuntime() ||
        !compute.modelComputeController() ||
        !compute.pointComputeRuntime() ||
        !compute.remeshController() ||
        !compute.voronoiSystemComputeController() ||
        !compute.heatSystemComputeController() ||
        !compute.cfdSystemComputeController()) {
        shutdown();
        return false;
    }

    payloadRegistryState = std::make_unique<NodePayloadRegistry>();
    runtimeModelComputeTransportState = std::make_unique<RuntimeModelComputeTransport>();
    runtimeModelDisplayTransportState = std::make_unique<RuntimeModelDisplayTransport>();
    runtimePointComputeTransportState = std::make_unique<RuntimePointComputeTransport>();
    runtimePointDisplayTransportState = std::make_unique<RuntimePointDisplayTransport>();
    runtimeRemeshDisplayTransportState = std::make_unique<RuntimeRemeshDisplayTransport>();
    runtimeRemeshTransportState = std::make_unique<RuntimeRemeshComputeTransport>();
    runtimeVoronoiDisplayTransportState = std::make_unique<RuntimeVoronoiDisplayTransport>();
    runtimeVoronoiComputeTransportState = std::make_unique<RuntimeVoronoiComputeTransport>();
    runtimeHeatComputeTransportState = std::make_unique<RuntimeHeatComputeTransport>();
    runtimeHeatDisplayTransportState = std::make_unique<RuntimeHeatDisplayTransport>();
    runtimeCFDDisplayTransportState = std::make_unique<RuntimeCFDDisplayTransport>();
    runtimeCFDComputeTransportState = std::make_unique<RuntimeCFDComputeTransport>();

    runtimeModelComputeTransportState->setController(compute.modelComputeController());
    runtimePointComputeTransportState->setRuntime(compute.pointComputeRuntime());
    runtimeRemeshTransportState->setController(compute.remeshController());
    runtimeVoronoiComputeTransportState->setController(compute.voronoiSystemComputeController());
    runtimeHeatComputeTransportState->setController(compute.heatSystemComputeController());
    runtimeCFDComputeTransportState->setController(compute.cfdSystemComputeController());

    runtimeModelDisplayTransportState->setController(&displayControllers.model);
    runtimePointDisplayTransportState->setController(&displayControllers.point);
    runtimeRemeshDisplayTransportState->setController(&displayControllers.remesh);
    runtimeVoronoiDisplayTransportState->setController(&displayControllers.voronoi);
    runtimeHeatDisplayTransportState->setController(&displayControllers.heat);
    runtimeCFDDisplayTransportState->setController(&displayControllers.cfd);

    RuntimeTransports transports{};
    transports.modelComputeTransport = runtimeModelComputeTransportState.get();
    transports.pointComputeTransport = runtimePointComputeTransportState.get();
    transports.remeshComputeTransport = runtimeRemeshTransportState.get();
    transports.voronoiComputeTransport = runtimeVoronoiComputeTransportState.get();
    transports.heatComputeTransport = runtimeHeatComputeTransportState.get();
    transports.cfdComputeTransport = runtimeCFDComputeTransportState.get();
    transports.modelDisplayTransport = runtimeModelDisplayTransportState.get();
    transports.pointDisplayTransport = runtimePointDisplayTransportState.get();
    transports.remeshDisplayTransport = runtimeRemeshDisplayTransportState.get();
    transports.voronoiDisplayTransport = runtimeVoronoiDisplayTransportState.get();
    transports.heatDisplayTransport = runtimeHeatDisplayTransportState.get();
    transports.cfdDisplayTransport = runtimeCFDDisplayTransportState.get();

    nodeGraphControllerState = std::make_unique<NodeGraphController>(
        *payloadRegistryState, compute, transports);

    initialized = true;
    return true;
}

void NodeGraphContext::shutdown() {
    if (nodeGraphControllerState && nodeGraphControllerState->getProductManager()) {
        nodeGraphControllerState->getProductManager()->clear();
    }
    nodeGraphControllerState.reset();
    runtimeModelDisplayTransportState.reset();
    runtimeRemeshDisplayTransportState.reset();
    runtimeRemeshTransportState.reset();
    runtimePointDisplayTransportState.reset();
    runtimePointComputeTransportState.reset();
    runtimeHeatComputeTransportState.reset();
    runtimeHeatDisplayTransportState.reset();
    runtimeCFDDisplayTransportState.reset();
    runtimeCFDComputeTransportState.reset();
    runtimeVoronoiDisplayTransportState.reset();
    runtimeVoronoiComputeTransportState.reset();
    runtimeModelComputeTransportState.reset();
    // Destroyed last because transports and graph resolve payload types through it
    payloadRegistryState.reset();
    initialized = false;
}

bool NodeGraphContext::isInitialized() const {
    return initialized;
}

NodeGraphController* NodeGraphContext::nodeGraphController() {
    return nodeGraphControllerState.get();
}

const NodeGraphController* NodeGraphContext::nodeGraphController() const {
    return nodeGraphControllerState.get();
}
