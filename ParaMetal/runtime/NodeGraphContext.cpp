#include "NodeGraphContext.hpp"

#include "ComputeContext.hpp"
#include "VulkanCoreContext.hpp"
#include "cuda/CudaVulkanDevice.hpp"

#include <cuda_runtime.h>
#include "nodegraph/NodeGraphController.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/RuntimeCFDComputeTransport.hpp"
#include "runtime/RuntimeConnections.hpp"
#include "runtime/RuntimeHeatComputeTransport.hpp"
#include "runtime/RuntimeHeatDisplayTransport.hpp"
#include "runtime/RuntimeModelComputeTransport.hpp"
#include "runtime/RuntimeModelDisplayTransport.hpp"
#include "runtime/RuntimePointComputeTransport.hpp"
#include "runtime/RuntimePointDisplayTransport.hpp"
#include "runtime/RuntimeProductManager.hpp"
#include "runtime/RuntimeRemeshComputeTransport.hpp"
#include "runtime/RuntimeRemeshDisplayTransport.hpp"
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
    runtimeHeatDisplayTransportState->setComputeController(compute.heatSystemComputeController());

    RuntimeConnections runtimeConnections{};
    runtimeConnections.modelComputeTransport = runtimeModelComputeTransportState.get();
    runtimeConnections.pointComputeTransport = runtimePointComputeTransportState.get();
    runtimeConnections.remeshComputeTransport = runtimeRemeshTransportState.get();
    runtimeConnections.voronoiComputeTransport = runtimeVoronoiComputeTransportState.get();
    runtimeConnections.heatComputeTransport = runtimeHeatComputeTransportState.get();
    runtimeConnections.cfdComputeTransport = runtimeCFDComputeTransportState.get();
    runtimeConnections.modelDisplayTransport = runtimeModelDisplayTransportState.get();
    runtimeConnections.pointDisplayTransport = runtimePointDisplayTransportState.get();
    runtimeConnections.remeshDisplayTransport = runtimeRemeshDisplayTransportState.get();
    runtimeConnections.voronoiDisplayTransport = runtimeVoronoiDisplayTransportState.get();
    runtimeConnections.heatDisplayTransport = runtimeHeatDisplayTransportState.get();
    runtimeConnections.modelComputeController = compute.modelComputeController();
    runtimeConnections.pointComputeRuntime = compute.pointComputeRuntime();
    runtimeConnections.remeshController = compute.remeshController();
    runtimeConnections.voronoiController = compute.voronoiSystemComputeController();
    runtimeConnections.heatController = compute.heatSystemComputeController();
    runtimeConnections.cfdController = compute.cfdSystemComputeController();
    runtimeConnections.simGraph = compute.simGraphRuntime();
    runtimeConnections.quiesceGpu = [&core]() {
        bool synchronized = true;
        const VkDevice device = core.device().getDevice();
        if (device != VK_NULL_HANDLE && vkDeviceWaitIdle(device) != VK_SUCCESS) {
            synchronized = false;
        }
        const int cudaDevice = cudaVulkan::findDevice(core.device().getPhysicalDevice());
        if (cudaDevice >= 0) {
            const cudaError_t setDeviceResult = cudaSetDevice(cudaDevice);
            if (setDeviceResult != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                synchronized = false;
            }
        } else if (cudaVulkan::deviceWasFound()) {
            synchronized = false;
        }
        return synchronized;
    };
    nodeGraphControllerState = std::make_unique<NodeGraphController>(
        *payloadRegistryState, runtimeConnections);

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
