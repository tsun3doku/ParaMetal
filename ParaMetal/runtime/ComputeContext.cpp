#include "ComputeContext.hpp"

#include "SceneContext.hpp"
#include "VulkanCoreContext.hpp"
#include "cfd/CFDSystemComputeController.hpp"
#include "framegraph/FrameSync.hpp"
#include "heat/HeatSystemComputeController.hpp"
#include "render/RenderConfig.hpp"
#include "runtime/ModelComputeController.hpp"
#include "runtime/PointComputeRuntime.hpp"
#include "runtime/RemeshController.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "scene/ModelUploader.hpp"
#include "simgraph/SimGraphRuntime.hpp"
#include "voronoi/VoronoiSystemComputeController.hpp"
#include "vulkan/ModelRegistry.hpp"

ComputeContext::ComputeContext() = default;

ComputeContext::~ComputeContext() {
    shutdown();
}

bool ComputeContext::initialize(
    VulkanCoreContext& core,
    SceneContext& scene,
    FrameSync& frameSync,
    std::atomic<bool>& runtimeBusy) {
    if (initialized) {
        return true;
    }

    auto* allocator = core.allocator();
    auto* commandPool = core.getRenderCommandPool();
    auto* transferCommandPool = core.getTransferCommandPool();
    auto* modelRegistry = scene.modelRegistry();
    auto* modelUploader = scene.modelUploader();
    if (!allocator || !commandPool || !transferCommandPool || !modelRegistry || !modelUploader) {
        shutdown();
        return false;
    }

    simGraphRuntimeState = std::make_unique<SimGraphRuntime>();
    if (!simGraphRuntimeState->initialize(
            core.device(),
            frameSync)) {
        shutdown();
        return false;
    }

    pointComputeRuntimeState = std::make_unique<PointComputeRuntime>(
        *allocator);
    modelComputeControllerState = std::make_unique<ModelComputeController>(
        *modelRegistry,
        *modelUploader,
        runtimeBusy);
    remeshControllerState = std::make_unique<RemeshController>(
        core.device(),
        *allocator,
        runtimeBusy);
    voronoiSystemComputeControllerState = std::make_unique<VoronoiSystemComputeController>(
        core.device(),
        *allocator,
        *transferCommandPool);
    heatSystemComputeControllerState = std::make_unique<HeatSystemComputeController>(
        core.device(),
        *allocator,
        *commandPool,
        *transferCommandPool,
        renderconfig::MaxFramesInFlight);
    cfdSystemComputeControllerState = std::make_unique<cfd::CFDSystemComputeController>(
        core.device(),
        *allocator,
        *commandPool,
        renderconfig::MaxFramesInFlight);

    initialized = true;
    return true;
}

void ComputeContext::shutdown() {
    simulationSockets.clear();
    remeshControllerState.reset();
    modelComputeControllerState.reset();
    pointComputeRuntimeState.reset();
    voronoiSystemComputeControllerState.reset();
    heatSystemComputeControllerState.reset();
    cfdSystemComputeControllerState.reset();
    if (simGraphRuntimeState) {
        simGraphRuntimeState->cleanup();
        simGraphRuntimeState.reset();
    }
    initialized = false;
}

bool ComputeContext::isInitialized() const {
    return initialized;
}

void ComputeContext::quiescedClear() {
    if (simGraphRuntimeState) {
        simGraphRuntimeState->clear();
    }
    simulationSockets.clear();
    if (modelComputeControllerState) {
        modelComputeControllerState->disableAll();
    }
    if (pointComputeRuntimeState) {
        pointComputeRuntimeState->quiescedClear();
    }
    if (remeshControllerState) {
        remeshControllerState->quiescedClear();
    }
    if (voronoiSystemComputeControllerState) {
        voronoiSystemComputeControllerState->quiescedClear();
    }
    if (heatSystemComputeControllerState) {
        heatSystemComputeControllerState->quiescedClear();
    }
    if (cfdSystemComputeControllerState) {
        cfdSystemComputeControllerState->quiescedClear();
    }
}

void ComputeContext::clearRetiringGenerations() {
    if (modelComputeControllerState) {
        modelComputeControllerState->clearAllRetiring();
    }
    if (pointComputeRuntimeState) {
        pointComputeRuntimeState->clearAllRetiring();
    }
    if (remeshControllerState) {
        remeshControllerState->clearAllRetiring();
    }
    if (voronoiSystemComputeControllerState) {
        voronoiSystemComputeControllerState->clearAllRetiring();
    }
    if (heatSystemComputeControllerState) {
        heatSystemComputeControllerState->clearAllRetiring();
    }
    if (cfdSystemComputeControllerState) {
        cfdSystemComputeControllerState->clearAllRetiring();
    }
}

void ComputeContext::tickDomainServices() {
    if (heatSystemComputeControllerState) {
        heatSystemComputeControllerState->updateSerialInputs();
    }
}

void ComputeContext::syncSimulationNodes(
    const RuntimePackageManager& packages) {
    SimGraphRuntime* simGraph = simGraphRuntime();
    if (simGraph == nullptr) {
        return;
    }

    std::unordered_set<uint64_t> desired;
    desired.reserve(packages.cfdSystems().size() + packages.heatSystems().size());

    if (cfdSystemComputeControllerState) {
        for (const auto& [socketKey, package] : packages.cfdSystems()) {
            (void)package;
            cfd::CFDSystem* system =
                cfdSystemComputeControllerState->getSystem(socketKey);
            if (system == nullptr) {
                continue;
            }
            simGraph->registerNode(socketKey, system);
            desired.insert(socketKey);
        }
    }

    if (heatSystemComputeControllerState) {
        for (const auto& [socketKey, package] : packages.heatSystems()) {
            (void)package;
            HeatSystem* system =
                heatSystemComputeControllerState->getSystem(socketKey);
            if (system == nullptr) {
                continue;
            }
            simGraph->registerNode(socketKey, system);
            desired.insert(socketKey);

            const HeatSystemComputeController::Config* heatConfig =
                heatSystemComputeControllerState->getConfig(socketKey);
            const uint64_t producerSocket =
                heatConfig && heatConfig->hasAdvection &&
                package.cfdProduct.isValid()
                    ? package.cfdProduct.outputSocketKey
                    : 0;
            if (producerSocket == 0 ||
                cfdSystemComputeControllerState == nullptr ||
                cfdSystemComputeControllerState->getSystem(producerSocket) ==
                    nullptr) {
                simGraph->disconnect(socketKey);
                continue;
            }
            SimConnection connection{};
            connection.connectionId = socketKey;
            connection.producerSocket = producerSocket;
            connection.consumerSocket = socketKey;
            connection.mode = SimConnectionMode::Staged;
            simGraph->connect(connection);
        }
    }

    for (uint64_t socketKey : simulationSockets) {
        if (desired.find(socketKey) == desired.end()) {
            simGraph->unregisterNode(socketKey);
        }
    }
    simulationSockets = std::move(desired);
}
