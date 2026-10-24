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
        *commandPool,
        renderconfig::MaxFramesInFlight);

    initialized = true;
    return true;
}

void ComputeContext::shutdown() {
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

SimGraphRuntime* ComputeContext::simGraphRuntime() {
    return simGraphRuntimeState.get();
}

const SimGraphRuntime* ComputeContext::simGraphRuntime() const {
    return simGraphRuntimeState.get();
}

ModelComputeController* ComputeContext::modelComputeController() {
    return modelComputeControllerState.get();
}

const ModelComputeController* ComputeContext::modelComputeController() const {
    return modelComputeControllerState.get();
}

PointComputeRuntime* ComputeContext::pointComputeRuntime() {
    return pointComputeRuntimeState.get();
}

const PointComputeRuntime* ComputeContext::pointComputeRuntime() const {
    return pointComputeRuntimeState.get();
}

RemeshController* ComputeContext::remeshController() {
    return remeshControllerState.get();
}

const RemeshController* ComputeContext::remeshController() const {
    return remeshControllerState.get();
}

VoronoiSystemComputeController* ComputeContext::voronoiSystemComputeController() {
    return voronoiSystemComputeControllerState.get();
}

const VoronoiSystemComputeController* ComputeContext::voronoiSystemComputeController() const {
    return voronoiSystemComputeControllerState.get();
}

HeatSystemComputeController* ComputeContext::heatSystemComputeController() {
    return heatSystemComputeControllerState.get();
}

const HeatSystemComputeController* ComputeContext::heatSystemComputeController() const {
    return heatSystemComputeControllerState.get();
}

cfd::CFDSystemComputeController* ComputeContext::cfdSystemComputeController() {
    return cfdSystemComputeControllerState.get();
}

const cfd::CFDSystemComputeController* ComputeContext::cfdSystemComputeController() const {
    return cfdSystemComputeControllerState.get();
}
