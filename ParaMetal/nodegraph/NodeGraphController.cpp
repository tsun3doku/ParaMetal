#include "NodeGraphController.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include <unordered_set>

#include "NodeGraph.hpp"
#include "NodeGraphDebugCache.hpp"
#include "NodePayloadRegistry.hpp"
#include "runtime/RuntimeContactComputeTransport.hpp"
#include "runtime/RuntimeContactDisplayTransport.hpp"
#include "runtime/RuntimeHeatComputeTransport.hpp"
#include "runtime/RuntimeHeatDisplayTransport.hpp"
#include "runtime/RuntimeModelComputeTransport.hpp"
#include "runtime/RuntimeModelDisplayTransport.hpp"
#include "runtime/RuntimePackageCompiler.hpp"
#include "runtime/RuntimePointComputeTransport.hpp"
#include "runtime/RuntimePointDisplayTransport.hpp"
#include "runtime/RuntimeRemeshComputeTransport.hpp"
#include "runtime/RuntimeRemeshDisplayTransport.hpp"
#include "runtime/RuntimeVoronoiComputeTransport.hpp"
#include "runtime/RuntimeVoronoiDisplayTransport.hpp"
#include "runtime/RuntimePackageManager.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "vulkan/MemoryAllocator.hpp"

NodeGraphController::NodeGraphController(NodeGraph* bridge, const NodeRuntimeServices& services)
    : bridge(bridge),
      runtimeServices(services),
      runtime(services) {
    plan.isValid = false;
    if (runtimeServices.vulkanDevice && runtimeServices.memoryAllocator) {
        productManager = std::make_unique<RuntimeProductManager>(
            *runtimeServices.vulkanDevice, *runtimeServices.memoryAllocator);
    }
    RuntimeProductManager* pm = productManager.get();
    if (runtimeServices.modelComputeTransport) { runtimeServices.modelComputeTransport->setProducts(pm); }
    if (runtimeServices.remeshComputeTransport) { runtimeServices.remeshComputeTransport->setProducts(pm); }
    if (runtimeServices.voronoiComputeTransport) { runtimeServices.voronoiComputeTransport->setProducts(pm); }
    if (runtimeServices.contactComputeTransport) { runtimeServices.contactComputeTransport->setProducts(pm); }
    if (runtimeServices.heatComputeTransport) { runtimeServices.heatComputeTransport->setProducts(pm); }
    if (runtimeServices.pointComputeTransport) { runtimeServices.pointComputeTransport->setProducts(pm); }
    if (runtimeServices.modelDisplayTransport) { runtimeServices.modelDisplayTransport->setProducts(pm); }
    if (runtimeServices.remeshDisplayTransport) { runtimeServices.remeshDisplayTransport->setProducts(pm); }
    if (runtimeServices.voronoiDisplayTransport) { runtimeServices.voronoiDisplayTransport->setProducts(pm); }
    if (runtimeServices.contactDisplayTransport) { runtimeServices.contactDisplayTransport->setProducts(pm); }
    if (runtimeServices.heatDisplayTransport) { runtimeServices.heatDisplayTransport->setProducts(pm); }
    if (runtimeServices.pointDisplayTransport) { runtimeServices.pointDisplayTransport->setProducts(pm); }
}

void NodeGraphController::consumePendingGraphDelta() {
    if (!bridge) {
        return;
    }

    NodeGraphDelta delta{};
    if (!bridge->consumeChanges(revisionSeen, delta)) {
        return;
    }
    runtime.applyDelta(delta);

    bool nonLayout = false;
    for (const NodeGraphChange& change : delta.changes) {
        if (change.reason != NodeGraphChangeReason::Layout) {
            nonLayout = true;
            break;
        }
    }
    if (!nonLayout) {
        return;
    }

    pendingPackageRevision = runtime.state().revision;
    NodeGraphDebugCache::instance().setState(runtime.state(), runtimeServices.payloadRegistry);
    plan = NodeGraphCompiler::compile(runtime.state());
}

void NodeGraphController::tick() {
    consumePendingGraphDelta();

    if (plan.isValid && completedPackageRevision != pendingPackageRevision) {
        compileRuntimePackages();
        completedPackageRevision = pendingPackageRevision;
    }

    updateDisplayTransports();
}

void NodeGraphController::compileRuntimePackages() {
    runtime.execute(plan);
    const NodeGraphEvaluationState& execState = runtime.evaluationState();

    RuntimePackageCompiler packageCompiler{};
    packageManager.beginCompile();

    // Compile and apply packages in topological order
    for (NodeGraphNodeId nodeId : plan.executionOrder) {
        const auto nodeIt = runtime.state().nodes.find(nodeId.value);
        if (nodeIt == runtime.state().nodes.end()) {
            continue;
        }
        const auto& node = nodeIt->second;

        packageCompiler.compileNode(runtime.state(), node, execState, runtimeServices.payloadRegistry, *productManager, packageManager);

        updateComputeTransports(node);
    }

    // Clean up stale packages
    if (runtimeServices.modelComputeTransport) {
        packageManager.forEachStale<ModelPackage>([this](uint64_t key, const ModelPackage&) {
            runtimeServices.modelComputeTransport->remove(key);
            runtime.setOutputProductHandle(key, {});
        });
    }
    if (runtimeServices.pointComputeTransport) {
        packageManager.forEachStale<PointPackage>([this](uint64_t key, const PointPackage&) {
            runtimeServices.pointComputeTransport->remove(key);
            runtime.setOutputProductHandle(key, {});
        });
    }
    if (runtimeServices.remeshComputeTransport) {
        packageManager.forEachStale<RemeshPackage>([this](uint64_t key, const RemeshPackage&) {
            runtimeServices.remeshComputeTransport->remove(key);
            runtime.setOutputProductHandle(key, {});
        });
    }
    if (runtimeServices.voronoiComputeTransport) {
        packageManager.forEachStale<VoronoiPackage>([this](uint64_t key, const VoronoiPackage&) {
            runtimeServices.voronoiComputeTransport->remove(key);
            runtime.setOutputProductHandle(key, {});
        });
    }
    if (runtimeServices.contactComputeTransport) {
        packageManager.forEachStale<ContactPackage>([this](uint64_t key, const ContactPackage&) {
            runtimeServices.contactComputeTransport->remove(key);
            runtime.setOutputProductHandle(key, {});
        });
    }
    if (runtimeServices.heatComputeTransport) {
        packageManager.forEachStale<HeatPackage>([this](uint64_t key, const HeatPackage&) {
            runtimeServices.heatComputeTransport->remove(key);
            runtime.setOutputProductHandle(key, {});
        });
    }

    packageManager.destroyStale();

    if (runtimeServices.modelComputeTransport) {
        runtimeServices.modelComputeTransport->flush();
    }

    NodeGraphDebugCache::instance().update(
        runtime.state().revision,
        execState.outputBySocket);
}

void NodeGraphController::updateComputeTransports(const NodeGraphNode& node) {
    for (const NodeGraphSocket& outputSocket : node.outputs) {
        const uint64_t socketKey = NodeSocketKey(node.id, outputSocket.id).value;
        updateComputeTransport(socketKey);
    }
}

void NodeGraphController::updateComputeTransport(uint64_t socketKey) {
    if (socketKey == 0) return;

    if (runtimeServices.modelComputeTransport) {
        if (const auto* pkg = packageManager.find<ModelPackage>(socketKey)) {
            ProductHandle handle = runtimeServices.modelComputeTransport->apply(socketKey, *pkg);
            if (handle.isValid()) {
                runtime.setOutputProductHandle(socketKey, handle);
                packageManager.setProductHandle(socketKey, handle);
            }
            return;
        }
    }
    if (runtimeServices.pointComputeTransport) {
        if (const auto* pkg = packageManager.find<PointPackage>(socketKey)) {
            ProductHandle handle = runtimeServices.pointComputeTransport->apply(socketKey, *pkg);
            if (handle.isValid()) {
                runtime.setOutputProductHandle(socketKey, handle);
                packageManager.setProductHandle(socketKey, handle);
            }
            return;
        }
    }
    if (runtimeServices.remeshComputeTransport) {
        if (const auto* pkg = packageManager.find<RemeshPackage>(socketKey)) {
            ProductHandle handle = runtimeServices.remeshComputeTransport->apply(socketKey, *pkg);
            if (handle.isValid()) {
                runtime.setOutputProductHandle(socketKey, handle);
                packageManager.setProductHandle(socketKey, handle);
            }
            return;
        }
    }
    if (runtimeServices.voronoiComputeTransport) {
        if (const auto* pkg = packageManager.find<VoronoiPackage>(socketKey)) {
            ProductHandle handle = runtimeServices.voronoiComputeTransport->apply(socketKey, *pkg);
            if (handle.isValid()) {
                runtime.setOutputProductHandle(socketKey, handle);
                packageManager.setProductHandle(socketKey, handle);
            }
            return;
        }
    }
    if (runtimeServices.contactComputeTransport) {
        if (const auto* pkg = packageManager.find<ContactPackage>(socketKey)) {
            ProductHandle handle = runtimeServices.contactComputeTransport->apply(socketKey, *pkg);
            if (handle.isValid()) {
                runtime.setOutputProductHandle(socketKey, handle);
                packageManager.setProductHandle(socketKey, handle);
            }
            return;
        }
    }
    if (runtimeServices.heatComputeTransport) {
        if (const auto* pkg = packageManager.find<HeatPackage>(socketKey)) {
            ProductHandle handle = runtimeServices.heatComputeTransport->apply(socketKey, *pkg);
            if (handle.isValid()) {
                runtime.setOutputProductHandle(socketKey, handle);
                packageManager.setProductHandle(socketKey, handle);
            }
            return;
        }
    }
}

void NodeGraphController::updateDisplayTransports() {
    const bool hasDisplayTransports = runtimeServices.modelDisplayTransport ||
        runtimeServices.remeshDisplayTransport ||
        runtimeServices.voronoiDisplayTransport ||
        runtimeServices.contactDisplayTransport ||
        runtimeServices.heatDisplayTransport ||
        runtimeServices.pointDisplayTransport;
    if (!hasDisplayTransports) {
        return;
    }

    const std::unordered_set<uint64_t> visibleKeys =
        nodeGraphDisplay.computeDisplayKeys(
            runtime.state(),
            runtime.evaluationState(),
            packageManager,
            runtimeServices.payloadRegistry);

    if (runtimeServices.modelDisplayTransport) {
        runtimeServices.modelDisplayTransport->sync(packageManager, visibleKeys);
        runtimeServices.modelDisplayTransport->finalizeSync();
    }
    if (runtimeServices.remeshDisplayTransport) {
        runtimeServices.remeshDisplayTransport->sync(packageManager, visibleKeys);
        runtimeServices.remeshDisplayTransport->finalizeSync();
    }
    if (runtimeServices.voronoiDisplayTransport) {
        runtimeServices.voronoiDisplayTransport->sync(packageManager, visibleKeys);
        runtimeServices.voronoiDisplayTransport->finalizeSync();
    }
    if (runtimeServices.contactDisplayTransport) {
        runtimeServices.contactDisplayTransport->sync(packageManager, visibleKeys);
        runtimeServices.contactDisplayTransport->finalizeSync();
    }
    if (runtimeServices.heatDisplayTransport) {
        runtimeServices.heatDisplayTransport->sync(packageManager, visibleKeys);
        runtimeServices.heatDisplayTransport->finalizeSync();
    }
    if (runtimeServices.pointDisplayTransport) {
        runtimeServices.pointDisplayTransport->sync(packageManager, visibleKeys);
        runtimeServices.pointDisplayTransport->finalizeSync();
    }
}

const NodeGraphCompiled& NodeGraphController::compiledState() const {
    return plan;
}
