#include "NodeGraphController.hpp"

#include "NodeGraphCompiler.hpp"
#include "NodeGraphDebugCache.hpp"
#include "cfd/CFDSystemComputeController.hpp"
#include "heat/HeatSystemComputeController.hpp"
#include "runtime/ModelComputeController.hpp"
#include "runtime/PointComputeRuntime.hpp"
#include "runtime/RemeshController.hpp"
#include "voronoi/VoronoiSystemComputeController.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "NodePayloadRegistry.hpp"
#include "runtime/RuntimeHeatComputeTransport.hpp"
#include "runtime/RuntimeHeatDisplayTransport.hpp"
#include "runtime/RuntimeModelDisplayTransport.hpp"
#include "runtime/RuntimePointDisplayTransport.hpp"
#include "runtime/RuntimeProductManager.hpp"
#include "runtime/RuntimeRemeshDisplayTransport.hpp"
#include "runtime/RuntimeVoronoiDisplayTransport.hpp"
#include "simgraph/SimGraphRuntime.hpp"

#include <iostream>
#include <unordered_set>

NodeGraphController::NodeGraphController(
    NodePayloadRegistry& payloadRegistry,
    const RuntimeConnections& connections)
    : payloadRegistry(payloadRegistry),
      runtimeConnections(connections),
      runtime(&payloadRegistry),
      packageController(connections) {
    plan.isValid = false;
}

void NodeGraphController::rebuildForDelta(const NodeGraphDelta& delta) {
    runtime.applyDelta(delta);

    bool nonLayout = false;
    for (const NodeGraphChange& change : delta.changes) {
        if (change.reason != NodeGraphChangeReason::Layout) {
            nonLayout = true;
            break;
        }
    }
    if (!nonLayout) return;

    packageCompilationPending = true;
    NodeGraphDebugCache::instance().setState(runtime.state(), &payloadRegistry);
    plan = NodeGraphCompiler::compile(runtime.state());
}

void NodeGraphController::tick() {
    if (plan.isValid && packageCompilationPending) {
        if (compileRuntimePackages()) packageCompilationPending = false;
        // sync simulation nodes before sim graph run
        syncSimulationNodes();
    }

    const RuntimeProductManager* products = packageController.products();
    const uint64_t stateRevision = runtime.state().revision;
    const uint64_t productRevision = products != nullptr ? products->getRevision() : 0;
    const uint64_t heatTraceRevision = runtimeConnections.heatComputeTransport
        ? runtimeConnections.heatComputeTransport->getDisplayRevision()
        : 0;
    const bool fullDisplaySync =
        stateRevision != lastDisplaySyncStateRevision ||
        productRevision != lastDisplaySyncProductRevision ||
        packageCompileRevision != lastDisplaySyncPackageRevision;
    if (fullDisplaySync) {
        updateDisplayTransports();
        lastDisplaySyncStateRevision = stateRevision;
        lastDisplaySyncProductRevision = productRevision;
        lastDisplaySyncPackageRevision = packageCompileRevision;
    } else if (heatTraceRevision != lastDisplaySyncHeatTraceRevision) {
        // update heat display transport only
        updateHeatDisplayTransport();
    }
    lastDisplaySyncHeatTraceRevision = heatTraceRevision;
}

void NodeGraphController::detachRuntimeBindings() {
    packageController.detachAllDisplayBindings();
    // unblock interop waits before quiescing gpu
    if (runtimeConnections.heatController) {
        runtimeConnections.heatController->releaseWaits();
    }
    if (runtimeConnections.cfdController) {
        runtimeConnections.cfdController->releaseWaits();
    }
}

void NodeGraphController::quiescedClear() {
    packageController.quiescedClear();
    plan = {};
    plan.isValid = false;
    simulationSockets.clear();
    lastDisplaySyncStateRevision = UINT64_MAX;
    lastDisplaySyncProductRevision = UINT64_MAX;
    lastDisplaySyncPackageRevision = UINT64_MAX;
    lastDisplaySyncHeatTraceRevision = UINT64_MAX;
    lastPackageErrors_.clear();
}

// sync committed simulation systems into sim graph
void NodeGraphController::syncSimulationNodes() {
    SimGraphRuntime* simGraph = runtimeConnections.simGraph;
    if (simGraph == nullptr) return;

    const RuntimePackageManager& packages = packageController.packages();
    std::unordered_set<uint64_t> desired;
    desired.reserve(packages.cfdSystems().size() + packages.heatSystems().size());

    if (runtimeConnections.cfdController) {
        for (const auto& [socketKey, package] : packages.cfdSystems()) {
            (void)package;
            cfd::CFDSystem* system = runtimeConnections.cfdController->getSystem(socketKey);
            if (system == nullptr) continue;
            simGraph->registerNode(socketKey, system);
            desired.insert(socketKey);
        }
    }

    if (runtimeConnections.heatController) {
        for (const auto& [socketKey, package] : packages.heatSystems()) {
            HeatSystem* system = runtimeConnections.heatController->getSystem(socketKey);
            if (system == nullptr) continue;
            simGraph->registerNode(socketKey, system);
            desired.insert(socketKey);

            const HeatSystemComputeController::Config* heatConfig =
                runtimeConnections.heatController->getConfig(socketKey);
            const uint64_t producerSocket =
                heatConfig && heatConfig->hasAdvection && package.cfdProduct.isValid()
                    ? package.cfdProduct.outputSocketKey
                    : 0;
            if (producerSocket == 0 ||
                runtimeConnections.cfdController == nullptr ||
                runtimeConnections.cfdController->getSystem(producerSocket) == nullptr) {
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

void NodeGraphController::resetGraph(const NodeGraphState& state) {
    NodeGraphDelta delta{};
    delta.fromRevision = runtime.state().revision;
    delta.toRevision = state.revision;
    NodeGraphChange resetChange{};
    resetChange.type = NodeGraphChangeType::Reset;
    resetChange.reason = NodeGraphChangeReason::Topology;
    delta.changes.push_back(resetChange);
    for (const auto& entry : state.nodes) {
        NodeGraphChange change{};
        change.type = NodeGraphChangeType::NodeUpsert;
        change.reason = NodeGraphChangeReason::Topology;
        change.node = entry.second;
        delta.changes.push_back(change);
    }
    for (const auto& entry : state.edges) {
        NodeGraphChange change{};
        change.type = NodeGraphChangeType::EdgeUpsert;
        change.reason = NodeGraphChangeReason::Topology;
        change.edge = entry.second;
        delta.changes.push_back(change);
    }
    rebuildForDelta(delta);
}

bool NodeGraphController::applyGraphDelta(const NodeGraphDelta& delta) {
    if (delta.fromRevision != runtime.state().revision) return false;
    rebuildForDelta(delta);
    return true;
}

const NodeGraphState& NodeGraphController::graphState() const {
    return runtime.state();
}

bool NodeGraphController::resolveGizmoTransformNode(
    uint64_t outputSocketKey,
    NodeGraphNodeId& outNodeId) const {
    outNodeId = {};
    return outputSocketKey != 0 &&
        findFirstUpstreamNodeByType(
            runtime.state(), outputSocketKey, nodegraphtypes::Transform, outNodeId);
}

void NodeGraphController::reclaimRetiringGenerations() {
    RuntimeProductManager* products = packageController.products();
    bool anyRetiring = products != nullptr && products->hasAnyRetiring();
    if (runtimeConnections.modelComputeController &&
        runtimeConnections.modelComputeController->hasAnyRetiring()) {
        anyRetiring = true;
    }
    if (runtimeConnections.pointComputeRuntime &&
        runtimeConnections.pointComputeRuntime->hasAnyRetiring()) {
        anyRetiring = true;
    }
    if (runtimeConnections.remeshController &&
        runtimeConnections.remeshController->hasAnyRetiring()) {
        anyRetiring = true;
    }
    if (runtimeConnections.voronoiController &&
        runtimeConnections.voronoiController->hasAnyRetiring()) {
        anyRetiring = true;
    }
    if (runtimeConnections.heatController &&
        runtimeConnections.heatController->hasAnyRetiring()) {
        anyRetiring = true;
    }
    if (runtimeConnections.cfdController &&
        runtimeConnections.cfdController->hasAnyRetiring()) {
        anyRetiring = true;
    }
    if (!anyRetiring) {
        return;
    }

    if (!runtimeConnections.quiesceGpu || !runtimeConnections.quiesceGpu()) {
        std::cerr << "[NodeGraphController] GPU quiesce failed; retiring generations retained"
                  << std::endl;
        return;
    }
    if (products) products->clearAllRetiring();
    if (runtimeConnections.modelComputeController) runtimeConnections.modelComputeController->clearAllRetiring();
    if (runtimeConnections.pointComputeRuntime) runtimeConnections.pointComputeRuntime->clearAllRetiring();
    if (runtimeConnections.remeshController) runtimeConnections.remeshController->clearAllRetiring();
    if (runtimeConnections.voronoiController) runtimeConnections.voronoiController->clearAllRetiring();
    if (runtimeConnections.heatController) runtimeConnections.heatController->clearAllRetiring();
    if (runtimeConnections.cfdController) runtimeConnections.cfdController->clearAllRetiring();
}

bool NodeGraphController::compileRuntimePackages() {
    reclaimRetiringGenerations();
    runtime.execute(plan);
    const NodeGraphEvaluation& evaluation = runtime.evaluation();
    const RuntimePackageCompiler::FrozenPackages frozenPackages =
        frozenPackageRebuildPending
            ? RuntimePackageCompiler::FrozenPackages::Rebuild
            : RuntimePackageCompiler::FrozenPackages::Preserve;
    std::vector<std::string> errors;
    if (!packageCompiler.validate(
            runtime.state(),
            plan,
            evaluation,
            payloadRegistry,
            errors)) {
        lastPackageErrors_.clear();
        for (const std::string& error : errors) {
            if (!lastPackageErrors_.empty()) lastPackageErrors_ += "; ";
            lastPackageErrors_ += error;
            std::cerr << "Runtime package compilation failed: " << error << '\n';
        }
        return false;
    }

    const RuntimeProductManager* products = packageController.products();
    if (!products) return false;
    packageController.beginCompilation();
    for (NodeGraphNodeId nodeId : plan.executionOrder) {
        const NodeGraphNode* node = runtime.state().node(nodeId);
        if (!node) {
            errors.push_back("Package schedule references a missing node.");
            lastPackageErrors_.clear();
            lastPackageErrors_ = errors.back();
            std::cerr << "Runtime package compilation failed: " << errors.back() << '\n';
            packageController.finishCompilation(/*success=*/false);
            return false;
        }

        if (!packageCompiler.compileNode(
                runtime.state(),
                *node,
                evaluation,
                payloadRegistry,
                *products,
                activeWorldUnit,
                frozenPackages,
                packageController.packages(),
                errors)) {
            packageController.invalidateOutputsForNode(*node);
            lastPackageErrors_.clear();
            for (const std::string& error : errors) {
                if (!lastPackageErrors_.empty()) lastPackageErrors_ += "; ";
                lastPackageErrors_ += error;
                std::cerr << "Runtime package compilation failed: " << error << '\n';
            }
            packageController.finishCompilation(/*success=*/false);
            return false;
        }

        if (!packageController.applyNode(*node)) {
            packageController.invalidateOutputsForNode(*node);
            errors.push_back("update failed; node invalidated");
            lastPackageErrors_.clear();
            for (const std::string& error : errors) {
                if (!lastPackageErrors_.empty()) lastPackageErrors_ += "; ";
                lastPackageErrors_ += error;
                std::cerr << "Runtime package compilation failed: " << error << '\n';
            }
            packageController.finishCompilation(/*success=*/false);
            return false;
        }
    }
    packageController.finishCompilation(/*success=*/true);
    lastPackageErrors_.clear();
    frozenPackageRebuildPending = false;

    NodeGraphDebugCache::instance().update(
        runtime.state().revision,
        evaluation.outputsBySocket);
    return true;
}

void NodeGraphController::setWorldUnit(units::LengthUnit unit) {
    if (unit == activeWorldUnit) return;
    activeWorldUnit = unit;
    packageCompilationPending = true;
    frozenPackageRebuildPending = true;
    ++packageCompileRevision;
}

void NodeGraphController::updateHeatDisplayTransport() {
    if (!runtimeConnections.heatDisplayTransport) return;

    RuntimePackageManager& packages = packageController.packages();
    const std::unordered_set<uint64_t> visibleKeys =
        nodeGraphDisplay.computeDisplayKeys(
            runtime.state(),
            runtime.evaluation(),
            packages,
            &payloadRegistry);

    runtimeConnections.heatDisplayTransport->sync(packages, visibleKeys);
    runtimeConnections.heatDisplayTransport->finalizeSync();
}

void NodeGraphController::updateDisplayTransports() {
    const bool hasDisplayTransports = runtimeConnections.modelDisplayTransport ||
        runtimeConnections.remeshDisplayTransport ||
        runtimeConnections.voronoiDisplayTransport ||
        runtimeConnections.heatDisplayTransport ||
        runtimeConnections.pointDisplayTransport;
    if (!hasDisplayTransports) return;

    RuntimePackageManager& packages = packageController.packages();
    const std::unordered_set<uint64_t> visibleKeys =
        nodeGraphDisplay.computeDisplayKeys(
            runtime.state(),
            runtime.evaluation(),
            packages,
            &payloadRegistry);

    if (runtimeConnections.modelDisplayTransport) {
        runtimeConnections.modelDisplayTransport->sync(packages, visibleKeys);
        runtimeConnections.modelDisplayTransport->finalizeSync();
    }
    if (runtimeConnections.remeshDisplayTransport) {
        runtimeConnections.remeshDisplayTransport->sync(packages, visibleKeys);
        runtimeConnections.remeshDisplayTransport->finalizeSync();
    }
    if (runtimeConnections.voronoiDisplayTransport) {
        runtimeConnections.voronoiDisplayTransport->sync(packages, visibleKeys);
        runtimeConnections.voronoiDisplayTransport->finalizeSync();
    }
    if (runtimeConnections.heatDisplayTransport) {
        runtimeConnections.heatDisplayTransport->sync(packages, visibleKeys);
        runtimeConnections.heatDisplayTransport->finalizeSync();
    }
    if (runtimeConnections.pointDisplayTransport) {
        runtimeConnections.pointDisplayTransport->sync(packages, visibleKeys);
        runtimeConnections.pointDisplayTransport->finalizeSync();
    }
}

const NodeGraphCompiled& NodeGraphController::compiledState() const {
    return plan;
}

void NodeGraphController::addRuntimeModelId(
    std::vector<uint32_t>& outIds,
    uint32_t id) const {
    if (id == 0) return;
    for (uint32_t existing : outIds) {
        if (existing == id) return;
    }
    outIds.push_back(id);
}

bool NodeGraphController::runtimeModelIdsForSocket(
    uint64_t socketKey,
    std::vector<uint32_t>& outIds) const {
    const RuntimeProductManager* products = packageController.products();
    const RuntimePackageManager& packages = packageController.packages();
    if (socketKey == 0 || !products) return false;

    if (const ModelPackage* package = packages.findAny<ModelPackage>(socketKey)) {
        if (const ModelProduct* product = products->resolve<ModelProduct>(package->productHandle))
            addRuntimeModelId(outIds, product->runtimeModelId);
        return true;
    }
    if (const RemeshPackage* package = packages.findAny<RemeshPackage>(socketKey)) {
        if (const RemeshProduct* product = products->resolve<RemeshProduct>(package->productHandle))
            addRuntimeModelId(outIds, product->runtimeModelId);
        return true;
    }
    if (const VoronoiPackage* package = packages.findAny<VoronoiPackage>(socketKey)) {
        if (const VoronoiProduct* product = products->resolve<VoronoiProduct>(package->productHandle)) {
            for (uint32_t modelId : product->globalDisplayRuntimeModelIds) {
                addRuntimeModelId(outIds, modelId);
            }
        }
        return true;
    }
    if (const HeatPackage* package = packages.findAny<HeatPackage>(socketKey)) {
        for (const HeatModelPackage& model : package->models) {
            if (const ModelProduct* product = products->resolve<ModelProduct>(model.modelProduct))
                addRuntimeModelId(outIds, product->runtimeModelId);
        }
        return true;
    }
    return packages.findAny<PointPackage>(socketKey) != nullptr;
}

bool NodeGraphController::runtimeModelIdsForNode(
    NodeGraphNodeId nodeId,
    std::vector<uint32_t>& outIds) const {
    outIds.clear();
    const auto nodeIt = runtime.state().nodes.find(nodeId.value);
    if (nodeIt == runtime.state().nodes.end()) return false;

    for (const NodeGraphSocket& output : nodeIt->second.outputs) {
        const uint64_t socketKey = NodeSocketKey(nodeIt->second.id, output.id).value;
        if (runtimeModelIdsForSocket(socketKey, outIds)) return true;
    }
    return false;
}
