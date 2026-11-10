#include "NodeGraphController.hpp"

#include "NodeGraphCompiler.hpp"
#include "NodeGraphDebugCache.hpp"
#include "cfd/CFDSystemComputeController.hpp"
#include "heat/HeatSystemComputeController.hpp"
#include "runtime/ComputeContext.hpp"
#include "runtime/ModelComputeController.hpp"
#include "runtime/PointComputeRuntime.hpp"
#include "runtime/RemeshController.hpp"
#include "voronoi/VoronoiSystemComputeController.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "NodePayloadRegistry.hpp"
#include "runtime/RuntimeHeatDisplayTransport.hpp"
#include "runtime/RuntimeCFDDisplayTransport.hpp"
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
    ComputeContext& computeContext,
    const RuntimeTransports& transportBindings)
    : payloadRegistry(payloadRegistry),
      compute(computeContext),
      transports(transportBindings),
      runtime(&payloadRegistry),
      packageController(transportBindings) {
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
    const bool fullDisplaySync =
        stateRevision != lastDisplaySyncStateRevision ||
        productRevision != lastDisplaySyncProductRevision ||
        packageCompileRevision != lastDisplaySyncPackageRevision;
    if (fullDisplaySync) {
        updateDisplayTransports();
        lastDisplaySyncStateRevision = stateRevision;
        lastDisplaySyncProductRevision = productRevision;
        lastDisplaySyncPackageRevision = packageCompileRevision;
    }
}

void NodeGraphController::detachRuntimeBindings() {
    packageController.detachAllDisplayBindings();
    // unblock interop waits before quiescing gpu
    if (compute.heatSystemComputeController()) {
        compute.heatSystemComputeController()->releaseWaits();
    }
    if (compute.cfdSystemComputeController()) {
        compute.cfdSystemComputeController()->releaseWaits();
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
    lastPackageErrors_.clear();
}

// sync committed simulation systems into sim graph
void NodeGraphController::syncSimulationNodes() {
    SimGraphRuntime* simGraph = compute.simGraphRuntime();
    if (simGraph == nullptr) return;

    const RuntimePackageManager& packages = packageController.packages();
    std::unordered_set<uint64_t> desired;
    desired.reserve(packages.cfdSystems().size() + packages.heatSystems().size());

    if (compute.cfdSystemComputeController()) {
        for (const auto& [socketKey, package] : packages.cfdSystems()) {
            (void)package;
            cfd::CFDSystem* system = compute.cfdSystemComputeController()->getSystem(socketKey);
            if (system == nullptr) continue;
            simGraph->registerNode(socketKey, system);
            desired.insert(socketKey);
        }
    }

    if (compute.heatSystemComputeController()) {
        for (const auto& [socketKey, package] : packages.heatSystems()) {
            HeatSystem* system = compute.heatSystemComputeController()->getSystem(socketKey);
            if (system == nullptr) continue;
            simGraph->registerNode(socketKey, system);
            desired.insert(socketKey);

            const HeatSystemComputeController::Config* heatConfig =
                compute.heatSystemComputeController()->getConfig(socketKey);
            const uint64_t producerSocket =
                heatConfig && heatConfig->hasAdvection && package.cfdProduct.isValid()
                    ? package.cfdProduct.outputSocketKey
                    : 0;
            if (producerSocket == 0 ||
                compute.cfdSystemComputeController() == nullptr ||
                compute.cfdSystemComputeController()->getSystem(producerSocket) == nullptr) {
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

bool NodeGraphController::hasRetiringGenerations() const {
    const RuntimeProductManager* products = packageController.products();
    if (products != nullptr && products->hasAnyRetiring()) {
        return true;
    }
    if (compute.modelComputeController() &&
        compute.modelComputeController()->hasAnyRetiring()) {
        return true;
    }
    if (compute.pointComputeRuntime() &&
        compute.pointComputeRuntime()->hasAnyRetiring()) {
        return true;
    }
    if (compute.remeshController() &&
        compute.remeshController()->hasAnyRetiring()) {
        return true;
    }
    if (compute.voronoiSystemComputeController() &&
        compute.voronoiSystemComputeController()->hasAnyRetiring()) {
        return true;
    }
    if (compute.heatSystemComputeController() &&
        compute.heatSystemComputeController()->hasAnyRetiring()) {
        return true;
    }
    if (compute.cfdSystemComputeController() &&
        compute.cfdSystemComputeController()->hasAnyRetiring()) {
        return true;
    }
    return false;
}

void NodeGraphController::clearRetiringGenerations() {
    if (RuntimeProductManager* products = packageController.products()) {
        products->clearAllRetiring();
    }
    if (compute.modelComputeController()) compute.modelComputeController()->clearAllRetiring();
    if (compute.pointComputeRuntime()) compute.pointComputeRuntime()->clearAllRetiring();
    if (compute.remeshController()) compute.remeshController()->clearAllRetiring();
    if (compute.voronoiSystemComputeController()) compute.voronoiSystemComputeController()->clearAllRetiring();
    if (compute.heatSystemComputeController()) compute.heatSystemComputeController()->clearAllRetiring();
    if (compute.cfdSystemComputeController()) compute.cfdSystemComputeController()->clearAllRetiring();
}

bool NodeGraphController::compileRuntimePackages() {
    runtime.execute(plan);
    const NodeGraphEvaluation& evaluation = runtime.evaluation();
    const RuntimePackageCompiler::FrozenPackages frozenPackages =
        frozenPackageRebuildPending
            ? RuntimePackageCompiler::FrozenPackages::Rebuild
            : RuntimePackageCompiler::FrozenPackages::Preserve;
    std::vector<std::string> messages;
    bool failed = false;
    if (!packageCompiler.validate(
            runtime.state(),
            plan,
            evaluation,
            payloadRegistry,
            messages)) {
        publishPackageErrors(messages);
        return false;
    }

    const RuntimeProductManager* products = packageController.products();
    if (!products) {
        return false;
    }
    packageController.beginCompilation();
    for (NodeGraphNodeId nodeId : plan.executionOrder) {
        const NodeGraphNode* node = runtime.state().node(nodeId);
        if (!node) {
            messages.push_back("Package schedule references a missing node.");
            failed = true;
            continue;
        }

        const std::string label = node->title.empty() ? node->typeId : node->title;
        for (const NodeGraphSocket& input : node->inputs) {
            if (!input.required) continue;
            if (!runtime.state().edges.incoming(node->id, input.id).empty()) continue;
            messages.push_back(label + ": input '" + input.name + "' is not connected.");
        }

        std::vector<std::string> nodeErrors;
        if (!packageCompiler.compileNode(
                runtime.state(),
                *node,
                evaluation,
                payloadRegistry,
                *products,
                activeWorldUnit,
                frozenPackages,
                packageController.packages(),
                nodeErrors)) {
            packageController.invalidateOutputsForNode(*node);
            messages.insert(messages.end(), nodeErrors.begin(), nodeErrors.end());
            failed = true;
            continue;
        }

        if (!packageController.applyNode(*node)) {
            packageController.invalidateOutputsForNode(*node);
            messages.push_back(label + ": runtime update failed; node invalidated.");
            failed = true;
        }
    }
    packageController.finishCompilation(/*success=*/true);
    publishPackageErrors(messages);
    if (failed) {
        return false;
    }
    frozenPackageRebuildPending = false;

    NodeGraphDebugCache::instance().update(
        runtime.state().revision,
        evaluation.outputsBySocket);
    return true;
}

void NodeGraphController::publishPackageErrors(const std::vector<std::string>& errors) {
    std::string joined;
    for (const std::string& error : errors) {
        if (!joined.empty()) joined += "; ";
        joined += error;
    }
    if (joined == lastPackageErrors_) return;
    lastPackageErrors_ = joined;
}

void NodeGraphController::setWorldUnit(units::LengthUnit unit) {
    if (unit == activeWorldUnit) return;
    activeWorldUnit = unit;
    packageCompilationPending = true;
    frozenPackageRebuildPending = true;
    ++packageCompileRevision;
}

void NodeGraphController::updateDisplayTransports() {
    const bool hasDisplayTransports = transports.modelDisplayTransport ||
        transports.remeshDisplayTransport ||
        transports.voronoiDisplayTransport ||
        transports.heatDisplayTransport ||
        transports.cfdDisplayTransport ||
        transports.pointDisplayTransport;
    if (!hasDisplayTransports) return;

    RuntimePackageManager& packages = packageController.packages();
    const std::unordered_set<uint64_t> visibleKeys =
        nodeGraphDisplay.computeDisplayKeys(
            runtime.state(),
            runtime.evaluation(),
            packages,
            &payloadRegistry);

    if (transports.modelDisplayTransport) {
        transports.modelDisplayTransport->sync(packages, visibleKeys);
        transports.modelDisplayTransport->finalizeSync();
    }
    if (transports.remeshDisplayTransport) {
        transports.remeshDisplayTransport->sync(packages, visibleKeys);
        transports.remeshDisplayTransport->finalizeSync();
    }
    if (transports.voronoiDisplayTransport) {
        transports.voronoiDisplayTransport->sync(packages, visibleKeys);
        transports.voronoiDisplayTransport->finalizeSync();
    }
    if (transports.heatDisplayTransport) {
        transports.heatDisplayTransport->sync(packages, visibleKeys);
        transports.heatDisplayTransport->finalizeSync();
    }
    if (transports.cfdDisplayTransport) {
        transports.cfdDisplayTransport->sync(packages, visibleKeys);
        transports.cfdDisplayTransport->finalizeSync();
    }
    if (transports.pointDisplayTransport) {
        transports.pointDisplayTransport->sync(packages, visibleKeys);
        transports.pointDisplayTransport->finalizeSync();
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
