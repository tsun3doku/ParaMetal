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
#include "NodeTransform.hpp"
#include "NodePayloadRegistry.hpp"
#include "domain/GeometryData.hpp"
#include "domain/RemeshData.hpp"
#include "mesh/SubselectionEditState.hpp"
#include "util/GeometryUtils.hpp"
#include "runtime/RuntimeHeatDisplayTransport.hpp"
#include "runtime/RuntimeCFDDisplayTransport.hpp"
#include "runtime/RuntimeModelDisplayTransport.hpp"
#include "runtime/RuntimePointDisplayTransport.hpp"
#include "runtime/RuntimeRemeshDisplayTransport.hpp"
#include "runtime/RuntimeVoronoiDisplayTransport.hpp"

#include <glm/gtc/matrix_inverse.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <unordered_set>
#include <utility>

NodeGraphController::NodeGraphController(
    NodePayloadRegistry& payloadRegistry,
    ComputeContext& computeContext,
    const RuntimeTransports& transportBindings)
    : payloadRegistry(payloadRegistry),
      compute(computeContext),
      runtime(&payloadRegistry),
      packageController(transportBindings) {
    plan.isValid = false;
}

static bool buildSelectionOverlayIndices(
    const GeometryData& geometry,
    std::vector<uint32_t>& outIndices) {
    if (geometry.renderIndices.empty()) {
        outIndices.clear();
        return true;
    }

    const std::size_t vertexCount = geometry.renderPositions.size() / 3;
    if (geometry.renderIndices.size() != geometry.triangleIndices.size() ||
        geometry.renderPositions.size() % 3 != 0 ||
        std::any_of(
            geometry.renderIndices.begin(),
            geometry.renderIndices.end(),
            [vertexCount](uint32_t index) {
                return index >= vertexCount;
            })) {
        return false;
    }

    outIndices = geometry.renderIndices;
    return true;
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
    packageCompilationFailed = false;
    NodeGraphDebugCache::instance().setState(runtime.state(), &payloadRegistry);
    plan = NodeGraphCompiler::compile(runtime.state());
}

void NodeGraphController::tick() {
    packageController.retryPendingRemovals();

    if (plan.isValid &&
        packageCompilationPending &&
        !packageController.hasRetiringGenerations() &&
        !packageController.hasPendingRemovals()) {
        packageCompilationPending = false;
        packageCompilationFailed = !compileRuntimePackages();
        ++evaluationRevisionCounter;
        // sync simulation nodes before sim graph run
        compute.syncSimulationNodes(packageController.packages());
    }

    const uint64_t stateRevision = runtime.state().revision;
    const uint64_t productRevision = packageController.publicationRevision();
    const bool fullDisplaySync =
        stateRevision != lastDisplaySyncStateRevision ||
        productRevision != lastDisplaySyncProductRevision ||
        packageCompileRevision != lastDisplaySyncPackageRevision;
    if (fullDisplaySync && updateDisplayTransports()) {
        lastDisplaySyncStateRevision = stateRevision;
        lastDisplaySyncProductRevision = productRevision;
        lastDisplaySyncPackageRevision = packageCompileRevision;
    }
}

void NodeGraphController::detachRuntimeBindings() {
    packageController.detachAllDisplayBindings();
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
    packageCompilationPending = false;
    packageCompilationFailed = false;
    lastDisplaySyncStateRevision = UINT64_MAX;
    lastDisplaySyncProductRevision = UINT64_MAX;
    lastDisplaySyncPackageRevision = UINT64_MAX;
    lastPackageErrors_.clear();
}

void NodeGraphController::resetGraph(const NodeGraphState& state) {
    NodeGraphDelta delta{};
    delta.graphInstanceId = state.instanceId;
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
    if (!canApplyNodeGraphDelta(runtime.state(), delta)) {
        return false;
    }
    rebuildForDelta(delta);
    return true;
}

const NodeGraphState& NodeGraphController::graphState() const {
    return runtime.state();
}

bool NodeGraphController::beginMeshSelection(
    const NodeInteractionSession& session,
    SubselectionEditState& outState) const {
    outState = {};
    if (session.sourceSocketKey == 0 || session.displaySocketKey == 0 ||
        session.runtimeModelId == 0) {
        return false;
    }

    const EvaluatedSocketValue* evaluated =
        runtime.evaluation().outputFor(session.sourceSocketKey);
    if (!evaluated || evaluated->status != EvaluatedSocketStatus::Value) {
        return false;
    }

    SubselectionEditState state{};
    state.sourceSocketKey = session.sourceSocketKey;
    state.displaySocketKey = session.displaySocketKey;
    state.binding = session.binding;
    state.runtimeModelId = session.runtimeModelId;

    if (session.binding == SurfaceBinding::Intrinsic) {
        NodeDataHandle remeshHandle{};
        const RemeshData* remesh = payloadRegistry.resolveRemesh(
            evaluated->data.payloadHandle, &remeshHandle);
        if (!remesh ||
            remesh->surfacePositions.empty() ||
            remesh->surfaceTriangleIndices.empty() ||
            remesh->surfaceTriangleIndices.size() % 3 != 0 ||
            remesh->surfaceFaceIds.size() !=
                remesh->surfaceTriangleIndices.size() / 3 ||
            remesh->surfaceVertexIds.size() != remesh->surfacePositions.size() ||
            remesh->surfaceEdgeIds.empty() ||
            !std::is_sorted(
                remesh->surfaceEdgeIds.begin(), remesh->surfaceEdgeIds.end()) ||
            std::adjacent_find(
                remesh->surfaceEdgeIds.begin(), remesh->surfaceEdgeIds.end()) !=
                remesh->surfaceEdgeIds.end() ||
            (session.sourceSurfaceIdentity != 0 &&
             session.sourceSurfaceIdentity != remesh->surfaceGeometryHash)) {
            return false;
        }

        state.surfaceIdentity = remesh->surfaceGeometryHash;
        state.pickGeometryIdentity = remesh->surfaceGeometryHash;
        state.faceCount =
            static_cast<uint32_t>(remesh->surfaceTriangleIndices.size() / 3);
        state.vertexPositions = remesh->surfacePositions;
        state.triangleIndices = remesh->surfaceTriangleIndices;
        state.surfaceFaceIds = remesh->surfaceFaceIds;
        state.surfaceVertexIds = remesh->surfaceVertexIds;
        state.surfaceEdgeIds = remesh->surfaceEdgeIds;
    } else {
        NodeDataHandle geometryHandle{};
        const GeometryData* geometry = payloadRegistry.resolveGeometry(
            evaluated->data.payloadHandle, &geometryHandle);
        if (!geometry ||
            geometry->vertexPositions.empty() ||
            geometry->triangleIndices.empty() ||
            geometry->triangleIndices.size() % 3 != 0) {
            return false;
        }

        const uint64_t geometryIdentity =
            payloadRegistry.resolveHash(
                geometryHandle, HashDomain::Geometry);
        if (geometryIdentity == 0 ||
            (session.sourceSurfaceIdentity != 0 &&
             session.sourceSurfaceIdentity != geometryIdentity)) {
            return false;
        }

        const float canonicalToWorld =
            units::canonicalToWorldScale(activeWorldUnit);

        state.surfaceIdentity = geometryIdentity;
        state.faceCount =
            static_cast<uint32_t>(geometry->triangleIndices.size() / 3);
        state.vertexPositions.reserve(geometry->vertexPositions.size() / 3);
        for (std::size_t index = 0;
             index + 2 < geometry->vertexPositions.size();
             index += 3) {
            state.vertexPositions.emplace_back(
                geometry->vertexPositions[index] * canonicalToWorld,
                geometry->vertexPositions[index + 1] * canonicalToWorld,
                geometry->vertexPositions[index + 2] * canonicalToWorld);
        }
        state.triangleIndices = geometry->triangleIndices;
        if (!buildSelectionOverlayIndices(*geometry, state.overlayIndices)) {
            return false;
        }
    }

    std::vector<uint32_t> modelIds;
    if (!packageController.appendRuntimeModelIdsForSocket(
            state.displaySocketKey, modelIds) ||
        modelIds.size() != 1 ||
        modelIds.front() != state.runtimeModelId) {
        return false;
    }

    outState = std::move(state);
    return true;
}

bool NodeGraphController::meshSelectionIsCurrent(
    const SubselectionEditState& state) const {
    if (!packageController.isDisplayActive(
            state.displaySocketKey, state.surfaceIdentity)) {
        return false;
    }

    uint64_t surfaceIdentity = 0;
    if (!sourceSurfaceIdentity(
            state.sourceSocketKey, surfaceIdentity) ||
        surfaceIdentity != state.surfaceIdentity) {
        return false;
    }

    std::vector<uint32_t> modelIds;
    if (!packageController.appendRuntimeModelIdsForSocket(
            state.displaySocketKey, modelIds) ||
        modelIds.size() != 1 ||
        modelIds.front() != state.runtimeModelId) {
        return false;
    }

    return true;
}

bool NodeGraphController::inspect(
    NodeGraphNodeId nodeId,
    NodeInspection& out) const {
    out = {};
    const NodeGraphNode* node = runtime.state().node(nodeId);
    if (!node) {
        return false;
    }
    const NodeKernel* kernel = runtime.kernelFor(node->typeId);
    if (!kernel) {
        return false;
    }

    const NodeAuthoringContext context{
        *node,
        runtime.state(),
        runtime.evaluation(),
        payloadRegistry,
    };
    kernel->inspect(context, out);
    return true;
}

bool NodeGraphController::beginAction(
    NodeGraphNodeId nodeId,
    uint64_t nodeInstanceId,
    uint32_t actionId,
    NodeInteractionRequest& outRequest,
    NodeInteractionSession& outSession,
    std::string& error) {
    outRequest = {};
    outSession = {};
    error.clear();

    const NodeGraphNode* node = runtime.state().node(nodeId);
    if (!node || node->instanceId != nodeInstanceId) {
        error = "Interaction target changed.";
        return false;
    }
    const NodeKernel* kernel = runtime.kernelFor(node->typeId);
    if (!kernel) {
        error = "Node does not provide interactions.";
        return false;
    }

    const NodeAuthoringContext context{
        *node,
        runtime.state(),
        runtime.evaluation(),
        payloadRegistry,
    };
    NodeInteractionRequest request{};
    if (!kernel->beginInteraction(
            context, actionId, request, error)) {
        return false;
    }

    NodeInteractionSession session{};
    session.nodeId = node->id;
    session.nodeInstanceId = node->instanceId;
    session.actionId = actionId;
    session.type = request.type;
    session.binding = request.binding;
    session.sourceSocketKey = request.sourceSocketKey;
    session.displaySocketKey = request.displaySocketKey;
    if (session.sourceSocketKey != 0 &&
        !sourceSurfaceIdentity(
            session.sourceSocketKey,
            session.sourceSurfaceIdentity)) {
        error = "Interaction source is not available.";
        return false;
    }

    if (request.type == NodeInteractionType::MeshSubselection) {
        std::vector<uint32_t> modelIds;
        if (!packageController.appendRuntimeModelIdsForSocket(
                session.displaySocketKey, modelIds) ||
            modelIds.size() != 1) {
            error = "Interaction requires one current display model.";
            return false;
        }
        session.runtimeModelId = modelIds.front();

        RuntimePackageController::DisplayedSurface surface{};
        if (!packageController.findDisplayedSurface(
                session.runtimeModelId, surface) ||
            surface.intrinsic !=
                (session.binding == SurfaceBinding::Intrinsic)) {
            error = "Interaction surface is not displayed.";
            return false;
        }
        session.displaySocketKey = surface.socketKey;
    }

    outRequest = std::move(request);
    outSession = session;
    return true;
}

bool NodeGraphController::completeAction(
    const NodeInteractionSession& session,
    const NodeInteractionResult& result,
    std::vector<NodeGraphParamValue>& outPatch,
    std::string& error) {
    outPatch.clear();
    error.clear();
    if (result.sessionId != session.id) {
        error = "Interaction result does not match the active session.";
        return false;
    }

    const NodeGraphNode* node = runtime.state().node(session.nodeId);
    if (!node || node->instanceId != session.nodeInstanceId) {
        error = "Interaction target changed.";
        return false;
    }
    const NodeKernel* kernel = runtime.kernelFor(node->typeId);
    if (!kernel) {
        error = "Node does not provide interactions.";
        return false;
    }

    if (session.sourceSocketKey != 0) {
        uint64_t currentSourceSurfaceIdentity = 0;
        if (!sourceSurfaceIdentity(
                session.sourceSocketKey,
                currentSourceSurfaceIdentity) ||
            currentSourceSurfaceIdentity != session.sourceSurfaceIdentity) {
            error = "Interaction source changed during editing.";
            return false;
        }
    }

    const NodeAuthoringContext context{
        *node,
        runtime.state(),
        runtime.evaluation(),
        payloadRegistry,
    };
    return kernel->completeInteraction(
        context, session, result, outPatch, error);
}

bool NodeGraphController::previewAction(
    const NodeInteractionSession& session,
    const NodeInteractionValue& authored,
    NodeInteractionValue& display,
    std::string& error) const {
    error.clear();

    const NodeGraphNode* node = runtime.state().node(session.nodeId);
    if (!node || node->instanceId != session.nodeInstanceId) {
        error = "Interaction target changed.";
        return false;
    }

    uint64_t currentSourceSurfaceIdentity = 0;
    if (session.sourceSocketKey &&
        (!sourceSurfaceIdentity(
             session.sourceSocketKey, currentSourceSurfaceIdentity) ||
         currentSourceSurfaceIdentity != session.sourceSurfaceIdentity)) {
        error = "Interaction source changed.";
        return false;
    }

    const NodeKernel* kernel = runtime.kernelFor(node->typeId);
    if (!kernel) {
        error = "Interaction kernel is unavailable.";
        return false;
    }

    const NodeAuthoringContext context{
        *node,
        runtime.state(),
        runtime.evaluation(),
        payloadRegistry,
    };
    return kernel->previewInteraction(
        context, session, authored, display, error);
}

uint64_t NodeGraphController::runtimeModelRevision() const {
    return packageController.publicationRevision();
}


bool NodeGraphController::resolveGizmoTransformNode(
    uint64_t outputSocketKey,
    NodeGraphNodeId& outNodeId) const {
    outNodeId = {};
    return outputSocketKey != 0 && findFirstUpstreamNodeByType(runtime.state(), outputSocketKey, nodegraphtypes::Transform, outNodeId);
}

bool NodeGraphController::hasRetiringGenerations() const {
    return packageController.hasRetiringGenerations();
}

void NodeGraphController::clearRetiringGenerations() {
    packageController.clearRetiringProducts();
    compute.clearRetiringGenerations();
    packageController.retryPendingRemovals();
}

bool NodeGraphController::compileRuntimePackages() {
    assert(!packageController.hasRetiringGenerations());
    assert(!packageController.hasPendingRemovals());
    runtime.beginEvaluation(plan);
    const NodeGraphEvaluation& evaluation = runtime.evaluation();
    const RuntimePackageCompiler::FrozenPackages frozenPackages =
        frozenPackageRebuildPending
            ? RuntimePackageCompiler::FrozenPackages::Rebuild
            : RuntimePackageCompiler::FrozenPackages::Preserve;
    std::vector<std::string> messages;
    bool failed = false;
    bool backpressured = false;

    packageController.beginCompilation();
    for (NodeGraphNodeId nodeId : plan.executionOrder) {
        const NodeGraphNode* node = runtime.state().node(nodeId);
        if (!node) {
            messages.push_back("Package schedule references a missing node.");
            failed = true;
            continue;
        }

        const std::string label = node->title.empty() ? node->typeId : node->title;
        std::vector<std::string> missingInputs;
        for (const NodeGraphSocket& input : node->inputs) {
            if (!input.required) continue;
            if (!runtime.state().edges.incoming(node->id, input.id).empty()) continue;
            missingInputs.push_back(
                label + ": input '" + input.name + "' is not connected.");
        }
        if (!missingInputs.empty()) {
            messages.insert(messages.end(), missingInputs.begin(), missingInputs.end());
            packageController.invalidateOutputsForNode(*node);
            runtime.failNode(nodeId, missingInputs.front());
            failed = true;
            continue;
        }

        runtime.evaluateNode(nodeId);
        if (runtime.nodeBackpressured(nodeId)) {
            packageCompilationPending = true;
            backpressured = true;
            break;
        }

        std::vector<std::string> nodeErrors;
        if (!packageCompiler.executeNode(
                runtime.state(), *node, runtime, payloadRegistry,
                activeWorldUnit, frozenPackages, packageController,
                nodeErrors, backpressured)) {
            if (backpressured) {
                packageCompilationPending = true;
                break;
            }
            packageController.invalidateOutputsForNode(*node);
            runtime.failNode(nodeId, nodeErrors.empty() ? "Node execution failed." : nodeErrors.front());
            messages.insert(messages.end(), nodeErrors.begin(), nodeErrors.end());
            failed = true;
        }
    }
    if (backpressured) {
        packageController.abortCompilationPreservingUnprocessed();
    } else {
        packageController.finishCompilation(!failed);
    }
    publishPackageErrors(messages);
    if (backpressured) {
        return true;
    }
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
    lastPackageErrors_ = std::move(joined);
}

void NodeGraphController::setWorldUnit(units::LengthUnit unit) {
    if (unit == activeWorldUnit) return;
    activeWorldUnit = unit;
    packageCompilationPending = true;
    packageCompilationFailed = false;
    frozenPackageRebuildPending = true;
    ++packageCompileRevision;
}

bool NodeGraphController::updateDisplayTransports() {
    return packageController.syncDisplay(
        nodeGraphDisplay.computeDisplayKeys(
            runtime.state(),
            runtime.evaluation(),
            packageController.packages(),
            &payloadRegistry));
}

const NodeGraphCompiled& NodeGraphController::compiledState() const {
    return plan;
}

bool NodeGraphController::sourceSurfaceIdentity(
    uint64_t socketKey,
    uint64_t& outSurfaceIdentity) const {
    outSurfaceIdentity = 0;

    const EvaluatedSocketValue* evaluated =
        runtime.evaluation().outputFor(socketKey);
    if (!evaluated ||
        evaluated->status != EvaluatedSocketStatus::Value) {
        return false;
    }

    NodeDataHandle geometryHandle{};
    if (payloadRegistry.resolveGeometry(
            evaluated->data.payloadHandle, &geometryHandle)) {
        outSurfaceIdentity = payloadRegistry.resolveHash(
            geometryHandle, HashDomain::Geometry);
        return outSurfaceIdentity != 0;
    }

    const RemeshData* remesh = payloadRegistry.resolveRemesh(
        evaluated->data.payloadHandle);
    if (!remesh || remesh->surfaceGeometryHash == 0) {
        return false;
    }
    outSurfaceIdentity = remesh->surfaceGeometryHash;
    return true;
}

bool NodeGraphController::runtimeModelIdsForNode(
    NodeGraphNodeId nodeId,
    std::vector<uint32_t>& outIds) const {
    outIds.clear();
    const auto nodeIt = runtime.state().nodes.find(nodeId.value);
    if (nodeIt == runtime.state().nodes.end()) return false;

    for (const NodeGraphSocket& output : nodeIt->second.outputs) {
        const uint64_t socketKey = NodeSocketKey(nodeIt->second.id, output.id).value;
        packageController.appendRuntimeModelIdsForSocket(socketKey, outIds);
    }
    return !outIds.empty();
}

bool NodeGraphController::primaryDisplayNode(NodeGraphNodeId& outNodeId) const {
    outNodeId = {};
    for (const auto& [id, node] : runtime.state().nodes) {
        (void)id;
        if (!node.state.isPrimaryDisplay()) {
            continue;
        }
        if (outNodeId.isValid()) {
            outNodeId = {};
            return false;
        }
        outNodeId = node.id;
    }
    return outNodeId.isValid();
}

