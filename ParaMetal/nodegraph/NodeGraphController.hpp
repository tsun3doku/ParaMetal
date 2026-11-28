#pragma once

#include "NodeGraphDisplay.hpp"
#include "NodeGraphRuntime.hpp"
#include "NodeInteraction.hpp"
#include "mesh/MeshElementType.hpp"
#include "runtime/RuntimeTransports.hpp"
#include "runtime/package/RuntimePackageCompiler.hpp"
#include "runtime/package/RuntimePackageController.hpp"
#include "util/Units.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

class ComputeContext;
class NodePayloadRegistry;
struct SubselectionEditState;

class NodeGraphController {
public:
    NodeGraphController(
        NodePayloadRegistry& payloadRegistry,
        ComputeContext& compute,
        const RuntimeTransports& transports);

    bool isValid() const { return runtime.isValid(); }
    void tick();
    void resetGraph(const NodeGraphState& state);
    bool applyGraphDelta(const NodeGraphDelta& delta);
    const NodeGraphState& graphState() const;
    const NodeGraphEvaluation& evaluation() const { return runtime.evaluation(); }
    const NodePayloadRegistry& payloads() const { return payloadRegistry; }
    bool beginMeshSelection(
        const NodeInteractionSession& session,
        SubselectionEditState& outState) const;
    bool meshSelectionIsCurrent(const SubselectionEditState& state) const;
    bool inspect(NodeGraphNodeId nodeId, NodeInspection& out) const;
    bool beginAction(
        NodeGraphNodeId nodeId,
        uint64_t nodeInstanceId,
        uint32_t actionId,
        NodeInteractionRequest& outRequest,
        NodeInteractionSession& outSession,
        std::string& error);
    bool completeAction(
        const NodeInteractionSession& session,
        const NodeInteractionResult& result,
        std::vector<NodeGraphParamValue>& outPatch,
        std::string& error);
    bool previewAction(
        const NodeInteractionSession& session,
        const NodeInteractionValue& authored,
        NodeInteractionValue& display,
        std::string& error) const;
    uint64_t runtimeModelRevision() const;
    uint64_t evaluationRevision() const { return evaluationRevisionCounter; }
    void setWorldUnit(units::LengthUnit unit);
    bool updateDisplayTransports();
    void detachRuntimeBindings();
    void quiescedClear();
    units::LengthUnit worldUnit() const { return activeWorldUnit; }
    bool runtimeCompilationPending() const {
        return packageCompilationPending || packageCompilationFailed;
    }
    bool runtimeCompilationFailed() const { return packageCompilationFailed; }
    const NodeGraphCompiled& compiledState() const;
    bool runtimeModelIdsForNode(NodeGraphNodeId nodeId, std::vector<uint32_t>& outIds) const;
    bool primaryDisplayNode(NodeGraphNodeId& outNodeId) const;
    bool resolveGizmoTransformNode(uint64_t outputSocketKey, NodeGraphNodeId& outNodeId) const;
    bool hasRetiringGenerations() const;
    void clearRetiringGenerations();

    const RuntimePackageController& getPackageController() const { return packageController; }
    const std::string& lastPackageErrors() const { return lastPackageErrors_; }
    const std::string& lastDisplayError() const { return packageController.lastDisplayError(); }

private:
    void rebuildForDelta(const NodeGraphDelta& delta);
    bool compileRuntimePackages();
    void publishPackageErrors(const std::vector<std::string>& errors);
    bool sourceSurfaceIdentity(
        uint64_t socketKey,
        uint64_t& outSurfaceIdentity) const;

    NodePayloadRegistry& payloadRegistry;
    ComputeContext& compute;
    NodeGraphRuntime runtime;
    bool packageCompilationPending = false;
    bool packageCompilationFailed = false;
    bool frozenPackageRebuildPending = false;
    uint64_t packageCompileRevision = 0;
    uint64_t evaluationRevisionCounter = 1;
    uint64_t lastDisplaySyncStateRevision = UINT64_MAX;
    uint64_t lastDisplaySyncProductRevision = UINT64_MAX;
    uint64_t lastDisplaySyncPackageRevision = UINT64_MAX;
    NodeGraphCompiled plan{};
    NodeGraphDisplay nodeGraphDisplay{};
    RuntimePackageCompiler packageCompiler{};
    RuntimePackageController packageController;
    units::LengthUnit activeWorldUnit = units::defaultLengthUnit();
    std::string lastPackageErrors_;
};
