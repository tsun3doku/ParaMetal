#pragma once

#include "NodeGraphDisplay.hpp"
#include "NodeGraphRuntime.hpp"
#include "runtime/RuntimeTransports.hpp"
#include "runtime/package/RuntimePackageCompiler.hpp"
#include "runtime/package/RuntimePackageController.hpp"
#include "util/Units.hpp"

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

class ComputeContext;
class NodePayloadRegistry;

class NodeGraphController {
public:
    NodeGraphController(
        NodePayloadRegistry& payloadRegistry,
        ComputeContext& compute,
        const RuntimeTransports& transports);

    void tick();
    void resetGraph(const NodeGraphState& state);
    bool applyGraphDelta(const NodeGraphDelta& delta);
    const NodeGraphState& graphState() const;
    const NodeGraphEvaluation& evaluation() const { return runtime.evaluation(); }
    const NodePayloadRegistry& payloads() const { return payloadRegistry; }
    bool resolveGizmoTransformNode(uint64_t outputSocketKey, NodeGraphNodeId& outNodeId) const;
    void updateDisplayTransports();
    void detachRuntimeBindings();
    void quiescedClear();
    void setWorldUnit(units::LengthUnit unit);
    units::LengthUnit worldUnit() const { return activeWorldUnit; }
    bool runtimeCompilationPending() const { return packageCompilationPending; }
    const NodeGraphCompiled& compiledState() const;
    bool runtimeModelIdsForNode(NodeGraphNodeId nodeId, std::vector<uint32_t>& outIds) const;
    bool hasRetiringGenerations() const;
    void clearRetiringGenerations();

    RuntimeProductManager* getProductManager() { return packageController.products(); }
    const RuntimeProductManager* getProductManager() const { return packageController.products(); }
    RuntimePackageManager* getPackageManager() { return &packageController.packages(); }
    const RuntimePackageManager* getPackageManager() const { return &packageController.packages(); }
    const std::string& lastPackageErrors() const { return lastPackageErrors_; }

private:
    void rebuildForDelta(const NodeGraphDelta& delta);
    void syncSimulationNodes();
    bool compileRuntimePackages();
    void publishPackageErrors(const std::vector<std::string>& errors);
    bool runtimeModelIdsForSocket(uint64_t socketKey, std::vector<uint32_t>& outIds) const;
    void addRuntimeModelId(std::vector<uint32_t>& outIds, uint32_t id) const;

    NodePayloadRegistry& payloadRegistry;
    ComputeContext& compute;
    RuntimeTransports transports{};
    NodeGraphRuntime runtime;
    bool packageCompilationPending = false;
    bool frozenPackageRebuildPending = false;
    uint64_t packageCompileRevision = 0;
    uint64_t lastDisplaySyncStateRevision = UINT64_MAX;
    uint64_t lastDisplaySyncProductRevision = UINT64_MAX;
    uint64_t lastDisplaySyncPackageRevision = UINT64_MAX;
    NodeGraphCompiled plan{};
    NodeGraphDisplay nodeGraphDisplay{};
    RuntimePackageCompiler packageCompiler{};
    RuntimePackageController packageController;
    std::unordered_set<uint64_t> simulationSockets;
    units::LengthUnit activeWorldUnit = units::defaultLengthUnit();
    std::string lastPackageErrors_;
};
