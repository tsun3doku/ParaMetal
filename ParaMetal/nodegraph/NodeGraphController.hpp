#pragma once

#include "NodeGraphDisplay.hpp"
#include "NodeGraphCompiler.hpp"
#include "NodeGraphRuntime.hpp"
#include "runtime/RuntimePackageManager.hpp"
#include "runtime/RuntimeProductManager.hpp"

#include <cstdint>
#include <memory>
#include <vector>

class VulkanDevice;
class MemoryAllocator;

class NodeGraphController {
public:
    explicit NodeGraphController(const NodeRuntimeServices& services = {});

    void tick();
    void resetGraph(const NodeGraphState& state);
    bool applyGraphDelta(const NodeGraphDelta& delta);
    const NodeGraphState& graphState() const;
    bool resolveGizmoTransformNode(uint64_t outputSocketKey, NodeGraphNodeId& outNodeId) const;
    void updateDisplayTransports();
    const NodeGraphCompiled& compiledState() const;

    bool runtimeModelIdsForNode(NodeGraphNodeId nodeId,
                                 std::vector<uint32_t>& outIds) const;

    RuntimeProductManager* getProductManager() { return productManager.get(); }
    RuntimePackageManager* getPackageManager() { return &packageManager; }

private:
    void rebuildForDelta(const NodeGraphDelta& delta);
    void compileRuntimePackages();
    void updateComputeTransport(uint64_t socketKey);
    void updateComputeTransports(const NodeGraphNode& node);
    bool runtimeModelIdsForSocket(uint64_t socketKey,
                                   std::vector<uint32_t>& outIds) const;
    void addRuntimeModelId(std::vector<uint32_t>& outIds, uint32_t id) const;

    NodeRuntimeServices runtimeServices{};
    NodeGraphRuntime runtime;
    uint64_t pendingPackageRevision = 0;  
    uint64_t completedPackageRevision = 0; 
    NodeGraphCompiled plan{};
    NodeGraphDisplay nodeGraphDisplay{};
    RuntimePackageManager packageManager{};
    std::unique_ptr<RuntimeProductManager> productManager;
};
