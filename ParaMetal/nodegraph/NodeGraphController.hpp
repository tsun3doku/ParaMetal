#pragma once

#include "NodeGraphDisplay.hpp"
#include "NodeGraphCompiler.hpp"
#include "NodeGraphRuntime.hpp"
#include "runtime/RuntimePackageManager.hpp"
#include "runtime/RuntimeProductManager.hpp"

#include <cstdint>
#include <memory>

class NodeGraph;
class VulkanDevice;
class MemoryAllocator;

class NodeGraphController {
public:
    NodeGraphController(
        NodeGraph* bridge = nullptr,
        const NodeRuntimeServices& services = {});

    void tick();
    void updateDisplayTransports();
    const NodeGraphCompiled& compiledState() const;

    RuntimeProductManager* getProductManager() { return productManager.get(); }
    RuntimePackageManager* getPackageManager() { return &packageManager; }

private:
    void consumePendingGraphDelta();
    void compileRuntimePackages();
    void updateComputeTransport(uint64_t socketKey);
    void updateComputeTransports(const NodeGraphNode& node);

    NodeGraph* bridge = nullptr;
    NodeRuntimeServices runtimeServices{};
    NodeGraphRuntime runtime;
    uint64_t revisionSeen = 0;
    uint64_t pendingPackageRevision = 0;  
    uint64_t completedPackageRevision = 0; 
    NodeGraphCompiled plan{};
    NodeGraphDisplay nodeGraphDisplay{};
    RuntimePackageManager packageManager{};
    std::unique_ptr<RuntimeProductManager> productManager;
};
