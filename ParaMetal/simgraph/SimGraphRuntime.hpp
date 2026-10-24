#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "SimConnection.hpp"

class FrameSync;
class SimNode;
class VulkanDevice;

enum class SimStatus {
    Idle,
    Ok,
    Failed,
    DeviceLost
};

struct SimTickResult {
    SimStatus status = SimStatus::Idle;
    std::optional<float> computeGpuMs;
    uint64_t completedStepCount = 0;
};

// Owns simulation scheduling: one node registry, the connection set, a cached
// topological order, and recording/submission on the compute queue.
//
// Nodes are updated and recorded in topological order. Connections are bound to
// their consumer immediately before that consumer updates, and only when the
// producer has a publication available; the graph then attaches the producer's
// ready wait to the consumer's submission and signals the producer's release
// once, after the last consuming submission. Synchronization therefore follows
// actual per-connection consumption rather than node activity.
class SimGraphRuntime {
public:
    ~SimGraphRuntime();

    bool initialize(VulkanDevice& vulkanDevice, FrameSync& frameSync);
    void cleanup();

    // Registrations are package-driven and persist across frames. A socket key is
    // the identity of a node, so re-registering it with a new node replaces the
    // old one.
    void registerNode(uint64_t socketKey, SimNode* node);
    void unregisterNode(uint64_t socketKey);

    // Connections are the execution edges. The connection id is the identity of
    // an edge, so re-registering an id rewires that edge. Self-connections,
    // duplicate socket pairs and zero-copy mode are rejected.
    void connect(const SimConnection& connection);
    void disconnect(uint64_t connectionId);

    void setEnabled(bool enabledValue) { enabled = enabledValue; }

    SimTickResult tick(uint32_t frameIndex);

    void clear();

private:
    struct RegisteredNode {
        uint64_t socketKey = 0;
        SimNode* node = nullptr;
    };

    int findNodeIndex(uint64_t socketKey) const;
    bool hasPublication(const SimNode& node) const;
    bool rebuildGraph();
    void bindIncomingConnections(size_t nodeIndex);
    SimTickResult recordAndSubmit(uint32_t frameIndex);

    std::vector<RegisteredNode> nodes;
    std::vector<SimConnection> connections;

    // Cached execution state, rebuilt whenever registrations or connections change.
    std::unordered_map<uint64_t, size_t> nodeIndexBySocket;
    std::vector<size_t> topologicalOrder;
    std::vector<std::vector<size_t>> incomingConnectionsByNode;
    bool graphDirty = true;
    bool orderValid = true;

    bool enabled = true;

    VulkanDevice* vulkanDevice = nullptr;
    FrameSync* frameSync = nullptr;
};
