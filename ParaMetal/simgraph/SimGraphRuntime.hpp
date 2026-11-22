#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "SimConnection.hpp"
#include "SimNode.hpp"

class FrameSync;
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

class SimGraphRuntime {
public:
    ~SimGraphRuntime();

    bool initialize(VulkanDevice& vulkanDevice, FrameSync& frameSync);
    void cleanup();

    void registerNode(uint64_t socketKey, SimNode* node);
    void unregisterNode(uint64_t socketKey);

    void connect(const SimConnection& connection);
    void disconnect(uint64_t connectionId);

    void setEnabled(bool enabledValue) { enabled = enabledValue; }

    SimTickResult tick(const SimTickContext& context);

    size_t registeredNodeCount() const { return nodes.size(); }
    bool hasNodeInState(SimNodeState state) const;
    SimNodeState getAggregateSimulationState() const;
    bool hasSimulationTimelineMetadata() const;
    float getSimulationTotalTime() const;
    float getSimulationDuration() const;
    uint32_t getSimulationTimelineFrameCount() const;
    uint32_t getSimulationRecordedTimelineFrames() const;
    uint32_t getPlaybackAvailableFrame() const;

    void clear();

private:
    struct RegisteredNode {
        uint64_t socketKey = 0;
        SimNode* node = nullptr;
    };

    int findNodeIndex(uint64_t socketKey) const;
    bool hasPublication(const SimNode& node) const;
    uint64_t simulationStepTotal() const;
    bool rebuildGraph();
    void bindIncomingConnections(size_t nodeIndex);
    SimTickResult recordAndSubmit(uint32_t renderFrameSlot);

    uint64_t tickCounter = 0;

    std::vector<RegisteredNode> nodes;
    std::vector<SimConnection> connections;

    std::unordered_map<uint64_t, size_t> nodeIndexBySocket;
    std::vector<size_t> topologicalOrder;
    std::vector<std::vector<size_t>> incomingConnectionsByNode;
    bool graphDirty = true;
    bool orderValid = true;

    bool enabled = true;

    VulkanDevice* vulkanDevice = nullptr;
    FrameSync* frameSync = nullptr;
};
