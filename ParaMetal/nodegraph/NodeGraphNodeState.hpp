#pragma once

#include "NodeGraphCoreTypes.hpp"

#include <cstdint>
#include <unordered_map>

struct NodeGraphNode;
class NodeGraphNodeState {
public:
    enum struct FrozenState {
        Live,
        Frozen
    };

    enum struct DisplayState {
        Hidden,
        Primary
    };

    struct Flags {
        bool frozenEnabled = false;
        bool displayEnabled = false;
    };

    FrozenState frozenState() const;
    DisplayState displayState() const;

    bool isFrozen() const;
    bool isPrimaryDisplay() const;

    const Flags& flags() const;
    bool restore(const Flags& flags);
    bool setFrozenState(FrozenState state);
    bool setDisplayState(DisplayState state);

    static bool setDisplayState(
        std::unordered_map<uint32_t, NodeGraphNode>& nodes,
        NodeGraphNodeId nodeId,
        DisplayState displayState);
    static bool toggleFrozen(std::unordered_map<uint32_t, NodeGraphNode>& nodes, NodeGraphNodeId nodeId);
    static bool toggleDisplay(std::unordered_map<uint32_t, NodeGraphNode>& nodes, NodeGraphNodeId nodeId);

    bool operator==(const NodeGraphNodeState& other) const;
    bool operator!=(const NodeGraphNodeState& other) const;

private:
    static bool set(bool& flag, bool enabled);

    Flags values{};
};
