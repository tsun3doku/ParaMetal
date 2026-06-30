#include "NodeGraphNodeState.hpp"

#include "NodeGraphTypes.hpp"

NodeGraphNodeState::FrozenState NodeGraphNodeState::frozenState() const {
    return values.frozenEnabled ? FrozenState::Frozen : FrozenState::Live;
}

NodeGraphNodeState::DisplayState NodeGraphNodeState::displayState() const {
    return values.displayEnabled ? DisplayState::Primary : DisplayState::Hidden;
}

bool NodeGraphNodeState::isFrozen() const {
    return frozenState() == FrozenState::Frozen;
}

bool NodeGraphNodeState::isPrimaryDisplay() const {
    return displayState() == DisplayState::Primary;
}

const NodeGraphNodeState::Flags& NodeGraphNodeState::flags() const {
    return values;
}

bool NodeGraphNodeState::restore(const Flags& flags) {
    bool changed = false;
    changed = set(values.frozenEnabled, flags.frozenEnabled) || changed;
    changed = set(values.displayEnabled, flags.displayEnabled) || changed;
    return changed;
}

bool NodeGraphNodeState::setFrozenState(FrozenState state) {
    return set(values.frozenEnabled, state == FrozenState::Frozen);
}

bool NodeGraphNodeState::setDisplayState(DisplayState state) {
    return set(values.displayEnabled, state == DisplayState::Primary);
}

bool NodeGraphNodeState::setDisplayState(
    std::unordered_map<uint32_t, NodeGraphNode>& nodes,
    NodeGraphNodeId nodeId,
    DisplayState displayState) {
    bool changed = false;
    if (displayState == DisplayState::Primary) {
        for (auto& [id, node] : nodes) {
            const DisplayState nodeDisplayState = (node.id == nodeId) ? DisplayState::Primary : DisplayState::Hidden;
            changed = node.state.setDisplayState(nodeDisplayState) || changed;
        }
        return changed;
    }

    const auto nodeIt = nodes.find(nodeId.value);
    if (nodeIt == nodes.end()) {
        return false;
    }
    return nodeIt->second.state.setDisplayState(DisplayState::Hidden);
}

bool NodeGraphNodeState::toggleFrozen(std::unordered_map<uint32_t, NodeGraphNode>& nodes, NodeGraphNodeId nodeId) {
    const auto nodeIt = nodes.find(nodeId.value);
    if (nodeIt == nodes.end()) {
        return false;
    }

    const FrozenState nextState = nodeIt->second.state.isFrozen() ? FrozenState::Live : FrozenState::Frozen;
    return nodeIt->second.state.setFrozenState(nextState);
}

bool NodeGraphNodeState::toggleDisplay(std::unordered_map<uint32_t, NodeGraphNode>& nodes, NodeGraphNodeId nodeId) {
    const auto nodeIt = nodes.find(nodeId.value);
    if (nodeIt == nodes.end()) {
        return false;
    }

    const DisplayState nextState = nodeIt->second.state.isPrimaryDisplay() ? DisplayState::Hidden : DisplayState::Primary;
    return setDisplayState(nodes, nodeId, nextState);
}

bool NodeGraphNodeState::operator==(const NodeGraphNodeState& other) const {
    return values.frozenEnabled == other.values.frozenEnabled &&
           values.displayEnabled == other.values.displayEnabled;
}

bool NodeGraphNodeState::operator!=(const NodeGraphNodeState& other) const {
    return !(*this == other);
}

bool NodeGraphNodeState::set(bool& flag, bool enabled) {
    if (flag == enabled) {
        return false;
    }

    flag = enabled;
    return true;
}
