#pragma once

#include <cstdint>

struct NodeGraphNodeId {
    uint32_t value = 0;
    bool isValid() const {
        return value != 0;
    }
};

struct NodeGraphSocketId {
    uint32_t value = 0;
    bool isValid() const {
        return value != 0;
    }
};

struct NodeGraphEdgeId {
    uint32_t value = 0;
    bool isValid() const {
        return value != 0;
    }
};

inline bool operator==(NodeGraphNodeId lhs, NodeGraphNodeId rhs) {
    return lhs.value == rhs.value;
}

inline bool operator!=(NodeGraphNodeId lhs, NodeGraphNodeId rhs) {
    return !(lhs == rhs);
}

inline bool operator==(NodeGraphSocketId lhs, NodeGraphSocketId rhs) {
    return lhs.value == rhs.value;
}

inline bool operator!=(NodeGraphSocketId lhs, NodeGraphSocketId rhs) {
    return !(lhs == rhs);
}

inline bool operator==(NodeGraphEdgeId lhs, NodeGraphEdgeId rhs) {
    return lhs.value == rhs.value;
}

inline bool operator!=(NodeGraphEdgeId lhs, NodeGraphEdgeId rhs) {
    return !(lhs == rhs);
}

struct NodeSocketKey {
    uint64_t value = 0;
    NodeSocketKey() = default;
    NodeSocketKey(uint64_t v) : value(v) {}
    NodeSocketKey(NodeGraphNodeId n, NodeGraphSocketId s)
        : value((static_cast<uint64_t>(n.value) << 32) | s.value) {}
    NodeGraphNodeId node() const {
        return {static_cast<uint32_t>(value >> 32)};
    }
    NodeGraphSocketId socket() const {
        return {static_cast<uint32_t>(value & 0xFFFFFFFF)};
    }
    operator uint64_t() const {
        return value;
    }
};

struct NodeDataHandle {
    uint64_t key = 0;

    bool operator==(const NodeDataHandle& other) const {
        return key == other.key;
    }

    bool operator<(const NodeDataHandle& other) const {
        return key < other.key;
    }
};
