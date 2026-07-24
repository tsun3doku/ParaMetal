#pragma once

#include "NodeGraphCoreTypes.hpp"
#include "NodeGraphNodeState.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

enum class NodeGraphValueType : uint8_t {
    None,
    Mesh,
    Remesh,
    HeatModel,
    Points,
    Volume,
    Field,
    Vector3,
    ScalarFloat,
    ScalarInt,
    ScalarBool
};

enum class NodeGraphSocketDirection {
    Input,
    Output
};

enum class NodeGraphNodeCategory {
    Geometry,
    Meshing,
    System,
    Misc
};

using NodeTypeId = std::string;

struct NodeGraphSocketContract {
    uint8_t producedPayloadType = 0;
};

struct NodeSocketSignature {
    std::string name;
    NodeGraphSocketDirection direction = NodeGraphSocketDirection::Input;
    NodeGraphValueType valueType = NodeGraphValueType::None;
    std::vector<NodeGraphValueType> acceptedValueTypes;
    NodeGraphSocketContract contract;
    bool variadic = false;
    bool required = true;
};

enum class NodeGraphParamType {
    Float,
    Int,
    Bool,
    String,
    Enum,
    Struct,
    Array
};

struct NodeGraphParamDefinition;

struct NodeGraphParamField {
    std::string name;
    std::shared_ptr<NodeGraphParamDefinition> definition;
};

struct NodeGraphParamDefinition {
    uint32_t id = 0;
    std::string name;
    NodeGraphParamType type = NodeGraphParamType::Float;
    double defaultFloatValue = 0.0;
    int64_t defaultIntValue = 0;
    bool defaultBoolValue = false;
    std::string defaultStringValue;
    bool isAction = false;
    std::vector<std::string> enumOptions;
    std::vector<NodeGraphParamField> fields;
    std::shared_ptr<NodeGraphParamDefinition> elementDefinition;
};

struct NodeGraphParamValue;

struct NodeGraphParamFieldValue {
    std::string name;
    std::shared_ptr<NodeGraphParamValue> value;
};

struct NodeGraphParamValue {
    uint32_t id = 0;
    NodeGraphParamType type = NodeGraphParamType::Float;
    double floatValue = 0.0;
    int64_t intValue = 0;
    bool boolValue = false;
    std::string stringValue;
    std::string enumValue;
    std::vector<NodeGraphParamFieldValue> fieldValues;
    std::vector<NodeGraphParamValue> arrayValues;
};

struct NodeTypeDefinition {
    NodeTypeId id;
    std::string displayName;
    NodeGraphNodeCategory category = NodeGraphNodeCategory::Misc;
    std::vector<NodeSocketSignature> sockets;
    std::vector<NodeGraphParamDefinition> parameters;
};

struct NodeGraphSocket : NodeSocketSignature {
    NodeGraphSocketId id{};

    NodeGraphSocket() = default;
    NodeGraphSocket(NodeGraphSocketId socketId, const NodeSocketSignature& sig)
        : NodeSocketSignature(sig), id(socketId) {}
};

struct NodeGraphNode {
    NodeGraphNodeId id{};
    NodeTypeId typeId;
    NodeGraphNodeCategory category = NodeGraphNodeCategory::Misc;
    std::string title;
    float x = 0.0f;
    float y = 0.0f;
    NodeGraphNodeState state{};
    std::vector<NodeGraphSocket> inputs;
    std::vector<NodeGraphSocket> outputs;
    std::vector<NodeGraphParamValue> parameters;

    const NodeGraphSocket* input(const char* name) const {
        for (const auto& s : inputs) {
            if (s.name == name) return &s;
        }
        return nullptr;
    }
    const NodeGraphSocket* input(NodeGraphValueType valueType) const {
        for (const auto& s : inputs) {
            if (s.valueType == valueType) return &s;
        }
        return nullptr;
    }
    const NodeGraphSocket* output(NodeGraphValueType valueType) const {
        for (const auto& s : outputs) {
            if (s.valueType == valueType) return &s;
        }
        return nullptr;
    }
    const NodeGraphSocket* input(NodeGraphSocketId socketId) const {
        for (const auto& s : inputs) {
            if (s.id == socketId) return &s;
        }
        return nullptr;
    }
    const NodeGraphSocket* output(NodeGraphSocketId socketId) const {
        for (const auto& s : outputs) {
            if (s.id == socketId) return &s;
        }
        return nullptr;
    }
    const NodeGraphSocket* outputOfType(uint8_t payloadType) const {
        for (const auto& s : outputs) {
            if (s.direction == NodeGraphSocketDirection::Output &&
                s.contract.producedPayloadType == payloadType) return &s;
        }
        return nullptr;
    }
};

struct NodeGraphEdge {
    NodeGraphEdgeId id{};
    NodeGraphNodeId fromNode{};
    NodeGraphSocketId fromSocket{};
    NodeGraphNodeId toNode{};
    NodeGraphSocketId toSocket{};
};

enum class NodeGraphChangeType : uint8_t {
    Reset = 0,
    NodeUpsert = 1,
    NodeRemoved = 2,
    EdgeUpsert = 3,
    EdgeRemoved = 4
};

enum class NodeGraphChangeReason : uint8_t {
    Topology = 0,
    Parameter = 1,
    Layout = 2,
    State = 3
};

struct NodeGraphChange {
    NodeGraphChangeType type = NodeGraphChangeType::Reset;
    NodeGraphChangeReason reason = NodeGraphChangeReason::Topology;
    NodeGraphNode node{};
    NodeGraphNodeId nodeId{};
    NodeGraphEdge edge{};
    NodeGraphEdgeId edgeId{};
};

struct NodeGraphDelta {
    uint64_t fromRevision = 0;
    uint64_t toRevision = 0;
    std::vector<NodeGraphChange> changes;
};

struct NodeGraphCompiled {
    uint64_t revision = 0;
    bool isValid = false;
    std::vector<NodeGraphNodeId> executionOrder;
    std::vector<std::string> compilationErrors;
};
