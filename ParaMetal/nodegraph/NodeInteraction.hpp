#pragma once

#include "NodeGraphEvaluation.hpp"
#include "NodeGraphTypes.hpp"
#include "mesh/MeshSelection.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

class NodePayloadRegistry;
struct NodeGraphState;

struct NodeChoice {
    std::string label;
    std::string value;
};

struct NodeParameterChoices {
    uint32_t parameterId = 0;
    std::vector<NodeChoice> choices;
};

struct NodeInspectionField {
    std::string name;
    std::string value;
};

struct NodeInspection {
    std::vector<NodeParameterChoices> parameterChoices;
    std::vector<NodeInspectionField> fields;
};

struct NodeAuthoringContext {
    const NodeGraphNode& node;
    const NodeGraphState& graph;
    const NodeGraphEvaluation& evaluation;
    const NodePayloadRegistry& payloads;
};

struct NodeActionMessage {
    enum class Type : uint8_t {
        Begin,
        Complete,
        Cancel,
        Clear
    };

    Type type = Type::Begin;
    NodeGraphNodeId nodeId{};
    uint64_t nodeInstanceId = 0;
    uint32_t actionId = 0;
};

using NodeInteractionValue = std::variant<
    std::monostate,
    MeshSelection,
    glm::mat4,
    glm::vec3,
    uint32_t
>;

struct NodeInteractionRequest {
    NodeInteractionType type = NodeInteractionType::MeshSubselection;
    SurfaceBinding binding = SurfaceBinding::Source;
    uint64_t sourceSocketKey = 0;
    uint64_t displaySocketKey = 0;
    NodeInteractionValue initialValue;
};

struct NodeInteractionSession {
    uint64_t id = 0;
    uint64_t nodeInstanceId = 0;
    NodeGraphNodeId nodeId{};
    uint32_t actionId = 0;
    NodeInteractionType type = NodeInteractionType::MeshSubselection;
    SurfaceBinding binding = SurfaceBinding::Source;
    uint64_t sourceSocketKey = 0;
    uint64_t displaySocketKey = 0;
    uint64_t sourceSurfaceIdentity = 0;
    uint32_t runtimeModelId = 0;
};

struct NodeInteractionResult {
    uint64_t sessionId = 0;
    NodeInteractionValue value;
};
