#pragma once

#include "NodeGraphTypes.hpp"

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

struct PayloadTypeDefinition {
    uint8_t id = 0;
    std::string name;
    NodeGraphValueType displayType = NodeGraphValueType::None;
};

class NodeGraphRegistry {
public:
    NodeGraphRegistry() = default;

    uint8_t registerPayloadType(const std::string& name, NodeGraphValueType displayType);
    NodeGraphValueType getPayloadDisplayType(uint8_t typeId) const;
    const std::string* getPayloadTypeName(uint8_t typeId) const;

    bool registerNodeType(NodeTypeDefinition definition);
    const NodeTypeDefinition* findNodeType(const NodeTypeId& typeId) const;
    const std::vector<NodeTypeDefinition>& allNodeTypes() const;
    bool isValid() const { return registrationValid; }

private:
    std::unordered_map<std::string, uint8_t> payloadNameToId;
    std::vector<PayloadTypeDefinition> payloadTypes;
    uint32_t nextPayloadTypeId = 1;

    std::vector<NodeTypeDefinition> nodeList;
    std::unordered_map<NodeTypeId, size_t> nodeTypeToIndex;
    bool registrationValid = true;
};
