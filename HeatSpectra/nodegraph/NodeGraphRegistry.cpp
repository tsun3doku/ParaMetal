#include "NodeGraphRegistry.hpp"

uint8_t NodeGraphRegistry::registerPayloadType(const std::string& name, NodeGraphValueType displayType) {
    return payloadTypes.registerPayloadType(name, displayType);
}

NodeGraphValueType NodeGraphRegistry::getPayloadDisplayType(uint8_t typeId) const {
    return payloadTypes.getDisplayType(typeId);
}

const std::string* NodeGraphRegistry::getPayloadTypeName(uint8_t typeId) const {
    return payloadTypes.getTypeName(typeId);
}

void NodeGraphRegistry::registerNodeType(NodeTypeDefinition definition) {
    nodeTypes[definition.id] = std::move(definition);
    nodeList.clear();
    for (const auto& pair : nodeTypes) {
        nodeList.push_back(pair.second);
    }
}

const NodeTypeDefinition* NodeGraphRegistry::findNodeType(const NodeTypeId& typeId) const {
    auto it = nodeTypes.find(typeId);
    if (it != nodeTypes.end()) {
        return &it->second;
    }
    return nullptr;
}

const std::vector<NodeTypeDefinition>& NodeGraphRegistry::allNodeTypes() const {
    return nodeList;
}
