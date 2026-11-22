#include "NodeGraphRegistry.hpp"

#include <limits>
#include <utility>

uint8_t NodeGraphRegistry::registerPayloadType(const std::string& name, NodeGraphValueType displayType) {
    if (name.empty() || payloadNameToId.count(name) ||
        nextPayloadTypeId > std::numeric_limits<uint8_t>::max()) {
        registrationValid = false;
        return 0;
    }

    const uint8_t id = static_cast<uint8_t>(nextPayloadTypeId++);
    payloadNameToId.emplace(name, id);
    payloadTypes.push_back(PayloadTypeDefinition{id, name, displayType});
    return id;
}

NodeGraphValueType NodeGraphRegistry::getPayloadDisplayType(uint8_t typeId) const {
    if (typeId == 0 || typeId > payloadTypes.size()) {
        return NodeGraphValueType::None;
    }
    return payloadTypes[typeId - 1].displayType;
}

const std::string* NodeGraphRegistry::getPayloadTypeName(uint8_t typeId) const {
    if (typeId == 0 || typeId > payloadTypes.size()) {
        return nullptr;
    }
    return &payloadTypes[typeId - 1].name;
}

bool NodeGraphRegistry::registerNodeType(NodeTypeDefinition definition) {
    if (definition.id.empty() || nodeTypeToIndex.count(definition.id)) {
        registrationValid = false;
        return false;
    }

    const size_t index = nodeList.size();
    nodeList.push_back(std::move(definition));
    nodeTypeToIndex.emplace(nodeList.back().id, index);
    return true;
}

const NodeTypeDefinition* NodeGraphRegistry::findNodeType(const NodeTypeId& typeId) const {
    const auto it = nodeTypeToIndex.find(typeId);
    if (it == nodeTypeToIndex.end()) {
        return nullptr;
    }
    return &nodeList[it->second];
}

const std::vector<NodeTypeDefinition>& NodeGraphRegistry::allNodeTypes() const {
    return nodeList;
}
