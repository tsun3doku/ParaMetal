#pragma once

#include "../hash/HashValues.hpp"
#include "NodeGraphCoreTypes.hpp"
#include "NodeGraphTypes.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class NodeGraphTypeRegistry;
class NodePayloadRegistry;

struct NodeDataBlock {
    uint8_t dataType = 0;
    NodeDataHandle payloadHandle{};
    HashValues hashes{};
    double scalarFloatValue = 0.0;
    int64_t scalarIntValue = 0;
    bool scalarBoolValue = false;
    bool isFrozen = false;
    std::unordered_map<std::string, std::string> metadata;
    std::vector<NodeGraphNodeId> lineageNodeIds;
};

void populateMetadata(NodeDataBlock& dataBlock, const NodeGraphTypeRegistry* typeRegistry = nullptr, const NodePayloadRegistry* payloadRegistry = nullptr);