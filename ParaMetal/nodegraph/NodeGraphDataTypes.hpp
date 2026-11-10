#pragma once

#include "../hash/HashValues.hpp"
#include "NodeGraphCoreTypes.hpp"
#include "NodeGraphTypes.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

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