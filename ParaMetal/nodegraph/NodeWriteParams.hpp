#pragma once

#include <string>
#include <cstdint>

struct WriteProductSelection {
    uint32_t producerNodeId = 0;
    uint32_t outputSocketId = 0;
    std::string productId;
    std::string format = "OBJ";

    bool empty() const { return producerNodeId == 0; }
};

struct NodeGraphNode;

struct WriteNodeParams {
    std::string filePath;
    WriteProductSelection product;
};

WriteNodeParams readWriteNodeParams(const NodeGraphNode& node);
