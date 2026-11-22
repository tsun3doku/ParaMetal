#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildWriteDefinition();

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

namespace nodegraphtypes {
inline constexpr const char* Write = "write";
}

namespace nodegraphparams::write {
constexpr uint32_t FilePath = 1;
constexpr uint32_t Product = 5;
}
