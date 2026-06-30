#pragma once

#include "NodeGraphState.hpp"
#include "NodeGraphTypes.hpp"

#include <string>
#include <vector>
#include <unordered_set>

class NodeGraphCompiler {
public:
    static NodeGraphCompiled compile(const NodeGraphState& state);

private:
    static bool buildTopologicalOrder(const NodeGraphState& state, std::vector<NodeGraphNodeId>& outOrder, std::string* outError = nullptr);
};
