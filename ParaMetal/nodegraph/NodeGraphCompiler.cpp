#include "NodeGraphCompiler.hpp"
#include "NodeGraphRegistry.hpp"

#include <algorithm>
#include <functional>
#include <queue>
#include <unordered_map>
#include <vector>

bool NodeGraphCompiler::buildTopologicalOrder(
    const NodeGraphState& state,
    std::vector<NodeGraphNodeId>& outOrder,
    std::string* outError) {
    outOrder.clear();
    if (outError) {
        outError->clear();
    }

    if (state.nodes.empty()) {
        return true;
    }

    std::vector<uint32_t> nodeIds;
    nodeIds.reserve(state.nodes.size());
    for (const auto& [id, node] : state.nodes) {
        (void)node;
        nodeIds.push_back(id);
    }
    std::sort(nodeIds.begin(), nodeIds.end());

    std::unordered_map<uint32_t, std::size_t> nodeIndexById;
    nodeIndexById.reserve(nodeIds.size() * 2);
    for (std::size_t index = 0; index < nodeIds.size(); ++index) {
        nodeIndexById[nodeIds[index]] = index;
    }

    std::vector<std::vector<std::size_t>> adjacency(nodeIds.size());
    std::vector<std::size_t> inDegree(nodeIds.size(), 0);
    for (const auto& [id, edge] : state.edges) {
        (void)id;
        const auto fromIt = nodeIndexById.find(edge.fromNode.value);
        const auto toIt = nodeIndexById.find(edge.toNode.value);
        if (fromIt == nodeIndexById.end() || toIt == nodeIndexById.end()) {
            continue;
        }
        adjacency[fromIt->second].push_back(toIt->second);
        ++inDegree[toIt->second];
    }

    std::priority_queue<
        std::size_t,
        std::vector<std::size_t>,
        std::greater<std::size_t>> ready;
    for (std::size_t index = 0; index < nodeIds.size(); ++index) {
        if (inDegree[index] == 0) {
            ready.push(index);
        }
    }

    outOrder.reserve(nodeIds.size());
    while (!ready.empty()) {
        const std::size_t current = ready.top();
        ready.pop();
        outOrder.push_back(NodeGraphNodeId{nodeIds[current]});

        for (const std::size_t downstream : adjacency[current]) {
            if (inDegree[downstream] == 0) {
                continue;
            }
            --inDegree[downstream];
            if (inDegree[downstream] == 0) {
                ready.push(downstream);
            }
        }
    }

    if (outOrder.size() != state.nodes.size()) {
        outOrder.clear();
        if (outError) {
            *outError = "Graph contains a cycle or invalid dependency chain.";
        }
        return false;
    }

    return true;
}
NodeGraphCompiled NodeGraphCompiler::compile(const NodeGraphState& state) {
    NodeGraphCompiled compiled{};
    compiled.revision = state.revision;

    std::string topoError;
    if (!buildTopologicalOrder(state, compiled.executionOrder, &topoError)) {
        compiled.isValid = false;
        if (!topoError.empty()) {
            compiled.compilationErrors.push_back(topoError);
        }
        return compiled;
    }

    compiled.isValid = true;
    return compiled;
}
