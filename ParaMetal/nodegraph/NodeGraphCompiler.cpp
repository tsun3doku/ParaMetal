#include "NodeGraphCompiler.hpp"
#include "NodeGraphRegistry.hpp"

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

    std::unordered_map<uint32_t, std::size_t> nodeIndexById;
    nodeIndexById.reserve(state.nodes.size() * 2);
    std::size_t index = 0;
    for (const auto& [id, node] : state.nodes) {
        nodeIndexById[id] = index++;
    }

    std::vector<std::vector<uint32_t>> adjacency(state.nodes.size());
    std::vector<uint32_t> inDegree(state.nodes.size(), 0);
    for (const auto& [id, edge] : state.edges) {
        const auto fromIt = nodeIndexById.find(edge.fromNode.value);
        const auto toIt = nodeIndexById.find(edge.toNode.value);
        if (fromIt == nodeIndexById.end() || toIt == nodeIndexById.end()) {
            continue;
        }

        adjacency[fromIt->second].push_back(edge.toNode.value);
        ++inDegree[toIt->second];
    }

    std::vector<uint32_t> readyNodeIds;
    readyNodeIds.reserve(state.nodes.size());
    for (const auto& [id, node] : state.nodes) {
        const auto nodeIt = nodeIndexById.find(id);
        if (nodeIt == nodeIndexById.end()) {
            continue;
        }
        if (inDegree[nodeIt->second] == 0) {
            readyNodeIds.push_back(id);
        }
    }

    outOrder.reserve(state.nodes.size());
    std::size_t cursor = 0;
    while (cursor < readyNodeIds.size()) {
        const uint32_t currentNodeId = readyNodeIds[cursor++];
        outOrder.push_back(NodeGraphNodeId{currentNodeId});

        const auto currentIt = nodeIndexById.find(currentNodeId);
        if (currentIt == nodeIndexById.end()) {
            continue;
        }

        for (const uint32_t downstreamNodeId : adjacency[currentIt->second]) {
            const auto downstreamIt = nodeIndexById.find(downstreamNodeId);
            if (downstreamIt == nodeIndexById.end()) {
                continue;
            }

            uint32_t& downstreamDegree = inDegree[downstreamIt->second];
            if (downstreamDegree == 0) {
                continue;
            }

            --downstreamDegree;
            if (downstreamDegree == 0) {
                readyNodeIds.push_back(downstreamNodeId);
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
