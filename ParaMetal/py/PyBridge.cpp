#include "PyBridge.hpp"

namespace pybridge {

static NodeGraph* g_graph = nullptr;

void setGraph(NodeGraph* graph) {
    g_graph = graph;
}

NodeGraph* getGraph() {
    return g_graph;
}

} // namespace pybridge
