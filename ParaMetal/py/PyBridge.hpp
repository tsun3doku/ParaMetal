#pragma once

class NodeGraph;

namespace pybridge {
    void setGraph(NodeGraph* graph);
    NodeGraph* getGraph();
}
