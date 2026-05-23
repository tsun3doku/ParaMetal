#pragma once

class NodeGraphBridge;

namespace pybridge {
    void setBridge(NodeGraphBridge* bridge);
    NodeGraphBridge* getBridge();
}
