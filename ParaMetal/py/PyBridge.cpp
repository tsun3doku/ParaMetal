#include "PyBridge.hpp"

namespace pybridge {

static NodeGraphBridge* g_bridge = nullptr;

void setBridge(NodeGraphBridge* bridge) {
    g_bridge = bridge;
}

NodeGraphBridge* getBridge() {
    return g_bridge;
}

} // namespace pybridge
