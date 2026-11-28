#pragma once

#include "GeometryData.hpp"
#include "nodegraph/NodeGraphCoreTypes.hpp"

struct GroupData {
    NodeDataHandle sourceHandle{};
    MeshGroup group{};
};
