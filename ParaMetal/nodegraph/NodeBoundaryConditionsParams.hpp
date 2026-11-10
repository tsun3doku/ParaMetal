#pragma once

#include "NodeGraphTypes.hpp"
#include "domain/BoundaryData.hpp"
#include <array>

struct NodeGraphNode;

struct BoundaryConditionsNodeParams {
    std::array<BoundaryData::Face, 6> faces{};
};

BoundaryConditionsNodeParams readBoundaryConditionsNodeParams(const NodeGraphNode& node);
