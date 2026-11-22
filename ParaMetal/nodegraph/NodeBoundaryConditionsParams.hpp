#pragma once

#include "NodeGraphTypes.hpp"
#include "domain/BoundaryData.hpp"
#include <array>

struct NodeGraphNode;

struct BoundaryConditionsNodeParams {
    std::array<BoundaryData::Face, 6> faces{};
};

BoundaryConditionsNodeParams readBoundaryConditionsNodeParams(const NodeGraphNode& node);

namespace nodegraphparams::boundaryconditions {
inline constexpr uint32_t flowTypeParam(uint32_t face)           { return 100 + face * 10; }
inline constexpr uint32_t velocityXParam(uint32_t face)          { return 101 + face * 10; }
inline constexpr uint32_t velocityYParam(uint32_t face)          { return 102 + face * 10; }
inline constexpr uint32_t velocityZParam(uint32_t face)          { return 103 + face * 10; }
inline constexpr uint32_t thermalTypeParam(uint32_t face)        { return 104 + face * 10; }
inline constexpr uint32_t temperatureParam(uint32_t face)        { return 105 + face * 10; }
inline constexpr uint32_t heatFluxParam(uint32_t face)           { return 106 + face * 10; }
inline constexpr uint32_t htcParam(uint32_t face)                { return 107 + face * 10; }
inline constexpr uint32_t inflowTemperatureParam(uint32_t face)  { return 108 + face * 10; }
}
