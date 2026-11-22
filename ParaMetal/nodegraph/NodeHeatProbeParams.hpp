#pragma once

#include "NodeGraphTypes.hpp"
#include "domain/HeatProbeData.hpp"
#include <glm/glm.hpp>
#include <string>
#include <vector>
struct NodeGraphNode;

struct HeatProbeNodeParams {
    bool enabled = true;
    uint32_t opcPort = 4840;
    bool showTemperature = true;
    bool showName = true;
    std::vector<HeatProbeItem> items;
};

std::string serializeHeatProbesJson(const std::vector<HeatProbeItem>& items);
std::vector<HeatProbeItem> deserializeHeatProbesJson(const std::string& json);

HeatProbeNodeParams readHeatProbeNodeParams(const NodeGraphNode& node);

namespace nodegraphtypes {
inline constexpr const char* HeatProbe = "heat_probe";
}

namespace nodegraphparams::heatprobe {
constexpr uint32_t Enabled = 1;
constexpr uint32_t OpcPort = 2;
constexpr uint32_t ProbesJson = 3;
constexpr uint32_t ShowTemperature = 4;
constexpr uint32_t ShowName = 5;
}
