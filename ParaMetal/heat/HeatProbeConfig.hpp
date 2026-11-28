#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

struct HeatProbeConfigItem {
    uint32_t id = 1;
    std::string name = "Probe_1";
    glm::vec3 worldPosition{0.0f};
    bool enabled = true;
};

struct HeatProbeConfig {
    bool enabled = true;
    uint32_t opcPort = 4840;
    bool showTemperature = true;
    bool showName = true;
    std::vector<HeatProbeConfigItem> items;
};
