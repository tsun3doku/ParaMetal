#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <glm/glm.hpp>

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects,
//                                                          backend/controller objects or GPU resources
//                                                        - This header must not be included in any backend ]

struct HeatProbeItem {
    uint32_t id = 1;
    std::string name = "Probe_1";
    glm::vec3 worldPosition{0.0f};
    bool enabled = true;
};

struct HeatProbeData {
    bool enabled = true;
    uint32_t opcPort = 4840;
    bool showTemperature = true;
    bool showName = true;
    std::vector<HeatProbeItem> items;
};
