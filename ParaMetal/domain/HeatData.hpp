#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"
#include "heat/HeatSystemPresets.hpp"

#include <vector>

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects, 
//                                                          backend/controller objects or GPU resources 
//                                                        - This header must not be included in any backend ]

struct HeatMaterialBinding {
    uint32_t modelNodeId = 0;
    HeatMaterialPresetId presetId = HeatMaterialPresetId::Aluminum;
};

struct HeatData {
    uint64_t payloadHash = 0;
    uint64_t voronoiPayloadHash = 0;
    uint64_t contactPayloadHash = 0;
    std::vector<NodeDataHandle> voronoiHandles;
    std::vector<NodeDataHandle> contactHandles;
    std::vector<NodeDataHandle> heatModelHandles;
    std::vector<HeatMaterialBinding> materialBindings;
    float contactThermalConductance = 16000.0f;
    float simulationDuration = 5.0f;
    bool active = false;
    bool paused = false;
    uint32_t resetCounter = 0;

    void sealPayload();
};
