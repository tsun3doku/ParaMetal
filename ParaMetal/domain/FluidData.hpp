#pragma once

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects,
//                                                          backend/controller objects or GPU resources
//                                                        - This header must not be included in any backend ]

struct FluidData {
    float density = 1.225f;
    float kinematicViscosity = 1.516e-5f;
    float specificHeat = 1005.0f;
    float conductivity = 0.026f;
    float initialTemperatureC = 20.0f;
};
