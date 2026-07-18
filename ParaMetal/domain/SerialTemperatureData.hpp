#pragma once

#include <cstdint>
#include <string>

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects,
//                                                          backend/controller objects or GPU resources
//                                                        - This header must not be included in any backend ]

struct SerialTemperatureData {
    bool enabled = true;
    std::string portName;
    uint32_t baudRate = 115200;
};
