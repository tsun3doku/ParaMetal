#pragma once

#include "nodegraph/NodeGraphCoreTypes.hpp"

#include <cstdint>
//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects,
//                                                          backend/controller objects or GPU resources
//                                                        - This header must not be included in any backend ]

struct ContactPairEndpoint {
    NodeDataHandle payloadHandle{};
    NodeDataHandle meshHandle{};
};

struct ContactPairData {
    bool hasValidContact = false;
    ContactPairEndpoint endpointA{};
    ContactPairEndpoint endpointB{};

    float minNormalDot = -0.65f;
    float contactRadius = 0.01f;
    NodeDataHandle contactPairsHandle{};
};

struct ContactData {
    uint64_t payloadHash = 0;
    uint64_t emitterPayloadHash = 0;
    uint64_t receiverPayloadHash = 0;
    ContactPairData pair{};
    bool active = false;

    void sealPayload();
};
