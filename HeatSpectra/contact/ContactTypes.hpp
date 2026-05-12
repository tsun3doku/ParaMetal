#pragma once

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

#include "contact/ContactGpuStructs.hpp"

struct ContactPair {
    contact::Sample samples[7];
    float contactArea;
};

struct ContactLineVertex {
    glm::vec3 position;
    glm::vec3 color;
};

struct ContactCoupling {
    uint32_t modelARuntimeModelId = 0;
    uint32_t modelBRuntimeModelId = 0;
    std::vector<uint32_t> modelBTriangleIndices;
    std::vector<ContactPair> contactPairs;
    uint32_t contactPairCount = 0;

    bool isValid() const {
        return modelARuntimeModelId != 0 &&
            modelBRuntimeModelId != 0 &&
            !modelBTriangleIndices.empty() &&
            !contactPairs.empty() &&
            contactPairCount != 0;
    }
};
