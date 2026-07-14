#pragma once

#include <cstdint>

namespace contact {

struct Sample {
    uint32_t modelATriangleIndex;
    float u;
    float v;
    float contactSampleArea;
};

static_assert(sizeof(Sample) == 16, "Sample must match GPU stride");

}
