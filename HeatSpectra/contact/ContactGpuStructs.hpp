#pragma once

#include <cstdint>

namespace contact {

struct Sample {
    uint32_t sourceTriangleIndex;
    float u;
    float v;
    float wArea;
};

static_assert(sizeof(Sample) == 16, "Sample must match GPU stride");

struct ContactSampleEntry {
    uint32_t weightOffset;    
    uint32_t weightCount;     
    float conductance;        
    uint32_t _pad;
};

static_assert(sizeof(ContactSampleEntry) == 16, "ContactSampleEntry must match GPU stride");

struct ContactSampleWeight {
    uint32_t cellIndex;
    float weight;
    uint32_t _pad0;
    uint32_t _pad1;
};

static_assert(sizeof(ContactSampleWeight) == 16, "ContactSampleWeight must match GPU stride");

struct PushConstant {
    uint32_t couplingKind;
    float heatSourceTemperature;
    uint32_t _pad1;
    uint32_t _pad2;
};

static_assert(sizeof(PushConstant) == 16, "PushConstant must match GPU stride");

struct ContactIndex {
    uint32_t offset;
    uint32_t count;
    float contactK;
    uint32_t _pad;
};

static_assert(sizeof(ContactIndex) == 16, "ContactIndex must match GPU stride");
}
