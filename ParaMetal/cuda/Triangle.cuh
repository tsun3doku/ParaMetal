#pragma once

#include <cuda_runtime.h>

namespace cudaGeometry {

struct DeviceTriangle {
    float4 v0;
    float4 v1;
    float4 v2;
};

static_assert(sizeof(DeviceTriangle) == sizeof(float4) * 3,
              "DeviceTriangle must match float4 triangle layout");

} // namespace cudaGeometry
