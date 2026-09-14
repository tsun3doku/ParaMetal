#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <cfloat>

namespace voronoi {

__host__ __device__ inline float contactGeometryTolerance(float contactResolution, float coordinateScale) {
    return fmaxf(32.0f * FLT_EPSILON * coordinateScale, 1e-4f * contactResolution);
}

__device__ inline bool contactActive(float gap, float contactRange, float tolerance) {
    return gap <= contactRange + tolerance;
}

} // namespace voronoi
