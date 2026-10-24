#pragma once

#include "RVDGeometry.cuh"

#include <cuda_runtime.h>

namespace voronoi::cudaGeometry {

bool buildBvhMoments(
    DeviceModelGeometry& geometry,
    cudaStream_t stream);

bool buildWindingSolidMask(
    const DeviceModelGeometryView* views,
    uint32_t viewCount,
    float3 gridMin,
    int3 gridDim,
    float cellSize,
    uint8_t* outMask,
    cudaStream_t stream);

} // namespace voronoi::cudaGeometry
