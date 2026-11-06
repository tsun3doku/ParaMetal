#pragma once

#include "CudaBuffer.cuh"
#include "Triangle.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace cudaGeometry {

struct DeviceVoxelGrid {
    float3 gridMin{0.0f, 0.0f, 0.0f};
    float scale = 1.0f;      // source -> grid coordinate scale
    float voxelSize = 1.0f;  // voxel size in grid coordinates
    int3 gridDim{1, 1, 1};
};

bool buildTriangleVoxelGrid(
    const DeviceTriangle* triangles,
    uint32_t triangleCount,
    const DeviceVoxelGrid& grid,
    cudaUtils::CudaBuffer<int32_t>& offsets,
    cudaUtils::CudaBuffer<int32_t>& triangleIds,
    cudaStream_t stream);

bool buildVoxelOccupancy(
    const DeviceTriangle* triangles,
    uint32_t triangleCount,
    const DeviceVoxelGrid& grid,
    const int32_t* voxelOffsets,
    const int32_t* voxelTriangleIds,
    uint32_t voxelTriangleIdCount,
    cudaUtils::CudaBuffer<uint8_t>& occupancy,
    cudaStream_t stream);

} // namespace cudaGeometry
