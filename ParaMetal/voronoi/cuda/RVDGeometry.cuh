#pragma once

#include "TriangleBvh.cuh"
#include "../../cuda/CudaBuffer.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

class VoxelGrid;

namespace voronoi::cudaGeometry {

inline constexpr float CanonicalDomainSize = 1000.0f;

struct DeviceVoxelGrid {
    float3 gridMin;
    float scale;
    int3 gridDim;
};

struct DeviceModelGeometryView {
    const DeviceTriangle* triangles = nullptr;
    const TriangleBvhNode* nodes = nullptr;
    const uint32_t* parents = nullptr;
    uint32_t root = InvalidNodeId;
    uint32_t runtimeModelId = 0;
};

struct DeviceModelGeometry {
    cudaUtils::CudaBuffer<DeviceTriangle> triangles;
    cudaUtils::CudaBuffer<uint8_t> occupancy;
    cudaUtils::CudaBuffer<int32_t> voxelTriangleIds;
    cudaUtils::CudaBuffer<int32_t> voxelOffsets;
    DeviceVoxelGrid voxelGrid{};
    uint32_t runtimeModelId = 0;
    TriangleBvh bvh;

    void reset() {
        triangles.reset();
        occupancy.reset();
        voxelTriangleIds.reset();
        voxelOffsets.reset();
        bvh.reset();
        voxelGrid = {};
        runtimeModelId = 0;
    }

    bool valid() const {
        const size_t cellCount = size_t(voxelGrid.gridDim.x) * size_t(voxelGrid.gridDim.y) * size_t(voxelGrid.gridDim.z);
        return triangles.size() != 0 &&
               occupancy.size() != 0 && voxelOffsets.size() == cellCount + 1 &&
               voxelGrid.gridDim.x > 0 && voxelGrid.gridDim.y > 0 && voxelGrid.gridDim.z > 0 &&
               voxelGrid.scale > 0.0f &&
               bvh.valid();
    }

    DeviceModelGeometryView view() const {
        return DeviceModelGeometryView{
            triangles.get(),
            bvh.nodes.get(),
            bvh.parents.get(),
            bvh.root,
            runtimeModelId
        };
    }
};

bool uploadGeometry(
    const std::vector<std::array<glm::vec4, 3>>& triangles,
    const VoxelGrid& voxelGrid,
    DeviceModelGeometry& geometry,
    cudaStream_t stream);

__device__ inline int cornerIndex(const DeviceVoxelGrid& grid, int x, int y, int z) {
    const int stride = grid.gridDim.x + 1;
    return z * stride * stride + y * stride + x;
}

__host__ __device__ inline float3 canonicalFromWorld(const DeviceVoxelGrid& grid, float3 point) {
    return make_float3((point.x - grid.gridMin.x) * grid.scale,
                       (point.y - grid.gridMin.y) * grid.scale,
                       (point.z - grid.gridMin.z) * grid.scale);
}

__device__ inline float voxelSize(const DeviceVoxelGrid& grid) {
    return CanonicalDomainSize / float(grid.gridDim.x);
}

} // namespace voronoi::cudaGeometry
