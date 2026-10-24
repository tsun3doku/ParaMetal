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
    const float4* moments = nullptr;
    uint32_t root = InvalidNodeId;
    uint32_t runtimeModelId = 0;
    float3 boundsMin{0.0f};
    float3 boundsMax{0.0f};
};

struct DeviceModelGeometry {
    cudaUtils::CudaBuffer<DeviceTriangle> triangles;
    cudaUtils::CudaBuffer<float4> bvhMoments;
    cudaUtils::CudaBuffer<int32_t> voxelTriangleIds;
    cudaUtils::CudaBuffer<int32_t> voxelOffsets;
    DeviceVoxelGrid voxelGrid{};
    uint32_t runtimeModelId = 0;
    float3 boundsMin{0.0f};
    float3 boundsMax{0.0f};
    TriangleBvh bvh;

    void reset() {
        triangles.reset();
        bvhMoments.reset();
        voxelTriangleIds.reset();
        voxelOffsets.reset();
        bvh.reset();
        voxelGrid = {};
        runtimeModelId = 0;
        boundsMin = make_float3(0.0f, 0.0f, 0.0f);
        boundsMax = make_float3(0.0f, 0.0f, 0.0f);
    }

    bool valid() const {
        const size_t cellCount = size_t(voxelGrid.gridDim.x) * size_t(voxelGrid.gridDim.y) * size_t(voxelGrid.gridDim.z);
        return triangles.size() != 0 &&
               bvhMoments.size() != 0 &&
               voxelOffsets.size() == cellCount + 1 &&
               voxelGrid.gridDim.x > 0 && voxelGrid.gridDim.y > 0 && voxelGrid.gridDim.z > 0 &&
               voxelGrid.scale > 0.0f &&
               bvh.valid();
    }

    DeviceModelGeometryView view() const {
        return DeviceModelGeometryView{
            triangles.get(),
            bvh.nodes.get(),
            bvh.parents.get(),
            bvhMoments.get(),
            bvh.root,
            runtimeModelId,
            boundsMin,
            boundsMax
        };
    }
};

bool uploadGeometry(
    const std::vector<std::array<glm::vec4, 3>>& triangles,
    const VoxelGrid& voxelGrid,
    DeviceModelGeometry& geometry,
    cudaStream_t stream,
    uint32_t runtimeModelId = 0);

__host__ __device__ inline float3 canonicalFromWorld(const DeviceVoxelGrid& grid, float3 point) {
    return make_float3((point.x - grid.gridMin.x) * grid.scale,
                       (point.y - grid.gridMin.y) * grid.scale,
                       (point.z - grid.gridMin.z) * grid.scale);
}

} // namespace voronoi::cudaGeometry
