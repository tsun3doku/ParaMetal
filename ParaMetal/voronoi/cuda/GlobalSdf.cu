#include "GlobalSdf.cuh"

#include "RVDGeometry.cuh"
#include "RVD.hpp"
#include "TriangleBvh.cuh"
#include "TriangleQueries.cuh"
#include "../../cuda/CudaBuffer.cuh"
#include "../../spatial/VoxelGrid.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>

namespace voronoi::globalsdf {

using cudaGeometry::DeviceTriangle;
using cudaGeometry::DeviceVoxelGrid;
using cudaGeometry::TriangleBvhNode;
using cudaGeometry::ClosestSurfaceHit;
using cudaGeometry::closestPoint;
using cudaGeometry::canonicalFromWorld;
using cudaGeometry::cornerIndex;
using cudaGeometry::voxelSize;
using cudaGeometry::sub3;
using cudaGeometry::cross3;
using cudaGeometry::dot3;

template <class T>
using DeviceBuffer = cudaUtils::CudaBuffer<T>;
constexpr auto cudaOk = cudaUtils::cudaCheck;
constexpr uint32_t BlockSize = 256;

struct SDFGrid {
    float3 gridMin;
    float cellSize;
    int3 gridDim;
    uint32_t sampleCount;
};

__device__ inline int clampInt(int value, int minimum, int maximum) {
    return max(minimum, min(value, maximum));
}

__global__ void minCombineKernel(float* dest, const float* src, uint32_t count) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dest[idx] = fminf(dest[idx], src[idx]);
    }
}

__global__ void evaluateGlobalSdfKernel(
    float* outSignedValues,
    SDFGrid grid,
    const DeviceTriangle* triangles,
    const TriangleBvhNode* bvhNodes,
    const uint32_t* bvhParents,
    uint32_t bvhRoot,
    const uint8_t* occupancy,
    DeviceVoxelGrid voxelGrid)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= grid.sampleCount) return;

    const int x = int(index % uint32_t(grid.gridDim.x));
    const int y = int((index / uint32_t(grid.gridDim.x)) % uint32_t(grid.gridDim.y));
    const int z = int(index / uint32_t(grid.gridDim.x * grid.gridDim.y));
    const float3 point = make_float3(grid.gridMin.x + x * grid.cellSize,
                                     grid.gridMin.y + y * grid.cellSize,
                                     grid.gridMin.z + z * grid.cellSize);

    const ClosestSurfaceHit hit = closestPoint(point, triangles, bvhNodes, bvhParents, bvhRoot);
    const float distance = sqrtf(fmaxf(0.0f, hit.closest.distanceSquared));

    const float canonicalVoxelSize = voxelSize(voxelGrid);
    const float3 canonical = canonicalFromWorld(voxelGrid, point);
    const int vx = clampInt(int(canonical.x / canonicalVoxelSize), 0, voxelGrid.gridDim.x);
    const int vy = clampInt(int(canonical.y / canonicalVoxelSize), 0, voxelGrid.gridDim.y);
    const int vz = clampInt(int(canonical.z / canonicalVoxelSize), 0, voxelGrid.gridDim.z);
    const uint8_t occ = occupancy[cornerIndex(voxelGrid, vx, vy, vz)];

    float signedValue = distance;
    if (occ == 2u) {
        signedValue = -distance;
    } else if (occ == 1u && hit.triangleId != InvalidId) {
        const DeviceTriangle tri = triangles[hit.triangleId];
        const float3 a = make_float3(tri.v0.x, tri.v0.y, tri.v0.z);
        const float3 b = make_float3(tri.v1.x, tri.v1.y, tri.v1.z);
        const float3 c = make_float3(tri.v2.x, tri.v2.y, tri.v2.z);
        const float3 n = cross3(sub3(b, a), sub3(c, a));
        if (dot3(n, n) > 1e-30f && dot3(sub3(point, hit.closest.point), n) < 0.0f) {
            signedValue = -distance;
        }
    }
    outSignedValues[index] = signedValue;
}

bool build(const BuildInput& input, BuildResult& result, cudaStream_t stream) {
    result = {};
    if (!input.geometries || input.geometries->empty() ||
        !(input.cellSize > 0.0f) || !std::isfinite(input.cellSize) ||
        input.gridDim.x < 2 || input.gridDim.y < 2 || input.gridDim.z < 2) return false;

    const size_t sampleCount64 = size_t(input.gridDim.x) * input.gridDim.y * input.gridDim.z;
    if (sampleCount64 == 0 || sampleCount64 > UINT32_MAX) return false;
    const size_t channelCount = input.geometries->size();
    if (channelCount > SIZE_MAX / sampleCount64) return false;


    const SDFGrid grid{make_float3(input.gridMin.x, input.gridMin.y, input.gridMin.z),
                       input.cellSize,
                       make_int3(input.gridDim.x, input.gridDim.y, input.gridDim.z),
                       uint32_t(sampleCount64)};

    result.values.resize(channelCount * sampleCount64);
    result.channelStride = uint32_t(sampleCount64);
    result.channelCount = uint32_t(channelCount);

    DeviceBuffer<float> signedValues;
    if (!signedValues.allocate(sampleCount64)) return false;
    const uint32_t gridBlocks = (grid.sampleCount + BlockSize - 1) / BlockSize;

    for (uint32_t channel = 0; channel < uint32_t(channelCount); ++channel) {
        const ModelGeometry& source = (*input.geometries)[channel];
        if (!source.voxelGrid || !source.triangles || source.triangles->empty()) {
            return false;
        }

        cudaGeometry::DeviceModelGeometry geometry;
        if (!cudaGeometry::uploadGeometry(*source.triangles, *source.voxelGrid, geometry, stream)) {
            std::cerr << "[GlobalSdf] geometry upload failed channel=" << channel << std::endl;
            return false;
        }

        evaluateGlobalSdfKernel<<<gridBlocks, BlockSize, 0, stream>>>(
            signedValues.get(),
            grid,
            geometry.triangles.get(),
            geometry.bvh.nodes.get(),
            geometry.bvh.parents.get(),
            geometry.bvh.root,
            geometry.occupancy.get(),
            geometry.voxelGrid);

        if (!cudaOk(cudaGetLastError(), "launch GlobalSdf evaluate kernel")) {
            geometry.reset();
            return false;
        }

        if (input.outDevUnionSdf) {
            if (channel == 0) {
                cudaMemcpyAsync(input.outDevUnionSdf, signedValues.get(), sampleCount64 * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            } else {
                minCombineKernel<<<gridBlocks, BlockSize, 0, stream>>>(input.outDevUnionSdf, signedValues.get(), uint32_t(sampleCount64));
            }
        }

        float* destination = result.values.data() + size_t(channel) * sampleCount64;
        if (!signedValues.downloadAsync(destination, sampleCount64, stream)) {
            geometry.reset();
            return false;
        }
        geometry.reset();
    }
    if (!cudaOk(cudaStreamSynchronize(stream), "synchronize GlobalSdf readback")) return false;
    return true;
}

} // namespace voronoi::globalsdf