#include "VoxelGrid.cuh"

#include "CudaCheck.cuh"
#include "CudaMath.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace cudaGeometry {

namespace {

constexpr uint32_t BlockSize = 256;

__device__ inline int clampVoxel(int value, int count) {
    return value < 0 ? 0 : (value >= count ? count - 1 : value);
}

__device__ inline float3 worldToGrid(const DeviceVoxelGrid& grid, const float4& point) {
    return make_float3(
        (point.x - grid.gridMin.x) * grid.scale,
        (point.y - grid.gridMin.y) * grid.scale,
        (point.z - grid.gridMin.z) * grid.scale);
}

__device__ inline bool triangleGridRange(
    const DeviceTriangle& triangle,
    const DeviceVoxelGrid& grid,
    float3 canonical[3],
    int3& voxelMinimum,
    int3& voxelMaximum) {
    canonical[0] = worldToGrid(grid, triangle.v0);
    canonical[1] = worldToGrid(grid, triangle.v1);
    canonical[2] = worldToGrid(grid, triangle.v2);

    for (int i = 0; i < 3; ++i) {
        if (!isfinite(canonical[i].x) || !isfinite(canonical[i].y) ||
            !isfinite(canonical[i].z)) {
            return false;
        }
    }

    const float3 minimum = make_float3(
        fminf(canonical[0].x, fminf(canonical[1].x, canonical[2].x)),
        fminf(canonical[0].y, fminf(canonical[1].y, canonical[2].y)),
        fminf(canonical[0].z, fminf(canonical[1].z, canonical[2].z)));
    const float3 maximum = make_float3(
        fmaxf(canonical[0].x, fmaxf(canonical[1].x, canonical[2].x)),
        fmaxf(canonical[0].y, fmaxf(canonical[1].y, canonical[2].y)),
        fmaxf(canonical[0].z, fmaxf(canonical[1].z, canonical[2].z)));

    const float invVoxelSize = 1.0f / grid.voxelSize;
    voxelMinimum = make_int3(
        clampVoxel(int(floorf(minimum.x * invVoxelSize)), grid.gridDim.x),
        clampVoxel(int(floorf(minimum.y * invVoxelSize)), grid.gridDim.y),
        clampVoxel(int(floorf(minimum.z * invVoxelSize)), grid.gridDim.z));
    voxelMaximum = make_int3(
        clampVoxel(int(ceilf(maximum.x * invVoxelSize)), grid.gridDim.x),
        clampVoxel(int(ceilf(maximum.y * invVoxelSize)), grid.gridDim.y),
        clampVoxel(int(ceilf(maximum.z * invVoxelSize)), grid.gridDim.z));
    return true;
}

__device__ inline bool triangleBoxOverlap(
    const float boxCenter[3],
    const float boxHalfSize[3],
    const float triangleVertices[3][3]) {
    float v0[3], v1[3], v2[3];
    float e0[3], e1[3], e2[3];
    float minimum, maximum, p0, p1, p2, radius;
    float fex, fey, fez;
    float normal[3];

    for (int i = 0; i < 3; ++i) {
        v0[i] = triangleVertices[0][i] - boxCenter[i];
        v1[i] = triangleVertices[1][i] - boxCenter[i];
        v2[i] = triangleVertices[2][i] - boxCenter[i];
    }

    for (int i = 0; i < 3; ++i) {
        e0[i] = v1[i] - v0[i];
        e1[i] = v2[i] - v1[i];
        e2[i] = v0[i] - v2[i];
    }

    fex = fabsf(e0[0]);
    fey = fabsf(e0[1]);
    fez = fabsf(e0[2]);

    p0 = e0[2] * v0[1] - e0[1] * v0[2];
    p2 = e0[2] * v2[1] - e0[1] * v2[2];
    if (p0 < p2) { minimum = p0; maximum = p2; }
    else { minimum = p2; maximum = p0; }
    radius = fez * boxHalfSize[1] + fey * boxHalfSize[2];
    if (minimum > radius || maximum < -radius) return false;

    p0 = -e0[2] * v0[0] + e0[0] * v0[2];
    p2 = -e0[2] * v2[0] + e0[0] * v2[2];
    if (p0 < p2) { minimum = p0; maximum = p2; }
    else { minimum = p2; maximum = p0; }
    radius = fez * boxHalfSize[0] + fex * boxHalfSize[2];
    if (minimum > radius || maximum < -radius) return false;

    p1 = e0[1] * v1[0] - e0[0] * v1[1];
    p2 = e0[1] * v2[0] - e0[0] * v2[1];
    if (p2 < p1) { minimum = p2; maximum = p1; }
    else { minimum = p1; maximum = p2; }
    radius = fey * boxHalfSize[0] + fex * boxHalfSize[1];
    if (minimum > radius || maximum < -radius) return false;

    fex = fabsf(e1[0]);
    fey = fabsf(e1[1]);
    fez = fabsf(e1[2]);

    p0 = e1[2] * v0[1] - e1[1] * v0[2];
    p2 = e1[2] * v2[1] - e1[1] * v2[2];
    if (p0 < p2) { minimum = p0; maximum = p2; }
    else { minimum = p2; maximum = p0; }
    radius = fez * boxHalfSize[1] + fey * boxHalfSize[2];
    if (minimum > radius || maximum < -radius) return false;

    p0 = -e1[2] * v0[0] + e1[0] * v0[2];
    p2 = -e1[2] * v2[0] + e1[0] * v2[2];
    if (p0 < p2) { minimum = p0; maximum = p2; }
    else { minimum = p2; maximum = p0; }
    radius = fez * boxHalfSize[0] + fex * boxHalfSize[2];
    if (minimum > radius || maximum < -radius) return false;

    p0 = e1[1] * v0[0] - e1[0] * v0[1];
    p1 = e1[1] * v1[0] - e1[0] * v1[1];
    if (p0 < p1) { minimum = p0; maximum = p1; }
    else { minimum = p1; maximum = p0; }
    radius = fey * boxHalfSize[0] + fex * boxHalfSize[1];
    if (minimum > radius || maximum < -radius) return false;

    fex = fabsf(e2[0]);
    fey = fabsf(e2[1]);
    fez = fabsf(e2[2]);

    p0 = e2[2] * v0[1] - e2[1] * v0[2];
    p1 = e2[2] * v1[1] - e2[1] * v1[2];
    if (p0 < p1) { minimum = p0; maximum = p1; }
    else { minimum = p1; maximum = p0; }
    radius = fez * boxHalfSize[1] + fey * boxHalfSize[2];
    if (minimum > radius || maximum < -radius) return false;

    p0 = -e2[2] * v0[0] + e2[0] * v0[2];
    p1 = -e2[2] * v1[0] + e2[0] * v1[2];
    if (p0 < p1) { minimum = p0; maximum = p1; }
    else { minimum = p1; maximum = p0; }
    radius = fez * boxHalfSize[0] + fex * boxHalfSize[2];
    if (minimum > radius || maximum < -radius) return false;

    p1 = e2[1] * v1[0] - e2[0] * v1[1];
    p2 = e2[1] * v2[0] - e2[0] * v2[1];
    if (p2 < p1) { minimum = p2; maximum = p1; }
    else { minimum = p1; maximum = p2; }
    radius = fey * boxHalfSize[0] + fex * boxHalfSize[1];
    if (minimum > radius || maximum < -radius) return false;

    minimum = fminf(fminf(v0[0], v1[0]), v2[0]);
    maximum = fmaxf(fmaxf(v0[0], v1[0]), v2[0]);
    if (minimum > boxHalfSize[0] || maximum < -boxHalfSize[0]) return false;

    minimum = fminf(fminf(v0[1], v1[1]), v2[1]);
    maximum = fmaxf(fmaxf(v0[1], v1[1]), v2[1]);
    if (minimum > boxHalfSize[1] || maximum < -boxHalfSize[1]) return false;

    minimum = fminf(fminf(v0[2], v1[2]), v2[2]);
    maximum = fmaxf(fmaxf(v0[2], v1[2]), v2[2]);
    if (minimum > boxHalfSize[2] || maximum < -boxHalfSize[2]) return false;

    normal[0] = e0[1] * e1[2] - e0[2] * e1[1];
    normal[1] = e0[2] * e1[0] - e0[0] * e1[2];
    normal[2] = e0[0] * e1[1] - e0[1] * e1[0];

    float vmin[3], vmax[3];
    for (int q = 0; q < 3; ++q) {
        const float v = v0[q];
        if (normal[q] > 0.0f) {
            vmin[q] = -boxHalfSize[q] - v;
            vmax[q] = boxHalfSize[q] - v;
        } else {
            vmin[q] = boxHalfSize[q] - v;
            vmax[q] = -boxHalfSize[q] - v;
        }
    }

    const float dotMinimum =
        normal[0] * vmin[0] + normal[1] * vmin[1] + normal[2] * vmin[2];
    const float dotMaximum =
        normal[0] * vmax[0] + normal[1] * vmax[1] + normal[2] * vmax[2];
    if (dotMinimum > 0.0f) return false;
    if (dotMaximum >= 0.0f) return true;
    return false;
}

__global__ void countTriangleVoxelsKernel(
    const DeviceTriangle* triangles,
    uint32_t triangleCount,
    DeviceVoxelGrid grid,
    int32_t* counts) {
    for (uint32_t triangleIndex = blockIdx.x * blockDim.x + threadIdx.x;
         triangleIndex < triangleCount;
         triangleIndex += gridDim.x * blockDim.x) {
        float3 canonical[3];
        int3 voxelMinimum;
        int3 voxelMaximum;
        if (!triangleGridRange(
                triangles[triangleIndex], grid, canonical, voxelMinimum, voxelMaximum)) {
            continue;
        }

        const float boxHalfSize[3] = {
            0.5f * grid.voxelSize,
            0.5f * grid.voxelSize,
            0.5f * grid.voxelSize};
        const float triangleVertices[3][3] = {
            {canonical[0].x, canonical[0].y, canonical[0].z},
            {canonical[1].x, canonical[1].y, canonical[1].z},
            {canonical[2].x, canonical[2].y, canonical[2].z}};

        for (int z = voxelMinimum.z; z <= voxelMaximum.z; ++z) {
            for (int y = voxelMinimum.y; y <= voxelMaximum.y; ++y) {
                for (int x = voxelMinimum.x; x <= voxelMaximum.x; ++x) {
                    const float boxCenter[3] = {
                        (float(x) + 0.5f) * grid.voxelSize,
                        (float(y) + 0.5f) * grid.voxelSize,
                        (float(z) + 0.5f) * grid.voxelSize};
                    if (!triangleBoxOverlap(boxCenter, boxHalfSize, triangleVertices)) {
                        continue;
                    }
                    const int32_t voxel =
                        x + y * grid.gridDim.x + z * grid.gridDim.x * grid.gridDim.y;
                    atomicAdd(&counts[voxel], 1);
                }
            }
        }
    }
}

__global__ void finalizeOffsetsKernel(
    int32_t* offsets,
    const int32_t* counts,
    int32_t voxelCount) {
    if (threadIdx.x == 0 && blockIdx.x == 0 && voxelCount > 0) {
        offsets[voxelCount] = offsets[voxelCount - 1] + counts[voxelCount - 1];
    }
}

__global__ void scatterTriangleVoxelsKernel(
    const DeviceTriangle* triangles,
    uint32_t triangleCount,
    DeviceVoxelGrid grid,
    int32_t* cursor,
    int32_t* triangleIds) {
    for (uint32_t triangleIndex = blockIdx.x * blockDim.x + threadIdx.x;
         triangleIndex < triangleCount;
         triangleIndex += gridDim.x * blockDim.x) {
        float3 canonical[3];
        int3 voxelMinimum;
        int3 voxelMaximum;
        if (!triangleGridRange(
                triangles[triangleIndex], grid, canonical, voxelMinimum, voxelMaximum)) {
            continue;
        }

        const float boxHalfSize[3] = {
            0.5f * grid.voxelSize,
            0.5f * grid.voxelSize,
            0.5f * grid.voxelSize};
        const float triangleVertices[3][3] = {
            {canonical[0].x, canonical[0].y, canonical[0].z},
            {canonical[1].x, canonical[1].y, canonical[1].z},
            {canonical[2].x, canonical[2].y, canonical[2].z}};

        for (int z = voxelMinimum.z; z <= voxelMaximum.z; ++z) {
            for (int y = voxelMinimum.y; y <= voxelMaximum.y; ++y) {
                for (int x = voxelMinimum.x; x <= voxelMaximum.x; ++x) {
                    const float boxCenter[3] = {
                        (float(x) + 0.5f) * grid.voxelSize,
                        (float(y) + 0.5f) * grid.voxelSize,
                        (float(z) + 0.5f) * grid.voxelSize};
                    if (!triangleBoxOverlap(boxCenter, boxHalfSize, triangleVertices)) {
                        continue;
                    }
                    const int32_t voxel =
                        x + y * grid.gridDim.x + z * grid.gridDim.x * grid.gridDim.y;
                    const int32_t slot = atomicAdd(&cursor[voxel], 1);
                    triangleIds[slot] = int32_t(triangleIndex);
                }
            }
        }
    }
}

} // namespace

bool buildTriangleVoxelGrid(
    const DeviceTriangle* triangles,
    uint32_t triangleCount,
    const DeviceVoxelGrid& grid,
    cudaUtils::CudaBuffer<int32_t>& offsets,
    cudaUtils::CudaBuffer<int32_t>& triangleIds,
    cudaStream_t stream) {
    offsets.reset();
    triangleIds.reset();

    if (triangles == nullptr || triangleCount == 0) return false;
    if (grid.gridDim.x <= 0 || grid.gridDim.y <= 0 || grid.gridDim.z <= 0 ||
        !(grid.scale > 0.0f) || !std::isfinite(grid.scale) ||
        !(grid.voxelSize > 0.0f) || !std::isfinite(grid.voxelSize) ||
        !std::isfinite(grid.gridMin.x) || !std::isfinite(grid.gridMin.y) ||
        !std::isfinite(grid.gridMin.z)) {
        std::cerr << "[VoxelGrid] invalid grid parameters" << std::endl;
        return false;
    }

    const uint64_t voxelCount64 =
        uint64_t(grid.gridDim.x) * uint64_t(grid.gridDim.y) * uint64_t(grid.gridDim.z);
    if (voxelCount64 == 0 || voxelCount64 > uint64_t(INT32_MAX)) {
        std::cerr << "[VoxelGrid] grid dimensions overflow" << std::endl;
        return false;
    }
    const int32_t voxelCount = int32_t(voxelCount64);

    cudaUtils::CudaBuffer<int32_t> counts;
    cudaUtils::CudaBuffer<int32_t> cursor;
    if (!counts.allocate(size_t(voxelCount)) ||
        !offsets.allocate(size_t(voxelCount) + 1) ||
        !cursor.allocate(size_t(voxelCount))) {
        return false;
    }

    if (!cudaUtils::cudaCheck(
            cudaMemsetAsync(counts.get(), 0, size_t(voxelCount) * sizeof(int32_t), stream),
            "clear voxel triangle counts")) {
        return false;
    }

    const uint32_t gridBlocks = (triangleCount + BlockSize - 1) / BlockSize;
    countTriangleVoxelsKernel<<<gridBlocks, BlockSize, 0, stream>>>(
        triangles, triangleCount, grid, counts.get());
    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch countTriangleVoxelsKernel")) {
        return false;
    }

    size_t tempStorageBytes = 0;
    if (cub::DeviceScan::ExclusiveSum(
            nullptr, tempStorageBytes,
            counts.get(), offsets.get(), voxelCount, stream) != cudaSuccess) {
        std::cerr << "[VoxelGrid] exclusive sum sizing failed" << std::endl;
        return false;
    }

    cudaUtils::CudaBuffer<uint8_t> tempStorage;
    if (!tempStorage.allocate(tempStorageBytes)) return false;
    if (cub::DeviceScan::ExclusiveSum(
            tempStorage.get(), tempStorageBytes,
            counts.get(), offsets.get(), voxelCount, stream) != cudaSuccess) {
        std::cerr << "[VoxelGrid] exclusive sum failed" << std::endl;
        return false;
    }

    finalizeOffsetsKernel<<<1, 1, 0, stream>>>(
        offsets.get(), counts.get(), voxelCount);
    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch finalizeOffsetsKernel")) {
        return false;
    }

    int32_t total = 0;
    if (!cudaUtils::cudaCheck(
            cudaMemcpyAsync(
                &total, offsets.get() + voxelCount, sizeof(int32_t),
                cudaMemcpyDeviceToHost, stream),
            "read voxel triangle count") ||
        !cudaUtils::cudaCheck(
            cudaStreamSynchronize(stream),
            "synchronize voxel triangle count")) {
        return false;
    }
    if (total < 0) {
        std::cerr << "[VoxelGrid] negative voxel triangle count" << std::endl;
        return false;
    }
    if (total == 0) return true;

    if (!triangleIds.allocate(size_t(total))) return false;
    if (!cudaUtils::cudaCheck(
            cudaMemcpyAsync(
                cursor.get(), offsets.get(), size_t(voxelCount) * sizeof(int32_t),
                cudaMemcpyDeviceToDevice, stream),
            "copy voxel offsets to cursor")) {
        triangleIds.reset();
        return false;
    }

    scatterTriangleVoxelsKernel<<<gridBlocks, BlockSize, 0, stream>>>(
        triangles, triangleCount, grid, cursor.get(), triangleIds.get());
    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch scatterTriangleVoxelsKernel")) {
        triangleIds.reset();
        return false;
    }
    return true;
}

constexpr double VoxelOccupancyBorderFraction = 0.1;
constexpr double VoxelOccupancyDuplicateFraction = 1e-3;

__device__ inline double3 voxelOccupancyWorldPoint(
    const DeviceVoxelGrid& grid, int x, int y, int z) {
    const double invScale = 1.0 / double(grid.scale);
    return make_double3(
        double(grid.gridMin.x) + double(x) * double(grid.voxelSize) * invScale,
        double(grid.gridMin.y) + double(y) * double(grid.voxelSize) * invScale,
        double(grid.gridMin.z) + double(z) * double(grid.voxelSize) * invScale);
}

__device__ inline bool voxelOccupancyRayTriangle(
    double3 origin,
    double3 direction,
    double3 a,
    double3 b,
    double3 c,
    double& t) {
    const double3 edge1 = cudaMath::sub3(b, a);
    const double3 edge2 = cudaMath::sub3(c, a);
    const double3 pvec = cudaMath::cross3(direction, edge2);
    const double det = cudaMath::dot3(edge1, pvec);
    if (fabs(det) < 1e-30) return false;

    const double invDet = 1.0 / det;
    const double3 tvec = cudaMath::sub3(origin, a);
    const double u = cudaMath::dot3(tvec, pvec) * invDet;
    if (u < -1e-12 || u > 1.0 + 1e-12) return false;

    const double3 qvec = cudaMath::cross3(tvec, edge1);
    const double v = cudaMath::dot3(direction, qvec) * invDet;
    if (v < -1e-12 || u + v > 1.0 + 1e-12) return false;

    t = cudaMath::dot3(edge2, qvec) * invDet;
    return t > 0.0;
}

__device__ inline double voxelOccupancyPointTriangleDistanceSquared(
    double3 p, double3 a, double3 b, double3 c) {
    const double3 ab = cudaMath::sub3(b, a);
    const double3 ac = cudaMath::sub3(c, a);
    const double3 ap = cudaMath::sub3(p, a);

    const double d1 = cudaMath::dot3(ab, ap);
    const double d2 = cudaMath::dot3(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return cudaMath::dot3(ap, ap);

    const double3 bp = cudaMath::sub3(p, b);
    const double d3 = cudaMath::dot3(ab, bp);
    const double d4 = cudaMath::dot3(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return cudaMath::dot3(bp, bp);

    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        const double3 q = cudaMath::add3(a, cudaMath::mul3(ab, v));
        const double3 qp = cudaMath::sub3(p, q);
        return cudaMath::dot3(qp, qp);
    }

    const double3 cp = cudaMath::sub3(p, c);
    const double d5 = cudaMath::dot3(ab, cp);
    const double d6 = cudaMath::dot3(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return cudaMath::dot3(cp, cp);

    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        const double3 q = cudaMath::add3(a, cudaMath::mul3(ac, w));
        const double3 qp = cudaMath::sub3(p, q);
        return cudaMath::dot3(qp, qp);
    }

    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const double3 q = cudaMath::add3(b, cudaMath::mul3(cudaMath::sub3(c, b), w));
        const double3 qp = cudaMath::sub3(p, q);
        return cudaMath::dot3(qp, qp);
    }

    const double denominator = 1.0 / (va + vb + vc);
    const double v = vb * denominator;
    const double w = vc * denominator;
    const double3 q = cudaMath::add3(
        a, cudaMath::add3(cudaMath::mul3(ab, v), cudaMath::mul3(ac, w)));
    const double3 qp = cudaMath::sub3(p, q);
    return cudaMath::dot3(qp, qp);
}

__global__ void voxelOccupancyBorderKernel(
    const DeviceTriangle* triangles,
    uint32_t triangleCount,
    DeviceVoxelGrid grid,
    const int32_t* voxelOffsets,
    const int32_t* voxelTriangleIds,
    uint32_t voxelTriangleIdCount,
    uint8_t* occupancy,
    uint32_t totalCorners,
    double borderThreshold) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= totalCorners) return;

    const uint32_t strideX = uint32_t(grid.gridDim.x) + 1u;
    const uint32_t strideY = uint32_t(grid.gridDim.y) + 1u;
    const int x = int(index % strideX);
    const int y = int((index / strideX) % strideY);
    const int z = int(index / (strideX * strideY));
    const double3 point = voxelOccupancyWorldPoint(grid, x, y, z);
    const double thresholdSquared = borderThreshold * borderThreshold;

    const int xCells[2] = {max(0, x - 1), min(grid.gridDim.x - 1, x)};
    const int yCells[2] = {max(0, y - 1), min(grid.gridDim.y - 1, y)};
    const int zCells[2] = {max(0, z - 1), min(grid.gridDim.z - 1, z)};
    for (int zi = 0; zi < 2; ++zi) {
        for (int yi = 0; yi < 2; ++yi) {
            for (int xi = 0; xi < 2; ++xi) {
                const int voxelIndex =
                    xCells[xi] + yCells[yi] * grid.gridDim.x +
                    zCells[zi] * grid.gridDim.x * grid.gridDim.y;
                const int32_t begin = voxelOffsets[voxelIndex];
                const int32_t end = voxelOffsets[voxelIndex + 1];
                if (begin < 0 || end < begin ||
                    uint32_t(end) > voxelTriangleIdCount) {
                    continue;
                }
                for (int32_t listIndex = begin; listIndex < end; ++listIndex) {
                    const int32_t triangleIndex = voxelTriangleIds[listIndex];
                    if (triangleIndex < 0 ||
                        uint32_t(triangleIndex) >= triangleCount) {
                        continue;
                    }
                    const DeviceTriangle& triangle = triangles[triangleIndex];
                    const double3 a = make_double3(
                        double(triangle.v0.x), double(triangle.v0.y),
                        double(triangle.v0.z));
                    const double3 b = make_double3(
                        double(triangle.v1.x), double(triangle.v1.y),
                        double(triangle.v1.z));
                    const double3 c = make_double3(
                        double(triangle.v2.x), double(triangle.v2.y),
                        double(triangle.v2.z));
                    if (voxelOccupancyPointTriangleDistanceSquared(point, a, b, c) <=
                        thresholdSquared) {
                        occupancy[index] = 1u;
                        return;
                    }
                }
            }
        }
    }
}

__global__ void voxelOccupancyCountHitsKernel(
    const DeviceTriangle* triangles,
    uint32_t triangleCount,
    DeviceVoxelGrid grid,
    const int32_t* voxelOffsets,
    const int32_t* voxelTriangleIds,
    uint32_t voxelTriangleIdCount,
    int32_t* hitCounts,
    uint32_t columns,
    double worldVoxelSize) {
    const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;

    const uint32_t strideY = uint32_t(grid.gridDim.y) + 1u;
    const int y = int(column % strideY);
    const int z = int(column / strideY);
    const double3 origin = make_double3(
        double(grid.gridMin.x) - worldVoxelSize,
        double(grid.gridMin.y) + double(y) * worldVoxelSize,
        double(grid.gridMin.z) + double(z) * worldVoxelSize);
    const double3 direction = make_double3(1.0, 0.0, 0.0);

    const int yCells[2] = {max(0, y - 1), min(grid.gridDim.y - 1, y)};
    const int zCells[2] = {max(0, z - 1), min(grid.gridDim.z - 1, z)};
    int count = 0;
    for (int zi = 0; zi < 2; ++zi) {
        for (int yi = 0; yi < 2; ++yi) {
            for (int xCell = 0; xCell < grid.gridDim.x; ++xCell) {
                const int voxelIndex =
                    xCell + yCells[yi] * grid.gridDim.x +
                    zCells[zi] * grid.gridDim.x * grid.gridDim.y;
                const int32_t begin = voxelOffsets[voxelIndex];
                const int32_t end = voxelOffsets[voxelIndex + 1];
                if (begin < 0 || end < begin ||
                    uint32_t(end) > voxelTriangleIdCount) {
                    continue;
                }
                for (int32_t listIndex = begin; listIndex < end; ++listIndex) {
                    const int32_t triangleIndex = voxelTriangleIds[listIndex];
                    if (triangleIndex < 0 ||
                        uint32_t(triangleIndex) >= triangleCount) {
                        continue;
                    }
                    const DeviceTriangle& triangle = triangles[triangleIndex];
                    const double3 a = make_double3(
                        double(triangle.v0.x), double(triangle.v0.y),
                        double(triangle.v0.z));
                    const double3 b = make_double3(
                        double(triangle.v1.x), double(triangle.v1.y),
                        double(triangle.v1.z));
                    const double3 c = make_double3(
                        double(triangle.v2.x), double(triangle.v2.y),
                        double(triangle.v2.z));
                    double t = 0.0;
                    if (voxelOccupancyRayTriangle(origin, direction, a, b, c, t)) {
                        ++count;
                    }
                }
            }
        }
    }
    hitCounts[column] = count;
}

__global__ void voxelOccupancyFillHitsKernel(
    const DeviceTriangle* triangles,
    uint32_t triangleCount,
    DeviceVoxelGrid grid,
    const int32_t* voxelOffsets,
    const int32_t* voxelTriangleIds,
    uint32_t voxelTriangleIdCount,
    const int32_t* hitOffsets,
    double* hitX,
    uint32_t columns,
    double worldVoxelSize) {
    const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;

    const uint32_t strideY = uint32_t(grid.gridDim.y) + 1u;
    const int y = int(column % strideY);
    const int z = int(column / strideY);
    const double3 origin = make_double3(
        double(grid.gridMin.x) - worldVoxelSize,
        double(grid.gridMin.y) + double(y) * worldVoxelSize,
        double(grid.gridMin.z) + double(z) * worldVoxelSize);
    const double3 direction = make_double3(1.0, 0.0, 0.0);

    int32_t cursor = hitOffsets[column];
    const int yCells[2] = {max(0, y - 1), min(grid.gridDim.y - 1, y)};
    const int zCells[2] = {max(0, z - 1), min(grid.gridDim.z - 1, z)};
    for (int zi = 0; zi < 2; ++zi) {
        for (int yi = 0; yi < 2; ++yi) {
            for (int xCell = 0; xCell < grid.gridDim.x; ++xCell) {
                const int voxelIndex =
                    xCell + yCells[yi] * grid.gridDim.x +
                    zCells[zi] * grid.gridDim.x * grid.gridDim.y;
                const int32_t begin = voxelOffsets[voxelIndex];
                const int32_t end = voxelOffsets[voxelIndex + 1];
                if (begin < 0 || end < begin ||
                    uint32_t(end) > voxelTriangleIdCount) {
                    continue;
                }
                for (int32_t listIndex = begin; listIndex < end; ++listIndex) {
                    const int32_t triangleIndex = voxelTriangleIds[listIndex];
                    if (triangleIndex < 0 ||
                        uint32_t(triangleIndex) >= triangleCount) {
                        continue;
                    }
                    const DeviceTriangle& triangle = triangles[triangleIndex];
                    const double3 a = make_double3(
                        double(triangle.v0.x), double(triangle.v0.y),
                        double(triangle.v0.z));
                    const double3 b = make_double3(
                        double(triangle.v1.x), double(triangle.v1.y),
                        double(triangle.v1.z));
                    const double3 c = make_double3(
                        double(triangle.v2.x), double(triangle.v2.y),
                        double(triangle.v2.z));
                    double t = 0.0;
                    if (voxelOccupancyRayTriangle(origin, direction, a, b, c, t)) {
                        hitX[cursor++] = t;
                    }
                }
            }
        }
    }
}

__global__ void voxelOccupancySortHitsKernel(
    const int32_t* hitOffsets,
    const int32_t* hitCounts,
    double* hitX,
    uint32_t columns) {
    const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;

    const int32_t begin = hitOffsets[column];
    const int32_t count = hitCounts[column];
    for (int32_t i = 1; i < count; ++i) {
        const double key = hitX[begin + i];
        int32_t j = i - 1;
        while (j >= 0 && hitX[begin + j] > key) {
            hitX[begin + j + 1] = hitX[begin + j];
            --j;
        }
        hitX[begin + j + 1] = key;
    }
}

__global__ void voxelOccupancyClassifyKernel(
    DeviceVoxelGrid grid,
    const int32_t* hitOffsets,
    const int32_t* hitCounts,
    const double* hitX,
    uint8_t* occupancy,
    uint32_t columns,
    double worldVoxelSize,
    double duplicateEpsilon) {
    const uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= columns) return;

    const uint32_t strideX = uint32_t(grid.gridDim.x) + 1u;
    const uint32_t strideY = uint32_t(grid.gridDim.y) + 1u;
    const int y = int(column % strideY);
    const int z = int(column / strideY);
    const int32_t begin = hitOffsets[column];
    const int32_t count = hitCounts[column];

    int32_t hitIndex = 0;
    int uniqueHits = 0;
    double lastHit = -1e300;
    for (int x = 0; x <= grid.gridDim.x; ++x) {
        const uint32_t cornerIndex =
            uint32_t(z) * strideX * strideY + uint32_t(y) * strideX +
            uint32_t(x);
        if (occupancy[cornerIndex] == 1u) continue;

        const double cornerT = double(x + 1) * worldVoxelSize;
        while (hitIndex < count) {
            const double hit = hitX[begin + hitIndex];
            if (hit > cornerT + 1e-12) break;
            if (hit > lastHit + duplicateEpsilon) {
                ++uniqueHits;
                lastHit = hit;
            }
            ++hitIndex;
        }
        occupancy[cornerIndex] = (uniqueHits & 1) ? 2u : 0u;
    }
}

bool buildVoxelOccupancy(
    const DeviceTriangle* triangles,
    uint32_t triangleCount,
    const DeviceVoxelGrid& grid,
    const int32_t* voxelOffsets,
    const int32_t* voxelTriangleIds,
    uint32_t voxelTriangleIdCount,
    cudaUtils::CudaBuffer<uint8_t>& occupancy,
    cudaStream_t stream) {
    occupancy.reset();
    if (triangles == nullptr || triangleCount == 0 ||
        voxelOffsets == nullptr || voxelTriangleIds == nullptr ||
        voxelTriangleIdCount == 0 ||
        grid.gridDim.x <= 0 || grid.gridDim.y <= 0 || grid.gridDim.z <= 0 ||
        !(grid.scale > 0.0f) || !(grid.voxelSize > 0.0f)) {
        return false;
    }

    const size_t strideX = size_t(grid.gridDim.x) + 1u;
    const size_t strideY = size_t(grid.gridDim.y) + 1u;
    const size_t cornerCount = strideX * strideY * (size_t(grid.gridDim.z) + 1u);
    const size_t columns = strideY * (size_t(grid.gridDim.z) + 1u);
    if (cornerCount == 0 || columns == 0 || cornerCount > UINT32_MAX ||
        columns > size_t(INT32_MAX)) {
        return false;
    }
    if (!occupancy.allocate(cornerCount)) return false;
    if (!cudaUtils::cudaCheck(
            cudaMemsetAsync(occupancy.get(), 0, cornerCount, stream),
            "clear voxel occupancy")) {
        return false;
    }

    const double worldVoxelSize = double(grid.voxelSize) / double(grid.scale);
    const double borderThreshold = worldVoxelSize * VoxelOccupancyBorderFraction;
    const uint32_t cornerBlocks =
        (uint32_t(cornerCount) + BlockSize - 1u) / BlockSize;
    voxelOccupancyBorderKernel<<<cornerBlocks, BlockSize, 0, stream>>>(
        triangles, triangleCount, grid, voxelOffsets, voxelTriangleIds,
        voxelTriangleIdCount, occupancy.get(), uint32_t(cornerCount),
        borderThreshold);
    if (!cudaUtils::cudaCheck(
            cudaGetLastError(), "launch voxelOccupancyBorderKernel")) {
        return false;
    }

    cudaUtils::CudaBuffer<int32_t> hitCounts;
    cudaUtils::CudaBuffer<int32_t> hitOffsets;
    cudaUtils::CudaBuffer<double> hitX;
    if (!hitCounts.allocate(columns) || !hitOffsets.allocate(columns + 1u)) {
        return false;
    }
    if (!cudaUtils::cudaCheck(
            cudaMemsetAsync(hitOffsets.get(), 0, sizeof(int32_t), stream),
            "clear voxel occupancy hit offsets")) {
        return false;
    }

    const uint32_t columnBlocks =
        (uint32_t(columns) + BlockSize - 1u) / BlockSize;
    voxelOccupancyCountHitsKernel<<<columnBlocks, BlockSize, 0, stream>>>(
        triangles, triangleCount, grid, voxelOffsets, voxelTriangleIds,
        voxelTriangleIdCount, hitCounts.get(), uint32_t(columns),
        worldVoxelSize);
    if (!cudaUtils::cudaCheck(
            cudaGetLastError(), "launch voxelOccupancyCountHitsKernel")) {
        return false;
    }

    size_t scanBytes = 0;
    cub::DeviceScan::InclusiveSum(
        nullptr, scanBytes, hitCounts.get(), hitOffsets.get() + 1,
        int(columns), stream);
    cudaUtils::CudaBuffer<uint8_t> scanTemp;
    if (!scanTemp.allocate(scanBytes)) return false;
    cub::DeviceScan::InclusiveSum(
        scanTemp.get(), scanBytes, hitCounts.get(), hitOffsets.get() + 1,
        int(columns), stream);
    if (!cudaUtils::cudaCheck(
            cudaGetLastError(), "scan voxel occupancy hit counts")) {
        return false;
    }

    int32_t totalHits = 0;
    if (!cudaUtils::cudaCheck(
            cudaMemcpyAsync(
                &totalHits, hitOffsets.get() + columns, sizeof(int32_t),
                cudaMemcpyDeviceToHost, stream),
            "read voxel occupancy hit count")) {
        return false;
    }
    if (!cudaUtils::cudaCheck(
            cudaStreamSynchronize(stream),
            "synchronize voxel occupancy hit count")) {
        return false;
    }
    if (totalHits < 0) return false;

    if (totalHits > 0) {
        if (!hitX.allocate(size_t(totalHits))) return false;
        voxelOccupancyFillHitsKernel<<<columnBlocks, BlockSize, 0, stream>>>(
            triangles, triangleCount, grid, voxelOffsets, voxelTriangleIds,
            voxelTriangleIdCount, hitOffsets.get(), hitX.get(),
            uint32_t(columns), worldVoxelSize);
        if (!cudaUtils::cudaCheck(
                cudaGetLastError(), "launch voxelOccupancyFillHitsKernel")) {
            return false;
        }
        voxelOccupancySortHitsKernel<<<columnBlocks, BlockSize, 0, stream>>>(
            hitOffsets.get(), hitCounts.get(), hitX.get(), uint32_t(columns));
        if (!cudaUtils::cudaCheck(
                cudaGetLastError(), "launch voxelOccupancySortHitsKernel")) {
            return false;
        }
    }

    voxelOccupancyClassifyKernel<<<columnBlocks, BlockSize, 0, stream>>>(
        grid, hitOffsets.get(), hitCounts.get(),
        totalHits > 0 ? hitX.get() : nullptr, occupancy.get(),
        uint32_t(columns), worldVoxelSize,
        worldVoxelSize * VoxelOccupancyDuplicateFraction);
    if (!cudaUtils::cudaCheck(
            cudaGetLastError(), "launch voxelOccupancyClassifyKernel")) {
        return false;
    }
    return cudaUtils::cudaCheck(
        cudaStreamSynchronize(stream), "synchronize voxel occupancy");
}

} // namespace cudaGeometry
