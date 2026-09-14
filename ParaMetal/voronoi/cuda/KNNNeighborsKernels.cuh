#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace voronoi {

constexpr uint32_t KNNBlockSize = 32;
constexpr uint32_t InvalidNeighborId = 0xffffffffu;

struct DeviceGrid {
    float3 gridMin;
    float invCellWidth;
    int3 gridDim;
    int dimXY;
    uint32_t totalCells;
};

struct alignas(16) BucketEntry {
    float4 position;
    uint32_t id;
    uint32_t padding[3];
};

struct DeviceKnn {
    const float4* seeds;
    uint32_t seedCount;
    const uint32_t* cellStart;
    const BucketEntry* bucketEntries;
    const int3* cellOffsets;
    const double* cellOffsetDistances;
    uint32_t cellOffsetCount;
    DeviceGrid grid;
};

__host__ __device__ inline int clampCellCoordinate(int value, int dimension) {
    return value < 0 ? 0 : (value >= dimension ? dimension - 1 : value);
}

__host__ __device__ inline int3 cellIndexOf(float3 position, const DeviceGrid& grid) {
    const double inverseCellWidth = double(grid.invCellWidth);
    const int x = position.x <= grid.gridMin.x
        ? 0 : int((double(position.x) - double(grid.gridMin.x)) * inverseCellWidth);
    const int y = position.y <= grid.gridMin.y
        ? 0 : int((double(position.y) - double(grid.gridMin.y)) * inverseCellWidth);
    const int z = position.z <= grid.gridMin.z
        ? 0 : int((double(position.z) - double(grid.gridMin.z)) * inverseCellWidth);
    return make_int3(clampCellCoordinate(x, grid.gridDim.x),
                     clampCellCoordinate(y, grid.gridDim.y),
                     clampCellCoordinate(z, grid.gridDim.z));
}

__host__ __device__ inline uint32_t linearCell(int x, int y, int z,
                                                const DeviceGrid& grid) {
    return uint32_t(z) * uint32_t(grid.dimXY) +
           uint32_t(y) * uint32_t(grid.gridDim.x) + uint32_t(x);
}

template<typename T>
__host__ __device__ inline T knnMin(T a, T b) { return (a < b) ? a : b; }

template<typename T>
__host__ __device__ inline T knnMax(T a, T b) { return (a > b) ? a : b; }

template<typename TVisitor>
__host__ __device__ inline void traverseKnnCells(const DeviceKnn& knn, float3 queryPoint, TVisitor&& visitor) {
    const int3 queryCell = cellIndexOf(queryPoint, knn.grid);
    for (uint32_t offsetIndex = 0; offsetIndex < knn.cellOffsetCount; ++offsetIndex) {
        if (!visitor.shouldContinue(knn.cellOffsetDistances[offsetIndex])) {
            break;
        }

        const int3 offset = knn.cellOffsets[offsetIndex];
        const int x = queryCell.x + offset.x;
        const int y = queryCell.y + offset.y;
        const int z = queryCell.z + offset.z;
        if (x < 0 || x >= knn.grid.gridDim.x ||
            y < 0 || y >= knn.grid.gridDim.y ||
            z < 0 || z >= knn.grid.gridDim.z) continue;

        const uint32_t cell = linearCell(x, y, z, knn.grid);
        const uint32_t begin = knnMin(knn.cellStart[cell], knn.seedCount);
        const uint32_t end = knnMin(knnMax(knn.cellStart[cell + 1], begin), knn.seedCount);
        for (uint32_t index = begin; index < end; ++index) {
            visitor.visit(knn.bucketEntries[index]);
        }
    }
}

__device__ inline uint32_t nearestSeed(float3 point, const DeviceKnn& knn) {
    uint32_t bestSeed = InvalidNeighborId;
    double bestDistSq = 1e30;

    struct NearestVisitor {
        float3 query;
        uint32_t& bestSeed;
        double& bestDistSq;

        __device__ bool shouldContinue(double cellLowerBoundDistSq) const {
            return bestDistSq > cellLowerBoundDistSq;
        }

        __device__ void visit(const BucketEntry& entry) {
            if (entry.id >= 0xffffffffu) return;
            const double dx = double(query.x) - double(entry.position.x);
            const double dy = double(query.y) - double(entry.position.y);
            const double dz = double(query.z) - double(entry.position.z);
            const double d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < bestDistSq) {
                bestDistSq = d2;
                bestSeed = entry.id;
            }
        }
    } visitor{point, bestSeed, bestDistSq};

    traverseKnnCells(knn, point, visitor);
    return bestSeed;
}

__global__ void knnKernel(DeviceKnn knn, uint32_t* neighbors, double* distanceHeap,
                          uint32_t k, uint32_t* searchFailure,
                          const uint32_t* deviceQueryIds, uint32_t queryCount,
                          uint32_t outputBase,
                          uint32_t outputStride, bool scatterBySeedId);

__global__ void gatherPrefixKernel(const uint32_t* neighbors, uint32_t seedCount,
                                   uint32_t neighborStride, uint32_t maxNeighbors,
                                   uint32_t* out);

} // namespace voronoi
