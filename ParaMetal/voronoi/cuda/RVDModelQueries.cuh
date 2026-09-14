#pragma once

#include "RVD.hpp"
#include "RVDGeometry.cuh"
#include "TriangleQueries.cuh"

#include <cuda_runtime.h>
#include <cmath>

namespace voronoi {

__device__ inline bool isRankedCloser(float dA, uint32_t idA, float dB, uint32_t idB) {
    if (dA != dB) return dA < dB;
    return idA < idB;
}

__device__ inline float distanceToModel(float3 p, const cudaGeometry::DeviceModelGeometryView& geom) {
    const cudaGeometry::ClosestSurfaceHit hit = cudaGeometry::closestPoint(
        p, geom.triangles, geom.nodes, geom.parents, geom.root);
    return sqrtf(fmaxf(0.0f, hit.closest.distanceSquared));
}

__device__ inline float nearestOtherDistance(
    float3 p,
    uint32_t modelIndex,
    const cudaGeometry::DeviceModelGeometryView* models,
    uint32_t modelCount)
{
    float minDist = 1e30f;
    for (uint32_t j = 0; j < modelCount; ++j) {
        if (j == modelIndex) continue;
        const float d = distanceToModel(p, models[j]);
        if (d < minDist) minDist = d;
    }
    return minDist;
}

__device__ inline bool areNearestTwoModels(
    float3 p,
    uint32_t modelIndexA,
    uint32_t modelIndexB,
    const cudaGeometry::DeviceModelGeometryView* models,
    uint32_t modelCount)
{
    if (modelCount < 2 || !models) return false;

    float dBest = 1e30f;
    float dSecond = 1e30f;
    uint32_t rBest = InvalidId;
    uint32_t rSecond = InvalidId;
    uint32_t idxBest = InvalidId;
    uint32_t idxSecond = InvalidId;

    for (uint32_t m = 0; m < modelCount; ++m) {
        const float d = distanceToModel(p, models[m]);
        const uint32_t rM = models[m].runtimeModelId;
        if (isRankedCloser(d, rM, dBest, rBest)) {
            dSecond = dBest;
            rSecond = rBest;
            idxSecond = idxBest;
            dBest = d;
            rBest = rM;
            idxBest = m;
        } else if (isRankedCloser(d, rM, dSecond, rSecond)) {
            dSecond = d;
            rSecond = rM;
            idxSecond = m;
        }
    }

    return (idxBest == modelIndexA && idxSecond == modelIndexB) ||
           (idxBest == modelIndexB && idxSecond == modelIndexA);
}

} // namespace voronoi
