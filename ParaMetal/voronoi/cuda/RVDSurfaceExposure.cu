#include "RVDSurfaceExposure.cuh"
#include "RVDModelQueries.cuh"
#include "TriangleQueries.cuh"
#include "RVDContactPredicates.cuh"
#include "../../cuda/CudaBuffer.cuh"

#include <cub/cub.cuh>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

namespace voronoi {

using cudaGeometry::DeviceModelGeometryView;
using cudaGeometry::sub3;
using cudaGeometry::add3;
using cudaGeometry::mul3;
using cudaGeometry::cross3;
using cudaGeometry::len3;

static constexpr double FaceAreaClosureFraction = 1e-3;
static constexpr uint32_t BlockSize = 256;

static bool exclusiveSum(
    const uint32_t* counts,
    uint32_t* offsets,
    uint32_t count,
    cudaUtils::CudaBuffer<uint8_t>& scratch,
    cudaStream_t stream)
{
    size_t bytes = 0;
    if (cub::DeviceScan::ExclusiveSum(
            nullptr,
            bytes,
            counts,
            offsets,
            int(count),
            stream) != cudaSuccess)
        return false;

    if (scratch.size() < bytes && !scratch.allocate(bytes))
        return false;

    return cub::DeviceScan::ExclusiveSum(
        scratch.get(),
        bytes,
        counts,
        offsets,
        int(count),
        stream) == cudaSuccess;
}

static inline bool readScanTotal(
    const cudaUtils::CudaBuffer<uint32_t>& off,
    const cudaUtils::CudaBuffer<uint32_t>& cnt,
    uint32_t count,
    cudaStream_t stream,
    uint32_t& total)
{
    if (count == 0) {
        total = 0;
        return true;
    }
    uint32_t o = 0, c = 0;
    if (cudaMemcpyAsync(&o, off.get() + (count - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(&c, cnt.get() + (count - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) return false;
    total = o + c;
    return true;
}

static __device__ inline float triangleDiameter(float3 a, float3 b, float3 c) {
    const float e0 = len3(sub3(b, a));
    const float e1 = len3(sub3(c, b));
    const float e2 = len3(sub3(a, c));
    return fmaxf(e0, fmaxf(e1, e2));
}

static __global__ void countSurfaceFanTrianglesKernel(
    const RestrictedSurface* surfaces,
    uint32_t totalSurfaces,
    uint32_t* outFanCounts)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSurfaces) return;
    const uint32_t vCnt = surfaces[idx].vertexCount;
    outFanCounts[idx] = (vCnt >= 3) ? (vCnt - 2) : 0;
}

static __global__ void emitSurfaceFanTrianglesKernel(
    const RestrictedSurface* surfaces,
    const float3* restrictedVertices,
    uint32_t totalSurfaces,
    uint32_t globalSurfaceBase,
    const uint32_t* fanOffsets,
    float3* outTriangles,
    uint32_t* outSurfaceIndex)
{
    const uint32_t sIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (sIdx >= totalSurfaces) return;

    const RestrictedSurface surf = surfaces[sIdx];
    if (surf.vertexCount < 3) return;

    const uint32_t vBase = surf.firstVertex;
    const uint32_t fBase = fanOffsets[sIdx];
    const float3 v0 = restrictedVertices[vBase];

    for (uint32_t k = 0; k < surf.vertexCount - 2; ++k) {
        const float3 v1 = restrictedVertices[vBase + k + 1];
        const float3 v2 = restrictedVertices[vBase + k + 2];
        const uint32_t dst = fBase + k;

        outTriangles[3 * dst + 0] = v0;
        outTriangles[3 * dst + 1] = v1;
        outTriangles[3 * dst + 2] = v2;
        outSurfaceIndex[dst] = globalSurfaceBase + sIdx;
    }
}

static __device__ inline void accumulateClippedSubtriangle(
    float3 a, float3 b, float3 c,
    float ha, float hb, float hc,
    uint32_t surfaceIdx,
    double4* devExposure)
{
    const bool inA = (ha > 0.0f);
    const bool inB = (hb > 0.0f);
    const bool inC = (hc > 0.0f);
    const int inCount = (inA ? 1 : 0) + (inB ? 1 : 0) + (inC ? 1 : 0);

    if (inCount == 0) return;

    if (inCount == 3) {
        const float3 va = cross3(sub3(b, a), sub3(c, a));
        const float area = 0.5f * len3(va);
        const float3 cent = mul3(add3(add3(a, b), c), 1.0f / 3.0f);
        atomicAdd(&devExposure[surfaceIdx].x, double(area * cent.x));
        atomicAdd(&devExposure[surfaceIdx].y, double(area * cent.y));
        atomicAdd(&devExposure[surfaceIdx].z, double(area * cent.z));
        atomicAdd(&devExposure[surfaceIdx].w, double(area));
        return;
    }

    if (inCount == 1) {
        float3 p0 = a, p1 = b, p2 = c;
        float h0 = ha, h1 = hb, h2 = hc;
        if (inB) { p0 = b; p1 = c; p2 = a; h0 = hb; h1 = hc; h2 = ha; }
        else if (inC) { p0 = c; p1 = a; p2 = b; h0 = hc; h1 = ha; h2 = hb; }

        const float t01 = -h0 / (h1 - h0);
        const float t02 = -h0 / (h2 - h0);
        const float3 q1 = add3(p0, mul3(sub3(p1, p0), t01));
        const float3 q2 = add3(p0, mul3(sub3(p2, p0), t02));

        const float3 va = cross3(sub3(q1, p0), sub3(q2, p0));
        const float area = 0.5f * len3(va);
        const float3 cent = mul3(add3(add3(p0, q1), q2), 1.0f / 3.0f);
        atomicAdd(&devExposure[surfaceIdx].x, double(area * cent.x));
        atomicAdd(&devExposure[surfaceIdx].y, double(area * cent.y));
        atomicAdd(&devExposure[surfaceIdx].z, double(area * cent.z));
        atomicAdd(&devExposure[surfaceIdx].w, double(area));
        return;
    }

    if (inCount == 2) {
        float3 p0 = a, p1 = b, p2 = c;
        float h0 = ha, h1 = hb, h2 = hc;
        if (!inA) { p0 = b; p1 = c; p2 = a; h0 = hb; h1 = hc; h2 = ha; }
        else if (!inB) { p0 = c; p1 = a; p2 = b; h0 = hc; h1 = ha; h2 = hb; }

        const float t02 = -h0 / (h2 - h0);
        const float t12 = -h1 / (h2 - h1);
        const float3 q0 = add3(p0, mul3(sub3(p2, p0), t02));
        const float3 q1 = add3(p1, mul3(sub3(p2, p1), t12));

        const float3 va1 = cross3(sub3(p1, p0), sub3(q1, p0));
        const float area1 = 0.5f * len3(va1);
        const float3 cent1 = mul3(add3(add3(p0, p1), q1), 1.0f / 3.0f);

        const float3 va2 = cross3(sub3(q1, p0), sub3(q0, p0));
        const float area2 = 0.5f * len3(va2);
        const float3 cent2 = mul3(add3(add3(p0, q1), q0), 1.0f / 3.0f);

        const float totalArea = area1 + area2;
        atomicAdd(&devExposure[surfaceIdx].x, double(area1 * cent1.x + area2 * cent2.x));
        atomicAdd(&devExposure[surfaceIdx].y, double(area1 * cent1.y + area2 * cent2.y));
        atomicAdd(&devExposure[surfaceIdx].z, double(area1 * cent1.z + area2 * cent2.z));
        atomicAdd(&devExposure[surfaceIdx].w, double(totalArea));
    }
}

static __global__ void evaluateAdaptiveTrianglesKernel(
    const float3* triangles,
    const uint32_t* surfaceIndices,
    uint32_t triangleCount,
    const DeviceModelGeometryView* models,
    uint32_t modelCount,
    uint32_t myModelIndex,
    float contactRange,
    float contactResolution,
    float lengthTol,
    uint32_t* outSubdivideCounts,
    double4* devExposure)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= triangleCount) return;

    const float3 v0 = triangles[3 * idx + 0];
    const float3 v1 = triangles[3 * idx + 1];
    const float3 v2 = triangles[3 * idx + 2];
    const uint32_t sIdx = surfaceIndices[idx];

    const float3 c = mul3(add3(add3(v0, v1), v2), 1.0f / 3.0f);
    const float R = fmaxf(len3(sub3(v0, c)), fmaxf(len3(sub3(v1, c)), len3(sub3(v2, c))));

    const float dCent = nearestOtherDistance(c, myModelIndex, models, modelCount);
    const float hc = dCent - contactRange;

    // 1-Lipschitz certificate of exposure
    if (hc - R > lengthTol) {
        const float3 va = cross3(sub3(v1, v0), sub3(v2, v0));
        const float area = 0.5f * len3(va);
        atomicAdd(&devExposure[sIdx].x, double(area * c.x));
        atomicAdd(&devExposure[sIdx].y, double(area * c.y));
        atomicAdd(&devExposure[sIdx].z, double(area * c.z));
        atomicAdd(&devExposure[sIdx].w, double(area));
        outSubdivideCounts[idx] = 0;
        return;
    }

    // 1-Lipschitz certificate of full contact 
    if (hc + R <= -lengthTol) {
        outSubdivideCounts[idx] = 0;
        return;
    }

    // Check diameter for subdivision vs terminal resolution
    const float diam = triangleDiameter(v0, v1, v2);
    if (diam > contactResolution) {
        outSubdivideCounts[idx] = 4;
        return;
    }

    // Terminal boundary triangle
    const float h0 = nearestOtherDistance(v0, myModelIndex, models, modelCount) - contactRange;
    const float h1 = nearestOtherDistance(v1, myModelIndex, models, modelCount) - contactRange;
    const float h2 = nearestOtherDistance(v2, myModelIndex, models, modelCount) - contactRange;

    accumulateClippedSubtriangle(v0, v1, c, h0, h1, hc, sIdx, devExposure);
    accumulateClippedSubtriangle(v1, v2, c, h1, h2, hc, sIdx, devExposure);
    accumulateClippedSubtriangle(v2, v0, c, h2, h0, hc, sIdx, devExposure);

    outSubdivideCounts[idx] = 0;
}

static __global__ void emitSubdividedTrianglesKernel(
    const float3* inTriangles,
    const uint32_t* inSurfaceIndices,
    uint32_t inCount,
    const uint32_t* subdivideCounts,
    const uint32_t* subdivideOffsets,
    float3* outTriangles,
    uint32_t* outSurfaceIndices)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inCount) return;
    if (subdivideCounts[idx] == 0) return;

    const uint32_t dstBase = subdivideOffsets[idx];
    const float3 v0 = inTriangles[3 * idx + 0];
    const float3 v1 = inTriangles[3 * idx + 1];
    const float3 v2 = inTriangles[3 * idx + 2];
    const uint32_t sIdx = inSurfaceIndices[idx];

    const float3 m01 = mul3(add3(v0, v1), 0.5f);
    const float3 m12 = mul3(add3(v1, v2), 0.5f);
    const float3 m20 = mul3(add3(v2, v0), 0.5f);

    outTriangles[3 * (dstBase + 0) + 0] = v0;
    outTriangles[3 * (dstBase + 0) + 1] = m01;
    outTriangles[3 * (dstBase + 0) + 2] = m20;
    outSurfaceIndices[dstBase + 0] = sIdx;

    outTriangles[3 * (dstBase + 1) + 0] = v1;
    outTriangles[3 * (dstBase + 1) + 1] = m12;
    outTriangles[3 * (dstBase + 1) + 2] = m01;
    outSurfaceIndices[dstBase + 1] = sIdx;

    outTriangles[3 * (dstBase + 2) + 0] = v2;
    outTriangles[3 * (dstBase + 2) + 1] = m20;
    outTriangles[3 * (dstBase + 2) + 2] = m12;
    outSurfaceIndices[dstBase + 2] = sIdx;

    outTriangles[3 * (dstBase + 3) + 0] = m01;
    outTriangles[3 * (dstBase + 3) + 1] = m12;
    outTriangles[3 * (dstBase + 3) + 2] = m20;
    outSurfaceIndices[dstBase + 3] = sIdx;
}

bool computeSurfaceExposure(
    const std::vector<ModelRestrictionData>& modelRestrictions,
    const cudaUtils::CudaBuffer<cudaGeometry::DeviceModelGeometryView>& devGeometries,
    const BaseCellsContext& baseCtx,
    float contactRange,
    float contactResolution,
    std::vector<SurfaceMeasure>& outExposure,
    cudaStream_t stream)
{
    outExposure.clear();
    const uint32_t modelCount = uint32_t(modelRestrictions.size());
    if (modelCount == 0) return true;

    if (devGeometries.size() != modelRestrictions.size()) {
        std::cerr << "[RVDSurfaceExposure] geometry view count mismatch" << std::endl;
        return false;
    }

    uint32_t totalSurfaces = 0;
    for (uint32_t m = 0; m < modelCount; ++m) {
        const auto& restriction = modelRestrictions[m];
        if (restriction.surfaceGeometry.size() != restriction.totalPatches ||
            restriction.devSurfaces.size() != restriction.totalPatches) {
            std::cerr << "[RVDSurfaceExposure] restriction patch count mismatch on model " << m << std::endl;
            return false;
        }
        if (restriction.surfaceBase != totalSurfaces) {
            std::cerr << "[RVDSurfaceExposure] non-contiguous surface ranges\n";
            return false;
        }
        totalSurfaces += restriction.totalPatches;
    }
    if (totalSurfaces == 0) return true;

    outExposure.assign(totalSurfaces, SurfaceMeasure{});
    const double invScale2 = baseCtx.invBaseScale * baseCtx.invBaseScale;

    // Surface 100% exposed to fluid
    if (modelCount == 1) {
        const uint32_t sCount = modelRestrictions[0].totalPatches;
        for (uint32_t s = 0; s < sCount; ++s) {
            const auto& geom = modelRestrictions[0].surfaceGeometry[s];
            const double origArea = double(geom.centroidArea.w) * invScale2;
            const glm::dvec3 polyCentroid = glm::dvec3(baseCtx.domainMin) +
                glm::dvec3(geom.centroidArea.x, geom.centroidArea.y, geom.centroidArea.z) * baseCtx.invBaseScale;
            outExposure[s].area = origArea;
            outExposure[s].moment = polyCentroid * origArea;
        }
        return true;
    }

    cudaUtils::CudaBuffer<double4> devExposure;
    if (!devExposure.allocate(totalSurfaces)) return false;
    if (cudaMemsetAsync(devExposure.get(), 0, totalSurfaces * sizeof(double4), stream) != cudaSuccess) return false;

    const glm::vec3 domainSize = baseCtx.domainMax - baseCtx.domainMin;
    const float coordinateScale = std::max({domainSize.x, domainSize.y, domainSize.z});
    const float lengthTol = contactGeometryTolerance(contactResolution, coordinateScale);

    cudaUtils::CudaBuffer<uint8_t> scanScratch;

    for (uint32_t m = 0; m < modelCount; ++m) {
        const auto& restriction = modelRestrictions[m];
        const uint32_t sCount = restriction.totalPatches;
        if (sCount == 0) continue;
        const uint32_t surfaceBase = restriction.surfaceBase;

        cudaUtils::CudaBuffer<uint32_t> devCounts;
        cudaUtils::CudaBuffer<uint32_t> devOffsets;
        if (!devCounts.allocate(sCount) || !devOffsets.allocate(sCount)) return false;

        const uint32_t grid = (sCount + BlockSize - 1) / BlockSize;
        countSurfaceFanTrianglesKernel<<<grid, BlockSize, 0, stream>>>(
            restriction.devSurfaces.get(), sCount, devCounts.get());
        if (cudaGetLastError() != cudaSuccess) return false;

        if (!exclusiveSum(devCounts.get(), devOffsets.get(), sCount, scanScratch, stream)) return false;
        uint32_t initialFans = 0;
        if (!readScanTotal(devOffsets, devCounts, sCount, stream, initialFans)) return false;
        if (initialFans > 0) {
            cudaUtils::CudaBuffer<float3> currentTriangles;
            cudaUtils::CudaBuffer<uint32_t> currentSurfaceIndices;
            if (!currentTriangles.allocate(3 * initialFans) ||
                !currentSurfaceIndices.allocate(initialFans)) return false;

            emitSurfaceFanTrianglesKernel<<<grid, BlockSize, 0, stream>>>(
                restriction.devSurfaces.get(),
                restriction.devRestrictedVertices.get(),
                sCount,
                surfaceBase,
                devOffsets.get(),
                currentTriangles.get(),
                currentSurfaceIndices.get());
            if (cudaGetLastError() != cudaSuccess) return false;

            uint32_t activeCount = initialFans;

            while (activeCount > 0) {
                const uint32_t gridEval = (activeCount + BlockSize - 1) / BlockSize;
                if (devCounts.size() < activeCount) {
                    if (!devCounts.allocate(activeCount) ||
                        !devOffsets.allocate(activeCount))
                        return false;
                }

                evaluateAdaptiveTrianglesKernel<<<gridEval, BlockSize, 0, stream>>>(
                    currentTriangles.get(),
                    currentSurfaceIndices.get(),
                    activeCount,
                    devGeometries.get(),
                    modelCount,
                    m,
                    contactRange,
                    contactResolution,
                    lengthTol,
                    devCounts.get(),
                    devExposure.get());
                if (cudaGetLastError() != cudaSuccess) return false;

                if (!exclusiveSum(devCounts.get(), devOffsets.get(), activeCount, scanScratch, stream)) return false;

                uint32_t nextTriCount = 0;
                if (!readScanTotal(devOffsets, devCounts, activeCount, stream, nextTriCount)) return false;
                if (nextTriCount == 0) break;

                cudaUtils::CudaBuffer<float3> nextTriangles;
                cudaUtils::CudaBuffer<uint32_t> nextSurfaceIndices;
                if (!nextTriangles.allocate(3 * nextTriCount) ||
                    !nextSurfaceIndices.allocate(nextTriCount)) return false;

                emitSubdividedTrianglesKernel<<<gridEval, BlockSize, 0, stream>>>(
                    currentTriangles.get(),
                    currentSurfaceIndices.get(),
                    activeCount,
                    devCounts.get(),
                    devOffsets.get(),
                    nextTriangles.get(),
                    nextSurfaceIndices.get());
                if (cudaGetLastError() != cudaSuccess) return false;

                currentTriangles.swap(nextTriangles);
                currentSurfaceIndices.swap(nextSurfaceIndices);
                activeCount = nextTriCount;
            }
        }
    }

    std::vector<double4> hostExposure(totalSurfaces);
    if (cudaMemcpyAsync(hostExposure.data(), devExposure.get(), totalSurfaces * sizeof(double4), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) return false;

    for (size_t i = 0; i < totalSurfaces; ++i) {
        outExposure[i].area = hostExposure[i].w;
        outExposure[i].moment = glm::dvec3(hostExposure[i].x, hostExposure[i].y, hostExposure[i].z);
    }

    // Exposure validation
    for (uint32_t m = 0; m < modelCount; ++m) {
        const auto& restriction = modelRestrictions[m];
        const uint32_t sCount = restriction.totalPatches;
        for (uint32_t s = 0; s < sCount; ++s) {
            const double origArea = double(restriction.surfaceGeometry[s].centroidArea.w) * invScale2;
            const double tol = std::max(baseCtx.areaTolerance, origArea * FaceAreaClosureFraction);
            const uint32_t globalSurface = restriction.surfaceBase + s;
            const double area = outExposure[globalSurface].area;
            if (area < -tol || area > origArea + tol) {
                std::cerr << "[RVDSurfaceExposure-Invariant] exposure area out of bounds on surface "
                          << globalSurface << " area=" << area << " origArea=" << origArea
                          << " tol=" << tol << std::endl;
                return false;
            }
        }
    }

    return true;
}

} // namespace voronoi
