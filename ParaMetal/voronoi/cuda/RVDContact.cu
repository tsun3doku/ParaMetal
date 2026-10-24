#include "RVDContact.cuh"
#include "TriangleBvh.cuh"
#include "RVDBaseCells.cuh"
#include "RVDGeometry.cuh"
#include "../../cuda/CudaBuffer.cuh"
#include "../../cuda/CudaMath.cuh"

#include <cub/cub.cuh>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

namespace voronoi {

using cudaMath::add3;
using cudaMath::sub3;
using cudaMath::mul3;
using cudaMath::dot3;
using cudaMath::cross3;
using cudaMath::length3;
using cudaMath::normalize3;

static constexpr uint32_t BlockSize = 256;
static constexpr int MaxMortarScratchVertices = 128;

struct MortarBasis {
    double3 origin;
    double3 u;
    double3 v;
    double3 normal;
};

struct DeviceModelInfo {
    uint32_t modelIndex = 0;
    uint32_t surfaceBase = 0;
    uint32_t totalPatches = 0;
    const RestrictedSurfaceGeometry* geom = nullptr;
    const RestrictedSurface* surfaces = nullptr;
    const double3* vertices = nullptr;
};

static cudaUtils::CudaBuffer<double2> overlapVertices;
static cudaUtils::CudaBuffer<uint32_t> overlapVertexOffsets;
static cudaUtils::CudaBuffer<uint32_t> overlapVertexCounts;
static cudaUtils::CudaBuffer<MortarBasis> mortarBases;

static __host__ __device__ inline double polygonSignedArea2D(const double2* poly, int count) {
    if (count < 3) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        int next = (i + 1 == count) ? 0 : i + 1;
        sum += poly[i].x * poly[next].y - poly[next].x * poly[i].y;
    }
    return 0.5 * sum;
}

static __host__ __device__ inline void reversePolygon2D(double2* poly, int count) {
    for (int i = 0; i < count / 2; ++i) {
        double2 tmp = poly[i];
        poly[i] = poly[count - 1 - i];
        poly[count - 1 - i] = tmp;
    }
}

static __device__ inline int clipPolygon2DByHalfPlane(
    const double2* inPoly,
    int inCount,
    double a,
    double b,
    double c,
    double2* outPoly,
    int* outCapacityError = nullptr)
{
    if (inCount < 3) return 0;
    int outCount = 0;

    double2 prev = inPoly[inCount - 1];
    double dPrev = a * prev.x + b * prev.y + c;

    for (int i = 0; i < inCount; ++i) {
        double2 curr = inPoly[i];
        double dCurr = a * curr.x + b * curr.y + c;

        if ((dPrev < 0.0 && dCurr >= 0.0) ||
            (dPrev >= 0.0 && dCurr < 0.0)) {

            if (outCount >= MaxMortarScratchVertices) {
                if (outCapacityError) atomicExch(outCapacityError, 1);
                return -1;
            }

            double t = dPrev / (dPrev - dCurr);
            outPoly[outCount++] = make_double2(
                prev.x + t * (curr.x - prev.x),
                prev.y + t * (curr.y - prev.y)
            );
        }

        if (dCurr >= 0.0) {
            if (outCount >= MaxMortarScratchVertices) {
                if (outCapacityError) atomicExch(outCapacityError, 1);
                return -1;
            }
            outPoly[outCount++] = curr;
        }

        prev = curr;
        dPrev = dCurr;
    }

    return outCount;
}

static __host__ __device__ inline double polygonAreaAndCentroid2D(
    const double2* poly,
    int count,
    double2& outCentroid)
{
    if (count < 3) {
        outCentroid = make_double2(0.0, 0.0);
        return 0.0;
    }
    double signedArea2 = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    for (int i = 0; i < count; ++i) {
        int next = (i + 1 == count) ? 0 : i + 1;
        double cross = poly[i].x * poly[next].y - poly[next].x * poly[i].y;
        signedArea2 += cross;
        cx += (poly[i].x + poly[next].x) * cross;
        cy += (poly[i].y + poly[next].y) * cross;
    }
    double area = 0.5 * fabs(signedArea2);
    if (area <= 1e-14) {
        outCentroid = make_double2(0.0, 0.0);
        return 0.0;
    }
    outCentroid = make_double2(cx / (3.0 * signedArea2), cy / (3.0 * signedArea2));
    return area;
}

template <typename T>
static bool ensureCapacity(cudaUtils::CudaBuffer<T>& buffer, size_t needed) {
    if (buffer.size() < needed) {
        return buffer.allocate(needed);
    }
    return true;
}

static bool exclusiveSum(
    const uint32_t* in,
    uint32_t* out,
    int count,
    cudaUtils::CudaBuffer<uint8_t>& scratch,
    cudaStream_t stream)
{
    size_t bytes = 0;
    if (cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, count, stream) != cudaSuccess) return false;
    if (!ensureCapacity(scratch, bytes)) return false;
    return cub::DeviceScan::ExclusiveSum(scratch.get(), bytes, in, out, count, stream) == cudaSuccess;
}

template <typename KeyT, typename ValueT>
static bool sortPairs(
    const KeyT* keysIn, KeyT* keysOut,
    const ValueT* valsIn, ValueT* valsOut,
    int count,
    cudaUtils::CudaBuffer<uint8_t>& scratch,
    cudaStream_t stream)
{
    size_t bytes = 0;
    if (cub::DeviceRadixSort::SortPairs(nullptr, bytes, keysIn, keysOut, valsIn, valsOut, count, 0, sizeof(KeyT) * 8, stream) != cudaSuccess) return false;
    if (!ensureCapacity(scratch, bytes)) return false;
    return cub::DeviceRadixSort::SortPairs(scratch.get(), bytes, keysIn, keysOut, valsIn, valsOut, count, 0, sizeof(KeyT) * 8, stream) == cudaSuccess;
}

static __global__ void initSurfaceExposureKernel(
    const RestrictedSurfaceGeometry* geom,
    uint32_t patchCount,
    uint32_t surfaceBase,
    double invBaseScale,
    float3 domainMin,
    double4* outDevExposure)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= patchCount) return;

    RestrictedSurfaceGeometry g = geom[idx];
    double invScale2 = invBaseScale * invBaseScale;
    double canArea = g.centroidArea.w;
    double totalArea = canArea * invScale2;

    double3 canCent = make_double3(g.centroidArea.x, g.centroidArea.y, g.centroidArea.z);
    double3 worldCent = make_double3(
        double(domainMin.x) + canCent.x * invBaseScale,
        double(domainMin.y) + canCent.y * invBaseScale,
        double(domainMin.z) + canCent.z * invBaseScale
    );
    double3 totalMoment = mul3(worldCent, totalArea);

    outDevExposure[surfaceBase + idx] = make_double4(totalMoment.x, totalMoment.y, totalMoment.z, totalArea);
}

static __global__ void extractPatchTrianglesKernel(
    const RestrictedSurface* surfaces,
    uint32_t patchCount,
    uint32_t* outTriIds,
    uint32_t* outPatchIndices)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= patchCount) return;
    outTriIds[idx] = surfaces[idx].triangleId;
    outPatchIndices[idx] = idx;
}

static __device__ inline int2 findPatchRangeForTriangle(
    const uint32_t* sortedTriangles,
    uint32_t totalPatches,
    uint32_t targetTriId)
{
    int l = 0, r = int(totalPatches);
    while (l < r) {
        int m = l + (r - l) / 2;
        if (sortedTriangles[m] < targetTriId) l = m + 1;
        else r = m;
    }
    int start = l;
    if (start >= int(totalPatches) || sortedTriangles[start] != targetTriId) {
        return make_int2(0, 0);
    }
    l = start; r = int(totalPatches);
    while (l < r) {
        int m = l + (r - l) / 2;
        if (sortedTriangles[m] <= targetTriId) l = m + 1;
        else r = m;
    }
    return make_int2(start, l - start);
}

static __global__ void countAndEmitCandidatePatchPairsKernel(
    const RestrictedSurface* surfacesA,
    const RestrictedSurfaceGeometry* geomA,
    const double3* verticesA,
    uint32_t patchCountA,
    uint32_t surfBaseA,
    cudaGeometry::DeviceModelGeometryView geomB,
    const uint32_t* sortedTrianglesB,
    const uint32_t* sortedPatchIndicesB,
    const RestrictedSurfaceGeometry* geomBData,
    uint32_t patchCountB,
    uint32_t surfBaseB,
    double contactRange,
    float3 domainMin,
    double invBaseScale,
    float coordScale,
    const uint32_t* offsetsA,
    uint32_t* outCountsA,
    uint2* outCandidatePairs,
    bool emitPass)
{
    uint32_t idxA = blockIdx.x * blockDim.x + threadIdx.x;
    if (idxA >= patchCountA) return;

    RestrictedSurface sA = surfacesA[idxA];
    if (sA.vertexCount < 3) {
        if (!emitPass) outCountsA[idxA] = 0;
        return;
    }

    float3 aabbMin = make_float3(1e30f, 1e30f, 1e30f);
    float3 aabbMax = make_float3(-1e30f, -1e30f, -1e30f);
    for (uint32_t v = 0; v < sA.vertexCount; ++v) {
        double3 pCan = verticesA[sA.firstVertex + v];
        float3 pWorld = make_float3(
            float(double(domainMin.x) + pCan.x * invBaseScale),
            float(double(domainMin.y) + pCan.y * invBaseScale),
            float(double(domainMin.z) + pCan.z * invBaseScale)
        );
        aabbMin.x = fminf(aabbMin.x, pWorld.x);
        aabbMin.y = fminf(aabbMin.y, pWorld.y);
        aabbMin.z = fminf(aabbMin.z, pWorld.z);
        aabbMax.x = fmaxf(aabbMax.x, pWorld.x);
        aabbMax.y = fmaxf(aabbMax.y, pWorld.y);
        aabbMax.z = fmaxf(aabbMax.z, pWorld.z);
    }

    RestrictedSurfaceGeometry gA = geomA[idxA];
    double3 nA = normalize3(make_double3(gA.plane.x, gA.plane.y, gA.plane.z));

    float margin = 16.0f * FLT_EPSILON * coordScale;
    float3 qMin = make_float3(aabbMin.x - float(contactRange) - margin,
                              aabbMin.y - float(contactRange) - margin,
                              aabbMin.z - float(contactRange) - margin);
    float3 qMax = make_float3(aabbMax.x + float(contactRange) + margin,
                              aabbMax.y + float(contactRange) + margin,
                              aabbMax.z + float(contactRange) + margin);
    float maxDist = float(contactRange) + margin;
    float maxDistSq = maxDist * maxDist;

    uint32_t stack[64];
    int stackPtr = 0;
    if (geomB.root != cudaGeometry::InvalidNodeId && geomB.nodes != nullptr) {
        stack[stackPtr++] = geomB.root;
    }

    uint32_t foundCount = 0;
    uint32_t dstBase = emitPass ? offsetsA[idxA] : 0;

    while (stackPtr > 0) {
        uint32_t nodeIdx = stack[--stackPtr];
        const cudaGeometry::TriangleBvhNode node = geomB.nodes[nodeIdx];

        if (node.maximum.x < qMin.x || node.minimum.x > qMax.x ||
            node.maximum.y < qMin.y || node.minimum.y > qMax.y ||
            node.maximum.z < qMin.z || node.minimum.z > qMax.z) {
            continue;
        }

        if (cudaGeometry::aabbDistanceSquared(node.minimum, node.maximum, aabbMin, aabbMax) > maxDistSq) {
            continue;
        }

        if (node.isLeaf()) {
            uint32_t triB = node.triangleId();
            int2 range = findPatchRangeForTriangle(sortedTrianglesB, patchCountB, triB);
            for (int r = 0; r < range.y; ++r) {
                uint32_t patchBIdx = sortedPatchIndicesB[range.x + r];
                RestrictedSurfaceGeometry gB = geomBData[patchBIdx];
                double3 nB = normalize3(make_double3(gB.plane.x, gB.plane.y, gB.plane.z));

                if (dot3(nA, nB) < -0.2) {
                    if (emitPass) {
                        outCandidatePairs[dstBase + foundCount] = make_uint2(surfBaseA + idxA, surfBaseB + patchBIdx);
                    }
                    foundCount++;
                }
            }
        } else {
            if (stackPtr < 62) {
                stack[stackPtr++] = node.leftChild();
                stack[stackPtr++] = node.rightChild();
            }
        }
    }

    if (!emitPass) {
        outCountsA[idxA] = foundCount;
    }
}

static __device__ inline bool clipMortarPair(
    uint32_t vCountA, const double3* vertsA, double3 nA, double3 centA,
    uint32_t vCountB, const double3* vertsB, double3 nB, double3 centB,
    double contactRange, double gPenTol,
    MortarOverlap& outOverlap,
    MortarBasis& outBasis,
    double2* outVertices,
    int& outVertexCount,
    int* outCapacityError = nullptr)
{
    if (vCountA < 3 || vCountB < 3) return false;
    if (vCountA > MaxMortarScratchVertices || vCountB > MaxMortarScratchVertices) {
        if (outCapacityError) atomicExch(outCapacityError, 1);
        return false;
    }

    double3 m = nA;
    double dotAm = 1.0;
    double dotBm = dot3(nB, m);
    if (dotBm > -0.5) return false;

    double3 ref = (fabs(m.x) < 0.9) ? make_double3(1.0, 0.0, 0.0) : make_double3(0.0, 1.0, 0.0);
    double3 uCross = cross3(ref, m);
    double uLen = length3(uCross);
    if (uLen <= 1e-14) return false;
    double3 u = mul3(uCross, 1.0 / uLen);
    double3 v = cross3(m, u);

    double3 pm = centA;

    double2 polyA[MaxMortarScratchVertices];
    for (uint32_t k = 0; k < vCountA; ++k) {
        double3 d = sub3(vertsA[k], pm);
        polyA[k] = make_double2(dot3(d, u), dot3(d, v));
    }
    if (polygonSignedArea2D(polyA, int(vCountA)) < 0.0) {
        reversePolygon2D(polyA, int(vCountA));
    }

    double2 polyB[MaxMortarScratchVertices];
    for (uint32_t k = 0; k < vCountB; ++k) {
        double3 d = sub3(vertsB[k], pm);
        polyB[k] = make_double2(dot3(d, u), dot3(d, v));
    }
    if (polygonSignedArea2D(polyB, int(vCountB)) < 0.0) {
        reversePolygon2D(polyB, int(vCountB));
    }

    double2 scratch1[MaxMortarScratchVertices];
    double2 scratch2[MaxMortarScratchVertices];
    int curCount = int(vCountA);
    const double2* src = polyA;
    double2* dst = scratch1;

    for (uint32_t e = 0; e < vCountB; ++e) {
        uint32_t eNext = (e + 1 == vCountB) ? 0 : e + 1;
        double2 b0 = polyB[e];
        double2 b1 = polyB[eNext];
        double aLine = b0.y - b1.y;
        double bLine = b1.x - b0.x;
        double cLine = b1.y * b0.x - b1.x * b0.y;

        int newCount = clipPolygon2DByHalfPlane(src, curCount, aLine, bLine, cLine, dst, outCapacityError);
        if (newCount < 3) return false;
        curCount = newCount;
        src = dst;
        dst = (dst == scratch1) ? scratch2 : scratch1;
    }

    double g0 = dot3(nB, sub3(centB, pm)) / dotBm - dot3(nA, sub3(centA, pm)) / dotAm;
    double gu = -dot3(nB, u) / dotBm + dot3(nA, u) / dotAm;
    double gv = -dot3(nB, v) / dotBm + dot3(nA, v) / dotAm;

    double gradG = sqrt(gu * gu + gv * gv);
    if (gradG > 1e-12) {
        int newCount = clipPolygon2DByHalfPlane(src, curCount, gu, gv, g0 + gPenTol, dst, outCapacityError);
        if (newCount < 3) return false;
        curCount = newCount;
        src = dst;
        dst = (dst == scratch1) ? scratch2 : scratch1;

        newCount = clipPolygon2DByHalfPlane(src, curCount, -gu, -gv, contactRange - g0, dst, outCapacityError);
        if (newCount < 3) return false;
        curCount = newCount;
        src = dst;
        dst = (dst == scratch1) ? scratch2 : scratch1;
    } else {
        if (g0 < -gPenTol || g0 > contactRange) return false;
    }

    double2 centroid2D;
    double Am = polygonAreaAndCentroid2D(src, curCount, centroid2D);
    if (Am <= 1e-14) return false;

    double3 Cm = add3(pm, add3(mul3(u, centroid2D.x), mul3(v, centroid2D.y)));
    double areaA = Am / fabs(dotAm);
    double areaB = Am / fabs(dotBm);

    double zA = (dot3(nA, sub3(centA, pm)) - centroid2D.x * dot3(nA, u) - centroid2D.y * dot3(nA, v)) / dotAm;
    double zB = (dot3(nB, sub3(centB, pm)) - centroid2D.x * dot3(nB, u) - centroid2D.y * dot3(nB, v)) / dotBm;

    double3 CA = add3(Cm, mul3(m, zA));
    double3 CB = add3(Cm, mul3(m, zB));

    outOverlap.couplingArea = Am;
    outOverlap.areaA = areaA;
    outOverlap.areaB = areaB;
    outOverlap.momentA = mul3(CA, areaA);
    outOverlap.momentB = mul3(CB, areaB);

    outBasis.origin = pm;
    outBasis.u = u;
    outBasis.v = v;
    outBasis.normal = m;

    if (outVertices != nullptr) {
        for (int k = 0; k < curCount; ++k) {
            outVertices[k] = src[k];
        }
    }
    outVertexCount = curCount;
    return true;
}

static __global__ void countAndEmitMortarOverlapsKernel(
    const uint2* candidatePairs,
    uint32_t candidateCount,
    const DeviceModelInfo* models,
    uint32_t modelCount,
    double contactRange,
    double gPenTol,
    float3 domainMin,
    double invBaseScale,
    const uint32_t* overlapOffsets,
    const uint32_t* vertexOffsets,
    uint32_t* outOverlapValid,
    uint32_t* outVertexCounts,
    MortarOverlap* outAcceptedOverlaps,
    MortarBasis* outBases,
    double2* outVertices,
    bool emitPass,
    int* outCapacityError)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= candidateCount) return;

    uint2 pair = candidatePairs[idx];
    uint32_t globalA = pair.x;
    uint32_t globalB = pair.y;

    const DeviceModelInfo* infoA = nullptr;
    const DeviceModelInfo* infoB = nullptr;
    for (uint32_t m = 0; m < modelCount; ++m) {
        if (globalA >= models[m].surfaceBase && globalA < models[m].surfaceBase + models[m].totalPatches) {
            infoA = &models[m];
        }
        if (globalB >= models[m].surfaceBase && globalB < models[m].surfaceBase + models[m].totalPatches) {
            infoB = &models[m];
        }
    }
    if (!infoA || !infoB) {
        if (!emitPass) {
            outOverlapValid[idx] = 0;
            outVertexCounts[idx] = 0;
        }
        return;
    }

    uint32_t localA = globalA - infoA->surfaceBase;
    uint32_t localB = globalB - infoB->surfaceBase;

    RestrictedSurface sA = infoA->surfaces[localA];
    RestrictedSurface sB = infoB->surfaces[localB];
    RestrictedSurfaceGeometry gA = infoA->geom[localA];
    RestrictedSurfaceGeometry gB = infoB->geom[localB];

    double3 nA = normalize3(make_double3(gA.plane.x, gA.plane.y, gA.plane.z));
    double3 nB = normalize3(make_double3(gB.plane.x, gB.plane.y, gB.plane.z));

    double3 centA = make_double3(
        double(domainMin.x) + gA.centroidArea.x * invBaseScale,
        double(domainMin.y) + gA.centroidArea.y * invBaseScale,
        double(domainMin.z) + gA.centroidArea.z * invBaseScale
    );
    double3 centB = make_double3(
        double(domainMin.x) + gB.centroidArea.x * invBaseScale,
        double(domainMin.y) + gB.centroidArea.y * invBaseScale,
        double(domainMin.z) + gB.centroidArea.z * invBaseScale
    );

    double3 vertsA[MaxMortarScratchVertices];
    for (uint32_t k = 0; k < sA.vertexCount && k < MaxMortarScratchVertices; ++k) {
        double3 pCan = infoA->vertices[sA.firstVertex + k];
        vertsA[k] = make_double3(
            double(domainMin.x) + pCan.x * invBaseScale,
            double(domainMin.y) + pCan.y * invBaseScale,
            double(domainMin.z) + pCan.z * invBaseScale
        );
    }

    double3 vertsB[MaxMortarScratchVertices];
    for (uint32_t k = 0; k < sB.vertexCount && k < MaxMortarScratchVertices; ++k) {
        double3 pCan = infoB->vertices[sB.firstVertex + k];
        vertsB[k] = make_double3(
            double(domainMin.x) + pCan.x * invBaseScale,
            double(domainMin.y) + pCan.y * invBaseScale,
            double(domainMin.z) + pCan.z * invBaseScale
        );
    }

    MortarOverlap overlap{};
    MortarBasis basis{};
    int vCount = 0;
    double2* vertDst = emitPass ? (outVertices + vertexOffsets[idx]) : nullptr;

    bool valid = clipMortarPair(
        sA.vertexCount, vertsA, nA, centA,
        sB.vertexCount, vertsB, nB, centB,
        contactRange, gPenTol,
        overlap, basis, vertDst, vCount,
        outCapacityError
    );

    if (!emitPass) {
        outOverlapValid[idx] = valid ? 1 : 0;
        outVertexCounts[idx] = valid ? uint32_t(vCount) : 0;
    } else {
        if (valid) {
            uint32_t dstOv = overlapOffsets[idx];
            overlap.surfaceA = globalA;
            overlap.surfaceB = globalB;
            outAcceptedOverlaps[dstOv] = overlap;
            outBases[dstOv] = basis;
            outVertexCounts[dstOv] = uint32_t(vCount);
        }
    }
}

static __global__ void verifyOwnershipSideKernel(
    const uint32_t* sortedKeys,
    const uint32_t* sortedIndices,
    uint32_t overlapCount,
    uint32_t sideIndex,
    const MortarOverlap* overlaps,
    const MortarBasis* bases,
    const double2* vertices,
    const uint32_t* vertexOffsets,
    const uint32_t* vertexCounts,
    const DeviceModelInfo* models,
    uint32_t modelCount,
    float3 domainMin,
    double invBaseScale,
    float coordScale,
    int* conflictFound,
    double2* conflictMeasure)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= overlapCount) return;

    uint32_t currKey = sortedKeys[idx];
    if (idx > 0 && sortedKeys[idx - 1] == currKey) return; // Only process at run start

    int runLength = 1;
    while (idx + runLength < overlapCount && sortedKeys[idx + runLength] == currKey) {
        runLength++;
    }
    if (runLength <= 1) return;

    const DeviceModelInfo* info = nullptr;
    for (uint32_t m = 0; m < modelCount; ++m) {
        if (currKey >= models[m].surfaceBase && currKey < models[m].surfaceBase + models[m].totalPatches) {
            info = &models[m];
            break;
        }
    }
    if (!info) return;

    uint32_t localPatch = currKey - info->surfaceBase;
    RestrictedSurfaceGeometry g = info->geom[localPatch];
    double3 nP = normalize3(make_double3(g.plane.x, g.plane.y, g.plane.z));
    double3 cP = make_double3(
        double(domainMin.x) + g.centroidArea.x * invBaseScale,
        double(domainMin.y) + g.centroidArea.y * invBaseScale,
        double(domainMin.z) + g.centroidArea.z * invBaseScale
    );

    double3 ref = (fabs(nP.x) < 0.9) ? make_double3(1.0, 0.0, 0.0) : make_double3(0.0, 1.0, 0.0);
    double3 uP = normalize3(cross3(ref, nP));
    double3 vP = cross3(nP, uP);

    double invScale2 = invBaseScale * invBaseScale;
    double patchArea = g.centroidArea.w * invScale2;

    for (int i = 0; i < runLength; ++i) {
        uint32_t ovIdx1 = sortedIndices[idx + i];
        uint32_t k1 = vertexCounts[ovIdx1];
        if (k1 < 3 || k1 > MaxMortarScratchVertices) continue;
        uint32_t off1 = vertexOffsets[ovIdx1];
        MortarBasis b1 = bases[ovIdx1];

        double denom1 = dot3(nP, b1.normal);
        if (fabs(denom1) < 1e-4) continue;

        double2 poly1[MaxMortarScratchVertices];
        for (uint32_t v = 0; v < k1; ++v) {
            double2 p2D = vertices[off1 + v];
            double3 p3D = add3(b1.origin, add3(mul3(b1.u, p2D.x), mul3(b1.v, p2D.y)));
            double t1 = dot3(nP, sub3(cP, p3D)) / denom1;
            double3 pPatch = add3(p3D, mul3(b1.normal, t1));
            double3 d = sub3(pPatch, cP);
            poly1[v] = make_double2(dot3(d, uP), dot3(d, vP));
        }
        if (polygonSignedArea2D(poly1, int(k1)) < 0.0) {
            reversePolygon2D(poly1, int(k1));
        }

        double perim1 = 0.0;
        for (uint32_t v = 0; v < k1; ++v) {
            uint32_t next = (v + 1 == k1) ? 0 : v + 1;
            double dx = poly1[next].x - poly1[v].x;
            double dy = poly1[next].y - poly1[v].y;
            perim1 += sqrt(dx * dx + dy * dy);
        }

        for (int j = i + 1; j < runLength; ++j) {
            uint32_t ovIdx2 = sortedIndices[idx + j];
            uint32_t k2 = vertexCounts[ovIdx2];
            if (k2 < 3 || k2 > MaxMortarScratchVertices) continue;
            uint32_t off2 = vertexOffsets[ovIdx2];
            MortarBasis b2 = bases[ovIdx2];

            double denom2 = dot3(nP, b2.normal);
            if (fabs(denom2) < 1e-4) continue;

            double2 poly2[MaxMortarScratchVertices];
            for (uint32_t v = 0; v < k2; ++v) {
                double2 p2D = vertices[off2 + v];
                double3 p3D = add3(b2.origin, add3(mul3(b2.u, p2D.x), mul3(b2.v, p2D.y)));
                double t2 = dot3(nP, sub3(cP, p3D)) / denom2;
                double3 pPatch = add3(p3D, mul3(b2.normal, t2));
                double3 d = sub3(pPatch, cP);
                poly2[v] = make_double2(dot3(d, uP), dot3(d, vP));
            }
            if (polygonSignedArea2D(poly2, int(k2)) < 0.0) {
                reversePolygon2D(poly2, int(k2));
            }

            double kappa = fmax(1.0 / fabs(denom1), 1.0 / fabs(denom2));
            double tol = kappa * (16.0 * FLT_EPSILON * coordScale * perim1)
                       + 64.0 * FLT_EPSILON * patchArea
                       + 1e-12 * coordScale * coordScale;

            double2 scratch1[MaxMortarScratchVertices];
            double2 scratch2[MaxMortarScratchVertices];
            int curCount = int(k1);
            const double2* src = poly1;
            double2* dst = scratch1;

            for (uint32_t e = 0; e < k2; ++e) {
                uint32_t eNext = (e + 1 == k2) ? 0 : e + 1;
                double2 b0 = poly2[e];
                double2 b1 = poly2[eNext];
                double aLine = b0.y - b1.y;
                double bLine = b1.x - b0.x;
                double cLine = b1.y * b0.x - b1.x * b0.y;

                int newCount = clipPolygon2DByHalfPlane(src, curCount, aLine, bLine, cLine, dst);
                if (newCount < 3) { curCount = 0; break; }
                curCount = newCount;
                src = dst;
                dst = (dst == scratch1) ? scratch2 : scratch1;
            }

            if (curCount >= 3) {
                double interArea = fabs(polygonSignedArea2D(src, curCount));
                if (interArea > tol) {
                    if (atomicCAS(conflictFound, 0, 1) == 0) {
                        uint32_t oppSurf1 = (sideIndex == 0) ? overlaps[ovIdx1].surfaceB : overlaps[ovIdx1].surfaceA;
                        uint32_t oppSurf2 = (sideIndex == 0) ? overlaps[ovIdx2].surfaceB : overlaps[ovIdx2].surfaceA;

                        RestrictedSurfaceGeometry oppGeom1{};
                        bool oppGeom1Found = false;
                        for (uint32_t m = 0; m < modelCount; ++m) {
                            if (oppSurf1 >= models[m].surfaceBase && oppSurf1 < models[m].surfaceBase + models[m].totalPatches) {
                                oppGeom1 = models[m].geom[oppSurf1 - models[m].surfaceBase];
                                oppGeom1Found = true;
                                break;
                            }
                        }

                        RestrictedSurfaceGeometry oppGeom2{};
                        bool oppGeom2Found = false;
                        for (uint32_t m = 0; m < modelCount; ++m) {
                            if (oppSurf2 >= models[m].surfaceBase && oppSurf2 < models[m].surfaceBase + models[m].totalPatches) {
                                oppGeom2 = models[m].geom[oppSurf2 - models[m].surfaceBase];
                                oppGeom2Found = true;
                                break;
                            }
                        }

                        double singleDirArea = 0.0;
                        if (oppGeom1Found && oppGeom2Found) {
                            double3 nOpp1 = normalize3(make_double3(oppGeom1.plane.x, oppGeom1.plane.y, oppGeom1.plane.z));
                            double3 cOpp1 = make_double3(
                                double(domainMin.x) + oppGeom1.centroidArea.x * invBaseScale,
                                double(domainMin.y) + oppGeom1.centroidArea.y * invBaseScale,
                                double(domainMin.z) + oppGeom1.centroidArea.z * invBaseScale
                            );
                            double denomOpp1 = dot3(nOpp1, b1.normal);
                            for (uint32_t v = 0; v < k1; ++v) {
                                double2 p2D = vertices[off1 + v];
                                double3 p3D = add3(b1.origin, add3(mul3(b1.u, p2D.x), mul3(b1.v, p2D.y)));
                                double tB1 = (fabs(denomOpp1) > 1e-6) ? (dot3(nOpp1, sub3(cOpp1, p3D)) / denomOpp1) : 0.0;
                                double3 pOpp1 = add3(p3D, mul3(b1.normal, tB1));
                                double3 d1 = sub3(pOpp1, cP);
                                poly1[v] = make_double2(dot3(d1, uP), dot3(d1, vP));
                            }
                            if (polygonSignedArea2D(poly1, int(k1)) < 0.0) {
                                reversePolygon2D(poly1, int(k1));
                            }

                            double3 nOpp2 = normalize3(make_double3(oppGeom2.plane.x, oppGeom2.plane.y, oppGeom2.plane.z));
                            double3 cOpp2 = make_double3(
                                double(domainMin.x) + oppGeom2.centroidArea.x * invBaseScale,
                                double(domainMin.y) + oppGeom2.centroidArea.y * invBaseScale,
                                double(domainMin.z) + oppGeom2.centroidArea.z * invBaseScale
                            );
                            double denomOpp2 = dot3(nOpp2, b2.normal);
                            for (uint32_t v = 0; v < k2; ++v) {
                                double2 p2D = vertices[off2 + v];
                                double3 p3D = add3(b2.origin, add3(mul3(b2.u, p2D.x), mul3(b2.v, p2D.y)));
                                double tB2 = (fabs(denomOpp2) > 1e-6) ? (dot3(nOpp2, sub3(cOpp2, p3D)) / denomOpp2) : 0.0;
                                double3 pOpp2 = add3(p3D, mul3(b2.normal, tB2));
                                double3 d2 = sub3(pOpp2, cP);
                                poly2[v] = make_double2(dot3(d2, uP), dot3(d2, vP));
                            }
                            if (polygonSignedArea2D(poly2, int(k2)) < 0.0) {
                                reversePolygon2D(poly2, int(k2));
                            }

                            int testCount = int(k1);
                            const double2* testSrc = poly1;
                            double2* testDst = scratch1;

                            for (uint32_t e = 0; e < k2; ++e) {
                                uint32_t eNext = (e + 1 == k2) ? 0 : e + 1;
                                double2 b0 = poly2[e];
                                double2 b1p = poly2[eNext];
                                double aLine = b0.y - b1p.y;
                                double bLine = b1p.x - b0.x;
                                double cLine = b1p.y * b0.x - b1p.x * b0.y;

                                int newCount = clipPolygon2DByHalfPlane(testSrc, testCount, aLine, bLine, cLine, testDst);
                                if (newCount < 3) { testCount = 0; break; }
                                testCount = newCount;
                                testSrc = testDst;
                                testDst = (testDst == scratch1) ? scratch2 : scratch1;
                            }

                            if (testCount >= 3) {
                                singleDirArea = fabs(polygonSignedArea2D(testSrc, testCount));
                            }
                        }
                        *conflictMeasure = make_double2(singleDirArea, tol);
                    }
                }
            }
        }
    }
}

// Exposure accounting kernels

static __global__ void accumulateMortarExposureKernel(
    const MortarOverlap* overlaps,
    uint32_t overlapCount,
    double4* claimedAreaMoment)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= overlapCount) return;

    MortarOverlap ov = overlaps[idx];

    atomicAdd(&claimedAreaMoment[ov.surfaceA].w, ov.areaA);
    atomicAdd(&claimedAreaMoment[ov.surfaceA].x, ov.momentA.x);
    atomicAdd(&claimedAreaMoment[ov.surfaceA].y, ov.momentA.y);
    atomicAdd(&claimedAreaMoment[ov.surfaceA].z, ov.momentA.z);

    atomicAdd(&claimedAreaMoment[ov.surfaceB].w, ov.areaB);
    atomicAdd(&claimedAreaMoment[ov.surfaceB].x, ov.momentB.x);
    atomicAdd(&claimedAreaMoment[ov.surfaceB].y, ov.momentB.y);
    atomicAdd(&claimedAreaMoment[ov.surfaceB].z, ov.momentB.z);
}

static __global__ void finalizeSurfaceExposureKernel(
    const RestrictedSurfaceGeometry* geom,
    uint32_t patchCount,
    uint32_t surfaceBase,
    double invBaseScale,
    float3 domainMin,
    float coordScale,
    const double4* claimedAreaMoment,
    double4* outDevExposure)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= patchCount) return;

    uint32_t globalIdx = surfaceBase + idx;
    RestrictedSurfaceGeometry g = geom[idx];

    double invScale2 = invBaseScale * invBaseScale;
    double totalArea = g.centroidArea.w * invScale2;

    double3 canCent = make_double3(g.centroidArea.x, g.centroidArea.y, g.centroidArea.z);
    double3 worldCent = make_double3(
        double(domainMin.x) + canCent.x * invBaseScale,
        double(domainMin.y) + canCent.y * invBaseScale,
        double(domainMin.z) + canCent.z * invBaseScale
    );
    double3 totalMoment = mul3(worldCent, totalArea);

    const double4 claimed = claimedAreaMoment[globalIdx];
    double Acontact = claimed.w;
    double3 Mcontact = make_double3(claimed.x, claimed.y, claimed.z);

    if (Acontact > totalArea) {
        // Clamp area while keeping partial-contact centroid to prevent replacing with whole patch
        Acontact = totalArea;
    }

    double Aexposed = fmax(0.0, totalArea - Acontact);
    double3 Mexposed;
    if (Aexposed <= 256.0 * DBL_EPSILON * totalArea) {
        Aexposed = 0.0;
        Mexposed = make_double3(0.0, 0.0, 0.0);
    } else {
        Mexposed = sub3(totalMoment, Mcontact);
    }

    outDevExposure[globalIdx] = make_double4(Mexposed.x, Mexposed.y, Mexposed.z, Aexposed);
}

static __global__ void mapOverlapsToCouplingKeysKernel(
    const MortarOverlap* overlaps,
    uint32_t overlapCount,
    const uint32_t* contactFaceBySurface,
    uint64_t* outKeys,
    double* outValues)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= overlapCount) return;

    MortarOverlap ov = overlaps[idx];
    uint32_t faceA = contactFaceBySurface[ov.surfaceA];
    uint32_t faceB = contactFaceBySurface[ov.surfaceB];

    if (faceA != InvalidId && faceB != InvalidId) {
        uint32_t fMin = min(faceA, faceB);
        uint32_t fMax = max(faceA, faceB);
        outKeys[idx] = (uint64_t(fMin) << 32) | uint64_t(fMax);
        outValues[idx] = ov.couplingArea;
    } else {
        outKeys[idx] = 0xFFFFFFFFFFFFFFFFull;
        outValues[idx] = 0.0;
    }
}

bool buildMortarOverlaps(
    const std::vector<ModelRestriction>& models,
    const std::vector<cudaGeometry::DeviceModelGeometryView>& geometries,
    const BaseCellDomain& baseDomain,
    double contactRange,
    cudaUtils::CudaBuffer<MortarOverlap>& devOverlaps,
    uint32_t& overlapCount,
    cudaUtils::CudaBuffer<double4>& devExposure,
    cudaStream_t stream)
{
    overlapCount = 0;
    const uint32_t modelCount = uint32_t(models.size());
    if (modelCount == 0) return true;

    uint32_t totalSurfacePatches = 0;
    for (uint32_t m = 0; m < modelCount; ++m) {
        totalSurfacePatches += models[m].totalPatches;
    }

    if (!ensureCapacity(devExposure, totalSurfacePatches)) return false;

    const float3 domainMin = make_float3(baseDomain.domainMin.x, baseDomain.domainMin.y, baseDomain.domainMin.z);

    for (uint32_t m = 0; m < modelCount; ++m) {
        if (models[m].totalPatches == 0) continue;
        uint32_t blocks = (models[m].totalPatches + BlockSize - 1) / BlockSize;
        initSurfaceExposureKernel<<<blocks, BlockSize, 0, stream>>>(
            models[m].devSurfaceGeometry.get(),
            models[m].totalPatches,
            models[m].surfaceBase,
            baseDomain.invBaseScale,
            domainMin,
            devExposure.get()
        );
    }
    if (cudaGetLastError() != cudaSuccess) return false;

    if (modelCount < 2 || totalSurfacePatches == 0) return true;

    float coordScale = glm::length(baseDomain.domainMax - baseDomain.domainMin);
    if (coordScale <= 0.0f) coordScale = 1.0f;
    double gPenTol = 256.0 * DBL_EPSILON * double(coordScale);

    // Prepare DeviceModelInfo array
    std::vector<DeviceModelInfo> hostModelInfos(modelCount);
    for (uint32_t m = 0; m < modelCount; ++m) {
        hostModelInfos[m].modelIndex = m;
        hostModelInfos[m].surfaceBase = models[m].surfaceBase;
        hostModelInfos[m].totalPatches = models[m].totalPatches;
        hostModelInfos[m].geom = models[m].devSurfaceGeometry.get();
        hostModelInfos[m].surfaces = models[m].devSurfaces.get();
        hostModelInfos[m].vertices = models[m].devRestrictedVertices.get();
    }
    cudaUtils::CudaBuffer<DeviceModelInfo> devModelInfos;
    if (!devModelInfos.allocate(modelCount) ||
        !devModelInfos.uploadAsync(hostModelInfos.data(), modelCount, stream)) {
        return false;
    }

    cudaUtils::CudaBuffer<int> devCapacityError;
    if (!devCapacityError.allocate(1)) return false;
    cudaMemsetAsync(devCapacityError.get(), 0, sizeof(int), stream);

    std::vector<MortarOverlap> allOverlaps;
    std::vector<MortarBasis> allBases;
    std::vector<double2> allVertices;
    std::vector<uint32_t> allVertexCounts;
    std::vector<uint32_t> allVertexOffsets;

    cudaUtils::CudaBuffer<uint8_t> cubScratch;

    for (uint32_t ma = 0; ma < modelCount; ++ma) {
        for (uint32_t mb = ma + 1; mb < modelCount; ++mb) {
            uint32_t patchesA = models[ma].totalPatches;
            uint32_t patchesB = models[mb].totalPatches;
            if (patchesA == 0 || patchesB == 0) continue;

            cudaUtils::CudaBuffer<uint32_t> devTriKeysIn, devTriKeysOut;
            cudaUtils::CudaBuffer<uint32_t> devPatchValsIn, devPatchValsOut;
            if (!devTriKeysIn.allocate(patchesB) || !devTriKeysOut.allocate(patchesB) ||
                !devPatchValsIn.allocate(patchesB) || !devPatchValsOut.allocate(patchesB)) {
                return false;
            }

            uint32_t bBlocks = (patchesB + BlockSize - 1) / BlockSize;
            extractPatchTrianglesKernel<<<bBlocks, BlockSize, 0, stream>>>(
                models[mb].devSurfaces.get(),
                patchesB,
                devTriKeysIn.get(),
                devPatchValsIn.get()
            );

            if (!sortPairs(devTriKeysIn.get(), devTriKeysOut.get(),
                           devPatchValsIn.get(), devPatchValsOut.get(),
                           int(patchesB), cubScratch, stream)) {
                return false;
            }

            cudaUtils::CudaBuffer<uint32_t> devCountsA, devOffsetsA;
            if (!devCountsA.allocate(patchesA) || !devOffsetsA.allocate(patchesA)) return false;

            uint32_t aBlocks = (patchesA + BlockSize - 1) / BlockSize;
            countAndEmitCandidatePatchPairsKernel<<<aBlocks, BlockSize, 0, stream>>>(
                models[ma].devSurfaces.get(),
                models[ma].devSurfaceGeometry.get(),
                models[ma].devRestrictedVertices.get(),
                patchesA,
                models[ma].surfaceBase,
                geometries[mb],
                devTriKeysOut.get(),
                devPatchValsOut.get(),
                models[mb].devSurfaceGeometry.get(),
                patchesB,
                models[mb].surfaceBase,
                contactRange,
                domainMin,
                baseDomain.invBaseScale,
                coordScale,
                nullptr,
                devCountsA.get(),
                nullptr,
                false
            );

            if (!exclusiveSum(devCountsA.get(), devOffsetsA.get(), int(patchesA), cubScratch, stream)) {
                return false;
            }

            uint32_t lastCount = 0, lastOffset = 0;
            cudaMemcpyAsync(&lastCount, devCountsA.get() + (patchesA - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&lastOffset, devOffsetsA.get() + (patchesA - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            uint32_t numCandidates = lastOffset + lastCount;
            if (numCandidates == 0) continue;

            cudaUtils::CudaBuffer<uint2> devCandidatePairs;
            if (!devCandidatePairs.allocate(numCandidates)) return false;

            countAndEmitCandidatePatchPairsKernel<<<aBlocks, BlockSize, 0, stream>>>(
                models[ma].devSurfaces.get(),
                models[ma].devSurfaceGeometry.get(),
                models[ma].devRestrictedVertices.get(),
                patchesA,
                models[ma].surfaceBase,
                geometries[mb],
                devTriKeysOut.get(),
                devPatchValsOut.get(),
                models[mb].devSurfaceGeometry.get(),
                patchesB,
                models[mb].surfaceBase,
                contactRange,
                domainMin,
                baseDomain.invBaseScale,
                coordScale,
                devOffsetsA.get(),
                nullptr,
                devCandidatePairs.get(),
                true
            );

            cudaUtils::CudaBuffer<uint32_t> devOverlapValid, devOverlapOffsets;
            cudaUtils::CudaBuffer<uint32_t> devVertexCounts, devVertexOffsets;
            if (!devOverlapValid.allocate(numCandidates) || !devOverlapOffsets.allocate(numCandidates) ||
                !devVertexCounts.allocate(numCandidates) || !devVertexOffsets.allocate(numCandidates)) {
                return false;
            }

            uint32_t cBlocks = (numCandidates + BlockSize - 1) / BlockSize;
            countAndEmitMortarOverlapsKernel<<<cBlocks, BlockSize, 0, stream>>>(
                devCandidatePairs.get(),
                numCandidates,
                devModelInfos.get(),
                modelCount,
                contactRange,
                gPenTol,
                domainMin,
                baseDomain.invBaseScale,
                nullptr,
                nullptr,
                devOverlapValid.get(),
                devVertexCounts.get(),
                nullptr,
                nullptr,
                nullptr,
                false,
                devCapacityError.get()
            );

            if (!exclusiveSum(devOverlapValid.get(), devOverlapOffsets.get(), int(numCandidates), cubScratch, stream)) {
                return false;
            }
            if (!exclusiveSum(devVertexCounts.get(), devVertexOffsets.get(), int(numCandidates), cubScratch, stream)) {
                return false;
            }

            uint32_t lastValid = 0, lastValidOff = 0;
            uint32_t lastVertCount = 0, lastVertOff = 0;
            cudaMemcpyAsync(&lastValid, devOverlapValid.get() + (numCandidates - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&lastValidOff, devOverlapOffsets.get() + (numCandidates - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&lastVertCount, devVertexCounts.get() + (numCandidates - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&lastVertOff, devVertexOffsets.get() + (numCandidates - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            int capacityError = 0;
            cudaMemcpyAsync(&capacityError, devCapacityError.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (capacityError != 0) {
                std::cerr << "[RVDContact] Fatal error: Mortar scratch polygon capacity exceeded (MaxMortarScratchVertices = "
                          << MaxMortarScratchVertices << "). Clipping buffer overflowed!\n";
                return false;
            }

            uint32_t acceptedOverlaps = lastValidOff + lastValid;
            uint32_t acceptedVertices = lastVertOff + lastVertCount;
            if (acceptedOverlaps == 0) continue;

            cudaUtils::CudaBuffer<MortarOverlap> curAcceptedOverlaps;
            cudaUtils::CudaBuffer<MortarBasis> curBases;
            cudaUtils::CudaBuffer<uint32_t> curOverlapVertexCounts;
            cudaUtils::CudaBuffer<double2> curVertices;

            if (!curAcceptedOverlaps.allocate(acceptedOverlaps) ||
                !curBases.allocate(acceptedOverlaps) ||
                !curOverlapVertexCounts.allocate(acceptedOverlaps) ||
                !curVertices.allocate(acceptedVertices)) {
                return false;
            }

            countAndEmitMortarOverlapsKernel<<<cBlocks, BlockSize, 0, stream>>>(
                devCandidatePairs.get(),
                numCandidates,
                devModelInfos.get(),
                modelCount,
                contactRange,
                gPenTol,
                domainMin,
                baseDomain.invBaseScale,
                devOverlapOffsets.get(),
                devVertexOffsets.get(),
                devOverlapValid.get(),
                curOverlapVertexCounts.get(),
                curAcceptedOverlaps.get(),
                curBases.get(),
                curVertices.get(),
                true,
                devCapacityError.get()
            );
            cudaStreamSynchronize(stream);

            std::vector<MortarOverlap> pairOverlaps(acceptedOverlaps);
            std::vector<MortarBasis> pairBases(acceptedOverlaps);
            std::vector<uint32_t> pairCounts(acceptedOverlaps);
            std::vector<double2> pairVertices(acceptedVertices);

            if (!curAcceptedOverlaps.downloadAsync(pairOverlaps.data(), acceptedOverlaps, stream) ||
                !curBases.downloadAsync(pairBases.data(), acceptedOverlaps, stream) ||
                !curOverlapVertexCounts.downloadAsync(pairCounts.data(), acceptedOverlaps, stream) ||
                !curVertices.downloadAsync(pairVertices.data(), acceptedVertices, stream) ||
                cudaStreamSynchronize(stream) != cudaSuccess) {
                return false;
            }

            uint32_t baseVertOffset = uint32_t(allVertices.size());
            for (uint32_t i = 0; i < acceptedOverlaps; ++i) {
                allOverlaps.push_back(pairOverlaps[i]);
                allBases.push_back(pairBases[i]);
                allVertexCounts.push_back(pairCounts[i]);
                allVertexOffsets.push_back(baseVertOffset);
                baseVertOffset += pairCounts[i];
            }
            allVertices.insert(allVertices.end(), pairVertices.begin(), pairVertices.end());
        }
    }

    overlapCount = uint32_t(allOverlaps.size());
    if (overlapCount == 0) return true;

    if (!ensureCapacity(devOverlaps, overlapCount) ||
        !devOverlaps.uploadAsync(allOverlaps.data(), overlapCount, stream) ||
        !ensureCapacity(mortarBases, overlapCount) ||
        !mortarBases.uploadAsync(allBases.data(), overlapCount, stream) ||
        !ensureCapacity(overlapVertexCounts, overlapCount) ||
        !overlapVertexCounts.uploadAsync(allVertexCounts.data(), overlapCount, stream) ||
        !ensureCapacity(overlapVertexOffsets, overlapCount) ||
        !overlapVertexOffsets.uploadAsync(allVertexOffsets.data(), overlapCount, stream) ||
        !ensureCapacity(overlapVertices, allVertices.size()) ||
        !overlapVertices.uploadAsync(allVertices.data(), allVertices.size(), stream)) {
        return false;
    }

    cudaUtils::CudaBuffer<int> devConflictFound;
    cudaUtils::CudaBuffer<double2> devConflictMeasure;
    if (!devConflictFound.allocate(1) || !devConflictMeasure.allocate(1)) return false;
    cudaMemsetAsync(devConflictFound.get(), 0, sizeof(int), stream);
    cudaMemsetAsync(devConflictMeasure.get(), 0, sizeof(double2), stream);

    cudaUtils::CudaBuffer<uint32_t> devKeysAIn, devKeysAOut;
    cudaUtils::CudaBuffer<uint32_t> devIndicesAIn, devIndicesAOut;
    if (!devKeysAIn.allocate(overlapCount) || !devKeysAOut.allocate(overlapCount) ||
        !devIndicesAIn.allocate(overlapCount) || !devIndicesAOut.allocate(overlapCount)) return false;

    std::vector<uint32_t> hostKeysA(overlapCount);
    std::vector<uint32_t> hostIndicesA(overlapCount);
    for (uint32_t i = 0; i < overlapCount; ++i) {
        hostKeysA[i] = allOverlaps[i].surfaceA;
        hostIndicesA[i] = i;
    }
    devKeysAIn.uploadAsync(hostKeysA.data(), overlapCount, stream);
    devIndicesAIn.uploadAsync(hostIndicesA.data(), overlapCount, stream);

    if (!sortPairs(devKeysAIn.get(), devKeysAOut.get(),
                   devIndicesAIn.get(), devIndicesAOut.get(),
                   int(overlapCount), cubScratch, stream)) return false;

    uint32_t ovBlocks = (overlapCount + BlockSize - 1) / BlockSize;
    verifyOwnershipSideKernel<<<ovBlocks, BlockSize, 0, stream>>>(
        devKeysAOut.get(),
        devIndicesAOut.get(),
        overlapCount,
        0, // Side A
        devOverlaps.get(),
        mortarBases.get(),
        overlapVertices.get(),
        overlapVertexOffsets.get(),
        overlapVertexCounts.get(),
        devModelInfos.get(),
        modelCount,
        domainMin,
        baseDomain.invBaseScale,
        coordScale,
        devConflictFound.get(), devConflictMeasure.get()
    );

    std::vector<uint32_t> hostKeysB(overlapCount);
    for (uint32_t i = 0; i < overlapCount; ++i) {
        hostKeysB[i] = allOverlaps[i].surfaceB;
    }
    devKeysAIn.uploadAsync(hostKeysB.data(), overlapCount, stream);
    devIndicesAIn.uploadAsync(hostIndicesA.data(), overlapCount, stream);

    if (!sortPairs(devKeysAIn.get(), devKeysAOut.get(),
                   devIndicesAIn.get(), devIndicesAOut.get(),
                   int(overlapCount), cubScratch, stream)) return false;

    verifyOwnershipSideKernel<<<ovBlocks, BlockSize, 0, stream>>>(
        devKeysAOut.get(),
        devIndicesAOut.get(),
        overlapCount,
        1, // Side B
        devOverlaps.get(),
        mortarBases.get(),
        overlapVertices.get(),
        overlapVertexOffsets.get(),
        overlapVertexCounts.get(),
        devModelInfos.get(),
        modelCount,
        domainMin,
        baseDomain.invBaseScale,
        coordScale,
        devConflictFound.get(), devConflictMeasure.get()
    );

    int hostConflictFound = 0;
    double2 hostConflictMeasure = make_double2(0.0, 0.0);
    cudaMemcpyAsync(&hostConflictFound, devConflictFound.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&hostConflictMeasure, devConflictMeasure.get(), sizeof(double2), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (hostConflictFound != 0 && hostConflictMeasure.x > hostConflictMeasure.y) {
        std::cerr << "[RVDContact] geometric ownership conflict: intersection="
                  << hostConflictMeasure.x << " tolerance=" << hostConflictMeasure.y << std::endl;
        return false;
    }

    cudaUtils::CudaBuffer<double4> devClaimedAreaMoment;
    if (!devClaimedAreaMoment.allocate(totalSurfacePatches)) {
        return false;
    }
    cudaMemsetAsync(devClaimedAreaMoment.get(), 0, totalSurfacePatches * sizeof(double4), stream);

    accumulateMortarExposureKernel<<<ovBlocks, BlockSize, 0, stream>>>(
        devOverlaps.get(),
        overlapCount,
        devClaimedAreaMoment.get()
    );

    for (uint32_t m = 0; m < modelCount; ++m) {
        if (models[m].totalPatches == 0) continue;
        uint32_t blocks = (models[m].totalPatches + BlockSize - 1) / BlockSize;
        finalizeSurfaceExposureKernel<<<blocks, BlockSize, 0, stream>>>(
            models[m].devSurfaceGeometry.get(),
            models[m].totalPatches,
            models[m].surfaceBase,
            baseDomain.invBaseScale,
            domainMin,
            coordScale,
            devClaimedAreaMoment.get(),
            devExposure.get()
        );
    }

    return true;
}

bool finalizeContactCouplings(
    const cudaUtils::CudaBuffer<MortarOverlap>& devOverlaps,
    uint32_t overlapCount,
    const cudaUtils::CudaBuffer<uint32_t>& devContactFaceBySurface,
    RVDResult& result,
    cudaStream_t stream)
{
    result.contactCouplings.clear();
    result.contactMesh.clear();

    if (overlapCount == 0) {
        overlapVertices.reset();
        overlapVertexOffsets.reset();
        overlapVertexCounts.reset();
        mortarBases.reset();
        return true;
    }

    cudaUtils::CudaBuffer<uint64_t> devKeysIn, devKeysOut;
    cudaUtils::CudaBuffer<double> devValsIn, devValsOut;
    if (!devKeysIn.allocate(overlapCount) || !devKeysOut.allocate(overlapCount) ||
        !devValsIn.allocate(overlapCount) || !devValsOut.allocate(overlapCount)) {
        return false;
    }

    uint32_t blocks = (overlapCount + BlockSize - 1) / BlockSize;
    mapOverlapsToCouplingKeysKernel<<<blocks, BlockSize, 0, stream>>>(
        devOverlaps.get(),
        overlapCount,
        devContactFaceBySurface.get(),
        devKeysIn.get(),
        devValsIn.get()
    );

    cudaUtils::CudaBuffer<uint8_t> cubScratch;
    if (!sortPairs(devKeysIn.get(), devKeysOut.get(),
                   devValsIn.get(), devValsOut.get(),
                   int(overlapCount), cubScratch, stream)) {
        return false;
    }

    cudaUtils::CudaBuffer<uint64_t> devUniqueKeys;
    cudaUtils::CudaBuffer<double> devAggregatedAreas;
    cudaUtils::CudaBuffer<int> devNumRuns;
    if (!devUniqueKeys.allocate(overlapCount) ||
        !devAggregatedAreas.allocate(overlapCount) ||
        !devNumRuns.allocate(1)) {
        return false;
    }

    size_t reduceBytes = 0;
    if (cub::DeviceReduce::ReduceByKey(nullptr, reduceBytes,
                                       devKeysOut.get(), devUniqueKeys.get(),
                                       devValsOut.get(), devAggregatedAreas.get(),
                                       devNumRuns.get(), cub::Sum(), int(overlapCount), stream) != cudaSuccess) {
        return false;
    }
    if (!ensureCapacity(cubScratch, reduceBytes)) return false;
    if (cub::DeviceReduce::ReduceByKey(cubScratch.get(), reduceBytes,
                                       devKeysOut.get(), devUniqueKeys.get(),
                                       devValsOut.get(), devAggregatedAreas.get(),
                                       devNumRuns.get(), cub::Sum(), int(overlapCount), stream) != cudaSuccess) {
        return false;
    }

    int numRuns = 0;
    cudaMemcpyAsync(&numRuns, devNumRuns.get(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (numRuns > 0) {
        std::vector<uint64_t> hostKeys(numRuns);
        std::vector<double> hostAreas(numRuns);
        if (!devUniqueKeys.downloadAsync(hostKeys.data(), numRuns, stream) ||
            !devAggregatedAreas.downloadAsync(hostAreas.data(), numRuns, stream) ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
            return false;
        }

        for (int r = 0; r < numRuns; ++r) {
            uint64_t key = hostKeys[r];
            if (key == 0xFFFFFFFFFFFFFFFFull) continue;
            ContactCoupling cc{};
            cc.faceA = uint32_t(key >> 32);
            cc.faceB = uint32_t(key & 0xFFFFFFFFull);
            cc.couplingArea = hostAreas[r];
            result.contactCouplings.push_back(cc);
        }
    }

    if (overlapVertices.size() > 0 && overlapVertexCounts.size() >= overlapCount) {
        std::vector<uint32_t> hostCounts(overlapCount);
        std::vector<uint32_t> hostOffsets(overlapCount);
        std::vector<MortarBasis> hostBases(overlapCount);
        std::vector<double2> hostVertices(overlapVertices.size());

        if (overlapVertexCounts.downloadAsync(hostCounts.data(), overlapCount, stream) &&
            overlapVertexOffsets.downloadAsync(hostOffsets.data(), overlapCount, stream) &&
            mortarBases.downloadAsync(hostBases.data(), overlapCount, stream) &&
            overlapVertices.downloadAsync(hostVertices.data(), hostVertices.size(), stream) &&
            cudaStreamSynchronize(stream) == cudaSuccess) {

            for (uint32_t i = 0; i < overlapCount; ++i) {
                uint32_t K = hostCounts[i];
                if (K < 3) continue;
                uint32_t offset = hostOffsets[i];
                MortarBasis b = hostBases[i];

                double2 c2D;
                polygonAreaAndCentroid2D(hostVertices.data() + offset, int(K), c2D);

                glm::vec3 c3D = glm::vec3(
                    float(b.origin.x + c2D.x * b.u.x + c2D.y * b.v.x),
                    float(b.origin.y + c2D.x * b.u.y + c2D.y * b.v.y),
                    float(b.origin.z + c2D.x * b.u.z + c2D.y * b.v.z)
                );
                glm::vec3 n3D = glm::vec3(float(b.normal.x), float(b.normal.y), float(b.normal.z));

                for (uint32_t k = 0; k < K; ++k) {
                    uint32_t nextK = (k + 1 == K) ? 0 : k + 1;
                    double2 v0 = hostVertices[offset + k];
                    double2 v1 = hostVertices[offset + nextK];

                    glm::vec3 p0 = glm::vec3(
                        float(b.origin.x + v0.x * b.u.x + v0.y * b.v.x),
                        float(b.origin.y + v0.x * b.u.y + v0.y * b.v.y),
                        float(b.origin.z + v0.x * b.u.z + v0.y * b.v.z)
                    );
                    glm::vec3 p1 = glm::vec3(
                        float(b.origin.x + v1.x * b.u.x + v1.y * b.v.x),
                        float(b.origin.y + v1.x * b.u.y + v1.y * b.v.y),
                        float(b.origin.z + v1.x * b.u.z + v1.y * b.v.z)
                    );

                    ContactMeshVertex cv0{}, cv1{}, cv2{};
                    cv0.position = c3D;
                    cv0.normal = n3D;
                    cv1.position = p0;
                    cv1.normal = n3D;
                    cv2.position = p1;
                    cv2.normal = n3D;

                    result.contactMesh.push_back(cv0);
                    result.contactMesh.push_back(cv1);
                    result.contactMesh.push_back(cv2);
                }
            }
        }
    }

    overlapVertices.reset();
    overlapVertexOffsets.reset();
    overlapVertexCounts.reset();
    mortarBases.reset();

    return true;
}

} // namespace voronoi
