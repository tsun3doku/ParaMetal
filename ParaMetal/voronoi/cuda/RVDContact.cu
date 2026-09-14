#include "RVDContact.cuh"
#include "RVDModelQueries.cuh"
#include "RVDContactManifold.cuh"
#include "TriangleQueries.cuh"
#include "TriangleBvh.cuh"
#include "RVDBaseCells.cuh"
#include "KNNNeighborsKernels.cuh"
#include "../../cuda/CudaBuffer.cuh"

#include <cub/cub.cuh>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

namespace voronoi {

using cudaGeometry::canonicalFromWorld;
using cudaGeometry::DeviceVoxelGrid;
using cudaGeometry::ClosestSurfaceHit;
using cudaGeometry::closestPoint;
using cudaGeometry::DeviceModelGeometryView;
using cudaGeometry::sub3;
using cudaGeometry::add3;
using cudaGeometry::mul3;
using cudaGeometry::dot3;
using cudaGeometry::cross3;
using cudaGeometry::len3;

static constexpr uint32_t BlockSize = 256;

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

static __device__ uint32_t findPowerCellOwner(
    float3 worldPoint,
    uint32_t initialGuess,
    DeviceBaseCells cells,
    DeviceVoxelGrid baseGrid)
{
    if (initialGuess == InvalidId || initialGuess >= cells.cellCount) return InvalidId;

    const float3 canonicalPoint = canonicalFromWorld(baseGrid, worldPoint);
    uint32_t cell = initialGuess;

    for (uint32_t step = 0; step < cells.cellCount; ++step) {
        int worstPlane = -1;
        float worstD = -BasePlaneTolerance;

        const uint32_t pCount = basePlaneCount(cells, cell);
        for (uint32_t plane = 0; plane < pCount; ++plane) {
            const float4 pl = basePlane(cells, cell, plane);
            const float d = pl.x * canonicalPoint.x + pl.y * canonicalPoint.y + pl.z * canonicalPoint.z + pl.w;
            if (d < worstD) {
                worstD = d;
                worstPlane = int(plane);
            }
        }

        if (worstPlane < 0) return cell;

        const int neighbor = basePlaneNeighbor(cells, cell, worstPlane);
        if (neighbor < 0 || uint32_t(neighbor) >= cells.cellCount) return InvalidId;

        cell = uint32_t(neighbor);
    }
    return InvalidId;
}

static __device__ inline float triangleDiameter(float3 a, float3 b, float3 c) {
    const float dAB = len3(sub3(b, a));
    const float dBC = len3(sub3(c, b));
    const float dCA = len3(sub3(a, c));
    return fmaxf(dAB, fmaxf(dBC, dCA));
}

static __device__ inline bool sampleContactTopology(
    float3 p,
    DeviceModelGeometryView geomA,
    DeviceModelGeometryView geomB,
    const DeviceModelGeometryView* allModels,
    uint32_t modelCount,
    uint32_t modelIndexA,
    uint32_t modelIndexB,
    DeviceKnn seeds,
    DeviceBaseCells baseCells,
    DeviceVoxelGrid baseGrid,
    const uint32_t* fragByCellA,
    const uint32_t* fragByCellB,
    uint2& outTuple)
{
    const ClosestSurfaceHit hitA = closestPoint(p, geomA.triangles, geomA.nodes, geomA.parents, geomA.root);
    if (hitA.triangleId == InvalidId) return false;

    const ClosestSurfaceHit hitB = closestPoint(p, geomB.triangles, geomB.nodes, geomB.parents, geomB.root);
    if (hitB.triangleId == InvalidId) return false;

    if (!areNearestTwoModels(p, modelIndexA, modelIndexB, allModels, modelCount)) {
        return false;
    }

    const float3 qA = hitA.closest.point;
    const uint32_t seedA = nearestSeed(qA, seeds);
    const uint32_t cellA = findPowerCellOwner(qA, seedA, baseCells, baseGrid);
    if (cellA == InvalidId) return false;

    const float3 qB = hitB.closest.point;
    const uint32_t seedB = nearestSeed(qB, seeds);
    const uint32_t cellB = findPowerCellOwner(qB, seedB, baseCells, baseGrid);
    if (cellB == InvalidId) return false;

    if (!fragByCellA || !fragByCellB) return false;
    const uint32_t fragA = fragByCellA[cellA];
    const uint32_t fragB = fragByCellB[cellB];
    if (fragA == InvalidId || fragB == InvalidId) return false;

    outTuple = make_uint2(fragA, fragB);
    return true;
}

static __global__ void unpackManifoldMeshTrianglesKernel(
    const float3* meshVertices,
    const uint3* meshTriangles,
    uint32_t triangleCount,
    float3* outTriangles,
    uint2* outTopology)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= triangleCount) return;

    const uint3 t = meshTriangles[idx];
    outTriangles[3 * idx + 0] = meshVertices[t.x];
    outTriangles[3 * idx + 1] = meshVertices[t.y];
    outTriangles[3 * idx + 2] = meshVertices[t.z];
    outTopology[idx] = make_uint2(InvalidId, InvalidId);
}

static __global__ void classifyContactTrianglesKernel(
    const float3* inTriangles,
    uint2* inOutTopology,
    uint32_t triangleCount,
    DeviceModelGeometryView geomA,
    DeviceModelGeometryView geomB,
    const DeviceModelGeometryView* allModels,
    uint32_t modelCount,
    uint32_t modelIndexA,
    uint32_t modelIndexB,
    float contactResolution,
    DeviceKnn seeds,
    DeviceBaseCells baseCells,
    DeviceVoxelGrid baseGrid,
    const uint32_t* fragByCellA,
    const uint32_t* fragByCellB,
    uint32_t* outChildCounts,
    uint32_t* outHasSubdivision)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= triangleCount) return;

    if (inOutTopology[idx].x != InvalidId) {
        outChildCounts[idx] = 1;
        return;
    }

    const float3 v0 = inTriangles[3 * idx + 0];
    const float3 v1 = inTriangles[3 * idx + 1];
    const float3 v2 = inTriangles[3 * idx + 2];

    float3 p[7];
    p[0] = v0;
    p[1] = v1;
    p[2] = v2;
    p[3] = mul3(add3(v0, v1), 0.5f);
    p[4] = mul3(add3(v1, v2), 0.5f);
    p[5] = mul3(add3(v2, v0), 0.5f);
    p[6] = mul3(add3(add3(v0, v1), v2), 1.0f / 3.0f);

    uint2 tuple[7];
    bool valid[7];
    bool allValid = true;

    for (int k = 0; k < 7; ++k) {
        valid[k] = sampleContactTopology(
            p[k], geomA, geomB, allModels, modelCount,
            modelIndexA, modelIndexB,
            seeds, baseCells, baseGrid,
            fragByCellA, fragByCellB,
            tuple[k]);
        if (!valid[k]) allValid = false;
    }

    bool homogeneous = allValid;
    if (homogeneous) {
        const uint2 t0 = tuple[0];
        for (int k = 1; k < 7; ++k) {
            if (tuple[k].x != t0.x || tuple[k].y != t0.y) {
                homogeneous = false;
                break;
            }
        }
    }

    const float diam = triangleDiameter(v0, v1, v2);

    if (homogeneous) {
        inOutTopology[idx] = tuple[0];
        outChildCounts[idx] = 1;
    } else if (diam > contactResolution) {
        outChildCounts[idx] = 4;
        if (outHasSubdivision) {
            atomicExch(outHasSubdivision, 1u);
        }
    } else if (valid[6]) {
        inOutTopology[idx] = tuple[6];
        outChildCounts[idx] = 1;
    } else {
        outChildCounts[idx] = 0;
    }
}

static __global__ void emitContactTrianglesKernel(
    const float3* inTriangles,
    const uint2* inTopology,
    uint32_t inCount,
    const uint32_t* childCounts,
    const uint32_t* childOffsets,
    float3* outTriangles,
    uint2* outTopology)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= inCount) return;

    const uint32_t count = childCounts[idx];
    if (count == 0) return;
    const uint32_t dst = childOffsets[idx];
    const float3 v0 = inTriangles[3 * idx + 0];
    const float3 v1 = inTriangles[3 * idx + 1];
    const float3 v2 = inTriangles[3 * idx + 2];

    if (count == 1) {
        outTriangles[3 * dst + 0] = v0;
        outTriangles[3 * dst + 1] = v1;
        outTriangles[3 * dst + 2] = v2;
        outTopology[dst] = inTopology[idx];
    } else if (count == 4) {
        const float3 m01 = mul3(add3(v0, v1), 0.5f);
        const float3 m12 = mul3(add3(v1, v2), 0.5f);
        const float3 m20 = mul3(add3(v2, v0), 0.5f);

        outTriangles[3 * (dst + 0) + 0] = v0;
        outTriangles[3 * (dst + 0) + 1] = m01;
        outTriangles[3 * (dst + 0) + 2] = m20;
        outTopology[dst + 0] = make_uint2(InvalidId, InvalidId);

        outTriangles[3 * (dst + 1) + 0] = v1;
        outTriangles[3 * (dst + 1) + 1] = m12;
        outTriangles[3 * (dst + 1) + 2] = m01;
        outTopology[dst + 1] = make_uint2(InvalidId, InvalidId);

        outTriangles[3 * (dst + 2) + 0] = v2;
        outTriangles[3 * (dst + 2) + 1] = m20;
        outTriangles[3 * (dst + 2) + 2] = m12;
        outTopology[dst + 2] = make_uint2(InvalidId, InvalidId);

        outTriangles[3 * (dst + 3) + 0] = m01;
        outTriangles[3 * (dst + 3) + 1] = m12;
        outTriangles[3 * (dst + 3) + 2] = m20;
        outTopology[dst + 3] = make_uint2(InvalidId, InvalidId);
    }
}

static __global__ void computeContactFacesKernel(
    const float3* triangles,
    const uint2* topology,
    uint32_t triangleCount,
    DeviceModelGeometryView geomA,
    DeviceModelGeometryView geomB,
    float areaTolerance,
    float* outAreas,
    float4* outNormalGap,
    float2* outPlaneOffsets)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= triangleCount) return;

    outAreas[idx]        = 0.0f;
    outNormalGap[idx]    = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    outPlaneOffsets[idx] = make_float2(0.0f, 0.0f);

    const uint2 topo = topology[idx];
    const uint32_t fragA = topo.x;
    const uint32_t fragB = topo.y;

    if (fragA == InvalidId || fragB == InvalidId) {
        return;
    }

    const float3 p0 = triangles[3 * idx + 0];
    const float3 p1 = triangles[3 * idx + 1];
    const float3 p2 = triangles[3 * idx + 2];

    const float3 cM = mul3(add3(add3(p0, p1), p2), 1.0f / 3.0f);

    const ClosestSurfaceHit hitA = closestPoint(cM, geomA.triangles, geomA.nodes, geomA.parents, geomA.root);
    const ClosestSurfaceHit hitB = closestPoint(cM, geomB.triangles, geomB.nodes, geomB.parents, geomB.root);

    if (hitA.triangleId == InvalidId || hitB.triangleId == InvalidId) {
        return;
    }

    const float3 qA = hitA.closest.point;
    const float3 qB = hitB.closest.point;

    const float3 vectorArea2 = cross3(sub3(p1, p0), sub3(p2, p0));
    const float twiceArea = len3(vectorArea2);
    if (0.5f * twiceArea <= areaTolerance) {
        return;
    }

    const float3 medialNormal = mul3(vectorArea2, 1.0f / twiceArea);
    const float3 sep = sub3(qB, qA);
    const float gap = len3(sep);

    const float3 n = gap > 1e-12f ? mul3(sep, 1.0f / gap) : medialNormal;

    const float area = 0.5f * fabsf(dot3(vectorArea2, n));
    if (area <= areaTolerance) {
        return;
    }

    outAreas[idx]        = area;
    outNormalGap[idx]    = make_float4(n.x, n.y, n.z, gap);
    outPlaneOffsets[idx] = make_float2(dot3(n, qA), dot3(n, qB));
}

bool buildContactFaces(
    const std::vector<ModelRestrictionData>& models,
    const cudaUtils::CudaBuffer<cudaGeometry::DeviceModelGeometryView>& devGeometries,
    const std::vector<cudaGeometry::DeviceModelGeometryView>& hostGeometries,
    const DeviceKnn& seeds,
    const BaseCellsContext& baseCtx,
    float contactRange,
    float contactResolution,
    const std::vector<std::vector<uint32_t>>& fragmentIdByModelCell,
    const std::vector<Fragment>& fragments,
    RVDResult& result,
    cudaStream_t stream)
{
    const uint32_t modelCount = uint32_t(models.size());
    if (hostGeometries.size() != modelCount ||
        devGeometries.size() != modelCount ||
        fragmentIdByModelCell.size() != modelCount)
        return false;

    if (modelCount < 2)
        return true;

    for (uint32_t m = 0; m < modelCount; ++m) {
        if (fragmentIdByModelCell[m].size() != baseCtx.count)
            return false;
    }

    const float3 domainMin = make_float3(baseCtx.domainMin.x, baseCtx.domainMin.y, baseCtx.domainMin.z);
    const float3 domainMax = make_float3(baseCtx.domainMax.x, baseCtx.domainMax.y, baseCtx.domainMax.z);

    std::vector<cudaUtils::CudaBuffer<uint32_t>> devModelFragments(modelCount);
    for (uint32_t m = 0; m < modelCount; ++m) {
        if (!devModelFragments[m].allocate(baseCtx.count) ||
            !devModelFragments[m].uploadAsync(fragmentIdByModelCell[m].data(), baseCtx.count, stream)) {
            return false;
        }
    }

    cudaUtils::CudaBuffer<uint8_t> cubScratch;
    cudaUtils::CudaBuffer<uint32_t> devChildCounts;
    cudaUtils::CudaBuffer<uint32_t> devChildOffsets;
    cudaUtils::CudaBuffer<float3> curTriangles;
    cudaUtils::CudaBuffer<uint2> curTopology;
    cudaUtils::CudaBuffer<float3> nextTriangles;
    cudaUtils::CudaBuffer<uint2> nextTopology;
    cudaUtils::CudaBuffer<uint32_t> devHasSubdivision;
    if (!devHasSubdivision.allocate(1)) return false;

    cudaUtils::CudaBuffer<float> devAreas;
    cudaUtils::CudaBuffer<float4> devNormalGap;
    cudaUtils::CudaBuffer<float2> devPlaneOffsets;

    for (uint32_t ma = 0; ma < modelCount; ++ma) {
        for (uint32_t mb = ma + 1; mb < modelCount; ++mb) {
            if (models[ma].totalPatches == 0 || models[mb].totalPatches == 0) continue;

            ContactManifoldMesh manifoldMesh;
            if (!extractContactManifold(hostGeometries[ma], hostGeometries[mb], domainMin, domainMax, contactRange, contactResolution, manifoldMesh, stream)) {
                std::cerr << "[RVDContact] failed to extract manifold for pair (" << ma << ", " << mb << ")\n";
                return false;
            }
            if (manifoldMesh.triangles.size() == 0) continue;

            const uint32_t initialTriCount = uint32_t(manifoldMesh.triangles.size());
            if (!ensureCapacity(curTriangles, 3 * initialTriCount) || !ensureCapacity(curTopology, initialTriCount)) return false;

            unpackManifoldMeshTrianglesKernel<<<(initialTriCount + BlockSize - 1) / BlockSize, BlockSize, 0, stream>>>(
                manifoldMesh.vertices.get(),
                manifoldMesh.triangles.get(),
                initialTriCount,
                curTriangles.get(),
                curTopology.get());
            if (cudaGetLastError() != cudaSuccess) return false;

            uint32_t curCount = initialTriCount;

            while (curCount > 0) {
                if (!ensureCapacity(devChildCounts, curCount)) return false;

                if (cudaMemsetAsync(devHasSubdivision.get(), 0, sizeof(uint32_t), stream) != cudaSuccess) return false;

                const uint32_t grid = (curCount + BlockSize - 1) / BlockSize;
                classifyContactTrianglesKernel<<<grid, BlockSize, 0, stream>>>(
                    curTriangles.get(),
                    curTopology.get(),
                    curCount,
                    hostGeometries[ma],
                    hostGeometries[mb],
                    devGeometries.get(),
                    modelCount,
                    ma,
                    mb,
                    contactResolution,
                    seeds,
                    baseCtx.deviceView(),
                    baseCtx.baseGrid,
                    devModelFragments[ma].get(),
                    devModelFragments[mb].get(),
                    devChildCounts.get(),
                    devHasSubdivision.get());
                if (cudaGetLastError() != cudaSuccess) return false;

                if (!ensureCapacity(devChildOffsets, curCount) ||
                    !exclusiveSum(devChildCounts.get(), devChildOffsets.get(), int(curCount), cubScratch, stream)) return false;

                uint32_t lastOffset = 0;
                uint32_t lastCount = 0;
                uint32_t hostHasSubdivision = 0;

                if (cudaMemcpyAsync(&lastOffset, devChildOffsets.get() + (curCount - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                    cudaMemcpyAsync(&lastCount, devChildCounts.get() + (curCount - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                    cudaMemcpyAsync(&hostHasSubdivision, devHasSubdivision.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
                    cudaStreamSynchronize(stream) != cudaSuccess) {
                    return false;
                }

                const uint32_t nextCount = lastOffset + lastCount;
                if (nextCount == 0) {
                    curCount = 0;
                    break;
                }

                if (!ensureCapacity(nextTriangles, 3 * nextCount) || !ensureCapacity(nextTopology, nextCount)) return false;

                emitContactTrianglesKernel<<<grid, BlockSize, 0, stream>>>(
                    curTriangles.get(),
                    curTopology.get(),
                    curCount,
                    devChildCounts.get(),
                    devChildOffsets.get(),
                    nextTriangles.get(),
                    nextTopology.get());
                if (cudaGetLastError() != cudaSuccess) return false;

                curTriangles.swap(nextTriangles);
                curTopology.swap(nextTopology);
                curCount = nextCount;

                if (hostHasSubdivision == 0) {
                    break;
                }
            }

            if (curCount == 0) continue;

            if (!ensureCapacity(devAreas, curCount) ||
                !ensureCapacity(devNormalGap, curCount) ||
                !ensureCapacity(devPlaneOffsets, curCount)) return false;

            const uint32_t gridFaces = (curCount + BlockSize - 1) / BlockSize;
            computeContactFacesKernel<<<gridFaces, BlockSize, 0, stream>>>(
                curTriangles.get(),
                curTopology.get(),
                curCount,
                hostGeometries[ma],
                hostGeometries[mb],
                float(baseCtx.areaTolerance),
                devAreas.get(),
                devNormalGap.get(),
                devPlaneOffsets.get());
            if (cudaGetLastError() != cudaSuccess) return false;

            std::vector<float> hostAreas(curCount);
            std::vector<float4> hostNormalGap(curCount);
            std::vector<float2> hostPlaneOffsets(curCount);
            std::vector<uint2> hostTopology(curCount);
            std::vector<float3> hostTriangles(3 * curCount);

            if (!devAreas.downloadAsync(hostAreas.data(), curCount, stream) ||
                !devNormalGap.downloadAsync(hostNormalGap.data(), curCount, stream) ||
                !devPlaneOffsets.downloadAsync(hostPlaneOffsets.data(), curCount, stream) ||
                !curTopology.downloadAsync(hostTopology.data(), curCount, stream) ||
                !curTriangles.downloadAsync(hostTriangles.data(), 3 * curCount, stream) ||
                cudaStreamSynchronize(stream) != cudaSuccess) return false;

            for (uint32_t i = 0; i < curCount; ++i) {
                const float area = hostAreas[i];
                if (area <= baseCtx.areaTolerance) continue;

                const uint32_t fragA = hostTopology[i].x;
                const uint32_t fragB = hostTopology[i].y;

                if (fragA == InvalidId || fragB == InvalidId) continue;
                if (fragA >= fragments.size() || fragB >= fragments.size()) continue;

                const glm::dvec3 normal(hostNormalGap[i].x, hostNormalGap[i].y, hostNormalGap[i].z);
                const double gap = double(hostNormalGap[i].w);

                const double dotSolidA = double(hostPlaneOffsets[i].x);
                const double dotSolidB = double(hostPlaneOffsets[i].y);

                const glm::dvec3 cFragA = glm::dvec3(fragments[fragA].centroid);
                const glm::dvec3 cFragB = glm::dvec3(fragments[fragB].centroid);

                const double distA = std::abs(glm::dot(normal, cFragA) - dotSolidA);
                const double distB = std::abs(glm::dot(normal, cFragB) - dotSolidB);

                result.contactFaces.push_back({
                    fragA,
                    fragB,
                    double(area),
                    distA,
                    distB,
                    gap
                });

                const float3 p0 = hostTriangles[3 * i + 0];
                const float3 p1 = hostTriangles[3 * i + 1];
                const float3 p2 = hostTriangles[3 * i + 2];

                ContactMeshVertex cv0{}, cv1{}, cv2{};
                cv0.position = glm::vec3(p0.x, p0.y, p0.z);
                cv0.normal = glm::vec3(float(normal.x), float(normal.y), float(normal.z));
                cv1.position = glm::vec3(p1.x, p1.y, p1.z);
                cv1.normal = cv0.normal;
                cv2.position = glm::vec3(p2.x, p2.y, p2.z);
                cv2.normal = cv0.normal;

                result.contactMesh.push_back(cv0);
                result.contactMesh.push_back(cv1);
                result.contactMesh.push_back(cv2);
            }
        }
    }
    return true;
}

} // namespace voronoi
