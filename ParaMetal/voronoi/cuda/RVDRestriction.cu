#include "RVDRestriction.cuh"
#include "ConvexCell.cuh"

#include <cub/cub.cuh>
#include <iostream>
#include <algorithm>

namespace voronoi {

using rvdDevice::ConvexCell;
using rvdDevice::ConvexCellStorage;
using rvdDevice::DiscoveryLimits;
using rvdDevice::PredicateMode;
using rvdDevice::sub;
using rvdDevice::cross3;
using rvdDevice::dot3;
using rvdDevice::mul;
using rvdDevice::min3;
using rvdDevice::max3;
using rvdDevice::finite3;
using rvdDevice::domainFaceFromTag;
using rvdDevice::integrateCell;
using rvdDevice::clipTriangleToCell;
using rvdDevice::polygonArea;
using rvdDevice::polygonCentroid;
using rvdDevice::determinant3Double;
using rvdDevice::MaxClippedPolygonVertices;
using DeviceTriangle = cudaGeometry::DeviceTriangle;

struct RestrictedDeviceInput {
    DeviceKnn knn;
    const DeviceTriangle* triangles;
    uint32_t triangleCount;
    const uint8_t* occupancy;
    uint32_t occupancyCount;
    const int32_t* triangleList;
    uint32_t triangleListCount;
    const int32_t* offsets;
    uint32_t offsetCount;
    cudaGeometry::DeviceVoxelGrid baseGrid;
    cudaGeometry::DeviceVoxelGrid modelVoxelGrid;
    DeviceBaseCells baseCells;
};

struct RestrictedDeviceOutput {
    double4* moments = nullptr;
    float* restrictedSurfaceAreas = nullptr;
    uint32_t* planarPatchCounts = nullptr;
    const uint32_t* planarPatchOffsets = nullptr;
    uint32_t* planarVertexCounts = nullptr;
    const uint32_t* planarVertexOffsets = nullptr;
    RestrictedSurface* outSurfaces = nullptr;
    RestrictedSurfaceGeometry* outSurfaceGeometry = nullptr;
    float3* outRestrictedVertices = nullptr;
    const uint32_t* restrictedPlaneOffsets = nullptr;
    double* restrictedPlaneAreas = nullptr;
    RVDStatus* statuses = nullptr;
};

struct RestrictedCellStorage {
    ConvexCellStorage<DiscoveryLimits> cell;
    uint3 backupVertices[DiscoveryLimits::V];
    int32_t uniqueTriangles[DiscoveryLimits::T];
};

static __device__ inline float4 planeFromPoints(
    float3 a,
    float3 b,
    float3 c)
{
    const float3 normal = cross3(sub(b, a), sub(c, a));
    return make_float4(
        normal.x,
        normal.y,
        normal.z,
        -dot3(normal, a));
}

static constexpr auto cudaOk = cudaUtils::cudaCheck;

template <PredicateMode Mode>
static __device__ bool cellBounds(
    const ConvexCell<Mode>& cell,
    float3& minimum, float3& maximum) {
    minimum = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    maximum = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int vertex = 0; vertex < cell.vertexCount && cell.status == RVDStatus::Success; ++vertex) {
        const float3 point = cell.vertexPosition(cell.vertices[vertex]);
        minimum = min3(minimum, point);
        maximum = max3(maximum, point);
    }
    return cell.status == RVDStatus::Success && finite3(minimum) && finite3(maximum);
}

static __device__ void voxelBounds(
    const RestrictedDeviceInput& input,
    float3 minimum,
    float3 maximum,
    int3& voxelMinimum,
    int3& voxelMaximum)
{
    const cudaGeometry::DeviceVoxelGrid& vox = input.modelVoxelGrid;
    const float invVoxelSize = 1.0f / vox.scale;

    const int minX = int(floorf((minimum.x - vox.gridMin.x) * invVoxelSize));
    const int minY = int(floorf((minimum.y - vox.gridMin.y) * invVoxelSize));
    const int minZ = int(floorf((minimum.z - vox.gridMin.z) * invVoxelSize));

    const int maxX = int(floorf((maximum.x - vox.gridMin.x) * invVoxelSize));
    const int maxY = int(floorf((maximum.y - vox.gridMin.y) * invVoxelSize));
    const int maxZ = int(floorf((maximum.z - vox.gridMin.z) * invVoxelSize));

    voxelMinimum = make_int3(
        max(0, min(vox.gridDim.x - 1, minX)),
        max(0, min(vox.gridDim.y - 1, minY)),
        max(0, min(vox.gridDim.z - 1, minZ)));

    voxelMaximum = make_int3(
        max(0, min(vox.gridDim.x - 1, maxX)),
        max(0, min(vox.gridDim.y - 1, maxY)),
        max(0, min(vox.gridDim.z - 1, maxZ)));
}

static __device__ bool findCellOrigin(
    const RestrictedDeviceInput& input,
    int3 voxelMinimum,
    int3 voxelMaximum,
    float3& origin,
    bool& inside)
{
    const cudaGeometry::DeviceVoxelGrid& vox = input.modelVoxelGrid;
    const int cStride = vox.gridDim.x + 1;
    const float voxelSize = vox.scale;
    const int3 cornerMaximum = make_int3(
        min(vox.gridDim.x, voxelMaximum.x + 1),
        min(vox.gridDim.y, voxelMaximum.y + 1),
        min(vox.gridDim.z, voxelMaximum.z + 1));

    bool foundOutside = false;
    float3 outsideOrigin{};

    for (int z = voxelMinimum.z; z <= cornerMaximum.z; ++z) {
        for (int y = voxelMinimum.y; y <= cornerMaximum.y; ++y) {
            for (int x = voxelMinimum.x; x <= cornerMaximum.x; ++x) {
                const int index = z * cStride * cStride + y * cStride + x;
                if (index < 0 || uint32_t(index) >= input.occupancyCount) {
                    continue;
                }

                const uint32_t value = input.occupancy[index];
                const float3 point = make_float3(
                    vox.gridMin.x + float(x) * voxelSize,
                    vox.gridMin.y + float(y) * voxelSize,
                    vox.gridMin.z + float(z) * voxelSize);

                if (value == 2u) {
                    origin = point;
                    inside = true;
                    return true;
                }

                if (value == 0u && !foundOutside) {
                    outsideOrigin = point;
                    foundOutside = true;
                }
            }
        }
    }

    if (foundOutside) {
        origin = outsideOrigin;
        inside = false;
        return true;
    }

    const int faceCoordinates[3] = {
        voxelMaximum.x + 1,
        voxelMaximum.y + 1,
        voxelMaximum.z + 1
    };

    for (int axis = 2; axis >= 0; --axis) {
        const int limit =
            axis == 0 ? vox.gridDim.x :
            axis == 1 ? vox.gridDim.y :
                        vox.gridDim.z;

        if (faceCoordinates[axis] >= limit) {
            int x = voxelMinimum.x;
            int y = voxelMinimum.y;
            int z = voxelMinimum.z;

            if (axis == 0) x = vox.gridDim.x;
            if (axis == 1) y = vox.gridDim.y;
            if (axis == 2) z = vox.gridDim.z;

            origin = make_float3(
                vox.gridMin.x + float(x) * voxelSize,
                vox.gridMin.y + float(y) * voxelSize,
                vox.gridMin.z + float(z) * voxelSize);

            inside = false;
            return true;
        }
    }

    return false;
}

template <PredicateMode Mode>
static __device__ int findBoundaryPlane(
    const ConvexCell<Mode>& cell,
    int basePlaneCount,
    float3 a,
    float3 b,
    float3 c)
{
    for (int p = 0; p < basePlaneCount; ++p) {
        if (domainFaceFromTag(cell.planeNeighborIds[p]) < 0) {
            continue;
        }

        const float4 plane = cell.planes[p];
        const float n =
            sqrtf(plane.x * plane.x +
                  plane.y * plane.y +
                  plane.z * plane.z);

        const float eps = 1e-6f * n;

        const float da =
            plane.x * a.x + plane.y * a.y + plane.z * a.z + plane.w;
        const float db =
            plane.x * b.x + plane.y * b.y + plane.z * b.z + plane.w;
        const float dc =
            plane.x * c.x + plane.y * c.y + plane.z * c.z + plane.w;

        if (fabsf(da) <= eps &&
            fabsf(db) <= eps &&
            fabsf(dc) <= eps) {
            return p;
        }
    }

    return -1;
}

template <PredicateMode Mode>
static __device__ bool writeIntegratedCell(
    uint32_t outputId,
    ConvexCell<Mode>& cell,
    RestrictedDeviceOutput output,
    double4 moment,
    float patchArea,
    float canonicalScale) {
    if (cell.status != RVDStatus::Success) return false;
    const double inverseScale = 1.0 / double(canonicalScale);
    const double inverseAreaScale = inverseScale * inverseScale;
    const double inverseVolumeScale = inverseAreaScale * inverseScale;
    const double worldSignedVolume = moment.w * inverseVolumeScale;

    if (!isfinite(worldSignedVolume) || worldSignedVolume < 0.0) {
        cell.fail(RVDStatus::NonFiniteGeometry);
        return false;
    }

    if (output.moments != nullptr) {
        output.moments[outputId] = moment;
    }

    if (output.restrictedSurfaceAreas != nullptr) {
        const float inverseAreaScaleF = float(inverseAreaScale);
        const float worldPatchArea = patchArea * inverseAreaScaleF;
        output.restrictedSurfaceAreas[outputId] = worldPatchArea;
    }

    output.statuses[outputId] = RVDStatus::Success;
    return true;
}

static __device__ void processRestrictedCell(
    uint32_t seedId,
    const RestrictedDeviceInput& input,
    RestrictedDeviceOutput output,
    RestrictedCellStorage& storage) {

    ConvexCell<PredicateMode::FilteredDouble> cell;
    cell.attachStorage(storage.cell, RVDRetryPlaneCount, RVDRetryVertexCount);
    cell.cellId = seedId;

    if (output.restrictedSurfaceAreas != nullptr) output.restrictedSurfaceAreas[seedId] = 0.0f;
    if (output.planarPatchCounts != nullptr) output.planarPatchCounts[seedId] = 0u;
    if (output.planarVertexCounts != nullptr) output.planarVertexCounts[seedId] = 0u;

    if (seedId >= input.knn.seedCount) {
        output.statuses[seedId] = RVDStatus::InvalidIndex;
        return;
    }

    const int32_t retrySlot = input.baseCells.retrySlots[seedId];
    bool loaded = false;

    if (retrySlot >= 0) {
        if (uint32_t(retrySlot) >= input.baseCells.retryCount) {
            output.statuses[seedId] = RVDStatus::InvalidIndex;
            return;
        }
        loaded = cell.loadStorage(input.baseCells.retryCells[uint32_t(retrySlot)]);
    } else {
        loaded = cell.loadStorage(input.baseCells.cells[seedId]);
    }

    if (!loaded) {
        output.statuses[seedId] = cell.status == RVDStatus::Success ? RVDStatus::InvalidIndex : cell.status;
        return;
    }

    const int basePlaneCount = cell.planeCount;
    const int baseVertexCount = cell.vertexCount;
    for (int p = 0; p < basePlaneCount; ++p) {
        cell.planeAreas[p] = 0.0;
    }

    float3 cellMinimum, cellMaximum;
    if (!cellBounds(cell, cellMinimum, cellMaximum)) {
        output.statuses[seedId] = cell.status == RVDStatus::Success ? RVDStatus::NonFiniteGeometry : cell.status;
        return;
    }

    int3 voxelMinimum, voxelMaximum;
    voxelBounds(input, cellMinimum, cellMaximum, voxelMinimum, voxelMaximum);

    float3 cellOrigin = make_float3(0.0f, 0.0f, 0.0f);
    bool originInside = false;
    if (!findCellOrigin(input, voxelMinimum, voxelMaximum, cellOrigin, originInside)) {
        output.statuses[seedId] = RVDStatus::InconsistentBoundary;
        return;
    }

    const float4 seedPosition = input.knn.seeds[seedId];
    const float3 seedMomentCenter = canonicalFromWorld(
        input.baseGrid, make_float3(seedPosition.x, seedPosition.y, seedPosition.z));

    double4 moment = make_double4(0.0, 0.0, 0.0, 0.0);

    const int dimX = input.modelVoxelGrid.gridDim.x;
    const int dimY = input.modelVoxelGrid.gridDim.y;
    int uniqueCount = 0;

    for (int z = voxelMinimum.z; z <= voxelMaximum.z; ++z) {
        for (int y = voxelMinimum.y; y <= voxelMaximum.y; ++y) {
            for (int x = voxelMinimum.x; x <= voxelMaximum.x; ++x) {
                const int voxelIndex = z * dimX * dimY + y * dimX + x;
                if (voxelIndex < 0 || uint32_t(voxelIndex + 1) >= input.offsetCount) continue;
                const int32_t begin = input.offsets[voxelIndex];
                const int32_t end = input.offsets[voxelIndex + 1];

                for (int32_t listIndex = begin; listIndex < end; ++listIndex) {
                    if (listIndex < 0 || uint32_t(listIndex) >= input.triangleListCount) continue;
                    const int32_t rawTriangle = input.triangleList[listIndex];
                    if (rawTriangle < 0 || uint32_t(rawTriangle) >= input.triangleCount) continue;

                    bool seen = false;
                    for (int u = 0; u < uniqueCount; ++u) {
                        if (storage.uniqueTriangles[u] == rawTriangle) { seen = true; break; }
                    }
                    if (seen) continue;
                    if (uint32_t(uniqueCount) >= DiscoveryLimits::T) {
                        output.statuses[seedId] = RVDStatus::TriangleOverflow;
                        return;
                    }
                    storage.uniqueTriangles[uniqueCount++] = rawTriangle;
                }
            }
        }
    }

    const double4 fullCell = integrateCell(cell, seedMomentCenter, basePlaneCount);
    if (cell.status != RVDStatus::Success) {
        output.statuses[seedId] = cell.status;
        return;
    }

    if (originInside) {
        moment = make_double4(
            fullCell.x / 6.0,
            fullCell.y / 6.0,
            fullCell.z / 6.0,
            fullCell.w / 6.0);
    } else {
        for (int p = 0; p < basePlaneCount; ++p) {
            cell.planeAreas[p] = 0.0;
        }
    }

    for (int i = 0; i < baseVertexCount; ++i) {
        storage.backupVertices[i] = cell.vertices[i];
    }

    float surfacePatchArea = 0.0f;
    uint32_t planarPatchCount = 0u;
    uint32_t planarVertexCount = 0u;

    for (int u = 0; u < uniqueCount; ++u) {
        const int32_t triangleIndex = storage.uniqueTriangles[u];
        const DeviceTriangle tri = input.triangles[triangleIndex];
        float3 v0 = canonicalFromWorld(input.baseGrid, make_float3(tri.v0.x, tri.v0.y, tri.v0.z));
        float3 v1 = canonicalFromWorld(input.baseGrid, make_float3(tri.v1.x, tri.v1.y, tri.v1.z));
        float3 v2 = canonicalFromWorld(input.baseGrid, make_float3(tri.v2.x, tri.v2.y, tri.v2.z));

        if (!finite3(v0) || !finite3(v1) || !finite3(v2)) {
            output.statuses[seedId] = RVDStatus::NonFiniteGeometry;
            return;
        }

        // Surface patch clipping for contact output
        float3 poly[MaxClippedPolygonVertices];
        const int numClipped = clipTriangleToCell(v0, v1, v2, cell, basePlaneCount, poly, MaxClippedPolygonVertices);
        if (numClipped < 0) {
            output.statuses[seedId] = RVDStatus::VertexOverflow;
            return;
        }
        if (numClipped >= 3) {
            const float area = polygonArea(poly, numClipped);
            if (area > 1e-12f) {
                surfacePatchArea += area;

                if (output.planarPatchOffsets != nullptr) {
                    const uint32_t patchDst = output.planarPatchOffsets[seedId] + planarPatchCount;
                    const uint32_t vertDst = output.planarVertexOffsets ? (output.planarVertexOffsets[seedId] + planarVertexCount) : 0u;

                    if (output.outSurfaces != nullptr) {
                        RestrictedSurface surface{};
                        surface.cellId = seedId;
                        surface.firstVertex = vertDst;
                        surface.vertexCount = uint32_t(numClipped);
                        output.outSurfaces[patchDst] = surface;
                    }
                    if (output.outSurfaceGeometry != nullptr) {
                        const float3 centroid = polygonCentroid(poly, numClipped, area);
                        const float3 crossA = cross3(sub(v1, v0), sub(v2, v0));
                        const float lenA = sqrtf(dot3(crossA, crossA));
                        const float3 normal = lenA > 1e-12f ? mul(crossA, 1.0f / lenA) : make_float3(0.0f, 0.0f, 1.0f);
                        const float planeOffset = dot3(normal, v0);

                        RestrictedSurfaceGeometry geom{};
                        geom.centroidArea = make_float4(centroid.x, centroid.y, centroid.z, area);
                        geom.plane = make_float4(normal.x, normal.y, normal.z, planeOffset);
                        output.outSurfaceGeometry[patchDst] = geom;
                    }
                    if (output.outRestrictedVertices != nullptr) {
                        const float invScale = 1.0f / input.baseGrid.scale;
                        for (int v = 0; v < numClipped; ++v) {
                            const float3 pCan = poly[v];
                            output.outRestrictedVertices[vertDst + v] = make_float3(
                                pCan.x * invScale + input.baseGrid.gridMin.x,
                                pCan.y * invScale + input.baseGrid.gridMin.y,
                                pCan.z * invScale + input.baseGrid.gridMin.z);
                        }
                    }
                }
                planarPatchCount += 1u;
                planarVertexCount += uint32_t(numClipped);
            }
        }

        // 4-plane tetrahedral clipping and integration
        const double det = determinant3Double(
            sub(v0, cellOrigin),
            sub(v1, cellOrigin),
            sub(v2, cellOrigin));

        double sign = det > 0.0 ? 1.0 : -1.0;

        if (det < 0.0) {
            const float3 tmp = v1;
            v1 = v2;
            v2 = tmp;
        }

        if (originInside) {
            sign = -sign;
        }

        const int boundaryPlane =
            originInside
                ? findBoundaryPlane(
                      cell,
                      basePlaneCount,
                      v0,
                      v1,
                      v2)
                : -1;

        const double boundaryArea =
            boundaryPlane >= 0
                ? cell.planeAreas[boundaryPlane]
                : 0.0;

        const float4 tetraPlanes[4] = {
            planeFromPoints(cellOrigin, v0, v1),
            planeFromPoints(cellOrigin, v2, v0),
            planeFromPoints(cellOrigin, v1, v2),
            planeFromPoints(v1, v2, v0)
        };

        bool emptyIntersection = false;
        for (int plane = 0; plane < 4; ++plane) {
            if (!cell.addPlane(tetraPlanes[plane])) {
                if (cell.status == RVDStatus::EmptyCell) emptyIntersection = true;
                break;
            }
        }
        if (!emptyIntersection && cell.status != RVDStatus::Success) {
            output.statuses[seedId] = cell.status;
            return;
        }
        if (!emptyIntersection) {
            const double4 contribution = integrateCell(cell, seedMomentCenter, basePlaneCount, sign);
            moment.x += contribution.x / 6.0;
            moment.y += contribution.y / 6.0;
            moment.z += contribution.z / 6.0;
            moment.w += contribution.w / 6.0;

            if (boundaryPlane >= 0) {
                cell.planeAreas[boundaryPlane] = boundaryArea;
            }

            if (cell.status != RVDStatus::Success) {
                output.statuses[seedId] = cell.status;
                return;
            }
        }

        cell.vertexCount = baseVertexCount;
        cell.planeCount = basePlaneCount;
        cell.status = RVDStatus::Success;
        for (int i = 0; i < baseVertexCount; ++i) {
            cell.vertices[i] = storage.backupVertices[i];
        }
    }

    if (output.planarPatchCounts != nullptr) output.planarPatchCounts[seedId] = planarPatchCount;
    if (output.planarVertexCounts != nullptr) output.planarVertexCounts[seedId] = planarVertexCount;

    if (!isfinite(surfacePatchArea)) {
        output.statuses[seedId] = RVDStatus::NonFiniteGeometry;
        return;
    }

    if (output.restrictedPlaneAreas != nullptr && output.restrictedPlaneOffsets != nullptr) {
        const uint32_t dst = output.restrictedPlaneOffsets[seedId];
        const double areaScale =
            1.0 /
            (double(input.baseGrid.scale) *
             double(input.baseGrid.scale));
        for (int p = 0; p < basePlaneCount; ++p) {
            output.restrictedPlaneAreas[dst + p] = cell.planeAreas[p] * areaScale;
        }
    }

    constexpr double RestrictedVolumeEpsilon = 1.0;
    if (moment.w < -RestrictedVolumeEpsilon) {
        output.statuses[seedId] = RVDStatus::NegativeVolume;
        return;
    }
    if (moment.w <= RestrictedVolumeEpsilon) {
        output.statuses[seedId] = RVDStatus::EmptyCell;
        return;
    }

    if (!writeIntegratedCell(seedId, cell, output, moment, surfacePatchArea, input.baseGrid.scale)) {
        output.statuses[seedId] = cell.status == RVDStatus::Success ? RVDStatus::NonFiniteGeometry : cell.status;
    }
}

static __global__ void cellKernel(
    RestrictedDeviceInput input,
    RestrictedDeviceOutput output,
    uint32_t begin,
    uint32_t count,
    RestrictedCellStorage* storageMemory) {
    const uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;

    const uint32_t seedId = begin + local;
    processRestrictedCell(seedId, input, output, storageMemory[local]);
}

static bool launchRestrictionPass(
    const RestrictedDeviceInput& input,
    RestrictedDeviceOutput output,
    uint32_t count,
    uint32_t chunkSize,
    RestrictedCellStorage* storage,
    cudaStream_t stream)
{
    if (count == 0) return true;

    constexpr uint32_t BlockSize = 32;
    const uint32_t chunk = std::max(1u, chunkSize);

    for (uint32_t begin = 0; begin < count; begin += chunk) {
        const uint32_t n = std::min(chunk, count - begin);
        const uint32_t blocks = (n + BlockSize - 1) / BlockSize;

        cellKernel<<<blocks, BlockSize, 0, stream>>>(
            input, output, begin, n, storage);

        if (!cudaOk(cudaGetLastError(), "launch restricted kernel"))
            return false;
    }

    return cudaOk(
        cudaStreamSynchronize(stream),
        "synchronize restriction pass");
}

bool restrictModelGeometry(
    const RVDInput& input,
    const RVDConfig& config,
    const cudaGeometry::DeviceModelGeometry& devGeom,
    const BaseCellsContext& baseCtx,
    const TopologyContext& topoCtx,
    ModelRestrictionData& model,
    RVDResult& result,
    cudaStream_t stream) {

    if (!devGeom.valid()) return false;

    model.runtimeModelId = devGeom.runtimeModelId;
    model.cells.assign(baseCtx.count, ModelCellData{});
    model.faceAreas.assign(topoCtx.baseFaces.size(), 0.0);

    const cudaGeometry::DeviceVoxelGrid& modelGrid = devGeom.voxelGrid;
    const float modelVoxelSize =
        (cudaGeometry::CanonicalDomainSize / float(modelGrid.gridDim.x)) / modelGrid.scale;

    cudaGeometry::DeviceVoxelGrid voxelLookup{};
    voxelLookup.gridMin = make_float3(
        (modelGrid.gridMin.x - baseCtx.domainMin.x) * baseCtx.baseScale,
        (modelGrid.gridMin.y - baseCtx.domainMin.y) * baseCtx.baseScale,
        (modelGrid.gridMin.z - baseCtx.domainMin.z) * baseCtx.baseScale);
    voxelLookup.scale = modelVoxelSize * baseCtx.baseScale;
    voxelLookup.gridDim = modelGrid.gridDim;

    const uint32_t count = baseCtx.count;
    const uint32_t totalPlanes = baseCtx.cellPlaneOffsets.back();

    cudaUtils::CudaBuffer<double4> devMoments;
    cudaUtils::CudaBuffer<float> devRestrictedSurfaceAreas;
    cudaUtils::CudaBuffer<uint32_t> devPlanarPatchCounts;
    cudaUtils::CudaBuffer<uint32_t> devPlanarVertexCounts;
    cudaUtils::CudaBuffer<double> devRestrictedPlaneAreas;
    cudaUtils::CudaBuffer<RVDStatus> devStatuses;

    if (!devMoments.allocate(count) ||
        !devRestrictedSurfaceAreas.allocate(count) ||
        !devPlanarPatchCounts.allocate(count) ||
        !devPlanarVertexCounts.allocate(count) ||
        !devStatuses.allocate(count) ||
        (totalPlanes && !devRestrictedPlaneAreas.allocate(totalPlanes))) return false;

    if (totalPlanes && !cudaOk(cudaMemsetAsync(devRestrictedPlaneAreas.get(), 0, totalPlanes * sizeof(double), stream), "clear restricted plane areas"))
        return false;

    const uint32_t chunkSize = std::max(1u, config.restrictedChunkSize);
    const uint32_t storageCount = std::min(count, chunkSize);

    cudaUtils::CudaBuffer<RestrictedCellStorage> devStorage;
    if (!devStorage.allocate(storageCount))
        return false;

    RestrictedDeviceInput geomInput{};
    geomInput.knn = input.seedSpatialIndex->deviceKnn();
    geomInput.triangles = devGeom.triangles.get();
    geomInput.triangleCount = uint32_t(devGeom.triangles.size());
    geomInput.occupancy = devGeom.occupancy.get();
    geomInput.occupancyCount = uint32_t(devGeom.occupancy.size());
    geomInput.triangleList = devGeom.voxelTriangleIds.get();
    geomInput.triangleListCount = uint32_t(devGeom.voxelTriangleIds.size());
    geomInput.offsets = devGeom.voxelOffsets.get();
    geomInput.offsetCount = uint32_t(devGeom.voxelOffsets.size());
    geomInput.baseGrid = baseCtx.baseGrid;
    geomInput.modelVoxelGrid = voxelLookup;
    geomInput.baseCells = baseCtx.deviceView();

    RestrictedDeviceOutput geomOutput{};
    geomOutput.moments = devMoments.get();
    geomOutput.restrictedSurfaceAreas = devRestrictedSurfaceAreas.get();
    geomOutput.planarPatchCounts = devPlanarPatchCounts.get();
    geomOutput.planarVertexCounts = devPlanarVertexCounts.get();
    geomOutput.restrictedPlaneOffsets = baseCtx.devCellPlaneOffsets.get();
    geomOutput.restrictedPlaneAreas = devRestrictedPlaneAreas.get();
    geomOutput.statuses = devStatuses.get();

    if (!launchRestrictionPass(geomInput, geomOutput, count, chunkSize, devStorage.get(), stream)) {
        std::cerr << "[RVD] restriction count pass failed modelId=" << model.runtimeModelId << std::endl;
        return false;
    }

    cudaUtils::CudaBuffer<uint32_t> devPatchOffsets;
    cudaUtils::CudaBuffer<uint32_t> devVertexOffsets;
    if (!devPatchOffsets.allocate(count) || !devVertexOffsets.allocate(count)) return false;

    size_t patchScanBytes = 0;
    size_t vertexScanBytes = 0;

    if (!cudaOk(cub::DeviceScan::ExclusiveSum(
            nullptr, patchScanBytes,
            devPlanarPatchCounts.get(), devPatchOffsets.get(),
            count, stream), "query patch scan") ||
        !cudaOk(cub::DeviceScan::ExclusiveSum(
            nullptr, vertexScanBytes,
            devPlanarVertexCounts.get(), devVertexOffsets.get(),
            count, stream), "query vertex scan"))
        return false;

    cudaUtils::CudaBuffer<uint8_t> scanStorage;
    if (!scanStorage.allocate(std::max(patchScanBytes, vertexScanBytes)))
        return false;

    if (!cudaOk(cub::DeviceScan::ExclusiveSum(
            scanStorage.get(), patchScanBytes,
            devPlanarPatchCounts.get(), devPatchOffsets.get(),
            count, stream), "execute patch scan") ||
        !cudaOk(cub::DeviceScan::ExclusiveSum(
            scanStorage.get(), vertexScanBytes,
            devPlanarVertexCounts.get(), devVertexOffsets.get(),
            count, stream), "execute vertex scan"))
        return false;

    uint32_t lastPatchOffset = 0, lastPatchCount = 0;
    uint32_t lastVertexOffset = 0, lastVertexCount = 0;
    if (count > 0) {
        if (!cudaOk(cudaMemcpyAsync(&lastPatchOffset, devPatchOffsets.get() + (count - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), "download last patch offset") ||
            !cudaOk(cudaMemcpyAsync(&lastPatchCount, devPlanarPatchCounts.get() + (count - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), "download last patch count") ||
            !cudaOk(cudaMemcpyAsync(&lastVertexOffset, devVertexOffsets.get() + (count - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), "download last vertex offset") ||
            !cudaOk(cudaMemcpyAsync(&lastVertexCount, devPlanarVertexCounts.get() + (count - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), "download last vertex count") ||
            !cudaOk(cudaStreamSynchronize(stream), "sync counts")) return false;
    }
    const uint32_t totalPatches = lastPatchOffset + lastPatchCount;
    const uint32_t totalVertices = lastVertexOffset + lastVertexCount;

    cudaUtils::CudaBuffer<RestrictedSurface> devSurfaces;
    cudaUtils::CudaBuffer<RestrictedSurfaceGeometry> devSurfaceGeometry;
    cudaUtils::CudaBuffer<float3> devRestrictedVertices;

    if (totalPatches > 0) {
        if (!devSurfaces.allocate(totalPatches) ||
            !devSurfaceGeometry.allocate(totalPatches)) return false;
    }
    if (totalVertices > 0) {
        if (!devRestrictedVertices.allocate(totalVertices)) return false;
    }

    geomOutput.planarPatchOffsets = devPatchOffsets.get();
    geomOutput.planarVertexOffsets = devVertexOffsets.get();
    geomOutput.outSurfaces = devSurfaces.get();
    geomOutput.outSurfaceGeometry = devSurfaceGeometry.get();
    geomOutput.outRestrictedVertices = devRestrictedVertices.get();

    if (!launchRestrictionPass(geomInput, geomOutput, count, chunkSize, devStorage.get(), stream)) {
        std::cerr << "[RVD] restriction write pass failed modelId=" << model.runtimeModelId << std::endl;
        return false;
    }

    std::vector<RVDStatus> statuses(count);
    std::vector<double4> moments(count);
    std::vector<float> surfaceAreas(count);
    std::vector<double> planeAreas(totalPlanes, 0.0);
    std::vector<RestrictedSurface> surfaces(totalPatches);
    std::vector<RestrictedSurfaceGeometry> surfaceGeometry(totalPatches);

    if (!devStatuses.downloadAsync(statuses.data(), count, stream) ||
        !devMoments.downloadAsync(moments.data(), count, stream) ||
        !devRestrictedSurfaceAreas.downloadAsync(surfaceAreas.data(), count, stream) ||
        (totalPlanes && !devRestrictedPlaneAreas.downloadAsync(planeAreas.data(), totalPlanes, stream)) ||
        (totalPatches && !devSurfaces.downloadAsync(surfaces.data(), totalPatches, stream)) ||
        (totalPatches && !devSurfaceGeometry.downloadAsync(surfaceGeometry.data(), totalPatches, stream)) ||
        !cudaOk(cudaStreamSynchronize(stream), "download restricted results")) return false;

    model.surfaceBase = uint32_t(result.surfaces.size());

    result.surfaces.reserve(
        result.surfaces.size() + surfaces.size());

    for (const RestrictedSurface& surface : surfaces) {
        result.surfaces.push_back(surface);
    }

    for (uint32_t cell = 0; cell < count; ++cell) {
        const RVDStatus status = statuses[cell];
        if (status == RVDStatus::EmptyCell) continue;
        if (status != RVDStatus::Success) {
            std::cerr << "[RVD] restriction failed cell=" << cell
                      << " modelId=" << model.runtimeModelId
                      << " status=" << rvdStatusName(status) << std::endl;
            return false;
        }

        const double4 rawMoment = moments[cell];
        const double volume = rawMoment.w * baseCtx.invBaseVolumeScale;
        if (!(volume > 0.0) || !std::isfinite(volume) ||
            !std::isfinite(rawMoment.x) || !std::isfinite(rawMoment.y) || !std::isfinite(rawMoment.z))
            return false;

        model.cells[cell] = {
            volume,
            glm::dvec3(rawMoment.x, rawMoment.y, rawMoment.z) * baseCtx.invBaseMomentScale,
            surfaceAreas[cell]
        };

        const uint32_t planeOffset = baseCtx.cellPlaneOffsets[cell];
        const uint32_t planeCount = baseCtx.planeCount(cell);
        for (uint32_t p = 0; p < planeCount; ++p) {
            const uint32_t faceId = topoCtx.voronoiFaceIdByPlane[planeOffset + p];
            if (faceId != InvalidId) {
                const auto& bf = topoCtx.baseFaces[faceId];
                const uint32_t other = (bf.cellA == cell) ? bf.cellB : bf.cellA;
                if (other != InvalidId) {
                    if (statuses[other] != RVDStatus::Success)
                        continue;

                    const double otherVol = moments[other].w * baseCtx.invBaseVolumeScale;
                    if (!(otherVol > 0.0))
                        continue;
                }
                if (bf.cellA == cell || bf.slotA == InvalidId) {
                    model.faceAreas[faceId] = std::clamp(
                        double(planeAreas[planeOffset + p]), 0.0, bf.area);
                }
            }
        }
    }

    model.devSurfaces.swap(devSurfaces);
    model.surfaceGeometry.swap(surfaceGeometry);
    model.devRestrictedVertices.swap(devRestrictedVertices);
    model.totalPatches = totalPatches;
    return true;
}

} // namespace voronoi
