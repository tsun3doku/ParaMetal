#include "RVDRestriction.cuh"
#include "ConvexCell.cuh"
#include "TriangleWinding.cuh"
#include "../../cuda/CudaMath.cuh"

#include <cub/cub.cuh>
#include <cfloat>
#include <iostream>
#include <algorithm>
#include <limits>

namespace voronoi {

constexpr int MaxTrianglePlaneSegments = 64;

using rvdDevice::ConvexCell;

using rvdDevice::DiscoveryLimits;
using cudaMath::add3;
using cudaMath::sub3;
using cudaMath::cross3;
using cudaMath::dot3;
using cudaMath::mul3;
using rvdDevice::domainFaceFromTag;
using rvdDevice::integrateCell;
using rvdDevice::clipTriangleToCellSegments;
using rvdDevice::PlaneClippedSegment;
using rvdDevice::integratePolygon;
using rvdDevice::determinant3Double;
using rvdDevice::MaxClippedPolygonVertices;
using rvdDevice::NoPlaneNeighbor;
using cudaMath::min3;
using cudaMath::max3;
using DeviceTriangle = cudaGeometry::DeviceTriangle;

struct RestrictedDeviceInput {
    DeviceKnn knn;
    const DeviceTriangle* triangles;
    uint32_t triangleCount;
    const int32_t* triangleList;
    uint32_t triangleListCount;
    const int32_t* offsets;
    uint32_t offsetCount;
    cudaGeometry::DeviceVoxelGrid baseGrid;
    cudaGeometry::DeviceVoxelGrid modelVoxelGrid;
    cudaGeometry::DeviceModelGeometryView winding;
    double3 domainMin;
    double invBaseScale;
    float3 modelBoundsMin;
    float3 modelBoundsMax;
    DeviceBaseCells baseCells;
};

struct RestrictedDeviceOutput {
    double4* moments = nullptr;
    uint32_t* planarPatchCounts = nullptr;
    const uint32_t* planarPatchOffsets = nullptr;
    uint32_t* planarVertexCounts = nullptr;
    const uint32_t* planarVertexOffsets = nullptr;
    RestrictedSurface* outSurfaces = nullptr;
    RestrictedSurfaceGeometry* outSurfaceGeometry = nullptr;
    double3* outRestrictedVertices = nullptr;
    const uint32_t* restrictedPlaneOffsets = nullptr;
    double* restrictedPlaneAreas = nullptr;
    double* restrictedPlaneAbsSums = nullptr;
    double3* restrictedPlaneMoments = nullptr;
    uint32_t* faceSegmentCounts = nullptr;
    const uint32_t* faceSegmentOffsets = nullptr;
    FaceBoundarySegment* outFaceSegments = nullptr;
    RVDStatus* statuses = nullptr;
};

struct RestrictedCellStorage {
    rvdDevice::ConvexCellOverflowStorage cell;
    uint3 backupVertices[RVDOverflowVertexCount];
    int32_t uniqueTriangles[DiscoveryLimits::T];
};

static __device__ inline bool finite3(float3 a) {
    return isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
}
static __device__ inline bool finite3(double3 a) {
    return isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
}

static __device__ inline double4 planeFromPoints(
    double3 a,
    double3 b,
    double3 c)
{
    const double3 normal = cross3(sub3(b, a), sub3(c, a));
    return make_double4(
        normal.x,
        normal.y,
        normal.z,
        -dot3(normal, a));
}

static constexpr auto cudaOk = cudaUtils::cudaCheck;

static __device__ bool cellBounds(
    const ConvexCell& cell,
    float3& minimum, float3& maximum) {
    minimum = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    maximum = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int vertex = 0; vertex < cell.vertexCount && cell.status == RVDStatus::Success; ++vertex) {
        const auto point = cell.vertexPosition(cell.vertices[vertex]);
        const float3 pF = make_float3(float(point.x), float(point.y), float(point.z));
        minimum = min3(minimum, pF);
        maximum = max3(maximum, pF);
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

static __device__ bool writeIntegratedCell(
    uint32_t outputId,
    ConvexCell& cell,
    RestrictedDeviceOutput output,
    double4 moment,
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

    output.statuses[outputId] = RVDStatus::Success;
    return true;
}

static __device__ void processRestrictedCell(
    uint32_t seedId,
    const RestrictedDeviceInput& input,
    RestrictedDeviceOutput output,
    bool surfaceOnly,
    RestrictedCellStorage& storage) {

    ConvexCell cell;
    cell.attachStorage(storage.cell, RVDOverflowPlaneCount, RVDOverflowVertexCount);

    if (!surfaceOnly) {
        if (output.planarPatchCounts != nullptr) output.planarPatchCounts[seedId] = 0u;
        if (output.planarVertexCounts != nullptr) output.planarVertexCounts[seedId] = 0u;
    } else if (output.planarPatchCounts == nullptr ||
               output.planarPatchCounts[seedId] == 0u ||
               output.statuses == nullptr ||
               output.statuses[seedId] != RVDStatus::Success) {
        return;
    }

    if (seedId >= input.knn.seedCount) {
        output.statuses[seedId] = RVDStatus::InvalidIndex;
        return;
    }

    const bool loaded = cell.loadStorage(resolveBaseCell(input.baseCells, seedId));

    if (!loaded) {
        output.statuses[seedId] = cell.status == RVDStatus::Success ? RVDStatus::InvalidIndex : cell.status;
        return;
    }

    const int basePlaneCount = cell.planeCount;
    const int baseVertexCount = cell.vertexCount;
    uint32_t planeSegmentCursor[RVDOverflowPlaneCount];
    for (int p = 0; p < basePlaneCount && p < int(RVDOverflowPlaneCount); ++p) {
        planeSegmentCursor[p] = 0u;
    }
    if (!surfaceOnly) {
        cell.clearPlaneMeasures(basePlaneCount);

        if (output.restrictedPlaneOffsets != nullptr) {
            const uint32_t dst =
                output.restrictedPlaneOffsets[seedId];

            for (int p = 0; p < basePlaneCount; ++p) {
                if (output.restrictedPlaneAreas != nullptr) {
                    output.restrictedPlaneAreas[dst + p] = 0.0;
                }

                if (output.restrictedPlaneMoments != nullptr) {
                    output.restrictedPlaneMoments[dst + p] =
                        make_double3(0.0, 0.0, 0.0);
                }
            }
        }
    }

    float3 cellMinimum, cellMaximum;
    if (!cellBounds(cell, cellMinimum, cellMaximum)) {
        output.statuses[seedId] = cell.status == RVDStatus::Success ? RVDStatus::NonFiniteGeometry : cell.status;
        return;
    }

    const bool outsideModel =
        cellMaximum.x < input.modelBoundsMin.x || cellMinimum.x > input.modelBoundsMax.x ||
        cellMaximum.y < input.modelBoundsMin.y || cellMinimum.y > input.modelBoundsMax.y ||
        cellMaximum.z < input.modelBoundsMin.z || cellMinimum.z > input.modelBoundsMax.z;
    if (outsideModel) {
        if (!surfaceOnly) {
            output.moments[seedId] = make_double4(0.0, 0.0, 0.0, 0.0);
            output.statuses[seedId] = RVDStatus::EmptyCell;
        }
        return;
    }

    int3 voxelMinimum, voxelMaximum;
    voxelBounds(input, cellMinimum, cellMaximum, voxelMinimum, voxelMaximum);

    const cudaGeometry::DeviceVoxelGrid& voxelGrid = input.modelVoxelGrid;
    const float voxelSize = voxelGrid.scale;
    const float3 cellOrigin = make_float3(
        voxelGrid.gridMin.x + float(voxelMinimum.x) * voxelSize,
        voxelGrid.gridMin.y + float(voxelMinimum.y) * voxelSize,
        voxelGrid.gridMin.z + float(voxelMinimum.z) * voxelSize);
    bool originInside = false;
    if (!surfaceOnly) {
        const double3 originWorld = make_double3(
            input.domainMin.x + double(cellOrigin.x) * input.invBaseScale,
            input.domainMin.y + double(cellOrigin.y) * input.invBaseScale,
            input.domainMin.z + double(cellOrigin.z) * input.invBaseScale);
        const float3 originWorldF = make_float3(
            float(originWorld.x), float(originWorld.y), float(originWorld.z));
        const double originWindingValue = cudaGeometry::computeWinding(
            input.winding.triangles,
            input.winding.nodes,
            input.winding.parents,
            input.winding.root,
            input.winding.moments,
            originWorld,
            originWorldF);
        originInside =
            isfinite(originWindingValue) && fabs(originWindingValue) > 0.5;
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

    double4 fullCell = make_double4(0.0, 0.0, 0.0, 0.0);
    double absoluteVolumeSum = 0.0;
    if (!surfaceOnly) {
        fullCell = integrateCell(cell, seedMomentCenter, basePlaneCount);
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
            absoluteVolumeSum = fabs(fullCell.w / 6.0);
        } else {
            cell.clearPlaneMeasures(basePlaneCount);
        }

        for (int i = 0; i < baseVertexCount; ++i) {
            storage.backupVertices[i] = cell.vertices[i];
        }
    }

    double surfacePatchArea = 0.0;
    uint32_t planarPatchCount = 0u;
    uint32_t planarVertexCount = 0u;

    const double scale = double(input.baseGrid.scale);
    const float3 origin = input.baseGrid.gridMin;

    for (int u = 0; u < uniqueCount; ++u) {
        const int32_t triangleIndex = storage.uniqueTriangles[u];
        const DeviceTriangle tri = input.triangles[triangleIndex];

        const double3 v0D = make_double3(
            (double(tri.v0.x) - double(origin.x)) * scale,
            (double(tri.v0.y) - double(origin.y)) * scale,
            (double(tri.v0.z) - double(origin.z)) * scale);

        double3 v1D = make_double3(
            (double(tri.v1.x) - double(origin.x)) * scale,
            (double(tri.v1.y) - double(origin.y)) * scale,
            (double(tri.v1.z) - double(origin.z)) * scale);

        double3 v2D = make_double3(
            (double(tri.v2.x) - double(origin.x)) * scale,
            (double(tri.v2.y) - double(origin.y)) * scale,
            (double(tri.v2.z) - double(origin.z)) * scale);

        if (!finite3(v0D) || !finite3(v1D) || !finite3(v2D)) {
            output.statuses[seedId] = RVDStatus::NonFiniteGeometry;
            return;
        }

        // Surface patch clipping for contact output
        double3 poly[MaxClippedPolygonVertices];
        PlaneClippedSegment planeSegments[MaxTrianglePlaneSegments];
        int planeSegmentCount = 0;
        const int numClipped = clipTriangleToCellSegments(
            v0D, v1D, v2D, cell, basePlaneCount, poly, MaxClippedPolygonVertices,
            planeSegments, &planeSegmentCount, MaxTrianglePlaneSegments);
        if (numClipped < 0) {
            output.statuses[seedId] = planeSegmentCount >= MaxTrianglePlaneSegments
                ? RVDStatus::SegmentOverflow : RVDStatus::VertexOverflow;
            return;
        }

        if (numClipped >= 3) {
            double area = 0.0;
            double3 moment = make_double3(0.0, 0.0, 0.0);
            if (!integratePolygon(poly, numClipped, area, moment)) {
                output.statuses[seedId] = RVDStatus::NonFiniteGeometry;
                return;
            }
            if (area > 0.0 && isfinite(area)) {
                surfacePatchArea += area;

                if (!surfaceOnly) {
                    if (output.faceSegmentCounts != nullptr && output.restrictedPlaneOffsets != nullptr) {
                        const uint32_t planeBase = output.restrictedPlaneOffsets[seedId];
                        for (int s = 0; s < planeSegmentCount; ++s) {
                            const int plane = planeSegments[s].planeIndex;
                            if (plane < 0 || plane >= basePlaneCount) continue;
                            ++output.faceSegmentCounts[planeBase + uint32_t(plane)];
                        }
                    }
                } else if (output.outFaceSegments != nullptr &&
                           output.faceSegmentOffsets != nullptr &&
                           output.restrictedPlaneOffsets != nullptr) {
                    const uint32_t planeBase = output.restrictedPlaneOffsets[seedId];
                    const double invBaseScale = 1.0 / double(input.baseGrid.scale);
                    for (int s = 0; s < planeSegmentCount; ++s) {
                        const int plane = planeSegments[s].planeIndex;
                        if (plane < 0 || plane >= basePlaneCount) continue;
                        const uint32_t slot = planeBase + uint32_t(plane);
                        const uint32_t dst = output.faceSegmentOffsets[slot] + planeSegmentCursor[plane]++;
                        const double3& a = planeSegments[s].a;
                        const double3& b = planeSegments[s].b;
                        FaceBoundarySegment segment{};
                        segment.a = make_double3(
                            input.domainMin.x + a.x * invBaseScale,
                            input.domainMin.y + a.y * invBaseScale,
                            input.domainMin.z + a.z * invBaseScale);
                        segment.b = make_double3(
                            input.domainMin.x + b.x * invBaseScale,
                            input.domainMin.y + b.y * invBaseScale,
                            input.domainMin.z + b.z * invBaseScale);
                        segment.modelId = input.winding.runtimeModelId;
                        output.outFaceSegments[dst] = segment;
                    }
                }


                if (output.planarPatchOffsets != nullptr) {
                    const uint32_t patchDst = output.planarPatchOffsets[seedId] + planarPatchCount;
                    const uint32_t vertDst = output.planarVertexOffsets ? (output.planarVertexOffsets[seedId] + planarVertexCount) : 0u;

                    if (output.outSurfaces != nullptr) {
                        RestrictedSurface surface{};
                        surface.cellId = seedId;
                        surface.firstVertex = vertDst;
                        surface.vertexCount = uint32_t(numClipped);
                        surface.triangleId = uint32_t(triangleIndex);
                        output.outSurfaces[patchDst] = surface;
                    }
                    if (output.outSurfaceGeometry != nullptr) {
                        const double3 centroid = mul3(moment, 1.0 / area);
                        const double3 crossA = cross3(sub3(v1D, v0D), sub3(v2D, v0D));
                        const double lenA = sqrt(dot3(crossA, crossA));
                        if (!(lenA > 0.0) || !isfinite(lenA)) {
                            output.statuses[seedId] = RVDStatus::NonFiniteGeometry;
                            return;
                        }
                        const double3 normal = mul3(crossA, 1.0 / lenA);

                        RestrictedSurfaceGeometry geom{};
                        geom.centroidArea = make_double4(centroid.x, centroid.y, centroid.z, area);
                        geom.plane = make_double4(normal.x, normal.y, normal.z, dot3(normal, v0D));
                        output.outSurfaceGeometry[patchDst] = geom;
                    }
                    if (output.outRestrictedVertices != nullptr) {
                        for (int v = 0; v < numClipped; ++v) {
                            output.outRestrictedVertices[vertDst + v] = poly[v];
                        }
                    }
                }
                planarPatchCount += 1u;
                planarVertexCount += uint32_t(numClipped);
            }
        }

        if (surfaceOnly) {
            continue;
        }

        // 4-plane tetrahedral clipping and integration in double precision
        const double3 originD = make_double3(cellOrigin.x, cellOrigin.y, cellOrigin.z);

        const double3 d0 = sub3(v0D, originD);
        const double3 d1 = sub3(v1D, originD);
        const double3 d2 = sub3(v2D, originD);
        const double det = determinant3Double(d0, d1, d2);

        double sign = det > 0.0 ? 1.0 : -1.0;

        if (det < 0.0) {
            const double3 tmp = v1D;
            v1D = v2D;
            v2D = tmp;
        }

        const double4 tetraPlanes[4] = {
            planeFromPoints(originD, v0D, v1D),
            planeFromPoints(originD, v2D, v0D),
            planeFromPoints(originD, v1D, v2D),
            planeFromPoints(v1D, v2D, v0D)
        };

        constexpr uint64_t RestrictionTagBase = 0x8000000000000000ull;

        bool emptyIntersection = false;
        for (int plane = 0; plane < 4; ++plane) {
            const uint64_t tag =
                RestrictionTagBase |
                (uint64_t(uint32_t(triangleIndex)) << 2) |
                uint64_t(plane);

            if (!cell.addPlane(
                    tetraPlanes[plane],
                    NoPlaneNeighbor,
                    tag)) {
                if (cell.status == RVDStatus::EmptyCell) emptyIntersection = true;
                break;
            }
        }
        if (!emptyIntersection && cell.status != RVDStatus::Success) {
            output.statuses[seedId] = cell.status;
            return;
        }
        if (!emptyIntersection) {
            const double4 contribution = integrateCell(cell, seedMomentCenter, basePlaneCount, -sign);
            const double deltaVolume = contribution.w / 6.0;
            moment.x += contribution.x / 6.0;
            moment.y += contribution.y / 6.0;
            moment.z += contribution.z / 6.0;
            moment.w += deltaVolume;
            absoluteVolumeSum += fabs(deltaVolume);

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

    if (surfaceOnly) {
        return;
    }

    if (!isfinite(surfacePatchArea)) {
        output.statuses[seedId] =
            RVDStatus::NonFiniteGeometry;
        return;
    }

    const double fullCellVolume = fabs(fullCell.w / 6.0);
    const double volumeErrorBudget = fmax(
        256.0 * DBL_EPSILON * absoluteVolumeSum,
        1e-12 * fullCellVolume);

    if (!isfinite(moment.w)) {
        output.statuses[seedId] =
            RVDStatus::NonFiniteGeometry;
        return;
    }

    if (moment.w < -volumeErrorBudget) {
        output.statuses[seedId] =
            RVDStatus::NegativeVolume;
        return;
    }

    // Restricted volume within error budget is treated as empty cell
    if (moment.w <= volumeErrorBudget) {
        output.statuses[seedId] =
            RVDStatus::EmptyCell;
        return;
    }

    if (output.planarPatchCounts != nullptr) {
        output.planarPatchCounts[seedId] = planarPatchCount;
    }

    if (output.planarVertexCounts != nullptr) {
        output.planarVertexCounts[seedId] = planarVertexCount;
    }

    // Only positive volume model fragment owns restricted plane geometry
    if (output.restrictedPlaneAreas != nullptr &&
        output.restrictedPlaneOffsets != nullptr) {

        const uint32_t dst =
            output.restrictedPlaneOffsets[seedId];

        const double invBaseScale =
            1.0 / double(input.baseGrid.scale);

        const double areaScale =
            invBaseScale * invBaseScale;

        const double momentScale =
            areaScale * invBaseScale;

        for (int p = 0; p < basePlaneCount; ++p) {
            const double aWorld =
                cell.planeAreas[p] * areaScale;

            output.restrictedPlaneAreas[dst + p] =
                aWorld;

            if (output.restrictedPlaneAbsSums != nullptr) {
                output.restrictedPlaneAbsSums[dst + p] =
                    cell.planeAreaAbsSums[p] * areaScale;
            }

            if (output.restrictedPlaneMoments != nullptr) {
                if (cell.planeAreas[p] > 0.0 && cell.planeMoments != nullptr) {
                    const double mx =
                        double(input.baseGrid.gridMin.x) * aWorld +
                        cell.planeMoments[p].x * momentScale;

                    const double my =
                        double(input.baseGrid.gridMin.y) * aWorld +
                        cell.planeMoments[p].y * momentScale;

                    const double mz =
                        double(input.baseGrid.gridMin.z) * aWorld +
                        cell.planeMoments[p].z * momentScale;

                    if (!isfinite(mx) || !isfinite(my) || !isfinite(mz)) {
                        output.statuses[seedId] = RVDStatus::NonFiniteGeometry;
                        return;
                    }

                    output.restrictedPlaneMoments[dst + p] =
                        make_double3(mx, my, mz);
                } else {
                    output.restrictedPlaneMoments[dst + p] =
                        make_double3(0.0, 0.0, 0.0);
                }
            }
        }
    }

    if (!writeIntegratedCell(
            seedId,
            cell,
            output,
            moment,
            input.baseGrid.scale)) {

        output.statuses[seedId] =
            cell.status == RVDStatus::Success
                ? RVDStatus::NonFiniteGeometry
                : cell.status;
    }
}

static __global__ void cellKernel(
    RestrictedDeviceInput input,
    RestrictedDeviceOutput output,
    uint32_t begin,
    uint32_t count,
    RestrictedCellStorage* storageMemory,
    bool surfaceOnly) {
    const uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;

    const uint32_t seedId = begin + local;
    processRestrictedCell(seedId, input, output, surfaceOnly, storageMemory[local]);
}

static bool launchRestrictionPass(
    const RestrictedDeviceInput& input,
    RestrictedDeviceOutput output,
    uint32_t count,
    uint32_t chunkSize,
    RestrictedCellStorage* storage,
    cudaStream_t stream,
    bool surfaceOnly)
{
    if (count == 0) return true;

    constexpr uint32_t BlockSize = 32;
    const uint32_t chunk = std::max(1u, chunkSize);

    for (uint32_t begin = 0; begin < count; begin += chunk) {
        const uint32_t n = std::min(chunk, count - begin);
        const uint32_t blocks = (n + BlockSize - 1) / BlockSize;

        cellKernel<<<blocks, BlockSize, 0, stream>>>(
            input, output, begin, n, storage, surfaceOnly);

        if (!cudaOk(cudaGetLastError(), "launch restricted kernel"))
            return false;
    }

    return cudaOk(
        cudaStreamSynchronize(stream),
        "synchronize restriction pass");
}

static __global__ void normalizeRestrictedPlanesKernel(
    DeviceBaseCells baseCells,
    const uint32_t* cellPlaneOffsets,
    const RVDStatus* statuses,
    double4* devMoments,
    double* devRestrictedPlaneAreas,
    double3* devRestrictedPlaneMoments,
    const double* devRestrictedPlaneAbsSums,
    uint32_t count,
    uint32_t runtimeModelId,
    float nominalCellSize,
    double invBaseScale,
    double invBaseAreaScale,
    float3 domainMin,
    unsigned long long* devPackedError)
{
    const uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= count) return;

    const RVDStatus status = statuses[cell];
    if (status == RVDStatus::EmptyCell) {
        devMoments[cell] = make_double4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    if (status != RVDStatus::Success) {
        const unsigned long long packed = (static_cast<unsigned long long>(uint32_t(status)) << 32) | static_cast<unsigned long long>(cell);
        atomicCAS(devPackedError, 0ULL, packed);
        devMoments[cell] = make_double4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    const double4 rawMoment = devMoments[cell];
    if (!(rawMoment.w > 0.0) || !isfinite(rawMoment.w) ||
        !isfinite(rawMoment.x) || !isfinite(rawMoment.y) || !isfinite(rawMoment.z)) {
        const unsigned long long packed = (static_cast<unsigned long long>(101u) << 32) | static_cast<unsigned long long>(cell);
        atomicCAS(devPackedError, 0ULL, packed);
        devMoments[cell] = make_double4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    const BaseCellArrays cellState = resolveBaseCell(baseCells, cell);
    const uint32_t planeOffset = cellPlaneOffsets[cell];
    const uint32_t planeCount = cellState.planeCount;
    const double facetFloor = 64.0 * DBL_EPSILON * double(nominalCellSize * nominalCellSize);
    const double momentScale = invBaseAreaScale * invBaseScale;

    for (uint32_t p = 0; p < planeCount; ++p) {
        const uint32_t slot = planeOffset + p;
        const double canBaseArea = double(cellState.planeAreas[p]);
        const double baseArea = canBaseArea * invBaseAreaScale;
        const double areaTolerance = fmax(facetFloor, fabs(baseArea) * 1e-6);
        const double rawArea = devRestrictedPlaneAreas[slot];

        if (baseArea <= areaTolerance) {
            devRestrictedPlaneAreas[slot] = 0.0;
            devRestrictedPlaneMoments[slot] = make_double3(0.0, 0.0, 0.0);
            continue;
        }

        if (rawArea < -areaTolerance) {
            const unsigned long long packed = (static_cast<unsigned long long>(102u) << 32) | static_cast<unsigned long long>(cell);
            atomicCAS(devPackedError, 0ULL, packed);
            return;
        }

        if (!isfinite(rawArea) || rawArea > baseArea + areaTolerance) {
            const unsigned long long packed = (static_cast<unsigned long long>(102u) << 32) | static_cast<unsigned long long>(cell);
            atomicCAS(devPackedError, 0ULL, packed);
            return;
        }

        if (baseArea - rawArea <= areaTolerance) {
            devRestrictedPlaneAreas[slot] = baseArea;
            const double3 canMom = cellState.planeMoments[p];
            const double mx = double(domainMin.x) * baseArea + canMom.x * momentScale;
            const double my = double(domainMin.y) * baseArea + canMom.y * momentScale;
            const double mz = double(domainMin.z) * baseArea + canMom.z * momentScale;
            devRestrictedPlaneMoments[slot] = make_double3(mx, my, mz);
            continue;
        }
    }
}

bool restrictModelGeometry(
    const RVDInput& input,
    const RVDConfig& config,
    const cudaGeometry::DeviceModelGeometry& devGeom,
    const BaseCellDomain& baseDomain,
    ModelRestriction& restriction,
    cudaStream_t stream) {

    if (!devGeom.valid()) return false;

    restriction.runtimeModelId = devGeom.runtimeModelId;

    const cudaGeometry::DeviceVoxelGrid& modelGrid = devGeom.voxelGrid;
    const float modelVoxelSize =
        (cudaGeometry::CanonicalDomainSize / float(modelGrid.gridDim.x)) / modelGrid.scale;

    cudaGeometry::DeviceVoxelGrid voxelLookup{};
    voxelLookup.gridMin = make_float3(
        (modelGrid.gridMin.x - baseDomain.domainMin.x) * baseDomain.baseScale,
        (modelGrid.gridMin.y - baseDomain.domainMin.y) * baseDomain.baseScale,
        (modelGrid.gridMin.z - baseDomain.domainMin.z) * baseDomain.baseScale);
    voxelLookup.scale = modelVoxelSize * baseDomain.baseScale;
    voxelLookup.gridDim = modelGrid.gridDim;

    const uint32_t count = baseDomain.count;
    const uint32_t totalPlanes = baseDomain.totalPlanes;

    cudaUtils::CudaBuffer<double4> devMoments;
    cudaUtils::CudaBuffer<uint32_t> devPlanarPatchCounts;
    cudaUtils::CudaBuffer<uint32_t> devPlanarVertexCounts;
    cudaUtils::CudaBuffer<double> devRestrictedPlaneAreas;
    cudaUtils::CudaBuffer<double> devRestrictedPlaneAbsSums;
    cudaUtils::CudaBuffer<double3> devRestrictedPlaneMoments;
    cudaUtils::CudaBuffer<uint32_t> devFaceSegmentCounts;
    cudaUtils::CudaBuffer<uint32_t> devFaceSegmentOffsets;
    cudaUtils::CudaBuffer<RVDStatus> devStatuses;

    if (!devMoments.allocate(count) ||
        !devPlanarPatchCounts.allocate(count) ||
        !devPlanarVertexCounts.allocate(count) ||
        !devStatuses.allocate(count) ||
        (totalPlanes && !devRestrictedPlaneAreas.allocate(totalPlanes)) ||
        (totalPlanes && !devRestrictedPlaneAbsSums.allocate(totalPlanes)) ||
        (totalPlanes && !devRestrictedPlaneMoments.allocate(totalPlanes)) ||
        (totalPlanes && !devFaceSegmentCounts.allocate(size_t(totalPlanes) + 1)) ||
        (totalPlanes && !devFaceSegmentOffsets.allocate(size_t(totalPlanes) + 1))) return false;

    if (!cudaOk(cudaMemsetAsync(devMoments.get(), 0, count * sizeof(double4), stream), "clear moments"))
        return false;
    if (totalPlanes && !cudaOk(cudaMemsetAsync(devRestrictedPlaneAreas.get(), 0, totalPlanes * sizeof(double), stream), "clear restricted plane areas"))
        return false;
    if (totalPlanes && !cudaOk(cudaMemsetAsync(devRestrictedPlaneAbsSums.get(), 0, totalPlanes * sizeof(double), stream), "clear restricted plane abs sums"))
        return false;
    if (totalPlanes && !cudaOk(cudaMemsetAsync(devRestrictedPlaneMoments.get(), 0, totalPlanes * sizeof(double3), stream), "clear restricted plane moments"))
        return false;
    if (totalPlanes && !cudaOk(cudaMemsetAsync(devFaceSegmentCounts.get(), 0, (size_t(totalPlanes) + 1) * sizeof(uint32_t), stream), "clear face segment counts"))
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
    geomInput.triangleList = devGeom.voxelTriangleIds.get();
    geomInput.triangleListCount = uint32_t(devGeom.voxelTriangleIds.size());
    geomInput.offsets = devGeom.voxelOffsets.get();
    geomInput.offsetCount = uint32_t(devGeom.voxelOffsets.size());
    geomInput.baseGrid = baseDomain.baseGrid;
    geomInput.modelVoxelGrid = voxelLookup;
    geomInput.winding = devGeom.view();
    geomInput.domainMin = make_double3(
        double(baseDomain.domainMin.x), double(baseDomain.domainMin.y), double(baseDomain.domainMin.z));
    geomInput.invBaseScale = baseDomain.invBaseScale;
    geomInput.modelBoundsMin = make_float3(
        (devGeom.boundsMin.x - baseDomain.domainMin.x) * baseDomain.baseScale,
        (devGeom.boundsMin.y - baseDomain.domainMin.y) * baseDomain.baseScale,
        (devGeom.boundsMin.z - baseDomain.domainMin.z) * baseDomain.baseScale);
    geomInput.modelBoundsMax = make_float3(
        (devGeom.boundsMax.x - baseDomain.domainMin.x) * baseDomain.baseScale,
        (devGeom.boundsMax.y - baseDomain.domainMin.y) * baseDomain.baseScale,
        (devGeom.boundsMax.z - baseDomain.domainMin.z) * baseDomain.baseScale);
    geomInput.baseCells = baseDomain.deviceView();

    RestrictedDeviceOutput geomOutput{};
    geomOutput.moments = devMoments.get();
    geomOutput.planarPatchCounts = devPlanarPatchCounts.get();
    geomOutput.planarVertexCounts = devPlanarVertexCounts.get();
    geomOutput.restrictedPlaneOffsets = baseDomain.devCellPlaneOffsets.get();
    geomOutput.restrictedPlaneAreas = devRestrictedPlaneAreas.get();
    geomOutput.restrictedPlaneAbsSums = devRestrictedPlaneAbsSums.get();
    geomOutput.restrictedPlaneMoments = devRestrictedPlaneMoments.get();
    geomOutput.faceSegmentCounts = devFaceSegmentCounts.get();
    geomOutput.statuses = devStatuses.get();

    if (!launchRestrictionPass(geomInput, geomOutput, count, chunkSize, devStorage.get(), stream, false)) {
        std::cerr << "[RVD] restriction count pass failed modelId=" << restriction.runtimeModelId << std::endl;
        return false;
    }

    cudaUtils::CudaBuffer<uint32_t> devPatchOffsets;
    cudaUtils::CudaBuffer<uint32_t> devVertexOffsets;
    if (!devPatchOffsets.allocate(count) || !devVertexOffsets.allocate(count)) return false;

    size_t patchScanBytes = 0;
    size_t vertexScanBytes = 0;
    size_t segmentScanBytes = 0;

    if (!cudaOk(cub::DeviceScan::ExclusiveSum(
            nullptr, patchScanBytes,
            devPlanarPatchCounts.get(), devPatchOffsets.get(),
            count, stream), "query patch scan") ||
        !cudaOk(cub::DeviceScan::ExclusiveSum(
            nullptr, vertexScanBytes,
            devPlanarVertexCounts.get(), devVertexOffsets.get(),
            count, stream), "query vertex scan") ||
        (totalPlanes > 0 && !cudaOk(cub::DeviceScan::ExclusiveSum(
            nullptr, segmentScanBytes,
            devFaceSegmentCounts.get(), devFaceSegmentOffsets.get(),
            int(totalPlanes + 1), stream), "query face segment scan")))
        return false;

    cudaUtils::CudaBuffer<uint8_t> scanStorage;
    if (!scanStorage.allocate(std::max({patchScanBytes, vertexScanBytes, segmentScanBytes})))
        return false;

    if (!cudaOk(cub::DeviceScan::ExclusiveSum(
            scanStorage.get(), patchScanBytes,
            devPlanarPatchCounts.get(), devPatchOffsets.get(),
            count, stream), "execute patch scan") ||
        !cudaOk(cub::DeviceScan::ExclusiveSum(
            scanStorage.get(), vertexScanBytes,
            devPlanarVertexCounts.get(), devVertexOffsets.get(),
            count, stream), "execute vertex scan") ||
        (totalPlanes > 0 && !cudaOk(cub::DeviceScan::ExclusiveSum(
            scanStorage.get(), segmentScanBytes,
            devFaceSegmentCounts.get(), devFaceSegmentOffsets.get(),
            int(totalPlanes + 1), stream), "execute face segment scan")))
        return false;

    uint32_t lastPatchOffset = 0, lastPatchCount = 0;
    uint32_t lastVertexOffset = 0, lastVertexCount = 0;
    uint32_t totalFaceSegments = 0;
    if (count > 0) {
        if (!cudaOk(cudaMemcpyAsync(&lastPatchOffset, devPatchOffsets.get() + (count - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), "download last patch offset") ||
            !cudaOk(cudaMemcpyAsync(&lastPatchCount, devPlanarPatchCounts.get() + (count - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), "download last patch count") ||
            !cudaOk(cudaMemcpyAsync(&lastVertexOffset, devVertexOffsets.get() + (count - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), "download last vertex offset") ||
            !cudaOk(cudaMemcpyAsync(&lastVertexCount, devPlanarVertexCounts.get() + (count - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), "download last vertex count") ||
            (totalPlanes > 0 && !cudaOk(cudaMemcpyAsync(&totalFaceSegments, devFaceSegmentOffsets.get() + totalPlanes, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), "download face segment count")) ||
            !cudaOk(cudaStreamSynchronize(stream), "sync counts")) return false;
    }
    const uint32_t totalPatches = lastPatchOffset + lastPatchCount;
    const uint32_t totalVertices = lastVertexOffset + lastVertexCount;

    cudaUtils::CudaBuffer<RestrictedSurface> devSurfaces;
    cudaUtils::CudaBuffer<RestrictedSurfaceGeometry> devSurfaceGeometry;
    cudaUtils::CudaBuffer<double3> devRestrictedVertices;
    cudaUtils::CudaBuffer<FaceBoundarySegment> devFaceSegments;

    if (totalPatches > 0) {
        if (!devSurfaces.allocate(totalPatches) ||
            !devSurfaceGeometry.allocate(totalPatches)) return false;
    }
    if (totalVertices > 0) {
        if (!devRestrictedVertices.allocate(totalVertices)) return false;
    }
    if (totalFaceSegments > 0) {
        if (!devFaceSegments.allocate(totalFaceSegments)) return false;
    }

    geomOutput.planarPatchOffsets = devPatchOffsets.get();
    geomOutput.planarVertexOffsets = devVertexOffsets.get();
    geomOutput.outSurfaces = devSurfaces.get();
    geomOutput.outSurfaceGeometry = devSurfaceGeometry.get();
    geomOutput.outRestrictedVertices = devRestrictedVertices.get();
    geomOutput.faceSegmentCounts = nullptr;
    geomOutput.faceSegmentOffsets = devFaceSegmentOffsets.get();
    geomOutput.outFaceSegments = devFaceSegments.get();

    if (!launchRestrictionPass(geomInput, geomOutput, count, chunkSize, devStorage.get(), stream, true)) {
        std::cerr << "[RVD] restriction write pass failed modelId=" << restriction.runtimeModelId << std::endl;
        return false;
    }

    cudaUtils::CudaBuffer<unsigned long long> devPackedError;
    if (!devPackedError.allocate(1)) return false;
    cudaMemsetAsync(devPackedError.get(), 0, sizeof(unsigned long long), stream);

    if (count > 0) {
        constexpr uint32_t BlockSize = 256;
        const uint32_t blocks = (count + BlockSize - 1) / BlockSize;
        normalizeRestrictedPlanesKernel<<<blocks, BlockSize, 0, stream>>>(
            baseDomain.deviceView(),
            baseDomain.devCellPlaneOffsets.get(),
            devStatuses.get(),
            devMoments.get(),
            devRestrictedPlaneAreas.get(),
            devRestrictedPlaneMoments.get(),
            devRestrictedPlaneAbsSums.get(),
            count,
            restriction.runtimeModelId,
            baseDomain.nominalCellSize,
            baseDomain.invBaseScale,
            baseDomain.invBaseAreaScale,
            make_float3(baseDomain.domainMin.x, baseDomain.domainMin.y, baseDomain.domainMin.z),
            devPackedError.get());
    }

    unsigned long long packedError = 0;
    cudaMemcpyAsync(&packedError, devPackedError.get(), sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (packedError != 0) {
        const uint32_t errorStatus = static_cast<uint32_t>(packedError >> 32);
        const uint32_t errorCell = static_cast<uint32_t>(packedError & 0xFFFFFFFFULL);
        std::cerr << "[RVD] restriction failed cell=" << errorCell << " modelId=" << restriction.runtimeModelId
                  << " code=" << errorStatus << std::endl;
        return false;
    }

    restriction.devMoments = std::move(devMoments);
    restriction.devRestrictedPlaneAreas = std::move(devRestrictedPlaneAreas);
    restriction.devRestrictedPlaneAbsSums = std::move(devRestrictedPlaneAbsSums);
    restriction.devRestrictedPlaneMoments = std::move(devRestrictedPlaneMoments);
    restriction.devSurfaces = std::move(devSurfaces);
    restriction.devSurfaceGeometry = std::move(devSurfaceGeometry);
    restriction.devRestrictedVertices = std::move(devRestrictedVertices);
    restriction.devFaceSegmentOffsets = std::move(devFaceSegmentOffsets);
    restriction.devFaceSegments = std::move(devFaceSegments);
    restriction.totalPatches = totalPatches;
    restriction.totalFaceSegments = totalFaceSegments;
    return true;
}

} // namespace voronoi
