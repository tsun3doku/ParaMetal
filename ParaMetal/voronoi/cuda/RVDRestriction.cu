#include "RVDRestriction.cuh"
#include "ConvexCell.cuh"
#include "RVDModelGeometry.cuh"
#include "RVDHostUtils.cuh"
#include "../../cuda/CudaMath.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>

namespace voronoi {

namespace {

constexpr int MaxTrianglePlaneSegments = 64;
constexpr uint32_t RestrictionBlockSize = 32;
constexpr double FlatTetrahedronFraction = 1e-12;

const char *restrictionFailureName(uint32_t code) {
  if (code < uint32_t(RVDStatus::Count)) {
    return rvdStatusName(RVDStatus(code));
  }
  return rvdRestrictionErrorName(code);
}

using DeviceTriangle = cudaGeometry::DeviceTriangle;
using rvdDevice::ClipVertexProvenance;
using rvdDevice::ConvexCell;
using rvdDevice::CellClipOverflowWorkspace;
using rvdDevice::ConvexCellOverflowStorage;
using rvdDevice::DiscoveryLimits;
using rvdDevice::MaxClippedPolygonVertices;
using rvdDevice::NoPlaneNeighbor;
using rvdDevice::PlaneClippedSegment;
using rvdDevice::clipTriangleToCellSegments;
using rvdDevice::determinant3Double;
using rvdDevice::planeEvaluateFilteredExact;
using cudaMath::finite3;
using rvdDevice::integrateCell;
using rvdDevice::integratePolygon;

using cudaMath::cross3;
using cudaMath::dot3;
using cudaMath::max3;
using cudaMath::min3;
using cudaMath::mul3;
using cudaMath::sub3;

struct RestrictedDeviceInput {
  DeviceKnn knn;
  const DeviceTriangle *triangles;
  const uint3 *triangleVertexIds;
  uint32_t triangleCount;
  const uint8_t *occupancy;
  uint32_t occupancyCount;
  const int32_t *triangleList;
  uint32_t triangleListCount;
  const int32_t *offsets;
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
  double4 *moments = nullptr;
  uint32_t *planarPatchCounts = nullptr;
  const uint32_t *planarPatchOffsets = nullptr;
  uint32_t *planarVertexCounts = nullptr;
  const uint32_t *planarVertexOffsets = nullptr;
  RestrictedSurface *outSurfaces = nullptr;
  double3 *outSurfaceNormals = nullptr;
  double3 *outRestrictedVertices = nullptr;
  const uint32_t *restrictedPlaneOffsets = nullptr;
  double *restrictedPlaneAreas = nullptr;
  uint32_t *faceSegmentCounts = nullptr;
  const uint32_t *faceSegmentOffsets = nullptr;
  FaceBoundarySegment *outFaceSegments = nullptr;
  RvdPackedError *packedError = nullptr;
  RVDStatus *statuses = nullptr;
};

struct RestrictedCellStorage {
  ConvexCellOverflowStorage cell;
  CellClipOverflowWorkspace workspace;
  uint3 backupVertices[RVDOverflowVertexCount];
  int32_t uniqueTriangles[DiscoveryLimits::T];
};

struct RestrictedCellWork {
  uint32_t seedId = 0;
  ConvexCell cell;
  int basePlaneCount = 0;
  int baseVertexCount = 0;
  uint32_t planeSegmentCursor[RVDOverflowPlaneCount] = {};
  double4 fullCell = make_double4(0.0, 0.0, 0.0, 0.0);
  double absoluteVolumeSum = 0.0;
  double surfacePatchArea = 0.0;
  uint32_t planarPatchCount = 0;
  uint32_t planarVertexCount = 0;
  double4 moment = make_double4(0.0, 0.0, 0.0, 0.0);
  int3 voxelMinimum{};
  int3 voxelMaximum{};
  float3 cellOrigin{};
  float3 seedMomentCenter{};
  bool referenceInside = false;
  bool surfaceOnly = false;
  int uniqueCount = 0;
};

__device__ inline double4 planeFromPoints(double3 a, double3 b, double3 c) {
  const double3 normal = cross3(sub3(b, a), sub3(c, a));
  return make_double4(normal.x, normal.y, normal.z, -dot3(normal, a));
}

__device__ inline double3 toCanonical(float4 point, float3 origin, double scale) {
  return make_double3(
      (double(point.x) - double(origin.x)) * scale,
      (double(point.y) - double(origin.y)) * scale,
      (double(point.z) - double(origin.z)) * scale);
}

__device__ inline void fillFaceCutKey(
    FaceCutVertexKey &key, const ClipVertexProvenance &provenance,
    uint32_t triangleId, uint32_t planeBase) {
  key.triangleId = triangleId;
  if (provenance.triangleVertexB != InvalidId) {
    key.cadEdgeVertexA = provenance.triangleVertexA;
    key.cadEdgeVertexB = provenance.triangleVertexB;
  } else {
    key.cadVertexId = provenance.triangleVertexA;
  }
  if (provenance.planeA != InvalidId) {
    key.basePlaneA = planeBase + provenance.planeA;
  }
  if (provenance.planeB != InvalidId) {
    key.basePlaneB = planeBase + provenance.planeB;
  }
  if (provenance.planeC != InvalidId) {
    key.basePlaneC = planeBase + provenance.planeC;
  }
}



__device__ bool cellBounds(const ConvexCell &cell, float3 &minimum, float3 &maximum) {
  minimum = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  maximum = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

  for (int vertex = 0;
       vertex < cell.vertexCount && cell.status == RVDStatus::Success;
       ++vertex) {
    const double3 point = cell.vertexPosition(cell.vertices[vertex]);
    const float3 pointF = make_float3(float(point.x), float(point.y), float(point.z));
    minimum = min3(minimum, pointF);
    maximum = max3(maximum, pointF);
  }

  return cell.status == RVDStatus::Success && finite3(minimum) && finite3(maximum);
}

__device__ void voxelBounds(
    const RestrictedDeviceInput &input,
    float3 minimum,
    float3 maximum,
    int3 &voxelMinimum,
    int3 &voxelMaximum) {
  const cudaGeometry::DeviceVoxelGrid &vox = input.modelVoxelGrid;
  const float invVoxelSize = 1.0f / vox.voxelSize;

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

__device__ bool isOutsideModel(
    float3 cellMinimum,
    float3 cellMaximum,
    const RestrictedDeviceInput &input) {
  return cellMaximum.x < input.modelBoundsMin.x ||
         cellMinimum.x > input.modelBoundsMax.x ||
         cellMaximum.y < input.modelBoundsMin.y ||
         cellMinimum.y > input.modelBoundsMax.y ||
         cellMaximum.z < input.modelBoundsMin.z ||
         cellMinimum.z > input.modelBoundsMax.z;
}

__device__ bool findCellOrigin(
    const RestrictedDeviceInput &input,
    int3 voxelMinimum,
    int3 voxelMaximum,
    float3 &origin,
    bool &inside) {
  const cudaGeometry::DeviceVoxelGrid &vox = input.modelVoxelGrid;
  const int cStride = vox.gridDim.x + 1;
  const float voxelSize = vox.voxelSize;
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
        if (index < 0 || uint32_t(index) >= input.occupancyCount) continue;

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

  const int faceCoordinates[3] = {voxelMaximum.x + 1, voxelMaximum.y + 1, voxelMaximum.z + 1};
  for (int axis = 2; axis >= 0; --axis) {
    const int limit = axis == 0 ? vox.gridDim.x
        : axis == 1 ? vox.gridDim.y
                    : vox.gridDim.z;
    if (faceCoordinates[axis] < limit) continue;

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
  return false;
}

__device__ inline void failCell(
    uint32_t seedId,
    RVDStatus status,
    RestrictedDeviceOutput output) {
  output.statuses[seedId] = status;
  if (status != RVDStatus::Success &&
      status != RVDStatus::EmptyCell &&
      output.packedError != nullptr) {
    rvdReportErrorOnce(output.packedError, uint32_t(status), seedId);
  }
}

__device__ inline void zeroCellCounts(
    uint32_t seedId,
    RestrictedDeviceOutput output,
    int basePlaneCount) {
  if (output.planarPatchCounts != nullptr) output.planarPatchCounts[seedId] = 0u;
  if (output.planarVertexCounts != nullptr) output.planarVertexCounts[seedId] = 0u;
  if (output.faceSegmentCounts != nullptr &&
      output.restrictedPlaneOffsets != nullptr) {
    const uint32_t base = output.restrictedPlaneOffsets[seedId];
    for (int p = 0; p < basePlaneCount; ++p) {
      output.faceSegmentCounts[base + uint32_t(p)] = 0u;
    }
  }
}

__device__ bool initializeOutput(
    uint32_t seedId,
    RestrictedDeviceOutput output,
    bool surfaceOnly) {
  if (!surfaceOnly) {
    if (output.planarPatchCounts != nullptr) output.planarPatchCounts[seedId] = 0u;
    if (output.planarVertexCounts != nullptr) output.planarVertexCounts[seedId] = 0u;
    return true;
  }

  if (output.planarPatchCounts == nullptr ||
      output.planarPatchCounts[seedId] == 0u ||
      output.statuses == nullptr) {
    return false;
  }
  const RVDStatus status = output.statuses[seedId];
  return status == RVDStatus::Success;
}

__device__ bool prepareCell(
    uint32_t seedId,
    const RestrictedDeviceInput &input,
    RestrictedDeviceOutput output,
    RestrictedCellStorage &storage,
    RestrictedCellWork &work) {
  if (seedId >= input.knn.seedCount) {
    failCell(seedId, RVDStatus::InvalidIndex, output);
    return false;
  }

  ConvexCell &cell = work.cell;
  cell.attachStorage(storage.cell, storage.workspace, RVDOverflowPlaneCount, RVDOverflowVertexCount);
  if (!cell.loadStorage(resolveBaseCell(input.baseCells, seedId))) {
    failCell(seedId, cell.status == RVDStatus::Success ? RVDStatus::InvalidIndex : cell.status, output);
    return false;
  }

  work.basePlaneCount = cell.planeCount;
  work.baseVertexCount = cell.vertexCount;
  for (int p = 0; p < work.basePlaneCount && p < int(RVDOverflowPlaneCount); ++p) {
    work.planeSegmentCursor[p] = 0u;
  }

  if (!work.surfaceOnly) {
    cell.clearPlaneMeasures(work.basePlaneCount);
  }

  float3 cellMinimum;
  float3 cellMaximum;
  if (!cellBounds(cell, cellMinimum, cellMaximum)) {
    failCell(seedId, cell.status == RVDStatus::Success
        ? RVDStatus::NonFiniteGeometry
        : cell.status, output);
    return false;
  }

  if (isOutsideModel(cellMinimum, cellMaximum, input)) {
    if (!work.surfaceOnly) {
      output.moments[seedId] = make_double4(0.0, 0.0, 0.0, 0.0);
      failCell(seedId, RVDStatus::EmptyCell, output);
    }
    return false;
  }

  voxelBounds(input, cellMinimum, cellMaximum, work.voxelMinimum, work.voxelMaximum);

  if (!work.surfaceOnly) {
    if (!findCellOrigin(input, work.voxelMinimum, work.voxelMaximum, work.cellOrigin, work.referenceInside)) {
      if (output.packedError != nullptr &&
          rvdReportErrorOnce(
              output.packedError, uint32_t(RVDRestrictionError::MissingCellOrigin),
              seedId)) {
        printf("[RVDRestriction] missing cell origin cell=%u box=(%d %d %d)-(%d %d %d) "
               "grid=(%d %d %d)\n",
               seedId, work.voxelMinimum.x, work.voxelMinimum.y, work.voxelMinimum.z,
               work.voxelMaximum.x, work.voxelMaximum.y, work.voxelMaximum.z,
               input.modelVoxelGrid.gridDim.x, input.modelVoxelGrid.gridDim.y,
               input.modelVoxelGrid.gridDim.z);
      }
      failCell(seedId, RVDStatus::InconsistentBoundary, output);
      return false;
    }
  }

  const float4 seed = input.knn.seeds[seedId];
  work.seedMomentCenter = cudaGeometry::canonicalFromWorld(
      input.baseGrid, make_float3(seed.x, seed.y, seed.z));
  return true;
}

__device__ bool collectUniqueTriangles(
    uint32_t seedId,
    const RestrictedDeviceInput &input,
    RestrictedDeviceOutput output,
    RestrictedCellStorage &storage,
    RestrictedCellWork &work) {
  const int dimX = input.modelVoxelGrid.gridDim.x;
  const int dimY = input.modelVoxelGrid.gridDim.y;

  for (int z = work.voxelMinimum.z; z <= work.voxelMaximum.z; ++z) {
    for (int y = work.voxelMinimum.y; y <= work.voxelMaximum.y; ++y) {
      for (int x = work.voxelMinimum.x; x <= work.voxelMaximum.x; ++x) {
        const int voxelIndex = z * dimX * dimY + y * dimX + x;
        if (voxelIndex < 0 || uint32_t(voxelIndex + 1) >= input.offsetCount) continue;

        const int32_t begin = input.offsets[voxelIndex];
        const int32_t end = input.offsets[voxelIndex + 1];
        for (int32_t listIndex = begin; listIndex < end; ++listIndex) {
          if (listIndex < 0 || uint32_t(listIndex) >= input.triangleListCount) continue;

          const int32_t triangleIndex = input.triangleList[listIndex];
          if (triangleIndex < 0 || uint32_t(triangleIndex) >= input.triangleCount) continue;

          bool seen = false;
          for (int u = 0; u < work.uniqueCount; ++u) {
            if (storage.uniqueTriangles[u] == triangleIndex) {
              seen = true;
              break;
            }
          }
          if (seen) continue;

          if (uint32_t(work.uniqueCount) >= DiscoveryLimits::T) {
            failCell(seedId, RVDStatus::TriangleOverflow, output);
            return false;
          }
          storage.uniqueTriangles[work.uniqueCount++] = triangleIndex;
        }
      }
    }
  }
  return true;
}

__device__ bool integrateWholeCell(
    uint32_t seedId,
    RestrictedDeviceOutput output,
    RestrictedCellStorage &storage,
    RestrictedCellWork &work) {
  ConvexCell &cell = work.cell;
  work.fullCell = integrateCell(cell, work.seedMomentCenter, work.basePlaneCount);
  if (cell.status != RVDStatus::Success) {
    if (cell.status == RVDStatus::InconsistentBoundary &&
        output.packedError != nullptr) {
      rvdReportErrorOnce(
          output.packedError, uint32_t(RVDRestrictionError::BaseCellBoundary),
          seedId);
    }
    failCell(seedId, cell.status, output);
    return false;
  }

  if (work.referenceInside) {
    work.moment = make_double4(
        work.fullCell.x / 6.0,
        work.fullCell.y / 6.0,
        work.fullCell.z / 6.0,
        work.fullCell.w / 6.0);
    work.absoluteVolumeSum = fabs(work.fullCell.w / 6.0);
  } else {
    cell.clearPlaneMeasures(work.basePlaneCount);
  }

  for (int i = 0; i < work.baseVertexCount; ++i) {
    storage.backupVertices[i] = cell.vertices[i];
  }
  return true;
}

__device__ bool emitSurfacePatch(
    uint32_t seedId,
    int triangleIndex,
    double3 v0D,
    double3 v1D,
    double3 v2D,
    const double3 *polygon,
    int polygonCount,
    double area,
    const PlaneClippedSegment *planeSegments,
    int planeSegmentCount,
    const RestrictedDeviceInput &input,
    RestrictedDeviceOutput output,
    bool surfaceOnly,
    RestrictedCellWork &work) {
  double3 triangleCross = make_double3(0.0, 0.0, 0.0);
  double triangleNormalLength = 0.0;
  if (surfaceOnly) {
    triangleCross = cross3(sub3(v1D, v0D), sub3(v2D, v0D));
    triangleNormalLength = sqrt(dot3(triangleCross, triangleCross));
  }

  if (!surfaceOnly) {
    if (output.faceSegmentCounts != nullptr &&
        output.restrictedPlaneOffsets != nullptr) {
      const uint32_t planeBase = output.restrictedPlaneOffsets[seedId];
      for (int s = 0; s < planeSegmentCount; ++s) {
        const int plane = planeSegments[s].planeIndex;
        if (plane < 0 || plane >= work.basePlaneCount) continue;
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
      if (plane < 0 || plane >= work.basePlaneCount) continue;

      const uint32_t slot = planeBase + uint32_t(plane);
      const uint32_t dst =
          output.faceSegmentOffsets[slot] + work.planeSegmentCursor[plane]++;

      FaceBoundarySegment segment{};
      segment.a = cudaGeometry::canonicalToWorld(planeSegments[s].a, input.domainMin, invBaseScale);
      segment.b = cudaGeometry::canonicalToWorld(planeSegments[s].b, input.domainMin, invBaseScale);
      segment.planeNormal = triangleNormalLength > 0.0
          ? make_double3(
                triangleCross.x / triangleNormalLength,
                triangleCross.y / triangleNormalLength,
                triangleCross.z / triangleNormalLength)
          : make_double3(0.0, 0.0, 1.0);
      const uint32_t triangleId = uint32_t(triangleIndex);
      segment.triangleId = triangleId;
      segment.runtimeModelId = input.winding.runtimeModelId;
      const double4 basePlane = work.cell.planes[plane];
      const bool coplanar =
          planeEvaluateFilteredExact(basePlane, v0D).sign == 0 &&
          planeEvaluateFilteredExact(basePlane, v1D).sign == 0 &&
          planeEvaluateFilteredExact(basePlane, v2D).sign == 0;
      if (coplanar) {
        segment.segmentType = uint32_t(FaceCutSegmentType::Coplanar);
      }
      fillFaceCutKey(segment.keyA, planeSegments[s].provenanceA, triangleId, planeBase);
      fillFaceCutKey(segment.keyB, planeSegments[s].provenanceB, triangleId, planeBase);
      output.outFaceSegments[dst] = segment;
    }
  }

  if (output.planarPatchOffsets != nullptr) {
    const uint32_t patchDst = output.planarPatchOffsets[seedId] + work.planarPatchCount;
    const uint32_t vertexDst = output.planarVertexOffsets != nullptr
        ? output.planarVertexOffsets[seedId] + work.planarVertexCount
        : 0u;

    if (!(triangleNormalLength > 0.0) || !isfinite(triangleNormalLength)) {
      failCell(seedId, RVDStatus::NonFiniteGeometry, output);
      return false;
    }
    const double3 normal = mul3(triangleCross, 1.0 / triangleNormalLength);

    if (output.outSurfaces != nullptr) {
      RestrictedSurface surface{};
      surface.cellId = seedId;
      surface.firstVertex = vertexDst;
      surface.vertexCount = uint32_t(polygonCount);
      surface.triangleId = uint32_t(triangleIndex);
      output.outSurfaces[patchDst] = surface;
    }

    if (output.outSurfaceNormals != nullptr) {
      output.outSurfaceNormals[patchDst] = make_double3(normal.x, normal.y, normal.z);
    }

    if (output.outRestrictedVertices != nullptr) {
      for (int v = 0; v < polygonCount; ++v) {
        output.outRestrictedVertices[vertexDst + v] = polygon[v];
      }
    }
  }

  ++work.planarPatchCount;
  work.planarVertexCount += uint32_t(polygonCount);
  return true;
}

__device__ bool clipTriangleVolume(
    double3 v0D,
    double3 v1D,
    double3 v2D,
    double3 cellOrigin,
    const uint3 *backupVertices,
    uint32_t triangleIndex,
    RvdPackedError *packedError,
    RestrictedCellWork &work) {
  ConvexCell &cell = work.cell;
  const double3 d0 = sub3(v0D, cellOrigin);
  const double3 d1 = sub3(v1D, cellOrigin);
  const double3 d2 = sub3(v2D, cellOrigin);
  const double det = determinant3Double(d0, d1, d2);
  const double cellVolume = fabs(work.fullCell.w);
  if (!(cellVolume > 0.0) || fabs(det) <= FlatTetrahedronFraction * cellVolume) {
    return true;
  }

  const double sign = det > 0.0 ? 1.0 : -1.0;

  if (det < 0.0) {
    const double3 temporary = v1D;
    v1D = v2D;
    v2D = temporary;
  }

  const double4 tetrahedronPlanes[4] = {
      planeFromPoints(cellOrigin, v0D, v1D),
      planeFromPoints(cellOrigin, v2D, v0D),
      planeFromPoints(cellOrigin, v1D, v2D),
      planeFromPoints(v1D, v2D, v0D)};

  bool emptyIntersection = false;
  for (int plane = 0; plane < 4; ++plane) {
    if (!cell.addPlane(tetrahedronPlanes[plane], NoPlaneNeighbor)) {
      if (cell.status == RVDStatus::EmptyCell) {
        emptyIntersection = true;
      } else if (cell.status == RVDStatus::InconsistentBoundary &&
                 packedError != nullptr &&
                 rvdReportErrorOnce(
                     packedError, uint32_t(RVDRestrictionError::TriangleTopology),
                     work.seedId)) {
        printf("[RVDRestriction] triangle topology cell=%u triangle=%u plane=%d "
               "vertices=%d planes=%d\n",
               work.seedId, triangleIndex, plane, cell.vertexCount, cell.planeCount);
      }
      break;
    }
  }

  if (!emptyIntersection && cell.status != RVDStatus::Success) return false;
  if (!emptyIntersection) {
    const double4 contribution = integrateCell(cell, work.seedMomentCenter, work.basePlaneCount, -sign);
    const double deltaVolume = contribution.w / 6.0;
    work.moment.x += contribution.x / 6.0;
    work.moment.y += contribution.y / 6.0;
    work.moment.z += contribution.z / 6.0;
    work.moment.w += deltaVolume;
    work.absoluteVolumeSum += fabs(deltaVolume);

    if (cell.status != RVDStatus::Success) {
      if (cell.status == RVDStatus::InconsistentBoundary &&
          packedError != nullptr &&
          rvdReportErrorOnce(
              packedError, uint32_t(RVDRestrictionError::TriangleClipMeasures),
              work.seedId)) {
        printf("[RVDRestriction] triangle clip measures cell=%u triangle=%u "
               "vertices=%d planes=%d\n",
               work.seedId, triangleIndex, cell.vertexCount, cell.planeCount);
      }
      return false;
    }
  }

  cell.vertexCount = work.baseVertexCount;
  cell.planeCount = work.basePlaneCount;
  cell.status = RVDStatus::Success;
  for (int i = 0; i < work.baseVertexCount; ++i) {
    cell.vertices[i] = backupVertices[i];
  }
  return true;
}

__device__ bool processTriangle(
    uint32_t seedId,
    int triangleIndex,
    const RestrictedDeviceInput &input,
    RestrictedDeviceOutput output,
    bool surfaceOnly,
    RestrictedCellStorage &storage,
    RestrictedCellWork &work) {
  const DeviceTriangle triangle = input.triangles[triangleIndex];
  const double scale = double(input.baseGrid.scale);
  const float3 origin = input.baseGrid.gridMin;

  const double3 v0D = toCanonical(triangle.v0, origin, scale);
  double3 v1D = toCanonical(triangle.v1, origin, scale);
  double3 v2D = toCanonical(triangle.v2, origin, scale);
  if (!finite3(v0D) || !finite3(v1D) || !finite3(v2D)) {
    failCell(seedId, RVDStatus::NonFiniteGeometry, output);
    return false;
  }

  double3 polygon[MaxClippedPolygonVertices];
  PlaneClippedSegment planeSegments[MaxTrianglePlaneSegments];
  int planeSegmentCount = 0;
  ClipVertexProvenance inputProvenance[3];
  if (input.triangleVertexIds != nullptr) {
    const uint3 vertexIds = input.triangleVertexIds[triangleIndex];
    inputProvenance[0].triangleVertexA = vertexIds.x;
    inputProvenance[1].triangleVertexA = vertexIds.y;
    inputProvenance[2].triangleVertexA = vertexIds.z;
  }
  const int polygonCount = clipTriangleToCellSegments(
      v0D, v1D, v2D, work.cell, work.basePlaneCount, polygon,
      MaxClippedPolygonVertices, planeSegments, &planeSegmentCount,
      MaxTrianglePlaneSegments, inputProvenance);
  if (polygonCount < 0) {
    failCell(seedId, planeSegmentCount >= MaxTrianglePlaneSegments
        ? RVDStatus::SegmentOverflow
        : RVDStatus::VertexOverflow, output);
    return false;
  }

  if (polygonCount >= 3) {
    double area = 0.0;
    double3 patchMoment = make_double3(0.0, 0.0, 0.0);
    if (!integratePolygon(polygon, polygonCount, area, patchMoment)) {
      failCell(seedId, RVDStatus::NonFiniteGeometry, output);
      return false;
    }

    if (area > 0.0 && isfinite(area)) {
      work.surfacePatchArea += area;
      if (!emitSurfacePatch(
              seedId, triangleIndex, v0D, v1D, v2D, polygon,
              polygonCount, area,
              planeSegments, planeSegmentCount, input, output, surfaceOnly,
              work)) {
        return false;
      }
    }
  }

  if (surfaceOnly) return true;

  const double3 cellOrigin = make_double3(
      double(work.cellOrigin.x),
      double(work.cellOrigin.y),
      double(work.cellOrigin.z));
  if (!clipTriangleVolume(
          v0D, v1D, v2D, cellOrigin, storage.backupVertices,
          uint32_t(triangleIndex), output.packedError, work)) {
    failCell(seedId, work.cell.status, output);
    return false;
  }
  return true;
}

__device__ inline void reportPlaneAreaOutOfRange(
    RvdPackedError *packedError,
    uint32_t cell,
    uint32_t plane,
    const char *reason) {
  if (rvdReportErrorOnce(
          packedError, uint32_t(RVDRestrictionError::PlaneAreaOutOfRange), cell)) {
    printf("[RVDRestriction] PlaneAreaOutOfRange cell=%u plane=%u reason=%s\n",
           cell, plane, reason);
  }
}

__device__ bool writeRestrictedPlaneArea(
    uint32_t seedId,
    uint32_t plane,
    const RestrictedDeviceInput &input,
    const BaseCellArrays &cellState,
    RestrictedDeviceOutput output,
    double rawArea,
    double areaScale) {
  const uint32_t dst = output.restrictedPlaneOffsets[seedId];
  const double baseArea = cellState.planeAreas != nullptr
      ? double(cellState.planeAreas[plane]) * areaScale
      : 0.0;
  if (!isfinite(baseArea) || !isfinite(rawArea)) {
    reportPlaneAreaOutOfRange(output.packedError, seedId, plane, "nonFiniteArea");
    return false;
  }
  const double baseAbsSum = cellState.planeAreaAbsSums != nullptr
      ? cellState.planeAreaAbsSums[plane] * areaScale
      : fabs(baseArea);
  if (!isfinite(baseAbsSum) || baseAbsSum < 0.0) {
    reportPlaneAreaOutOfRange(output.packedError, seedId, plane, "invalidAbsoluteSum");
    return false;
  }
  const double extent = double(cudaGeometry::CanonicalDomainSize) * input.invBaseScale;
  const double facetFloor = 64.0 * DBL_EPSILON * extent * extent;
  const double areaTolerance = fmax(facetFloor, fabs(baseArea) * 1e-6);

  // reject non-positive plane area residuals
  if (!(baseArea > areaTolerance)) {
    if (rawArea > areaTolerance) {
      reportPlaneAreaOutOfRange(
          output.packedError, seedId, plane, "coverageOnDegenerateBaseFace");
      return false;
    }
    output.restrictedPlaneAreas[dst + plane] = 0.0;
    return true;
  }

  if (rawArea < 0.0) {
    if (rawArea < -areaTolerance) {
      reportPlaneAreaOutOfRange(
          output.packedError, seedId, plane, "negativeRestrictedArea");
      return false;
    }
    output.restrictedPlaneAreas[dst + plane] = 0.0;
    return true;
  }
  if (rawArea > baseArea + areaTolerance) {
    reportPlaneAreaOutOfRange(
        output.packedError, seedId, plane, "restrictedAreaExceedsBase");
    return false;
  }
  output.restrictedPlaneAreas[dst + plane] = rawArea;
  return true;
}

__device__ bool writeRestrictedPlaneMeasures(
    uint32_t seedId,
    const RestrictedDeviceInput &input,
    RestrictedDeviceOutput output,
    const RestrictedCellWork &work) {
  if (output.restrictedPlaneAreas == nullptr ||
      output.restrictedPlaneOffsets == nullptr) {
    return true;
  }

  const BaseCellArrays cellState = resolveBaseCell(input.baseCells, seedId);
  const double invBaseScale = 1.0 / double(input.baseGrid.scale);
  const double areaScale = invBaseScale * invBaseScale;

  for (int p = 0; p < work.basePlaneCount && p < int(cellState.planeCount); ++p) {
    const double worldArea = work.cell.planeAreas[p] * areaScale;
    if (!writeRestrictedPlaneArea(
            seedId, uint32_t(p), input, cellState, output, worldArea, areaScale)) {
      return false;
    }
  }
  return true;
}

__device__ bool writeIntegratedCell(
    uint32_t seedId,
    ConvexCell &cell,
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
    output.moments[seedId] = moment;
  }
  output.statuses[seedId] = RVDStatus::Success;
  return true;
}

__device__ inline bool writeEmptyRestrictedCell(
    uint32_t seedId,
    RestrictedDeviceOutput output) {
  if (output.moments != nullptr) {
    output.moments[seedId] = make_double4(0.0, 0.0, 0.0, 0.0);
  }
  failCell(seedId, RVDStatus::EmptyCell, output);
  return false;
}

__device__ inline bool writeFullBaseCell(
    uint32_t seedId,
    const RestrictedDeviceInput &input,
    RestrictedDeviceOutput output,
    const RestrictedCellWork &work) {
  const BaseCellArrays cellState = resolveBaseCell(input.baseCells, seedId);
  const double invBaseScale = 1.0 / double(input.baseGrid.scale);
  const double areaScale = invBaseScale * invBaseScale;

  if (output.restrictedPlaneOffsets != nullptr &&
      output.restrictedPlaneAreas != nullptr) {
    for (int p = 0; p < work.basePlaneCount && p < int(cellState.planeCount); ++p) {
      const double worldArea = cellState.planeAreas != nullptr
          ? double(cellState.planeAreas[p]) * areaScale
          : 0.0;
      if (!writeRestrictedPlaneArea(
              seedId, uint32_t(p), input, cellState, output, worldArea, areaScale)) {
        return false;
      }
    }
  }

  const double4 fullMoment = make_double4(
      work.fullCell.x / 6.0,
      work.fullCell.y / 6.0,
      work.fullCell.z / 6.0,
      work.fullCell.w / 6.0);

  if (output.moments != nullptr) {
    output.moments[seedId] = fullMoment;
  }
  output.statuses[seedId] = RVDStatus::Success;
  return true;
}

__device__ bool finalizeRestrictedCell(
    uint32_t seedId,
    const RestrictedDeviceInput &input,
    RestrictedDeviceOutput output,
    RestrictedCellWork &work) {
  if (!isfinite(work.surfacePatchArea)) {
    failCell(seedId, RVDStatus::NonFiniteGeometry, output);
    return false;
  }

  if (output.planarPatchCounts != nullptr) {
    output.planarPatchCounts[seedId] = work.planarPatchCount;
  }
  if (output.planarVertexCounts != nullptr) {
    output.planarVertexCounts[seedId] = work.planarVertexCount;
  }

  if (work.planarPatchCount == 0) {
    double3 samplePoint = make_double3(
        double(input.knn.seeds[seedId].x),
        double(input.knn.seeds[seedId].y),
        double(input.knn.seeds[seedId].z));
    if (work.fullCell.w > 0.0 && isfinite(work.fullCell.w)) {
      const double invW = 1.0 / work.fullCell.w;
      const double3 canCentroid = make_double3(
          double(work.seedMomentCenter.x) + work.fullCell.x * invW,
          double(work.seedMomentCenter.y) + work.fullCell.y * invW,
          double(work.seedMomentCenter.z) + work.fullCell.z * invW);
      if (isfinite(canCentroid.x) && isfinite(canCentroid.y) && isfinite(canCentroid.z)) {
        samplePoint = cudaGeometry::canonicalToWorld(
            canCentroid, input.domainMin, input.invBaseScale);
      }
    }

    const uint32_t classification = cudaGeometry::classifyPoint(
        input.winding.triangles,
        input.winding.nodes,
        input.winding.root,
        samplePoint);
    if (classification == cudaGeometry::PointOnSurface ||
        classification == cudaGeometry::PointUnresolved) {
      if (output.packedError != nullptr &&
          rvdReportErrorOnce(
              output.packedError, uint32_t(RVDRestrictionError::WindingOnSurface),
              seedId)) {
        printf("[RVDRestriction] winding on surface cell=%u classification=%u "
               "sample=(%g %g %g)\n",
               seedId, classification, samplePoint.x, samplePoint.y,
               samplePoint.z);
      }
      failCell(seedId, RVDStatus::InconsistentBoundary, output);
      return false;
    }

    const int inside = classification == cudaGeometry::PointInside ? 1 : 0;

    if (!inside) {
      return writeEmptyRestrictedCell(seedId, output);
    }
    return writeFullBaseCell(seedId, input, output, work);
  }

  const double volumeErrorBudget = 256.0 * DBL_EPSILON * work.absoluteVolumeSum;
  const double fullCellVolume = fabs(work.fullCell.w / 6.0);
  // filter negative cancellation residue below facet floor
  const double negativeClosureErrorBudget = volumeErrorBudget + 256.0 * DBL_EPSILON * fullCellVolume;

  if (!isfinite(work.moment.w)) {
    failCell(seedId, RVDStatus::NonFiniteGeometry, output);
    return false;
  }
  if (work.moment.w < -negativeClosureErrorBudget) {
    failCell(seedId, RVDStatus::NegativeVolume, output);
    return false;
  }

  if (work.moment.w <= 0.0) {
    zeroCellCounts(seedId, output, work.basePlaneCount);
    failCell(seedId, RVDStatus::EmptyCell, output);
    return false;
  }

  if (!writeRestrictedPlaneMeasures(seedId, input, output, work)) return false;

  if (!writeIntegratedCell(
          seedId, work.cell, output, work.moment, input.baseGrid.scale)) {
    failCell(seedId, work.cell.status == RVDStatus::Success
        ? RVDStatus::NonFiniteGeometry
        : work.cell.status, output);
    return false;
  }
  return true;
}

__device__ void processRestrictedCell(
    uint32_t seedId,
    const RestrictedDeviceInput &input,
    RestrictedDeviceOutput output,
    bool surfaceOnly,
    RestrictedCellStorage &storage) {
  if (!initializeOutput(seedId, output, surfaceOnly)) return;

  RestrictedCellWork work;
  work.seedId = seedId;
  work.surfaceOnly = surfaceOnly;

  if (!prepareCell(seedId, input, output, storage, work)) return;
  if (!collectUniqueTriangles(seedId, input, output, storage, work)) return;
  if (!surfaceOnly && !integrateWholeCell(seedId, output, storage, work)) {
    return;
  }

  for (int u = 0; u < work.uniqueCount; ++u) {
    if (!processTriangle(seedId, storage.uniqueTriangles[u], input, output, surfaceOnly, storage, work)) {
      return;
    }
  }

  if (surfaceOnly) return;
  finalizeRestrictedCell(seedId, input, output, work);
}

__global__ void restrictedCellKernel(
    RestrictedDeviceInput input,
    RestrictedDeviceOutput output,
    uint32_t begin,
    uint32_t count,
    RestrictedCellStorage *storageMemory,
    bool surfaceOnly) {
  const uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local >= count) return;

  processRestrictedCell(begin + local, input, output, surfaceOnly, storageMemory[local]);
}

bool launchRestrictionPass(
    const RestrictedDeviceInput &input,
    RestrictedDeviceOutput output,
    uint32_t count,
    uint32_t chunkSize,
    RestrictedCellStorage *storage,
    cudaStream_t stream,
    bool surfaceOnly) {
  if (count == 0) return true;

  const uint32_t chunk = std::max(1u, chunkSize);
  for (uint32_t begin = 0; begin < count; begin += chunk) {
    const uint32_t chunkCount = std::min(chunk, count - begin);
    restrictedCellKernel<<<
        rvdHost::blockCount(chunkCount, RestrictionBlockSize),
        RestrictionBlockSize, 0, stream>>>(
        input, output, begin, chunkCount, storage, surfaceOnly);

    if (!rvdHost::checkLaunch("launch restricted kernel")) return false;
  }
  return rvdHost::synchronize(stream, "synchronize restriction pass");
}

struct RestrictionPassBuffers {
  cudaUtils::CudaBuffer<double4> moments;
  cudaUtils::CudaBuffer<uint32_t> planarPatchCounts;
  cudaUtils::CudaBuffer<uint32_t> planarVertexCounts;
  cudaUtils::CudaBuffer<uint32_t> planarPatchOffsets;
  cudaUtils::CudaBuffer<uint32_t> planarVertexOffsets;
  cudaUtils::CudaBuffer<double> restrictedPlaneAreas;
  cudaUtils::CudaBuffer<uint32_t> faceSegmentCounts;
  cudaUtils::CudaBuffer<uint32_t> faceSegmentOffsets;
  cudaUtils::CudaBuffer<RVDStatus> statuses;
  cudaUtils::CudaBuffer<RestrictedCellStorage> storage;
  cudaUtils::CudaBuffer<RestrictedSurface> surfaces;
  cudaUtils::CudaBuffer<double3> surfaceNormals;
  cudaUtils::CudaBuffer<double3> restrictedVertices;
  cudaUtils::CudaBuffer<FaceBoundarySegment> faceSegments;
};

struct RestrictionTotals {
  uint32_t patches = 0;
  uint32_t vertices = 0;
  uint32_t faceSegments = 0;
};

bool allocatePassBuffers(
    uint32_t count,
    uint32_t totalPlanes,
    uint32_t storageCount,
    RestrictionPassBuffers &buffers) {
  if (!buffers.moments.allocate(count) ||
      !buffers.planarPatchCounts.allocate(count) ||
      !buffers.planarVertexCounts.allocate(count) ||
      !buffers.planarPatchOffsets.allocate(count) ||
      !buffers.planarVertexOffsets.allocate(count) ||
      !buffers.statuses.allocate(count) ||
      !buffers.storage.allocate(storageCount)) {
    return false;
  }

  if (totalPlanes == 0) return true;
  return buffers.restrictedPlaneAreas.allocate(totalPlanes) &&
         buffers.faceSegmentCounts.allocate(size_t(totalPlanes) + 1) &&
         buffers.faceSegmentOffsets.allocate(size_t(totalPlanes) + 1);
}

bool clearPassBuffers(
    RestrictionPassBuffers &buffers,
    uint32_t count,
    uint32_t totalPlanes,
    cudaStream_t stream) {
  if (!rvdHost::zero(
          buffers.moments.get(), count * sizeof(double4), stream, "clear moments")) {
    return false;
  }
  if (totalPlanes == 0) return true;

  return rvdHost::zero(
             buffers.restrictedPlaneAreas.get(),
             totalPlanes * sizeof(double), stream, "clear restricted plane areas") &&
         rvdHost::zero(
             buffers.faceSegmentCounts.get(),
             (size_t(totalPlanes) + 1) * sizeof(uint32_t), stream,
             "clear face segment counts");
}

RestrictedDeviceInput makeDeviceInput(
    const RVDInput &input,
    const cudaGeometry::DeviceModelGeometry &devGeom,
    const BaseCellDomain &baseDomain,
    cudaGeometry::DeviceVoxelGrid voxelLookup) {
  RestrictedDeviceInput deviceInput{};
  deviceInput.knn = input.seedSpatialIndex->deviceKnn();
  deviceInput.triangles = devGeom.triangles.get();
  deviceInput.triangleVertexIds = devGeom.triangleVertexIds.get();
  deviceInput.triangleCount = uint32_t(devGeom.triangles.size());
  deviceInput.triangleList = devGeom.voxelTriangleIds.get();
  deviceInput.triangleListCount = uint32_t(devGeom.voxelTriangleIds.size());
  deviceInput.offsets = devGeom.voxelOffsets.get();
  deviceInput.offsetCount = uint32_t(devGeom.voxelOffsets.size());
  deviceInput.baseGrid = baseDomain.baseGrid;
  deviceInput.modelVoxelGrid = voxelLookup;
  deviceInput.winding = devGeom.view();
  deviceInput.domainMin = make_double3(
      double(baseDomain.domainMin.x),
      double(baseDomain.domainMin.y),
      double(baseDomain.domainMin.z));
  deviceInput.invBaseScale = baseDomain.invBaseScale;
  deviceInput.modelBoundsMin = cudaGeometry::canonicalFromWorld(baseDomain.baseGrid, devGeom.boundsMin);
  deviceInput.modelBoundsMax = cudaGeometry::canonicalFromWorld(baseDomain.baseGrid, devGeom.boundsMax);
  deviceInput.baseCells = baseDomain.deviceView();
  return deviceInput;
}

RestrictedDeviceOutput makeCountPassOutput(
    const BaseCellDomain &baseDomain,
    RestrictionPassBuffers &buffers,
    RvdPackedError *packedError) {
  RestrictedDeviceOutput output{};
  output.moments = buffers.moments.get();
  output.planarPatchCounts = buffers.planarPatchCounts.get();
  output.planarVertexCounts = buffers.planarVertexCounts.get();
  output.restrictedPlaneOffsets = baseDomain.devCellPlaneOffsets.get();
  output.restrictedPlaneAreas = buffers.restrictedPlaneAreas.get();
  output.faceSegmentCounts = buffers.faceSegmentCounts.get();
  output.packedError = packedError;
  output.statuses = buffers.statuses.get();
  return output;
}

bool scanRestrictionCounts(
    const BaseCellDomain &baseDomain,
    RestrictionPassBuffers &buffers,
    RestrictionTotals &totals,
    cudaStream_t stream) {
  const uint32_t count = baseDomain.count;
  const uint32_t totalPlanes = baseDomain.totalPlanes;

  cudaUtils::CudaBuffer<uint8_t> scanScratch;
  if (!rvdHost::exclusiveSum(
          buffers.planarPatchCounts.get(),
          buffers.planarPatchOffsets.get(),
          int(count),
          scanScratch,
          stream) ||
      !rvdHost::exclusiveSum(
          buffers.planarVertexCounts.get(),
          buffers.planarVertexOffsets.get(),
          int(count),
          scanScratch,
          stream)) {
    return false;
  }

  if (totalPlanes > 0 &&
      !rvdHost::exclusiveSum(
          buffers.faceSegmentCounts.get(),
          buffers.faceSegmentOffsets.get(),
          int(totalPlanes + 1),
          scanScratch,
          stream)) {
    return false;
  }

  if (!rvdHost::readLastExclusiveScanTotal(
          buffers.planarPatchCounts.get(),
          buffers.planarPatchOffsets.get(),
          count,
          stream,
          totals.patches,
          "read restricted patch count") ||
      !rvdHost::readLastExclusiveScanTotal(
          buffers.planarVertexCounts.get(),
          buffers.planarVertexOffsets.get(),
          count,
          stream,
          totals.vertices,
          "read restricted vertex count")) {
    return false;
  }

  if (totalPlanes > 0 &&
      !rvdHost::readAt(
          totals.faceSegments,
          buffers.faceSegmentOffsets.get(),
          totalPlanes,
          stream,
          "read face segment count")) {
    return false;
  }
  return true;
}

bool allocateSurfaceBuffers(const RestrictionTotals &totals, RestrictionPassBuffers &buffers) {
  if (totals.patches > 0 &&
      (!buffers.surfaces.allocate(totals.patches) ||
       !buffers.surfaceNormals.allocate(totals.patches))) {
    return false;
  }
  if (totals.vertices > 0 && !buffers.restrictedVertices.allocate(totals.vertices)) {
    return false;
  }
  if (totals.faceSegments > 0 && !buffers.faceSegments.allocate(totals.faceSegments)) {
    return false;
  }
  return true;
}

RestrictedDeviceOutput makeWritePassOutput(
    const BaseCellDomain &baseDomain,
    RestrictionPassBuffers &buffers,
    RvdPackedError *packedError) {
  RestrictedDeviceOutput output{};
  output.moments = buffers.moments.get();
  output.planarPatchCounts = buffers.planarPatchCounts.get();
  output.planarVertexCounts = buffers.planarVertexCounts.get();
  output.planarPatchOffsets = buffers.planarPatchOffsets.get();
  output.planarVertexOffsets = buffers.planarVertexOffsets.get();
  output.outSurfaces = buffers.surfaces.get();
  output.outSurfaceNormals = buffers.surfaceNormals.get();
  output.outRestrictedVertices = buffers.restrictedVertices.get();
  output.faceSegmentOffsets = buffers.faceSegmentOffsets.get();
  output.outFaceSegments = buffers.faceSegments.get();
  output.restrictedPlaneOffsets = baseDomain.devCellPlaneOffsets.get();
  output.restrictedPlaneAreas = buffers.restrictedPlaneAreas.get();
  output.packedError = packedError;
  output.statuses = buffers.statuses.get();
  return output;
}

void moveRestrictionOutputs(RestrictionPassBuffers &buffers, const RestrictionTotals &totals, ModelRestriction &restriction) {
  restriction.devMoments = std::move(buffers.moments);
  restriction.devRestrictedPlaneAreas = std::move(buffers.restrictedPlaneAreas);
  restriction.devSurfaces = std::move(buffers.surfaces);
  restriction.devSurfaceNormals = std::move(buffers.surfaceNormals);
  restriction.devRestrictedVertices = std::move(buffers.restrictedVertices);
  restriction.devFaceSegmentOffsets = std::move(buffers.faceSegmentOffsets);
  restriction.devFaceSegments = std::move(buffers.faceSegments);
  restriction.totalPatches = totals.patches;
}

} // namespace

bool restrictModelGeometry(
    const RVDInput &input,
    const RVDConfig &config,
    const cudaGeometry::DeviceModelGeometry &devGeom,
    const BaseCellDomain &baseDomain,
    ModelRestriction &restriction,
    cudaStream_t stream) {
  if (!devGeom.valid()) return false;

  restriction.runtimeModelId = devGeom.runtimeModelId;

  const cudaGeometry::DeviceVoxelGrid &modelGrid = devGeom.voxelGrid;
  const float modelVoxelSize = modelGrid.voxelSize / modelGrid.scale;

  cudaGeometry::DeviceVoxelGrid voxelLookup{};
  voxelLookup.gridMin = cudaGeometry::canonicalFromWorld(baseDomain.baseGrid, modelGrid.gridMin);
  voxelLookup.voxelSize = modelVoxelSize * baseDomain.baseScale;
  voxelLookup.gridDim = modelGrid.gridDim;

  const uint32_t count = baseDomain.count;
  const uint32_t totalPlanes = baseDomain.totalPlanes;
  const uint32_t chunkSize = std::max(1u, config.baseChunkSize);
  const uint32_t storageCount = std::min(count, chunkSize);

  RestrictionPassBuffers buffers;
  if (!allocatePassBuffers(count, totalPlanes, storageCount, buffers) ||
      !clearPassBuffers(buffers, count, totalPlanes, stream)) {
    return false;
  }

  cudaUtils::CudaBuffer<uint8_t> occupancy;
  if (!::cudaGeometry::buildVoxelOccupancy(
          devGeom.triangles.get(), uint32_t(devGeom.triangles.size()),
          devGeom.voxelGrid, devGeom.voxelOffsets.get(),
          devGeom.voxelTriangleIds.get(),
          uint32_t(devGeom.voxelTriangleIds.size()), occupancy, stream)) {
    std::cerr << "[RVD] occupancy build failed modelId="
              << restriction.runtimeModelId << std::endl;
    return false;
  }

  RestrictedDeviceInput deviceInput =
      makeDeviceInput(input, devGeom, baseDomain, voxelLookup);
  deviceInput.occupancy = occupancy.get();
  deviceInput.occupancyCount = uint32_t(occupancy.size());

  cudaUtils::CudaBuffer<RvdPackedError> packedError;
  if (!packedError.allocate(1) ||
      !rvdHost::zero(
          packedError.get(), sizeof(RvdPackedError), stream,
          "clear restriction error")) {
    return false;
  }
  const RestrictedDeviceOutput countOutput =
      makeCountPassOutput(baseDomain, buffers, packedError.get());

  if (!launchRestrictionPass(deviceInput, countOutput, count, chunkSize, buffers.storage.get(), stream, false)) {
    std::cerr << "[RVD] restriction count pass failed modelId=" << restriction.runtimeModelId << std::endl;
    return false;
  }

  RvdPackedError siteError = 0;
  if (!rvdHost::readPackedError(packedError, stream, siteError)) return false;
  if (siteError != 0) {
    const uint32_t code = rvdErrorCode(siteError);
    const uint32_t cell = rvdErrorId(siteError);
    std::cerr << "[RVD] restriction count pass site failed cell=" << cell
              << " modelId=" << restriction.runtimeModelId
              << " code=" << code
              << " (" << restrictionFailureName(code) << ")" << std::endl;
    return false;
  }

  RestrictionTotals totals;
  if (!scanRestrictionCounts(baseDomain, buffers, totals, stream)) return false;
  if (!allocateSurfaceBuffers(totals, buffers)) return false;

  const RestrictedDeviceOutput writeOutput =
      makeWritePassOutput(baseDomain, buffers, packedError.get());
  if (!launchRestrictionPass(deviceInput, writeOutput, count, chunkSize, buffers.storage.get(), stream, true)) {
    std::cerr << "[RVD] restriction write pass failed modelId=" << restriction.runtimeModelId << std::endl;
    return false;
  }

  RvdPackedError writeError = 0;
  if (!rvdHost::readPackedError(packedError, stream, writeError)) return false;
  if (writeError != 0) {
    const uint32_t code = rvdErrorCode(writeError);
    const uint32_t cell = rvdErrorId(writeError);
    std::cerr << "[RVD] restriction write pass failed cell=" << cell
              << " modelId=" << restriction.runtimeModelId
              << " code=" << code
              << " (" << restrictionFailureName(code) << ")" << std::endl;
    return false;
  }

  moveRestrictionOutputs(buffers, totals, restriction);
  return true;
}

} // namespace voronoi
