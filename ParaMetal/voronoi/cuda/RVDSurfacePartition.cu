#include "RVDSurfacePartition.hpp"
#include "RVDHostUtils.cuh"
#include "RVDModelGeometry.cuh"
#include "../../cuda/CudaMath.cuh"
#include "../../cuda/CudaBuffer.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <vector>

namespace voronoi {

static constexpr uint32_t SurfaceBlockSize = 128;
static constexpr uint32_t SurfacePolygonBlockSize = 32;
static constexpr uint32_t SurfacePatchCapacity = 48;
static constexpr uint32_t MaxPatchCuts = 8;
static constexpr uint32_t MaxPatchPartners = 8;
static constexpr uint32_t SurfaceClipCapacity = SurfacePatchCapacity + MaxPatchCuts + 1;
static constexpr double SurfaceInfinity = std::numeric_limits<double>::infinity();

enum SurfacePartitionError : uint32_t {
  SurfaceInvalidPatch = 1,
  SurfaceInvalidPhaseVolume = 2,
  SurfaceInvalidNormal = 3,
  SurfacePolygonOverflow = 4,
  SurfaceDegenerateFrame = 5,
  SurfaceNonPositiveArea = 6,
  SurfaceMissingSourceTriangle = 7,
  SurfaceCutOverflow = 8,
  SurfacePartnerOverflow = 9,
  SurfaceSplitFailure = 10,
  SurfacePartitionFailure = 11,
  SurfaceConservation = 12,
};

enum SideFailure : uint32_t {
  SideSuccess = 0,
  SideInvalidOwner = 1,
  SideDegenerateRay = 2,
  SideOwnerNotDistinct = 3,
  SideOwnerOverlap = 4,
  SideMultipleAdjacent = 5,
};

struct PatchClassification {
  uint32_t adjacentModelIndex = InvalidId;
  uint32_t valid = 1;
  uint32_t flipNormal = 0;
  uint32_t failure = SideSuccess;
  uint32_t offendingModel = InvalidId;
  double windingMinus = 0.0;
  double windingPlus = 0.0;
};

struct SurfaceModelView {
  const RestrictedSurface *surfaces;
  const double3 *normals;
  const double4 *moments;
  const double3 *vertices;
  uint32_t patchCount;
  uint32_t vertexCount;
  uint32_t runtimeModelId;
};

struct PreparedSurfacePatch {
  double3 origin;
  double3 u;
  double3 v;
  double3 normal;
  double2 centroid;
  double area;
  uint32_t cellId;
  uint32_t modelIndex;
  uint32_t runtimeModelId;
  uint32_t sourceTriangleId;
  uint32_t firstVertex;
  uint32_t vertexCount;
  uint32_t reversed;
  uint32_t active;
};

struct SurfaceCut {
  double3 point;
  double3 normal;
};

struct SurfacePiece {
  double2 centroid;
  double area;
  uint32_t patchId;
  uint32_t firstVertex;
  uint32_t vertexCount;
  uint32_t contactFacing;
  uint32_t original;
};

struct SurfaceRegionOutput {
  double3 moment;
  double3 normal;
  double area;
  uint32_t cellId;
  uint32_t modelIndex;
  uint32_t runtimeModelId;
  uint32_t sourceTriangleId;
  uint32_t outsideMaterialId;
  uint32_t firstVertex;
  uint32_t vertexCount;
  uint32_t contactFacing;
};

struct SurfaceConservationReport {
  double expected;
  double actual;
  double difference;
  double tolerance;
};

struct SurfacePartitionResult {
  uint32_t insideCount;
  uint32_t insideVertices;
  uint32_t outsideCount;
  uint32_t outsideVertices;
  double insideArea;
  double2 insideCentroid;
  bool overflow;
};

struct SurfaceCutResult {
  uint32_t positiveCount;
  uint32_t negativeCount;
  double positiveArea;
  double negativeArea;
  double2 positiveCentroid;
  double2 negativeCentroid;
  bool split;
  bool overflow;
};

__device__ inline double surfaceCross2(double2 a, double2 b) {
  return a.x * b.y - a.y * b.x;
}

__device__ inline double surfaceLength2(double2 a) {
  return sqrt(a.x * a.x + a.y * a.y);
}

template <typename T>
__device__ inline void surfaceReverse(T *p, uint32_t n) {
  for (uint32_t i = 0; i < n / 2; ++i) {
    T tmp = p[i]; p[i] = p[n - 1 - i]; p[n - 1 - i] = tmp;
  }
}

__device__ inline double3 surfaceWorldVertex(
    const SurfaceModelView &view, const PreparedSurfacePatch &patch,
    uint32_t vertex, double3 domainMin, double invBaseScale) {
  return cudaMath::add3(domainMin, cudaMath::mul3(view.vertices[patch.firstVertex + vertex], invBaseScale));
}

__device__ inline void surfaceProjectWorldPolygon(
    const SurfaceModelView &view, const PreparedSurfacePatch &patch,
    double3 origin, double3 u, double3 v, double3 domainMin, double invBaseScale, double2 *out) {
  for (uint32_t i = 0; i < patch.vertexCount; ++i) {
    const double3 delta = cudaMath::sub3(surfaceWorldVertex(view, patch, i, domainMin, invBaseScale), origin);
    out[i] = make_double2(cudaMath::dot3(delta, u), cudaMath::dot3(delta, v));
  }
}

__device__ inline void surfaceMaterializePatch(
    const SurfaceModelView &view, const PreparedSurfacePatch &patch,
    double3 domainMin, double invBaseScale, double2 *out) {
  surfaceProjectWorldPolygon(view, patch, patch.origin, patch.u, patch.v, domainMin, invBaseScale, out);
  if (patch.reversed != 0) surfaceReverse(out, patch.vertexCount);
}

__device__ inline double3 surfaceLocalToWorld(const PreparedSurfacePatch &patch, double2 pt) {
  return cudaMath::add3(
      cudaMath::add3(patch.origin, cudaMath::mul3(patch.u, pt.x)),
      cudaMath::mul3(patch.v, pt.y));
}

__device__ inline uint32_t surfaceClipPolygon(
    const double2 *input, uint32_t inputCount, double a, double b, double c, double2 *output, bool &overflow) {
  overflow = false;
  if (inputCount == 0) return 0;
  uint32_t count = 0;
  double2 prev = input[inputCount - 1];
  double prevDist = a * prev.x + b * prev.y + c;
  for (uint32_t i = 0; i < inputCount; ++i) {
    const double2 curr = input[i];
    const double currDist = a * curr.x + b * curr.y + c;
    if ((prevDist < 0.0 && currDist > 0.0) || (prevDist > 0.0 && currDist < 0.0)) {
      const double t = prevDist / (prevDist - currDist);
      const double2 hit = make_double2(prev.x + t * (curr.x - prev.x), prev.y + t * (curr.y - prev.y));
      if (count == 0 || hit.x != output[count - 1].x || hit.y != output[count - 1].y) {
        if (count < SurfaceClipCapacity) output[count++] = hit;
        else { overflow = true; return 0; }
      }
    }
    if (currDist >= 0.0) {
      if (count == 0 || curr.x != output[count - 1].x || curr.y != output[count - 1].y) {
        if (count < SurfaceClipCapacity) output[count++] = curr;
        else { overflow = true; return 0; }
      }
    }
    prev = curr;
    prevDist = currDist;
  }
  if (count > 1 && output[0].x == output[count - 1].x && output[0].y == output[count - 1].y) --count;
  return count;
}

__device__ inline double surfacePolygonArea(const double2 *poly, uint32_t count) {
  double twiceArea = 0.0;
  for (uint32_t i = 0; i < count; ++i) twiceArea += surfaceCross2(poly[i], poly[(i + 1) % count]);
  return 0.5 * fabs(twiceArea);
}

__device__ inline bool surfaceMeasurePolygon(const double2 *poly, uint32_t count, double &area, double2 &centroid) {
  area = 0.0;
  centroid = make_double2(0.0, 0.0);
  if (count < 3) return true;
  const double2 origin = poly[0];
  double twiceArea = 0.0, absSum = 0.0;
  double2 num = make_double2(0.0, 0.0);
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t next = (i + 1) % count;
    const double2 a = make_double2(poly[i].x - origin.x, poly[i].y - origin.y);
    const double2 b = make_double2(poly[next].x - origin.x, poly[next].y - origin.y);
    const double term = surfaceCross2(a, b);
    twiceArea += term;
    absSum += fabs(term);
    num.x += (a.x + b.x) * term;
    num.y += (a.y + b.y) * term;
  }
  const double roundoff = 256.0 * DBL_EPSILON * absSum;
  area = 0.5 * fabs(twiceArea);
  if (!isfinite(area) || !isfinite(roundoff)) return false;
  if (area <= roundoff) { area = 0.0; return true; }
  centroid = make_double2(origin.x + num.x / (3.0 * twiceArea), origin.y + num.y / (3.0 * twiceArea));
  return isfinite(centroid.x) && isfinite(centroid.y);
}

__device__ inline bool surfaceLineInterval(
    const double2 *poly, uint32_t count, const PreparedSurfacePatch &frame,
    double3 linePoint, double3 lineDir, double tol, double &minBound, double &maxBound) {
  const double3 delta = cudaMath::sub3(linePoint, frame.origin);
  const double2 pt = make_double2(cudaMath::dot3(delta, frame.u), cudaMath::dot3(delta, frame.v));
  const double2 dir = make_double2(cudaMath::dot3(lineDir, frame.u), cudaMath::dot3(lineDir, frame.v));
  minBound = -SurfaceInfinity;
  maxBound = SurfaceInfinity;
  for (uint32_t i = 0; i < count; ++i) {
    const double2 a = poly[i];
    const double2 b = poly[(i + 1) % count];
    const double2 edge = make_double2(b.x - a.x, b.y - a.y);
    const double2 offset = make_double2(pt.x - a.x, pt.y - a.y);
    const double constant = surfaceCross2(edge, offset);
    const double slope = surfaceCross2(edge, dir);
    const double edgeLen = surfaceLength2(edge);
    const double edgeTol = tol * edgeLen;
    if (fabs(slope) <= 64.0 * DBL_EPSILON * edgeLen) {
      if (constant < -edgeTol) return false;
      continue;
    }
    const double bound = (-edgeTol - constant) / slope;
    if (slope > 0.0) minBound = fmax(minBound, bound);
    else maxBound = fmin(maxBound, bound);
    if (minBound > maxBound) return false;
  }
  return true;
}

__device__ inline int surfaceIntersectInArea(
    const SurfaceModelView *views, const PreparedSurfacePatch &patchA, double2 *polyA,
    const PreparedSurfacePatch &patchB, double2 *polyB,
    double3 domainMin, double invBaseScale, double posTol, int &coplanar) {
  coplanar = 0;
  const double3 crossNormals = cudaMath::cross3(patchA.normal, patchB.normal);
  const double dirLen = cudaMath::length3(crossNormals);
  const double planeDist = fabs(cudaMath::dot3(patchA.normal, cudaMath::sub3(patchB.origin, patchA.origin)));
  if (dirLen <= 512.0 * DBL_EPSILON) {
    const double nDot = cudaMath::dot3(patchA.normal, patchB.normal);
    if (nDot > -1.0 + 512.0 * DBL_EPSILON || planeDist > posTol) return 0;
    const double frameArea = surfacePolygonArea(polyA, patchA.vertexCount);
    surfaceProjectWorldPolygon(views[patchB.modelIndex], patchB, patchA.origin, patchA.u, patchA.v, domainMin, invBaseScale, polyB);
    double signedArea = 0.0;
    for (uint32_t i = 0; i < patchB.vertexCount; ++i) signedArea += surfaceCross2(polyB[i], polyB[(i + 1) % patchB.vertexCount]);
    if (signedArea < 0.0) surfaceReverse(polyB, patchB.vertexCount);

    double2 scratchA[SurfaceClipCapacity], scratchB[SurfaceClipCapacity];
    uint32_t overlapCount = patchA.vertexCount;
    for (uint32_t i = 0; i < overlapCount; ++i) scratchA[i] = polyA[i];
    double2 *curr = scratchA, *next = scratchB;
    for (uint32_t edge = 0; edge < patchB.vertexCount; ++edge) {
      const double2 p0 = polyB[edge], p1 = polyB[(edge + 1) % patchB.vertexCount];
      bool clipOverflow = false;
      overlapCount = surfaceClipPolygon(curr, overlapCount, p0.y - p1.y, p1.x - p0.x, p1.y * p0.x - p1.x * p0.y, next, clipOverflow);
      if (clipOverflow) return -1;
      double2 *tmp = curr; curr = next; next = tmp;
      if (overlapCount < 3) return 0;
    }
    coplanar = surfacePolygonArea(curr, overlapCount) > 256.0 * DBL_EPSILON * frameArea;
    return coplanar ? 1 : 0;
  }

  const double3 dir = make_double3(crossNormals.x / dirLen, crossNormals.y / dirLen, crossNormals.z / dirLen);
  const double offA = cudaMath::dot3(patchA.normal, patchA.origin);
  const double offB = cudaMath::dot3(patchB.normal, patchB.origin);
  const double3 num = cudaMath::add3(cudaMath::mul3(cudaMath::cross3(patchB.normal, crossNormals), offA),
                                     cudaMath::mul3(cudaMath::cross3(crossNormals, patchA.normal), offB));
  const double denom = dirLen * dirLen;
  const double3 linePt = make_double3(num.x / denom, num.y / denom, num.z / denom);
  double minA, maxA, minB, maxB;
  if (!surfaceLineInterval(polyA, patchA.vertexCount, patchA, linePt, dir, posTol, minA, maxA) ||
      !surfaceLineInterval(polyB, patchB.vertexCount, patchB, linePt, dir, posTol, minB, maxB)) {
    return 0;
  }
  return (fmin(maxA, maxB) - fmax(minA, minB)) > posTol ? 1 : 0;
}

__device__ inline bool surfaceApplyCut(
    const double2 *poly, uint32_t count, const SurfaceCut &cut, const PreparedSurfacePatch &patch,
    double2 *pos, double2 *neg, SurfaceCutResult &result) {
  result = SurfaceCutResult{};
  const double a = cudaMath::dot3(cut.normal, patch.u);
  const double b = cudaMath::dot3(cut.normal, patch.v);
  const double c = cudaMath::dot3(cut.normal, cudaMath::sub3(patch.origin, cut.point));
  bool clipOverflow = false;
  result.positiveCount = surfaceClipPolygon(poly, count, a, b, c, pos, clipOverflow);
  if (clipOverflow) { result.overflow = true; return false; }
  result.negativeCount = surfaceClipPolygon(poly, count, -a, -b, -c, neg, clipOverflow);
  if (clipOverflow) { result.overflow = true; return false; }
  if (!surfaceMeasurePolygon(pos, result.positiveCount, result.positiveArea, result.positiveCentroid) ||
      !surfaceMeasurePolygon(neg, result.negativeCount, result.negativeArea, result.negativeCentroid)) {
    return false;
  }
  result.split = !(result.positiveArea == 0.0 || result.negativeArea == 0.0);
  return true;
}

__device__ inline bool surfacePartitionPiece(
    const SurfaceModelView *views, const PreparedSurfacePatch &patch,
    const double2 *source, uint32_t sourceCount, double sourceArea,
    const PreparedSurfacePatch &opposing, double3 domainMin, double invBaseScale,
    double2 *curr, double2 *next, double2 *projected, double2 *outside,
    double2 *insideOut, double2 *outsideOut, uint32_t *outsideCounts,
    SurfacePartitionResult &result) {
  result = SurfacePartitionResult{};
  if (sourceCount < 3 || opposing.vertexCount < 3) return false;

  surfaceProjectWorldPolygon(views[opposing.modelIndex], opposing, patch.origin, patch.u, patch.v,
                             domainMin, invBaseScale, projected);
  double signedAreaOpp = 0.0;
  for (uint32_t i = 0; i < opposing.vertexCount; ++i)
    signedAreaOpp += surfaceCross2(projected[i], projected[(i + 1) % opposing.vertexCount]);
  if (!isfinite(signedAreaOpp) || signedAreaOpp == 0.0) return false;
  if (signedAreaOpp < 0.0) surfaceReverse(projected, opposing.vertexCount);

  for (uint32_t i = 0; i < sourceCount; ++i) curr[i] = source[i];
  double signedAreaSrc = 0.0;
  for (uint32_t i = 0; i < sourceCount; ++i)
    signedAreaSrc += surfaceCross2(curr[i], curr[(i + 1) % sourceCount]);
  if (!isfinite(signedAreaSrc) || signedAreaSrc == 0.0) return false;
  if (signedAreaSrc < 0.0) surfaceReverse(curr, sourceCount);

  uint32_t currCount = sourceCount;
  double outsideAreaSum = 0.0;
  for (uint32_t edge = 0; edge < opposing.vertexCount; ++edge) {
    if (currCount < 3) break;
    const double2 p0 = projected[edge], p1 = projected[(edge + 1) % opposing.vertexCount];
    const double a = p0.y - p1.y, b = p1.x - p0.x, c = p1.y * p0.x - p1.x * p0.y;

    bool clipOverflow = false;
    const uint32_t outCount = surfaceClipPolygon(curr, currCount, -a, -b, -c, outside, clipOverflow);
    if (clipOverflow) { result.overflow = true; return false; }
    double outArea = 0.0;
    double2 outCentroid;
    if (!surfaceMeasurePolygon(outside, outCount, outArea, outCentroid)) return false;
    if (outCount >= 3 && outArea > 0.0) {
      if (outsideOut) {
        for (uint32_t v = 0; v < outCount; ++v) outsideOut[result.outsideVertices + v] = outside[v];
        outsideCounts[result.outsideCount] = outCount;
      }
      ++result.outsideCount;
      result.outsideVertices += outCount;
      outsideAreaSum += outArea;
    }

    currCount = surfaceClipPolygon(curr, currCount, a, b, c, next, clipOverflow);
    if (clipOverflow) { result.overflow = true; return false; }
    double2 *tmp = curr; curr = next; next = tmp;
  }

  double overlapArea = 0.0;
  double2 overlapCentroid = make_double2(0.0, 0.0);
  if (currCount >= 3 && !surfaceMeasurePolygon(curr, currCount, overlapArea, overlapCentroid)) return false;

  if (!(overlapArea > 0.0)) {
    if (outsideOut) {
      for (uint32_t v = 0; v < sourceCount; ++v) outsideOut[v] = source[v];
      outsideCounts[0] = sourceCount;
    }
    result.outsideCount = 1;
    // strips written before this reset can exceed the source polygon
    if (result.outsideVertices < sourceCount) result.outsideVertices = sourceCount;
    return true;
  }

  if (insideOut) {
    for (uint32_t v = 0; v < currCount; ++v) insideOut[v] = curr[v];
  }
  result.insideCount = 1;
  result.insideVertices = currCount;
  result.insideArea = overlapArea;
  result.insideCentroid = overlapCentroid;

  const double totalArea = overlapArea + outsideAreaSum;
  const double tol = 512.0 * DBL_EPSILON * sourceArea * double(opposing.vertexCount + 1);
  return fabs(totalArea - sourceArea) <= tol;
}

__device__ inline bool pointInsideModelBounds(const cudaGeometry::DeviceModelGeometryView &view, double3 p) {
  return view.nodes && p.x >= double(view.boundsMin.x) && p.x <= double(view.boundsMax.x) &&
                       p.y >= double(view.boundsMin.y) && p.y <= double(view.boundsMax.y) &&
                       p.z >= double(view.boundsMin.z) && p.z <= double(view.boundsMax.z);
}

__device__ inline bool sideWinding(
    const cudaGeometry::DeviceModelGeometryView &view, double3 pt, double3 normal, double &winding) {
  winding = 0.0;
  if (!view.nodes || !view.triangles || view.root == cudaGeometry::InvalidNodeId) return true;
  const cudaGeometry::SideWinding side = cudaGeometry::computeSideWinding(
      view.triangles, view.nodes, view.root, pt, normal, cudaGeometry::InvalidNodeId);
  winding = side.value;
  return !side.ambiguous;
}

__global__ void prepareSurfacePatchesKernel(
    const SurfaceModelView *views, uint32_t modelCount, uint32_t patchCount, uint32_t cellCount,
    double3 domainMin, double invBaseScale, PreparedSurfacePatch *prepared, uint32_t *activeFlags,
    uint32_t *cellKeys, uint32_t *cellValues, uint32_t *cellCounts, uint32_t *pieceVertexCounts,
    RvdPackedError *packedError) {
  const uint32_t patchId = blockIdx.x * blockDim.x + threadIdx.x;
  if (patchId >= patchCount) return;
  prepared[patchId] = PreparedSurfacePatch{};
  activeFlags[patchId] = 0;
  cellKeys[patchId] = cellCount;
  cellValues[patchId] = patchId;
  pieceVertexCounts[patchId] = 0;

  uint32_t modelIndex = 0, remaining = patchId;
  while (modelIndex < modelCount) {
    const uint32_t count = views[modelIndex].patchCount;
    if (remaining < count) break;
    remaining -= count;
    ++modelIndex;
  }
  if (modelIndex >= modelCount) {
    rvdReportError(packedError, SurfaceInvalidPatch, patchId);
    return;
  }
  const SurfaceModelView &view = views[modelIndex];
  const RestrictedSurface surface = view.surfaces[remaining];
  if (surface.cellId >= cellCount || surface.vertexCount < 3 ||
      surface.firstVertex > view.vertexCount ||
      surface.vertexCount > view.vertexCount - surface.firstVertex) {
    rvdReportError(packedError, SurfaceInvalidPatch, patchId);
    return;
  }
  if (surface.vertexCount > SurfacePatchCapacity) {
    rvdReportError(packedError, SurfacePolygonOverflow, patchId);
    return;
  }
  const double phaseVolume = view.moments[surface.cellId].w;
  if (phaseVolume == 0.0) return;
  if (!(phaseVolume > 0.0) || !isfinite(phaseVolume)) {
    rvdReportError(packedError, SurfaceInvalidPhaseVolume, patchId);
    return;
  }
  if (surface.triangleId == InvalidId) {
    rvdReportError(packedError, SurfaceMissingSourceTriangle, patchId);
    return;
  }

  const double3 rawNormal = view.normals[remaining];
  const double normalLength = cudaMath::length3(rawNormal);
  if (!(normalLength > 0.0) || !isfinite(normalLength)) {
    rvdReportError(packedError, SurfaceInvalidNormal, patchId);
    return;
  }

  PreparedSurfacePatch record{};
  record.normal = cudaMath::mul3(rawNormal, 1.0 / normalLength);
  record.cellId = surface.cellId;
  record.modelIndex = modelIndex;
  record.runtimeModelId = view.runtimeModelId;
  record.sourceTriangleId = surface.triangleId;
  record.firstVertex = surface.firstVertex;
  record.vertexCount = surface.vertexCount;
  record.origin = cudaMath::add3(domainMin, cudaMath::mul3(view.vertices[surface.firstVertex], invBaseScale));
  record.u = rvdHost::makeBasisU(record.normal);
  record.v = cudaMath::cross3(record.normal, record.u);

  double2 polygon[SurfaceClipCapacity];
  surfaceProjectWorldPolygon(view, record, record.origin, record.u, record.v, domainMin, invBaseScale, polygon);
  double signedArea = 0.0;
  for (uint32_t i = 0; i < surface.vertexCount; ++i)
    signedArea += surfaceCross2(polygon[i], polygon[(i + 1) % surface.vertexCount]);
  if (!isfinite(signedArea) || signedArea == 0.0) {
    rvdReportError(packedError, SurfaceDegenerateFrame, patchId);
    return;
  }
  record.reversed = signedArea < 0.0 ? 1u : 0u;
  if (record.reversed != 0) surfaceReverse(polygon, surface.vertexCount);

  double area = 0.0;
  double2 centroid;
  if (!surfaceMeasurePolygon(polygon, surface.vertexCount, area, centroid) || !(area > 0.0)) {
    rvdReportError(packedError, SurfaceNonPositiveArea, patchId);
    return;
  }
  record.area = area;
  record.centroid = centroid;
  record.active = 1;
  prepared[patchId] = record;
  activeFlags[patchId] = 1;
  cellKeys[patchId] = record.cellId;
  cellValues[patchId] = patchId;
  atomicAdd(&cellCounts[record.cellId], 1u);
  pieceVertexCounts[patchId] = record.vertexCount;
}

__global__ void enumerateSurfacePairsKernel(
    const SurfaceModelView *views, const PreparedSurfacePatch *prepared, uint32_t patchCount,
    const uint32_t *sortedCellKeys, const uint32_t *sortedPatchIds,
    const uint32_t *cellCounts, const uint32_t *cellOffsets,
    double3 domainMin, double invBaseScale, double posTol,
    SurfaceCut *cuts, uint32_t *partners, uint32_t *cutCounts, uint32_t *partnerCounts,
    uint32_t *maxLevels, RvdPackedError *packedError) {
  const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= patchCount) return;
  const uint32_t patchA = sortedPatchIds[slot];
  if (prepared[patchA].active == 0) return;
  const uint32_t cell = sortedCellKeys[slot];
  const uint32_t begin = cellOffsets[cell], count = cellCounts[cell];

  double2 polyA[SurfaceClipCapacity], polyB[SurfaceClipCapacity];
  surfaceMaterializePatch(views[prepared[patchA].modelIndex], prepared[patchA], domainMin, invBaseScale, polyA);

  uint32_t cutSlots = 0, partnerSlots = 0;
  for (uint32_t entry = 0; entry < count; ++entry) {
    const uint32_t patchB = sortedPatchIds[begin + entry];
    if (patchB == patchA || prepared[patchB].active == 0) continue;
    if (prepared[patchB].modelIndex == prepared[patchA].modelIndex) continue;
    surfaceMaterializePatch(views[prepared[patchB].modelIndex], prepared[patchB], domainMin, invBaseScale, polyB);
    int coplanar = 0;
    const int hit = patchA < patchB
        ? surfaceIntersectInArea(views, prepared[patchA], polyA, prepared[patchB], polyB, domainMin, invBaseScale, posTol, coplanar)
        : surfaceIntersectInArea(views, prepared[patchB], polyB, prepared[patchA], polyA, domainMin, invBaseScale, posTol, coplanar);
    if (hit < 0) {
      rvdReportError(packedError, SurfacePolygonOverflow, patchA);
      return;
    }
    if (hit == 0) continue;
    if (coplanar != 0) {
      if (partnerSlots >= MaxPatchPartners) {
        rvdReportError(packedError, SurfacePartnerOverflow, patchA);
        return;
      }
      partners[patchA * MaxPatchPartners + partnerSlots++] = patchB;
    } else {
      if (cutSlots >= MaxPatchCuts) {
        rvdReportError(packedError, SurfaceCutOverflow, patchA);
        return;
      }
      cuts[patchA * MaxPatchCuts + cutSlots].point =
          surfaceWorldVertex(views[prepared[patchB].modelIndex], prepared[patchB], 0u, domainMin, invBaseScale);
      cuts[patchA * MaxPatchCuts + cutSlots].normal = prepared[patchB].normal;
      ++cutSlots;
    }
  }
  cutCounts[patchA] = cutSlots;
  partnerCounts[patchA] = partnerSlots;
  if (cutSlots > 0) atomicMax(&maxLevels[0], cutSlots);
  if (partnerSlots > 0) atomicMax(&maxLevels[1], partnerSlots);
}

__global__ void initPiecesKernel(
    const SurfaceModelView *views, const PreparedSurfacePatch *prepared, uint32_t patchCount,
    const uint32_t *pieceIndex, const uint32_t *pieceVertexOffset,
    double3 domainMin, double invBaseScale, SurfacePiece *pieces, double2 *vertices) {
  const uint32_t patchId = blockIdx.x * blockDim.x + threadIdx.x;
  if (patchId >= patchCount) return;
  const PreparedSurfacePatch &patch = prepared[patchId];
  if (patch.active == 0) return;
  SurfacePiece piece{};
  piece.centroid = patch.centroid;
  piece.area = patch.area;
  piece.patchId = patchId;
  piece.firstVertex = pieceVertexOffset[patchId];
  piece.vertexCount = patch.vertexCount;
  piece.contactFacing = 0;
  piece.original = 1;
  pieces[pieceIndex[patchId]] = piece;
  surfaceMaterializePatch(views[patch.modelIndex], patch, domainMin, invBaseScale, vertices + piece.firstVertex);
}

__global__ void countCutOutputsKernel(
    const SurfacePiece *pieces, const double2 *vertices, uint32_t pieceCount,
    const PreparedSurfacePatch *prepared, const SurfaceCut *cuts, const uint32_t *cutCounts,
    uint32_t level, uint32_t *outputCounts, uint32_t *outputVertexSums, RvdPackedError *packedError) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pieceCount) return;
  const SurfacePiece piece = pieces[index];
  const PreparedSurfacePatch &patch = prepared[piece.patchId];
  uint32_t outCount = 1, vertexSum = piece.vertexCount;
  if (level < cutCounts[piece.patchId]) {
    double2 pos[SurfaceClipCapacity], neg[SurfaceClipCapacity];
    SurfaceCutResult cutResult{};
    if (!surfaceApplyCut(vertices + piece.firstVertex, piece.vertexCount, cuts[piece.patchId * MaxPatchCuts + level],
                         patch, pos, neg, cutResult)) {
      rvdReportError(packedError, cutResult.overflow ? SurfacePolygonOverflow : SurfaceSplitFailure, piece.patchId);
      return;
    }
    if (cutResult.split) {
      outCount = 2;
      vertexSum = cutResult.positiveCount + cutResult.negativeCount;
    }
  }
  outputCounts[index] = outCount;
  outputVertexSums[index] = vertexSum;
}

__global__ void emitCutOutputsKernel(
    const SurfacePiece *pieces, const double2 *vertices, uint32_t pieceCount,
    const PreparedSurfacePatch *prepared, const SurfaceCut *cuts, const uint32_t *cutCounts,
    uint32_t level, const uint32_t *outputOffsets, const uint32_t *outputVertexOffsets,
    SurfacePiece *outPieces, double2 *outVertices, RvdPackedError *packedError) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pieceCount) return;
  const SurfacePiece piece = pieces[index];
  const PreparedSurfacePatch &patch = prepared[piece.patchId];
  const uint32_t base = outputOffsets[index], vertexBase = outputVertexOffsets[index];
  if (level < cutCounts[piece.patchId]) {
    double2 pos[SurfaceClipCapacity], neg[SurfaceClipCapacity];
    SurfaceCutResult cutResult{};
    if (!surfaceApplyCut(vertices + piece.firstVertex, piece.vertexCount, cuts[piece.patchId * MaxPatchCuts + level],
                         patch, pos, neg, cutResult)) {
      rvdReportError(packedError, cutResult.overflow ? SurfacePolygonOverflow : SurfaceSplitFailure, piece.patchId);
      return;
    }
    if (cutResult.split) {
      SurfacePiece out = piece;
      out.original = 0; out.area = cutResult.positiveArea; out.centroid = cutResult.positiveCentroid;
      out.vertexCount = cutResult.positiveCount; out.firstVertex = vertexBase;
      for (uint32_t v = 0; v < cutResult.positiveCount; ++v) outVertices[vertexBase + v] = pos[v];
      outPieces[base] = out;

      out.area = cutResult.negativeArea; out.centroid = cutResult.negativeCentroid;
      out.vertexCount = cutResult.negativeCount; out.firstVertex = vertexBase + cutResult.positiveCount;
      for (uint32_t v = 0; v < cutResult.negativeCount; ++v) outVertices[vertexBase + cutResult.positiveCount + v] = neg[v];
      outPieces[base + 1] = out;
      return;
    }
  }
  SurfacePiece out = piece;
  out.firstVertex = vertexBase;
  for (uint32_t v = 0; v < piece.vertexCount; ++v) outVertices[vertexBase + v] = vertices[piece.firstVertex + v];
  outPieces[base] = out;
}

__global__ void countPartitionOutputsKernel(
    const SurfacePiece *pieces, const double2 *vertices, uint32_t pieceCount,
    const SurfaceModelView *views, const PreparedSurfacePatch *prepared,
    const uint32_t *partners, const uint32_t *partnerCounts, uint32_t level,
    double3 domainMin, double invBaseScale,
    uint32_t *insideCounts, uint32_t *insideVertexCounts, uint32_t *outsideCounts, uint32_t *outsideVertexSums,
    RvdPackedError *packedError) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pieceCount) return;
  const SurfacePiece piece = pieces[index];
  if (level >= partnerCounts[piece.patchId]) {
    insideCounts[index] = 0; insideVertexCounts[index] = 0;
    outsideCounts[index] = 1; outsideVertexSums[index] = piece.vertexCount;
    return;
  }
  const uint32_t partnerId = partners[piece.patchId * MaxPatchPartners + level];
  double2 curr[SurfaceClipCapacity], next[SurfaceClipCapacity], projected[SurfaceClipCapacity], outside[SurfaceClipCapacity];
  uint32_t outsideStrips[SurfaceClipCapacity];
  SurfacePartitionResult result{};
  if (!surfacePartitionPiece(views, prepared[piece.patchId], vertices + piece.firstVertex, piece.vertexCount, piece.area,
                             prepared[partnerId], domainMin, invBaseScale, curr, next, projected, outside,
                             nullptr, nullptr, outsideStrips, result)) {
    rvdReportError(packedError, result.overflow ? SurfacePolygonOverflow : SurfacePartitionFailure, piece.patchId);
    return;
  }
  insideCounts[index] = result.insideCount;
  insideVertexCounts[index] = result.insideVertices;
  outsideCounts[index] = result.outsideCount;
  outsideVertexSums[index] = result.outsideVertices;
}

__global__ void emitPartitionOutputsKernel(
    const SurfacePiece *pieces, const double2 *vertices, uint32_t pieceCount,
    const SurfaceModelView *views, const PreparedSurfacePatch *prepared,
    const uint32_t *partners, const uint32_t *partnerCounts, uint32_t level,
    double3 domainMin, double invBaseScale,
    const uint32_t *insideOffsets, const uint32_t *insideVertexOffsets,
    const uint32_t *outsideOffsets, const uint32_t *outsideVertexOffsets,
    uint32_t terminalPieceBase, uint32_t terminalVertexBase,
    SurfacePiece *outPieces, double2 *outVertices, SurfacePiece *terminalPieces, double2 *terminalVertices,
    RvdPackedError *packedError) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pieceCount) return;
  const SurfacePiece piece = pieces[index];
  if (level >= partnerCounts[piece.patchId]) {
    SurfacePiece out = piece;
    out.firstVertex = outsideVertexOffsets[index];
    for (uint32_t v = 0; v < piece.vertexCount; ++v) outVertices[out.firstVertex + v] = vertices[piece.firstVertex + v];
    outPieces[outsideOffsets[index]] = out;
    return;
  }
  const uint32_t partnerId = partners[piece.patchId * MaxPatchPartners + level];
  double2 curr[SurfaceClipCapacity], next[SurfaceClipCapacity], projected[SurfaceClipCapacity], outside[SurfaceClipCapacity];
  uint32_t outsideStrips[SurfaceClipCapacity];
  SurfacePartitionResult result{};
  if (!surfacePartitionPiece(views, prepared[piece.patchId], vertices + piece.firstVertex, piece.vertexCount, piece.area,
                             prepared[partnerId], domainMin, invBaseScale, curr, next, projected, outside,
                             terminalVertices + terminalVertexBase + insideVertexOffsets[index],
                             outVertices + outsideVertexOffsets[index], outsideStrips, result)) {
    rvdReportError(packedError, result.overflow ? SurfacePolygonOverflow : SurfacePartitionFailure, piece.patchId);
    return;
  }
  if (result.insideCount != 0) {
    SurfacePiece inside = piece;
    inside.original = 0; inside.contactFacing = 1;
    inside.area = result.insideArea; inside.centroid = result.insideCentroid;
    inside.vertexCount = result.insideVertices;
    inside.firstVertex = terminalVertexBase + insideVertexOffsets[index];
    terminalPieces[terminalPieceBase + insideOffsets[index]] = inside;
  }
  uint32_t vertexOffset = 0;
  for (uint32_t strip = 0; strip < result.outsideCount; ++strip) {
    const uint32_t stripVertices = outsideStrips[strip];
    double area = 0.0;
    double2 centroid;
    if (!surfaceMeasurePolygon(outVertices + outsideVertexOffsets[index] + vertexOffset, stripVertices, area, centroid)) {
      rvdReportError(packedError, SurfacePartitionFailure, piece.patchId);
      return;
    }
    SurfacePiece outPiece = piece;
    outPiece.original = 0; outPiece.contactFacing = 0;
    outPiece.area = area; outPiece.centroid = centroid;
    outPiece.vertexCount = stripVertices;
    outPiece.firstVertex = outsideVertexOffsets[index] + vertexOffset;
    outPieces[outsideOffsets[index] + strip] = outPiece;
    vertexOffset += stripVertices;
  }
}

__global__ void appendSurvivorsKernel(
    const SurfacePiece *pieces, const double2 *vertices, uint32_t pieceCount,
    uint32_t terminalPieceBase, uint32_t terminalVertexBase,
    SurfacePiece *terminalPieces, double2 *terminalVertices) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pieceCount) return;
  const SurfacePiece piece = pieces[index];
  SurfacePiece out = piece;
  out.contactFacing = 0;
  out.firstVertex = terminalVertexBase + piece.firstVertex;
  for (uint32_t v = 0; v < piece.vertexCount; ++v) terminalVertices[out.firstVertex + v] = vertices[piece.firstVertex + v];
  terminalPieces[terminalPieceBase + index] = out;
}

__global__ void accumulatePatchConservationKernel(
    const SurfacePiece *pieces, uint32_t terminalCount, double *patchSummedArea, uint32_t *patchPieceCounts) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= terminalCount) return;
  const SurfacePiece &p = pieces[index];
  atomicAdd(&patchSummedArea[p.patchId], p.area);
  atomicAdd(&patchPieceCounts[p.patchId], 1u);
}

__global__ void checkPatchConservationKernel(
    const double *patchSummedArea, const uint32_t *patchPieceCounts,
    const PreparedSurfacePatch *prepared, uint32_t patchCount,
    const uint32_t *cutCounts, const uint32_t *partners, const uint32_t *partnerCounts,
    RvdPackedError *packedError, SurfaceConservationReport *report) {
  const uint32_t patchId = blockIdx.x * blockDim.x + threadIdx.x;
  if (patchId >= patchCount) return;
  const PreparedSurfacePatch &patch = prepared[patchId];
  if (patch.active == 0) return;
  const uint32_t count = patchPieceCounts[patchId];
  const double total = patchSummedArea[patchId];
  uint32_t cutEdges = cutCounts[patchId];
  for (uint32_t i = 0; i < partnerCounts[patchId]; ++i)
    cutEdges += prepared[partners[patchId * MaxPatchPartners + i]].vertexCount;
  const double tol = 512.0 * DBL_EPSILON * patch.area * double(cutEdges + 1);
  const double diff = fabs(total - patch.area);
  if (count == 0 || diff > tol) {
    if (rvdReportErrorOnce(packedError, SurfaceConservation, patchId)) {
      report->expected = patch.area;
      report->actual = total;
      report->difference = diff;
      report->tolerance = tol;
    }
  }
}

__global__ void classifyMaterialRegionsKernel(
    const SurfacePiece *pieces, uint32_t terminalCount, const PreparedSurfacePatch *prepared,
    const cudaGeometry::DeviceModelGeometryView *models, uint32_t modelCount,
    uint32_t *vertexCounts, PatchClassification *classifications) {
  const uint32_t region = blockIdx.x * blockDim.x + threadIdx.x;
  if (region >= terminalCount) return;
  const SurfacePiece piece = pieces[region];
  const PreparedSurfacePatch &patch = prepared[piece.patchId];
  vertexCounts[region] = piece.vertexCount;
  const double3 centroid = surfaceLocalToWorld(patch, piece.centroid);
  const double3 normal = patch.normal;
  PatchClassification result{};
  if (patch.modelIndex >= modelCount || patch.sourceTriangleId == InvalidId) {
    result.valid = 0;
    result.failure = SideInvalidOwner;
    classifications[region] = result;
    return;
  }

  const cudaGeometry::DeviceModelGeometryView owner = models[patch.modelIndex];
  const cudaGeometry::SideWinding ownerSide = cudaGeometry::computeSideWinding(
      owner.triangles, owner.nodes, owner.root, centroid, normal, patch.sourceTriangleId);
  if (ownerSide.ambiguous) {
    result.valid = 0; result.failure = SideDegenerateRay; result.offendingModel = patch.modelIndex;
    classifications[region] = result;
    return;
  }
  const double windingPlus = ownerSide.value;
  result.windingPlus = windingPlus;
  result.windingMinus = windingPlus + 1.0;

  const bool insidePlus = fabs(result.windingPlus) > 0.5;
  const bool insideMinus = fabs(result.windingMinus) > 0.5;
  if (insidePlus == insideMinus) {
    result.valid = 0; result.failure = SideOwnerNotDistinct;
    classifications[region] = result;
    return;
  }

  result.flipNormal = insidePlus ? 1u : 0u;
  const double3 flipped = make_double3(-normal.x, -normal.y, -normal.z);
  const double3 inward = insidePlus ? normal : flipped;
  const double3 outward = insidePlus ? flipped : normal;

  if (piece.contactFacing != 0) {
    result.valid = 1;
    result.adjacentModelIndex = InvalidId;
    classifications[region] = result;
    return;
  }

  uint32_t adjacent = InvalidId;
  for (uint32_t m = 0; m < modelCount; ++m) {
    if (m == patch.modelIndex || !pointInsideModelBounds(models[m], centroid)) continue;
    double inWinding = 0.0;
    if (!sideWinding(models[m], centroid, inward, inWinding)) {
      result.valid = 0; result.failure = SideDegenerateRay; result.offendingModel = m;
      classifications[region] = result;
      return;
    }
    if (fabs(inWinding) > 0.5) {
      result.valid = 0; result.failure = SideOwnerOverlap; result.offendingModel = m;
      classifications[region] = result;
      return;
    }
    double outWinding = 0.0;
    if (!sideWinding(models[m], centroid, outward, outWinding)) {
      result.valid = 0; result.failure = SideDegenerateRay; result.offendingModel = m;
      classifications[region] = result;
      return;
    }
    if (fabs(outWinding) <= 0.5) continue;
    if (adjacent != InvalidId) {
      result.valid = 0; result.failure = SideMultipleAdjacent; result.offendingModel = m;
      classifications[region] = result;
      return;
    }
    adjacent = m;
  }
  result.valid = 1;
  result.adjacentModelIndex = adjacent;
  classifications[region] = result;
}

__global__ void emitSurfaceRegionsKernel(
    const SurfacePiece *pieces, uint32_t terminalCount, const double2 *terminalVertices,
    const PreparedSurfacePatch *prepared, const SurfaceModelView *views,
    const PatchClassification *classifications, const uint32_t *modelRuntimeIds,
    double3 domainMin, double invBaseScale, const uint32_t *vertexOffsets,
    SurfaceRegionOutput *outputs, double3 *outVertices) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= terminalCount) return;
  const SurfacePiece piece = pieces[index];
  const PreparedSurfacePatch &patch = prepared[piece.patchId];
  const PatchClassification &classification = classifications[index];
  const uint32_t vertexBase = vertexOffsets[index];
  const uint32_t count = piece.vertexCount;

  if (piece.original != 0) {
    const SurfaceModelView &view = views[patch.modelIndex];
    for (uint32_t v = 0; v < count; ++v)
      outVertices[vertexBase + v] = surfaceWorldVertex(view, patch, v, domainMin, invBaseScale);
  } else {
    for (uint32_t v = 0; v < count; ++v) {
      outVertices[vertexBase + v] = surfaceLocalToWorld(patch, terminalVertices[piece.firstVertex + v]);
    }
  }
  if (classification.flipNormal != 0) surfaceReverse(outVertices + vertexBase, count);

  SurfaceRegionOutput output{};
  output.cellId = patch.cellId;
  output.modelIndex = patch.modelIndex;
  output.runtimeModelId = patch.runtimeModelId;
  output.sourceTriangleId = patch.sourceTriangleId;
  output.outsideMaterialId = classification.adjacentModelIndex == InvalidId ? InvalidId : modelRuntimeIds[classification.adjacentModelIndex];
  output.area = piece.area;
  const double3 centroid = surfaceLocalToWorld(patch, piece.centroid);
  output.moment = cudaMath::mul3(centroid, piece.area);
  output.normal = classification.flipNormal != 0 ? make_double3(-patch.normal.x, -patch.normal.y, -patch.normal.z) : patch.normal;
  output.firstVertex = vertexBase;
  output.vertexCount = count;
  output.contactFacing = piece.contactFacing;
  outputs[index] = output;
}

static const char *surfaceErrorName(uint32_t code) {
  static const char *names[] = {
      "none", "invalidRestrictedPatch", "invalidPhaseVolume", "invalidSurfaceNormal",
      "patchVertexCapacityExceeded", "degenerateSurfaceFrame", "nonPositivePatchArea",
      "missingSourceTriangle", "patchCutCapacityExceeded", "patchPartnerCapacityExceeded",
      "cutSplitFailure", "coplanarPartitionFailure", "areaNotConserved"};
  return (code <= SurfaceConservation) ? names[code] : "unknown";
}

static const char *sideFailureName(uint32_t f) {
  static const char *names[] = {
      "success", "invalidOwner", "degenerateRay", "ownerBoundingNotDistinct",
      "ownerInsideOverlapped", "multipleAdjacentMaterials"};
  return (f <= SideMultipleAdjacent) ? names[f] : "unknown";
}

template <typename T>
static bool growPreserving(cudaUtils::CudaBuffer<T> &buffer, size_t needed, size_t retained, cudaStream_t stream) {
  if (buffer.size() >= needed) return true;
  cudaUtils::CudaBuffer<T> grown;
  if (!grown.allocate(needed)) return false;
  if (retained > 0 &&
      !cudaUtils::cudaCheck(cudaMemcpyAsync(grown.get(), buffer.get(), sizeof(T) * retained,
                                           cudaMemcpyDeviceToDevice, stream), "grow surface arena")) {
    return false;
  }
  buffer.swap(grown);
  return true;
}

template <size_t N>
static bool readScanTotals(
    const cudaUtils::CudaBuffer<uint32_t> (&counts)[4],
    const cudaUtils::CudaBuffer<uint32_t> (&offsets)[4],
    uint32_t count, cudaStream_t stream, uint32_t (&totals)[N], const char *label) {
  if (count == 0) {
    for (size_t i = 0; i < N; ++i) totals[i] = 0;
    return true;
  }
  uint32_t last[2 * N] = {};
  for (size_t i = 0; i < N; ++i) {
    if (!rvdHost::copyToHostAsync(&last[2 * i], counts[i].get() + (count - 1), 1, stream, label) ||
        !rvdHost::copyToHostAsync(&last[2 * i + 1], offsets[i].get() + (count - 1), 1, stream, label)) return false;
  }
  if (!cudaUtils::cudaCheck(cudaStreamSynchronize(stream), label)) return false;
  for (size_t i = 0; i < N; ++i) totals[i] = last[2 * i] + last[2 * i + 1];
  return true;
}

static bool reportSurfaceFailure(
    const char *stage, cudaUtils::CudaBuffer<RvdPackedError> &deviceError,
    cudaUtils::CudaBuffer<PreparedSurfacePatch> &prepared,
    cudaUtils::CudaBuffer<SurfaceConservationReport> &conservation, cudaStream_t stream) {
  RvdPackedError packed = 0;
  if (!rvdHost::readPackedError(deviceError, stream, packed)) return true;
  if (packed == 0) return false;
  const uint32_t code = rvdErrorCode(packed), patchId = rvdErrorId(packed);
  std::cerr << "[RVDSurfacePartition] " << stage << " failed reason=" << surfaceErrorName(code) << " patch=" << patchId;
  if (patchId < prepared.size()) {
    PreparedSurfacePatch patch{};
    if (cudaMemcpy(&patch, prepared.get() + patchId, sizeof(PreparedSurfacePatch), cudaMemcpyDeviceToHost) == cudaSuccess) {
      std::cerr << " modelId=" << patch.runtimeModelId << " cell=" << patch.cellId << " vertices=" << patch.vertexCount;
    }
  }
  if (code == SurfaceConservation && conservation.size() > 0) {
    SurfaceConservationReport vals{};
    if (cudaMemcpy(&vals, conservation.get(), sizeof(SurfaceConservationReport), cudaMemcpyDeviceToHost) == cudaSuccess) {
      std::cerr << " expected=" << vals.expected << " actual=" << vals.actual
                << " diff=" << vals.difference << " tol=" << vals.tolerance;
    }
  }
  std::cerr << std::endl;
  return true;
}

bool buildSurfacePartition(
    const std::vector<ModelRestriction> &models,
    const std::vector<cudaGeometry::DeviceModelGeometryView> &geometries,
    const BaseCellDomain &baseDomain,
    std::vector<SurfaceRegion> &regions,
    cudaStream_t stream) {
  regions.clear();
  if (models.empty()) return true;
  if (geometries.size() != models.size()) return false;

  const double3 domainMin = make_double3(baseDomain.domainMin.x, baseDomain.domainMin.y, baseDomain.domainMin.z);
  const double invBaseScale = double(baseDomain.invBaseScale);

  std::vector<SurfaceModelView> hostViews(models.size());
  std::vector<uint32_t> modelRuntimeIds(models.size());
  uint32_t totalPatches = 0;
  for (uint32_t m = 0; m < models.size(); ++m) {
    SurfaceModelView &v = hostViews[m];
    v.surfaces = models[m].devSurfaces.get();
    v.normals = models[m].devSurfaceNormals.get();
    v.moments = models[m].devMoments.get();
    v.vertices = models[m].devRestrictedVertices.get();
    v.patchCount = models[m].totalPatches;
    v.vertexCount = uint32_t(models[m].devRestrictedVertices.size());
    v.runtimeModelId = models[m].runtimeModelId;
    modelRuntimeIds[m] = models[m].runtimeModelId;
    totalPatches += models[m].totalPatches;
  }
  if (totalPatches == 0 || baseDomain.count == 0) return true;

  float domainScale = glm::length(baseDomain.domainMax - baseDomain.domainMin);
  if (!(domainScale > 0.0f) || !std::isfinite(domainScale)) domainScale = 1.0f;
  const double coordMag = std::max(
      std::max(std::abs(double(baseDomain.domainMin.x)), std::abs(double(baseDomain.domainMax.x))),
      std::max(std::max(std::abs(double(baseDomain.domainMin.y)), std::abs(double(baseDomain.domainMax.y))),
               std::max(std::abs(double(baseDomain.domainMin.z)), std::abs(double(baseDomain.domainMax.z)))));
  const double intersectionTolerance = 512.0 * DBL_EPSILON * std::max(double(domainScale), coordMag);

  cudaUtils::CudaBuffer<SurfaceModelView> deviceViews;
  cudaUtils::CudaBuffer<uint32_t> deviceModelRuntimeIds;
  cudaUtils::CudaBuffer<PreparedSurfacePatch> prepared;
  cudaUtils::CudaBuffer<uint32_t> activeFlags;
  cudaUtils::CudaBuffer<uint32_t> cellKeys, cellValues, sortedCellKeys, sortedPatchIds;
  cudaUtils::CudaBuffer<uint32_t> cellCounts, cellOffsets;
  cudaUtils::CudaBuffer<uint32_t> pieceVertexCounts, pieceVertexOffsets;
  cudaUtils::CudaBuffer<RvdPackedError> deviceError;
  cudaUtils::CudaBuffer<SurfaceConservationReport> conservation;
  cudaUtils::CudaBuffer<uint8_t> scanScratch;

  if (!deviceViews.allocate(hostViews.size()) ||
      !deviceViews.uploadAsync(hostViews.data(), hostViews.size(), stream) ||
      !deviceModelRuntimeIds.allocate(modelRuntimeIds.size()) ||
      !deviceModelRuntimeIds.uploadAsync(modelRuntimeIds.data(), modelRuntimeIds.size(), stream) ||
      !prepared.allocate(totalPatches) ||
      !activeFlags.allocate(totalPatches) ||
      !cellKeys.allocate(totalPatches) ||
      !cellValues.allocate(totalPatches) ||
      !sortedCellKeys.allocate(totalPatches) ||
      !sortedPatchIds.allocate(totalPatches) ||
      !cellCounts.allocate(baseDomain.count) ||
      !cellOffsets.allocate(baseDomain.count) ||
      !pieceVertexCounts.allocate(totalPatches) ||
      !pieceVertexOffsets.allocate(totalPatches) ||
      !deviceError.allocate(1) ||
      !conservation.allocate(1) ||
      !rvdHost::zero(cellCounts.get(), sizeof(uint32_t) * baseDomain.count, stream, "reset cell counts") ||
      !rvdHost::zero(deviceError.get(), sizeof(RvdPackedError), stream, "reset error") ||
      !rvdHost::zero(conservation.get(), sizeof(SurfaceConservationReport), stream, "reset conservation")) {
    return false;
  }

  prepareSurfacePatchesKernel<<<rvdHost::blockCount(totalPatches, SurfaceBlockSize), SurfaceBlockSize, 0, stream>>>(
      deviceViews.get(), uint32_t(hostViews.size()), totalPatches, baseDomain.count, domainMin, invBaseScale,
      prepared.get(), activeFlags.get(), cellKeys.get(), cellValues.get(), cellCounts.get(), pieceVertexCounts.get(),
      deviceError.get());
  if (!rvdHost::checkLaunch("launch prepareSurfacePatchesKernel") ||
      reportSurfaceFailure("prepare", deviceError, prepared, conservation, stream)) {
    return false;
  }

  if (!rvdHost::sortPairs(cellKeys.get(), sortedCellKeys.get(), cellValues.get(), sortedPatchIds.get(),
                          int(totalPatches), scanScratch, stream) ||
      !rvdHost::exclusiveSum(cellCounts.get(), cellOffsets.get(), int(baseDomain.count), scanScratch, stream)) {
    return false;
  }

  cudaUtils::CudaBuffer<SurfaceCut> cuts;
  cudaUtils::CudaBuffer<uint32_t> partners, cutCounts, partnerCounts, maxLevels;
  if (!cuts.allocate(size_t(totalPatches) * MaxPatchCuts) ||
      !partners.allocate(size_t(totalPatches) * MaxPatchPartners) ||
      !cutCounts.allocate(totalPatches) ||
      !partnerCounts.allocate(totalPatches) ||
      !maxLevels.allocate(2) ||
      !rvdHost::zero(cutCounts.get(), sizeof(uint32_t) * totalPatches, stream, "reset cut counts") ||
      !rvdHost::zero(partnerCounts.get(), sizeof(uint32_t) * totalPatches, stream, "reset partner counts") ||
      !rvdHost::zero(maxLevels.get(), sizeof(uint32_t) * 2, stream, "reset max levels")) {
    return false;
  }

  enumerateSurfacePairsKernel<<<rvdHost::blockCount(totalPatches, SurfaceBlockSize), SurfaceBlockSize, 0, stream>>>(
      deviceViews.get(), prepared.get(), totalPatches, sortedCellKeys.get(), sortedPatchIds.get(),
      cellCounts.get(), cellOffsets.get(), domainMin, invBaseScale, intersectionTolerance,
      cuts.get(), partners.get(), cutCounts.get(), partnerCounts.get(), maxLevels.get(), deviceError.get());
  if (!rvdHost::checkLaunch("launch enumerateSurfacePairsKernel") ||
      reportSurfaceFailure("pair enumeration", deviceError, prepared, conservation, stream)) {
    return false;
  }

  uint32_t maxLevelValues[2] = {0, 0};
  if (!rvdHost::copyToHost(maxLevelValues, maxLevels.get(), 2, stream, "read surface levels")) return false;
  const uint32_t maxCutLevel = maxLevelValues[0], maxPartnerLevel = maxLevelValues[1];

  cudaUtils::CudaBuffer<uint32_t> pieceIndex;
  if (!pieceIndex.allocate(totalPatches) ||
      !rvdHost::exclusiveSum(activeFlags.get(), pieceIndex.get(), int(totalPatches), scanScratch, stream) ||
      !rvdHost::exclusiveSum(pieceVertexCounts.get(), pieceVertexOffsets.get(), int(totalPatches), scanScratch, stream)) {
    return false;
  }

  uint32_t pieceCount = 0, vertexCountA = 0;
  if (!rvdHost::readLastExclusiveScanTotal(activeFlags.get(), pieceIndex.get(), totalPatches, stream, pieceCount, "read active patch count") ||
      !rvdHost::readLastExclusiveScanTotal(pieceVertexCounts.get(), pieceVertexOffsets.get(), totalPatches, stream, vertexCountA, "read patch vertex count")) {
    return false;
  }
  if (pieceCount == 0) return true;

  const size_t maxEstPieces = size_t(totalPatches) * 4 + 1024;
  const size_t maxEstVertices = maxEstPieces * 16 + 4096;

  cudaUtils::CudaBuffer<SurfacePiece> pieceA, pieceB, terminalPieces;
  cudaUtils::CudaBuffer<double2> vertexA, vertexB, terminalVertices;
  cudaUtils::CudaBuffer<uint32_t> counts[4], offsets[4];

  if (!pieceA.allocate(maxEstPieces) || !pieceB.allocate(maxEstPieces) || !terminalPieces.allocate(maxEstPieces) ||
      !vertexA.allocate(maxEstVertices) || !vertexB.allocate(maxEstVertices) || !terminalVertices.allocate(maxEstVertices)) {
    return false;
  }
  for (int i = 0; i < 4; ++i) {
    if (!counts[i].allocate(maxEstPieces) || !offsets[i].allocate(maxEstPieces)) return false;
  }

  initPiecesKernel<<<rvdHost::blockCount(totalPatches, SurfaceBlockSize), SurfaceBlockSize, 0, stream>>>(
      deviceViews.get(), prepared.get(), totalPatches, pieceIndex.get(), pieceVertexOffsets.get(),
      domainMin, invBaseScale, pieceA.get(), vertexA.get());
  if (!rvdHost::checkLaunch("launch initPiecesKernel")) return false;

  for (uint32_t level = 0; level < maxCutLevel; ++level) {
    if (pieceCount == 0) break;
    for (int i = 0; i < 2; ++i) {
      if (!rvdHost::ensureCapacity(counts[i], pieceCount) || !rvdHost::ensureCapacity(offsets[i], pieceCount)) return false;
    }
    countCutOutputsKernel<<<rvdHost::blockCount(pieceCount, SurfacePolygonBlockSize), SurfacePolygonBlockSize, 0, stream>>>(
        pieceA.get(), vertexA.get(), pieceCount, prepared.get(), cuts.get(), cutCounts.get(), level,
        counts[0].get(), counts[1].get(), deviceError.get());
    if (!rvdHost::checkLaunch("launch countCutOutputsKernel") ||
        reportSurfaceFailure("cut split", deviceError, prepared, conservation, stream)) return false;

    if (!rvdHost::exclusiveSum(counts[0].get(), offsets[0].get(), int(pieceCount), scanScratch, stream) ||
        !rvdHost::exclusiveSum(counts[1].get(), offsets[1].get(), int(pieceCount), scanScratch, stream)) return false;

    uint32_t splitTotals[2] = {0, 0};
    if (!readScanTotals<2>(counts, offsets, pieceCount, stream, splitTotals, "read split totals")) return false;
    const uint32_t nextPieceCount = splitTotals[0], nextVertexCount = splitTotals[1];

    if (!rvdHost::ensureCapacity(pieceB, nextPieceCount) || !rvdHost::ensureCapacity(vertexB, nextVertexCount)) return false;
    if (nextPieceCount > 0) {
      emitCutOutputsKernel<<<rvdHost::blockCount(pieceCount, SurfacePolygonBlockSize), SurfacePolygonBlockSize, 0, stream>>>(
          pieceA.get(), vertexA.get(), pieceCount, prepared.get(), cuts.get(), cutCounts.get(), level,
          offsets[0].get(), offsets[1].get(), pieceB.get(), vertexB.get(), deviceError.get());
      if (!rvdHost::checkLaunch("launch emitCutOutputsKernel") ||
          reportSurfaceFailure("cut split", deviceError, prepared, conservation, stream)) return false;
    }
    pieceA.swap(pieceB);
    vertexA.swap(vertexB);
    pieceCount = nextPieceCount;
    vertexCountA = nextVertexCount;
  }

  uint32_t terminalPieceCount = 0, terminalVertexCount = 0;
  for (uint32_t level = 0; level < maxPartnerLevel; ++level) {
    if (pieceCount == 0) break;
    for (int i = 0; i < 4; ++i) {
      if (!rvdHost::ensureCapacity(counts[i], pieceCount) || !rvdHost::ensureCapacity(offsets[i], pieceCount)) return false;
    }
    countPartitionOutputsKernel<<<rvdHost::blockCount(pieceCount, SurfacePolygonBlockSize), SurfacePolygonBlockSize, 0, stream>>>(
        pieceA.get(), vertexA.get(), pieceCount, deviceViews.get(), prepared.get(), partners.get(), partnerCounts.get(),
        level, domainMin, invBaseScale, counts[0].get(), counts[1].get(), counts[2].get(), counts[3].get(), deviceError.get());
    if (!rvdHost::checkLaunch("launch countPartitionOutputsKernel") ||
        reportSurfaceFailure("coplanar partition", deviceError, prepared, conservation, stream)) return false;

    for (int i = 0; i < 4; ++i) {
      if (!rvdHost::exclusiveSum(counts[i].get(), offsets[i].get(), int(pieceCount), scanScratch, stream)) return false;
    }

    uint32_t partTotals[4] = {0};
    if (!readScanTotals<4>(counts, offsets, pieceCount, stream, partTotals, "read partition totals")) return false;
    const uint32_t levelInsidePieces = partTotals[0], levelInsideVertices = partTotals[1];
    const uint32_t nextPieceCount = partTotals[2], nextVertexCount = partTotals[3];

    if (size_t(terminalPieceCount) + levelInsidePieces > terminalPieces.size() &&
        !growPreserving(terminalPieces, size_t(terminalPieceCount) + levelInsidePieces, terminalPieceCount, stream)) return false;
    if (size_t(terminalVertexCount) + levelInsideVertices > terminalVertices.size() &&
        !growPreserving(terminalVertices, size_t(terminalVertexCount) + levelInsideVertices, terminalVertexCount, stream)) return false;
    if (!rvdHost::ensureCapacity(pieceB, nextPieceCount) || !rvdHost::ensureCapacity(vertexB, nextVertexCount)) return false;

    if (levelInsidePieces > 0 || nextPieceCount > 0) {
      emitPartitionOutputsKernel<<<rvdHost::blockCount(pieceCount, SurfacePolygonBlockSize), SurfacePolygonBlockSize, 0, stream>>>(
          pieceA.get(), vertexA.get(), pieceCount, deviceViews.get(), prepared.get(), partners.get(), partnerCounts.get(),
          level, domainMin, invBaseScale, offsets[0].get(), offsets[1].get(), offsets[2].get(), offsets[3].get(),
          terminalPieceCount, terminalVertexCount, pieceB.get(), vertexB.get(), terminalPieces.get(), terminalVertices.get(),
          deviceError.get());
      if (!rvdHost::checkLaunch("launch emitPartitionOutputsKernel") ||
          reportSurfaceFailure("coplanar partition", deviceError, prepared, conservation, stream)) return false;
    }
    terminalPieceCount += levelInsidePieces;
    terminalVertexCount += levelInsideVertices;
    pieceA.swap(pieceB);
    vertexA.swap(vertexB);
    pieceCount = nextPieceCount;
    vertexCountA = nextVertexCount;
  }

  if (pieceCount > 0) {
    if (size_t(terminalPieceCount) + pieceCount > terminalPieces.size() &&
        !growPreserving(terminalPieces, size_t(terminalPieceCount) + pieceCount, terminalPieceCount, stream)) return false;
    if (size_t(terminalVertexCount) + vertexCountA > terminalVertices.size() &&
        !growPreserving(terminalVertices, size_t(terminalVertexCount) + vertexCountA, terminalVertexCount, stream)) return false;
    appendSurvivorsKernel<<<rvdHost::blockCount(pieceCount, SurfacePolygonBlockSize), SurfacePolygonBlockSize, 0, stream>>>(
        pieceA.get(), vertexA.get(), pieceCount, terminalPieceCount, terminalVertexCount,
        terminalPieces.get(), terminalVertices.get());
    if (!rvdHost::checkLaunch("launch appendSurvivorsKernel")) return false;
    terminalPieceCount += pieceCount;
    terminalVertexCount += vertexCountA;
  }

  if (terminalPieceCount == 0) {
    std::cerr << "[RVDSurfacePartition] active patches produced no terminal surface pieces" << std::endl;
    return false;
  }

  cudaUtils::CudaBuffer<double> patchSummedArea;
  cudaUtils::CudaBuffer<uint32_t> patchPieceCounts;
  if (!patchSummedArea.allocate(totalPatches) ||
      !patchPieceCounts.allocate(totalPatches) ||
      !rvdHost::zero(patchSummedArea.get(), sizeof(double) * totalPatches, stream, "reset patch areas") ||
      !rvdHost::zero(patchPieceCounts.get(), sizeof(uint32_t) * totalPatches, stream, "reset patch piece counts")) {
    return false;
  }
  accumulatePatchConservationKernel<<<rvdHost::blockCount(terminalPieceCount, SurfaceBlockSize), SurfaceBlockSize, 0, stream>>>(
      terminalPieces.get(), terminalPieceCount, patchSummedArea.get(), patchPieceCounts.get());
  checkPatchConservationKernel<<<rvdHost::blockCount(totalPatches, SurfaceBlockSize), SurfaceBlockSize, 0, stream>>>(
      patchSummedArea.get(), patchPieceCounts.get(), prepared.get(), totalPatches, cutCounts.get(),
      partners.get(), partnerCounts.get(), deviceError.get(), conservation.get());
  if (!rvdHost::checkLaunch("launch checkPatchConservationKernel") ||
      reportSurfaceFailure("area conservation", deviceError, prepared, conservation, stream)) {
    return false;
  }

  cudaUtils::CudaBuffer<PatchClassification> classifications;
  cudaUtils::CudaBuffer<cudaGeometry::DeviceModelGeometryView> deviceModels;
  cudaUtils::CudaBuffer<uint32_t> regionVertexCounts, regionVertexOffsets;

  if (!classifications.allocate(terminalPieceCount) ||
      !deviceModels.allocate(geometries.size()) ||
      !deviceModels.uploadAsync(geometries.data(), geometries.size(), stream) ||
      !regionVertexCounts.allocate(terminalPieceCount) ||
      !regionVertexOffsets.allocate(terminalPieceCount)) {
    return false;
  }

  classifyMaterialRegionsKernel<<<rvdHost::blockCount(terminalPieceCount, SurfaceBlockSize), SurfaceBlockSize, 0, stream>>>(
      terminalPieces.get(), terminalPieceCount, prepared.get(), deviceModels.get(), uint32_t(geometries.size()),
      regionVertexCounts.get(), classifications.get());
  if (!rvdHost::checkLaunch("launch classifyMaterialRegionsKernel") ||
      !rvdHost::exclusiveSum(regionVertexCounts.get(), regionVertexOffsets.get(), int(terminalPieceCount), scanScratch, stream)) {
    return false;
  }

  uint32_t totalRegionVertices = 0;
  if (!rvdHost::readLastExclusiveScanTotal(regionVertexCounts.get(), regionVertexOffsets.get(), terminalPieceCount,
                                          stream, totalRegionVertices, "read region vertex count")) {
    return false;
  }

  cudaUtils::CudaBuffer<SurfaceRegionOutput> outputs;
  cudaUtils::CudaBuffer<double3> outVertices;
  if (!outputs.allocate(terminalPieceCount) || !outVertices.allocate(totalRegionVertices)) return false;

  emitSurfaceRegionsKernel<<<rvdHost::blockCount(terminalPieceCount, SurfaceBlockSize), SurfaceBlockSize, 0, stream>>>(
      terminalPieces.get(), terminalPieceCount, terminalVertices.get(), prepared.get(), deviceViews.get(),
      classifications.get(), deviceModelRuntimeIds.get(), domainMin, invBaseScale, regionVertexOffsets.get(),
      outputs.get(), outVertices.get());
  if (!rvdHost::checkLaunch("launch emitSurfaceRegionsKernel")) return false;

  std::vector<SurfaceRegionOutput> hostOutputs(terminalPieceCount);
  std::vector<PatchClassification> hostClassifications(terminalPieceCount);
  std::vector<double3> hostVertices(totalRegionVertices);
  if (!outputs.downloadAsync(hostOutputs.data(), hostOutputs.size(), stream) ||
      !classifications.downloadAsync(hostClassifications.data(), hostClassifications.size(), stream) ||
      !outVertices.downloadAsync(hostVertices.data(), hostVertices.size(), stream) ||
      !rvdHost::synchronize(stream, "synchronize surface partition output")) {
    return false;
  }

  for (uint32_t i = 0; i < terminalPieceCount; ++i) {
    const SurfaceRegionOutput &out = hostOutputs[i];
    const PatchClassification &cls = hostClassifications[i];
    if (cls.valid == 0) {
      const glm::dvec3 center = glm::dvec3(out.moment.x, out.moment.y, out.moment.z) / out.area;
      std::cerr << "[RVDSurfacePartition] unresolved side ownership cell=" << out.cellId
                << " modelId=" << out.runtimeModelId << " region=" << i
                << " reason=" << sideFailureName(cls.failure) << " ownerTriangle=" << out.sourceTriangleId
                << " ownerWindings=(" << cls.windingMinus << ", " << cls.windingPlus << ")"
                << " centroid=(" << center.x << ", " << center.y << ", " << center.z << ")"
                << " offendingModel=" << cls.offendingModel << std::endl;
      regions.clear();
      return false;
    }
  }

  regions.reserve(terminalPieceCount);
  for (uint32_t i = 0; i < terminalPieceCount; ++i) {
    const SurfaceRegionOutput &out = hostOutputs[i];
    SurfaceRegion reg;
    reg.cellId = out.cellId;
    reg.modelIndex = out.modelIndex;
    reg.runtimeModelId = out.runtimeModelId;
    reg.outsideMaterialId = out.outsideMaterialId;
    reg.area = out.area;
    reg.moment = glm::dvec3(out.moment.x, out.moment.y, out.moment.z);
    reg.normal = glm::dvec3(out.normal.x, out.normal.y, out.normal.z);
    reg.sourceTriangleId = out.sourceTriangleId;
    reg.contactFacing = out.contactFacing != 0;
    reg.polygon.reserve(out.vertexCount);
    for (uint32_t v = 0; v < out.vertexCount; ++v) {
      const double3 &pt = hostVertices[out.firstVertex + v];
      reg.polygon.emplace_back(pt.x, pt.y, pt.z);
    }
    regions.push_back(std::move(reg));
  }

  return true;
}

} // namespace voronoi
