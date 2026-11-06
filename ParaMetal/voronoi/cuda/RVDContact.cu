#include "RVDContact.cuh"
#include "RVDHostUtils.cuh"
#include "../../cuda/CudaBuffer.cuh"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace voronoi {

namespace {

constexpr uint32_t ContactBlockSize = rvdHost::BlockSize;

struct RegionPlane {
  glm::dvec3 centroid{0.0};
  glm::dvec3 normal{0.0};
  glm::dvec3 u{1.0, 0.0, 0.0};
  glm::dvec3 v{0.0, 1.0, 0.0};
  glm::dvec3 minimum{0.0};
  glm::dvec3 maximum{0.0};
};

struct GridKey {
  int64_t x = 0;
  int64_t y = 0;
  int64_t z = 0;
};

struct BinEntry {
  uint64_t key;
  uint32_t region;
};

struct BinEntryLess {
  __host__ __device__ bool operator()(const BinEntry &a, const BinEntry &b) const {
    return a.key < b.key || (a.key == b.key && a.region < b.region);
  }
};

struct SpatialHashRange {
  GridKey minimum;
  GridKey maximum;
};

struct DeviceContactRegion {
  double3 centroid;
  double3 normal;
  double3 u;
  double3 v;
  double3 boundsMin;
  double3 boundsMax;
  uint32_t vertexOffset;
  uint32_t vertexCount;
  uint32_t faceId;
  uint32_t fragmentId;
};

constexpr uint32_t MaxContactPolygonVertices = 32;

struct RawCandidateOverlap {
  double projectionScale;
  double area;
  uint32_t vertexCount;
  uint32_t sourceScratchOffset;
};

struct CompactContactOverlap {
  uint32_t faceA;
  uint32_t fragmentA;
  uint32_t regionA;
  uint32_t faceB;
  uint32_t fragmentB;
  uint32_t regionB;
  double projectionScale;
  double area;
  uint32_t vertexOffset;
  uint32_t vertexCount;
};

struct GpuContactCoupling {
  uint32_t faceA;
  uint32_t fragmentA;
  uint32_t faceB;
  uint32_t fragmentB;
  double area;
};

struct ContactOverlapGeometry {
  uint32_t vertexCount;
  double localArea;
  double projectionScale;
};

enum ContactUnresolvedReason : uint32_t {
  ContactResolved = 0,
  ContactInitialClipCapacity = 1,
  ContactPenetrationClipCapacity = 2,
  ContactRangeClipCapacity = 3,
  ContactNonFiniteArea = 4,
  ContactAreaCancellation = 5,
};

__device__ inline double2 projectToFrame(
    const double3 &point,
    const DeviceContactRegion &frame) {
  const double3 delta = make_double3(
      point.x - frame.centroid.x,
      point.y - frame.centroid.y,
      point.z - frame.centroid.z);
  return make_double2(
      delta.x * frame.u.x + delta.y * frame.u.y + delta.z * frame.u.z,
      delta.x * frame.v.x + delta.y * frame.v.y + delta.z * frame.v.z);
}

__device__ inline double2 projectAlongNormal(
    const double3 &point,
    const double3 &projNormal,
    const DeviceContactRegion &dstFrame) {
  const double denom =
      dstFrame.normal.x * projNormal.x +
      dstFrame.normal.y * projNormal.y +
      dstFrame.normal.z * projNormal.z;
  if (fabs(denom) < 1e-6) {
    return projectToFrame(point, dstFrame);
  }
  const double3 cDiff = make_double3(
      dstFrame.centroid.x - point.x,
      dstFrame.centroid.y - point.y,
      dstFrame.centroid.z - point.z);
  const double t = (cDiff.x * dstFrame.normal.x +
                    cDiff.y * dstFrame.normal.y +
                    cDiff.z * dstFrame.normal.z) / denom;
  const double3 pDst = make_double3(
      point.x + t * projNormal.x,
      point.y + t * projNormal.y,
      point.z + t * projNormal.z);
  return projectToFrame(pDst, dstFrame);
}

__device__ inline double projectedPolygonSignedDoubleArea(
    const DeviceContactRegion &polygon,
    const DeviceContactRegion &frame,
    const double3 *vertices) {
  if (polygon.vertexCount < 3) return 0.0;
  const double2 origin = projectToFrame(
      vertices[polygon.vertexOffset], frame);
  double area = 0.0;
  double compensation = 0.0;
  for (uint32_t i = 1; i + 1 < polygon.vertexCount; ++i) {
    const double2 pointA = projectToFrame(
        vertices[polygon.vertexOffset + i], frame);
    const double2 pointB = projectToFrame(
        vertices[polygon.vertexOffset + i + 1], frame);
    const double ax = pointA.x - origin.x;
    const double ay = pointA.y - origin.y;
    const double bx = pointB.x - origin.x;
    const double by = pointB.y - origin.y;
    const double crossTerm = ax * by - bx * ay;
    const double adjusted = crossTerm - compensation;
    const double nextArea = area + adjusted;
    compensation = (nextArea - area) - adjusted;
    area = nextArea;
  }
  return area;
}

__device__ inline int clipHalfPlane(
    const double2 *input,
    int inputCount,
    double a,
    double b,
    double c,
    double2 *output,
    uint32_t capacity) {
  if (inputCount < 3) return 0;
  int outputCount = 0;
  double2 previous = input[inputCount - 1];
  double previousDistance = a * previous.x + b * previous.y + c;
  for (int i = 0; i < inputCount; ++i) {
    const double2 current = input[i];
    const double currentDistance = a * current.x + b * current.y + c;
    if ((previousDistance < 0.0 && currentDistance > 0.0) ||
        (previousDistance > 0.0 && currentDistance < 0.0)) {
      const double t = previousDistance / (previousDistance - currentDistance);
      const double2 intersection = make_double2(
          previous.x + t * (current.x - previous.x),
          previous.y + t * (current.y - previous.y));
      if (outputCount == 0 || intersection.x != output[outputCount - 1].x ||
          intersection.y != output[outputCount - 1].y) {
        if (uint32_t(outputCount) >= capacity) return -1;
        output[outputCount++] = intersection;
      }
    }
    if (currentDistance >= 0.0 &&
        (outputCount == 0 || current.x != output[outputCount - 1].x ||
         current.y != output[outputCount - 1].y)) {
      if (uint32_t(outputCount) >= capacity) return -1;
      output[outputCount++] = current;
    }
    previous = current;
    previousDistance = currentDistance;
  }
  if (outputCount > 1 && output[0].x == output[outputCount - 1].x &&
      output[0].y == output[outputCount - 1].y) {
    --outputCount;
  }
  return outputCount;
}

__device__ inline bool convexOverlapInContactRange(
    const DeviceContactRegion &a,
    const DeviceContactRegion &b,
    const double3 *vertices,
    double contactRange,
    double penetrationTolerance,
    double2 *scratchA,
    double2 *scratchB,
    uint32_t capacity,
    double &area,
    double2* &outSource,
    uint32_t &unresolvedReason,
    ContactOverlapGeometry &geometry) {
  area = 0.0;
  outSource = nullptr;
  unresolvedReason = ContactResolved;
  geometry = ContactOverlapGeometry{};
  double3 mortarNormal = make_double3(
      a.normal.x - b.normal.x,
      a.normal.y - b.normal.y,
      a.normal.z - b.normal.z);
  const double mortarLength = sqrt(
      mortarNormal.x * mortarNormal.x + mortarNormal.y * mortarNormal.y +
      mortarNormal.z * mortarNormal.z);
  if (!(mortarLength > 0.0) || !isfinite(mortarLength)) return false;
  mortarNormal.x /= mortarLength;
  mortarNormal.y /= mortarLength;
  mortarNormal.z /= mortarLength;

  int count = int(a.vertexCount);
  const double signedA = projectedPolygonSignedDoubleArea(a, a, vertices);
  for (uint32_t i = 0; i < a.vertexCount; ++i) {
    const uint32_t source = signedA >= 0.0 ? i : (a.vertexCount - i) % a.vertexCount;
    scratchA[i] = projectToFrame(vertices[a.vertexOffset + source], a);
  }

  const double signedB = projectedPolygonSignedDoubleArea(b, a, vertices);
  const bool bCounterClockwise = signedB >= 0.0;
  double2 *source = scratchA;
  double2 *destination = scratchB;
  for (uint32_t edge = 0; edge < b.vertexCount; ++edge) {
    const uint32_t ia = bCounterClockwise
        ? edge : (b.vertexCount - edge) % b.vertexCount;
    const uint32_t ib = bCounterClockwise
        ? (edge + 1) % b.vertexCount
        : (b.vertexCount - edge - 1) % b.vertexCount;
    const double2 p0 = projectToFrame(vertices[b.vertexOffset + ia], a);
    const double2 p1 = projectToFrame(vertices[b.vertexOffset + ib], a);
    const double lineA = p0.y - p1.y;
    const double lineB = p1.x - p0.x;
    const double lineC = p1.y * p0.x - p1.x * p0.y;
    const int clipped = clipHalfPlane(
        source, count, lineA, lineB, lineC, destination, capacity);
    if (clipped < 0) {
      unresolvedReason = ContactInitialClipCapacity;
      geometry.vertexCount = uint32_t(count);
      return false;
    }
    if (clipped < 3) return false;
    count = clipped;
    double2 *temporary = source;
    source = destination;
    destination = temporary;
  }

  const double normalDot =a.normal.x * b.normal.x + a.normal.y * b.normal.y + a.normal.z * b.normal.z;
  if (normalDot > -0.2) return false;
  const double gap0 =
      (b.normal.x * (b.centroid.x - a.centroid.x) +
       b.normal.y * (b.centroid.y - a.centroid.y) +
       b.normal.z * (b.centroid.z - a.centroid.z)) / normalDot;
  const double gapU = -(b.normal.x * a.u.x + b.normal.y * a.u.y + b.normal.z * a.u.z) / normalDot;
  const double gapV = -(b.normal.x * a.v.x + b.normal.y * a.v.y + b.normal.z * a.v.z) / normalDot;
  int clipped = clipHalfPlane(source, count, gapU, gapV, gap0 + penetrationTolerance, destination, capacity);
  if (clipped < 0) {
    unresolvedReason = ContactPenetrationClipCapacity;
    geometry.vertexCount = uint32_t(count);
    return false;
  }
  if (clipped < 3) return false;
  count = clipped;
  double2 *temporary = source;
  source = destination;
  destination = temporary;
  clipped = clipHalfPlane(source, count, -gapU, -gapV, contactRange - gap0, destination, capacity);
  if (clipped < 0) {
    unresolvedReason = ContactRangeClipCapacity;
    geometry.vertexCount = uint32_t(count);
    return false;
  }
  if (clipped < 3) return false;
  count = clipped;
  source = destination;

  double signedDoubleArea = 0.0;
  double absoluteDoubleArea = 0.0;
  const double2 origin = source[0];
  double areaCompensation = 0.0;
  for (int i = 1; i + 1 < count; ++i) {
    const double ax = source[i].x - origin.x;
    const double ay = source[i].y - origin.y;
    const double bx = source[i + 1].x - origin.x;
    const double by = source[i + 1].y - origin.y;
    const double crossTerm = ax * by - bx * ay;
    const double adjusted = crossTerm - areaCompensation;
    const double nextArea = signedDoubleArea + adjusted;
    areaCompensation = (nextArea - signedDoubleArea) - adjusted;
    signedDoubleArea = nextArea;
    absoluteDoubleArea += fabs(crossTerm);
  }
  geometry.vertexCount = uint32_t(count);
  geometry.localArea = 0.5 * fabs(signedDoubleArea);
  geometry.projectionScale = fabs(a.normal.x * mortarNormal.x + a.normal.y * mortarNormal.y + a.normal.z * mortarNormal.z);
  area = geometry.localArea * geometry.projectionScale;
  outSource = source;
  if (!isfinite(area) || !isfinite(absoluteDoubleArea) ||
      !isfinite(geometry.projectionScale)) {
    unresolvedReason = ContactNonFiniteArea;
    return false;
  }
  if (area == 0.0) return false;
  if (geometry.localArea <= 256.0 * DBL_EPSILON * (0.5 * absoluteDoubleArea)) {
    unresolvedReason = ContactAreaCancellation;
    return false;
  }
  return true;
}

__global__ void clipContactCandidatesKernel(
    const uint2 *pairs,
    uint32_t pairCount,
    const DeviceContactRegion *regions,
    const double3 *vertices,
    const uint32_t *scratchOffsets,
    const uint32_t *scratchCapacities,
    double2 *scratch,
    double contactRange,
    double penetrationTolerance,
    RawCandidateOverlap *rawCandidates,
    uint32_t *valid,
    uint32_t *unresolvedReasons) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pairCount) return;
  valid[index] = 0;
  unresolvedReasons[index] = ContactResolved;

  const uint2 pair = pairs[index];
  const DeviceContactRegion &a = regions[pair.x];
  const DeviceContactRegion &b = regions[pair.y];
  if (a.faceId == InvalidId || b.faceId == InvalidId || a.faceId == b.faceId) {
    return;
  }
  const double3 gap = make_double3(
      fmax(0.0, fmax(a.boundsMin.x - b.boundsMax.x, b.boundsMin.x - a.boundsMax.x)),
      fmax(0.0, fmax(a.boundsMin.y - b.boundsMax.y, b.boundsMin.y - a.boundsMax.y)),
      fmax(0.0, fmax(a.boundsMin.z - b.boundsMax.z, b.boundsMin.z - a.boundsMax.z)));
  if (sqrt(gap.x * gap.x + gap.y * gap.y + gap.z * gap.z) > contactRange) {
    return;
  }

  const uint32_t capacity = scratchCapacities[index];
  const uint32_t offset = scratchOffsets[index];
  double2 *scratchA = scratch + offset;
  double2 *scratchB = scratchA + capacity;
  double area = 0.0;
  double2 *outSource = nullptr;
  uint32_t unresolvedReason = ContactResolved;
  ContactOverlapGeometry geometry{};
  if (!convexOverlapInContactRange(
          a, b, vertices, contactRange, penetrationTolerance,
          scratchA, scratchB, capacity, area,
          outSource, unresolvedReason, geometry)) {
    if (unresolvedReason != ContactResolved) {
      unresolvedReasons[index] = unresolvedReason;
    }
    return;
  }

  rawCandidates[index].projectionScale = geometry.projectionScale;
  rawCandidates[index].area = area;
  rawCandidates[index].vertexCount = geometry.vertexCount;
  rawCandidates[index].sourceScratchOffset = uint32_t(outSource - scratch);
  valid[index] = 1;
}

__global__ void gatherAcceptedVertexCountsKernel(
    uint32_t pairCount,
    const uint32_t *valid,
    const uint32_t *compactOffsets,
    const RawCandidateOverlap *rawCandidates,
    uint32_t *acceptedVertexCounts) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pairCount) return;
  if (valid[index] == 0) return;
  const uint32_t outIdx = compactOffsets[index];
  acceptedVertexCounts[outIdx] = rawCandidates[index].vertexCount;
}

__global__ void compactAcceptedOverlapsKernel(
    const uint2 *pairs,
    uint32_t pairCount,
    const DeviceContactRegion *regions,
    const RawCandidateOverlap *rawCandidates,
    const double2 *scratch,
    const uint32_t *valid,
    const uint32_t *compactOffsets,
    const uint32_t *acceptedVertexOffsets,
    CompactContactOverlap *compactOverlaps,
    double2 *compactVertices) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pairCount) return;
  if (valid[index] == 0) return;

  const uint32_t outIdx = compactOffsets[index];
  const RawCandidateOverlap &raw = rawCandidates[index];
  const uint2 pair = pairs[index];
  const DeviceContactRegion &a = regions[pair.x];
  const DeviceContactRegion &b = regions[pair.y];

  CompactContactOverlap compact{};
  compact.faceA = a.faceId;
  compact.fragmentA = a.fragmentId;
  compact.regionA = pair.x;
  compact.faceB = b.faceId;
  compact.fragmentB = b.fragmentId;
  compact.regionB = pair.y;
  compact.projectionScale = raw.projectionScale;
  compact.area = raw.area;
  compact.vertexOffset = acceptedVertexOffsets[outIdx];
  compact.vertexCount = raw.vertexCount;

  const double2 *srcVertices = scratch + raw.sourceScratchOffset;
  double2 *dstVertices = compactVertices + compact.vertexOffset;
  for (uint32_t v = 0; v < raw.vertexCount; ++v) {
    dstVertices[v] = srcVertices[v];
  }
  compactOverlaps[outIdx] = compact;
}

__global__ void emitCandidateInstancesKernel(
    const CompactContactOverlap *compactOverlaps,
    uint32_t acceptedCount,
    uint32_t *instanceFaceIds,
    uint32_t *instanceCandData) {
  const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= acceptedCount) return;
  const CompactContactOverlap &cand = compactOverlaps[k];
  instanceFaceIds[2 * k] = cand.faceA;
  instanceCandData[2 * k] = (k << 1) | 0;
  instanceFaceIds[2 * k + 1] = cand.faceB;
  instanceCandData[2 * k + 1] = (k << 1) | 1;
}

__device__ inline uint32_t binarySearchLower(
    const uint32_t *array, uint32_t size, uint32_t key) {
  uint32_t l = 0, r = size;
  while (l < r) {
    uint32_t m = l + (r - l) / 2;
    if (array[m] < key) l = m + 1;
    else r = m;
  }
  return l;
}

__device__ inline uint32_t binarySearchUpper(const uint32_t *array, uint32_t size, uint32_t key) {
  uint32_t l = 0, r = size;
  while (l < r) {
    uint32_t m = l + (r - l) / 2;
    if (array[m] <= key) l = m + 1;
    else r = m;
  }
  return l;
}

constexpr int MaxLocalVertices = 32;
constexpr int MaxPieces = 32;

struct LocalPolygon {
  int count;
  double2 v[MaxLocalVertices];
};

__device__ inline double localPolygonArea(const double2 *v, int count) {
  if (count < 3) return 0.0;
  double area = 0.0;
  for (int i = 1; i + 1 < count; ++i) {
    area += (v[i].x - v[0].x) * (v[i + 1].y - v[0].y) - (v[i + 1].x - v[0].x) * (v[i].y - v[0].y);
  }
  return 0.5 * fabs(area);
}

__device__ inline void ensurePolygonCCW(double2 *v, int count) {
  if (count < 3) return;
  double signedArea = 0.0;
  for (int i = 1; i + 1 < count; ++i) {
    signedArea += (v[i].x - v[0].x) * (v[i + 1].y - v[0].y) - (v[i + 1].x - v[0].x) * (v[i].y - v[0].y);
  }
  if (signedArea < 0.0) {
    for (int i = 0; i < count / 2; ++i) {
      double2 tmp = v[i];
      v[i] = v[count - 1 - i];
      v[count - 1 - i] = tmp;
    }
  }
}

__device__ inline bool subtractConvexObstacle(
    const double2 *obsVertices,
    int obsCount,
    LocalPolygon *piecesIn,
    int numPiecesIn,
    LocalPolygon *piecesOut,
    int &numPiecesOut,
    int maxPieces) {
  if (obsCount < 3) {
    numPiecesOut = numPiecesIn;
    for (int i = 0; i < numPiecesIn; ++i) piecesOut[i] = piecesIn[i];
    return true;
  }

  double2 obs[MaxLocalVertices];
  int count = obsCount < MaxLocalVertices ? obsCount : MaxLocalVertices;
  for (int i = 0; i < count; ++i) obs[i] = obsVertices[i];
  ensurePolygonCCW(obs, count);

  numPiecesOut = 0;
  for (int p = 0; p < numPiecesIn; ++p) {
    LocalPolygon currentInside = piecesIn[p];
    for (int e = 0; e < count && currentInside.count >= 3; ++e) {
      int nextE = (e + 1 == count) ? 0 : e + 1;
      double lineA = obs[e].y - obs[nextE].y;
      double lineB = obs[nextE].x - obs[e].x;
      double lineC = obs[nextE].y * obs[e].x - obs[nextE].x * obs[e].y;

      if (numPiecesOut < maxPieces) {
        int outCount = clipHalfPlane(
            currentInside.v, currentInside.count, -lineA, -lineB, -lineC,
            piecesOut[numPiecesOut].v, MaxLocalVertices);
        if (outCount >= 3) {
          double a = localPolygonArea(piecesOut[numPiecesOut].v, outCount);
          if (a > 1e-18) {
            piecesOut[numPiecesOut].count = outCount;
            numPiecesOut++;
          }
        }
      } else {
        return false;
      }

      double2 nextInside[MaxLocalVertices];
      int inCount = clipHalfPlane(
          currentInside.v, currentInside.count, lineA, lineB, lineC,
          nextInside, MaxLocalVertices);
      currentInside.count = inCount;
      for (int i = 0; i < inCount; ++i) currentInside.v[i] = nextInside[i];
    }
  }
  return true;
}

__global__ void resolveMortarCouplingsKernel(
    const CompactContactOverlap *compactOverlaps,
    const double2 *compactVertices,
    uint32_t overlapCount,
    const DeviceContactRegion *regions,
    const uint32_t *sortedFaceIds,
    const uint32_t *sortedCandData,
    uint32_t instanceCount,
    double *resolvedAreas,
    uint32_t *resolvedValid) {
  const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= overlapCount) return;
  resolvedValid[k] = 0;
  resolvedAreas[k] = 0.0;

  const CompactContactOverlap &cand = compactOverlaps[k];

  const uint32_t startA = binarySearchLower(sortedFaceIds, instanceCount, cand.faceA);
  const uint32_t endA = binarySearchUpper(sortedFaceIds, instanceCount, cand.faceA);
  const uint32_t countA = endA - startA;

  const uint32_t startB = binarySearchLower(sortedFaceIds, instanceCount, cand.faceB);
  const uint32_t endB = binarySearchUpper(sortedFaceIds, instanceCount, cand.faceB);
  const uint32_t countB = endB - startB;

  if (countA <= 1 && countB <= 1) {
    resolvedAreas[k] = cand.area;
    resolvedValid[k] = (cand.area > 0.0) ? 1 : 0;
    return;
  }

  const DeviceContactRegion &regA = regions[cand.regionA];
  const DeviceContactRegion &regB = regions[cand.regionB];
  const double3 C_A = regA.centroid;
  const double3 n_A = regA.normal;
  const double3 u_A = regA.u;
  const double3 v_A = regA.v;

  const double3 C_B = regB.centroid;
  const double3 n_B = regB.normal;
  const double3 u_B = regB.u;
  const double3 v_B = regB.v;

  const double denomAB = n_A.x * n_B.x + n_A.y * n_B.y + n_A.z * n_B.z;
  if (fabs(denomAB) < 1e-6) return;

  const double3 cDiffAB = make_double3(C_B.x - C_A.x, C_B.y - C_A.y, C_B.z - C_A.z);
  const double candG0_A = (cDiffAB.x * n_B.x + cDiffAB.y * n_B.y + cDiffAB.z * n_B.z) / denomAB;
  const double candGU_A = -(u_A.x * n_B.x + u_A.y * n_B.y + u_A.z * n_B.z) / denomAB;
  const double candGV_A = -(v_A.x * n_B.x + v_A.y * n_B.y + v_A.z * n_B.z) / denomAB;

  const double3 cDiffBA = make_double3(C_A.x - C_B.x, C_A.y - C_B.y, C_A.z - C_B.z);
  const double candG0_B = (cDiffBA.x * n_A.x + cDiffBA.y * n_A.y + cDiffBA.z * n_A.z) / denomAB;
  const double candGU_B = -(u_B.x * n_A.x + u_B.y * n_A.y + u_B.z * n_A.z) / denomAB;
  const double candGV_B = -(v_B.x * n_A.x + v_B.y * n_A.y + v_B.z * n_A.z) / denomAB;

  LocalPolygon piecesBufA[MaxPieces];
  LocalPolygon piecesBufB[MaxPieces];
  LocalPolygon *currentPieces = piecesBufA;
  LocalPolygon *nextPieces = piecesBufB;

  int numCurrentPieces = 1;
  currentPieces[0].count = cand.vertexCount < MaxLocalVertices ? cand.vertexCount : MaxLocalVertices;
  for (int v = 0; v < currentPieces[0].count; ++v) {
    currentPieces[0].v[v] = compactVertices[cand.vertexOffset + v];
  }

  // process competitors on face A
  for (uint32_t i = startA; i < endA; ++i) {
    if (numCurrentPieces == 0) break;
    uint32_t compCandIdx = sortedCandData[i] >> 1;
    if (compCandIdx == k) continue;
    const CompactContactOverlap &comp = compactOverlaps[compCandIdx];
    uint32_t partnerRegion = (comp.faceA == cand.faceA) ? comp.regionB : comp.regionA;
    uint32_t partnerFace = (comp.faceA == cand.faceA) ? comp.faceB : comp.faceA;
    const DeviceContactRegion &partner = regions[partnerRegion];

    const double denomComp = n_A.x * partner.normal.x + n_A.y * partner.normal.y + n_A.z * partner.normal.z;
    if (fabs(denomComp) < 1e-6) continue;

    const double3 cDiff = make_double3(partner.centroid.x - C_A.x, partner.centroid.y - C_A.y, partner.centroid.z - C_A.z);
    const double compG0_A = (cDiff.x * partner.normal.x + cDiff.y * partner.normal.y + cDiff.z * partner.normal.z) / denomComp;
    const double compGU_A = -(u_A.x * partner.normal.x + u_A.y * partner.normal.y + u_A.z * partner.normal.z) / denomComp;
    const double compGV_A = -(v_A.x * partner.normal.x + v_A.y * partner.normal.y + v_A.z * partner.normal.z) / denomComp;

    double lineA = candGU_A - compGU_A;
    double lineB = candGV_A - compGV_A;
    double lineC = candG0_A - compG0_A;
    if (fabs(lineA) < 1e-12 && fabs(lineB) < 1e-12 && fabs(lineC) < 1e-12) {
      if (partnerFace > cand.faceB) continue;
      lineC = 1.0;
    }

    const DeviceContactRegion &compFrame = regions[comp.regionA];
    double2 compPoly[MaxLocalVertices];
    int compPolyCount = comp.vertexCount < MaxLocalVertices ? comp.vertexCount : MaxLocalVertices;
    for (int v = 0; v < compPolyCount; ++v) {
      double2 localV = compactVertices[comp.vertexOffset + v];
      double3 p3d = make_double3(
          compFrame.centroid.x + localV.x * compFrame.u.x + localV.y * compFrame.v.x,
          compFrame.centroid.y + localV.x * compFrame.u.y + localV.y * compFrame.v.y,
          compFrame.centroid.z + localV.x * compFrame.u.z + localV.y * compFrame.v.z);
      compPoly[v] = projectAlongNormal(p3d, compFrame.normal, regA);
    }

    double2 obsPoly[MaxLocalVertices];
    int obsPolyCount = clipHalfPlane(
        compPoly, compPolyCount, lineA, lineB, lineC, obsPoly, MaxLocalVertices);
    if (obsPolyCount >= 3) {
      int numNextPieces = 0;
      if (!subtractConvexObstacle(
              obsPoly, obsPolyCount, currentPieces, numCurrentPieces,
              nextPieces, numNextPieces, MaxPieces)) {
        return;
      }
      LocalPolygon *tmp = currentPieces;
      currentPieces = nextPieces;
      nextPieces = tmp;
      numCurrentPieces = numNextPieces;
    }
  }

  // process competitors on face B
  for (uint32_t i = startB; i < endB; ++i) {
    if (numCurrentPieces == 0) break;
    uint32_t compCandIdx = sortedCandData[i] >> 1;
    if (compCandIdx == k) continue;
    const CompactContactOverlap &comp = compactOverlaps[compCandIdx];
    uint32_t partnerRegion = (comp.faceA == cand.faceB) ? comp.regionB : comp.regionA;
    uint32_t partnerFace = (comp.faceA == cand.faceB) ? comp.faceB : comp.faceA;
    const DeviceContactRegion &partner = regions[partnerRegion];

    const double denomComp = n_B.x * partner.normal.x + n_B.y * partner.normal.y + n_B.z * partner.normal.z;
    if (fabs(denomComp) < 1e-6) continue;

    const double3 cDiff = make_double3(partner.centroid.x - C_B.x, partner.centroid.y - C_B.y, partner.centroid.z - C_B.z);
    const double compG0_B = (cDiff.x * partner.normal.x + cDiff.y * partner.normal.y + cDiff.z * partner.normal.z) / denomComp;
    const double compGU_B = -(u_B.x * partner.normal.x + u_B.y * partner.normal.y + u_B.z * partner.normal.z) / denomComp;
    const double compGV_B = -(v_B.x * partner.normal.x + v_B.y * partner.normal.y + v_B.z * partner.normal.z) / denomComp;

    double lineA_B = candGU_B - compGU_B;
    double lineB_B = candGV_B - compGV_B;
    double lineC_B = candG0_B - compG0_B;
    if (fabs(lineA_B) < 1e-12 && fabs(lineB_B) < 1e-12 && fabs(lineC_B) < 1e-12) {
      if (partnerFace > cand.faceA) continue;
      lineC_B = 1.0;
    }

    const DeviceContactRegion &compFrame = regions[comp.regionA];
    double2 compPoly_B[MaxLocalVertices];
    int compPolyCount = comp.vertexCount < MaxLocalVertices ? comp.vertexCount : MaxLocalVertices;
    for (int v = 0; v < compPolyCount; ++v) {
      double2 localV = compactVertices[comp.vertexOffset + v];
      double3 p3d = make_double3(
          compFrame.centroid.x + localV.x * compFrame.u.x + localV.y * compFrame.v.x,
          compFrame.centroid.y + localV.x * compFrame.u.y + localV.y * compFrame.v.y,
          compFrame.centroid.z + localV.x * compFrame.u.z + localV.y * compFrame.v.z);
      compPoly_B[v] = projectAlongNormal(p3d, compFrame.normal, regB);
    }

    double2 clippedObs_B[MaxLocalVertices];
    int clippedObsCount = clipHalfPlane(compPoly_B, compPolyCount, lineA_B, lineB_B, lineC_B, clippedObs_B, MaxLocalVertices);
    if (clippedObsCount >= 3) {
        double2 obsOnA[MaxLocalVertices];
      for (int v = 0; v < clippedObsCount; ++v) {
        double3 p3d = make_double3(
            C_B.x + clippedObs_B[v].x * u_B.x + clippedObs_B[v].y * v_B.x,
            C_B.y + clippedObs_B[v].x * u_B.y + clippedObs_B[v].y * v_B.y,
            C_B.z + clippedObs_B[v].x * u_B.z + clippedObs_B[v].y * v_B.z);
        obsOnA[v] = projectAlongNormal(p3d, n_A, regA);
      }

      int numNextPieces = 0;
      if (!subtractConvexObstacle(
              obsOnA, clippedObsCount, currentPieces, numCurrentPieces,
              nextPieces, numNextPieces, MaxPieces)) {
        return;
      }
      LocalPolygon *tmp = currentPieces;
      currentPieces = nextPieces;
      nextPieces = tmp;
      numCurrentPieces = numNextPieces;
    }
  }

  double netLocalArea = 0.0;
  for (int p = 0; p < numCurrentPieces; ++p) {
    netLocalArea += localPolygonArea(currentPieces[p].v, currentPieces[p].count);
  }
  double finalMortarArea = netLocalArea * cand.projectionScale;
  resolvedAreas[k] = finalMortarArea;
  resolvedValid[k] = (finalMortarArea > 1e-15) ? 1 : 0;
}

__global__ void emitFinalCouplingsKernel(
    const CompactContactOverlap *compactOverlaps,
    const double *resolvedAreas,
    const uint32_t *resolvedValid,
    const uint32_t *resolvedOffsets,
    uint32_t overlapCount,
    GpuContactCoupling *finalCouplings) {
  const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= overlapCount) return;
  if (resolvedValid[k] == 0) return;

  const uint32_t outIdx = resolvedOffsets[k];
  const CompactContactOverlap &cand = compactOverlaps[k];
  GpuContactCoupling coupling{};
  coupling.faceA = cand.faceA;
  coupling.fragmentA = cand.fragmentA;
  coupling.faceB = cand.faceB;
  coupling.fragmentB = cand.fragmentB;
  coupling.area = resolvedAreas[k];
  finalCouplings[outIdx] = coupling;
}

__device__ inline bool contactAabbWithinRange(
    const double3 &minA, const double3 &maxA,
    const double3 &minB, const double3 &maxB,
    double contactRange) {
  const double dx = fmax(0.0, fmax(minA.x - maxB.x, minB.x - maxA.x));
  const double dy = fmax(0.0, fmax(minA.y - maxB.y, minB.y - maxA.y));
  const double dz = fmax(0.0, fmax(minA.z - maxB.z, minB.z - maxA.z));
  return (dx * dx + dy * dy + dz * dz) <= (contactRange * contactRange);
}

__global__ void countPairsInBinsKernel(
    const BinEntry *entries,
    uint32_t entryCount,
    const uint32_t *regionModels,
    const double3 *regionBoundsMin,
    const double3 *regionBoundsMax,
    double contactRange,
    uint32_t *pairCounts,
    uint32_t *overflow) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= entryCount) return;
  pairCounts[index] = 0;
  if (index > 0 && entries[index - 1].key == entries[index].key) return;

  uint32_t left = index + 1;
  uint32_t right = entryCount;
  while (left < right) {
    const uint32_t middle = left + (right - left) / 2;
    if (entries[middle].key <= entries[index].key) left = middle + 1;
    else right = middle;
  }
  uint64_t count = 0;
  for (uint32_t a = index; a < left; ++a) {
    const uint32_t regA = entries[a].region;
    const uint32_t modelA = regionModels[regA];
    const double3 minA = regionBoundsMin[regA];
    const double3 maxA = regionBoundsMax[regA];
    for (uint32_t b = a + 1; b < left; ++b) {
      const uint32_t regB = entries[b].region;
      if (modelA != regionModels[regB]) {
        if (contactAabbWithinRange(minA, maxA, regionBoundsMin[regB], regionBoundsMax[regB], contactRange)) {
          ++count;
        }
      }
    }
  }
  if (count > UINT32_MAX) {
    atomicExch(overflow, 1u);
    return;
  }
  pairCounts[index] = uint32_t(count);
}

__global__ void emitPairsFromBinsKernel(
    const BinEntry *entries,
    uint32_t entryCount,
    const uint32_t *regionModels,
    const double3 *regionBoundsMin,
    const double3 *regionBoundsMax,
    double contactRange,
    const uint32_t *pairOffsets,
    uint64_t *pairKeys) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= entryCount) return;
  if (index > 0 && entries[index - 1].key == entries[index].key) return;

  uint32_t left = index + 1;
  uint32_t right = entryCount;
  while (left < right) {
    const uint32_t middle = left + (right - left) / 2;
    if (entries[middle].key <= entries[index].key) left = middle + 1;
    else right = middle;
  }
  uint32_t output = pairOffsets[index];
  for (uint32_t a = index; a < left; ++a) {
    const uint32_t regA = entries[a].region;
    const uint32_t modelA = regionModels[regA];
    const double3 minA = regionBoundsMin[regA];
    const double3 maxA = regionBoundsMax[regA];
    for (uint32_t b = a + 1; b < left; ++b) {
      const uint32_t regB = entries[b].region;
      if (modelA == regionModels[regB]) continue;
      if (!contactAabbWithinRange(minA, maxA, regionBoundsMin[regB], regionBoundsMax[regB], contactRange)) continue;
      const uint32_t regionA = min(regA, regB);
      const uint32_t regionB = max(regA, regB);
      pairKeys[output++] = (uint64_t(regionA) << 32) | uint64_t(regionB);
    }
  }
}

__global__ void countSpatialHashEntriesKernel(
    const SpatialHashRange *ranges,
    const uint8_t *active,
    uint32_t regionCount,
    uint32_t *counts,
    uint32_t *overflow) {
  const uint32_t region = blockIdx.x * blockDim.x + threadIdx.x;
  if (region >= regionCount) return;
  counts[region] = 0;
  if (!active[region]) return;
  const SpatialHashRange range = ranges[region];
  const uint64_t nx = uint64_t(range.maximum.x - range.minimum.x + 1);
  const uint64_t ny = uint64_t(range.maximum.y - range.minimum.y + 1);
  const uint64_t nz = uint64_t(range.maximum.z - range.minimum.z + 1);
  if (nx == 0 || ny == 0 || nz == 0 ||
      nx > UINT32_MAX / ny || nx * ny > UINT32_MAX / nz) {
    atomicExch(overflow, 1u);
    return;
  }
  const uint64_t count = nx * ny * nz;
  if (count > UINT32_MAX) {
    atomicExch(overflow, 1u);
    return;
  }
  counts[region] = uint32_t(count);
}

__global__ void emitSpatialHashEntriesKernel(
    const SpatialHashRange *ranges,
    const uint8_t *active,
    uint32_t regionCount,
    const uint32_t *offsets,
    uint64_t dimX,
    uint64_t dimY,
    BinEntry *entries) {
  const uint32_t region = blockIdx.x * blockDim.x + threadIdx.x;
  if (region >= regionCount || !active[region]) return;
  const SpatialHashRange range = ranges[region];
  uint32_t destination = offsets[region];
  for (int64_t z = range.minimum.z; z <= range.maximum.z; ++z) {
    for (int64_t y = range.minimum.y; y <= range.maximum.y; ++y) {
      for (int64_t x = range.minimum.x; x <= range.maximum.x; ++x) {
        const uint64_t key =
            (uint64_t(z) * dimY + uint64_t(y)) * dimX + uint64_t(x);
        entries[destination++] = {key, region};
      }
    }
  }
}

// host contact AABB hash coordinates
bool gridCoordinate(double value, double minimum, double cellSize, int64_t &result) {
  if (!(cellSize > 0.0) || !std::isfinite(cellSize) || !std::isfinite(value) || !std::isfinite(minimum)) {
    return false;
  }
  const double coordinate = std::floor((value - minimum) / cellSize);
  if (!std::isfinite(coordinate) ||
      coordinate < double(std::numeric_limits<int64_t>::min()) ||
      coordinate >= double(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  result = int64_t(coordinate);
  return true;
}

bool spatialHashRange(
    const RegionPlane &plane,
    double tolerance,
      const glm::dvec3 &origin,
      double cellSize,
    SpatialHashRange &range) {
  return gridCoordinate(
             plane.minimum.x - tolerance, origin.x, cellSize,
             range.minimum.x) &&
         gridCoordinate(
             plane.minimum.y - tolerance, origin.y, cellSize,
             range.minimum.y) &&
         gridCoordinate(
             plane.minimum.z - tolerance, origin.z, cellSize,
             range.minimum.z) &&
         gridCoordinate(
             plane.maximum.x + tolerance, origin.x, cellSize,
             range.maximum.x) &&
         gridCoordinate(
             plane.maximum.y + tolerance, origin.y, cellSize,
             range.maximum.y) &&
         gridCoordinate(
             plane.maximum.z + tolerance, origin.z, cellSize,
             range.maximum.z);
}

bool buildRegionPlane(const SurfaceRegion &region, RegionPlane &plane) {
  plane = RegionPlane{};
  if (!(region.area > 0.0) || !std::isfinite(region.area) ||
      region.polygon.size() < 3) {
    return false;
  }
  const double normalLength = glm::length(region.normal);
  if (!(normalLength > 0.0) || !std::isfinite(normalLength)) {
    return false;
  }
  plane.centroid = region.moment / region.area;
  plane.normal = region.normal / normalLength;
  plane.u = rvdHost::makeBasisU(plane.normal);
  plane.v = glm::cross(plane.normal, plane.u);

  plane.minimum = glm::dvec3(std::numeric_limits<double>::max());
  plane.maximum = glm::dvec3(-std::numeric_limits<double>::max());
  for (const glm::dvec3 &world : region.polygon) {
    if (!std::isfinite(world.x) || !std::isfinite(world.y) ||
        !std::isfinite(world.z)) {
      return false;
    }
    plane.minimum = glm::min(plane.minimum, world);
    plane.maximum = glm::max(plane.maximum, world);
  }
  return region.polygon.size() >= 3 && std::isfinite(plane.centroid.x) &&
      std::isfinite(plane.centroid.y) && std::isfinite(plane.centroid.z);
}

uint32_t solidFragmentForRegion(
    const SurfaceRegion &region,
    const RVDResult &rvd) {
  if (region.faceId >= rvd.fragmentFaces.size()) return InvalidId;

  const FragmentFace &face = rvd.fragmentFaces[region.faceId];
  uint32_t ownerFragment = InvalidId;
  for (uint32_t fragment : {face.fragmentA, face.fragmentB}) {
    if (fragment >= rvd.fragments.size() || rvd.fragments[fragment].ownerModelId != region.runtimeModelId) {
      continue;
    }
    if (ownerFragment != InvalidId) return InvalidId;
    ownerFragment = fragment;
  }
  return ownerFragment;
}

bool reportUnresolvedContacts(const std::vector<uint32_t> &reasons) {
  size_t unresolvedCount = 0;
  for (size_t i = 0; i < reasons.size(); ++i) {
    if (reasons[i] != ContactResolved) ++unresolvedCount;
  }
  if (unresolvedCount == 0) return false;
  std::cerr << "[RVDContact] unresolvedCandidateCount=" << unresolvedCount
            << std::endl;
  return true;
}

double computeDryContactGapTolerance(const BaseCellDomain &baseDomain) {
  float coordScale = glm::length(baseDomain.domainMax - baseDomain.domainMin);
  if (!(coordScale > 0.0f) || !std::isfinite(coordScale)) coordScale = 1.0f;
  return std::max(
      64.0 * double(FLT_EPSILON) * double(coordScale),
      256.0 * DBL_EPSILON * double(coordScale));
}

} // namespace

bool buildContactCouplings(
    const std::vector<SurfaceRegion> &regions,
    const RVDResult &rvd,
    const BaseCellDomain &baseDomain,
    double contactRange,
    std::vector<ContactCoupling> &couplings,
    cudaStream_t stream) {
  couplings.clear();
  if (regions.empty()) return true;
  if (!(contactRange > 0.0) || !std::isfinite(contactRange)) {
    std::cerr << "[RVDContact] invalid contactRange=" << contactRange << std::endl;
    return false;
  }

  const double numerical = computeDryContactGapTolerance(baseDomain);
  const double tolerance = std::max(contactRange, numerical);

  std::vector<RegionPlane> planes(regions.size());
  std::vector<uint32_t> solidFragments(regions.size(), InvalidId);
  for (size_t i = 0; i < regions.size(); ++i) {
    if (regions[i].faceId == InvalidId) continue;
    solidFragments[i] = solidFragmentForRegion(regions[i], rvd);
    if (solidFragments[i] == InvalidId) {
      std::cerr << "[RVDContact] cannot resolve solid-side incidence for "
                << "region=" << i << " face=" << regions[i].faceId
                << " modelId=" << regions[i].runtimeModelId << std::endl;
      return false;
    }
    if (!buildRegionPlane(regions[i], planes[i])) {
      std::cerr << "[RVDContact] invalid surface region plane " << i << std::endl;
      return false;
    }
  }

  std::vector<uint32_t> contactSurfaces;
  glm::dvec3 boundsMin(std::numeric_limits<double>::max());
  glm::dvec3 boundsMax(-std::numeric_limits<double>::max());
  double extentSum = 0.0;
  double maximumExtent = 0.0;
  for (uint32_t i = 0; i < regions.size(); ++i) {
    if (regions[i].faceId == InvalidId) {
      continue;
    }
    contactSurfaces.push_back(i);
    boundsMin = glm::min(boundsMin, planes[i].minimum);
    boundsMax = glm::max(boundsMax, planes[i].maximum);
    const glm::dvec3 regionExtent = planes[i].maximum - planes[i].minimum;
    const double extent = std::max(regionExtent.x, std::max(regionExtent.y, regionExtent.z));
    extentSum += extent;
    maximumExtent = std::max(maximumExtent, extent);
  }

  std::vector<uint32_t> hostRegionModels(regions.size());
  for (uint32_t i = 0; i < regions.size(); ++i) {
    hostRegionModels[i] = regions[i].modelIndex;
  }

  if (!contactSurfaces.empty()) {
    const double averageExtent = extentSum / double(contactSurfaces.size());
    const double gridSize = std::max(std::max(tolerance * 8.0, averageExtent * 1.5), maximumExtent / 16.0);
    if (!(gridSize > 0.0) || !std::isfinite(gridSize)) return false;

    const glm::dvec3 gridMin = boundsMin - glm::dvec3(tolerance);
    std::vector<SpatialHashRange> ranges(regions.size());
    std::vector<uint8_t> activeRegions(regions.size(), 0u);
    GridKey maximumBin{};
    for (uint32_t region : contactSurfaces) {
      if (!spatialHashRange(planes[region], tolerance, gridMin, gridSize,
                            ranges[region])) return false;
      activeRegions[region] = 1u;
      maximumBin.x = std::max(maximumBin.x, ranges[region].maximum.x);
      maximumBin.y = std::max(maximumBin.y, ranges[region].maximum.y);
      maximumBin.z = std::max(maximumBin.z, ranges[region].maximum.z);
    }
    if (maximumBin.x < 0 || maximumBin.y < 0 || maximumBin.z < 0) return false;
    const uint64_t dimX = uint64_t(maximumBin.x) + 1;
    const uint64_t dimY = uint64_t(maximumBin.y) + 1;
    if (dimX != 0 && dimY > UINT64_MAX / dimX) return false;
    const uint64_t xy = dimX * dimY;
    const uint64_t dimZ = uint64_t(maximumBin.z) + 1;
    if (xy != 0 && dimZ > UINT64_MAX / xy) return false;

    std::vector<double3> hostBoundsMin(regions.size(), make_double3(0.0, 0.0, 0.0));
    std::vector<double3> hostBoundsMax(regions.size(), make_double3(0.0, 0.0, 0.0));
    for (uint32_t region : contactSurfaces) {
      hostBoundsMin[region] = make_double3(planes[region].minimum.x, planes[region].minimum.y, planes[region].minimum.z);
      hostBoundsMax[region] = make_double3(planes[region].maximum.x, planes[region].maximum.y, planes[region].maximum.z);
    }

    cudaUtils::CudaBuffer<SpatialHashRange> deviceRanges;
    cudaUtils::CudaBuffer<uint8_t> deviceActiveRegions;
    cudaUtils::CudaBuffer<uint32_t> deviceRegionModels;
    cudaUtils::CudaBuffer<double3> deviceRegionBoundsMin;
    cudaUtils::CudaBuffer<double3> deviceRegionBoundsMax;
    if (!deviceRanges.allocate(ranges.size()) ||
        !deviceRanges.uploadAsync(ranges.data(), ranges.size(), stream) ||
        !deviceActiveRegions.allocate(activeRegions.size()) ||
        !deviceActiveRegions.uploadAsync(activeRegions.data(), activeRegions.size(), stream) ||
        !deviceRegionModels.allocate(regions.size()) ||
        !deviceRegionModels.uploadAsync(hostRegionModels.data(), hostRegionModels.size(), stream) ||
        !deviceRegionBoundsMin.allocate(regions.size()) ||
        !deviceRegionBoundsMin.uploadAsync(hostBoundsMin.data(), hostBoundsMin.size(), stream) ||
        !deviceRegionBoundsMax.allocate(regions.size()) ||
        !deviceRegionBoundsMax.uploadAsync(hostBoundsMax.data(), hostBoundsMax.size(), stream)) {
      return false;
    }

    cudaUtils::CudaBuffer<uint32_t> deviceBinCounts, deviceBinOffsets;
    cudaUtils::CudaBuffer<uint32_t> deviceOverflow;
    cudaUtils::CudaBuffer<uint8_t> binScanScratch;
    if (!deviceBinCounts.allocate(regions.size() + 1) ||
        !deviceBinOffsets.allocate(regions.size() + 1) ||
        !deviceOverflow.allocate(1) ||
        !rvdHost::zero(deviceOverflow.get(), sizeof(uint32_t), stream, "clear contact bin overflow") ||
        !rvdHost::zero(
            deviceBinCounts.get() + regions.size(), sizeof(uint32_t), stream,
            "clear contact bin count tail")) {
      return false;
    }
    countSpatialHashEntriesKernel<<<
        rvdHost::blockCount(uint32_t(regions.size()), ContactBlockSize),
        ContactBlockSize, 0, stream>>>(
        deviceRanges.get(), deviceActiveRegions.get(), uint32_t(regions.size()),
        deviceBinCounts.get(), deviceOverflow.get());
    if (!rvdHost::checkLaunch("launch countSpatialHashEntriesKernel") ||
        !rvdHost::exclusiveSum(
            deviceBinCounts.get(), deviceBinOffsets.get(),
            int(regions.size() + 1), binScanScratch, stream)) {
      return false;
    }
    uint32_t overflow = 0, entryCount = 0;
    if (!rvdHost::readAt(overflow, deviceOverflow.get(), 0, stream, "read contact bin overflow") ||
        !rvdHost::readAt(entryCount, deviceBinOffsets.get(), regions.size(), stream, "read contact bin entry count")) {
      return false;
    }
    if (overflow != 0) return false;
    if (entryCount == 0) return true;

    cudaUtils::CudaBuffer<BinEntry> deviceEntries;
    if (!deviceEntries.allocate(entryCount)) return false;
    emitSpatialHashEntriesKernel<<<
        rvdHost::blockCount(uint32_t(regions.size()), ContactBlockSize),
        ContactBlockSize, 0, stream>>>(
        deviceRanges.get(), deviceActiveRegions.get(), uint32_t(regions.size()),
        deviceBinOffsets.get(), dimX, dimY, deviceEntries.get());
    if (!rvdHost::checkLaunch("launch emitSpatialHashEntriesKernel")) return false;
    thrust::sort(
        thrust::cuda::par.on(stream),
        thrust::device_ptr<BinEntry>(deviceEntries.get()),
        thrust::device_ptr<BinEntry>(deviceEntries.get()) + entryCount,
        BinEntryLess());

    cudaUtils::CudaBuffer<uint32_t> devicePairCounts, devicePairOffsets;
    cudaUtils::CudaBuffer<uint8_t> pairScratch;
    if (!devicePairCounts.allocate(size_t(entryCount) + 1) ||
        !devicePairOffsets.allocate(size_t(entryCount) + 1) ||
        !rvdHost::zero(deviceOverflow.get(), sizeof(uint32_t), stream,
                       "clear contact pair overflow") ||
        !rvdHost::zero(
            devicePairCounts.get() + entryCount, sizeof(uint32_t), stream,
            "clear contact pair count tail")) {
      return false;
    }
    countPairsInBinsKernel<<<
        rvdHost::blockCount(entryCount, ContactBlockSize), ContactBlockSize, 0, stream>>>(
        deviceEntries.get(), entryCount, deviceRegionModels.get(),
        deviceRegionBoundsMin.get(), deviceRegionBoundsMax.get(),
        contactRange, devicePairCounts.get(), deviceOverflow.get());
    if (!rvdHost::checkLaunch("launch countPairsInBinsKernel") ||
        !rvdHost::exclusiveSum(devicePairCounts.get(), devicePairOffsets.get(), int(entryCount + 1), pairScratch, stream)) {
      return false;
    }
    uint32_t pairOverflow = 0, pairCount = 0;
    if (!rvdHost::readAt(pairOverflow, deviceOverflow.get(), 0, stream, "read contact pair overflow") ||
        !rvdHost::readAt(pairCount, devicePairOffsets.get(), entryCount, stream, "read contact pair count")) {
      return false;
    }
    if (pairOverflow != 0) return false;
    if (pairCount == 0) return true;

    cudaUtils::CudaBuffer<uint64_t> devicePairKeys;
    if (!devicePairKeys.allocate(pairCount)) return false;
    emitPairsFromBinsKernel<<<
        rvdHost::blockCount(entryCount, ContactBlockSize), ContactBlockSize, 0, stream>>>(
        deviceEntries.get(), entryCount, deviceRegionModels.get(),
        deviceRegionBoundsMin.get(), deviceRegionBoundsMax.get(),
        contactRange, devicePairOffsets.get(), devicePairKeys.get());
    if (!rvdHost::checkLaunch("launch emitPairsFromBinsKernel")) return false;
    thrust::sort(
        thrust::cuda::par.on(stream),
        thrust::device_ptr<uint64_t>(devicePairKeys.get()),
        thrust::device_ptr<uint64_t>(devicePairKeys.get()) + pairCount);
    const auto uniqueEnd = thrust::unique(
        thrust::cuda::par.on(stream),
        thrust::device_ptr<uint64_t>(devicePairKeys.get()),
        thrust::device_ptr<uint64_t>(devicePairKeys.get()) + pairCount);
    const uint32_t uniquePairCount = uint32_t(uniqueEnd - thrust::device_ptr<uint64_t>(devicePairKeys.get()));
    std::vector<uint64_t> hostPairKeys(uniquePairCount);
    if (!rvdHost::copyToHost(hostPairKeys.data(), devicePairKeys.get(), uniquePairCount, stream, "download contact pair keys")) {
      return false;
    }
    std::vector<uint2> hostPairs;
    hostPairs.reserve(uniquePairCount);
    for (uint64_t key : hostPairKeys) {
      hostPairs.push_back(make_uint2(uint32_t(key >> 32), uint32_t(key)));
    }

    std::vector<DeviceContactRegion> hostRegions(regions.size());
    std::vector<double3> hostVertices;
    for (uint32_t region : contactSurfaces) {
      DeviceContactRegion &device = hostRegions[region];
      const RegionPlane &plane = planes[region];
      device.centroid = make_double3(plane.centroid.x, plane.centroid.y, plane.centroid.z);
      device.normal = make_double3(plane.normal.x, plane.normal.y, plane.normal.z);
      device.u = make_double3(plane.u.x, plane.u.y, plane.u.z);
      device.v = make_double3(plane.v.x, plane.v.y, plane.v.z);
      device.boundsMin = make_double3(plane.minimum.x, plane.minimum.y, plane.minimum.z);
      device.boundsMax = make_double3(plane.maximum.x, plane.maximum.y, plane.maximum.z);
      device.vertexOffset = uint32_t(hostVertices.size());
      device.vertexCount = uint32_t(regions[region].polygon.size());
      device.faceId = regions[region].faceId;
      device.fragmentId = solidFragments[region];
      if (hostVertices.size() + device.vertexCount > UINT32_MAX) return false;
      for (const glm::dvec3 &point : regions[region].polygon) {
        hostVertices.push_back(make_double3(point.x, point.y, point.z));
      }
    }

    std::vector<uint32_t> scratchOffsets(hostPairs.size());
    std::vector<uint32_t> scratchCapacities(hostPairs.size());
    size_t totalScratch = 0;
    for (size_t i = 0; i < hostPairs.size(); ++i) {
      const uint32_t countA = hostRegions[hostPairs[i].x].vertexCount;
      const uint32_t countB = hostRegions[hostPairs[i].y].vertexCount;
      const uint64_t capacity = uint64_t(countA) + countB + 4;
      if (capacity > UINT32_MAX / 2 ||
          totalScratch > UINT32_MAX - 2 * capacity ||
          totalScratch > (SIZE_MAX / sizeof(double2)) - 2 * capacity) {
        return false;
      }
      scratchOffsets[i] = uint32_t(totalScratch);
      scratchCapacities[i] = uint32_t(capacity);
      totalScratch += size_t(2 * capacity);
    }

    cudaUtils::CudaBuffer<DeviceContactRegion> deviceRegions;
    cudaUtils::CudaBuffer<double3> deviceVertices;
    cudaUtils::CudaBuffer<uint2> devicePairs;
    cudaUtils::CudaBuffer<uint32_t> deviceScratchOffsets;
    cudaUtils::CudaBuffer<uint32_t> deviceScratchCapacities;
    cudaUtils::CudaBuffer<double2> deviceScratch;
    cudaUtils::CudaBuffer<RawCandidateOverlap> deviceRawCandidates;
    cudaUtils::CudaBuffer<uint32_t> deviceCandidateValid;
    cudaUtils::CudaBuffer<uint32_t> deviceCompactOffsets;
    cudaUtils::CudaBuffer<uint32_t> deviceUnresolvedReasons;
    cudaUtils::CudaBuffer<uint8_t> compactScanScratch;
    if (!deviceRegions.allocate(hostRegions.size()) ||
        !deviceRegions.uploadAsync(hostRegions.data(), hostRegions.size(), stream) ||
        !deviceVertices.allocate(hostVertices.size()) ||
        !deviceVertices.uploadAsync(hostVertices.data(), hostVertices.size(), stream) ||
        !devicePairs.allocate(hostPairs.size()) ||
        !devicePairs.uploadAsync(hostPairs.data(), hostPairs.size(), stream) ||
        !deviceScratchOffsets.allocate(scratchOffsets.size()) ||
        !deviceScratchOffsets.uploadAsync(scratchOffsets.data(), scratchOffsets.size(), stream) ||
        !deviceScratchCapacities.allocate(scratchCapacities.size()) ||
        !deviceScratchCapacities.uploadAsync(scratchCapacities.data(), scratchCapacities.size(), stream) ||
        !deviceScratch.allocate(totalScratch) ||
        !deviceRawCandidates.allocate(hostPairs.size()) ||
        !deviceCandidateValid.allocate(hostPairs.size() + 1) ||
        !deviceCompactOffsets.allocate(hostPairs.size() + 1) ||
        !deviceUnresolvedReasons.allocate(hostPairs.size()) ||
        !rvdHost::zero(deviceCandidateValid.get() + hostPairs.size(), sizeof(uint32_t), stream, "clear candidate valid tail")) {
      return false;
    }
    clipContactCandidatesKernel<<<
        rvdHost::blockCount(uint32_t(hostPairs.size()), ContactBlockSize),
        ContactBlockSize, 0, stream>>>(
        devicePairs.get(), uint32_t(hostPairs.size()), deviceRegions.get(),
        deviceVertices.get(), deviceScratchOffsets.get(),
        deviceScratchCapacities.get(), deviceScratch.get(), contactRange,
        numerical, deviceRawCandidates.get(), deviceCandidateValid.get(),
        deviceUnresolvedReasons.get());
    if (!rvdHost::checkLaunch("launch clipContactCandidatesKernel")) return false;

    std::vector<uint32_t> hostUnresolvedReasons(hostPairs.size());
    if (!rvdHost::copyToHost(
            hostUnresolvedReasons.data(), deviceUnresolvedReasons.get(),
            hostPairs.size(), stream, "download unresolved contact reasons")) {
      return false;
    }
    if (reportUnresolvedContacts(hostUnresolvedReasons)) {
      return false;
    }

    if (!rvdHost::exclusiveSum(
            deviceCandidateValid.get(), deviceCompactOffsets.get(),
            int(hostPairs.size() + 1), compactScanScratch, stream)) {
      return false;
    }

    uint32_t acceptedOverlapCount = 0;
    if (!rvdHost::readAt(
            acceptedOverlapCount, deviceCompactOffsets.get(), hostPairs.size(),
            stream, "read accepted overlap count")) {
      return false;
    }

    if (acceptedOverlapCount == 0) return true;

    // gather candidate vertex counts
    cudaUtils::CudaBuffer<uint32_t> deviceAcceptedVertexCounts;
    cudaUtils::CudaBuffer<uint32_t> deviceAcceptedVertexOffsets;
    if (!deviceAcceptedVertexCounts.allocate(acceptedOverlapCount + 1) ||
        !deviceAcceptedVertexOffsets.allocate(acceptedOverlapCount + 1) ||
        !rvdHost::zero(
            deviceAcceptedVertexCounts.get() + acceptedOverlapCount, sizeof(uint32_t), stream,
            "clear accepted vertex count tail")) {
      return false;
    }
    gatherAcceptedVertexCountsKernel<<<
        rvdHost::blockCount(uint32_t(hostPairs.size()), ContactBlockSize),
        ContactBlockSize, 0, stream>>>(
        uint32_t(hostPairs.size()), deviceCandidateValid.get(),
        deviceCompactOffsets.get(), deviceRawCandidates.get(),
        deviceAcceptedVertexCounts.get());
    if (!rvdHost::checkLaunch("launch gatherAcceptedVertexCountsKernel")) return false;

    if (!rvdHost::exclusiveSum(
            deviceAcceptedVertexCounts.get(), deviceAcceptedVertexOffsets.get(),
            int(acceptedOverlapCount + 1), compactScanScratch, stream)) {
      return false;
    }

    uint32_t totalCompactVertices = 0;
    if (!rvdHost::readAt(
            totalCompactVertices, deviceAcceptedVertexOffsets.get(), acceptedOverlapCount,
            stream, "read total compact vertices")) {
      return false;
    }

    cudaUtils::CudaBuffer<CompactContactOverlap> deviceCompactOverlaps;
    cudaUtils::CudaBuffer<double2> deviceCompactVertices;
    if (!deviceCompactOverlaps.allocate(acceptedOverlapCount) ||
        !deviceCompactVertices.allocate(totalCompactVertices)) {
      return false;
    }

    compactAcceptedOverlapsKernel<<<
        rvdHost::blockCount(uint32_t(hostPairs.size()), ContactBlockSize),
        ContactBlockSize, 0, stream>>>(
        devicePairs.get(), uint32_t(hostPairs.size()), deviceRegions.get(),
        deviceRawCandidates.get(), deviceScratch.get(),
        deviceCandidateValid.get(), deviceCompactOffsets.get(),
        deviceAcceptedVertexOffsets.get(),
        deviceCompactOverlaps.get(), deviceCompactVertices.get());
    if (!rvdHost::checkLaunch("launch compactAcceptedOverlapsKernel")) return false;

    // build face-to-candidate inverted index
    const uint32_t instanceCount = acceptedOverlapCount * 2;
    cudaUtils::CudaBuffer<uint32_t> deviceInstanceFaceIds;
    cudaUtils::CudaBuffer<uint32_t> deviceInstanceCandData;
    cudaUtils::CudaBuffer<uint32_t> deviceSortedFaceIds;
    cudaUtils::CudaBuffer<uint32_t> deviceSortedCandData;
    cudaUtils::CudaBuffer<uint8_t> sortScratch;

    if (!deviceInstanceFaceIds.allocate(instanceCount) ||
        !deviceInstanceCandData.allocate(instanceCount) ||
        !deviceSortedFaceIds.allocate(instanceCount) ||
        !deviceSortedCandData.allocate(instanceCount)) {
      return false;
    }

    emitCandidateInstancesKernel<<<
        rvdHost::blockCount(acceptedOverlapCount, ContactBlockSize),
        ContactBlockSize, 0, stream>>>(
        deviceCompactOverlaps.get(), acceptedOverlapCount,
        deviceInstanceFaceIds.get(), deviceInstanceCandData.get());
    if (!rvdHost::checkLaunch("launch emitCandidateInstancesKernel")) return false;

    if (!rvdHost::sortPairs(
            deviceInstanceFaceIds.get(), deviceSortedFaceIds.get(),
            deviceInstanceCandData.get(), deviceSortedCandData.get(),
            int(instanceCount), sortScratch, stream)) {
      return false;
    }

    // resolve mortar couplings
    cudaUtils::CudaBuffer<double> deviceResolvedAreas;
    cudaUtils::CudaBuffer<uint32_t> deviceResolvedValid;
    cudaUtils::CudaBuffer<uint32_t> deviceResolvedOffsets;
    if (!deviceResolvedAreas.allocate(acceptedOverlapCount) ||
        !deviceResolvedValid.allocate(acceptedOverlapCount + 1) ||
        !deviceResolvedOffsets.allocate(acceptedOverlapCount + 1) ||
        !rvdHost::zero(
            deviceResolvedValid.get(),
            (acceptedOverlapCount + 1) * sizeof(uint32_t), stream,
            "clear resolved valid")) {
      return false;
    }

    resolveMortarCouplingsKernel<<<
        rvdHost::blockCount(acceptedOverlapCount, ContactBlockSize),
        ContactBlockSize, 0, stream>>>(
        deviceCompactOverlaps.get(), deviceCompactVertices.get(),
        acceptedOverlapCount,
        deviceRegions.get(),
        deviceSortedFaceIds.get(), deviceSortedCandData.get(),
        instanceCount,
        deviceResolvedAreas.get(), deviceResolvedValid.get());
    if (!rvdHost::checkLaunch("launch resolveMortarCouplingsKernel")) return false;

    if (!rvdHost::exclusiveSum(
            deviceResolvedValid.get(), deviceResolvedOffsets.get(),
            int(acceptedOverlapCount + 1), sortScratch, stream)) {
      return false;
    }

    uint32_t finalCouplingCount = 0;
    if (!rvdHost::readAt(
            finalCouplingCount, deviceResolvedOffsets.get(), acceptedOverlapCount,
            stream, "read final coupling count")) {
      return false;
    }

    if (finalCouplingCount == 0) return true;

    cudaUtils::CudaBuffer<GpuContactCoupling> deviceFinalCouplings;
    if (!deviceFinalCouplings.allocate(finalCouplingCount) ||
        !rvdHost::zero(
            deviceFinalCouplings.get(),
            finalCouplingCount * sizeof(GpuContactCoupling), stream,
            "clear final couplings")) {
      return false;
    }

    emitFinalCouplingsKernel<<<
        rvdHost::blockCount(acceptedOverlapCount, ContactBlockSize),
        ContactBlockSize, 0, stream>>>(
        deviceCompactOverlaps.get(), deviceResolvedAreas.get(),
        deviceResolvedValid.get(), deviceResolvedOffsets.get(),
        acceptedOverlapCount, deviceFinalCouplings.get());
    if (!rvdHost::checkLaunch("launch emitFinalCouplingsKernel")) return false;

    if (!rvdHost::synchronize(stream, "finish emitFinalCouplingsKernel")) {
      return false;
    }

    std::vector<GpuContactCoupling> hostFinalCouplings(finalCouplingCount);
    if (!rvdHost::copyToHost(
            hostFinalCouplings.data(), deviceFinalCouplings.get(),
            finalCouplingCount, stream, "download final contact couplings")) {
      return false;
    }

    couplings.reserve(couplings.size() + finalCouplingCount);
    for (const auto &fc : hostFinalCouplings) {
      if (fc.faceA == InvalidId || fc.faceB == InvalidId ||
          fc.faceA == fc.faceB || !(fc.area > 0.0) || !std::isfinite(fc.area)) {
        continue;
      }
      ContactCoupling coupling{};
      coupling.faceA = fc.faceA;
      coupling.fragmentA = fc.fragmentA;
      coupling.faceB = fc.faceB;
      coupling.fragmentB = fc.fragmentB;
      coupling.couplingArea = fc.area;
      couplings.push_back(coupling);
    }
  }

  return true;
}

} // namespace voronoi
