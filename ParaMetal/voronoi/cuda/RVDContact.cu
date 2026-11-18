#include "RVDContact.cuh"
#include "RVDHostUtils.cuh"
#include "RVDPredicates.cuh"
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
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

namespace voronoi {

constexpr uint32_t ContactBlockSize = rvdHost::BlockSize;
constexpr double NumericalTolerance = 256.0 * DBL_EPSILON;

struct GridCoord {
  int64_t x = 0;
  int64_t y = 0;
  int64_t z = 0;
};

struct SpatialHashEntry {
  uint64_t key;
  uint32_t model;
  uint32_t region;

  __host__ __device__ bool operator<(const SpatialHashEntry &other) const {
    if (key != other.key) return key < other.key;
    if (model != other.model) return model < other.model;
    return region < other.region;
  }
};

struct SpatialHashRange {
  GridCoord minimum;
  GridCoord maximum;
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
  uint32_t modelIndex;
};

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

struct ContactOverlapGeometry {
  uint32_t vertexCount;
  double projectionScale;
};

struct HalfPlane2D {
  double a;
  double b;
  double c;
};

constexpr int MaxLocalVertices = 32;
constexpr int MaxPieces = 64;
constexpr uint32_t MortarChunkSize = 1024;

struct LocalPolygon {
  int count;
  double2 v[MaxLocalVertices];
};

struct FaceCandidateIndex {
  const uint32_t *sortedFaceIds;
  const uint32_t *sortedCandData;
  uint32_t instanceCount;
};

enum SubtractionStatus {
  SubtractionSuccess = 0,
  SubtractionPieceCapacity = 1,
  SubtractionVertexCapacity = 2,
  SubtractionUnresolvedArea = 3,
};

enum ContactUnresolvedReason : uint32_t {
  ContactResolved = 0,
  ContactInitialClipCapacity = 1,
  ContactPenetrationClipCapacity = 2,
  ContactRangeClipCapacity = 3,
  ContactNonFiniteArea = 4,
  ContactAreaCancellation = 5,
  ContactMortarVertexCapacity = 6,
  ContactMortarProjection = 7,
  ContactMortarNonFiniteArea = 8,
  ContactMortarPieceCapacity = 9,
  ContactMortarPieceArea = 10,
  ContactMortarTotalArea = 11,
};

static __device__ inline void recordContactError(
    uint64_t *errorRecord,
    uint32_t index,
    uint32_t reason) {
  atomicMin(errorRecord, (uint64_t(index) << 32) | uint64_t(reason));
}

static __device__ inline double2 projectToFrame(
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

static __device__ inline double3 unprojectToWorld(
    const double2 &local,
    const DeviceContactRegion &frame) {
  return make_double3(
      frame.centroid.x + local.x * frame.u.x + local.y * frame.v.x,
      frame.centroid.y + local.x * frame.u.y + local.y * frame.v.y,
      frame.centroid.z + local.x * frame.u.z + local.y * frame.v.z);
}

static __device__ inline bool projectAlongNormal(
    const double3 &point,
    const double3 &projNormal,
    const DeviceContactRegion &dstFrame,
    double2 &projected) {
  const double denom =
      dstFrame.normal.x * projNormal.x +
      dstFrame.normal.y * projNormal.y +
      dstFrame.normal.z * projNormal.z;
  if (!(fabs(denom) > NumericalTolerance)) {
    return false;
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
  projected = projectToFrame(pDst, dstFrame);
  return true;
}

static __device__ inline double regionExtent(const DeviceContactRegion &region) {
  const double dx = region.boundsMax.x - region.boundsMin.x;
  const double dy = region.boundsMax.y - region.boundsMin.y;
  const double dz = region.boundsMax.z - region.boundsMin.z;
  return sqrt(dx * dx + dy * dy + dz * dz);
}

static __device__ inline double projectedPolygonSignedDoubleArea(
    const DeviceContactRegion &polygon,
    const DeviceContactRegion &frame,
    const double3 *vertices) {
  if (polygon.vertexCount < 3) return 0.0;
  const double2 origin = projectToFrame(vertices[polygon.vertexOffset], frame);
  double area = 0.0;
  double compensation = 0.0;
  for (uint32_t i = 1; i + 1 < polygon.vertexCount; ++i) {
    const double2 pointA = projectToFrame(vertices[polygon.vertexOffset + i], frame);
    const double2 pointB = projectToFrame(vertices[polygon.vertexOffset + i + 1], frame);
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

static __device__ inline void evaluatePolygon2DArea(
    const double2 *v,
    int count,
    double &signedDoubleArea,
    double &absoluteDoubleArea) {
  signedDoubleArea = 0.0;
  absoluteDoubleArea = 0.0;
  if (count < 3) return;

  const double2 origin = v[0];
  double compensation = 0.0;
  for (int i = 1; i + 1 < count; ++i) {
    const double ax = v[i].x - origin.x;
    const double ay = v[i].y - origin.y;
    const double bx = v[i + 1].x - origin.x;
    const double by = v[i + 1].y - origin.y;
    const double crossTerm = ax * by - bx * ay;
    const double adjusted = crossTerm - compensation;
    const double nextArea = signedDoubleArea + adjusted;
    compensation = (nextArea - signedDoubleArea) - adjusted;
    signedDoubleArea = nextArea;
    absoluteDoubleArea += fabs(crossTerm);
  }
}

static __device__ inline void ensurePolygonCCW(double2 *v, int count) {
  if (count < 3) return;
  double signedDoubleArea = 0.0;
  double absoluteDoubleArea = 0.0;
  evaluatePolygon2DArea(v, count, signedDoubleArea, absoluteDoubleArea);
  if (signedDoubleArea < 0.0) {
    for (int i = 0; i < count / 2; ++i) {
      double2 tmp = v[i];
      v[i] = v[count - 1 - i];
      v[count - 1 - i] = tmp;
    }
  }
}

static __device__ inline bool withinMergeTolerance(
    const double2 &lhs,
    const double2 &rhs,
    double mergeTolerance) {
  return fabs(lhs.x - rhs.x) <= mergeTolerance &&
         fabs(lhs.y - rhs.y) <= mergeTolerance;
}

static __device__ inline int clipHalfPlane(
    const double2 *input,
    int inputCount,
    HalfPlane2D plane,
    double2 *output,
    uint32_t capacity,
    double mergeTolerance) {
  if (inputCount < 3) return 0;
  int outputCount = 0;
  const auto append = [&](const double2 &point) -> bool {
    if (outputCount > 0 &&
        withinMergeTolerance(point, output[outputCount - 1], mergeTolerance)) {
      return true;
    }
    if (uint32_t(outputCount) >= capacity) return false;
    output[outputCount++] = point;
    return true;
  };
  double2 previous = input[inputCount - 1];
  double previousDistance = plane.a * previous.x + plane.b * previous.y + plane.c;
  for (int i = 0; i < inputCount; ++i) {
    const double2 current = input[i];
    const double currentDistance = plane.a * current.x + plane.b * current.y + plane.c;
    if ((previousDistance < 0.0 && currentDistance > 0.0) ||
        (previousDistance > 0.0 && currentDistance < 0.0)) {
      const double t = previousDistance / (previousDistance - currentDistance);
      const double2 intersection = make_double2(
          previous.x + t * (current.x - previous.x),
          previous.y + t * (current.y - previous.y));
      if (!append(intersection)) return -1;
    }
    if (currentDistance >= 0.0 && !append(current)) return -1;
    previous = current;
    previousDistance = currentDistance;
  }
  while (outputCount > 1 &&
         withinMergeTolerance(output[outputCount - 1], output[0], mergeTolerance)) {
    --outputCount;
  }
  return outputCount;
}

static __device__ inline uint32_t binarySearchLower(
    const uint32_t *array, uint32_t size, uint32_t key) {
  uint32_t l = 0, r = size;
  while (l < r) {
    uint32_t m = l + (r - l) / 2;
    if (array[m] < key) l = m + 1;
    else r = m;
  }
  return l;
}

static __device__ inline uint32_t binarySearchUpper(
    const uint32_t *array, uint32_t size, uint32_t key) {
  uint32_t l = 0, r = size;
  while (l < r) {
    uint32_t m = l + (r - l) / 2;
    if (array[m] <= key) l = m + 1;
    else r = m;
  }
  return l;
}

static __device__ inline bool contactAabbWithinRange(
    const double3 &minA, const double3 &maxA,
    const double3 &minB, const double3 &maxB,
    double contactRange) {
  const double dx = fmax(0.0, fmax(minA.x - maxB.x, minB.x - maxA.x));
  const double dy = fmax(0.0, fmax(minA.y - maxB.y, minB.y - maxA.y));
  const double dz = fmax(0.0, fmax(minA.z - maxB.z, minB.z - maxA.z));
  return (dx * dx + dy * dy + dz * dz) <= (contactRange * contactRange);
}

static __global__ void countSpatialHashEntriesKernel(
    const SpatialHashRange *ranges,
    uint32_t regionCount,
    uint32_t *counts,
    uint32_t *overflow) {
  const uint32_t region = blockIdx.x * blockDim.x + threadIdx.x;
  if (region >= regionCount) return;
  counts[region] = 0;
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

static __global__ void emitSpatialHashEntriesKernel(
    const SpatialHashRange *ranges,
    uint32_t regionCount,
    const uint32_t *offsets,
    const DeviceContactRegion *regions,
    uint64_t dimX,
    uint64_t dimY,
    SpatialHashEntry *entries) {
  const uint32_t region = blockIdx.x * blockDim.x + threadIdx.x;
  if (region >= regionCount) return;
  const SpatialHashRange range = ranges[region];
  const uint32_t model = regions[region].modelIndex;
  uint32_t destination = offsets[region];
  for (int64_t z = range.minimum.z; z <= range.maximum.z; ++z) {
    for (int64_t y = range.minimum.y; y <= range.maximum.y; ++y) {
      for (int64_t x = range.minimum.x; x <= range.maximum.x; ++x) {
        const uint64_t key =
            (uint64_t(z) * dimY + uint64_t(y)) * dimX + uint64_t(x);
        entries[destination++] = {key, model, region};
      }
    }
  }
}

static __global__ void countPairsInSpatialEntriesKernel(
    const SpatialHashEntry *entries,
    uint32_t entryCount,
    const DeviceContactRegion *regions,
    double contactRange,
    uint32_t *pairCounts,
    uint32_t *overflow) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= entryCount) return;
  pairCounts[index] = 0;
  if (index > 0 && entries[index - 1].key == entries[index].key &&
      entries[index - 1].model == entries[index].model) {
    return;
  }

  uint32_t left = index + 1;
  uint32_t right = entryCount;
  while (left < right) {
    const uint32_t middle = left + (right - left) / 2;
    if (entries[middle].key <= entries[index].key) left = middle + 1;
    else right = middle;
  }
  uint32_t runEnd = index + 1;
  while (runEnd < left && entries[runEnd].model == entries[index].model) {
    ++runEnd;
  }

  uint64_t count = 0;
  for (uint32_t a = index; a < runEnd; ++a) {
    const DeviceContactRegion &regionA = regions[entries[a].region];
    for (uint32_t b = runEnd; b < left; ++b) {
      const DeviceContactRegion &regionB = regions[entries[b].region];
      if (contactAabbWithinRange(
              regionA.boundsMin, regionA.boundsMax,
              regionB.boundsMin, regionB.boundsMax, contactRange)) {
        ++count;
      }
    }
  }
  if (count > UINT32_MAX) {
    atomicExch(overflow, 1u);
    return;
  }
  pairCounts[index] = uint32_t(count);
}

static __global__ void emitPairsFromSpatialEntriesKernel(
    const SpatialHashEntry *entries,
    uint32_t entryCount,
    const DeviceContactRegion *regions,
    double contactRange,
    const uint32_t *pairOffsets,
    uint64_t *pairKeys) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= entryCount) return;
  if (index > 0 && entries[index - 1].key == entries[index].key &&
      entries[index - 1].model == entries[index].model) {
    return;
  }

  uint32_t left = index + 1;
  uint32_t right = entryCount;
  while (left < right) {
    const uint32_t middle = left + (right - left) / 2;
    if (entries[middle].key <= entries[index].key) left = middle + 1;
    else right = middle;
  }
  uint32_t runEnd = index + 1;
  while (runEnd < left && entries[runEnd].model == entries[index].model) {
    ++runEnd;
  }
  uint32_t output = pairOffsets[index];
  for (uint32_t a = index; a < runEnd; ++a) {
    const uint32_t regA = entries[a].region;
    const DeviceContactRegion &regionA = regions[regA];
    for (uint32_t b = runEnd; b < left; ++b) {
      const uint32_t regB = entries[b].region;
      const DeviceContactRegion &regionB = regions[regB];
      if (!contactAabbWithinRange(
              regionA.boundsMin, regionA.boundsMax,
              regionB.boundsMin, regionB.boundsMax, contactRange)) continue;
      const uint32_t first = min(regA, regB);
      const uint32_t second = max(regA, regB);
      pairKeys[output++] = (uint64_t(first) << 32) | uint64_t(second);
    }
  }
}

static bool gridCoordinate(double value, double minimum, double cellSize, int64_t &result) {
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

static bool spatialHashRange(
    const DeviceContactRegion &region,
    double tolerance,
    const glm::dvec3 &origin,
    double cellSize,
    SpatialHashRange &range) {
  return gridCoordinate(
             region.boundsMin.x - tolerance, origin.x, cellSize,
             range.minimum.x) &&
         gridCoordinate(
             region.boundsMin.y - tolerance, origin.y, cellSize,
             range.minimum.y) &&
         gridCoordinate(
             region.boundsMin.z - tolerance, origin.z, cellSize,
             range.minimum.z) &&
         gridCoordinate(
             region.boundsMax.x + tolerance, origin.x, cellSize,
             range.maximum.x) &&
         gridCoordinate(
             region.boundsMax.y + tolerance, origin.y, cellSize,
             range.maximum.y) &&
         gridCoordinate(
             region.boundsMax.z + tolerance, origin.z, cellSize,
             range.maximum.z);
}

static __device__ inline bool convexOverlapInContactRange(
    const DeviceContactRegion &a,
    const DeviceContactRegion &b,
    const double3 *vertices,
    double contactRange,
    double penetrationTolerance,
    double2 *scratch,
    uint32_t capacity,
    double &area,
    double2* &outSource,
    uint32_t &unresolvedReason,
    ContactOverlapGeometry &geometry) {
  area = 0.0;
  outSource = nullptr;
  unresolvedReason = ContactResolved;
  geometry = ContactOverlapGeometry{};
  const double normalDot =
      a.normal.x * b.normal.x + a.normal.y * b.normal.y + a.normal.z * b.normal.z;
  if (normalDot > -0.2) return false;
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

  const double mergeTolerance = NumericalTolerance * regionExtent(a);
  int count = int(a.vertexCount);
  const double signedA = projectedPolygonSignedDoubleArea(a, a, vertices);
  for (uint32_t i = 0; i < a.vertexCount; ++i) {
    const uint32_t source = signedA >= 0.0 ? i : (a.vertexCount - i) % a.vertexCount;
    scratch[i] = projectToFrame(vertices[a.vertexOffset + source], a);
  }

  const double signedB = projectedPolygonSignedDoubleArea(b, a, vertices);
  const bool bCounterClockwise = signedB >= 0.0;
  double2 *source = scratch;
  double2 *destination = scratch + capacity;
  for (uint32_t edge = 0; edge < b.vertexCount; ++edge) {
    const uint32_t ia = bCounterClockwise ? edge : (b.vertexCount - edge) % b.vertexCount;
    const uint32_t ib = bCounterClockwise ? (edge + 1) % b.vertexCount : (b.vertexCount - edge - 1) % b.vertexCount;
    const double2 p0 = projectToFrame(vertices[b.vertexOffset + ia], a);
    const double2 p1 = projectToFrame(vertices[b.vertexOffset + ib], a);
    const double lineA = p0.y - p1.y;
    const double lineB = p1.x - p0.x;
    const double lineC = p1.y * p0.x - p1.x * p0.y;
    const int clipped = clipHalfPlane(
        source, count, HalfPlane2D{lineA, lineB, lineC}, destination, capacity,
        mergeTolerance);
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

  const double gap0 =
      (b.normal.x * (b.centroid.x - a.centroid.x) +
       b.normal.y * (b.centroid.y - a.centroid.y) +
       b.normal.z * (b.centroid.z - a.centroid.z)) / normalDot;
  const double gapU = -(b.normal.x * a.u.x + b.normal.y * a.u.y + b.normal.z * a.u.z) / normalDot;
  const double gapV = -(b.normal.x * a.v.x + b.normal.y * a.v.y + b.normal.z * a.v.z) / normalDot;
  int clipped = clipHalfPlane(
      source, count, HalfPlane2D{gapU, gapV, gap0 + penetrationTolerance},
      destination, capacity, mergeTolerance);
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
  clipped = clipHalfPlane(
      source, count, HalfPlane2D{-gapU, -gapV, contactRange - gap0}, destination,
      capacity, mergeTolerance);
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
  evaluatePolygon2DArea(source, count, signedDoubleArea, absoluteDoubleArea);

  const double localArea = 0.5 * fabs(signedDoubleArea);
  geometry.vertexCount = uint32_t(count);
  geometry.projectionScale = fabs(a.normal.x * mortarNormal.x + a.normal.y * mortarNormal.y + a.normal.z * mortarNormal.z);
  area = localArea * geometry.projectionScale;
  outSource = source;
  if (!isfinite(area) || !isfinite(absoluteDoubleArea) ||
      !isfinite(geometry.projectionScale)) {
    unresolvedReason = ContactNonFiniteArea;
    return false;
  }
  if (area == 0.0) return false;
  if (localArea <= NumericalTolerance * (0.5 * absoluteDoubleArea)) {
    unresolvedReason = ContactAreaCancellation;
    return false;
  }
  return true;
}

static __global__ void countPairScratchCapacitiesKernel(
    const uint64_t *pairKeys,
    uint32_t pairCount,
    const DeviceContactRegion *regions,
    uint32_t *capacities,
    uint32_t *overflow) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pairCount) return;
  const uint32_t regionA = uint32_t(pairKeys[index] >> 32);
  const uint32_t regionB = uint32_t(pairKeys[index]);
  const uint64_t capacity =
      uint64_t(regions[regionA].vertexCount) +
      uint64_t(regions[regionB].vertexCount) + 4;
  if (capacity > UINT32_MAX) {
    atomicExch(overflow, 1u);
    capacities[index] = 0;
    return;
  }
  capacities[index] = uint32_t(capacity);
}

static __global__ void clipContactCandidatesKernel(
    const uint64_t *pairKeys,
    uint32_t pairCount,
    const DeviceContactRegion *regions,
    const double3 *vertices,
    const uint32_t *scratchOffsets,
    double2 *scratch,
    double contactRange,
    double penetrationTolerance,
    RawCandidateOverlap *rawCandidates,
    uint32_t *valid,
    uint64_t *errorRecord) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pairCount) return;
  valid[index] = 0;

  const uint32_t regionA = uint32_t(pairKeys[index] >> 32);
  const uint32_t regionB = uint32_t(pairKeys[index]);
  const DeviceContactRegion &a = regions[regionA];
  const DeviceContactRegion &b = regions[regionB];
  if (a.faceId == InvalidId || b.faceId == InvalidId || a.faceId == b.faceId) {
    return;
  }

  const uint32_t offset = 2u * scratchOffsets[index];
  const uint32_t capacity = scratchOffsets[index + 1] - scratchOffsets[index];
  double area = 0.0;
  double2 *outSource = nullptr;
  uint32_t unresolvedReason = ContactResolved;
  ContactOverlapGeometry geometry{};
  if (!convexOverlapInContactRange(
          a, b, vertices, contactRange, penetrationTolerance,
          scratch + offset, capacity, area,
          outSource, unresolvedReason, geometry)) {
    if (unresolvedReason != ContactResolved) {
      recordContactError(errorRecord, index, unresolvedReason);
    }
    return;
  }

  rawCandidates[index].projectionScale = geometry.projectionScale;
  rawCandidates[index].area = area;
  rawCandidates[index].vertexCount = geometry.vertexCount;
  rawCandidates[index].sourceScratchOffset = uint32_t(outSource - scratch);
  valid[index] = 1;
}

static __global__ void gatherAcceptedVertexCountsKernel(
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

static __global__ void compactAcceptedOverlapsKernel(
    const uint64_t *pairKeys,
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
  const uint32_t regionA = uint32_t(pairKeys[index] >> 32);
  const uint32_t regionB = uint32_t(pairKeys[index]);
  const DeviceContactRegion &a = regions[regionA];
  const DeviceContactRegion &b = regions[regionB];

  CompactContactOverlap compact{};
  compact.faceA = a.faceId;
  compact.fragmentA = a.fragmentId;
  compact.regionA = regionA;
  compact.faceB = b.faceId;
  compact.fragmentB = b.fragmentId;
  compact.regionB = regionB;
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

static __device__ inline SubtractionStatus subtractConvexObstacle(
    const double2 *obsVertices,
    int obsCount,
    LocalPolygon *piecesIn,
    int numPiecesIn,
    LocalPolygon *piecesOut,
    int &numPiecesOut,
    double mergeTolerance) {
  if (obsCount < 3) {
    if (numPiecesIn > MaxPieces) return SubtractionPieceCapacity;
    numPiecesOut = numPiecesIn;
    for (int i = 0; i < numPiecesIn; ++i) piecesOut[i] = piecesIn[i];
    return SubtractionSuccess;
  }
  if (obsCount > MaxLocalVertices) return SubtractionVertexCapacity;

  double2 obs[MaxLocalVertices];
  for (int i = 0; i < obsCount; ++i) obs[i] = obsVertices[i];
  ensurePolygonCCW(obs, obsCount);

  numPiecesOut = 0;
  for (int p = 0; p < numPiecesIn; ++p) {
    LocalPolygon currentInside = piecesIn[p];
    for (int e = 0; e < obsCount && currentInside.count >= 3; ++e) {
      int nextE = (e + 1 == obsCount) ? 0 : e + 1;
      double lineA = obs[e].y - obs[nextE].y;
      double lineB = obs[nextE].x - obs[e].x;
      double lineC = obs[nextE].y * obs[e].x - obs[nextE].x * obs[e].y;

      double2 outPoly[MaxLocalVertices];
      int outCount = clipHalfPlane(
          currentInside.v, currentInside.count, HalfPlane2D{-lineA, -lineB, -lineC},
          outPoly, MaxLocalVertices, mergeTolerance);
      if (outCount < 0) return SubtractionVertexCapacity;
      if (outCount >= 3) {
        bool allCollinear = true;
        for (int i = 2; i < outCount; ++i) {
          const rvdDevice::Orient2DResult orient =
              rvdDevice::orient2dFilteredDouble(outPoly[0], outPoly[1], outPoly[i]);
          if (orient.sign != 0) {
            allCollinear = false;
            break;
          }
        }
        if (!allCollinear) {
          double pieceDoubleArea = 0.0;
          double absoluteSum = 0.0;
          evaluatePolygon2DArea(outPoly, outCount, pieceDoubleArea, absoluteSum);
          const double roundoffBound = NumericalTolerance * absoluteSum;
          if (fabs(pieceDoubleArea) > roundoffBound) {
            if (numPiecesOut >= MaxPieces) return SubtractionPieceCapacity;
            piecesOut[numPiecesOut].count = outCount;
            for (int i = 0; i < outCount; ++i) {
              piecesOut[numPiecesOut].v[i] = outPoly[i];
            }
            numPiecesOut++;
          } else {
            return SubtractionUnresolvedArea;
          }
        }
      }

      double2 nextInside[MaxLocalVertices];
      int inCount = clipHalfPlane(
          currentInside.v, currentInside.count, HalfPlane2D{lineA, lineB, lineC},
          nextInside, MaxLocalVertices, mergeTolerance);
      if (inCount < 0) return SubtractionVertexCapacity;
      currentInside.count = inCount;
      for (int i = 0; i < inCount; ++i) currentInside.v[i] = nextInside[i];
    }
  }
  return SubtractionSuccess;
}

static __global__ void emitCandidateInstancesKernel(
    const CompactContactOverlap *compactOverlaps,
    uint32_t acceptedCount,
    uint32_t *instanceFaceIds,
    uint32_t *instanceCandData) {
  const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= acceptedCount) return;
  const CompactContactOverlap &cand = compactOverlaps[k];
  instanceFaceIds[2 * k] = cand.faceA;
  instanceCandData[2 * k] = k;
  instanceFaceIds[2 * k + 1] = cand.faceB;
  instanceCandData[2 * k + 1] = k;
}

static __global__ void resolveMortarCouplingsKernel(
    const CompactContactOverlap *compactOverlaps,
    const double2 *compactVertices,
    uint32_t chunkOffset,
    uint32_t chunkCount,
    const DeviceContactRegion *regions,
    FaceCandidateIndex candidateIndex,
    double numerical,
    LocalPolygon *mortarScratch,
    double *resolvedAreas,
    uint32_t *resolvedValid,
    uint64_t *errorRecord) {
  const uint32_t localK = blockIdx.x * blockDim.x + threadIdx.x;
  if (localK >= chunkCount) return;
  const uint32_t k = chunkOffset + localK;
  resolvedValid[k] = 0;
  resolvedAreas[k] = 0.0;

  const CompactContactOverlap &cand = compactOverlaps[k];

  const uint32_t startA = binarySearchLower(
      candidateIndex.sortedFaceIds, candidateIndex.instanceCount, cand.faceA);
  const uint32_t endA = binarySearchUpper(
      candidateIndex.sortedFaceIds, candidateIndex.instanceCount, cand.faceA);
  const uint32_t countA = endA - startA;

  const uint32_t startB = binarySearchLower(
      candidateIndex.sortedFaceIds, candidateIndex.instanceCount, cand.faceB);
  const uint32_t endB = binarySearchUpper(
      candidateIndex.sortedFaceIds, candidateIndex.instanceCount, cand.faceB);
  const uint32_t countB = endB - startB;

  if (countA <= 1 && countB <= 1) {
    resolvedAreas[k] = cand.area;
    resolvedValid[k] = (cand.area > 0.0) ? 1 : 0;
    return;
  }

  if (cand.vertexCount > uint32_t(MaxLocalVertices)) {
    recordContactError(errorRecord, k, ContactMortarVertexCapacity);
    return;
  }
  if (cand.vertexCount < 3) {
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

  const double3 cDiffAB = make_double3(C_B.x - C_A.x, C_B.y - C_A.y, C_B.z - C_A.z);
  const double candG0_A = (cDiffAB.x * n_B.x + cDiffAB.y * n_B.y + cDiffAB.z * n_B.z) / denomAB;
  const double candGU_A = -(u_A.x * n_B.x + u_A.y * n_B.y + u_A.z * n_B.z) / denomAB;
  const double candGV_A = -(v_A.x * n_B.x + v_A.y * n_B.y + v_A.z * n_B.z) / denomAB;

  const double3 cDiffBA = make_double3(C_A.x - C_B.x, C_A.y - C_B.y, C_A.z - C_B.z);
  const double candG0_B = (cDiffBA.x * n_A.x + cDiffBA.y * n_A.y + cDiffBA.z * n_A.z) / denomAB;
  const double candGU_B = -(u_B.x * n_A.x + u_B.y * n_A.y + u_B.z * n_A.z) / denomAB;
  const double candGV_B = -(v_B.x * n_A.x + v_B.y * n_A.y + v_B.z * n_A.z) / denomAB;

  const double extentA = regionExtent(regA);
  const double extentB = regionExtent(regB);
  const double mergeToleranceA = NumericalTolerance * extentA;
  const double mergeToleranceB = NumericalTolerance * extentB;

  LocalPolygon *piecesBufA = mortarScratch + size_t(localK) * (2 * MaxPieces);
  LocalPolygon *piecesBufB = piecesBufA + MaxPieces;
  LocalPolygon *currentPieces = piecesBufA;
  LocalPolygon *nextPieces = piecesBufB;

  int numCurrentPieces = 1;
  currentPieces[0].count = int(cand.vertexCount);
  for (int v = 0; v < currentPieces[0].count; ++v) {
    currentPieces[0].v[v] = compactVertices[cand.vertexOffset + v];
  }

  // process competitors on face A
  for (uint32_t i = startA; i < endA; ++i) {
    if (numCurrentPieces == 0) break;
    const uint32_t compCandIdx = candidateIndex.sortedCandData[i];
    if (compCandIdx == k) continue;
    const CompactContactOverlap &comp = compactOverlaps[compCandIdx];
    uint32_t partnerRegion = (comp.faceA == cand.faceA) ? comp.regionB : comp.regionA;
    uint32_t partnerFace = (comp.faceA == cand.faceA) ? comp.faceB : comp.faceA;
    const DeviceContactRegion &partner = regions[partnerRegion];
    if (comp.vertexCount > uint32_t(MaxLocalVertices)) {
      recordContactError(errorRecord, k, ContactMortarVertexCapacity);
      return;
    }

    const double denomComp = n_A.x * partner.normal.x + n_A.y * partner.normal.y + n_A.z * partner.normal.z;
    if (!(fabs(denomComp) > NumericalTolerance)) {
      recordContactError(errorRecord, k, ContactMortarProjection);
      return;
    }

    const double3 cDiff = make_double3(partner.centroid.x - C_A.x, partner.centroid.y - C_A.y, partner.centroid.z - C_A.z);
    const double compG0_A = (cDiff.x * partner.normal.x + cDiff.y * partner.normal.y + cDiff.z * partner.normal.z) / denomComp;
    const double compGU_A = -(u_A.x * partner.normal.x + u_A.y * partner.normal.y + u_A.z * partner.normal.z) / denomComp;
    const double compGV_A = -(v_A.x * partner.normal.x + v_A.y * partner.normal.y + v_A.z * partner.normal.z) / denomComp;

    const double lineA = candGU_A - compGU_A;
    const double lineB = candGV_A - compGV_A;
    const double lineC = candG0_A - compG0_A;
    const bool coincident = (fabs(lineA) + fabs(lineB)) * extentA + fabs(lineC) <= numerical;

    const DeviceContactRegion &compFrame = regions[comp.regionA];
    double2 compPoly[MaxLocalVertices];
    for (uint32_t v = 0; v < comp.vertexCount; ++v) {
      const double2 localV = compactVertices[comp.vertexOffset + v];
      const double3 p3d = unprojectToWorld(localV, compFrame);
      if (!projectAlongNormal(p3d, compFrame.normal, regA, compPoly[v])) {
        recordContactError(errorRecord, k, ContactMortarProjection);
        return;
      }
    }

    double2 obsPoly[MaxLocalVertices];
    int obsPolyCount = 0;
    if (coincident) {
      if (partnerFace > cand.faceB) continue;
      for (uint32_t v = 0; v < comp.vertexCount; ++v) obsPoly[v] = compPoly[v];
      obsPolyCount = int(comp.vertexCount);
    } else {
      obsPolyCount = clipHalfPlane(
          compPoly, int(comp.vertexCount), HalfPlane2D{lineA, lineB, lineC},
          obsPoly, MaxLocalVertices, mergeToleranceA);
      if (obsPolyCount < 0) {
        recordContactError(errorRecord, k, ContactMortarVertexCapacity);
        return;
      }
    }
    if (obsPolyCount >= 3) {
      int numNextPieces = 0;
      const SubtractionStatus subStatus = subtractConvexObstacle(
          obsPoly, obsPolyCount, currentPieces, numCurrentPieces,
          nextPieces, numNextPieces, mergeToleranceA);
      if (subStatus == SubtractionPieceCapacity) {
        recordContactError(errorRecord, k, ContactMortarPieceCapacity);
        return;
      }
      if (subStatus == SubtractionVertexCapacity) {
        recordContactError(errorRecord, k, ContactMortarVertexCapacity);
        return;
      }
      if (subStatus == SubtractionUnresolvedArea) {
        recordContactError(errorRecord, k, ContactMortarPieceArea);
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
    const uint32_t compCandIdx = candidateIndex.sortedCandData[i];
    if (compCandIdx == k) continue;
    const CompactContactOverlap &comp = compactOverlaps[compCandIdx];
    uint32_t partnerRegion = (comp.faceA == cand.faceB) ? comp.regionB : comp.regionA;
    uint32_t partnerFace = (comp.faceA == cand.faceB) ? comp.faceB : comp.faceA;
    const DeviceContactRegion &partner = regions[partnerRegion];
    if (comp.vertexCount > uint32_t(MaxLocalVertices)) {
      recordContactError(errorRecord, k, ContactMortarVertexCapacity);
      return;
    }

    const double denomComp = n_B.x * partner.normal.x + n_B.y * partner.normal.y + n_B.z * partner.normal.z;
    if (!(fabs(denomComp) > NumericalTolerance)) {
      recordContactError(errorRecord, k, ContactMortarProjection);
      return;
    }

    const double3 cDiff = make_double3(partner.centroid.x - C_B.x, partner.centroid.y - C_B.y, partner.centroid.z - C_B.z);
    const double compG0_B = (cDiff.x * partner.normal.x + cDiff.y * partner.normal.y + cDiff.z * partner.normal.z) / denomComp;
    const double compGU_B = -(u_B.x * partner.normal.x + u_B.y * partner.normal.y + u_B.z * partner.normal.z) / denomComp;
    const double compGV_B = -(v_B.x * partner.normal.x + v_B.y * partner.normal.y + v_B.z * partner.normal.z) / denomComp;

    const double lineA = candGU_B - compGU_B;
    const double lineB = candGV_B - compGV_B;
    const double lineC = candG0_B - compG0_B;
    const bool coincident = (fabs(lineA) + fabs(lineB)) * extentB + fabs(lineC) <= numerical;

    const DeviceContactRegion &compFrame = regions[comp.regionA];
    double2 compPoly[MaxLocalVertices];
    for (uint32_t v = 0; v < comp.vertexCount; ++v) {
      const double2 localV = compactVertices[comp.vertexOffset + v];
      const double3 p3d = unprojectToWorld(localV, compFrame);
      if (!projectAlongNormal(p3d, compFrame.normal, regB, compPoly[v])) {
        recordContactError(errorRecord, k, ContactMortarProjection);
        return;
      }
    }

    double2 clippedObs[MaxLocalVertices];
    int clippedObsCount = 0;
    if (coincident) {
      if (partnerFace > cand.faceA) continue;
      for (uint32_t v = 0; v < comp.vertexCount; ++v) clippedObs[v] = compPoly[v];
      clippedObsCount = int(comp.vertexCount);
    } else {
      clippedObsCount = clipHalfPlane(
          compPoly, int(comp.vertexCount), HalfPlane2D{lineA, lineB, lineC},
          clippedObs, MaxLocalVertices, mergeToleranceB);
      if (clippedObsCount < 0) {
        recordContactError(errorRecord, k, ContactMortarVertexCapacity);
        return;
      }
    }
    if (clippedObsCount >= 3) {
      double2 obsOnA[MaxLocalVertices];
      for (int v = 0; v < clippedObsCount; ++v) {
        const double3 p3d = unprojectToWorld(clippedObs[v], regB);
        if (!projectAlongNormal(p3d, n_A, regA, obsOnA[v])) {
          recordContactError(errorRecord, k, ContactMortarProjection);
          return;
        }
      }

      int numNextPieces = 0;
      const SubtractionStatus subStatus = subtractConvexObstacle(
          obsOnA, clippedObsCount, currentPieces, numCurrentPieces,
          nextPieces, numNextPieces, mergeToleranceA);
      if (subStatus == SubtractionPieceCapacity) {
        recordContactError(errorRecord, k, ContactMortarPieceCapacity);
        return;
      }
      if (subStatus == SubtractionVertexCapacity) {
        recordContactError(errorRecord, k, ContactMortarVertexCapacity);
        return;
      }
      if (subStatus == SubtractionUnresolvedArea) {
        recordContactError(errorRecord, k, ContactMortarPieceArea);
        return;
      }
      LocalPolygon *tmp = currentPieces;
      currentPieces = nextPieces;
      nextPieces = tmp;
      numCurrentPieces = numNextPieces;
    }
  }

  double netDoubleArea = 0.0;
  double netCompensation = 0.0;
  double absoluteDoubleArea = 0.0;
  for (int p = 0; p < numCurrentPieces; ++p) {
    const LocalPolygon &piece = currentPieces[p];
    double pieceDoubleArea = 0.0;
    double pieceAbsoluteSum = 0.0;
    evaluatePolygon2DArea(piece.v, piece.count, pieceDoubleArea, pieceAbsoluteSum);
    absoluteDoubleArea += pieceAbsoluteSum;
    const double adjusted = fabs(pieceDoubleArea) - netCompensation;
    const double nextArea = netDoubleArea + adjusted;
    netCompensation = (nextArea - netDoubleArea) - adjusted;
    netDoubleArea = nextArea;
  }
  const double localArea = 0.5 * netDoubleArea;
  const double finalMortarArea = localArea * cand.projectionScale;
  if (!isfinite(finalMortarArea) || !isfinite(absoluteDoubleArea)) {
    recordContactError(errorRecord, k, ContactMortarNonFiniteArea);
    return;
  }
  if (absoluteDoubleArea > 0.0 &&
      localArea <= NumericalTolerance * (0.5 * absoluteDoubleArea)) {
    recordContactError(errorRecord, k, ContactMortarTotalArea);
    return;
  }
  resolvedAreas[k] = finalMortarArea;
  resolvedValid[k] = (finalMortarArea > 0.0) ? 1 : 0;
}

static __global__ void emitFinalCouplingsKernel(
    const CompactContactOverlap *compactOverlaps,
    const double *resolvedAreas,
    const uint32_t *resolvedValid,
    const uint32_t *resolvedOffsets,
    uint32_t overlapCount,
    ContactCoupling *finalCouplings) {
  const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= overlapCount) return;
  if (resolvedValid[k] == 0) return;

  const uint32_t outIdx = resolvedOffsets[k];
  const CompactContactOverlap &cand = compactOverlaps[k];
  ContactCoupling coupling{};
  coupling.faceA = cand.faceA;
  coupling.fragmentA = cand.fragmentA;
  coupling.faceB = cand.faceB;
  coupling.fragmentB = cand.fragmentB;
  coupling.couplingArea = resolvedAreas[k];
  finalCouplings[outIdx] = coupling;
}

static bool buildContactRegion(
    const SurfaceRegion &region,
    uint32_t fragmentId,
    std::vector<double3> &vertices,
    DeviceContactRegion &descriptor) {
  descriptor = DeviceContactRegion{};
  if (!(region.area > 0.0) || !std::isfinite(region.area) || region.polygon.size() < 3) {
    return false;
  }
  const double normalLength = glm::length(region.normal);
  if (!(normalLength > 0.0) || !std::isfinite(normalLength)) {
    return false;
  }
  const glm::dvec3 centroid = region.moment / region.area;
  const glm::dvec3 normal = region.normal / normalLength;
  if (!std::isfinite(centroid.x) || !std::isfinite(centroid.y) || !std::isfinite(centroid.z)) {
    return false;
  }
  const glm::dvec3 u = rvdHost::makeBasisU(normal);
  const glm::dvec3 v = glm::cross(normal, u);

  glm::dvec3 minimum(std::numeric_limits<double>::max());
  glm::dvec3 maximum(-std::numeric_limits<double>::max());
  for (const glm::dvec3 &world : region.polygon) {
    if (!std::isfinite(world.x) || !std::isfinite(world.y) || !std::isfinite(world.z)) {
      return false;
    }
    minimum = glm::min(minimum, world);
    maximum = glm::max(maximum, world);
  }
  if (vertices.size() + region.polygon.size() > UINT32_MAX) return false;

  descriptor.centroid = make_double3(centroid.x, centroid.y, centroid.z);
  descriptor.normal = make_double3(normal.x, normal.y, normal.z);
  descriptor.u = make_double3(u.x, u.y, u.z);
  descriptor.v = make_double3(v.x, v.y, v.z);
  descriptor.boundsMin = make_double3(minimum.x, minimum.y, minimum.z);
  descriptor.boundsMax = make_double3(maximum.x, maximum.y, maximum.z);
  descriptor.vertexOffset = uint32_t(vertices.size());
  descriptor.vertexCount = uint32_t(region.polygon.size());
  descriptor.faceId = region.faceId;
  descriptor.fragmentId = fragmentId;
  descriptor.modelIndex = region.modelIndex;
  for (const glm::dvec3 &point : region.polygon) {
    vertices.push_back(make_double3(point.x, point.y, point.z));
  }
  return true;
}

static uint32_t solidFragmentForRegion(const SurfaceRegion &region, const RVDResult &rvd) {
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

static const char *contactReasonName(uint32_t reason) {
  switch (reason) {
    case ContactResolved: return "resolved";
    case ContactInitialClipCapacity: return "initial clip capacity";
    case ContactPenetrationClipCapacity: return "penetration clip capacity";
    case ContactRangeClipCapacity: return "range clip capacity";
    case ContactNonFiniteArea: return "non-finite area";
    case ContactAreaCancellation: return "area cancellation";
    case ContactMortarVertexCapacity: return "mortar vertex capacity";
    case ContactMortarProjection: return "mortar projection";
    case ContactMortarNonFiniteArea: return "mortar non-finite area";
    case ContactMortarPieceCapacity: return "mortar piece capacity";
    case ContactMortarPieceArea: return "mortar piece area";
    case ContactMortarTotalArea: return "mortar total area";
  }
  return "unknown";
}

static bool reportContactError(const cudaUtils::CudaBuffer<uint64_t> &errorRecord, cudaStream_t stream, const char *stage) {
  uint64_t record = 0;
  if (!rvdHost::readAt(record, errorRecord.get(), 0, stream, "read contact error")) {
    return true;
  }
  if (record == UINT64_MAX) return false;
  std::cerr << "[RVDContact] unresolved " << stage << " " << uint32_t(record >> 32) << ": " << contactReasonName(uint32_t(record)) << std::endl;
  return true;
}

static uint64_t contactTotal(const uint32_t *counts, size_t count, cudaStream_t stream) {
  return thrust::reduce(
      thrust::cuda::par.on(stream),
      thrust::device_ptr<const uint32_t>(counts),
      thrust::device_ptr<const uint32_t>(counts) + count,
      uint64_t(0),
      thrust::plus<uint64_t>());
}

static bool contactIndexRangeFits(uint64_t total, const char *label) {
  if (total < uint64_t(std::numeric_limits<int>::max())) return true;
  std::cerr << "[RVDContact] " << label << " " << total
            << " exceeds supported index range" << std::endl;
  return false;
}

static double computeDryContactGapTolerance(const BaseCellDomain &baseDomain) {
  float coordScale = glm::length(baseDomain.domainMax - baseDomain.domainMin);
  if (!(coordScale > 0.0f) || !std::isfinite(coordScale)) coordScale = 1.0f;
  return std::max(
      64.0 * double(FLT_EPSILON) * double(coordScale),
      256.0 * DBL_EPSILON * double(coordScale));
}

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

  bool multipleModels = false;
  const SurfaceRegion *referenceModelRegion = nullptr;
  for (const SurfaceRegion &region : regions) {
    if (region.faceId == InvalidId) continue;
    if (referenceModelRegion == nullptr) {
      referenceModelRegion = &region;
      continue;
    }
    if (region.modelIndex != referenceModelRegion->modelIndex) {
      multipleModels = true;
      break;
    }
  }
  if (!multipleModels) return true;

  const double numerical = computeDryContactGapTolerance(baseDomain);
  const double tolerance = std::max(contactRange, numerical);

  std::vector<DeviceContactRegion> hostRegions;
  std::vector<double3> hostVertices;
  for (uint32_t i = 0; i < regions.size(); ++i) {
    if (regions[i].faceId == InvalidId) continue;
    const uint32_t fragmentId = solidFragmentForRegion(regions[i], rvd);
    if (fragmentId == InvalidId) {
      std::cerr << "[RVDContact] cannot resolve solid-side incidence for "
                << "region=" << i << " face=" << regions[i].faceId
                << " modelId=" << regions[i].runtimeModelId << std::endl;
      return false;
    }
    DeviceContactRegion descriptor{};
    if (!buildContactRegion(regions[i], fragmentId, hostVertices, descriptor)) {
      std::cerr << "[RVDContact] invalid surface region " << i << std::endl;
      return false;
    }
    hostRegions.push_back(descriptor);
  }

  glm::dvec3 boundsMin(std::numeric_limits<double>::max());
  glm::dvec3 boundsMax(-std::numeric_limits<double>::max());
  double extentSum = 0.0;
  double maximumExtent = 0.0;
  for (const DeviceContactRegion &region : hostRegions) {
    const glm::dvec3 minimum(region.boundsMin.x, region.boundsMin.y, region.boundsMin.z);
    const glm::dvec3 maximum(region.boundsMax.x, region.boundsMax.y, region.boundsMax.z);
    boundsMin = glm::min(boundsMin, minimum);
    boundsMax = glm::max(boundsMax, maximum);
    const glm::dvec3 regionExtent = maximum - minimum;
    const double extent = std::max(regionExtent.x, std::max(regionExtent.y, regionExtent.z));
    extentSum += extent;
    maximumExtent = std::max(maximumExtent, extent);
  }

  const uint32_t compactRegionCount = uint32_t(hostRegions.size());
  const double averageExtent = extentSum / double(compactRegionCount);
  const double gridSize = std::max(std::max(tolerance * 8.0, averageExtent * 1.5), maximumExtent / 16.0);
  if (!(gridSize > 0.0) || !std::isfinite(gridSize)) return false;

  const glm::dvec3 gridMin = boundsMin - glm::dvec3(tolerance);
  std::vector<SpatialHashRange> ranges(compactRegionCount);
  GridCoord maximumCoord{};
  for (uint32_t region = 0; region < compactRegionCount; ++region) {
    if (!spatialHashRange(hostRegions[region], tolerance, gridMin, gridSize,
                          ranges[region])) return false;
    maximumCoord.x = std::max(maximumCoord.x, ranges[region].maximum.x);
    maximumCoord.y = std::max(maximumCoord.y, ranges[region].maximum.y);
    maximumCoord.z = std::max(maximumCoord.z, ranges[region].maximum.z);
  }
  if (maximumCoord.x < 0 || maximumCoord.y < 0 || maximumCoord.z < 0) return false;
  const uint64_t dimX = uint64_t(maximumCoord.x) + 1;
  const uint64_t dimY = uint64_t(maximumCoord.y) + 1;
  if (dimX != 0 && dimY > UINT64_MAX / dimX) return false;
  const uint64_t xy = dimX * dimY;
  const uint64_t dimZ = uint64_t(maximumCoord.z) + 1;
  if (xy != 0 && dimZ > UINT64_MAX / xy) return false;

  cudaUtils::CudaBuffer<DeviceContactRegion> deviceRegions;
  cudaUtils::CudaBuffer<double3> deviceVertices;
  cudaUtils::CudaBuffer<SpatialHashRange> deviceRanges;
  if (!deviceRegions.allocate(hostRegions.size()) ||
      !deviceRegions.uploadAsync(hostRegions.data(), hostRegions.size(), stream) ||
      !deviceVertices.allocate(hostVertices.size()) ||
      !deviceVertices.uploadAsync(hostVertices.data(), hostVertices.size(), stream) ||
      !deviceRanges.allocate(ranges.size()) ||
      !deviceRanges.uploadAsync(ranges.data(), ranges.size(), stream)) {
    return false;
  }

  cudaUtils::CudaBuffer<uint32_t> deviceEntryCounts, deviceEntryOffsets;
  cudaUtils::CudaBuffer<uint32_t> deviceOverflow;
  cudaUtils::CudaBuffer<uint64_t> deviceContactError;
  cudaUtils::CudaBuffer<uint8_t> scanScratch;
  if (!deviceEntryCounts.allocate(compactRegionCount + 1) ||
      !deviceEntryOffsets.allocate(compactRegionCount + 1) ||
      !deviceOverflow.allocate(1) ||
      !deviceContactError.allocate(1) ||
      !rvdHost::zero(deviceOverflow.get(), sizeof(uint32_t), stream, "clear contact entry overflow") ||
      !rvdHost::zero(
          deviceEntryCounts.get() + compactRegionCount, sizeof(uint32_t), stream,
          "clear contact entry count tail")) {
    return false;
  }
  countSpatialHashEntriesKernel<<<
      rvdHost::blockCount(compactRegionCount, ContactBlockSize),
      ContactBlockSize, 0, stream>>>(
      deviceRanges.get(), compactRegionCount, deviceEntryCounts.get(),
      deviceOverflow.get());
  if (!rvdHost::checkLaunch("launch countSpatialHashEntriesKernel")) return false;
  if (!contactIndexRangeFits(
          contactTotal(deviceEntryCounts.get(), compactRegionCount, stream),
          "contact hash entries")) {
    return false;
  }
  if (!rvdHost::exclusiveSum(
          deviceEntryCounts.get(), deviceEntryOffsets.get(),
          int(compactRegionCount + 1), scanScratch, stream)) {
    return false;
  }
  uint32_t overflow = 0, entryCount = 0;
  if (!rvdHost::readAt(overflow, deviceOverflow.get(), 0, stream, "read contact entry overflow") ||
      !rvdHost::readAt(
          entryCount, deviceEntryOffsets.get(), compactRegionCount, stream,
          "read contact entry count")) {
    return false;
  }
  if (overflow != 0) return false;
  if (entryCount == 0) return true;

  cudaUtils::CudaBuffer<SpatialHashEntry> deviceEntries;
  if (!deviceEntries.allocate(entryCount)) return false;
  emitSpatialHashEntriesKernel<<<
      rvdHost::blockCount(compactRegionCount, ContactBlockSize),
      ContactBlockSize, 0, stream>>>(
      deviceRanges.get(), compactRegionCount, deviceEntryOffsets.get(),
      deviceRegions.get(), dimX, dimY, deviceEntries.get());
  if (!rvdHost::checkLaunch("launch emitSpatialHashEntriesKernel")) return false;
  thrust::sort(
      thrust::cuda::par.on(stream),
      thrust::device_ptr<SpatialHashEntry>(deviceEntries.get()),
      thrust::device_ptr<SpatialHashEntry>(deviceEntries.get()) + entryCount);

  cudaUtils::CudaBuffer<uint32_t> devicePairCounts, devicePairOffsets;
  if (!devicePairCounts.allocate(size_t(entryCount) + 1) ||
      !devicePairOffsets.allocate(size_t(entryCount) + 1) ||
      !rvdHost::zero(deviceOverflow.get(), sizeof(uint32_t), stream, "clear contact pair overflow") ||
      !rvdHost::zero(devicePairCounts.get() + entryCount, sizeof(uint32_t), stream, "clear contact pair count tail")) {
    return false;
  }
  countPairsInSpatialEntriesKernel<<<
      rvdHost::blockCount(entryCount, ContactBlockSize), ContactBlockSize, 0, stream>>>(
      deviceEntries.get(), entryCount, deviceRegions.get(),
      contactRange, devicePairCounts.get(), deviceOverflow.get());
  if (!rvdHost::checkLaunch("launch countPairsInSpatialEntriesKernel")) return false;
  if (!contactIndexRangeFits(contactTotal(devicePairCounts.get(), entryCount, stream), "contact pairs")) return false; 
  if (!rvdHost::exclusiveSum(devicePairCounts.get(), devicePairOffsets.get(), int(entryCount + 1), scanScratch, stream)) return false;

  uint32_t pairOverflow = 0, pairCount = 0;
  if (!rvdHost::readAt(pairOverflow, deviceOverflow.get(), 0, stream, "read contact pair overflow") ||
      !rvdHost::readAt(pairCount, devicePairOffsets.get(), entryCount, stream, "read contact pair count")) {
    return false;
  }
  if (pairOverflow != 0) return false;
  if (pairCount == 0) return true;

  cudaUtils::CudaBuffer<uint64_t> devicePairKeys;
  if (!devicePairKeys.allocate(pairCount)) return false;
  emitPairsFromSpatialEntriesKernel<<<
      rvdHost::blockCount(entryCount, ContactBlockSize), ContactBlockSize, 0, stream>>>(
      deviceEntries.get(), entryCount, deviceRegions.get(),
      contactRange, devicePairOffsets.get(), devicePairKeys.get());
  if (!rvdHost::checkLaunch("launch emitPairsFromSpatialEntriesKernel")) return false;
  thrust::sort(
      thrust::cuda::par.on(stream),
      thrust::device_ptr<uint64_t>(devicePairKeys.get()),
      thrust::device_ptr<uint64_t>(devicePairKeys.get()) + pairCount);
  const auto uniqueEnd = thrust::unique(
      thrust::cuda::par.on(stream),
      thrust::device_ptr<uint64_t>(devicePairKeys.get()),
      thrust::device_ptr<uint64_t>(devicePairKeys.get()) + pairCount);
  const uint32_t uniquePairCount =
      uint32_t(uniqueEnd - thrust::device_ptr<uint64_t>(devicePairKeys.get()));
  if (uniquePairCount == 0) return true;

  cudaUtils::CudaBuffer<uint32_t> deviceScratchCapacities;
  cudaUtils::CudaBuffer<uint32_t> deviceScratchOffsets;
  if (!deviceScratchCapacities.allocate(size_t(uniquePairCount) + 1) ||
      !deviceScratchOffsets.allocate(size_t(uniquePairCount) + 1) ||
      !rvdHost::zero(deviceScratchCapacities.get() + uniquePairCount, sizeof(uint32_t), stream, "clear contact scratch capacity tail") ||
      !rvdHost::zero(deviceOverflow.get(), sizeof(uint32_t), stream, "clear contact scratch overflow")) {
    return false;
  }
  countPairScratchCapacitiesKernel<<<
      rvdHost::blockCount(uniquePairCount, ContactBlockSize),
      ContactBlockSize, 0, stream>>>(
      devicePairKeys.get(), uniquePairCount, deviceRegions.get(),
      deviceScratchCapacities.get(), deviceOverflow.get());
  if (!rvdHost::checkLaunch("launch countPairScratchCapacitiesKernel")) return false;
  if (!contactIndexRangeFits(
          contactTotal(deviceScratchCapacities.get(), uniquePairCount, stream),
          "contact scratch units")) {
    return false;
  }
  if (!rvdHost::exclusiveSum(deviceScratchCapacities.get(), deviceScratchOffsets.get(), int(uniquePairCount + 1), scanScratch, stream)) {
    return false;
  }
  uint32_t scratchOverflow = 0, scratchUnits = 0;
  if (!rvdHost::copyToHostAsync(&scratchOverflow, deviceOverflow.get(), 1, stream,"read contact scratch overflow") ||
      !rvdHost::copyToHostAsync(&scratchUnits, deviceScratchOffsets.get() + uniquePairCount, 1, stream,"read contact scratch units") ||
      !rvdHost::synchronize(stream, "read contact scratch layout")) {
    return false;
  }
  if (scratchOverflow != 0) {
    std::cerr << "[RVDContact] contact scratch capacity overflow" << std::endl;
    return false;
  }

  cudaUtils::CudaBuffer<double2> deviceScratch;
  cudaUtils::CudaBuffer<RawCandidateOverlap> deviceRawCandidates;
  cudaUtils::CudaBuffer<uint32_t> deviceCandidateValid;
  cudaUtils::CudaBuffer<uint32_t> deviceCompactOffsets;
  const size_t scratchElementCount = size_t(scratchUnits) * 2;
  if (!deviceScratch.allocate(scratchElementCount) ||
      !deviceRawCandidates.allocate(uniquePairCount) ||
      !deviceCandidateValid.allocate(size_t(uniquePairCount) + 1) ||
      !deviceCompactOffsets.allocate(size_t(uniquePairCount) + 1) ||
      !rvdHost::zero(deviceCandidateValid.get() + uniquePairCount, sizeof(uint32_t), stream, "clear candidate valid tail") ||
      !rvdHost::fill(deviceContactError.get(), 0xFF, sizeof(uint64_t), stream, "clear contact candidate error")) {
    return false;
  }
  clipContactCandidatesKernel<<<
      rvdHost::blockCount(uniquePairCount, ContactBlockSize),
      ContactBlockSize, 0, stream>>>(
      devicePairKeys.get(), uniquePairCount, deviceRegions.get(),
      deviceVertices.get(), deviceScratchOffsets.get(), deviceScratch.get(),
      contactRange, numerical, deviceRawCandidates.get(),
      deviceCandidateValid.get(), deviceContactError.get());
  if (!rvdHost::checkLaunch("launch clipContactCandidatesKernel")) return false;
  if (reportContactError(deviceContactError, stream, "contact candidate")) {
    return false;
  }

  if (!rvdHost::exclusiveSum(deviceCandidateValid.get(), deviceCompactOffsets.get(), int(uniquePairCount + 1), scanScratch, stream)) {
    return false;
  }

  uint32_t acceptedOverlapCount = 0;
  if (!rvdHost::readAt(acceptedOverlapCount, deviceCompactOffsets.get(), uniquePairCount, stream, "read accepted overlap count")) {
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
      rvdHost::blockCount(uniquePairCount, ContactBlockSize),
      ContactBlockSize, 0, stream>>>(
      uniquePairCount, deviceCandidateValid.get(),
      deviceCompactOffsets.get(), deviceRawCandidates.get(),
      deviceAcceptedVertexCounts.get());
  if (!rvdHost::checkLaunch("launch gatherAcceptedVertexCountsKernel")) return false;
  if (!contactIndexRangeFits(
          contactTotal(deviceAcceptedVertexCounts.get(), acceptedOverlapCount, stream),
          "contact vertices")) {
    return false;
  }
  if (!rvdHost::exclusiveSum(
          deviceAcceptedVertexCounts.get(), deviceAcceptedVertexOffsets.get(),
          int(acceptedOverlapCount + 1), scanScratch, stream)) {
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
      rvdHost::blockCount(uniquePairCount, ContactBlockSize),
      ContactBlockSize, 0, stream>>>(
      devicePairKeys.get(), uniquePairCount, deviceRegions.get(),
      deviceRawCandidates.get(), deviceScratch.get(),
      deviceCandidateValid.get(), deviceCompactOffsets.get(),
      deviceAcceptedVertexOffsets.get(),
      deviceCompactOverlaps.get(), deviceCompactVertices.get());
  if (!rvdHost::checkLaunch("launch compactAcceptedOverlapsKernel")) return false;

  if (!contactIndexRangeFits(
          uint64_t(acceptedOverlapCount) * 2u, "contact face instances")) {
    return false;
  }
  const uint32_t instanceCount = acceptedOverlapCount * 2;
  cudaUtils::CudaBuffer<uint32_t> deviceInstanceFaceIds;
  cudaUtils::CudaBuffer<uint32_t> deviceInstanceCandData;
  cudaUtils::CudaBuffer<uint32_t> deviceSortedFaceIds;
  cudaUtils::CudaBuffer<uint32_t> deviceSortedCandData;

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
          int(instanceCount), scanScratch, stream)) {
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
          deviceResolvedValid.get() + acceptedOverlapCount, sizeof(uint32_t), stream,
          "clear resolved valid tail") ||
      !rvdHost::fill(deviceContactError.get(), 0xFF, sizeof(uint64_t), stream, "clear contact mortar error")) {
    return false;
  }

  const uint32_t scratchCandidateCount = std::min(acceptedOverlapCount, MortarChunkSize);
  cudaUtils::CudaBuffer<LocalPolygon> deviceMortarScratch;
  if (scratchCandidateCount > 0 &&
      !deviceMortarScratch.allocate(size_t(scratchCandidateCount) * (2 * MaxPieces))) {
    return false;
  }

  const FaceCandidateIndex candidateIndex{
      deviceSortedFaceIds.get(), deviceSortedCandData.get(), instanceCount};
  for (uint32_t chunkOffset = 0; chunkOffset < acceptedOverlapCount; chunkOffset += MortarChunkSize) {
    const uint32_t chunkCount = std::min(MortarChunkSize, acceptedOverlapCount - chunkOffset);
    resolveMortarCouplingsKernel<<<
        rvdHost::blockCount(chunkCount, ContactBlockSize),
        ContactBlockSize, 0, stream>>>(
        deviceCompactOverlaps.get(), deviceCompactVertices.get(),
        chunkOffset, chunkCount,
        deviceRegions.get(), candidateIndex, numerical,
        deviceMortarScratch.get(),
        deviceResolvedAreas.get(), deviceResolvedValid.get(),
        deviceContactError.get());
    if (!rvdHost::checkLaunch("launch resolveMortarCouplingsKernel")) return false;
  }
  if (reportContactError(deviceContactError, stream, "contact mortar")) {
    return false;
  }

  if (!rvdHost::exclusiveSum(
          deviceResolvedValid.get(), deviceResolvedOffsets.get(),
          int(acceptedOverlapCount + 1), scanScratch, stream)) {
    return false;
  }

  uint32_t finalCouplingCount = 0;
  if (!rvdHost::readAt(finalCouplingCount, deviceResolvedOffsets.get(), acceptedOverlapCount, stream, "read final coupling count")) {
    return false;
  }

  if (finalCouplingCount == 0) return true;

  cudaUtils::CudaBuffer<ContactCoupling> deviceFinalCouplings;
  if (!deviceFinalCouplings.allocate(finalCouplingCount)) return false;

  emitFinalCouplingsKernel<<<
      rvdHost::blockCount(acceptedOverlapCount, ContactBlockSize),
      ContactBlockSize, 0, stream>>>(
      deviceCompactOverlaps.get(), deviceResolvedAreas.get(),
      deviceResolvedValid.get(), deviceResolvedOffsets.get(),
      acceptedOverlapCount, deviceFinalCouplings.get());
  if (!rvdHost::checkLaunch("launch emitFinalCouplingsKernel")) return false;

  couplings.resize(finalCouplingCount);
  if (!rvdHost::copyToHost(
          couplings.data(), deviceFinalCouplings.get(),
          finalCouplingCount, stream, "download final contact couplings")) {
    return false;
  }

  return true;
}

} // namespace voronoi
