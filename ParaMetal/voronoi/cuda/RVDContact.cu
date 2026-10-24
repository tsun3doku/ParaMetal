#include "RVDContact.cuh"
#include "RVDHostUtils.cuh"
#include "TriangleBvh.cuh"

#include "../../cuda/CudaMath.cuh"
#include "../../cuda/CudaBuffer.cuh"

#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace voronoi {

namespace {

using cudaMath::add3;
using cudaMath::cross3;
using cudaMath::dot3;
using cudaMath::length3;
using cudaMath::mul3;
using cudaMath::normalize3;
using cudaMath::sub3;

constexpr uint32_t BlockSize = rvdHost::BlockSize;
constexpr int MaxMortarScratchVertices = 128;
constexpr uint64_t InvalidCouplingKey = 0xFFFFFFFFFFFFFFFFull;

struct MortarBasis {
  double3 origin;
  double3 u;
  double3 v;
  double3 normal;
};

struct DeviceModelInfo {
  uint32_t surfaceBase = 0;
  uint32_t totalPatches = 0;
  const RestrictedSurfaceGeometry *geom = nullptr;
  const RestrictedSurface *surfaces = nullptr;
  const double3 *vertices = nullptr;
};

struct MortarMeshBuffers {
  cudaUtils::CudaBuffer<double2> vertices;
  cudaUtils::CudaBuffer<uint32_t> vertexOffsets;
  cudaUtils::CudaBuffer<uint32_t> vertexCounts;
  cudaUtils::CudaBuffer<MortarBasis> bases;
  uint32_t vertexCount = 0;

  void reset() {
    vertices.reset();
    vertexOffsets.reset();
    vertexCounts.reset();
    bases.reset();
    vertexCount = 0;
  }
};

MortarMeshBuffers gMortarMeshBuffers;

struct PairOverlapBuffers {
  cudaUtils::CudaBuffer<MortarOverlap> overlaps;
  cudaUtils::CudaBuffer<MortarBasis> bases;
  cudaUtils::CudaBuffer<uint32_t> vertexCounts;
  cudaUtils::CudaBuffer<double2> vertices;
  uint32_t overlapCount = 0;
  uint32_t vertexCount = 0;
};

struct PairWorkBuffers {
  cudaUtils::CudaBuffer<uint32_t> triangleKeysIn;
  cudaUtils::CudaBuffer<uint32_t> triangleKeysOut;
  cudaUtils::CudaBuffer<uint32_t> patchIndicesIn;
  cudaUtils::CudaBuffer<uint32_t> patchIndicesOut;
  cudaUtils::CudaBuffer<uint32_t> countsA;
  cudaUtils::CudaBuffer<uint32_t> offsetsA;
  cudaUtils::CudaBuffer<uint2> candidatePairs;
  cudaUtils::CudaBuffer<uint32_t> overlapValid;
  cudaUtils::CudaBuffer<uint32_t> overlapOffsets;
  cudaUtils::CudaBuffer<uint32_t> vertexCounts;
  cudaUtils::CudaBuffer<uint32_t> vertexOffsets;
  cudaUtils::CudaBuffer<uint8_t> scratch;
};

__host__ __device__ inline double polygonSignedArea2D(const double2 *polygon, int count) {
  if (count < 3) return 0.0;

  double sum = 0.0;
  for (int i = 0; i < count; ++i) {
    const int next = (i + 1 == count) ? 0 : i + 1;
    sum += polygon[i].x * polygon[next].y - polygon[next].x * polygon[i].y;
  }
  return 0.5 * sum;
}

__host__ __device__ inline void reversePolygon2D(double2 *polygon, int count) {
  for (int i = 0; i < count / 2; ++i) {
    const double2 temporary = polygon[i];
    polygon[i] = polygon[count - 1 - i];
    polygon[count - 1 - i] = temporary;
  }
}

__host__ __device__ inline void ensureCounterClockwise(double2 *polygon, int count) {
  if (polygonSignedArea2D(polygon, count) < 0.0) {
    reversePolygon2D(polygon, count);
  }
}

__device__ inline int clipPolygon2DByHalfPlane(
    const double2 *input,
    int inputCount,
    double a,
    double b,
    double c,
    double2 *output,
    int *capacityError = nullptr) {
  if (inputCount < 3) return 0;

  int outputCount = 0;
  double2 previous = input[inputCount - 1];
  double previousDistance = a * previous.x + b * previous.y + c;

  for (int i = 0; i < inputCount; ++i) {
    const double2 current = input[i];
    const double currentDistance = a * current.x + b * current.y + c;

    if ((previousDistance < 0.0 && currentDistance >= 0.0) ||
        (previousDistance >= 0.0 && currentDistance < 0.0)) {
      if (outputCount >= MaxMortarScratchVertices) {
        if (capacityError) atomicExch(capacityError, 1);
        return -1;
      }

      const double t = previousDistance / (previousDistance - currentDistance);
      output[outputCount++] = make_double2(
          previous.x + t * (current.x - previous.x),
          previous.y + t * (current.y - previous.y));
    }

    if (currentDistance >= 0.0) {
      if (outputCount >= MaxMortarScratchVertices) {
        if (capacityError) atomicExch(capacityError, 1);
        return -1;
      }
      output[outputCount++] = current;
    }

    previous = current;
    previousDistance = currentDistance;
  }
  return outputCount;
}

struct ClippedPolygon2D {
  const double2 *points = nullptr;
  int count = 0;
};

__device__ inline ClippedPolygon2D clipConvexPolygon2D(
    const double2 *subject,
    int subjectCount,
    const double2 *clipPolygon,
    int clipCount,
    double2 *scratchA,
    double2 *scratchB,
    int *capacityError = nullptr) {
  if (subjectCount < 3 || clipCount < 3) return {};

  const double2 *source = subject;
  int count = subjectCount;
  double2 *destination = scratchA;

  for (int edge = 0; edge < clipCount; ++edge) {
    const int next = (edge + 1 == clipCount) ? 0 : edge + 1;
    const double2 a = clipPolygon[edge];
    const double2 b = clipPolygon[next];

    const double lineA = a.y - b.y;
    const double lineB = b.x - a.x;
    const double lineC = b.y * a.x - b.x * a.y;

    const int newCount = clipPolygon2DByHalfPlane(source, count, lineA, lineB, lineC, destination, capacityError);
    if (newCount < 3) return {};

    count = newCount;
    source = destination;
    destination = destination == scratchA ? scratchB : scratchA;
  }

  return {source, count};
}

__host__ __device__ inline double polygonAreaAndCentroid2D(
    const double2 *polygon,
    int count,
    double2 &centroid) {
  if (count < 3) {
    centroid = make_double2(0.0, 0.0);
    return 0.0;
  }

  double signedDoubleArea = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  for (int i = 0; i < count; ++i) {
    const int next = (i + 1 == count) ? 0 : i + 1;
    const double cross = polygon[i].x * polygon[next].y - polygon[next].x * polygon[i].y;
    signedDoubleArea += cross;
    cx += (polygon[i].x + polygon[next].x) * cross;
    cy += (polygon[i].y + polygon[next].y) * cross;
  }

  const double area = 0.5 * fabs(signedDoubleArea);
  if (area <= 1e-14) {
    centroid = make_double2(0.0, 0.0);
    return 0.0;
  }
  centroid = make_double2(cx / (3.0 * signedDoubleArea), cy / (3.0 * signedDoubleArea));
  return area;
}

__device__ inline const DeviceModelInfo *findModelInfo(
    const DeviceModelInfo *models,
    uint32_t modelCount,
    uint32_t surface) {
  for (uint32_t m = 0; m < modelCount; ++m) {
    if (surface >= models[m].surfaceBase &&
        surface < models[m].surfaceBase + models[m].totalPatches) {
      return &models[m];
    }
  }
  return nullptr;
}

__device__ inline double3 geometryNormal(const RestrictedSurfaceGeometry &geometry) {
  return normalize3(geometry.normal);
}

__device__ inline double3 geometryCentroidWorld(
    const RestrictedSurfaceGeometry &geometry,
    float3 domainMin,
    double invBaseScale) {
  return make_double3(
      double(domainMin.x) + geometry.centroidArea.x * invBaseScale,
      double(domainMin.y) + geometry.centroidArea.y * invBaseScale,
      double(domainMin.z) + geometry.centroidArea.z * invBaseScale);
}

__device__ inline bool clipMortarPair(
    uint32_t vertexCountA,
    const double3 *verticesA,
    double3 normalA,
    double3 centroidA,
    uint32_t vertexCountB,
    const double3 *verticesB,
    double3 normalB,
    double3 centroidB,
    double contactRange,
    double gapPenetrationTolerance,
    MortarOverlap &overlap,
    MortarBasis &basis,
    double2 *outputVertices,
    int &outputVertexCount,
    int *capacityError = nullptr) {
  if (vertexCountA < 3 || vertexCountB < 3) return false;
  if (vertexCountA > MaxMortarScratchVertices ||
      vertexCountB > MaxMortarScratchVertices) {
    if (capacityError) atomicExch(capacityError, 1);
    return false;
  }

  const double3 mortarNormal = normalA;
  const double dotAMortar = 1.0;
  const double dotBMortar = dot3(normalB, mortarNormal);
  if (dotBMortar > -0.5) return false;

  const double3 reference =
      fabs(mortarNormal.x) < 0.9
      ? make_double3(1.0, 0.0, 0.0)
      : make_double3(0.0, 1.0, 0.0);
  const double3 uCross = cross3(reference, mortarNormal);
  const double uLength = length3(uCross);
  if (uLength <= 1e-14) return false;

  const double3 u = mul3(uCross, 1.0 / uLength);
  const double3 v = cross3(mortarNormal, u);
  const double3 mortarPoint = centroidA;

  double2 polygonA[MaxMortarScratchVertices];
  for (uint32_t k = 0; k < vertexCountA; ++k) {
    const double3 delta = sub3(verticesA[k], mortarPoint);
    polygonA[k] = make_double2(dot3(delta, u), dot3(delta, v));
  }
  ensureCounterClockwise(polygonA, int(vertexCountA));

  double2 polygonB[MaxMortarScratchVertices];
  for (uint32_t k = 0; k < vertexCountB; ++k) {
    const double3 delta = sub3(verticesB[k], mortarPoint);
    polygonB[k] = make_double2(dot3(delta, u), dot3(delta, v));
  }
  ensureCounterClockwise(polygonB, int(vertexCountB));

  double2 scratch1[MaxMortarScratchVertices];
  double2 scratch2[MaxMortarScratchVertices];

  ClippedPolygon2D clipped = clipConvexPolygon2D(
      polygonA, int(vertexCountA), polygonB, int(vertexCountB),
      scratch1, scratch2, capacityError);
  if (clipped.count < 3) return false;

  const double2 *source = clipped.points;
  double2 *destination = source == scratch1 ? scratch2 : scratch1;
  int currentCount = clipped.count;

  const double g0 =
      dot3(normalB, sub3(centroidB, mortarPoint)) / dotBMortar -
      dot3(normalA, sub3(centroidA, mortarPoint)) / dotAMortar;
  const double gu = -dot3(normalB, u) / dotBMortar + dot3(normalA, u) / dotAMortar;
  const double gv = -dot3(normalB, v) / dotBMortar + dot3(normalA, v) / dotAMortar;

  const double gradientMagnitude = sqrt(gu * gu + gv * gv);
  if (gradientMagnitude > 1e-12) {
    int newCount = clipPolygon2DByHalfPlane(
        source, currentCount, gu, gv, g0 + gapPenetrationTolerance,
        destination, capacityError);
    if (newCount < 3) return false;
    currentCount = newCount;
    source = destination;
    destination = destination == scratch1 ? scratch2 : scratch1;

    newCount = clipPolygon2DByHalfPlane(
        source, currentCount, -gu, -gv, contactRange - g0,
        destination, capacityError);
    if (newCount < 3) return false;
    currentCount = newCount;
    source = destination;
    destination = destination == scratch1 ? scratch2 : scratch1;
  } else if (g0 < -gapPenetrationTolerance || g0 > contactRange) {
    return false;
  }

  double2 centroid2D;
  const double couplingArea = polygonAreaAndCentroid2D(source, currentCount, centroid2D);
  if (couplingArea <= 1e-14) return false;

  const double3 couplingCentroid = add3(
      mortarPoint,
      add3(mul3(u, centroid2D.x), mul3(v, centroid2D.y)));
  const double areaA = couplingArea / fabs(dotAMortar);
  const double areaB = couplingArea / fabs(dotBMortar);

  const double zA =
      (dot3(normalA, sub3(centroidA, mortarPoint)) -
       centroid2D.x * dot3(normalA, u) -
       centroid2D.y * dot3(normalA, v)) /
      dotAMortar;
  const double zB =
      (dot3(normalB, sub3(centroidB, mortarPoint)) -
       centroid2D.x * dot3(normalB, u) -
       centroid2D.y * dot3(normalB, v)) /
      dotBMortar;

  const double3 projectedA = add3(couplingCentroid, mul3(mortarNormal, zA));
  const double3 projectedB = add3(couplingCentroid, mul3(mortarNormal, zB));

  overlap.couplingArea = couplingArea;
  overlap.areaA = areaA;
  overlap.areaB = areaB;
  overlap.momentA = mul3(projectedA, areaA);
  overlap.momentB = mul3(projectedB, areaB);

  basis.origin = mortarPoint;
  basis.u = u;
  basis.v = v;
  basis.normal = mortarNormal;

  if (outputVertices != nullptr) {
    for (int k = 0; k < currentCount; ++k) {
      outputVertices[k] = source[k];
    }
  }
  outputVertexCount = currentCount;
  return true;
}

__device__ inline int2 findPatchRangeForTriangle(
    const uint32_t *sortedTriangles,
    uint32_t patchCount,
    uint32_t targetTriangle) {
  int left = 0;
  int right = int(patchCount);
  while (left < right) {
    const int middle = left + (right - left) / 2;
    if (sortedTriangles[middle] < targetTriangle) {
      left = middle + 1;
    } else {
      right = middle;
    }
  }

  const int start = left;
  if (start >= int(patchCount) || sortedTriangles[start] != targetTriangle) {
    return make_int2(0, 0);
  }

  left = start;
  right = int(patchCount);
  while (left < right) {
    const int middle = left + (right - left) / 2;
    if (sortedTriangles[middle] <= targetTriangle) {
      left = middle + 1;
    } else {
      right = middle;
    }
  }
  return make_int2(start, left - start);
}

__global__ void initSurfaceExposureKernel(
    const RestrictedSurfaceGeometry *geometry,
    uint32_t patchCount,
    uint32_t surfaceBase,
    double invBaseScale,
    float3 domainMin,
    double4 *surfaceExposure) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= patchCount) return;

  const RestrictedSurfaceGeometry patch = geometry[index];
  const double inverseScaleSquared = invBaseScale * invBaseScale;
  const double area = double(patch.centroidArea.w) * inverseScaleSquared;
  const double3 centroid = make_double3(patch.centroidArea.x, patch.centroidArea.y, patch.centroidArea.z);
  const double3 worldCentroid = cudaGeometry::canonicalToWorld(centroid, make_double3(domainMin.x, domainMin.y, domainMin.z), invBaseScale);
  const double3 moment = mul3(worldCentroid, area);

  surfaceExposure[surfaceBase + index] = make_double4(moment.x, moment.y, moment.z, area);
}

__global__ void extractPatchTrianglesKernel(
    const RestrictedSurface *surfaces,
    uint32_t patchCount,
    uint32_t *triangleIds,
    uint32_t *patchIndices) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= patchCount) return;

  triangleIds[index] = surfaces[index].triangleId;
  patchIndices[index] = index;
}

__global__ void countAndEmitCandidatePatchPairsKernel(
    const RestrictedSurface *surfacesA,
    const RestrictedSurfaceGeometry *geometryA,
    const double3 *verticesA,
    uint32_t patchCountA,
    uint32_t surfaceBaseA,
    cudaGeometry::DeviceModelGeometryView geometryB,
    const uint32_t *sortedTrianglesB,
    const uint32_t *sortedPatchIndicesB,
    const RestrictedSurfaceGeometry *geometryBData,
    uint32_t patchCountB,
    uint32_t surfaceBaseB,
    double contactRange,
    float3 domainMin,
    double invBaseScale,
    float coordScale,
    const uint32_t *offsetsA,
    uint32_t *countsA,
    uint2 *candidatePairs,
    bool emitPass) {
  const uint32_t indexA = blockIdx.x * blockDim.x + threadIdx.x;
  if (indexA >= patchCountA) return;

  const RestrictedSurface surfaceA = surfacesA[indexA];
  if (surfaceA.vertexCount < 3) {
    if (!emitPass) countsA[indexA] = 0;
    return;
  }

  const double3 domainMinD = make_double3(domainMin.x, domainMin.y, domainMin.z);
  float3 boundsMin = make_float3(1e30f, 1e30f, 1e30f);
  float3 boundsMax = make_float3(-1e30f, -1e30f, -1e30f);
  for (uint32_t v = 0; v < surfaceA.vertexCount; ++v) {
    const double3 canonical = verticesA[surfaceA.firstVertex + v];
    const double3 world = cudaGeometry::canonicalToWorld(canonical, domainMinD, invBaseScale);
    const float3 point = make_float3(float(world.x), float(world.y), float(world.z));
    boundsMin.x = fminf(boundsMin.x, point.x);
    boundsMin.y = fminf(boundsMin.y, point.y);
    boundsMin.z = fminf(boundsMin.z, point.z);
    boundsMax.x = fmaxf(boundsMax.x, point.x);
    boundsMax.y = fmaxf(boundsMax.y, point.y);
    boundsMax.z = fmaxf(boundsMax.z, point.z);
  }

  const double3 normalA = geometryNormal(geometryA[indexA]);
  const float margin = 16.0f * FLT_EPSILON * coordScale;
  const float contactRangeF = float(contactRange);
  const float3 queryMin = make_float3(
      boundsMin.x - contactRangeF - margin,
      boundsMin.y - contactRangeF - margin,
      boundsMin.z - contactRangeF - margin);
  const float3 queryMax = make_float3(
      boundsMax.x + contactRangeF + margin,
      boundsMax.y + contactRangeF + margin,
      boundsMax.z + contactRangeF + margin);

  const float maxDistance = contactRangeF + margin;
  const float maxDistanceSquared = maxDistance * maxDistance;

  uint32_t stack[64];
  int stackPointer = 0;
  if (geometryB.root != cudaGeometry::InvalidNodeId && geometryB.nodes != nullptr) {
    stack[stackPointer++] = geometryB.root;
  }

  uint32_t foundCount = 0;
  const uint32_t destinationBase = emitPass ? offsetsA[indexA] : 0;

  while (stackPointer > 0) {
    const uint32_t nodeIndex = stack[--stackPointer];
    const cudaGeometry::TriangleBvhNode node = geometryB.nodes[nodeIndex];

    if (node.maximum.x < queryMin.x || node.minimum.x > queryMax.x ||
        node.maximum.y < queryMin.y || node.minimum.y > queryMax.y ||
        node.maximum.z < queryMin.z || node.minimum.z > queryMax.z) {
      continue;
    }
    if (cudaGeometry::aabbDistanceSquared(
            node.minimum, node.maximum, boundsMin, boundsMax) >
        maxDistanceSquared) {
      continue;
    }

    if (node.isLeaf()) {
      const uint32_t triangleB = node.triangleId();
      const int2 range = findPatchRangeForTriangle(sortedTrianglesB, patchCountB, triangleB);

      for (int r = 0; r < range.y; ++r) {
        const uint32_t patchBIndex = sortedPatchIndicesB[range.x + r];
        const double3 normalB = geometryNormal(geometryBData[patchBIndex]);
        if (dot3(normalA, normalB) < -0.2) {
          if (emitPass) {
            candidatePairs[destinationBase + foundCount] =
                make_uint2(surfaceBaseA + indexA, surfaceBaseB + patchBIndex);
          }
          ++foundCount;
        }
      }
    } else if (stackPointer < 62) {
      stack[stackPointer++] = node.leftChild();
      stack[stackPointer++] = node.rightChild();
    }
  }

  if (!emitPass) {
    countsA[indexA] = foundCount;
  }
}

__global__ void countAndEmitMortarOverlapsKernel(
    const uint2 *candidatePairs,
    uint32_t candidateCount,
    const DeviceModelInfo *models,
    uint32_t modelCount,
    double contactRange,
    double gapPenetrationTolerance,
    float3 domainMin,
    double invBaseScale,
    const uint32_t *overlapOffsets,
    const uint32_t *vertexOffsets,
    uint32_t *overlapValid,
    uint32_t *vertexCounts,
    MortarOverlap *acceptedOverlaps,
    MortarBasis *bases,
    double2 *vertices,
    bool emitPass,
    int *capacityError) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= candidateCount) return;

  const uint2 pair = candidatePairs[index];
  const uint32_t globalA = pair.x;
  const uint32_t globalB = pair.y;

  const DeviceModelInfo *infoA = findModelInfo(models, modelCount, globalA);
  const DeviceModelInfo *infoB = findModelInfo(models, modelCount, globalB);
  if (!infoA || !infoB) {
    if (!emitPass) {
      overlapValid[index] = 0;
      vertexCounts[index] = 0;
    }
    return;
  }

  const uint32_t localA = globalA - infoA->surfaceBase;
  const uint32_t localB = globalB - infoB->surfaceBase;

  const RestrictedSurface surfaceA = infoA->surfaces[localA];
  const RestrictedSurface surfaceB = infoB->surfaces[localB];
  const RestrictedSurfaceGeometry geometryA = infoA->geom[localA];
  const RestrictedSurfaceGeometry geometryB = infoB->geom[localB];

  const double3 normalA = geometryNormal(geometryA);
  const double3 normalB = geometryNormal(geometryB);
  const double3 centroidA = geometryCentroidWorld(geometryA, domainMin, invBaseScale);
  const double3 centroidB = geometryCentroidWorld(geometryB, domainMin, invBaseScale);
  const double3 domainMinD = make_double3(domainMin.x, domainMin.y, domainMin.z);

  double3 verticesA[MaxMortarScratchVertices];
  for (uint32_t k = 0; k < surfaceA.vertexCount && k < MaxMortarScratchVertices; ++k) {
    const double3 canonical = infoA->vertices[surfaceA.firstVertex + k];
    verticesA[k] = cudaGeometry::canonicalToWorld(canonical, domainMinD, invBaseScale);
  }

  double3 verticesB[MaxMortarScratchVertices];
  for (uint32_t k = 0; k < surfaceB.vertexCount && k < MaxMortarScratchVertices; ++k) {
    const double3 canonical = infoB->vertices[surfaceB.firstVertex + k];
    verticesB[k] = cudaGeometry::canonicalToWorld(canonical, domainMinD, invBaseScale);
  }

  MortarOverlap overlap{};
  MortarBasis basis{};
  int vertexCount = 0;
  double2 *destination = emitPass ? vertices + vertexOffsets[index] : nullptr;

  const bool valid = clipMortarPair(
      surfaceA.vertexCount, verticesA, normalA, centroidA,
      surfaceB.vertexCount, verticesB, normalB, centroidB,
      contactRange, gapPenetrationTolerance,
      overlap, basis, destination, vertexCount, capacityError);

  if (!emitPass) {
    overlapValid[index] = valid ? 1 : 0;
    vertexCounts[index] = valid ? uint32_t(vertexCount) : 0;
    return;
  }
  if (!valid) return;

  const uint32_t destinationIndex = overlapOffsets[index];
  overlap.surfaceA = globalA;
  overlap.surfaceB = globalB;
  acceptedOverlaps[destinationIndex] = overlap;
  bases[destinationIndex] = basis;
  vertexCounts[destinationIndex] = uint32_t(vertexCount);
}

__global__ void accumulateMortarExposureKernel(
    const MortarOverlap *overlaps,
    uint32_t overlapCount,
    double4 *claimedAreaMoment) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= overlapCount) return;

  const MortarOverlap overlap = overlaps[index];
  atomicAdd(&claimedAreaMoment[overlap.surfaceA].w, overlap.areaA);
  atomicAdd(&claimedAreaMoment[overlap.surfaceA].x, overlap.momentA.x);
  atomicAdd(&claimedAreaMoment[overlap.surfaceA].y, overlap.momentA.y);
  atomicAdd(&claimedAreaMoment[overlap.surfaceA].z, overlap.momentA.z);

  atomicAdd(&claimedAreaMoment[overlap.surfaceB].w, overlap.areaB);
  atomicAdd(&claimedAreaMoment[overlap.surfaceB].x, overlap.momentB.x);
  atomicAdd(&claimedAreaMoment[overlap.surfaceB].y, overlap.momentB.y);
  atomicAdd(&claimedAreaMoment[overlap.surfaceB].z, overlap.momentB.z);
}

__global__ void finalizeSurfaceExposureKernel(
    const RestrictedSurfaceGeometry *geometry,
    uint32_t patchCount,
    uint32_t surfaceBase,
    double invBaseScale,
    float3 domainMin,
    const double4 *claimedAreaMoment,
    double4 *surfaceExposure) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= patchCount) return;

  const uint32_t globalIndex = surfaceBase + index;
  const RestrictedSurfaceGeometry patch = geometry[index];
  const double inverseScaleSquared = invBaseScale * invBaseScale;
  const double totalArea = patch.centroidArea.w * inverseScaleSquared;

  const double3 canonicalCentroid = make_double3(patch.centroidArea.x, patch.centroidArea.y, patch.centroidArea.z);
  const double3 worldCentroid = make_double3(
      double(domainMin.x) + canonicalCentroid.x * invBaseScale,
      double(domainMin.y) + canonicalCentroid.y * invBaseScale,
      double(domainMin.z) + canonicalCentroid.z * invBaseScale);
  const double3 totalMoment = mul3(worldCentroid, totalArea);

  const double4 claimed = claimedAreaMoment[globalIndex];
  const double contactArea = fmin(claimed.w, totalArea);
  const double3 contactMoment = make_double3(claimed.x, claimed.y, claimed.z);

  double exposedArea = fmax(0.0, totalArea - contactArea);
  double3 exposedMoment;
  if (exposedArea <= 256.0 * DBL_EPSILON * totalArea) {
    exposedArea = 0.0;
    exposedMoment = make_double3(0.0, 0.0, 0.0);
  } else {
    exposedMoment = sub3(totalMoment, contactMoment);
  }

  surfaceExposure[globalIndex] =
      make_double4(exposedMoment.x, exposedMoment.y, exposedMoment.z, exposedArea);
}

__global__ void mapOverlapsToCouplingKeysKernel(
    const MortarOverlap *overlaps,
    uint32_t overlapCount,
    const uint32_t *contactFaceBySurface,
    uint64_t *keys,
    double *values) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= overlapCount) return;

  const MortarOverlap overlap = overlaps[index];
  const uint32_t faceA = contactFaceBySurface[overlap.surfaceA];
  const uint32_t faceB = contactFaceBySurface[overlap.surfaceB];

  if (faceA != InvalidId && faceB != InvalidId) {
    const uint32_t minimumFace = min(faceA, faceB);
    const uint32_t maximumFace = max(faceA, faceB);
    keys[index] = (uint64_t(minimumFace) << 32) | uint64_t(maximumFace);
    values[index] = overlap.couplingArea;
  } else {
    keys[index] = InvalidCouplingKey;
    values[index] = 0.0;
  }
}

bool initializeSurfaceExposure(
    const std::vector<ModelRestriction> &models,
    uint32_t totalPatches,
    const BaseCellDomain &baseDomain,
    cudaUtils::CudaBuffer<double4> &surfaceExposure,
    cudaStream_t stream) {
  if (!rvdHost::ensureCapacity(surfaceExposure, totalPatches)) return false;

  const float3 domainMin = make_float3(
      baseDomain.domainMin.x, baseDomain.domainMin.y, baseDomain.domainMin.z);
  for (const ModelRestriction &model : models) {
    if (model.totalPatches == 0) continue;

    initSurfaceExposureKernel<<<
        rvdHost::blockCount(model.totalPatches), BlockSize, 0, stream>>>(
        model.devSurfaceGeometry.get(),
        model.totalPatches,
        model.surfaceBase,
        baseDomain.invBaseScale,
        domainMin,
        surfaceExposure.get());
  }
  return rvdHost::checkLaunch("launch initSurfaceExposureKernel");
}

bool uploadModelInfos(
    const std::vector<ModelRestriction> &models,
    cudaStream_t stream,
    cudaUtils::CudaBuffer<DeviceModelInfo> &modelInfos) {
  std::vector<DeviceModelInfo> hostInfos(models.size());
  for (uint32_t m = 0; m < models.size(); ++m) {
    hostInfos[m].surfaceBase = models[m].surfaceBase;
    hostInfos[m].totalPatches = models[m].totalPatches;
    hostInfos[m].geom = models[m].devSurfaceGeometry.get();
    hostInfos[m].surfaces = models[m].devSurfaces.get();
    hostInfos[m].vertices = models[m].devRestrictedVertices.get();
  }

  return modelInfos.allocate(models.size()) &&
         modelInfos.uploadAsync(
             hostInfos.data(), models.size(), stream);
}

bool buildPairOverlaps(
    const ModelRestriction &modelA,
    const ModelRestriction &modelB,
    const cudaGeometry::DeviceModelGeometryView &geometryB,
    const cudaUtils::CudaBuffer<DeviceModelInfo> &modelInfos,
    uint32_t modelCount,
    double contactRange,
    double gapPenetrationTolerance,
    float3 domainMin,
    double invBaseScale,
    float coordScale,
    int *capacityError,
    cudaStream_t stream,
    PairWorkBuffers &work,
    PairOverlapBuffers &pair) {
  const uint32_t patchesA = modelA.totalPatches;
  const uint32_t patchesB = modelB.totalPatches;
  if (patchesA == 0 || patchesB == 0) return true;

  if (!rvdHost::ensureCapacity(work.triangleKeysIn, patchesB) ||
      !rvdHost::ensureCapacity(work.triangleKeysOut, patchesB) ||
      !rvdHost::ensureCapacity(work.patchIndicesIn, patchesB) ||
      !rvdHost::ensureCapacity(work.patchIndicesOut, patchesB)) {
    return false;
  }

  extractPatchTrianglesKernel<<<
      rvdHost::blockCount(patchesB), BlockSize, 0, stream>>>(
      modelB.devSurfaces.get(), patchesB,
      work.triangleKeysIn.get(), work.patchIndicesIn.get());
  if (!rvdHost::checkLaunch("launch extractPatchTrianglesKernel")) return false;

  if (!rvdHost::sortPairs(
          work.triangleKeysIn.get(), work.triangleKeysOut.get(),
          work.patchIndicesIn.get(), work.patchIndicesOut.get(),
          int(patchesB), work.scratch, stream)) {
    return false;
  }

  if (!rvdHost::ensureCapacity(work.countsA, patchesA) ||
      !rvdHost::ensureCapacity(work.offsetsA, patchesA)) {
    return false;
  }

  countAndEmitCandidatePatchPairsKernel<<<
      rvdHost::blockCount(patchesA), BlockSize, 0, stream>>>(
      modelA.devSurfaces.get(),
      modelA.devSurfaceGeometry.get(),
      modelA.devRestrictedVertices.get(),
      patchesA,
      modelA.surfaceBase,
      geometryB,
      work.triangleKeysOut.get(),
      work.patchIndicesOut.get(),
      modelB.devSurfaceGeometry.get(),
      patchesB,
      modelB.surfaceBase,
      contactRange,
      domainMin,
      invBaseScale,
      coordScale,
      nullptr,
      work.countsA.get(),
      nullptr,
      false);
  if (!rvdHost::checkLaunch("launch candidate pair count pass")) return false;

  if (!rvdHost::exclusiveSum(
          work.countsA.get(), work.offsetsA.get(), int(patchesA),
          work.scratch, stream)) {
    return false;
  }

  uint32_t candidateCount = 0;
  if (!rvdHost::readLastExclusiveScanTotal(
          work.countsA.get(), work.offsetsA.get(), patchesA, stream,
          candidateCount, "read candidate pair count")) {
    return false;
  }
  if (candidateCount == 0) return true;

  if (!rvdHost::ensureCapacity(work.candidatePairs, candidateCount)) {
    return false;
  }

  countAndEmitCandidatePatchPairsKernel<<<
      rvdHost::blockCount(patchesA), BlockSize, 0, stream>>>(
      modelA.devSurfaces.get(),
      modelA.devSurfaceGeometry.get(),
      modelA.devRestrictedVertices.get(),
      patchesA,
      modelA.surfaceBase,
      geometryB,
      work.triangleKeysOut.get(),
      work.patchIndicesOut.get(),
      modelB.devSurfaceGeometry.get(),
      patchesB,
      modelB.surfaceBase,
      contactRange,
      domainMin,
      invBaseScale,
      coordScale,
      work.offsetsA.get(),
      nullptr,
      work.candidatePairs.get(),
      true);
  if (!rvdHost::checkLaunch("launch candidate pair emit pass")) return false;

  if (!rvdHost::ensureCapacity(work.overlapValid, candidateCount) ||
      !rvdHost::ensureCapacity(work.overlapOffsets, candidateCount) ||
      !rvdHost::ensureCapacity(work.vertexCounts, candidateCount) ||
      !rvdHost::ensureCapacity(work.vertexOffsets, candidateCount)) {
    return false;
  }

  countAndEmitMortarOverlapsKernel<<<
      rvdHost::blockCount(candidateCount), BlockSize, 0, stream>>>(
      work.candidatePairs.get(),
      candidateCount,
      modelInfos.get(),
      modelCount,
      contactRange,
      gapPenetrationTolerance,
      domainMin,
      invBaseScale,
      nullptr,
      nullptr,
      work.overlapValid.get(),
      work.vertexCounts.get(),
      nullptr,
      nullptr,
      nullptr,
      false,
      capacityError);
  if (!rvdHost::checkLaunch("launch mortar overlap count pass")) return false;

  if (!rvdHost::exclusiveSum(
          work.overlapValid.get(), work.overlapOffsets.get(),
          int(candidateCount), work.scratch, stream) ||
      !rvdHost::exclusiveSum(
          work.vertexCounts.get(), work.vertexOffsets.get(),
          int(candidateCount), work.scratch, stream)) {
    return false;
  }

  uint32_t acceptedOverlaps = 0;
  uint32_t acceptedVertices = 0;
  if (!rvdHost::readLastExclusiveScanTotal(
          work.overlapValid.get(), work.overlapOffsets.get(), candidateCount,
          stream, acceptedOverlaps, "read accepted overlap count") ||
      !rvdHost::readLastExclusiveScanTotal(
          work.vertexCounts.get(), work.vertexOffsets.get(), candidateCount,
          stream, acceptedVertices, "read accepted overlap vertices")) {
    return false;
  }

  int hostCapacityError = 0;
  if (!rvdHost::copyToHost(
          &hostCapacityError, capacityError, 1, stream,
          "read mortar scratch capacity")) {
    return false;
  }
  if (hostCapacityError != 0) {
    std::cerr << "[RVDContact] Fatal error: Mortar scratch polygon capacity exceeded ("
              << "MaxMortarScratchVertices = " << MaxMortarScratchVertices
              << "). Clipping buffer overflowed!" << std::endl;
    return false;
  }

  if (acceptedOverlaps == 0) return true;

  if (!pair.overlaps.allocate(acceptedOverlaps) ||
      !pair.bases.allocate(acceptedOverlaps) ||
      !pair.vertexCounts.allocate(acceptedOverlaps) ||
      !pair.vertices.allocate(acceptedVertices)) {
    return false;
  }

  countAndEmitMortarOverlapsKernel<<<
      rvdHost::blockCount(candidateCount), BlockSize, 0, stream>>>(
      work.candidatePairs.get(),
      candidateCount,
      modelInfos.get(),
      modelCount,
      contactRange,
      gapPenetrationTolerance,
      domainMin,
      invBaseScale,
      work.overlapOffsets.get(),
      work.vertexOffsets.get(),
      work.overlapValid.get(),
      pair.vertexCounts.get(),
      pair.overlaps.get(),
      pair.bases.get(),
      pair.vertices.get(),
      true,
      capacityError);
  if (!rvdHost::checkLaunch("launch mortar overlap emit pass")) return false;

  pair.overlapCount = acceptedOverlaps;
  pair.vertexCount = acceptedVertices;

  return rvdHost::synchronize(stream, "finish pair overlaps");
}

bool allocateMortarMeshBuffers(uint32_t overlapCount, uint32_t vertexCount) {
  if (!rvdHost::ensureCapacity(gMortarMeshBuffers.vertices, vertexCount) ||
      !rvdHost::ensureCapacity(gMortarMeshBuffers.vertexCounts, overlapCount) ||
      !rvdHost::ensureCapacity(gMortarMeshBuffers.vertexOffsets, overlapCount) ||
      !rvdHost::ensureCapacity(gMortarMeshBuffers.bases, overlapCount)) {
    return false;
  }
  gMortarMeshBuffers.vertexCount = vertexCount;
  return true;
}

bool computeSurfaceExposure(
    const std::vector<ModelRestriction> &models,
    uint32_t totalPatches,
    const BaseCellDomain &baseDomain,
    const cudaUtils::CudaBuffer<MortarOverlap> &overlaps,
    uint32_t overlapCount,
    cudaUtils::CudaBuffer<double4> &surfaceExposure,
    cudaStream_t stream) {
  cudaUtils::CudaBuffer<double4> claimedAreaMoment;
  if (!claimedAreaMoment.allocate(totalPatches) ||
      !rvdHost::zero(
          claimedAreaMoment.get(), totalPatches * sizeof(double4), stream,
          "clear claimed surface area")) {
    return false;
  }

  accumulateMortarExposureKernel<<<
      rvdHost::blockCount(overlapCount), BlockSize, 0, stream>>>(
      overlaps.get(), overlapCount, claimedAreaMoment.get());
  if (!rvdHost::checkLaunch("launch accumulateMortarExposureKernel")) {
    return false;
  }

  const float3 domainMin = make_float3(
      baseDomain.domainMin.x, baseDomain.domainMin.y, baseDomain.domainMin.z);
  for (const ModelRestriction &model : models) {
    if (model.totalPatches == 0) continue;

    finalizeSurfaceExposureKernel<<<
        rvdHost::blockCount(model.totalPatches), BlockSize, 0, stream>>>(
        model.devSurfaceGeometry.get(),
        model.totalPatches,
        model.surfaceBase,
        baseDomain.invBaseScale,
        domainMin,
        claimedAreaMoment.get(),
        surfaceExposure.get());
  }
  return rvdHost::checkLaunch("launch finalizeSurfaceExposureKernel");
}

bool buildContactCouplings(
    const cudaUtils::CudaBuffer<MortarOverlap> &overlaps,
    uint32_t overlapCount,
    const cudaUtils::CudaBuffer<uint32_t> &contactFaceBySurface,
    std::vector<ContactCoupling> &couplings,
    cudaStream_t stream) {
  cudaUtils::CudaBuffer<uint64_t> keysIn;
  cudaUtils::CudaBuffer<uint64_t> keysOut;
  cudaUtils::CudaBuffer<double> valuesIn;
  cudaUtils::CudaBuffer<double> valuesOut;
  cudaUtils::CudaBuffer<uint64_t> uniqueKeys;
  cudaUtils::CudaBuffer<double> aggregatedAreas;
  cudaUtils::CudaBuffer<int> runCount;
  if (!keysIn.allocate(overlapCount) ||
      !keysOut.allocate(overlapCount) ||
      !valuesIn.allocate(overlapCount) ||
      !valuesOut.allocate(overlapCount) ||
      !uniqueKeys.allocate(overlapCount) ||
      !aggregatedAreas.allocate(overlapCount) ||
      !runCount.allocate(1)) {
    return false;
  }

  mapOverlapsToCouplingKeysKernel<<<
      rvdHost::blockCount(overlapCount), BlockSize, 0, stream>>>(
      overlaps.get(),
      overlapCount,
      contactFaceBySurface.get(),
      keysIn.get(),
      valuesIn.get());
  if (!rvdHost::checkLaunch("launch mapOverlapsToCouplingKeysKernel")) {
    return false;
  }

  cudaUtils::CudaBuffer<uint8_t> reduceScratch;
  if (!rvdHost::sortPairs(
          keysIn.get(), keysOut.get(), valuesIn.get(), valuesOut.get(),
          int(overlapCount), reduceScratch, stream) ||
      !rvdHost::reduceByKeySum(
          keysOut.get(), uniqueKeys.get(), valuesOut.get(), aggregatedAreas.get(),
          runCount.get(), int(overlapCount), reduceScratch, stream)) {
    return false;
  }

  int hostRunCount = 0;
  if (!rvdHost::copyToHost(
          &hostRunCount, runCount.get(), 1, stream,
          "read contact coupling count")) {
    return false;
  }
  if (hostRunCount <= 0) return true;

  std::vector<uint64_t> hostKeys(hostRunCount);
  std::vector<double> hostAreas(hostRunCount);
  if (!uniqueKeys.downloadAsync(hostKeys.data(), hostRunCount, stream) ||
      !aggregatedAreas.downloadAsync(hostAreas.data(), hostRunCount, stream) ||
      !rvdHost::synchronize(stream, "download contact couplings")) {
    return false;
  }

  couplings.reserve(couplings.size() + size_t(hostRunCount));
  for (int r = 0; r < hostRunCount; ++r) {
    const uint64_t key = hostKeys[r];
    if (key == InvalidCouplingKey) continue;

    ContactCoupling coupling{};
    coupling.faceA = uint32_t(key >> 32);
    coupling.faceB = uint32_t(key & 0xFFFFFFFFull);
    coupling.couplingArea = hostAreas[r];
    couplings.push_back(coupling);
  }
  return true;
}

bool buildContactMesh(uint32_t overlapCount, std::vector<ContactMeshVertex> &contactMesh, cudaStream_t stream) {
  if (gMortarMeshBuffers.vertexCount == 0 ||
      gMortarMeshBuffers.vertexCounts.size() < overlapCount) {
    return true;
  }

  std::vector<uint32_t> hostCounts(overlapCount);
  std::vector<uint32_t> hostOffsets(overlapCount);
  std::vector<MortarBasis> hostBases(overlapCount);
  std::vector<double2> hostVertices(gMortarMeshBuffers.vertexCount);
  if (!gMortarMeshBuffers.vertexCounts.downloadAsync(
          hostCounts.data(), overlapCount, stream) ||
      !gMortarMeshBuffers.vertexOffsets.downloadAsync(
          hostOffsets.data(), overlapCount, stream) ||
      !gMortarMeshBuffers.bases.downloadAsync(
          hostBases.data(), overlapCount, stream) ||
      !gMortarMeshBuffers.vertices.downloadAsync(
          hostVertices.data(), hostVertices.size(), stream) ||
      !rvdHost::synchronize(stream, "download contact mesh")) {
    return false;
  }

  for (uint32_t i = 0; i < overlapCount; ++i) {
    const uint32_t vertexCount = hostCounts[i];
    if (vertexCount < 3) continue;

    const uint32_t offset = hostOffsets[i];
    const MortarBasis basis = hostBases[i];
    double2 centroid2D;
    polygonAreaAndCentroid2D(hostVertices.data() + offset, int(vertexCount), centroid2D);

    const glm::vec3 centroid(
        float(basis.origin.x + centroid2D.x * basis.u.x + centroid2D.y * basis.v.x),
        float(basis.origin.y + centroid2D.x * basis.u.y + centroid2D.y * basis.v.y),
        float(basis.origin.z + centroid2D.x * basis.u.z + centroid2D.y * basis.v.z));
    const glm::vec3 normal(float(basis.normal.x), float(basis.normal.y), float(basis.normal.z));

    for (uint32_t k = 0; k < vertexCount; ++k) {
      const uint32_t next = (k + 1 == vertexCount) ? 0 : k + 1;
      const double2 point0 = hostVertices[offset + k];
      const double2 point1 = hostVertices[offset + next];

      const glm::vec3 position0(
          float(basis.origin.x + point0.x * basis.u.x + point0.y * basis.v.x),
          float(basis.origin.y + point0.x * basis.u.y + point0.y * basis.v.y),
          float(basis.origin.z + point0.x * basis.u.z + point0.y * basis.v.z));
      const glm::vec3 position1(
          float(basis.origin.x + point1.x * basis.u.x + point1.y * basis.v.x),
          float(basis.origin.y + point1.x * basis.u.y + point1.y * basis.v.y),
          float(basis.origin.z + point1.x * basis.u.z + point1.y * basis.v.z));

      ContactMeshVertex v0{};
      ContactMeshVertex v1{};
      ContactMeshVertex v2{};
      v0.position = centroid;
      v0.normal = normal;
      v1.position = position0;
      v1.normal = normal;
      v2.position = position1;
      v2.normal = normal;

      contactMesh.push_back(v0);
      contactMesh.push_back(v1);
      contactMesh.push_back(v2);
    }
  }
  return true;
}

} // namespace

bool buildMortarOverlaps(
    const std::vector<ModelRestriction> &models,
    const std::vector<cudaGeometry::DeviceModelGeometryView> &geometries,
    const BaseCellDomain &baseDomain,
    double contactRange,
    cudaUtils::CudaBuffer<MortarOverlap> &overlaps,
    uint32_t &overlapCount,
    cudaUtils::CudaBuffer<double4> &surfaceExposure,
    cudaStream_t stream) {
  overlapCount = 0;

  const uint32_t modelCount = uint32_t(models.size());
  if (modelCount == 0) return true;

  uint32_t totalSurfacePatches = 0;
  for (const ModelRestriction &model : models) {
    totalSurfacePatches += model.totalPatches;
  }

  if (!initializeSurfaceExposure(
          models, totalSurfacePatches, baseDomain, surfaceExposure, stream)) {
    return false;
  }
  if (modelCount < 2 || totalSurfacePatches == 0) return true;

  float coordScale = glm::length(baseDomain.domainMax - baseDomain.domainMin);
  if (coordScale <= 0.0f) coordScale = 1.0f;
  const double gapPenetrationTolerance = 256.0 * DBL_EPSILON * double(coordScale);

  cudaUtils::CudaBuffer<DeviceModelInfo> modelInfos;
  if (!uploadModelInfos(models, stream, modelInfos)) return false;

  cudaUtils::CudaBuffer<int> capacityError;
  if (!capacityError.allocate(1) ||
      !rvdHost::zero(capacityError.get(), sizeof(int), stream, "clear mortar capacity error")) {
    return false;
  }

  const float3 domainMin = make_float3(baseDomain.domainMin.x, baseDomain.domainMin.y, baseDomain.domainMin.z);

  std::vector<PairOverlapBuffers> pairs;
  pairs.reserve(size_t(modelCount) * (modelCount - 1) / 2);
  PairWorkBuffers work;

  for (uint32_t modelA = 0; modelA < modelCount; ++modelA) {
    for (uint32_t modelB = modelA + 1; modelB < modelCount; ++modelB) {
      if (models[modelA].totalPatches == 0 || models[modelB].totalPatches == 0) {
        continue;
      }

      pairs.emplace_back();
      if (!buildPairOverlaps(
              models[modelA], models[modelB], geometries[modelB], modelInfos,
              modelCount, contactRange, gapPenetrationTolerance, domainMin,
              baseDomain.invBaseScale, coordScale, capacityError.get(), stream,
              work, pairs.back())) {
        return false;
      }
    }
  }

  uint32_t totalOverlaps = 0;
  uint32_t totalVertices = 0;
  for (const PairOverlapBuffers &pair : pairs) {
    totalOverlaps += pair.overlapCount;
    totalVertices += pair.vertexCount;
  }

  overlapCount = totalOverlaps;
  if (overlapCount == 0) return true;

  if (!rvdHost::ensureCapacity(overlaps, totalOverlaps) ||
      !allocateMortarMeshBuffers(totalOverlaps, totalVertices)) {
    return false;
  }

  uint32_t overlapOffset = 0;
  uint32_t vertexOffset = 0;
  for (const PairOverlapBuffers &pair : pairs) {
    if (pair.overlapCount == 0) continue;

    if (!rvdHost::copyDeviceToDeviceAsync(
            overlaps.get() + overlapOffset, pair.overlaps.get(),
            pair.overlapCount, stream, "copy mortar overlaps") ||
        !rvdHost::copyDeviceToDeviceAsync(
            gMortarMeshBuffers.bases.get() + overlapOffset, pair.bases.get(),
            pair.overlapCount, stream, "copy mortar bases") ||
        !rvdHost::copyDeviceToDeviceAsync(
            gMortarMeshBuffers.vertexCounts.get() + overlapOffset,
            pair.vertexCounts.get(), pair.overlapCount, stream,
            "copy mortar vertex counts") ||
        !rvdHost::copyDeviceToDeviceAsync(
            gMortarMeshBuffers.vertices.get() + vertexOffset,
            pair.vertices.get(), pair.vertexCount, stream,
            "copy mortar vertices")) {
      return false;
    }
    overlapOffset += pair.overlapCount;
    vertexOffset += pair.vertexCount;
  }

  cudaUtils::CudaBuffer<uint8_t> scanScratch;
  if (!rvdHost::exclusiveSum(
          gMortarMeshBuffers.vertexCounts.get(),
          gMortarMeshBuffers.vertexOffsets.get(),
          int(totalOverlaps),
          scanScratch,
          stream)) {
    return false;
  }

  if (!computeSurfaceExposure(
          models, totalSurfacePatches, baseDomain, overlaps, overlapCount,
          surfaceExposure, stream)) {
    return false;
  }

  return rvdHost::synchronize(stream, "finish mortar overlaps");
}

bool finalizeContactCouplings(
    const cudaUtils::CudaBuffer<MortarOverlap> &overlaps,
    uint32_t overlapCount,
    const cudaUtils::CudaBuffer<uint32_t> &contactFaceBySurface,
    RVDResult &result,
    cudaStream_t stream) {
  result.contactCouplings.clear();
  result.contactMesh.clear();

  if (overlapCount == 0) {
    gMortarMeshBuffers.reset();
    return true;
  }

  const bool couplingsBuilt = buildContactCouplings(overlaps, overlapCount, contactFaceBySurface, result.contactCouplings, stream);
  if (!couplingsBuilt) return false;

  if (!buildContactMesh(overlapCount, result.contactMesh, stream)) {
    return false;
  }

  gMortarMeshBuffers.reset();
  return true;
}

} // namespace voronoi
