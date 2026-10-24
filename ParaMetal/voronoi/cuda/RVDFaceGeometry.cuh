#pragma once

#include "../../cuda/CudaMath.cuh"
#include "RVDFacePartition.hpp"
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

namespace voronoi {

using cudaMath::add3;
using cudaMath::cross3;
using cudaMath::dot3;
using cudaMath::mul3;
using cudaMath::normalize3;
using cudaMath::sub3;

static constexpr uint32_t MaxFaceVertices = 64;

struct FacePoint2 {
  double x;
  double y;
};
struct FacePolygon {
  uint32_t count;
  FacePoint2 vertices[MaxFaceVertices];
};
__device__ inline FacePoint2 faceAdd(FacePoint2 a, FacePoint2 b) {
  return {a.x + b.x, a.y + b.y};
}
__device__ inline FacePoint2 faceSub(FacePoint2 a, FacePoint2 b) {
  return {a.x - b.x, a.y - b.y};
}
__device__ inline FacePoint2 faceMul(FacePoint2 a, double s) {
  return {a.x * s, a.y * s};
}
__device__ inline double faceDot(FacePoint2 a, FacePoint2 b) {
  return a.x * b.x + a.y * b.y;
}
__device__ inline double faceCross(FacePoint2 a, FacePoint2 b) {
  return a.x * b.y - a.y * b.x;
}
__device__ inline FacePoint2 faceProject(double3 p, double3 o, double3 u,
                                         double3 v) {
  const double3 d = sub3(p, o);
  return {dot3(d, u), dot3(d, v)};
}
__device__ inline double facePolygonArea(const FacePolygon &p) {
  if (p.count < 3)
    return 0.0;
  const FacePoint2 origin = p.vertices[0];
  double twice = 0.0;
  for (uint32_t i = 1; i + 1 < p.count; ++i) {
    twice += faceCross(faceSub(p.vertices[i], origin),
                       faceSub(p.vertices[i + 1], origin));
  }
  return 0.5 * twice;
}
__device__ inline bool facePointInsideConvex(FacePoint2 p,
                                             const FacePolygon &polygon,
                                             double tolerance) {
  for (uint32_t i = 0; i < polygon.count; ++i) {
    const FacePoint2 a = polygon.vertices[i];
    const FacePoint2 b = polygon.vertices[(i + 1) % polygon.count];
    const FacePoint2 edge = faceSub(b, a);
    const double length = hypot(edge.x, edge.y);
    if (!(length > 0.0))
      continue;
    if (faceCross(edge, faceSub(p, a)) < -tolerance * length)
      return false;
  }
  return true;
}
__device__ inline bool facePolygonIntegral(const FacePolygon &p, double3 o,
                                           double3 u, double3 v, double &area,
                                           double3 &moment, double &roundoff) {
  area = 0.0;
  moment = make_double3(0.0, 0.0, 0.0);
  roundoff = 0.0;
  if (p.count < 3)
    return true;
  const FacePoint2 p0 = p.vertices[0];
  double mu = 0.0, mv = 0.0, absArea = 0.0;
  // Convex fan: avoids cancellation from large absolute shoelace products
  // and evaluates both area and first moments in the same local origin.
  for (uint32_t i = 1; i + 1 < p.count; ++i) {
    const FacePoint2 p1 = p.vertices[i], p2 = p.vertices[i + 1];
    const double triangleArea =
        0.5 * faceCross(faceSub(p1, p0), faceSub(p2, p0));
    const double cu = (p0.x + p1.x + p2.x) / 3.0;
    const double cv = (p0.y + p1.y + p2.y) / 3.0;
    area += triangleArea;
    mu += triangleArea * cu;
    mv += triangleArea * cv;
    absArea += fabs(triangleArea);
  }
  roundoff = 256.0 * DBL_EPSILON * absArea;
  moment = add3(mul3(o, area), add3(mul3(u, mu), mul3(v, mv)));
  return isfinite(area) && isfinite(moment.x) && isfinite(moment.y) &&
         isfinite(moment.z);
}
__device__ inline void faceReverse(FacePolygon &p) {
  for (uint32_t i = 0, j = p.count - 1; i < j; ++i, --j) {
    const FacePoint2 t = p.vertices[i];
    p.vertices[i] = p.vertices[j];
    p.vertices[j] = t;
  }
}
__device__ inline bool
faceSetupPolygon(uint32_t faceId, const uint32_t *offsets,
                 const double3 *vertices, const BaseFaceMeasure &measure,
                 FacePolygon &polygon, double3 &origin, double3 &axisU,
                 double3 &axisV, double3 &normal, double &extent,
                 double3 &bboxMin, double3 &bboxMax) {
  const uint32_t begin = offsets[faceId], end = offsets[faceId + 1];
  const uint32_t count = end - begin;
  if (count < 3 || count > MaxFaceVertices || !measure.valid)
    return false;
  normal = normalize3(measure.normal);
  if (!(dot3(normal, normal) > 0.9))
    return false;
  const double3 ref =
      fabs(normal.x) < 0.8 ? make_double3(1, 0, 0) : make_double3(0, 1, 0);
  axisU = normalize3(cross3(ref, normal));
  axisV = cross3(normal, axisU);
  origin = vertices[begin];
  bboxMin = bboxMax = origin;
  FacePoint2 center{0.0, 0.0};
  for (uint32_t i = 0; i < count; ++i) {
    const double3 w = vertices[begin + i];
    bboxMin = make_double3(fmin(bboxMin.x, w.x), fmin(bboxMin.y, w.y),
                           fmin(bboxMin.z, w.z));
    bboxMax = make_double3(fmax(bboxMax.x, w.x), fmax(bboxMax.y, w.y),
                           fmax(bboxMax.z, w.z));
    polygon.vertices[i] = faceProject(w, origin, axisU, axisV);
    center = faceAdd(center, polygon.vertices[i]);
  }
  center.x /= count;
  center.y /= count;
  // Base-cell vertex triplets are not guaranteed to be stored in boundary
  // order. Ordering the vertices of a strictly convex face about an interior
  // point produces the exact perimeter order without any polygon graph.
  double vertexAngle[MaxFaceVertices];
  for (uint32_t i = 0; i < count; ++i) {
    const FacePoint2 p = polygon.vertices[i];
    vertexAngle[i] = atan2(p.y - center.y, p.x - center.x);
  }
  for (uint32_t i = 1; i < count; ++i) {
    const FacePoint2 p = polygon.vertices[i];
    const double key = vertexAngle[i];
    uint32_t j = i;
    while (j > 0 && vertexAngle[j - 1] > key) {
      polygon.vertices[j] = polygon.vertices[j - 1];
      vertexAngle[j] = vertexAngle[j - 1];
      --j;
    }
    polygon.vertices[j] = p;
    vertexAngle[j] = key;
  }
  extent = hypot(bboxMax.x - bboxMin.x,
                 hypot(bboxMax.y - bboxMin.y, bboxMax.z - bboxMin.z));
  if (!isfinite(extent) || !(extent > 0.0))
    return false;

  // Coincident vertices can arrive through different clipping paths with a
  // few ULPs of projection noise. Keep one canonical graph vertex for them;
  // otherwise the zero-length boundary edge is dropped later while its two
  // endpoint IDs remain disconnected.
  const double vertexTolerance = 128.0 * DBL_EPSILON * extent;
  uint32_t uniqueCount = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const FacePoint2 p = polygon.vertices[i];
    const FacePoint2 previous =
        uniqueCount > 0 ? polygon.vertices[uniqueCount - 1] : p;
    if (uniqueCount == 0 ||
        hypot(p.x - previous.x, p.y - previous.y) > vertexTolerance) {
      polygon.vertices[uniqueCount++] = p;
    }
  }
  if (uniqueCount > 1 &&
      hypot(polygon.vertices[0].x - polygon.vertices[uniqueCount - 1].x,
            polygon.vertices[0].y - polygon.vertices[uniqueCount - 1].y) <=
          vertexTolerance) {
    --uniqueCount;
  }
  if (uniqueCount < 3)
    return false;
  polygon.count = uniqueCount;
  double area = facePolygonArea(polygon);
  if (area < 0.0) {
    faceReverse(polygon);
    area = -area;
  }
  return isfinite(area) && area > 0.0;
}
__device__ inline uint32_t
faceEmitRegion(FaceRegion *groups, uint32_t &groupCount, uint32_t capacity,
               double area, double3 moment, double3 normal, uint32_t materialA,
               uint32_t materialB, double areaError) {
  if (!(area > 0.0) || !isfinite(area) || !isfinite(moment.x) ||
      !isfinite(moment.y) || !isfinite(moment.z)) {
    return InvalidFacePolygon;
  }
  if (groupCount == capacity)
    return TooManyRegions;
  FaceRegion &region = groups[groupCount++];
  region = FaceRegion{};
  region.measure.area = area;
  region.measure.areaError = areaError;
  region.measure.moment = moment;
  region.measure.normal = normal;
  region.measure.valid = 1u;
  region.materialA = materialA;
  region.materialB = materialB;
  return FacePartitionSuccess;
}
__device__ inline uint32_t faceValidateConservation(
    const FaceRegion *groups, uint32_t groupCount, const FacePolygon &boundary,
    double3 origin, double3 u, double3 v, double scale,
    const BaseFaceMeasure &base) {
  double totalArea = 0.0;
  double3 totalMoment = make_double3(0, 0, 0);
  double error = 0.0;
  for (uint32_t i = 0; i < groupCount; ++i) {
    const auto &r = groups[i].measure;
    totalArea += r.area;
    totalMoment = add3(totalMoment, r.moment);
    error += r.areaError;
  }
  double polygonArea, baseRoundoff;
  double3 polygonMoment;
  if (!facePolygonIntegral(boundary, origin, u, v, polygonArea, polygonMoment,
                           baseRoundoff))
    return InvalidFacePolygon;
  const double tol =
      fmax(1024.0 * DBL_EPSILON * polygonArea, error + baseRoundoff) +
      256.0 * DBL_EPSILON * scale * scale;
  if (fabs(totalArea - polygonArea) > tol) {
    return AreaConservation;
  }
  const double momentScale =
      fmax(fabs(origin.x), fmax(fabs(origin.y), fabs(origin.z))) + scale;
  const double mTol =
      2048.0 * DBL_EPSILON * polygonArea * momentScale + tol * momentScale;
  const double3 delta = sub3(totalMoment, polygonMoment);
  if (fabs(delta.x) > mTol || fabs(delta.y) > mTol || fabs(delta.z) > mTol)
    return MomentConservation;
  // The separate canonical measure may be sourced from the base-cell
  // integration, so validate it too, without replacing it opportunistically.
  // The base-cell planes originate in the project's float input geometry.
  // Bound possible vertex-position quantization by perimeter * position ULP,
  // rather than applying any minimum positive-area cutoff.
  double perimeter = 0.0;
  for (uint32_t i = 0; i < boundary.count; ++i) {
    const FacePoint2 edge = faceSub(boundary.vertices[(i + 1) % boundary.count],
                                    boundary.vertices[i]);
    perimeter += hypot(edge.x, edge.y);
  }
  const double coordinateMagnitude =
      fmax(fmax(fabs(origin.x), fabs(origin.y)), fabs(origin.z));
  const double positionError =
      2.0 * FLT_EPSILON * fmax(coordinateMagnitude, scale);
  const double baseTol =
      base.areaError + perimeter * positionError + baseRoundoff;
  if (fabs(base.area - polygonArea) > baseTol)
    return BaseAreaConservation;
  const double3 referenceDelta = sub3(base.moment, polygonMoment);
  const double referenceMomentTol = baseTol * (coordinateMagnitude + scale) +
                                    512.0 * DBL_EPSILON *
                                        fmax(polygonArea, base.area) *
                                        (coordinateMagnitude + scale);
  if (fabs(referenceDelta.x) > referenceMomentTol ||
      fabs(referenceDelta.y) > referenceMomentTol ||
      fabs(referenceDelta.z) > referenceMomentTol)
    return BaseMomentConservation;
  return FacePartitionSuccess;
}

} // namespace voronoi
