#include "RVDBaseFacePartition.cuh"
#include "RVDBaseFaceTopology.hpp"
#include "RVDModelGeometry.cuh"
#include "RVDPredicates.cuh"
#include "RVDHostUtils.cuh"
#include "../../cuda/CudaMath.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

namespace voronoi {

using cudaMath::add3, cudaMath::cross3, cudaMath::dot3, cudaMath::mul3, cudaMath::normalize3, cudaMath::sub3;
using rvdDevice::orient2dFilteredDouble;

namespace {

constexpr uint32_t MaxFaceVertices = 64, BaseVertexCanonicalBit = 0x80000000u, MaxRegionsPerCutFace = 16;
constexpr uint32_t UnresolvedMaterial = 0xfffffffeu, OverlayBlockSize = 64, OverlayBatchSize = 2048;

enum FaceCanonicalKeyType : uint32_t {
  FaceCanonicalBaseVertex = 0,
  FaceCanonicalCadVertex = 1,
  FaceCanonicalIntersection = 2,
};

struct FaceCanonicalKey {
  uint32_t type = FaceCanonicalBaseVertex;
  uint32_t a = InvalidId;
  uint32_t b = InvalidId;
  uint32_t c = InvalidId;
};

__device__ inline bool faceCanonicalKeyEqual(
    const FaceCanonicalKey &lhs, const FaceCanonicalKey &rhs) {
  return lhs.type == rhs.type && lhs.a == rhs.a && lhs.b == rhs.b &&
         lhs.c == rhs.c;
}

struct FacePoint2 { double x, y; };

struct FacePolygon {
  uint32_t count;
  FacePoint2 vertices[MaxFaceVertices];
  uint32_t sourceVertexId[MaxFaceVertices];
};

__device__ inline FacePoint2 faceAdd(FacePoint2 a, FacePoint2 b) { return {a.x + b.x, a.y + b.y}; }
__device__ inline FacePoint2 faceSub(FacePoint2 a, FacePoint2 b) { return {a.x - b.x, a.y - b.y}; }
__device__ inline FacePoint2 faceMul(FacePoint2 a, double s) { return {a.x * s, a.y * s}; }
__device__ inline double faceDot(FacePoint2 a, FacePoint2 b) { return a.x * b.x + a.y * b.y; }
__device__ inline double faceCross(FacePoint2 a, FacePoint2 b) { return a.x * b.y - a.y * b.x; }

__device__ inline FacePoint2 faceProject(double3 p, double3 o, double3 u, double3 v) {
  const double3 d = sub3(p, o); return {dot3(d, u), dot3(d, v)};
}

__device__ inline bool aabbContainsOrIntersects(float3 pt, float3 bMin, float3 bMax, float tol = 1e-4f) {
  return pt.x >= bMin.x - tol && pt.x <= bMax.x + tol && pt.y >= bMin.y - tol && pt.y <= bMax.y + tol && pt.z >= bMin.z - tol && pt.z <= bMax.z + tol;
}

__device__ inline double facePolygonArea(const FacePolygon &p) {
  if (p.count < 3) return 0.0;
  double twice = 0.0;
  for (uint32_t i = 1; i + 1 < p.count; ++i) twice += faceCross(faceSub(p.vertices[i], p.vertices[0]), faceSub(p.vertices[i + 1], p.vertices[0]));
  return 0.5 * twice;
}

__device__ inline bool facePolygonIntegral(const FacePolygon &p, double3 o, double3 u, double3 v,
                                           double &area, double3 &moment, double &roundoff) {
  area = roundoff = 0.0; moment = make_double3(0, 0, 0);
  if (p.count < 3) return true;
  double mu = 0.0, mv = 0.0, absArea = 0.0;
  for (uint32_t i = 1; i + 1 < p.count; ++i) {
    const FacePoint2 p1 = p.vertices[i], p2 = p.vertices[i + 1];
    const double ta = 0.5 * faceCross(faceSub(p1, p.vertices[0]), faceSub(p2, p.vertices[0]));
    area += ta; mu += ta * ((p.vertices[0].x + p1.x + p2.x) / 3.0);
    mv += ta * ((p.vertices[0].y + p1.y + p2.y) / 3.0); absArea += fabs(ta);
  }
  roundoff = 256.0 * DBL_EPSILON * absArea;
  moment = add3(mul3(o, area), add3(mul3(u, mu), mul3(v, mv)));
  return isfinite(area) && isfinite(moment.x) && isfinite(moment.y) && isfinite(moment.z);
}

__device__ inline void faceReverse(FacePolygon &p) {
  for (uint32_t i = 0, j = p.count - 1; i < j; ++i, --j) {
    const FacePoint2 t = p.vertices[i]; p.vertices[i] = p.vertices[j]; p.vertices[j] = t;
    const uint32_t id = p.sourceVertexId[i]; p.sourceVertexId[i] = p.sourceVertexId[j]; p.sourceVertexId[j] = id;
  }
}

__device__ inline bool faceSetupPolygon(
    uint32_t faceId, const uint32_t *offsets, const double3 *vertices,
    const uint32_t *vertexIds, const BaseFaceMeasure &measure,
    FacePolygon &polygon, double3 &origin, double3 &axisU, double3 &axisV,
    double3 &normal, double &extent) {
  const uint32_t begin = offsets[faceId], end = offsets[faceId + 1];
  const uint32_t count = end - begin;
  if (count < 3 || count > MaxFaceVertices || !measure.valid) return false;
  normal = normalize3(measure.normal);
  if (!(dot3(normal, normal) > 0.9)) return false;
  axisU = rvdHost::makeBasisU(normal);
  axisV = cross3(normal, axisU);
  double3 bboxMin = vertices[begin], bboxMax = vertices[begin];
  origin = vertices[begin];
  FacePoint2 center{0.0, 0.0};
  for (uint32_t i = 0; i < count; ++i) {
    const double3 w = vertices[begin + i];
    bboxMin = make_double3(fmin(bboxMin.x, w.x), fmin(bboxMin.y, w.y), fmin(bboxMin.z, w.z));
    bboxMax = make_double3(fmax(bboxMax.x, w.x), fmax(bboxMax.y, w.y), fmax(bboxMax.z, w.z));
    polygon.vertices[i] = faceProject(w, origin, axisU, axisV);
    polygon.sourceVertexId[i] = vertexIds != nullptr ? vertexIds[begin + i] : (begin + i);
    center = faceAdd(center, polygon.vertices[i]);
  }
  center.x /= count; center.y /= count;
  double vertexAngle[MaxFaceVertices];
  for (uint32_t i = 0; i < count; ++i) {
    vertexAngle[i] = atan2(polygon.vertices[i].y - center.y, polygon.vertices[i].x - center.x);
  }
  for (uint32_t i = 1; i < count; ++i) {
    const FacePoint2 p = polygon.vertices[i];
    const uint32_t sId = polygon.sourceVertexId[i];
    const double key = vertexAngle[i];
    uint32_t j = i;
    while (j > 0 && vertexAngle[j - 1] > key) {
      polygon.vertices[j] = polygon.vertices[j - 1];
      polygon.sourceVertexId[j] = polygon.sourceVertexId[j - 1];
      vertexAngle[j] = vertexAngle[j - 1];
      --j;
    }
    polygon.vertices[j] = p; polygon.sourceVertexId[j] = sId; vertexAngle[j] = key;
  }
  extent = hypot(bboxMax.x - bboxMin.x, hypot(bboxMax.y - bboxMin.y, bboxMax.z - bboxMin.z));
  if (!isfinite(extent) || !(extent > 0.0)) return false;

  const double vertexTolerance = 128.0 * DBL_EPSILON * extent;
  uint32_t uniqueCount = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const FacePoint2 p = polygon.vertices[i];
    const FacePoint2 prev = uniqueCount > 0 ? polygon.vertices[uniqueCount - 1] : p;
    if (uniqueCount == 0 || hypot(p.x - prev.x, p.y - prev.y) > vertexTolerance) {
      polygon.vertices[uniqueCount] = p;
      polygon.sourceVertexId[uniqueCount++] = polygon.sourceVertexId[i];
    }
  }
  if (uniqueCount > 1 &&
      hypot(polygon.vertices[0].x - polygon.vertices[uniqueCount - 1].x,
            polygon.vertices[0].y - polygon.vertices[uniqueCount - 1].y) <= vertexTolerance) {
    --uniqueCount;
  }
  if (uniqueCount < 3) return false;
  polygon.count = uniqueCount;
  double area = facePolygonArea(polygon);
  if (area < 0.0) {
    faceReverse(polygon);
    area = -area;
  }
  return isfinite(area) && area > 0.0;
}

__device__ inline uint32_t faceEmitRegion(
    FaceRegion *groups, uint32_t &groupCount, uint32_t capacity,
    double area, double3 moment, double3 normal, uint32_t materialA,
    uint32_t materialB, double areaError) {
  if (!(area > 0.0) || !isfinite(area) || !isfinite(moment.x) || !isfinite(moment.y) || !isfinite(moment.z)) {
    return InvalidFacePolygon;
  }
  if (groupCount == capacity) return TooManyRegions;
  FaceRegion &r = groups[groupCount++];
  r = FaceRegion{};
  r.measure.area = area; r.measure.areaError = areaError; r.measure.moment = moment;
  r.measure.normal = normal; r.measure.valid = 1u;
  r.materialA = materialA; r.materialB = materialB;
  return FacePartitionSuccess;
}

__device__ inline uint32_t faceValidateConservation(
    const FaceRegion *groups, uint32_t groupCount, const FacePolygon &boundary,
    double3 origin, double3 u, double3 v, double scale) {
  double totalArea = 0.0, error = 0.0, polygonArea, baseRoundoff;
  double3 totalMoment = make_double3(0, 0, 0), polygonMoment;
  for (uint32_t i = 0; i < groupCount; ++i) {
    totalArea += groups[i].measure.area; totalMoment = add3(totalMoment, groups[i].measure.moment); error += groups[i].measure.areaError;
  }
  if (!facePolygonIntegral(boundary, origin, u, v, polygonArea, polygonMoment, baseRoundoff)) return InvalidFacePolygon;
  const double tol = fmax(1024.0 * DBL_EPSILON * polygonArea, error + baseRoundoff) + 256.0 * DBL_EPSILON * scale * scale;
  if (fabs(totalArea - polygonArea) > tol) {
    return AreaConservation;
  }
  const double momentScale = fmax(fabs(origin.x), fmax(fabs(origin.y), fabs(origin.z))) + scale;
  const double mTol = (2048.0 * DBL_EPSILON * polygonArea + tol) * momentScale;
  const double3 delta = sub3(totalMoment, polygonMoment);
  if (fabs(delta.x) > mTol || fabs(delta.y) > mTol || fabs(delta.z) > mTol) {
    return MomentConservation;
  }
  return FacePartitionSuccess;
}

struct Segment {
  FacePoint2 a, b;
  FaceCanonicalKey keyA, keyB;
  uint32_t segmentId = InvalidId, runtimeModelId = InvalidId, segmentType = uint32_t(FaceCutSegmentType::Crossing);
};
struct Edge { int a = -1, b = -1; };
struct Cycle {
  uint32_t firstPoint = 0, pointCount = 0;
  double area = 0.0, areaRoundoff = 0.0;
  FacePoint2 moment{};
  int parent = -1;
};
struct SplitParam { double t = 0.0; FaceCanonicalKey partnerKey; };

struct FaceOverlayWorkspace {
  Segment *segments = nullptr, *splitSegments = nullptr;
  double3 *segmentPlaneNormals = nullptr;
  FacePoint2 *points = nullptr, *cyclePoints = nullptr, *netMoment = nullptr;
  int *pointEdge = nullptr, *cycleOfDirectedEdge = nullptr, *regionOfCycle = nullptr;
  int *splitSources = nullptr, *splitPointA = nullptr, *splitPointB = nullptr;
  FaceCanonicalKey *pointKey = nullptr;
  Edge *edges = nullptr;
  uint32_t *edgeModelId = nullptr, *edgeSegmentType = nullptr, *regionMaterial = nullptr;
  Cycle *cycles = nullptr;
  bool *regionResolved = nullptr, *regionCoplanar = nullptr;
  double *netArea = nullptr, *netRoundoff = nullptr;
  uint8_t *visited = nullptr;
  FaceRegion *groups = nullptr;
  SplitParam *splitParams = nullptr;
  uint32_t segmentCapacity = 0, splitSegmentCapacity = 0, pointCapacity = 0, edgeCapacity = 0;
  uint32_t cycleCapacity = 0, cyclePointCapacity = 0, groupCapacity = 0, splitCapacity = 0;
  uint32_t segmentCount = 0, edgeCount = 0, cycleCount = 0, cyclePointCount = 0, groupCount = 0;
};

__device__ inline bool faceClipSegmentToConvex(FacePoint2 &a, FacePoint2 &b,
                                               const FacePolygon &face, double lineEps) {
  double t0 = 0.0, t1 = 1.0;
  const FacePoint2 d = faceSub(b, a);
  const double segmentLength = hypot(d.x, d.y);
  for (uint32_t i = 0; i < face.count; ++i) {
    const FacePoint2 v = face.vertices[i];
    const FacePoint2 e = faceSub(face.vertices[(i + 1) % face.count], v);
    const double edgeLength = hypot(e.x, e.y);
    const double denom = faceCross(e, d);
    const double dist = faceCross(e, faceSub(a, v));
    const double determinantScale = edgeLength * segmentLength;
    if (!(determinantScale > 0.0) || fabs(denom) <= 64.0 * DBL_EPSILON * determinantScale) {
      if (dist < -lineEps * edgeLength) return false;
      continue;
    }
    const double t = -dist / denom;
    if (denom > 0.0) t0 = fmax(t0, t);
    else t1 = fmin(t1, t);
    if (t0 > t1) return false;
  }
  const FacePoint2 clippedA = {a.x + d.x * t0, a.y + d.y * t0};
  const FacePoint2 clippedB = {a.x + d.x * t1, a.y + d.y * t1};
  if (hypot(clippedB.x - clippedA.x, clippedB.y - clippedA.y) <= lineEps) return false;
  a = clippedA;
  b = clippedB;
  return true;
}


__device__ inline uint32_t faceAddSegment(FaceOverlayWorkspace &work,
                                          FacePoint2 a, FacePoint2 b,
                                          FaceCanonicalKey keyA, FaceCanonicalKey keyB,
                                          uint32_t segmentId, const FacePolygon &boundary,
                                          double scale, double3 planeNormal,
                                          uint32_t runtimeModelId, uint32_t segmentType) {
  const double lineEps = 1024.0 * DBL_EPSILON * fmax(scale, 1.0);
  if (!faceClipSegmentToConvex(a, b, boundary, lineEps)) return FacePartitionSuccess;
  const FacePoint2 e = faceSub(b, a);
  const double len = hypot(e.x, e.y);
  if (!(len > lineEps)) return FacePartitionSuccess;

  double nx = e.y / len, ny = -e.x / len, c = -(nx * a.x + ny * a.y);
  if (nx < 0.0 || (nx == 0.0 && ny < 0.0)) { nx = -nx; ny = -ny; c = -c; }
  double minimum = DBL_MAX, maximum = -DBL_MAX;
  for (uint32_t i = 0; i < boundary.count; ++i) {
    const double d = nx * boundary.vertices[i].x + ny * boundary.vertices[i].y + c;
    minimum = fmin(minimum, d); maximum = fmax(maximum, d);
  }
  if (minimum >= -lineEps || maximum <= lineEps) return FacePartitionSuccess;

  for (uint32_t k = 0; k < work.segmentCount; ++k) {
    const Segment &q = work.segments[k];
    if (q.runtimeModelId != runtimeModelId || q.segmentType != segmentType) continue;
    if ((fabs(q.a.x - a.x) <= lineEps && fabs(q.a.y - a.y) <= lineEps &&
         fabs(q.b.x - b.x) <= lineEps && fabs(q.b.y - b.y) <= lineEps) ||
        (fabs(q.a.x - b.x) <= lineEps && fabs(q.a.y - b.y) <= lineEps &&
         fabs(q.b.x - a.x) <= lineEps && fabs(q.b.y - a.y) <= lineEps)) return FacePartitionSuccess;
  }
  if (work.segmentCount == work.segmentCapacity) return TooManySegments;
  work.segmentPlaneNormals[work.segmentCount] = planeNormal;
  work.segments[work.segmentCount++] = {a, b, keyA, keyB, segmentId, runtimeModelId, segmentType};
  return FacePartitionSuccess;
}

__device__ inline bool facePointInPolygon(FacePoint2 p, const FacePoint2 *polygon, int count) {
  int winding = 0;
  for (int i = 0; i < count; ++i) {
    const FacePoint2 a = polygon[i], b = polygon[(i + 1) % count];
    if (a.y <= p.y) {
      if (b.y > p.y && faceCross(faceSub(b, a), faceSub(p, a)) > 0.0) ++winding;
    } else {
      if (b.y <= p.y && faceCross(faceSub(b, a), faceSub(p, a)) < 0.0) --winding;
    }
  }
  return winding != 0;
}

__device__ inline void faceComputeCycleAreaAndMoment(
    const FacePoint2 *points, uint32_t count, double &area, double &roundoff,
    FacePoint2 &moment) {
  double twiceArea = 0.0, absolute = 0.0;
  moment = FacePoint2{0.0, 0.0};
  for (uint32_t i = 0; i < count; ++i) {
    const FacePoint2 &a = points[i], &b = points[(i + 1) % count];
    const double c = faceCross(a, b);
    twiceArea += c;
    absolute += fabs(c);
    moment.x += (a.x + b.x) * c / 6.0;
    moment.y += (a.y + b.y) * c / 6.0;
  }
  area = 0.5 * twiceArea;
  roundoff = 128.0 * DBL_EPSILON * absolute;
}

__device__ inline bool faceSegmentIntersection(const Segment &s0, const Segment &s1,
                                               FacePoint2 &out, double eps) {
  const FacePoint2 d0 = faceSub(s0.b, s0.a), d1 = faceSub(s1.b, s1.a);
  const auto denom = orient2dFilteredDouble(
      make_double2(0.0, 0.0), make_double2(d0.x, d0.y), make_double2(d1.x, d1.y));
  if (denom.sign == 0 || denom.value == 0.0) return false;

  const auto tOrient = orient2dFilteredDouble(
      make_double2(s1.a.x, s1.a.y), make_double2(s1.b.x, s1.b.y), make_double2(s0.a.x, s0.a.y));
  const auto uOrient = orient2dFilteredDouble(
      make_double2(s0.a.x, s0.a.y), make_double2(s0.b.x, s0.b.y), make_double2(s1.a.x, s1.a.y));
  const double t = tOrient.value / denom.value, u = -uOrient.value / denom.value;
  const double len0 = hypot(d0.x, d0.y), len1 = hypot(d1.x, d1.y);
  if (t < -eps / fmax(len0, 1.0e-300) || t > 1.0 + eps / fmax(len0, 1.0e-300) ||
      u < -eps / fmax(len1, 1.0e-300) || u > 1.0 + eps / fmax(len1, 1.0e-300)) {
    return false;
  }
  out = faceAdd(s0.a, faceMul(d0, t));
  return true;
}

__device__ inline FaceCanonicalKey faceIntersectionKey(uint32_t segmentA, uint32_t segmentB) {
  FaceCanonicalKey key{FaceCanonicalIntersection, segmentA, segmentB, InvalidId};
  if (key.a > key.b) { const uint32_t t = key.a; key.a = key.b; key.b = t; }
  return key;
}

__device__ inline int faceSplitSegments(
    FaceOverlayWorkspace &work,
    Segment *output, int *outputSource, double eps) {
  int outputCount = 0;
  const Segment *input = work.segments;
  const int inputCount = int(work.segmentCount);
  for (int i = 0; i < inputCount; ++i) {
    SplitParam *params = work.splitParams;
    int paramCount = 0;
    params[paramCount++] = {0.0, FaceCanonicalKey{}};
    params[paramCount++] = {1.0, FaceCanonicalKey{}};
    const FacePoint2 d = faceSub(input[i].b, input[i].a);
    const double sqLen = faceDot(d, d);
    const double tEps = eps / fmax(sqrt(sqLen), 1.0e-300);
    for (int j = 0; j < inputCount; ++j) {
      if (i == j) continue;
      const FaceCanonicalKey interKey = faceIntersectionKey(input[i].segmentId, input[j].segmentId);
      const FacePoint2 otherDir = faceSub(input[j].b, input[j].a);
      const double otherSqLen = faceDot(otherDir, otherDir);
      FacePoint2 p;
      if (faceSegmentIntersection(input[i], input[j], p, eps) && sqLen > 0.0) {
        const double t = faceDot(faceSub(p, input[i].a), d) / sqLen;
        if (t >= -tEps && t <= 1.0 + tEps) {
          if (paramCount >= int(work.splitCapacity)) return -1;
          FaceCanonicalKey partner = interKey;
          if (otherSqLen > 0.0) {
            const double oT = faceDot(faceSub(p, input[j].a), otherDir) / otherSqLen;
            if (oT <= tEps) partner = input[j].keyA;
            else if (oT >= 1.0 - tEps) partner = input[j].keyB;
          }
          params[paramCount++] = {fmin(1.0, fmax(0.0, t)), partner};
        }
      }
      const FacePoint2 endpoints[2] = {input[j].a, input[j].b};
      for (int e = 0; e < 2; ++e) {
        if (!(sqLen > 0.0)) continue;
        const double t = faceDot(faceSub(endpoints[e], input[i].a), d) / sqLen;
        if (t <= tEps || t >= 1.0 - tEps) continue;
        const FacePoint2 proj = faceAdd(input[i].a, faceMul(d, t));
        if (hypot(proj.x - endpoints[e].x, proj.y - endpoints[e].y) > eps) continue;
        if (paramCount >= int(work.splitCapacity)) return -1;
        params[paramCount++] = {t, e == 0 ? input[j].keyA : input[j].keyB};
      }
    }
    for (int a = 1; a < paramCount; ++a) {
      const SplitParam val = params[a];
      int b = a - 1;
      while (b >= 0 && params[b].t > val.t) { params[b + 1] = params[b]; --b; }
      params[b + 1] = val;
    }
    int uniqueCount = 0;
    for (int p = 0; p < paramCount; ++p) {
      if (uniqueCount > 0 && fabs(params[p].t - params[uniqueCount - 1].t) <= tEps) {
        if (params[p].t == 0.0 || params[p].t == 1.0) params[uniqueCount - 1] = params[p];
        continue;
      }
      params[uniqueCount++] = params[p];
    }
    for (int p = 0; p + 1 < uniqueCount; ++p) {
      if (params[p + 1].t - params[p].t <= tEps) continue;
      if (outputCount >= int(work.splitSegmentCapacity)) return -1;
      Segment s{};
      s.a = params[p].t == 0.0 ? input[i].a : faceAdd(input[i].a, faceMul(d, params[p].t));
      s.b = params[p + 1].t == 1.0 ? input[i].b : faceAdd(input[i].a, faceMul(d, params[p + 1].t));
      s.keyA = params[p].t == 0.0 ? input[i].keyA : params[p].partnerKey;
      s.keyB = params[p + 1].t == 1.0 ? input[i].keyB : params[p + 1].partnerKey;
      s.segmentId = input[i].segmentId;
      s.runtimeModelId = input[i].runtimeModelId;
      s.segmentType = input[i].segmentType;
      output[outputCount] = s;
      outputSource[outputCount++] = i;
    }
  }
  return outputCount;
}

__device__ inline int faceFindSplitPoint(const FacePoint2 *points, int count,
                                         int canonicalCount, const int *pointEdge,
                                         int edge, FacePoint2 p, double eps) {
  for (int i = 0; i < count; ++i) {
    if (points[i].x == p.x && points[i].y == p.y &&
        (i < canonicalCount || pointEdge == nullptr || pointEdge[i] == edge)) return i;
  }
  int bestCanonical = -1; double bestCanonicalDist = eps;
  for (int i = 0; i < canonicalCount; ++i) {
    const double dist = hypot(points[i].x - p.x, points[i].y - p.y);
    if (dist <= bestCanonicalDist) { bestCanonical = i; bestCanonicalDist = dist; }
  }
  if (bestCanonical >= 0) return bestCanonical;
  if (pointEdge == nullptr) return -1;
  int best = -1; double bestDist = eps;
  for (int i = canonicalCount; i < count; ++i) {
    if (pointEdge[i] != edge) continue;
    const double dist = hypot(points[i].x - p.x, points[i].y - p.y);
    if (dist <= bestDist) { best = i; bestDist = dist; }
  }
  return best;
}

__device__ inline int faceFindCanonicalPoint(
    const FaceCanonicalKey *keys, const FacePoint2 *points, int count,
    const FaceCanonicalKey &key, FacePoint2 p, double eps) {
  if (key.a == InvalidId) return -1;
  const double epsSq = eps * eps;
  for (int i = 0; i < count; ++i) {
    if (faceCanonicalKeyEqual(keys[i], key) && faceDot(faceSub(points[i], p), faceSub(points[i], p)) <= 16.0 * epsSq) return i;
  }
  return -1;
}

__device__ inline int faceBuildPoints(
    const FacePolygon &boundary,
    const Segment *segments, const int *segmentSource, int boundaryCount,
    int count, FaceOverlayWorkspace &work, double eps) {
  const int canonicalCount = int(boundary.count);
  if (canonicalCount > int(work.pointCapacity)) return -1;
  int pointCount = 0;
  for (int i = 0; i < canonicalCount; ++i) {
    work.points[pointCount] = boundary.vertices[i];
    work.pointEdge[pointCount] = -1;
    work.pointKey[pointCount++] = FaceCanonicalKey{FaceCanonicalBaseVertex, boundary.sourceVertexId[i], InvalidId, InvalidId};
  }
  for (int i = 0; i < count; ++i) {
    const int source = segmentSource[i];
    for (int k = 0; k < 2; ++k) {
      const FacePoint2 p = k == 0 ? segments[i].a : segments[i].b;
      const FaceCanonicalKey key = k == 0 ? segments[i].keyA : segments[i].keyB;
      int id = faceFindCanonicalPoint(work.pointKey, work.points, pointCount, key, p, eps);
      if (id < 0) {
        id = source < boundaryCount
                 ? faceFindSplitPoint(work.points, pointCount, canonicalCount, work.pointEdge, source, p, eps)
                 : faceFindSplitPoint(work.points, pointCount, pointCount, nullptr, -1, p, eps);
      }
      if (id < 0) {
        if (pointCount >= int(work.pointCapacity)) return -1;
        work.points[pointCount] = p;
        work.pointEdge[pointCount] = source < boundaryCount ? source : -1;
        work.pointKey[pointCount] = key;
        id = pointCount++;
      } else if (key.a != InvalidId && work.pointKey[id].a == InvalidId) {
        work.pointKey[id] = key;
      }
      if (k == 0) work.splitPointA[i] = id;
      else work.splitPointB[i] = id;
    }
  }
  return pointCount;
}

struct FaceDart { int edge, direction; };

__device__ inline FacePoint2 faceDartDirection(const FaceOverlayWorkspace &work, FaceDart dart) {
  const Edge &edge = work.edges[dart.edge];
  return dart.direction == 0 ? faceSub(work.points[edge.b], work.points[edge.a])
                             : faceSub(work.points[edge.a], work.points[edge.b]);
}

__device__ inline int faceDartHalf(FacePoint2 direction) {
  return (direction.y > 0.0 || (direction.y == 0.0 && direction.x > 0.0)) ? 0 : 1;
}

__device__ inline bool faceDartAngleLess(const FaceOverlayWorkspace &work, FaceDart a, FaceDart b) {
  const FacePoint2 da = faceDartDirection(work, a), db = faceDartDirection(work, b);
  const int halfA = faceDartHalf(da), halfB = faceDartHalf(db);
  if (halfA != halfB) return halfA < halfB;
  const auto orient = orient2dFilteredDouble(
      make_double2(0.0, 0.0), make_double2(da.x, da.y), make_double2(db.x, db.y));
  if (orient.sign != 0) return orient.sign > 0;
  if (a.edge != b.edge) return a.edge < b.edge;
  return a.direction < b.direction;
}

__device__ inline FaceDart faceNextDart(const FaceOverlayWorkspace &work, int edge, int direction) {
  const int vertex = direction == 0 ? work.edges[edge].b : work.edges[edge].a;
  const FaceDart reverse{edge, 1 - direction};
  FaceDart bestLess{-1, 0}, bestOverall{-1, 0};
  bool haveLess = false, haveOverall = false;

  for (uint32_t candidateEdge = 0; candidateEdge < work.edgeCount; ++candidateEdge) {
    for (int candidateDirection = 0; candidateDirection < 2; ++candidateDirection) {
      const int from = candidateDirection == 0 ? work.edges[candidateEdge].a : work.edges[candidateEdge].b;
      if (from != vertex) continue;
      const FaceDart candidate{int(candidateEdge), candidateDirection};
      if (candidate.edge == reverse.edge && candidate.direction == reverse.direction) continue;

      if (!haveOverall || faceDartAngleLess(work, bestOverall, candidate)) {
        bestOverall = candidate; haveOverall = true;
      }
      if (faceDartAngleLess(work, candidate, reverse) &&
          (!haveLess || faceDartAngleLess(work, bestLess, candidate))) {
        bestLess = candidate; haveLess = true;
      }
    }
  }
  if (haveLess) return bestLess;
  if (haveOverall) return bestOverall;
  return reverse;
}

__device__ inline uint32_t faceClassifyRegion(
    FacePoint2 sample, double3 sample3, double3 normal,
    const cudaGeometry::DeviceModelGeometryView *views, uint32_t modelCount,
    const FaceOverlayWorkspace &work, uint32_t &materialA, uint32_t &materialB) {
  materialA = InvalidId; materialB = InvalidId;
  for (uint32_t model = 0; model < modelCount; ++model) {
    const cudaGeometry::DeviceModelGeometryView &view = views[model];
    const uint32_t runtimeModelId = view.runtimeModelId;
    bool hasCoplanar = false;
    double3 coplanarNormal = make_double3(0.0, 0.0, 0.0);
    int coplanarWinding = 0;
    for (uint32_t s = 0; s < work.segmentCount; ++s) {
      const Segment &segment = work.segments[s];
      if (segment.runtimeModelId != runtimeModelId ||
          segment.segmentType != uint32_t(FaceCutSegmentType::Coplanar)) {
        continue;
      }
      hasCoplanar = true;
      coplanarNormal = work.segmentPlaneNormals[s];
      const auto orient = orient2dFilteredDouble(
          make_double2(segment.a.x, segment.a.y),
          make_double2(segment.b.x, segment.b.y),
          make_double2(sample.x, sample.y));
      if (segment.a.y <= sample.y) {
        if (segment.b.y > sample.y && orient.sign > 0) ++coplanarWinding;
      } else {
        if (segment.b.y <= sample.y && orient.sign < 0) --coplanarWinding;
      }
    }
    if (hasCoplanar && coplanarWinding != 0) {
      const double side = dot3(normal, coplanarNormal);
      if (side > 0.0) {
        if (materialB != InvalidId) return OverlappingModels;
        materialB = runtimeModelId;
      } else if (side < 0.0) {
        if (materialA != InvalidId) return OverlappingModels;
        materialA = runtimeModelId;
      } else {
        return AmbiguousOwnership;
      }
      continue;
    }
    const float3 sampleF = make_float3(float(sample3.x), float(sample3.y), float(sample3.z));
    if (!aabbContainsOrIntersects(sampleF, view.boundsMin, view.boundsMax)) continue;

    const double winding = cudaGeometry::computeWinding(
        view.triangles, view.nodes, view.parents, view.root, view.moments, sample3, sampleF);
    if (!isfinite(winding) || fabs(fabs(winding) - 0.5) < 0.01) return AmbiguousOwnership;
    if (fabs(winding) > 0.5) {
      if (materialA != InvalidId || materialB != InvalidId) return OverlappingModels;
      materialA = materialB = runtimeModelId;
      break;
    }
  }
  return FacePartitionSuccess;
}

__device__ inline bool faceFindRegionSample(
    const FaceOverlayWorkspace &work, uint32_t i, double eps, FacePoint2 &sample) {
  const FacePoint2 *cyclePoints = work.cyclePoints + work.cycles[i].firstPoint;
  sample = faceMul(work.cycles[i].moment, 1.0 / work.cycles[i].area);
  auto insideParentAndNoChild = [&](FacePoint2 pt) {
    if (!facePointInPolygon(pt, cyclePoints, work.cycles[i].pointCount)) return false;
    for (uint32_t j = 0; j < work.cycleCount; ++j) {
      if (work.cycles[j].parent == int(i) &&
          facePointInPolygon(pt, work.cyclePoints + work.cycles[j].firstPoint, work.cycles[j].pointCount)) {
        return false;
      }
    }
    return true;
  };
  if (insideParentAndNoChild(sample)) return true;

  for (uint32_t p = 0; p < work.cycles[i].pointCount; ++p) {
    const FacePoint2 a = cyclePoints[p], b = cyclePoints[(p + 1) % work.cycles[i].pointCount];
    const FacePoint2 e = faceSub(b, a);
    const double len = hypot(e.x, e.y);
    if (len <= eps) continue;
    const FacePoint2 inward{-e.y / len, e.x / len};
    const double estWidth = fmax(fabs(work.cycles[i].area) / len, 1e-14);
    const double probeDist = fmin(0.1 * estWidth, fmin(0.25 * len, fmax(1e-15, len * 1e-4)));
    const FacePoint2 probe = faceAdd(faceMul(faceAdd(a, b), 0.5), faceMul(inward, probeDist));
    if (insideParentAndNoChild(probe)) {
      sample = probe;
      return true;
    }
  }
  return false;
}

__device__ inline uint32_t faceCollectRestrictedSegments(
    const BaseFace &face, const FacePolygon &boundary, double3 origin,
    double3 u, double3 v, double scale,
    const FaceBoundarySegment *const *modelSegments,
    const uint32_t *const *modelSegmentOffsets, uint32_t modelCount,
    uint32_t &boundaryCount, FaceOverlayWorkspace &work) {
  work.segmentCount = boundaryCount = 0;
  for (uint32_t i = 0; i < boundary.count; ++i) {
    if (work.segmentCount == work.segmentCapacity) return TooManySegments;
    work.segmentPlaneNormals[work.segmentCount] = make_double3(0, 0, 0);
    FaceCanonicalKey kA{FaceCanonicalBaseVertex, boundary.sourceVertexId[i], InvalidId, InvalidId};
    FaceCanonicalKey kB{FaceCanonicalBaseVertex, boundary.sourceVertexId[(i + 1) % boundary.count], InvalidId, InvalidId};
    work.segments[work.segmentCount++] = {
        boundary.vertices[i], boundary.vertices[(i + 1) % boundary.count],
        kA, kB, BaseVertexCanonicalBit | i, InvalidId, uint32_t(FaceCutSegmentType::Crossing)};
  }
  boundaryCount = work.segmentCount;
  if (!modelSegments || !modelSegmentOffsets) return FacePartitionSuccess;

  const uint32_t slots[2] = {face.slotA, face.slotB};
  for (uint32_t s = 0; s < 2; ++s) {
    const uint32_t slot = slots[s];
    if (slot == InvalidId) continue;
    for (uint32_t g = 0; g < modelCount; ++g) {
      if (!modelSegments[g] || !modelSegmentOffsets[g]) continue;
      const uint32_t begin = modelSegmentOffsets[g][slot], end = modelSegmentOffsets[g][slot + 1];
      for (uint32_t i = begin; i < end; ++i) {
        const FaceBoundarySegment &seg = modelSegments[g][i];
        FaceCanonicalKey kA{FaceCanonicalCadVertex, seg.canonicalA, InvalidId, InvalidId};
        FaceCanonicalKey kB{FaceCanonicalCadVertex, seg.canonicalB, InvalidId, InvalidId};
        const uint32_t err = faceAddSegment(
            work, faceProject(seg.a, origin, u, v), faceProject(seg.b, origin, u, v),
            kA, kB, seg.segmentId, boundary, scale, seg.planeNormal,
            seg.runtimeModelId, seg.segmentType);
        if (err) return err;
      }
    }
  }
  return FacePartitionSuccess;
}

__device__ inline void faceDiscardCycle(
    FaceOverlayWorkspace &work, int cycleIndex, uint32_t firstPoint) {
  work.cyclePointCount = firstPoint;
  for (uint32_t i = 0; i < work.edgeCount * 2; ++i) {
    if (work.cycleOfDirectedEdge[i] == cycleIndex) work.cycleOfDirectedEdge[i] = -1;
  }
}

__device__ inline uint32_t faceBuildOverlay(
    uint32_t faceId, const BaseFaceMeasure &base, const FacePolygon &boundary,
    double3 origin, double3 u, double3 v,
    double3 normal, double scale, const cudaGeometry::DeviceModelGeometryView *models,
    uint32_t modelCount, uint32_t boundaryCount, FaceOverlayWorkspace &work) {
  const double eps = 1024.0 * DBL_EPSILON * fmax(scale, 1.0);
  Segment *split = work.splitSegments;
  int *splitSource = work.splitSources;
  const int splitCount = faceSplitSegments(work, split, splitSource, eps);
  if (splitCount < 0) return TooManySegments;

  const int pointCount = faceBuildPoints(boundary, split, splitSource, int(boundaryCount), splitCount, work, eps);
  if (pointCount < 0) return TooManyRegions;

  work.edgeCount = 0;
  for (int i = 0; i < splitCount; ++i) {
    const int source = splitSource[i];
    int a = work.splitPointA[i], b = work.splitPointB[i];
    if (a < 0 || b < 0 || a == b) continue;
    if (a > b) { const int t = a; a = b; b = t; }
    bool duplicate = false;
    for (uint32_t e = 0; e < work.edgeCount; ++e) {
      if (work.edges[e].a == a && work.edges[e].b == b) { duplicate = true; break; }
    }
    if (duplicate) continue;
    if (work.edgeCount >= work.edgeCapacity) return TooManyRegions;
    work.edgeModelId[work.edgeCount] = work.segments[source].runtimeModelId;
    work.edgeSegmentType[work.edgeCount] = work.segments[source].segmentType;
    work.edges[work.edgeCount++] = {a, b};
  }

  for (uint32_t i = 0; i < work.edgeCount * 2; ++i) {
    work.visited[i] = 0;
    work.cycleOfDirectedEdge[i] = -1;
  }
  work.cycleCount = 0;
  for (uint32_t e = 0; e < work.edgeCount; ++e) {
    for (int dir = 0; dir < 2; ++dir) {
      const int startFrom = dir == 0 ? work.edges[e].a : work.edges[e].b;
      const int startTo = dir == 0 ? work.edges[e].b : work.edges[e].a;
      if (work.visited[e * 2 + dir]) continue;
      int from = startFrom, to = startTo, edge = int(e), direction = dir;
      const int cycleIndex = int(work.cycleCount);
      Cycle cycle{};
      cycle.firstPoint = work.cyclePointCount;
      bool closed = false;
      for (int guard = 0; guard < int(work.edgeCount) * 4 + 16; ++guard) {
        if (work.visited[edge * 2 + direction]) {
          closed = (from == startFrom && to == startTo);
          break;
        }
        work.visited[edge * 2 + direction] = 1;
        work.cycleOfDirectedEdge[edge * 2 + direction] = cycleIndex;
        if (work.cyclePointCount >= work.cyclePointCapacity) return TooManyRegions;
        work.cyclePoints[work.cyclePointCount++] = work.points[from];
        ++cycle.pointCount;
        const FaceDart next = faceNextDart(work, edge, direction);
        if (next.edge < 0) break;
        from = to;
        edge = next.edge;
        direction = next.direction;
        to = direction == 0 ? work.edges[edge].b : work.edges[edge].a;
      }
      if (!closed) return UnresolvedRegion;
      if (cycle.pointCount < 3) {
        faceDiscardCycle(work, cycleIndex, cycle.firstPoint);
        continue;
      }
      faceComputeCycleAreaAndMoment(
          work.cyclePoints + cycle.firstPoint, cycle.pointCount,
          cycle.area, cycle.areaRoundoff, cycle.moment);
      if (!isfinite(cycle.area)) return UnresolvedRegion;
      if (fabs(cycle.area) <= cycle.areaRoundoff) {
        faceDiscardCycle(work, cycleIndex, cycle.firstPoint);
        continue;
      }
      if (work.cycleCount >= work.cycleCapacity) return TooManyRegions;
      work.cycles[work.cycleCount++] = cycle;
    }
  }

  for (uint32_t i = 0; i < work.cycleCount; ++i) work.cycles[i].parent = -1;
  for (uint32_t i = 0; i < work.cycleCount; ++i) {
    if (work.cycles[i].area >= 0.0) continue;
    FacePoint2 probe{};
    bool haveProbe = false;
    const FacePoint2 *cyclePoints = work.cyclePoints + work.cycles[i].firstPoint;
    for (uint32_t p = 0; p < work.cycles[i].pointCount; ++p) {
      const FacePoint2 a = cyclePoints[p], b = cyclePoints[(p + 1) % work.cycles[i].pointCount];
      const FacePoint2 e = faceSub(b, a);
      const double len = hypot(e.x, e.y);
      if (!(len > eps)) continue;
      const FacePoint2 outward{-e.y / len, e.x / len};
      const double probeDist = fmin(0.25 * len, fmax(1e-15, len * 1e-4));
      probe = faceAdd(faceMul(faceAdd(a, b), 0.5), faceMul(outward, probeDist));
      haveProbe = true;
      break;
    }
    if (!haveProbe) return UnresolvedRegion;
    if (!facePointInPolygon(probe, boundary.vertices, int(boundary.count))) {
      work.cycles[i].parent = -2;
      continue;
    }
    for (uint32_t j = 0; j < work.cycleCount; ++j) {
      if (work.cycles[j].area <= 0.0) continue;
      if (facePointInPolygon(probe, work.cyclePoints + work.cycles[j].firstPoint,
                             work.cycles[j].pointCount)) {
        if (work.cycles[i].parent < 0 ||
            work.cycles[j].area < work.cycles[work.cycles[i].parent].area) {
          work.cycles[i].parent = int(j);
        }
      }
    }
  }

  for (uint32_t i = 0; i < work.cycleCount; ++i) {
    if (work.cycles[i].area < 0.0 && work.cycles[i].parent == -1) return UnresolvedRegion;
  }

  double *netArea = work.netArea, *netRoundoff = work.netRoundoff;
  FacePoint2 *netMoment = work.netMoment;
  int *regionOfCycle = work.regionOfCycle;
  for (uint32_t i = 0; i < work.cycleCount; ++i) {
    netArea[i] = work.cycles[i].area; netRoundoff[i] = work.cycles[i].areaRoundoff; netMoment[i] = work.cycles[i].moment;
    if (work.cycles[i].area > 0.0) {
      for (uint32_t j = 0; j < work.cycleCount; ++j) {
        if (work.cycles[j].area < 0.0 && work.cycles[j].parent == int(i)) {
          netArea[i] += work.cycles[j].area; netRoundoff[i] += work.cycles[j].areaRoundoff;
          netMoment[i] = faceAdd(netMoment[i], work.cycles[j].moment);
        }
      }
    }
    work.regionMaterial[i] = UnresolvedMaterial; work.regionResolved[i] = false; work.regionCoplanar[i] = false;
    regionOfCycle[i] = work.cycles[i].area >= 0.0 ? int(i) : work.cycles[i].parent;
  }

  for (uint32_t e = 0; e < work.edgeCount; ++e) {
    if (work.edgeSegmentType[e] != uint32_t(FaceCutSegmentType::Coplanar)) continue;
    const int c0 = work.cycleOfDirectedEdge[e * 2], c1 = work.cycleOfDirectedEdge[e * 2 + 1];
    if (c0 >= 0 && regionOfCycle[c0] >= 0) work.regionCoplanar[regionOfCycle[c0]] = true;
    if (c1 >= 0 && regionOfCycle[c1] >= 0) work.regionCoplanar[regionOfCycle[c1]] = true;
  }

  int seed = -1;
  for (uint32_t i = 0; i < work.cycleCount; ++i) {
    if (work.cycles[i].area >= 0.0 && !work.regionCoplanar[i]) { seed = int(i); break; }
  }
  if (seed >= 0) {
    FacePoint2 sample{};
    if (faceFindRegionSample(work, uint32_t(seed), eps, sample)) {
      const double3 sample3 = add3(origin, add3(mul3(u, sample.x), mul3(v, sample.y)));
      uint32_t materialA = InvalidId, materialB = InvalidId;
      const uint32_t classify = faceClassifyRegion(
          sample, sample3, normal, models, modelCount, work, materialA, materialB);
      if (classify) return classify;
      if (materialA == materialB) {
        work.regionMaterial[seed] = materialA;
        work.regionResolved[seed] = true;
      }
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (uint32_t e = 0; e < work.edgeCount; ++e) {
      if (work.edgeSegmentType[e] == uint32_t(FaceCutSegmentType::Coplanar)) continue;
      const int c0 = work.cycleOfDirectedEdge[e * 2], c1 = work.cycleOfDirectedEdge[e * 2 + 1];
      if (c0 < 0 || c1 < 0) continue;
      const int r0 = regionOfCycle[c0], r1 = regionOfCycle[c1];
      if (r0 < 0 || r1 < 0 || r0 == r1) continue;
      const uint32_t edgeModel = work.edgeModelId[e];
      for (int pass = 0; pass < 2; ++pass) {
        const int fromR = pass == 0 ? r0 : r1, toR = pass == 0 ? r1 : r0;
        if (work.regionResolved[fromR] && !work.regionResolved[toR]) {
          uint32_t next = work.regionMaterial[fromR];
          if (edgeModel != InvalidId) next = (next == edgeModel) ? InvalidId : (next == InvalidId ? edgeModel : next);
          work.regionMaterial[toR] = next;
          work.regionResolved[toR] = true;
          changed = true;
        }
      }
    }
  }

  for (uint32_t e = 0; e < work.edgeCount; ++e) {
    if (work.edgeSegmentType[e] == uint32_t(FaceCutSegmentType::Coplanar)) continue;
    const int c0 = work.cycleOfDirectedEdge[e * 2], c1 = work.cycleOfDirectedEdge[e * 2 + 1];
    if (c0 < 0 || c1 < 0) continue;
    const int r0 = regionOfCycle[c0], r1 = regionOfCycle[c1];
    if (r0 < 0 || r1 < 0 || r0 == r1) continue;
    if (!work.regionResolved[r0] || !work.regionResolved[r1]) continue;
    const uint32_t m0 = work.regionMaterial[r0], m1 = work.regionMaterial[r1];
    const uint32_t edgeModel = work.edgeModelId[e];
    if (edgeModel == InvalidId) {
      if (m0 != m1) return MaterialInconsistency;
      continue;
    }
    if ((m0 != InvalidId && m0 != edgeModel) || (m1 != InvalidId && m1 != edgeModel)) continue;
    const uint32_t expected = (m0 == edgeModel) ? InvalidId : ((m0 == InvalidId) ? edgeModel : m0);
    if (m1 != expected) return MaterialInconsistency;
  }

  work.groupCount = 0;
  for (uint32_t i = 0; i < work.cycleCount; ++i) {
    if (work.cycles[i].area < 0.0) continue;
    if (!isfinite(netArea[i]) || netArea[i] <= netRoundoff[i]) return UnresolvedRegion;
    uint32_t materialA = InvalidId, materialB = InvalidId;
    if (work.regionResolved[i] && !work.regionCoplanar[i]) {
      materialA = materialB = work.regionMaterial[i];
    } else {
      FacePoint2 sample{};
      if (!faceFindRegionSample(work, i, eps, sample)) return UnresolvedRegion;
      const double3 sample3 = add3(origin, add3(mul3(u, sample.x), mul3(v, sample.y)));
      const uint32_t classify = faceClassifyRegion(
          sample, sample3, normal, models, modelCount, work, materialA, materialB);
      if (classify) return classify;
    }
    const double3 worldMoment = add3(mul3(origin, netArea[i]),
                                     add3(mul3(u, netMoment[i].x), mul3(v, netMoment[i].y)));
    const uint32_t emit = faceEmitRegion(
        work.groups, work.groupCount, work.groupCapacity, netArea[i], worldMoment, normal,
        materialA, materialB,
        fmax(base.areaError, fmax(256.0 * DBL_EPSILON * netArea[i], netRoundoff[i])));
    if (emit) return emit;
  }
  return FacePartitionSuccess;
}

const char *facePartitionErrorName(uint32_t code) {
  static const char *const names[] = {
      "FacePartitionSuccess", "InvalidFacePolygon", "TooManyRegions", "AmbiguousOwnership",
      "OverlappingModels", "AreaConservation", "MomentConservation", "PassMismatch",
      "UnresolvedRegion", "TooManySegments", "EmptyFacePartition", "MaterialInconsistency"
  };
  return code <= MaterialInconsistency ? names[code] : "Unknown";
}

struct FacePartitionInput {
  const BaseFace *baseFaces = nullptr;
  const uint32_t *faceVertexOffsets = nullptr, *faceVertexIds = nullptr;
  const double3 *faceVertices = nullptr;
  const BaseFaceMeasure *baseMeasures = nullptr;
  const cudaGeometry::DeviceModelGeometryView *models = nullptr;
  const FaceBoundarySegment *const *modelSegments = nullptr;
  const uint32_t *const *modelSegmentOffsets = nullptr;
  const double *const *modelRestrictedPlaneAreas = nullptr;
  double facetFloor = 0.0;
  uint32_t modelCount = 0;
};

__device__ inline uint32_t classifyUncutFaceSlot(
    uint32_t slot, double baseArea, double tolerance,
    const FacePartitionInput &input, uint32_t &outMaterial) {
  outMaterial = InvalidId;
  if (slot == InvalidId) return FacePartitionSuccess;

  for (uint32_t m = 0; m < input.modelCount; ++m) {
    const double *restrictedAreas =
        input.modelRestrictedPlaneAreas != nullptr ? input.modelRestrictedPlaneAreas[m] : nullptr;
    if (restrictedAreas == nullptr) return PassMismatch;

    const double area = restrictedAreas[slot];
    if (area <= tolerance) {
      continue;
    }
    if (fabs(area - baseArea) <= tolerance) {
      if (outMaterial != InvalidId) return OverlappingModels;
      outMaterial = input.models[m].runtimeModelId;
      continue;
    }
    return AreaConservation;
  }
  return FacePartitionSuccess;
}

__device__ inline uint32_t classifyAndEmitUncutFace(
    uint32_t faceId, const FacePartitionInput &input, FaceRegion &outRegion) {
  const BaseFaceMeasure base = input.baseMeasures[faceId];
  if (!base.valid || !(base.area > 0.0) || !isfinite(base.area)) return InvalidFacePolygon;

  const BaseFace &baseFace = input.baseFaces[faceId];
  const double tolerance = fmax(input.facetFloor, fabs(base.area) * 1e-6);

  uint32_t materialA = InvalidId, materialB = InvalidId;
  uint32_t err = classifyUncutFaceSlot(baseFace.slotA, base.area, tolerance, input, materialA);
  if (err) return err;

  if (baseFace.slotB != InvalidId) {
    err = classifyUncutFaceSlot(baseFace.slotB, base.area, tolerance, input, materialB);
    if (err) return err;
  }

  outRegion = FaceRegion{};
  outRegion.measure = base;
  outRegion.measure.normal = normalize3(base.normal);
  outRegion.measure.valid = 1u;
  outRegion.materialA = materialA;
  outRegion.materialB = materialB;
  return FacePartitionSuccess;
}

__device__ inline uint32_t partitionFaceArrangement(
    uint32_t faceId, const FacePartitionInput &input, FaceOverlayWorkspace &work) {
  work.segmentCount = work.edgeCount = work.cycleCount =
      work.cyclePointCount = work.groupCount = 0;

  const BaseFace &baseFace = input.baseFaces[faceId];
  const BaseFaceMeasure base = input.baseMeasures[faceId];

  FacePolygon boundary{};
  double3 origin, u, v, normal;
  double scale;
  if (!faceSetupPolygon(faceId, input.faceVertexOffsets, input.faceVertices,
                        input.faceVertexIds, base, boundary, origin, u, v,
                        normal, scale)) {
    return InvalidFacePolygon;
  }

  uint32_t boundaryCount = 0;
  const uint32_t segmentError = faceCollectRestrictedSegments(
      baseFace, boundary, origin, u, v, scale, input.modelSegments,
      input.modelSegmentOffsets, input.modelCount, boundaryCount, work);
  if (segmentError) return segmentError;

  if (work.segmentCount == boundaryCount) {
    if (work.groupCapacity == 0) return TooManyRegions;
    const uint32_t err = classifyAndEmitUncutFace(faceId, input, work.groups[0]);
    if (err) return err;
    work.groupCount = 1;
    return FacePartitionSuccess;
  }

  const uint32_t overlayError = faceBuildOverlay(
      faceId, base, boundary, origin, u, v, normal, scale,
      input.models, input.modelCount, boundaryCount, work);
  if (overlayError) return overlayError;

  return faceValidateConservation(work.groups, work.groupCount, boundary, origin, u, v, scale);
}

__global__ void countFaceInputSegmentsKernel(
    const BaseFace *baseFaces, const uint32_t *faceVertexOffsets,
    const uint32_t *const *modelSegmentOffsets, uint32_t modelCount,
    uint32_t faceCount, uint32_t *outCounts, uint32_t *outCadCounts) {
  const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
  if (faceId >= faceCount) return;
  const uint32_t baseCount = faceVertexOffsets[faceId + 1] - faceVertexOffsets[faceId];
  uint32_t cadCount = 0;
  const BaseFace &face = baseFaces[faceId];
  const uint32_t slots[2] = {face.slotA, face.slotB};
  for (uint32_t s = 0; s < 2; ++s) {
    const uint32_t slot = slots[s];
    if (slot == InvalidId) continue;
    for (uint32_t g = 0; g < modelCount; ++g) {
      if (modelSegmentOffsets[g] == nullptr) continue;
      cadCount += modelSegmentOffsets[g][slot + 1] - modelSegmentOffsets[g][slot];
    }
  }
  outCounts[faceId] = baseCount + cadCount;
  outCadCounts[faceId] = cadCount;
}

__global__ void setAllRegionCountsToOneKernel(uint32_t faceCount, uint32_t *regionCounts) {
  const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
  if (faceId < faceCount) regionCounts[faceId] = 1;
}

__global__ void emitUncutFacesKernel(
    uint32_t faceCount, const uint32_t *cadSegmentCounts,
    FacePartitionInput input, const uint32_t *regionOffsets,
    FaceRegion *regions, RvdPackedError *packedError) {
  const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
  if (faceId >= faceCount || cadSegmentCounts[faceId] > 0) return;

  FaceRegion region{};
  const uint32_t err = classifyAndEmitUncutFace(faceId, input, region);
  if (err) { rvdReportError(packedError, err, faceId); return; }
  regions[regionOffsets[faceId]] = region;
}

__global__ void buildFaceArrangementKernel(
    uint32_t batchStart, uint32_t batchEnd, const uint32_t *cutFaceIndices,
    FacePartitionInput input, uint32_t *outRegionCounts,
    FaceRegion *outCutRegions, uint32_t maxRegionsPerFace,
    FaceOverlayWorkspace *workspaces, RvdPackedError *packedError) {
  const uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t idx = batchStart + local;
  if (idx >= batchEnd) return;

  const uint32_t faceId = cutFaceIndices[idx];
  auto &workspace = workspaces[local];
  const uint32_t result = partitionFaceArrangement(faceId, input, workspace);
  if (result) { rvdReportError(packedError, result, faceId); return; }
  if (workspace.groupCount == 0 || workspace.groupCount > maxRegionsPerFace) {
    rvdReportError(packedError, workspace.groupCount == 0 ? EmptyFacePartition : TooManyRegions, faceId);
    return;
  }
  outRegionCounts[faceId] = workspace.groupCount;
  FaceRegion *dst = outCutRegions + idx * maxRegionsPerFace;
  for (uint32_t i = 0; i < workspace.groupCount; ++i) dst[i] = workspace.groups[i];
}

__global__ void scatterCutFaceRegionsKernel(
    uint32_t cutFaceCount, const uint32_t *cutFaceIndices,
    const uint32_t *regionOffsets, const FaceRegion *cutFaceEmittedRegions,
    uint32_t maxRegionsPerFace, FaceRegion *regions) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= cutFaceCount) return;

  const uint32_t faceId = cutFaceIndices[idx];
  const uint32_t dst = regionOffsets[faceId];
  const uint32_t count = regionOffsets[faceId + 1] - dst;
  const FaceRegion *src = cutFaceEmittedRegions + idx * maxRegionsPerFace;
  for (uint32_t i = 0; i < count; ++i) regions[dst + i] = src[i];
}

struct FaceArrangementCapacities {
  uint32_t inputSegments = 0, splitParams = 0, splitSegments = 0, points = 0;
  uint32_t edges = 0, cycles = 0, cyclePoints = 0, groups = 0;
};

inline FaceArrangementCapacities makeFaceArrangementCapacities(uint32_t inputSegments) {
  FaceArrangementCapacities c{};
  c.inputSegments = inputSegments;
  const uint64_t sp = 2ull + 2ull * (inputSegments > 0 ? uint64_t(inputSegments) - 1ull : 0ull);
  const uint64_t ss = std::min(uint64_t(inputSegments) * sp, 512ull);
  c.splitParams = uint32_t(sp);
  c.splitSegments = uint32_t(ss);
  c.points = uint32_t(2ull * ss + uint64_t(inputSegments) + 8ull);
  c.edges = uint32_t(ss);
  c.cycles = 64u;
  c.cyclePoints = uint32_t(std::max(2ull * ss + 8ull, 512ull));
  c.groups = MaxRegionsPerCutFace;
  return c;
}

inline size_t bindFaceOverlayWorkspace(
    const FaceArrangementCapacities &c, FaceOverlayWorkspace &work, uint8_t *base) {
  size_t off = 0;
  auto alloc = [&](auto &ptr, size_t count) {
    using Ptr = std::decay_t<decltype(ptr)>;
    off = (off + 127) & ~size_t(127);
    if (base) ptr = reinterpret_cast<Ptr>(base + off);
    off += count * sizeof(*ptr);
  };
  alloc(work.segments, c.inputSegments); alloc(work.segmentPlaneNormals, c.inputSegments);
  alloc(work.points, c.points); alloc(work.pointEdge, c.points); alloc(work.pointKey, c.points);
  alloc(work.edges, c.edges); alloc(work.edgeModelId, c.edges); alloc(work.edgeSegmentType, c.edges);
  alloc(work.cycleOfDirectedEdge, c.edges * 2); alloc(work.cycles, c.cycles); alloc(work.cyclePoints, c.cyclePoints);
  alloc(work.regionMaterial, c.cycles); alloc(work.regionResolved, c.cycles); alloc(work.regionCoplanar, c.cycles);
  alloc(work.netArea, c.cycles); alloc(work.netRoundoff, c.cycles); alloc(work.netMoment, c.cycles);
  alloc(work.regionOfCycle, c.cycles); alloc(work.visited, c.edges * 2); alloc(work.groups, c.groups);
  alloc(work.splitSegments, c.splitSegments); alloc(work.splitSources, c.splitSegments);
  alloc(work.splitPointA, c.splitSegments); alloc(work.splitPointB, c.splitSegments);
  alloc(work.splitParams, c.splitParams);
  work.segmentCapacity = c.inputSegments; work.splitSegmentCapacity = c.splitSegments;
  work.splitCapacity = c.splitParams; work.pointCapacity = c.points; work.edgeCapacity = c.edges;
  work.cycleCapacity = c.cycles; work.cyclePointCapacity = c.cyclePoints; work.groupCapacity = c.groups;
  return (off + 127) & ~size_t(127);
}

struct FacePartitionBuffers {
  cudaUtils::CudaBuffer<uint32_t> regionCounts;
  cudaUtils::CudaBuffer<uint32_t> cadSegmentCounts;
  cudaUtils::CudaBuffer<uint32_t> cutFaceIndices;
  cudaUtils::CudaBuffer<FaceRegion> cutFaceEmittedRegions;
  cudaUtils::CudaBuffer<RvdPackedError> packedError;
  cudaUtils::CudaBuffer<FaceOverlayWorkspace> workspaces;
  cudaUtils::CudaBuffer<uint8_t> arena;
  std::vector<FaceOverlayWorkspace> hostWorkspaces;
};

bool finishPartitionPass(const cudaUtils::CudaBuffer<RvdPackedError> &packedError, cudaStream_t stream) {
  RvdPackedError hostError = 0;
  if (!rvdHost::readPackedError(packedError, stream, hostError)) return false;
  if (hostError != 0) {
    std::cerr << "[RVD-FacePartition] " << facePartitionErrorName(rvdErrorCode(hostError))
              << " face=" << rvdErrorId(hostError) << std::endl;
    return false;
  }
  return true;
}

bool runArrangementPass(
    uint32_t cutFaceCount, const FacePartitionInput &input,
    const std::vector<uint32_t> &hostCutFaceIndices,
    const std::vector<uint32_t> &inputSegmentCounts, FacePartitionBuffers &buffers,
    cudaStream_t stream) {
  if (cutFaceCount == 0) return true;

  const uint32_t batchSize = std::min(cutFaceCount, OverlayBatchSize);
  for (uint32_t start = 0; start < cutFaceCount; start += batchSize) {
    const uint32_t count = std::min(batchSize, cutFaceCount - start);
    uint32_t maxInputSegments = 0;
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t fid = hostCutFaceIndices[start + i];
      maxInputSegments = std::max(maxInputSegments, inputSegmentCounts[fid]);
    }
    const FaceArrangementCapacities capacities = makeFaceArrangementCapacities(maxInputSegments);
    FaceOverlayWorkspace dummy{};
    const size_t bytesPerWorkspace = bindFaceOverlayWorkspace(capacities, dummy, nullptr);
    if (!buffers.arena.ensureCapacity(count * bytesPerWorkspace)) return false;

    uint8_t *arenaPtr = buffers.arena.get();
    for (uint32_t i = 0; i < count; ++i) {
      bindFaceOverlayWorkspace(capacities, buffers.hostWorkspaces[i], arenaPtr + i * bytesPerWorkspace);
    }
    if (!buffers.workspaces.uploadAsync(buffers.hostWorkspaces.data(), count, stream)) return false;

    buildFaceArrangementKernel<<<
        rvdHost::blockCount(count, OverlayBlockSize), OverlayBlockSize, 0, stream>>>(
        start, start + count, buffers.cutFaceIndices.get(), input,
        buffers.regionCounts.get(), buffers.cutFaceEmittedRegions.get(),
        MaxRegionsPerCutFace, buffers.workspaces.get(), buffers.packedError.get());

    if (!rvdHost::checkLaunch("launch buildFaceArrangementKernel") ||
        !rvdHost::synchronize(stream, "synchronize arrangement batch")) return false;
  }

  return finishPartitionPass(buffers.packedError, stream);
}

struct ModelFacePointers {
  cudaUtils::CudaBuffer<cudaGeometry::DeviceModelGeometryView> views;
  cudaUtils::CudaBuffer<const FaceBoundarySegment *> segments;
  cudaUtils::CudaBuffer<const uint32_t *> segmentOffsets;
  cudaUtils::CudaBuffer<const double *> restrictedPlaneAreas;
};

bool uploadModelFacePointers(
    const std::vector<cudaGeometry::DeviceModelGeometryView> &modelViews,
    const std::vector<ModelRestriction> &models, cudaStream_t stream,
    ModelFacePointers &pointers) {
  if (!pointers.views.allocate(modelViews.size()) ||
      !pointers.views.uploadAsync(modelViews.data(), modelViews.size(), stream)) return false;
  if (models.empty()) return true;

  const uint32_t count = uint32_t(models.size());
  std::vector<const FaceBoundarySegment *> segs(count);
  std::vector<const uint32_t *> offsets(count);
  std::vector<const double *> areas(count);
  for (uint32_t i = 0; i < count; ++i) {
    segs[i] = models[i].devFaceSegments.get();
    offsets[i] = models[i].devFaceSegmentOffsets.get();
    areas[i] = models[i].devRestrictedPlaneAreas.get();
  }
  return pointers.segments.allocate(count) && pointers.segmentOffsets.allocate(count) &&
         pointers.restrictedPlaneAreas.allocate(count) &&
         pointers.segments.uploadAsync(segs.data(), count, stream) &&
         pointers.segmentOffsets.uploadAsync(offsets.data(), count, stream) &&
         pointers.restrictedPlaneAreas.uploadAsync(areas.data(), count, stream);
}

FacePartitionInput makePartitionInput(
    BaseFaceTopology &faceTopology, const ModelFacePointers &pointers,
    uint32_t modelCount, double facetFloor) {
  FacePartitionInput in{};
  in.baseFaces = faceTopology.devBaseFaces.get(); in.faceVertexOffsets = faceTopology.devFaceVertexOffsets.get();
  in.faceVertices = faceTopology.devFaceVertices.get(); in.faceVertexIds = faceTopology.devFaceVertexIds.get();
  in.baseMeasures = faceTopology.devMeasures.get(); in.models = pointers.views.get();
  in.modelSegments = pointers.segments.get(); in.modelSegmentOffsets = pointers.segmentOffsets.get();
  in.modelRestrictedPlaneAreas = pointers.restrictedPlaneAreas.get();
  in.facetFloor = facetFloor;
  in.modelCount = modelCount;
  return in;
}

bool allocatePartitionBuffers(
    uint32_t faceCount, uint32_t cutFaceCount, uint32_t peakBatchInputSegments,
    BaseFaceTopology &faceTopology, cudaStream_t stream, FacePartitionBuffers &buffers) {
  const uint32_t batchCap = std::min(std::max(cutFaceCount, 1u), OverlayBatchSize);
  if (!buffers.regionCounts.allocate(size_t(faceCount) + 1) ||
      !faceTopology.devRegionOffsets.allocate(size_t(faceCount) + 1) ||
      !buffers.packedError.allocate(1) ||
      !buffers.workspaces.allocate(batchCap)) return false;
  if (cutFaceCount > 0 &&
      !buffers.cutFaceEmittedRegions.allocate(size_t(cutFaceCount) * MaxRegionsPerCutFace)) return false;
  buffers.hostWorkspaces.resize(batchCap);

  if (cutFaceCount > 0 && peakBatchInputSegments > 0) {
    const FaceArrangementCapacities peakCapacities = makeFaceArrangementCapacities(peakBatchInputSegments);
    FaceOverlayWorkspace dummy{};
    const size_t bytesPerWorkspace = bindFaceOverlayWorkspace(peakCapacities, dummy, nullptr);
    if (!buffers.arena.ensureCapacity(batchCap * bytesPerWorkspace)) return false;
  }
  return rvdHost::zero(buffers.packedError.get(), sizeof(RvdPackedError), stream, "clear face partition error") &&
         rvdHost::zero(buffers.regionCounts.get() + faceCount, sizeof(uint32_t), stream, "clear face count tail");
}

} // namespace

bool buildFacePartitionsGPU(
    const BaseCellDomain &baseDomain, BaseFaceTopology &faceTopology,
    const std::vector<cudaGeometry::DeviceModelGeometryView> &modelViews,
    const std::vector<ModelRestriction> &models, cudaStream_t stream) {
  const uint32_t faceCount = faceTopology.totalBaseFaces;
  if (modelViews.size() != models.size()) return false;

  if (faceCount == 0) {
    faceTopology.devRegionOffsets.reset();
    faceTopology.devRegions.reset();
    return true;
  }

  if (faceTopology.devFaceVertexOffsets.size() < size_t(faceCount) + 1 ||
      faceTopology.devFaceVertices.get() == nullptr ||
      faceTopology.devMeasures.size() < faceCount) {
    return false;
  }

  ModelFacePointers pointers;
  if (!uploadModelFacePointers(modelViews, models, stream, pointers)) return false;

  FacePartitionInput input =
      makePartitionInput(faceTopology, pointers, uint32_t(modelViews.size()), baseDomain.facetFloor());

  FacePartitionBuffers buffers;
  cudaUtils::CudaBuffer<uint32_t> inputSegmentCounts;
  if (!inputSegmentCounts.allocate(faceCount) || !buffers.cadSegmentCounts.allocate(faceCount)) return false;
  constexpr uint32_t CountBlockSize = 256;
  countFaceInputSegmentsKernel<<<
      rvdHost::blockCount(faceCount, CountBlockSize), CountBlockSize, 0, stream>>>(
      input.baseFaces, input.faceVertexOffsets, input.modelSegmentOffsets,
      input.modelCount, faceCount, inputSegmentCounts.get(), buffers.cadSegmentCounts.get());
  if (!rvdHost::checkLaunch("launch countFaceInputSegmentsKernel")) return false;

  std::vector<uint32_t> hostInputSegmentCounts(faceCount);
  std::vector<uint32_t> hostCadCounts(faceCount);
  if (!inputSegmentCounts.downloadAsync(hostInputSegmentCounts.data(), faceCount, stream) ||
      !buffers.cadSegmentCounts.downloadAsync(hostCadCounts.data(), faceCount, stream) ||
      !rvdHost::synchronize(stream, "synchronize face input segment counts")) {
    return false;
  }

  std::vector<uint32_t> hostCutFaceIndices;
  hostCutFaceIndices.reserve(faceCount / 8);
  uint32_t peakBatchInputSegments = 0;
  for (uint32_t i = 0; i < faceCount; ++i) {
    if (hostCadCounts[i] > 0) {
      hostCutFaceIndices.push_back(i);
      peakBatchInputSegments = std::max(peakBatchInputSegments, hostInputSegmentCounts[i]);
    }
  }
  const uint32_t cutFaceCount = uint32_t(hostCutFaceIndices.size());
  if (cutFaceCount > 0) {
    std::sort(hostCutFaceIndices.begin(), hostCutFaceIndices.end(),
              [&](uint32_t a, uint32_t b) {
                return hostInputSegmentCounts[a] < hostInputSegmentCounts[b];
              });
    if (!buffers.cutFaceIndices.allocate(cutFaceCount) ||
        !buffers.cutFaceIndices.uploadAsync(hostCutFaceIndices.data(), cutFaceCount, stream)) {
      return false;
    }
  }

  if (!allocatePartitionBuffers(faceCount, cutFaceCount, peakBatchInputSegments, faceTopology, stream, buffers)) return false;

  constexpr uint32_t BlockSize = 256;
  setAllRegionCountsToOneKernel<<<rvdHost::blockCount(faceCount, BlockSize), BlockSize, 0, stream>>>(
      faceCount, buffers.regionCounts.get());
  if (!rvdHost::checkLaunch("launch setAllRegionCountsToOneKernel")) return false;

  if (!runArrangementPass(cutFaceCount, input, hostCutFaceIndices, hostInputSegmentCounts, buffers, stream)) return false;

  cudaUtils::CudaBuffer<uint8_t> scanScratch;
  if (!rvdHost::exclusiveSum(buffers.regionCounts.get(), faceTopology.devRegionOffsets.get(), int(faceCount + 1), scanScratch, stream)) return false;

  uint32_t totalRegions = 0;
  if (!rvdHost::readAt(totalRegions, faceTopology.devRegionOffsets.get(), faceCount, stream, "read face region count")) return false;
  if (totalRegions == 0 || !faceTopology.devRegions.allocate(totalRegions)) return false;

  emitUncutFacesKernel<<<rvdHost::blockCount(faceCount, BlockSize), BlockSize, 0, stream>>>(
      faceCount, buffers.cadSegmentCounts.get(), input,
      faceTopology.devRegionOffsets.get(), faceTopology.devRegions.get(),
      buffers.packedError.get());
  if (!rvdHost::checkLaunch("launch emitUncutFacesKernel")) return false;

  if (cutFaceCount > 0) {
    scatterCutFaceRegionsKernel<<<
        rvdHost::blockCount(cutFaceCount, BlockSize), BlockSize, 0, stream>>>(
        cutFaceCount, buffers.cutFaceIndices.get(), faceTopology.devRegionOffsets.get(),
        buffers.cutFaceEmittedRegions.get(), MaxRegionsPerCutFace,
        faceTopology.devRegions.get());
    if (!rvdHost::checkLaunch("launch scatterCutFaceRegionsKernel")) return false;
  }
  return finishPartitionPass(buffers.packedError, stream);
}

} // namespace voronoi
