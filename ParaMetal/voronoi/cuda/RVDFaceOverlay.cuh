#pragma once

#include "RVDFaceCoverage.cuh"
#include "TriangleWinding.cuh"
#include <cfloat>
#include <cmath>
#include <cstdint>

namespace voronoi {

static constexpr uint32_t MaxFaceSegments = 256;
static constexpr uint32_t MaxFacePoints = 1024;
static constexpr uint32_t MaxFaceEdges = 1024;
static constexpr uint32_t MaxFaceCycles = 64;
static constexpr uint32_t MaxCycleVertices = 128;
static constexpr uint32_t MaxFaceGroups = 64;

struct Segment {
  FacePoint2 a;
  FacePoint2 b;
};
struct Edge {
  int a = -1;
  int b = -1;
};
struct Cycle {
  FacePoint2 points[MaxCycleVertices];
  int count = 0;
  double area = 0.0;
  double areaRoundoff = 0.0;
  FacePoint2 moment{};
  int parent = -1;
};
struct FaceOverlayWorkspace {
  Segment segments[MaxFaceSegments];
  FacePoint2 points[MaxFacePoints];
  int pointEdge[MaxFacePoints];
  Edge edges[MaxFaceEdges];
  Cycle cycles[MaxFaceCycles];
  uint8_t visited[MaxFaceEdges * 2];
  FaceRegion groups[MaxFaceGroups];
  uint32_t segmentCount;
  uint32_t pointCount;
  uint32_t edgeCount;
  uint32_t cycleCount;
  uint32_t groupCount;
};
__device__ inline bool faceClipSegmentToConvex(FacePoint2 &a, FacePoint2 &b,
                                               const FacePolygon &face,
                                               double lineEps) {
  double t0 = 0.0;
  double t1 = 1.0;
  const FacePoint2 d = faceSub(b, a);
  const double segmentLength = hypot(d.x, d.y);
  for (uint32_t i = 0; i < face.count; ++i) {
    const FacePoint2 v = face.vertices[i];
    const FacePoint2 e = faceSub(face.vertices[(i + 1) % face.count], v);
    const double edgeLength = hypot(e.x, e.y);
    const double denom = faceCross(e, d);
    const double dist = faceCross(e, faceSub(a, v));
    const double determinantScale = edgeLength * segmentLength;
    if (!(determinantScale > 0.0) ||
        fabs(denom) <= 64.0 * DBL_EPSILON * determinantScale) {
      if (dist < -lineEps * edgeLength)
        return false;
      continue;
    }
    const double t = -dist / denom;
    if (denom > 0.0)
      t0 = fmax(t0, t);
    else
      t1 = fmin(t1, t);
    if (t0 > t1)
      return false;
  }
  const FacePoint2 clippedA = {a.x + d.x * t0, a.y + d.y * t0};
  const FacePoint2 clippedB = {a.x + d.x * t1, a.y + d.y * t1};
  if (hypot(clippedB.x - clippedA.x, clippedB.y - clippedA.y) <= lineEps)
    return false;
  a = clippedA;
  b = clippedB;
  return true;
}
__device__ inline uint32_t faceAddSegment(FaceOverlayWorkspace &work,
                                          FacePoint2 a, FacePoint2 b,
                                          const FacePolygon &boundary,
                                          double scale) {
  const double lineEps = 1024.0 * DBL_EPSILON * fmax(scale, 1.0);
  if (!faceClipSegmentToConvex(a, b, boundary, lineEps))
    return FacePartitionSuccess;
  const FacePoint2 e = faceSub(b, a);
  const double len = hypot(e.x, e.y);
  if (!(len > lineEps))
    return FacePartitionSuccess;
  double nx = e.y / len;
  double ny = -e.x / len;
  double c = -(nx * a.x + ny * a.y);
  if (nx < 0.0 || (nx == 0.0 && ny < 0.0)) {
    nx = -nx;
    ny = -ny;
    c = -c;
  }
  double minimum = DBL_MAX;
  double maximum = -DBL_MAX;
  for (uint32_t i = 0; i < boundary.count; ++i) {
    const double d =
        nx * boundary.vertices[i].x + ny * boundary.vertices[i].y + c;
    minimum = fmin(minimum, d);
    maximum = fmax(maximum, d);
  }
  if (minimum >= -lineEps || maximum <= lineEps)
    return FacePartitionSuccess;
  for (uint32_t k = 0; k < work.segmentCount; ++k) {
    const Segment &q = work.segments[k];
    if (fabs(q.a.x - a.x) <= lineEps && fabs(q.a.y - a.y) <= lineEps &&
        fabs(q.b.x - b.x) <= lineEps && fabs(q.b.y - b.y) <= lineEps) {
      return FacePartitionSuccess;
    }
    if (fabs(q.a.x - b.x) <= lineEps && fabs(q.a.y - b.y) <= lineEps &&
        fabs(q.b.x - a.x) <= lineEps && fabs(q.b.y - a.y) <= lineEps) {
      return FacePartitionSuccess;
    }
  }
  if (work.segmentCount == MaxFaceSegments)
    return TooManySegments;
  work.segments[work.segmentCount++] = {a, b};
  return FacePartitionSuccess;
}
__device__ inline bool
facePointInPolygon(FacePoint2 p, const FacePoint2 *polygon, int count) {
  int winding = 0;
  for (int i = 0; i < count; ++i) {
    const FacePoint2 a = polygon[i];
    const FacePoint2 b = polygon[(i + 1) % count];
    if (a.y <= p.y) {
      if (b.y > p.y && faceCross(faceSub(b, a), faceSub(p, a)) > 0.0)
        ++winding;
    } else {
      if (b.y <= p.y && faceCross(faceSub(b, a), faceSub(p, a)) < 0.0)
        --winding;
    }
  }
  return winding != 0;
}
__device__ inline double faceCycleArea(const Cycle &cycle, double &roundoff) {
  double area = 0.0;
  double absolute = 0.0;
  for (int i = 0; i < cycle.count; ++i) {
    const double term =
        faceCross(cycle.points[i], cycle.points[(i + 1) % cycle.count]);
    area += term;
    absolute += fabs(term);
  }
  roundoff = 128.0 * DBL_EPSILON * absolute;
  return 0.5 * area;
}
__device__ inline bool faceSegmentIntersection(const Segment &s0,
                                               const Segment &s1,
                                               FacePoint2 &out, double eps) {
  const FacePoint2 d0 = faceSub(s0.b, s0.a);
  const FacePoint2 d1 = faceSub(s1.b, s1.a);
  const double denom = faceCross(d0, d1);
  const double determinantScale = hypot(d0.x, d0.y) * hypot(d1.x, d1.y);
  if (!(determinantScale > 0.0) ||
      fabs(denom) <= 64.0 * DBL_EPSILON * determinantScale) {
    return false;
  }
  const double t = faceCross(faceSub(s1.a, s0.a), d1) / denom;
  const double u = faceCross(faceSub(s1.a, s0.a), d0) / denom;
  const double len0 = hypot(d0.x, d0.y);
  const double len1 = hypot(d1.x, d1.y);
  const double tEps = eps / fmax(len0, 1.0e-300);
  const double uEps = eps / fmax(len1, 1.0e-300);
  if (t < -tEps || t > 1.0 + tEps || u < -uEps || u > 1.0 + uEps)
    return false;
  out = faceAdd(s0.a, faceMul(d0, t));
  return true;
}
__device__ inline int faceSplitSegments(const Segment *input, int inputCount,
                                        int boundaryInputCount, Segment *output,
                                        int *outputSource, double eps) {
  int outputCount = 0;
  for (int i = 0; i < inputCount; ++i) {
    double params[MaxFaceSegments];
    int paramCount = 0;
    params[paramCount++] = 0.0;
    params[paramCount++] = 1.0;
    const FacePoint2 d = faceSub(input[i].b, input[i].a);
    const double squaredLength = faceDot(d, d);
    const double tEps = eps / fmax(sqrt(squaredLength), 1.0e-300);
    for (int j = 0; j < inputCount; ++j) {
      if (i == j)
        continue;
      FacePoint2 p;
      if (faceSegmentIntersection(input[i], input[j], p, eps) &&
          squaredLength > 0.0) {
        const double t = faceDot(faceSub(p, input[i].a), d) / squaredLength;
        if (t >= -tEps && t <= 1.0 + tEps) {
          if (paramCount >= int(MaxFaceSegments))
            return -1;
          params[paramCount++] = fmin(1.0, fmax(0.0, t));
        }
      }
      const FacePoint2 endpoints[2] = {input[j].a, input[j].b};
      for (int e = 0; e < 2; ++e) {
        if (!(squaredLength > 0.0))
          continue;
        const double t =
            faceDot(faceSub(endpoints[e], input[i].a), d) / squaredLength;
        if (t <= tEps || t >= 1.0 - tEps)
          continue;
        const FacePoint2 projected = faceAdd(input[i].a, faceMul(d, t));
        if (hypot(projected.x - endpoints[e].x, projected.y - endpoints[e].y) >
            eps)
          continue;
        if (paramCount >= int(MaxFaceSegments))
          return -1;
        params[paramCount++] = t;
      }
    }
    for (int a = 1; a < paramCount; ++a) {
      const double value = params[a];
      int b = a - 1;
      while (b >= 0 && params[b] > value) {
        params[b + 1] = params[b];
        --b;
      }
      params[b + 1] = value;
    }
    int uniqueCount = 0;
    for (int p = 0; p < paramCount; ++p) {
      if (uniqueCount > 0 &&
          fabs(params[p] - params[uniqueCount - 1]) <= tEps) {
        if (params[p] == 0.0 || params[p] == 1.0) {
          params[uniqueCount - 1] = params[p];
        }
        continue;
      }
      params[uniqueCount++] = params[p];
    }
    for (int p = 0; p + 1 < uniqueCount; ++p) {
      if (params[p + 1] - params[p] <= tEps)
        continue;
      Segment segment{};
      segment.a = params[p] == 0.0 ? input[i].a
                                   : faceAdd(input[i].a, faceMul(d, params[p]));
      segment.b = params[p + 1] == 1.0
                      ? input[i].b
                      : faceAdd(input[i].a, faceMul(d, params[p + 1]));
      if (outputCount >= int(MaxFaceSegments))
        return -1;
      output[outputCount] = segment;
      outputSource[outputCount] = i;
      ++outputCount;
    }
  }
  return outputCount;
}
__device__ inline int faceFindPoint(const FacePoint2 *points, int count,
                                    FacePoint2 p, double eps) {
  for (int i = 0; i < count; ++i) {
    if (points[i].x == p.x && points[i].y == p.y)
      return i;
  }
  int best = -1;
  double bestDistance = eps;
  for (int i = 0; i < count; ++i) {
    const double distance = hypot(points[i].x - p.x, points[i].y - p.y);
    if (distance <= bestDistance) {
      best = i;
      bestDistance = distance;
    }
  }
  return best;
}
__device__ inline int faceFindSplitPoint(const FacePoint2 *points, int count,
                                         int canonicalCount,
                                         const int *pointEdge, int edge,
                                         FacePoint2 p, double eps) {
  for (int i = 0; i < count; ++i) {
    if (points[i].x == p.x && points[i].y == p.y &&
        (i < canonicalCount || pointEdge[i] == edge)) {
      return i;
    }
  }
  int bestCanonical = -1;
  double bestCanonicalDistance = eps;
  for (int i = 0; i < canonicalCount; ++i) {
    const double distance = hypot(points[i].x - p.x, points[i].y - p.y);
    if (distance <= bestCanonicalDistance) {
      bestCanonical = i;
      bestCanonicalDistance = distance;
    }
  }
  if (bestCanonical >= 0)
    return bestCanonical;
  int best = -1;
  double bestDistance = eps;
  for (int i = canonicalCount; i < count; ++i) {
    if (pointEdge[i] != edge)
      continue;
    const double distance = hypot(points[i].x - p.x, points[i].y - p.y);
    if (distance <= bestDistance) {
      best = i;
      bestDistance = distance;
    }
  }
  return best;
}
__device__ inline int
faceBuildPoints(const FacePoint2 *canonical, int canonicalCount,
                const Segment *segments, const int *segmentSource,
                int boundaryCount, int count, FacePoint2 *points,
                int *pointEdge, double eps) {
  if (canonicalCount > int(MaxFacePoints))
    return -1;
  int pointCount = 0;
  for (int i = 0; i < canonicalCount; ++i) {
    points[pointCount] = canonical[i];
    pointEdge[pointCount] = -1;
    ++pointCount;
  }
  for (int i = 0; i < count; ++i) {
    const int source = segmentSource[i];
    for (int k = 0; k < 2; ++k) {
      const FacePoint2 p = k == 0 ? segments[i].a : segments[i].b;
      const int id =
          source < boundaryCount
              ? faceFindSplitPoint(points, pointCount, canonicalCount,
                                   pointEdge, source, p, eps)
              : faceFindPoint(points, pointCount, p, eps);
      if (id < 0) {
        if (pointCount >= int(MaxFacePoints))
          return -1;
        points[pointCount] = p;
        pointEdge[pointCount] = source < boundaryCount ? source : -1;
        ++pointCount;
      }
    }
  }
  return pointCount;
}
__device__ inline double faceAngleOf(FacePoint2 d) { return atan2(d.y, d.x); }
__device__ inline uint32_t
faceClassify(double3 sample, double3 normal, double scale,
             const cudaGeometry::DeviceModelGeometryView *views,
             uint32_t modelCount, const FaceSideCoverage &sideA,
             const FaceSideCoverage &sideB, uint32_t &materialA,
             uint32_t &materialB) {
  using namespace cudaGeometry;
  materialA = sideA.resolved ? sideA.material : InvalidId;
  materialB = sideB.resolved ? sideB.material : InvalidId;
  const bool unresolvedA = !sideA.resolved;
  const bool unresolvedB = !sideB.resolved;
  if (!unresolvedA && !unresolvedB)
    return FacePartitionSuccess;

  const double positionScale =
      fmax(fmax(fabs(sample.x), fabs(sample.y)), fabs(sample.z));
  const double delta = fmax(16.0 * DBL_EPSILON * fmax(positionScale, 1.0),
                            8.0 * DBL_EPSILON * fmax(scale, 1.0e-20));
  const double3 pointA = sub3(sample, mul3(normal, delta));
  const double3 pointB = add3(sample, mul3(normal, delta));
  for (uint32_t i = 0; i < modelCount; ++i) {
    const auto &m = views[i];
    const TriangleBvhNode root = m.nodes[m.root];
    const double extent = 2.0 * delta;
    double wa = 0.0;
    double wb = 0.0;
    const bool considerA =
        unresolvedA && (sideA.positiveCount != 1 || i == sideA.modelIndex);
    const bool considerB =
        unresolvedB && (sideB.positiveCount != 1 || i == sideB.modelIndex);
    if (considerA) {
      const bool outsideA = pointA.x < double(root.minimum.x) - extent ||
                            pointA.x > double(root.maximum.x) + extent ||
                            pointA.y < double(root.minimum.y) - extent ||
                            pointA.y > double(root.maximum.y) + extent ||
                            pointA.z < double(root.minimum.z) - extent ||
                            pointA.z > double(root.maximum.z) + extent;
      if (!outsideA) {
        wa = computeWinding(
            m.triangles, m.nodes, m.parents, m.root, m.moments, pointA,
            make_float3(float(pointA.x), float(pointA.y), float(pointA.z)));
      }
      if (!isfinite(wa))
        return AmbiguousOwnership;
      if (fabs(fabs(wa) - 0.5) < 0.01)
        return AmbiguousOwnership;
      if (fabs(wa) > 0.5) {
        if (materialA != InvalidId)
          return OverlappingModels;
        materialA = m.runtimeModelId;
      }
    }
    if (considerB) {
      const bool outsideB = pointB.x < double(root.minimum.x) - extent ||
                            pointB.x > double(root.maximum.x) + extent ||
                            pointB.y < double(root.minimum.y) - extent ||
                            pointB.y > double(root.maximum.y) + extent ||
                            pointB.z < double(root.minimum.z) - extent ||
                            pointB.z > double(root.maximum.z) + extent;
      if (!outsideB) {
        wb = computeWinding(
            m.triangles, m.nodes, m.parents, m.root, m.moments, pointB,
            make_float3(float(pointB.x), float(pointB.y), float(pointB.z)));
      }
      if (!isfinite(wb))
        return AmbiguousOwnership;
      if (fabs(fabs(wb) - 0.5) < 0.01)
        return AmbiguousOwnership;
      if (fabs(wb) > 0.5) {
        if (materialB != InvalidId)
          return OverlappingModels;
        materialB = m.runtimeModelId;
      }
    }
  }
  return FacePartitionSuccess;
}
__device__ inline uint32_t faceCollectRestrictedSegments(
    const BaseFace &face, const FacePolygon &boundary, double3 origin,
    double3 u, double3 v, double scale,
    const FaceBoundarySegment *const *modelSegments,
    const uint32_t *const *modelSegmentOffsets, uint32_t modelCount,
    uint32_t &boundaryCount, FaceOverlayWorkspace &work) {
  work.segmentCount = 0;
  boundaryCount = 0;
  for (uint32_t i = 0; i < boundary.count; ++i) {
    if (work.segmentCount == MaxFaceSegments)
      return TooManySegments;
    work.segments[work.segmentCount++] = {
        boundary.vertices[i], boundary.vertices[(i + 1) % boundary.count]};
  }
  boundaryCount = work.segmentCount;
  if (modelSegments == nullptr || modelSegmentOffsets == nullptr)
    return FacePartitionSuccess;
  const uint32_t before = work.segmentCount;
  const uint32_t slots[2] = {face.slotA, face.slotB};
  for (uint32_t s = 0; s < 2; ++s) {
    const uint32_t slot = slots[s];
    if (slot == InvalidId)
      continue;
    for (uint32_t g = 0; g < modelCount; ++g) {
      if (modelSegments[g] == nullptr || modelSegmentOffsets[g] == nullptr)
        continue;
      const uint32_t begin = modelSegmentOffsets[g][slot];
      const uint32_t end = modelSegmentOffsets[g][slot + 1];
      for (uint32_t i = begin; i < end; ++i) {
        const FaceBoundarySegment &segment = modelSegments[g][i];
        const FacePoint2 a = faceProject(segment.a, origin, u, v);
        const FacePoint2 b = faceProject(segment.b, origin, u, v);
        const uint32_t error = faceAddSegment(work, a, b, boundary, scale);
        if (error)
          return error;
      }
    }
    if (s == 0 && work.segmentCount > before)
      break;
  }
  return FacePartitionSuccess;
}
__device__ inline uint32_t faceBuildOverlay(
    uint32_t faceId, const BaseFaceMeasure &base, const FacePolygon &boundary,
    double3 origin, double3 u, double3 v, double3 normal, double scale,
    const cudaGeometry::DeviceModelGeometryView *models, uint32_t modelCount,
    const FaceSideCoverage &sideA, const FaceSideCoverage &sideB,
    uint32_t boundaryCount, FaceOverlayWorkspace &work) {
  const double eps = 1024.0 * DBL_EPSILON * fmax(scale, 1.0);
  Segment split[MaxFaceSegments];
  int splitSource[MaxFaceSegments];
  const int splitCount =
      faceSplitSegments(work.segments, int(work.segmentCount),
                        int(boundaryCount), split, splitSource, eps);
  if (splitCount < 0) {
    printf("[RVD-FacePartition] segmentOverflowSplit face=%u\n", faceId);
    return TooManySegments;
  }
  const int pointCount = faceBuildPoints(
      boundary.vertices, int(boundary.count), split, splitSource,
      int(boundaryCount), splitCount, work.points, work.pointEdge, eps);
  if (pointCount < 0) {
    printf("[RVD-FacePartition] pointOverflow face=%u\n", faceId);
    return TooManyRegions;
  }
  work.pointCount = uint32_t(pointCount);

  work.edgeCount = 0;
  for (int i = 0; i < splitCount; ++i) {
    const int source = splitSource[i];
    const bool isBoundary = source < int(boundaryCount);
    int a =
        isBoundary
            ? faceFindSplitPoint(work.points, pointCount, int(boundary.count),
                                 work.pointEdge, source, split[i].a, eps)
            : faceFindPoint(work.points, pointCount, split[i].a, eps);
    int b =
        isBoundary
            ? faceFindSplitPoint(work.points, pointCount, int(boundary.count),
                                 work.pointEdge, source, split[i].b, eps)
            : faceFindPoint(work.points, pointCount, split[i].b, eps);
    if (a < 0 || b < 0 || a == b)
      continue;
    if (a > b) {
      const int t = a;
      a = b;
      b = t;
    }
    bool duplicate = false;
    for (uint32_t e = 0; e < work.edgeCount; ++e) {
      if (work.edges[e].a == a && work.edges[e].b == b) {
        duplicate = true;
        break;
      }
    }
    if (duplicate)
      continue;
    if (work.edgeCount >= MaxFaceEdges) {
      printf("[RVD-FacePartition] edgeOverflow face=%u\n", faceId);
      return TooManyRegions;
    }
    work.edges[work.edgeCount++] = {a, b};
  }

  for (uint32_t i = 0; i < work.edgeCount * 2; ++i)
    work.visited[i] = 0;
  work.cycleCount = 0;
  for (uint32_t e = 0; e < work.edgeCount; ++e) {
    for (int dir = 0; dir < 2; ++dir) {
      const int startFrom = dir == 0 ? work.edges[e].a : work.edges[e].b;
      const int startTo = dir == 0 ? work.edges[e].b : work.edges[e].a;
      if (work.visited[e * 2 + dir])
        continue;
      int from = startFrom;
      int to = startTo;
      int edge = int(e);
      int direction = dir;
      Cycle cycle{};
      bool closed = false;
      for (int guard = 0; guard < int(work.edgeCount) * 4 + 16; ++guard) {
        if (work.visited[edge * 2 + direction]) {
          closed = (from == startFrom && to == startTo);
          break;
        }
        work.visited[edge * 2 + direction] = 1;
        if (cycle.count >= int(MaxCycleVertices)) {
          printf("[RVD-FacePartition] cycleVertexOverflow face=%u\n", faceId);
          return TooManyRegions;
        }
        cycle.points[cycle.count++] = work.points[from];
        const double reverseAngle =
            faceAngleOf(faceSub(work.points[from], work.points[to]));
        int bestEdge = -1;
        int bestDir = 0;
        double bestAngle = -1e300;
        int wrapEdge = -1;
        int wrapDir = 0;
        double wrapAngle = -1e300;
        for (uint32_t j = 0; j < work.edgeCount; ++j) {
          if (work.edges[j].a == to) {
            const double angle = faceAngleOf(
                faceSub(work.points[work.edges[j].b], work.points[to]));
            if (angle > wrapAngle) {
              wrapAngle = angle;
              wrapEdge = int(j);
              wrapDir = 0;
            }
            if (angle < reverseAngle - 1e-12 && angle > bestAngle) {
              bestAngle = angle;
              bestEdge = int(j);
              bestDir = 0;
            }
          }
          if (work.edges[j].b == to) {
            const double angle = faceAngleOf(
                faceSub(work.points[work.edges[j].a], work.points[to]));
            if (angle > wrapAngle) {
              wrapAngle = angle;
              wrapEdge = int(j);
              wrapDir = 1;
            }
            if (angle < reverseAngle - 1e-12 && angle > bestAngle) {
              bestAngle = angle;
              bestEdge = int(j);
              bestDir = 1;
            }
          }
        }
        if (bestEdge < 0) {
          bestEdge = wrapEdge;
          bestDir = wrapDir;
        }
        if (bestEdge < 0)
          break;
        from = to;
        edge = bestEdge;
        direction = bestDir;
        to = direction == 0 ? work.edges[edge].b : work.edges[edge].a;
      }
      if (!closed || cycle.count < 3) {
        continue;
      }
      cycle.area = faceCycleArea(cycle, cycle.areaRoundoff);
      if (fabs(cycle.area) <= cycle.areaRoundoff)
        continue;
      for (int i = 0; i < cycle.count; ++i) {
        const FacePoint2 &a = cycle.points[i];
        const FacePoint2 &b = cycle.points[(i + 1) % cycle.count];
        const double c = faceCross(a, b);
        cycle.moment.x += (a.x + b.x) * c / 6.0;
        cycle.moment.y += (a.y + b.y) * c / 6.0;
      }
      if (work.cycleCount >= MaxFaceCycles) {
        printf("[RVD-FacePartition] cycleOverflow face=%u\n", faceId);
        return TooManyRegions;
      }
      work.cycles[work.cycleCount++] = cycle;
    }
  }

  for (uint32_t i = 0; i < work.cycleCount; ++i)
    work.cycles[i].parent = -1;
  for (uint32_t i = 0; i < work.cycleCount; ++i) {
    if (work.cycles[i].area >= 0.0)
      continue;
    FacePoint2 probe{};
    bool haveProbe = false;
    for (int p = 0; p < work.cycles[i].count; ++p) {
      const FacePoint2 a = work.cycles[i].points[p];
      const FacePoint2 b =
          work.cycles[i].points[(p + 1) % work.cycles[i].count];
      const FacePoint2 e = faceSub(b, a);
      const double len = hypot(e.x, e.y);
      if (!(len > eps))
        continue;
      const FacePoint2 outward{-e.y / len, e.x / len};
      probe = faceAdd(faceMul(faceAdd(a, b), 0.5),
                      faceMul(outward, fmax(eps, len * 1e-6)));
      haveProbe = true;
      break;
    }
    if (!haveProbe)
      continue;
    for (uint32_t j = 0; j < work.cycleCount; ++j) {
      if (work.cycles[j].area <= 0.0)
        continue;
      if (facePointInPolygon(probe, work.cycles[j].points,
                             work.cycles[j].count)) {
        if (work.cycles[i].parent < 0 ||
            work.cycles[j].area < work.cycles[work.cycles[i].parent].area) {
          work.cycles[i].parent = int(j);
        }
      }
    }
  }

  double netArea[MaxFaceCycles];
  double netRoundoff[MaxFaceCycles];
  FacePoint2 netMoment[MaxFaceCycles];
  for (uint32_t i = 0; i < work.cycleCount; ++i) {
    netArea[i] = work.cycles[i].area;
    netRoundoff[i] = work.cycles[i].areaRoundoff;
    netMoment[i] = work.cycles[i].moment;
    if (work.cycles[i].area <= 0.0)
      continue;
    for (uint32_t j = 0; j < work.cycleCount; ++j) {
      if (work.cycles[j].area < 0.0 && work.cycles[j].parent == int(i)) {
        netArea[i] += work.cycles[j].area;
        netRoundoff[i] += work.cycles[j].areaRoundoff;
        netMoment[i] = faceAdd(netMoment[i], work.cycles[j].moment);
      }
    }
  }

  work.groupCount = 0;
  for (uint32_t i = 0; i < work.cycleCount; ++i) {
    if (netArea[i] <= netRoundoff[i])
      continue;
    FacePoint2 sample =
        faceMul(work.cycles[i].moment, 1.0 / work.cycles[i].area);
    bool foundSample =
        facePointInPolygon(sample, work.cycles[i].points, work.cycles[i].count);
    if (foundSample) {
      for (uint32_t j = 0; j < work.cycleCount; ++j) {
        if (work.cycles[j].parent == int(i) &&
            facePointInPolygon(sample, work.cycles[j].points,
                               work.cycles[j].count)) {
          foundSample = false;
          break;
        }
      }
    }
    if (!foundSample) {
      for (int p = 0; p < work.cycles[i].count; ++p) {
        const FacePoint2 a = work.cycles[i].points[p];
        const FacePoint2 b =
            work.cycles[i].points[(p + 1) % work.cycles[i].count];
        const FacePoint2 e = faceSub(b, a);
        const double len = hypot(e.x, e.y);
        if (len <= eps)
          continue;
        const FacePoint2 inward{-e.y / len, e.x / len};
        const FacePoint2 probe =
            faceAdd(faceMul(faceAdd(a, b), 0.5),
                    faceMul(inward, fmax(eps, len * 1e-6)));
        bool insideChild = false;
        for (uint32_t j = 0; j < work.cycleCount; ++j) {
          if (work.cycles[j].parent == int(i) &&
              facePointInPolygon(probe, work.cycles[j].points,
                                 work.cycles[j].count)) {
            insideChild = true;
            break;
          }
        }
        if (!insideChild && facePointInPolygon(probe, work.cycles[i].points,
                                               work.cycles[i].count)) {
          sample = probe;
          foundSample = true;
          break;
        }
      }
    }
    if (!foundSample) {
      printf("[RVD-FacePartition] noRegionSample face=%u cycle=%u\n", faceId,
             i);
      return UnresolvedRegion;
    }

    const double3 sample3 =
        add3(origin, add3(mul3(u, sample.x), mul3(v, sample.y)));
    uint32_t materialA = InvalidId;
    uint32_t materialB = InvalidId;
    const uint32_t classify =
        faceClassify(sample3, normal, scale, models, modelCount, sideA, sideB,
                     materialA, materialB);
    if (classify)
      return classify;
    const double3 worldMoment =
        add3(mul3(origin, netArea[i]),
             add3(mul3(u, netMoment[i].x), mul3(v, netMoment[i].y)));
    const uint32_t emit = faceEmitRegion(
        work.groups, work.groupCount, MaxFaceGroups, netArea[i], worldMoment,
        normal, materialA, materialB,
        fmax(base.areaError,
             fmax(256.0 * DBL_EPSILON * netArea[i], netRoundoff[i])));
    if (emit)
      return emit;
  }
  return FacePartitionSuccess;
}

} // namespace voronoi
