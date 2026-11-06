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

namespace {

constexpr uint32_t BlockSize = rvdHost::BlockSize;
enum SideFailure : uint32_t {
  SideSuccess = 0,
  SideInvalidProbe = 1,
  SideNonFiniteWinding = 2,
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

struct SurfaceFrame {
  glm::dvec3 origin{0.0};
  glm::dvec3 u{1.0, 0.0, 0.0};
  glm::dvec3 v{0.0, 1.0, 0.0};
  std::vector<glm::dvec2> polygon;
};

struct SurfaceCut {
  glm::dvec3 point{0.0};
  glm::dvec3 normal{0.0};
};

struct HostSurfacePatch {
  SurfaceRegion region;
  std::vector<SurfaceCut> cuts;
};

struct MaterialProbe {
  double3 centroid;
  double3 normal;
  double probeDistance;
  uint32_t ownerModelIndex;
};

double cross2(const glm::dvec2 &a, const glm::dvec2 &b) {
  return a.x * b.y - a.y * b.x;
}

bool makeSurfaceFrame(const SurfaceRegion &region, SurfaceFrame &frame) {
  if (region.polygon.size() < 3) return false;
  frame.origin = region.polygon[0];
  frame.u = rvdHost::makeBasisU(region.normal);
  frame.v = glm::cross(region.normal, frame.u);
  frame.polygon.clear();
  frame.polygon.reserve(region.polygon.size());
  for (const glm::dvec3 &point : region.polygon) {
    const glm::dvec3 delta = point - frame.origin;
    frame.polygon.emplace_back(glm::dot(delta, frame.u), glm::dot(delta, frame.v));
  }
  double signedArea = 0.0;
  for (size_t i = 0; i < frame.polygon.size(); ++i) {
    signedArea += cross2(frame.polygon[i],
                         frame.polygon[(i + 1) % frame.polygon.size()]);
  }
  if (signedArea < 0.0) std::reverse(frame.polygon.begin(), frame.polygon.end());
  return std::isfinite(signedArea) && signedArea != 0.0;
}

bool clipPolygon(
    const std::vector<glm::dvec2> &input,
    double a,
    double b,
    double c,
    std::vector<glm::dvec2> &output) {
  output.clear();
  if (input.empty()) return true;
  glm::dvec2 previous = input.back();
  double previousDistance = a * previous.x + b * previous.y + c;
  for (const glm::dvec2 &current : input) {
    const double currentDistance = a * current.x + b * current.y + c;
    if ((previousDistance < 0.0 && currentDistance > 0.0) ||
        (previousDistance > 0.0 && currentDistance < 0.0)) {
      const double t = previousDistance / (previousDistance - currentDistance);
      const glm::dvec2 intersection = previous + t * (current - previous);
      if (output.empty() || intersection.x != output.back().x ||
          intersection.y != output.back().y) {
        output.push_back(intersection);
      }
    }
    if (currentDistance >= 0.0 &&
        (output.empty() || current.x != output.back().x ||
         current.y != output.back().y)) {
      output.push_back(current);
    }
    previous = current;
    previousDistance = currentDistance;
  }
  if (output.size() > 1 && output.front().x == output.back().x &&
      output.front().y == output.back().y) {
    output.pop_back();
  }
  return true;
}

double polygonArea(const std::vector<glm::dvec2> &polygon) {
  double twiceArea = 0.0;
  for (size_t i = 0; i < polygon.size(); ++i) {
    twiceArea += cross2(polygon[i], polygon[(i + 1) % polygon.size()]);
  }
  return 0.5 * std::abs(twiceArea);
}

bool lineInterval(
    const SurfaceFrame &frame,
    const glm::dvec3 &linePoint,
    const glm::dvec3 &lineDirection,
    double tolerance,
    double &minimum,
    double &maximum) {
  const glm::dvec3 delta = linePoint - frame.origin;
  const glm::dvec2 point(glm::dot(delta, frame.u), glm::dot(delta, frame.v));
  const glm::dvec2 direction(
      glm::dot(lineDirection, frame.u), glm::dot(lineDirection, frame.v));
  minimum = -std::numeric_limits<double>::infinity();
  maximum = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < frame.polygon.size(); ++i) {
    const glm::dvec2 a = frame.polygon[i];
    const glm::dvec2 edge = frame.polygon[(i + 1) % frame.polygon.size()] - a;
    const double constant = cross2(edge, point - a);
    const double slope = cross2(edge, direction);
    const double edgeTolerance = tolerance * glm::length(edge);
    if (std::abs(slope) <= 64.0 * DBL_EPSILON * glm::length(edge)) {
      if (constant < -edgeTolerance) return false;
      continue;
    }
    const double bound = (-edgeTolerance - constant) / slope;
    if (slope > 0.0) minimum = std::max(minimum, bound);
    else maximum = std::min(maximum, bound);
    if (minimum > maximum) return false;
  }
  return true;
}

bool surfacesIntersectInArea(
    const SurfaceRegion &regionA,
    const SurfaceFrame &frameA,
    const SurfaceRegion &regionB,
    const SurfaceFrame &frameB,
    double positionTolerance,
    bool &coplanar) {
  coplanar = false;
  const glm::dvec3 crossNormals = glm::cross(regionA.normal, regionB.normal);
  const double directionLength = glm::length(crossNormals);
  const double planeDistance =
      std::abs(glm::dot(regionA.normal, frameB.origin - frameA.origin));
  if (directionLength <= 512.0 * DBL_EPSILON) {
    if (planeDistance > positionTolerance) return false;
    std::vector<glm::dvec2> projectedB;
    projectedB.reserve(regionB.polygon.size());
    for (const glm::dvec3 &point : regionB.polygon) {
      const glm::dvec3 delta = point - frameA.origin;
      projectedB.emplace_back(glm::dot(delta, frameA.u), glm::dot(delta, frameA.v));
    }
    double signedArea = 0.0;
    for (size_t i = 0; i < projectedB.size(); ++i) {
      signedArea += cross2(projectedB[i], projectedB[(i + 1) % projectedB.size()]);
    }
    if (signedArea < 0.0) std::reverse(projectedB.begin(), projectedB.end());
    std::vector<glm::dvec2> overlap = frameA.polygon;
    std::vector<glm::dvec2> next;
    for (size_t edge = 0; edge < projectedB.size(); ++edge) {
      const glm::dvec2 a = projectedB[edge];
      const glm::dvec2 b = projectedB[(edge + 1) % projectedB.size()];
      if (!clipPolygon(overlap, a.y - b.y, b.x - a.x,
                       b.y * a.x - b.x * a.y, next)) {
        return true;
      }
      overlap.swap(next);
      if (overlap.size() < 3) return false;
    }
    coplanar = polygonArea(overlap) >
        256.0 * DBL_EPSILON * polygonArea(frameA.polygon);
    return coplanar;
  }

  const glm::dvec3 direction = crossNormals / directionLength;
  const double offsetA = glm::dot(regionA.normal, frameA.origin);
  const double offsetB = glm::dot(regionB.normal, frameB.origin);
  const glm::dvec3 linePoint =
      (offsetA * glm::cross(regionB.normal, crossNormals) +
       offsetB * glm::cross(crossNormals, regionA.normal)) /
      (directionLength * directionLength);
  double minA, maxA, minB, maxB;
  if (!lineInterval(frameA, linePoint, direction, positionTolerance, minA, maxA) ||
      !lineInterval(frameB, linePoint, direction, positionTolerance, minB, maxB)) {
    return false;
  }
  return std::min(maxA, maxB) - std::max(minA, minB) > positionTolerance;
}

bool measurePolygon(
    const std::vector<glm::dvec2> &polygon,
    double &area,
    glm::dvec2 &centroid) {
  area = 0.0;
  centroid = glm::dvec2(0.0);
  if (polygon.size() < 3) return true;
  double twiceArea = 0.0;
  double absoluteSum = 0.0;
  glm::dvec2 centroidNumerator(0.0);
  for (size_t i = 0; i < polygon.size(); ++i) {
    const glm::dvec2 &a = polygon[i];
    const glm::dvec2 &b = polygon[(i + 1) % polygon.size()];
    const double term = cross2(a, b);
    twiceArea += term;
    absoluteSum += std::abs(term);
    centroidNumerator += (a + b) * term;
  }
  const double roundoff = 256.0 * DBL_EPSILON * absoluteSum;
  area = 0.5 * std::abs(twiceArea);
  if (!std::isfinite(area) || !std::isfinite(roundoff)) return false;
  if (area <= roundoff) {
    area = 0.0;
    return true;
  }
  centroid = centroidNumerator / (3.0 * twiceArea);
  return std::isfinite(centroid.x) && std::isfinite(centroid.y);
}

bool splitPatchAtCuts(
    const SurfaceRegion &source,
    const SurfaceFrame &frame,
    const std::vector<SurfaceCut> &cuts,
    std::vector<SurfaceRegion> &pieces) {
  double sourcePolygonArea = 0.0;
  glm::dvec2 sourceCentroid;
  if (!measurePolygon(frame.polygon, sourcePolygonArea, sourceCentroid) ||
      !(sourcePolygonArea > 0.0)) {
    return false;
  }

  std::vector<std::vector<glm::dvec2>> polygons{frame.polygon};
  for (const SurfaceCut &cut : cuts) {
    const double a = glm::dot(cut.normal, frame.u);
    const double b = glm::dot(cut.normal, frame.v);
    const double c = glm::dot(cut.normal, frame.origin - cut.point);
    std::vector<std::vector<glm::dvec2>> split;
    for (const std::vector<glm::dvec2> &polygon : polygons) {
      std::vector<glm::dvec2> positive, negative;
      clipPolygon(polygon, a, b, c, positive);
      clipPolygon(polygon, -a, -b, -c, negative);
      double positiveArea = 0.0, negativeArea = 0.0;
      glm::dvec2 ignored;
      if (!measurePolygon(positive, positiveArea, ignored) ||
          !measurePolygon(negative, negativeArea, ignored)) {
        return false;
      }
      if (positiveArea == 0.0 || negativeArea == 0.0) {
        split.push_back(polygon);
      } else {
        split.push_back(std::move(positive));
        split.push_back(std::move(negative));
      }
    }
    polygons.swap(split);
  }

  double totalArea = 0.0;
  pieces.clear();
  pieces.reserve(polygons.size());
  for (const std::vector<glm::dvec2> &polygon : polygons) {
    double area = 0.0;
    glm::dvec2 centroid2;
    if (!measurePolygon(polygon, area, centroid2)) return false;
    if (!(area > 0.0)) continue;
    SurfaceRegion piece = source;
    piece.polygon.clear();
    piece.polygon.reserve(polygon.size());
    for (const glm::dvec2 &point : polygon) {
      piece.polygon.push_back(
          frame.origin + frame.u * point.x + frame.v * point.y);
    }
    const glm::dvec3 centroid =
        frame.origin + frame.u * centroid2.x + frame.v * centroid2.y;
    piece.area = area;
    piece.moment = centroid * area;
    piece.faceId = InvalidId;
    totalArea += area;
    pieces.push_back(std::move(piece));
  }
  const double tolerance =
      512.0 * DBL_EPSILON * sourcePolygonArea * double(cuts.size() + 1);
  return !pieces.empty() &&
      std::abs(totalArea - sourcePolygonArea) <= tolerance;
}

__device__ inline bool pointInsideModelBounds(
    const cudaGeometry::DeviceModelGeometryView &view,
    double3 point) {
  if (view.nodes == nullptr) return false;
  return point.x >= double(view.boundsMin.x) &&
         point.x <= double(view.boundsMax.x) &&
         point.y >= double(view.boundsMin.y) &&
         point.y <= double(view.boundsMax.y) &&
         point.z >= double(view.boundsMin.z) &&
         point.z <= double(view.boundsMax.z);
}

__device__ inline double modelWinding(
    const cudaGeometry::DeviceModelGeometryView &view,
    double3 point) {
  if (view.nodes == nullptr || view.triangles == nullptr ||
      view.parents == nullptr || view.moments == nullptr ||
      view.root == cudaGeometry::InvalidNodeId) {
    return 0.0;
  }
  return cudaGeometry::computeWinding(
      view.triangles, view.nodes, view.parents, view.root, view.moments,
      point, make_float3(float(point.x), float(point.y), float(point.z)));
}

__global__ void classifyMaterialRegionsKernel(
    const MaterialProbe *probes,
    uint32_t probeCount,
    const cudaGeometry::DeviceModelGeometryView *models,
    uint32_t modelCount,
    PatchClassification *classifications) {
  const uint32_t region = blockIdx.x * blockDim.x + threadIdx.x;
  if (region >= probeCount) return;
  const MaterialProbe probe = probes[region];
  PatchClassification result{};
  if (probe.ownerModelIndex >= modelCount ||
      !(probe.probeDistance > 0.0) || !isfinite(probe.probeDistance)) {
    result.valid = 0;
    result.failure = SideInvalidProbe;
    classifications[region] = result;
    return;
  }

  const double3 minus = make_double3(
      probe.centroid.x - probe.normal.x * probe.probeDistance,
      probe.centroid.y - probe.normal.y * probe.probeDistance,
      probe.centroid.z - probe.normal.z * probe.probeDistance);
  const double3 plus = make_double3(
      probe.centroid.x + probe.normal.x * probe.probeDistance,
      probe.centroid.y + probe.normal.y * probe.probeDistance,
      probe.centroid.z + probe.normal.z * probe.probeDistance);
  result.windingMinus = modelWinding(models[probe.ownerModelIndex], minus);
  result.windingPlus = modelWinding(models[probe.ownerModelIndex], plus);
  if (!isfinite(result.windingMinus) || !isfinite(result.windingPlus)) {
    result.valid = 0;
    result.failure = SideNonFiniteWinding;
    classifications[region] = result;
    return;
  }

  const bool insideMinus = fabs(result.windingMinus) > 0.5;
  const bool insidePlus = fabs(result.windingPlus) > 0.5;
  if (insideMinus == insidePlus) {
    result.valid = 0;
    result.failure = SideOwnerNotDistinct;
    classifications[region] = result;
    return;
  }

  // orientation determined by owner solid
  result.flipNormal = insidePlus ? 1u : 0u;
  const double3 inside = insideMinus ? minus : plus;
  const double3 outside = insideMinus ? plus : minus;

  uint32_t adjacent = InvalidId;
  for (uint32_t model = 0; model < modelCount; ++model) {
    if (model == probe.ownerModelIndex ||
        (!pointInsideModelBounds(models[model], inside) &&
         !pointInsideModelBounds(models[model], outside))) {
      continue;
    }
    const double insideWinding = modelWinding(models[model], inside);
    const double outsideWinding = modelWinding(models[model], outside);
    if (!isfinite(insideWinding) || !isfinite(outsideWinding)) {
      result.valid = 0;
      result.failure = SideNonFiniteWinding;
      result.offendingModel = model;
      classifications[region] = result;
      return;
    }
    if (fabs(insideWinding) > 0.5) {
      result.valid = 0;
      result.failure = SideOwnerOverlap;
      result.offendingModel = model;
      classifications[region] = result;
      return;
    }
    if (fabs(outsideWinding) <= 0.5) continue;
    if (adjacent != InvalidId) {
      result.valid = 0;
      result.failure = SideMultipleAdjacent;
      result.offendingModel = model;
      classifications[region] = result;
      return;
    }
    adjacent = model;
  }
  result.valid = 1;
  result.adjacentModelIndex = adjacent;
  classifications[region] = result;
}

const char *sideFailureName(uint32_t failure) {
  switch (failure) {
  case SideSuccess: return "success";
  case SideInvalidProbe: return "invalidProbe";
  case SideNonFiniteWinding: return "nonFiniteWinding";
  case SideOwnerNotDistinct: return "ownerBoundingNotDistinct";
  case SideOwnerOverlap: return "ownerInsideOverlapped";
  case SideMultipleAdjacent: return "multipleAdjacentMaterials";
  default: return "unknown";
  }
}

bool appendSurfaceRegion(
    SurfaceRegion &piece,
    float domainScale,
    double coordinateMagnitude,
    double intersectionTolerance,
    std::vector<MaterialProbe> &probes,
    std::vector<SurfaceRegion> &regions) {
  const glm::dvec3 center = piece.moment / piece.area;
  const double patchScale = std::sqrt(std::max(piece.area, 0.0));
  const double positionResolution = 16.0 * FLT_EPSILON *
      std::max(double(domainScale), coordinateMagnitude);
  double probeDistance = std::max(intersectionTolerance, positionResolution);
  probeDistance = std::max(probeDistance, patchScale * 1e-3);
  probeDistance = std::min(probeDistance, patchScale * 0.125);
  if (!(probeDistance > 0.0) || !std::isfinite(probeDistance)) {
    std::cerr << "[RVDSurfacePartition] patch has no usable probe scale "
              << "cell=" << piece.cellId << " modelId="
              << piece.runtimeModelId << " area=" << piece.area
              << std::endl;
    return false;
  }
  MaterialProbe probe{};
  probe.centroid = make_double3(center.x, center.y, center.z);
  probe.normal = make_double3(piece.normal.x, piece.normal.y, piece.normal.z);
  probe.probeDistance = probeDistance;
  probe.ownerModelIndex = piece.modelIndex;
  probes.push_back(probe);
  regions.push_back(std::move(piece));
  return true;
}

} // namespace

bool buildSurfacePartition(
    const std::vector<ModelRestriction> &models,
    const std::vector<cudaGeometry::DeviceModelGeometryView> &geometries,
    const BaseCellDomain &baseDomain,
    std::vector<SurfaceRegion> &regions,
    cudaStream_t stream) {
  regions.clear();
  if (models.empty()) return true;
  if (geometries.size() != models.size()) return false;

  const double3 domainMin = make_double3(
      baseDomain.domainMin.x, baseDomain.domainMin.y, baseDomain.domainMin.z);
  const double invBaseScale = double(baseDomain.invBaseScale);
  std::vector<HostSurfacePatch> patches;
  for (uint32_t modelIndex = 0; modelIndex < models.size(); ++modelIndex) {
    const ModelRestriction &model = models[modelIndex];
    std::vector<RestrictedSurface> surfaces(model.totalPatches);
    std::vector<RestrictedSurfaceGeometry> geometry(model.totalPatches);
    std::vector<double3> vertices(model.devRestrictedVertices.size());
    std::vector<double4> cellMoments(baseDomain.count);
    if (!model.devSurfaces.download(surfaces.data(), surfaces.size()) ||
        !model.devSurfaceGeometry.download(geometry.data(), geometry.size()) ||
        !model.devMoments.download(cellMoments.data(), cellMoments.size()) ||
        (!vertices.empty() &&
         !model.devRestrictedVertices.download(vertices.data(), vertices.size()))) {
      return false;
    }

    for (uint32_t patchIndex = 0; patchIndex < model.totalPatches; ++patchIndex) {
      const RestrictedSurface &surface = surfaces[patchIndex];
      if (surface.cellId >= baseDomain.count || surface.vertexCount < 3 ||
          surface.firstVertex > vertices.size() ||
          surface.vertexCount > vertices.size() - surface.firstVertex) {
        std::cerr << "[RVDSurfacePartition] invalid restricted patch modelId="
                  << model.runtimeModelId << " patch=" << patchIndex
                  << " cell=" << surface.cellId << std::endl;
        return false;
      }
      const double phaseVolume = cellMoments[surface.cellId].w;
      if (phaseVolume == 0.0) continue;
      if (!(phaseVolume > 0.0) || !std::isfinite(phaseVolume)) {
        std::cerr << "[RVDSurfacePartition] invalid owning phase volume "
                  << "modelId=" << model.runtimeModelId
                  << " patch=" << patchIndex
                  << " cell=" << surface.cellId
                  << " volume=" << phaseVolume << std::endl;
        return false;
      }

      const RestrictedSurfaceGeometry &measure = geometry[patchIndex];
      SurfaceRegion region;
      region.cellId = surface.cellId;
      region.modelIndex = modelIndex;
      region.runtimeModelId = model.runtimeModelId;
      region.sourceTriangleId = surface.triangleId;
      region.normal = glm::dvec3(
          measure.normal.x, measure.normal.y, measure.normal.z);
      const double normalLength = glm::length(region.normal);
      if (!(normalLength > 0.0) || !std::isfinite(normalLength)) return false;
      region.normal /= normalLength;
      region.polygon.reserve(surface.vertexCount);
      for (uint32_t vertex = 0; vertex < surface.vertexCount; ++vertex) {
        const double3 &canonical = vertices[surface.firstVertex + vertex];
        region.polygon.push_back(
            glm::dvec3(domainMin.x, domainMin.y, domainMin.z) +
            glm::dvec3(canonical.x, canonical.y, canonical.z) * invBaseScale);
      }

      SurfaceFrame frame;
      if (!makeSurfaceFrame(region, frame)) return false;
      double area = 0.0;
      glm::dvec2 center2;
      if (!measurePolygon(frame.polygon, area, center2) || !(area > 0.0)) {
        std::cerr << "[RVDSurfacePartition] unresolved positive-area patch "
                  << "modelId=" << model.runtimeModelId
                  << " patch=" << patchIndex << std::endl;
        return false;
      }
      region.area = area;
      region.moment = (frame.origin + frame.u * center2.x +
                       frame.v * center2.y) * area;
      patches.push_back({std::move(region), {}});
    }
  }

  float domainScale = glm::length(baseDomain.domainMax - baseDomain.domainMin);
  if (!(domainScale > 0.0f) || !std::isfinite(domainScale)) domainScale = 1.0f;
  const double coordinateMagnitude = std::max(
      std::max(std::abs(double(baseDomain.domainMin.x)),
               std::abs(double(baseDomain.domainMax.x))),
      std::max(std::max(std::abs(double(baseDomain.domainMin.y)),
                        std::abs(double(baseDomain.domainMax.y))),
               std::max(std::abs(double(baseDomain.domainMin.z)),
                        std::abs(double(baseDomain.domainMax.z)))));
  const double intersectionTolerance = 512.0 * DBL_EPSILON *
      std::max(double(domainScale), coordinateMagnitude);
  std::vector<std::vector<uint32_t>> regionsByCell(baseDomain.count);
  std::vector<SurfaceFrame> frames(patches.size());
  for (uint32_t i = 0; i < patches.size(); ++i) {
    if (!makeSurfaceFrame(patches[i].region, frames[i])) {
      std::cerr << "[RVDSurfacePartition] invalid restricted patch cell="
                << patches[i].region.cellId << " modelId="
                << patches[i].region.runtimeModelId << " patch=" << i
                << std::endl;
      return false;
    }
    regionsByCell[patches[i].region.cellId].push_back(i);
  }
  for (uint32_t cell = 0; cell < regionsByCell.size(); ++cell) {
    const std::vector<uint32_t> &cellRegions = regionsByCell[cell];
    for (size_t a = 0; a < cellRegions.size(); ++a) {
      const uint32_t regionA = cellRegions[a];
      for (size_t b = a + 1; b < cellRegions.size(); ++b) {
      const uint32_t regionB = cellRegions[b];
        if (patches[regionA].region.modelIndex ==
            patches[regionB].region.modelIndex) continue;
        bool coplanar = false;
        if (!surfacesIntersectInArea(
                patches[regionA].region, frames[regionA],
                patches[regionB].region, frames[regionB],
                intersectionTolerance, coplanar)) {
          continue;
        }
        if (coplanar) {
          std::cerr << "[RVDSurfacePartition] ambiguous coplanar overlap cell="
                    << cell << " modelA="
                    << patches[regionA].region.runtimeModelId
                    << " modelB=" << patches[regionB].region.runtimeModelId
                    << std::endl;
          return false;
        }
        patches[regionA].cuts.push_back({
            patches[regionB].region.polygon[0],
            patches[regionB].region.normal});
        patches[regionB].cuts.push_back({
            patches[regionA].region.polygon[0],
            patches[regionA].region.normal});
      }
    }
  }

  regions.clear();
  std::vector<MaterialProbe> probes;
  for (uint32_t patchIndex = 0; patchIndex < patches.size(); ++patchIndex) {
    HostSurfacePatch &patch = patches[patchIndex];
    if (patch.cuts.empty()) {
      if (!appendSurfaceRegion(
              patch.region, domainScale, coordinateMagnitude,
              intersectionTolerance, probes, regions)) {
        return false;
      }
      continue;
    }
    std::vector<SurfaceRegion> pieces;
    if (!splitPatchAtCuts(patch.region, frames[patchIndex], patch.cuts, pieces)) {
      std::cerr << "[RVDSurfacePartition] area was not conserved while "
                   "splitting modelId=" << patch.region.runtimeModelId
                << " cell=" << patch.region.cellId << std::endl;
      return false;
    }
    for (SurfaceRegion &piece : pieces) {
      if (!appendSurfaceRegion(
              piece, domainScale, coordinateMagnitude,
              intersectionTolerance, probes, regions)) {
        return false;
      }
    }
  }

  cudaUtils::CudaBuffer<cudaGeometry::DeviceModelGeometryView> deviceModels;
  cudaUtils::CudaBuffer<MaterialProbe> deviceProbes;
  cudaUtils::CudaBuffer<PatchClassification> deviceClassifications;
  if (!deviceModels.allocate(geometries.size()) ||
      !deviceModels.uploadAsync(geometries.data(), geometries.size(), stream) ||
      !deviceProbes.allocate(probes.size()) ||
      !deviceProbes.uploadAsync(probes.data(), probes.size(), stream) ||
      !deviceClassifications.allocate(probes.size())) {
    return false;
  }
  classifyMaterialRegionsKernel<<<
      rvdHost::blockCount(uint32_t(probes.size()), BlockSize),
      BlockSize, 0, stream>>>(
      deviceProbes.get(), uint32_t(probes.size()),
      deviceModels.get(), uint32_t(geometries.size()),
      deviceClassifications.get());
  if (!rvdHost::checkLaunch("launch classifyMaterialRegionsKernel")) return false;

  std::vector<PatchClassification> classifications(probes.size());
  if (!deviceClassifications.downloadAsync(
          classifications.data(), classifications.size(), stream) ||
      !rvdHost::synchronize(stream, "synchronize material side classification")) {
    return false;
  }
  for (uint32_t region = 0; region < regions.size(); ++region) {
    const PatchClassification &classification = classifications[region];
    const SurfaceRegion &surface = regions[region];
    if (classification.valid == 0) {
      const glm::dvec3 center = regions[region].moment / regions[region].area;
      std::cerr << "[RVDSurfacePartition] unresolved side ownership cell="
                << regions[region].cellId << " modelId="
                << regions[region].runtimeModelId << " region=" << region
                << " reason=" << sideFailureName(classification.failure)
                << " probeDistance=" << probes[region].probeDistance
                << " ownerWindings=(" << classification.windingMinus << ", "
                << classification.windingPlus << ")"
                << " centroid=(" << center.x << ", " << center.y << ", "
                << center.z << ")"
                << " offendingModel=" << classification.offendingModel
                << std::endl;
      return false;
    }
    if (classification.flipNormal != 0) {
      std::reverse(
          regions[region].polygon.begin(), regions[region].polygon.end());
      regions[region].normal = -regions[region].normal;
    }
    const uint32_t adjacent = classification.adjacentModelIndex;
    regions[region].outsideMaterialId = adjacent == InvalidId
        ? InvalidId
        : models[adjacent].runtimeModelId;
  }

  return true;
}

} // namespace voronoi
