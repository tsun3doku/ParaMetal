#include "RVDDiagnostics.hpp"

#include <vector>

#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <utility>

namespace voronoi {

namespace {

constexpr size_t NearestSeedCount = 64;

struct NeighborSeed {
  double distanceSquared = 0.0;
  size_t index = 0;
};

struct BisectorPlane {
  glm::dvec3 normal{0.0};
  double offset = 0.0;
};

struct DomainBounds {
  glm::vec3 minimum{0.0f};
  glm::vec3 maximum{0.0f};
};

DomainBounds computeDomainBounds(const std::array<glm::vec3, 8> &corners) {
  DomainBounds bounds;
  bounds.minimum = corners[0];
  bounds.maximum = corners[0];
  for (const glm::vec3 &corner : corners) {
    bounds.minimum = glm::min(bounds.minimum, corner);
    bounds.maximum = glm::max(bounds.maximum, corner);
  }
  return bounds;
}

std::vector<NeighborSeed> findNearestSeeds(
    const std::vector<glm::vec4> &seeds,
    size_t cell) {
  const glm::vec4 origin = seeds[cell];
  std::vector<NeighborSeed> nearest;
  nearest.reserve(NearestSeedCount);

  for (size_t other = 0; other < seeds.size(); ++other) {
    if (other == cell) continue;

    const glm::vec4 point = seeds[other];
    const glm::dvec3 delta(
        point.x - origin.x, point.y - origin.y, point.z - origin.z);
    const double distanceSquared = glm::dot(delta, delta);

    if (nearest.size() < NearestSeedCount) {
      nearest.push_back({distanceSquared, other});
      if (nearest.size() == NearestSeedCount) {
        std::sort(nearest.begin(), nearest.end(),
                  [](const NeighborSeed &a, const NeighborSeed &b) {
                    return a.distanceSquared < b.distanceSquared;
                  });
      }
    } else if (distanceSquared < nearest.back().distanceSquared) {
      nearest.back() = {distanceSquared, other};
      std::sort(nearest.begin(), nearest.end(),
                [](const NeighborSeed &a, const NeighborSeed &b) {
                  return a.distanceSquared < b.distanceSquared;
                });
    }
  }
  return nearest;
}

double minimumNeighborPairDistance(
    const std::vector<glm::vec4> &seeds,
    const std::vector<NeighborSeed> &nearest) {
  double minimum = std::numeric_limits<double>::max();
  for (size_t i = 0; i < nearest.size(); ++i) {
    for (size_t j = i + 1; j < nearest.size(); ++j) {
      const glm::vec4 a = seeds[nearest[i].index];
      const glm::vec4 b = seeds[nearest[j].index];
      const glm::dvec3 delta(a.x - b.x, a.y - b.y, a.z - b.z);
      minimum = std::min(minimum, glm::length(delta));
    }
  }
  return minimum;
}

std::vector<BisectorPlane> buildBisectorPlanes(
    const std::vector<glm::vec4> &seeds,
    size_t cell,
    const std::vector<NeighborSeed> &nearest) {
  const glm::vec4 origin = seeds[cell];
  const glm::dvec3 seed(origin.x, origin.y, origin.z);

  std::vector<BisectorPlane> planes;
  planes.reserve(nearest.size());
  for (const NeighborSeed &neighbor : nearest) {
    const glm::vec4 point4 = seeds[neighbor.index];
    const glm::dvec3 point(point4.x, point4.y, point4.z);
    const glm::dvec3 direction = point - seed;
    const double length = glm::length(direction);
    if (!(length > 0.0)) continue;

    const glm::dvec3 normal = direction / length;
    const glm::dvec3 midpoint = (seed + point) * 0.5;
    planes.push_back({normal, glm::dot(normal, midpoint)});
  }
  return planes;
}

void widestPlanePair(
    const std::vector<BisectorPlane> &planes,
    double &maxPlaneDot,
    double &offsetAtMaxDot) {
  maxPlaneDot = 0.0;
  offsetAtMaxDot = 0.0;

  for (size_t i = 0; i < planes.size(); ++i) {
    for (size_t j = i + 1; j < planes.size(); ++j) {
      const double dot = std::fabs(glm::dot(planes[i].normal, planes[j].normal));
      if (dot > maxPlaneDot) {
        maxPlaneDot = dot;
        offsetAtMaxDot = std::fabs(planes[i].offset - planes[j].offset);
      }
    }
  }
}

glm::dvec3 boundaryGap(
    const glm::dvec3 &seed,
    const DomainBounds &bounds) {
  return glm::dvec3(
      std::min(seed.x - double(bounds.minimum.x), double(bounds.maximum.x) - seed.x),
      std::min(seed.y - double(bounds.minimum.y), double(bounds.maximum.y) - seed.y),
      std::min(seed.z - double(bounds.minimum.z), double(bounds.maximum.z) - seed.z));
}

void reportCellFailure(
    const RVDInput &input,
    const std::vector<glm::vec4> &seeds,
    const DomainBounds &bounds,
    size_t cell,
    RVDStatus status) {
  const glm::vec4 seed4 = seeds[cell];
  const glm::dvec3 seed(seed4.x, seed4.y, seed4.z);

  const std::vector<NeighborSeed> nearest = findNearestSeeds(seeds, cell);
  const std::vector<BisectorPlane> planes = buildBisectorPlanes(seeds, cell, nearest);

  double maxPlaneDot = 0.0;
  double offsetAtMaxDot = 0.0;
  widestPlanePair(planes, maxPlaneDot, offsetAtMaxDot);

  const glm::dvec3 gap = boundaryGap(seed, bounds);
  const double nearestDistance =
      std::sqrt(nearest.empty() ? 0.0 : nearest.front().distanceSquared);

  std::cerr << "[RVD] base failure cell=" << cell
            << " status=" << rvdStatusName(status)
            << " nearestDist=" << nearestDistance
            << " minNeighborPairDist=" << minimumNeighborPairDistance(seeds, nearest)
            << " maxPlaneDot=" << maxPlaneDot
            << " offsetAtMaxDot=" << offsetAtMaxDot
            << " boundaryGap=(" << gap.x << "," << gap.y << "," << gap.z << ")"
            << " nominalCellSize=" << input.nominalCellSize << std::endl;
}

} // namespace

void reportBaseCellFailures(
    const RVDInput &input,
    const std::vector<RVDStatus> &statuses) {
  if (!input.seeds || input.seeds->empty() || !input.domainCorners) return;

  const std::vector<glm::vec4> &seeds = *input.seeds;
  const DomainBounds bounds = computeDomainBounds(*input.domainCorners);

  for (size_t cell = 0; cell < statuses.size(); ++cell) {
    if (statuses[cell] != RVDStatus::Success) {
      reportCellFailure(input, seeds, bounds, cell, statuses[cell]);
    }
  }
}

} // namespace voronoi
