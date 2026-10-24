#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDGeometry.cuh"
#include "RVDRestriction.cuh"
#include "RVDTopology.hpp"

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace voronoi {

enum FacePartitionError : uint32_t {
  FacePartitionSuccess = 0,
  InvalidInput = 1,
  InvalidFacePolygon = 2,
  InvalidModelBvh = 3,
  TooManyLines = 4,
  TooManyVertices = 5,
  TooManyRegions = 6,
  InvalidIntersection = 7,
  AmbiguousOwnership = 8,
  OverlappingModels = 9,
  AreaConservation = 10,
  MomentConservation = 11,
  PassMismatch = 12,
  ClipAreaConservation = 13,
  CoverageAreaConservation = 14,
  BaseAreaConservation = 15,
  BaseMomentConservation = 16,
  UnresolvedRegion = 17,
  UnresolvedCoverageArea = 18,
  UnresolvedCoverageMoment = 19,
  TooManySegments = 20,
  EmptyFacePartition = 21
};

bool buildFacePartitionsGPU(
    const BaseCellDomain &baseDomain, BaseFaceTopology &faceTopology,
    const std::vector<cudaGeometry::DeviceModelGeometryView> &modelViews,
    const std::vector<ModelRestriction> &models, cudaStream_t stream);

} // namespace voronoi
