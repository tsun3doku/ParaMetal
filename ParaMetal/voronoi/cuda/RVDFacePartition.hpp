#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDGeometry.cuh"
#include "RVDRestriction.cuh"
#include "RVDTopology.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace voronoi {

enum FacePartitionError : uint32_t {
  FacePartitionSuccess = 0,
  InvalidInput = 1,
  InvalidFacePolygon = 2,
  TooManyRegions = 3,
  AmbiguousOwnership = 4,
  OverlappingModels = 5,
  AreaConservation = 6,
  MomentConservation = 7,
  PassMismatch = 8,
  CoverageAreaConservation = 9,
  BaseAreaConservation = 10,
  BaseMomentConservation = 11,
  UnresolvedRegion = 12,
  UnresolvedCoverageArea = 13,
  UnresolvedCoverageMoment = 14,
  TooManySegments = 15,
  EmptyFacePartition = 16
};

bool buildFacePartitionsGPU(
    const BaseCellDomain &baseDomain,
    BaseFaceTopology &faceTopology,
    const std::vector<cudaGeometry::DeviceModelGeometryView> &modelViews,
    const std::vector<ModelRestriction> &models,
    cudaStream_t stream);

} // namespace voronoi
