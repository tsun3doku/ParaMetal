#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDModelGeometry.cuh"
#include "RVDRestriction.cuh"
#include "RVDBaseFaceTopology.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace voronoi {

enum FacePartitionError : uint32_t {
  FacePartitionSuccess = 0,
  InvalidFacePolygon = 1,
  TooManyRegions = 2,
  AmbiguousOwnership = 3,
  OverlappingModels = 4,
  AreaConservation = 5,
  MomentConservation = 6,
  PassMismatch = 7,
  UnresolvedRegion = 8,
  TooManySegments = 9,
  EmptyFacePartition = 10,
  MaterialInconsistency = 11
};

// partitions reciprocal Voronoi base faces into material regions
bool buildFacePartitionsGPU(
    const BaseCellDomain &baseDomain,
    BaseFaceTopology &faceTopology,
    const std::vector<cudaGeometry::DeviceModelGeometryView> &modelViews,
    const std::vector<ModelRestriction> &models,
    cudaStream_t stream = nullptr);

} // namespace voronoi
