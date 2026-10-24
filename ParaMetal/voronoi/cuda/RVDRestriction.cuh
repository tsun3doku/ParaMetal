#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDTopology.hpp"
#include "RVDGeometry.cuh"
#include "RVDStatus.cuh"

#include "../../cuda/CudaBuffer.cuh"

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace voronoi {

struct RestrictedSurfaceGeometry {
  double4 centroidArea; // xyz = canonical centroid, w = canonical area
  double3 normal;       // unit triangle normal
};

struct FaceBoundarySegment {
  double3 a;
  double3 b;
};

enum class RVDRestrictionError : uint32_t {
  None = 0,
  NonPositiveMoment = 101,
  PlaneAreaOutOfRange = 102,
};

inline const char *rvdRestrictionErrorName(uint32_t code) {
  switch (RVDRestrictionError(code)) {
  case RVDRestrictionError::None: return "None";
  case RVDRestrictionError::NonPositiveMoment: return "NonPositiveMoment";
  case RVDRestrictionError::PlaneAreaOutOfRange: return "PlaneAreaOutOfRange";
  default: return "Unknown";
  }
}

struct ModelRestriction {
  uint32_t runtimeModelId = 0;

  cudaUtils::CudaBuffer<double4> devMoments;
  cudaUtils::CudaBuffer<double> devRestrictedPlaneAreas;
  cudaUtils::CudaBuffer<double> devRestrictedPlaneAbsSums;
  cudaUtils::CudaBuffer<double3> devRestrictedPlaneMoments;
  cudaUtils::CudaBuffer<RestrictedSurfaceGeometry> devSurfaceGeometry;
  cudaUtils::CudaBuffer<RestrictedSurface> devSurfaces;
  cudaUtils::CudaBuffer<double3> devRestrictedVertices;
  cudaUtils::CudaBuffer<uint32_t> devFaceSegmentOffsets;
  cudaUtils::CudaBuffer<FaceBoundarySegment> devFaceSegments;
  uint32_t totalPatches = 0;
  uint32_t surfaceBase = 0;
};

bool restrictModelGeometry(
    const RVDInput &input,
    const RVDConfig &config,
    const cudaGeometry::DeviceModelGeometry &devGeom,
    const BaseCellDomain &baseDomain,
    ModelRestriction &restriction,
    cudaStream_t stream);

} // namespace voronoi
