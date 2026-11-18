#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDBaseFaceTopology.hpp"
#include "RVDModelGeometry.cuh"
#include "RVDStatus.cuh"

#include "../../cuda/CudaBuffer.cuh"

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace voronoi {

enum class FaceCutSegmentType : uint32_t {
  Crossing = 0,
  Coplanar = 1,
};

struct FaceCutVertexKey {
  uint32_t triangleId = InvalidId;
  uint32_t cadVertexId = InvalidId;
  uint32_t cadEdgeVertexA = InvalidId;
  uint32_t cadEdgeVertexB = InvalidId;
  uint32_t basePlaneA = InvalidId;
  uint32_t basePlaneB = InvalidId;
  uint32_t basePlaneC = InvalidId;
};

struct FaceBoundarySegment {
  double3 a;
  double3 b;
  double3 planeNormal;
  uint32_t runtimeModelId = InvalidId;
  uint32_t segmentId = InvalidId;
  uint32_t canonicalA = InvalidId;
  uint32_t canonicalB = InvalidId;
  uint32_t segmentType = uint32_t(FaceCutSegmentType::Crossing);
  uint32_t triangleId = InvalidId;
  FaceCutVertexKey keyA;
  FaceCutVertexKey keyB;
};

enum class RVDRestrictionError : uint32_t {
  None = 0,
  PlaneAreaOutOfRange = 102,
  InvalidFaceCutCanonicalization = 103,
  MissingCellOrigin = 104,
  BaseCellBoundary = 105,
  TriangleTopology = 106,
  TriangleClipMeasures = 107,
  WindingOnSurface = 108,
};

inline const char *rvdRestrictionErrorName(uint32_t code) {
  switch (RVDRestrictionError(code)) {
  case RVDRestrictionError::None: return "None";
  case RVDRestrictionError::PlaneAreaOutOfRange: return "PlaneAreaOutOfRange";
  case RVDRestrictionError::InvalidFaceCutCanonicalization: return "InvalidFaceCutCanonicalization";
  case RVDRestrictionError::MissingCellOrigin: return "MissingCellOrigin";
  case RVDRestrictionError::BaseCellBoundary: return "BaseCellBoundary";
  case RVDRestrictionError::TriangleTopology: return "TriangleTopology";
  case RVDRestrictionError::TriangleClipMeasures: return "TriangleClipMeasures";
  case RVDRestrictionError::WindingOnSurface: return "WindingOnSurface";
  default: return "Unknown";
  }
}

struct ModelRestriction {
  uint32_t runtimeModelId = 0;

  cudaUtils::CudaBuffer<double4> devMoments;
  cudaUtils::CudaBuffer<double> devRestrictedPlaneAreas;
  cudaUtils::CudaBuffer<double3> devSurfaceNormals;
  cudaUtils::CudaBuffer<RestrictedSurface> devSurfaces;
  cudaUtils::CudaBuffer<double3> devRestrictedVertices;
  cudaUtils::CudaBuffer<uint32_t> devFaceSegmentOffsets;
  cudaUtils::CudaBuffer<FaceBoundarySegment> devFaceSegments;
  uint32_t totalPatches = 0;
};

bool restrictModelGeometry(
    const RVDInput &input,
    const RVDConfig &config,
    const cudaGeometry::DeviceModelGeometry &devGeom,
    const BaseCellDomain &baseDomain,
    ModelRestriction &restriction,
    cudaStream_t stream);

} // namespace voronoi
