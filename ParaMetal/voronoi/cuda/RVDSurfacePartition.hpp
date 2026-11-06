#pragma once

#include "RVD.hpp"
#include "RVDRestriction.cuh"
#include "RVDBaseCells.cuh"
#include "RVDModelGeometry.cuh"
#include "../../cuda/CudaBuffer.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace voronoi {

// restricted CAD surface patch
struct SurfaceRegion {
  uint32_t cellId = 0;
  uint32_t modelIndex = 0;
  uint32_t runtimeModelId = 0;
  // outward material ID (InvalidId for fluid)
  uint32_t outsideMaterialId = 0xffffffffu;
  double area = 0.0;
  glm::dvec3 moment{0.0};
  glm::dvec3 normal{0.0, 0.0, 1.0};
  std::vector<glm::dvec3> polygon;
  uint32_t faceId = 0xffffffffu;
  uint32_t sourceTriangleId = 0xffffffffu;
};

// split transverse intersections into planar material regions
bool buildSurfacePartition(
    const std::vector<ModelRestriction> &models,
    const std::vector<cudaGeometry::DeviceModelGeometryView> &geometries,
    const BaseCellDomain &baseDomain,
    std::vector<SurfaceRegion> &regions,
    cudaStream_t stream = nullptr);

} // namespace voronoi
