#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDTopology.hpp"
#include "RVDGeometry.cuh"

#include <vector>
#include <cstdint>
#include <glm/glm.hpp>

#include "../../cuda/CudaBuffer.cuh"

namespace voronoi {

struct RestrictedSurfaceGeometry {
    double4 centroidArea; // xyz = canonical centroid, w = canonical area
    double4 plane;        // xyz = unit triangle normal, w = dot(n,p)
};

struct FaceBoundarySegment {
    double3 a;
    double3 b;
    uint32_t modelId = 0;
};

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
    uint32_t totalFaceSegments = 0;
    uint32_t surfaceBase = 0;
};

bool restrictModelGeometry(
    const RVDInput& input,
    const RVDConfig& config,
    const cudaGeometry::DeviceModelGeometry& devGeom,
    const BaseCellDomain& baseDomain,
    ModelRestriction& restriction,
    cudaStream_t stream);

} // namespace voronoi
