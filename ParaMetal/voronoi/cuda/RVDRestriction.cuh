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

struct ModelCellData {
    double volume = 0.0;
    glm::dvec3 localMoment{0.0};
    float surfaceArea = 0.0f;
};

struct RestrictedSurfaceGeometry {
    float4 centroidArea; // xyz = canonical centroid, w = canonical area
    float4 plane;        // xyz = unit triangle normal, w = dot(n,p)
};

struct ModelRestrictionData {
    uint32_t runtimeModelId = 0;
    std::vector<ModelCellData> cells;
    std::vector<double> faceAreas;
    std::vector<RestrictedSurfaceGeometry> surfaceGeometry;

    cudaUtils::CudaBuffer<RestrictedSurface> devSurfaces;
    cudaUtils::CudaBuffer<float3> devRestrictedVertices;
    uint32_t totalPatches = 0;
    uint32_t surfaceBase = 0;
};

bool restrictModelGeometry(
    const RVDInput& input,
    const RVDConfig& config,
    const cudaGeometry::DeviceModelGeometry& devGeom,
    const BaseCellsContext& baseCtx,
    const TopologyContext& topoCtx,
    ModelRestrictionData& model,
    RVDResult& result,
    cudaStream_t stream);

} // namespace voronoi
