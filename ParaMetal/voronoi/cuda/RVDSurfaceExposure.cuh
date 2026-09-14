#pragma once

#include "RVD.hpp"
#include "RVDRestriction.cuh"
#include "RVDBaseCells.cuh"
#include "RVDGeometry.cuh"

#include <vector>
#include <cuda_runtime.h>
#include "../../cuda/CudaBuffer.cuh"

namespace voronoi {

bool computeSurfaceExposure(
    const std::vector<ModelRestrictionData>& modelRestrictions,
    const cudaUtils::CudaBuffer<cudaGeometry::DeviceModelGeometryView>& devGeometries,
    const BaseCellsContext& baseCtx,
    float contactRange,
    float contactResolution,
    std::vector<SurfaceMeasure>& outExposure,
    cudaStream_t stream);

} // namespace voronoi
