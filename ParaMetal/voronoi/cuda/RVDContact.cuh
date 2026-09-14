#pragma once

#include "RVD.hpp"
#include "RVDRestriction.cuh"
#include "RVDBaseCells.cuh"
#include "RVDGeometry.cuh"
#include "KNNNeighborsKernels.cuh"

#include <vector>
#include <cuda_runtime.h>

namespace voronoi {

bool buildContactFaces(
    const std::vector<ModelRestrictionData>& models,
    const cudaUtils::CudaBuffer<cudaGeometry::DeviceModelGeometryView>& devGeometries,
    const std::vector<cudaGeometry::DeviceModelGeometryView>& hostGeometries,
    const DeviceKnn& seeds,
    const BaseCellsContext& baseCtx,
    float contactRange,
    float contactResolution,
    const std::vector<std::vector<uint32_t>>& fragmentIdByModelCell,
    const std::vector<Fragment>& fragments,
    RVDResult& result,
    cudaStream_t stream);

} // namespace voronoi
