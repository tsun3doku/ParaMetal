#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDSurfacePartition.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace voronoi {



bool agglomerateFragments(
    const BaseCellDomain &baseDomain,
    std::vector<SurfaceRegion> &surfaceRegions,
    uint32_t &totalBaseFragmentFaces,
    RVDResult &result,
    cudaStream_t stream = nullptr);

} // namespace voronoi
