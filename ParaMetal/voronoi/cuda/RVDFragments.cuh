#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDBaseFaceTopology.hpp"
#include "RVDRestriction.cuh"
#include "RVDSurfacePartition.hpp"

#include <cstdint>
#include <vector>

namespace voronoi {

bool verifySolidAreas(
    const BaseFaceTopology &faceTopology,
    const BaseCellDomain &baseDomain,
    const std::vector<ModelRestriction> &models,
    cudaStream_t stream = cudaStream_t(0));

bool buildFragmentsAndFaces(
    const RVDInput &input,
    const BaseCellDomain &baseDomain,
    const BaseFaceTopology &faceTopology,
    const std::vector<ModelRestriction> &models,
    std::vector<SurfaceRegion> &surfaceRegions,
    uint32_t &totalBaseFragmentFaces,
    RVDResult &result,
    cudaStream_t stream = cudaStream_t(0));

} // namespace voronoi
