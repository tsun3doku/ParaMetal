#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDBaseFaceTopology.hpp"
#include "RVDRestriction.cuh"
#include "RVDSurfacePartition.hpp"

#include <cstdint>
#include <vector>

namespace voronoi {

// build fragments and physical faces from restricted phase measures
bool buildFragmentsAndFaces(
    const RVDInput &input,
    const BaseCellDomain &baseDomain,
    const BaseFaceTopology &faceTopology,
    const std::vector<ModelRestriction> &models,
    std::vector<SurfaceRegion> &surfaceRegions,
    uint32_t &totalBaseFragmentFaces,
    RVDResult &result);

} // namespace voronoi
