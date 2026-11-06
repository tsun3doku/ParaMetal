#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDSurfacePartition.hpp"

#include <cstdint>
#include <vector>

namespace voronoi {

// emit thermal couplings between opposing CAD regions
bool buildContactCouplings(
    const std::vector<SurfaceRegion> &regions,
    const RVDResult &rvd,
    const BaseCellDomain &baseDomain,
    double contactRange,
    std::vector<ContactCoupling> &couplings,
    cudaStream_t stream = nullptr);

} // namespace voronoi
