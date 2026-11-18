#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDSurfacePartition.hpp"

#include <cstdint>
#include <vector>

namespace voronoi {

bool buildContactCouplings(
    const std::vector<SurfaceRegion> &regions,
    const RVDResult &rvd,
    const BaseCellDomain &baseDomain,
    double contactRange,
    std::vector<ContactCoupling> &couplings,
    cudaStream_t stream = nullptr);

} // namespace voronoi
