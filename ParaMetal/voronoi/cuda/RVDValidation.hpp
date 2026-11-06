#pragma once

#include "RVD.hpp"

#include <cstdint>

namespace voronoi {

// host-side geometric audit of final fragments and faces
bool validateFragmentTopology(
    const RVDResult &result,
    uint32_t totalBaseFragmentFaces,
    double toleranceClosure = 0.005,
    double toleranceGauss = 0.01);

// verify coupling area does not exceed physical area on either side
bool validateContactCouplings(
    const RVDResult &result,
    double relativeTolerance = 0.01);

} // namespace voronoi
