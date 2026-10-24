#pragma once

#include "RVD.hpp"

#include <cstdint>

namespace voronoi {

// Host-side geometric audit of the final fragment/face output. This is
// deliberately separate from the GPU pipeline: it is diagnostic-only and is
// the last gate before results are handed back to the caller.
bool validateFragmentTopology(
    const RVDResult &result,
    uint32_t totalBaseFragmentFaces,
    double toleranceClosure = 0.005,
    double toleranceGauss = 0.01);

} // namespace voronoi
