#pragma once

#include "RVD.hpp"
#include "RVDStatus.cuh"

#include <vector>

namespace voronoi {

// Host-side diagnosis for base cells that failed the unrestricted pass.
// Kept separate from the pipeline so failure analysis cannot leak into the
// hot path.
void reportBaseCellFailures(
    const RVDInput &input,
    const std::vector<RVDStatus> &statuses);

} // namespace voronoi
