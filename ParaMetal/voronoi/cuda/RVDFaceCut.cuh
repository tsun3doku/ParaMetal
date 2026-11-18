#pragma once

#include "RVDRestriction.cuh"

#include <vector>

namespace voronoi {

bool canonicalizeFaceCutVertices(std::vector<ModelRestriction> &models, cudaStream_t stream);

} // namespace voronoi
