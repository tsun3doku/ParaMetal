#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace voronoi {

bool updateLloydSeeds(
    const RVDInput &input,
    const RVDConfig &config,
    const BaseCellDomain &baseDomain,
    std::vector<glm::vec4> &seeds,
    uint32_t &movedCount,
    double &maxDisplacement,
    cudaStream_t stream);

} // namespace voronoi
