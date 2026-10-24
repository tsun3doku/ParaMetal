#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "KNNNeighborsKernels.cuh"

namespace voronoi {

inline constexpr uint32_t InvalidSeed = 0xFFFFFFFFu;
inline constexpr uint32_t SolidSeed   = 0xFFFFFFFEu;

bool buildSeedIdGrid(
    const DeviceKnn& knn,
    const uint8_t* devSolidMask,
    const std::vector<glm::vec4>& seeds,
    glm::vec3 gridMin,
    glm::ivec3 gridDim,
    float cellSize,
    uint32_t* outSeedIds,
    cudaStream_t stream);

} // namespace voronoi
