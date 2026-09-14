#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace voronoi {
struct ModelGeometry;
}

namespace voronoi::globalsdf {

struct BuildInput {
    const std::vector<ModelGeometry>* geometries = nullptr;
    glm::vec3 gridMin{};
    glm::ivec3 gridDim{};
    float cellSize = 0.0f;
    float* outDevUnionSdf = nullptr;
};

struct BuildResult {
    std::vector<float> values;
    uint32_t channelStride = 0;
    uint32_t channelCount = 0;
};

bool build(const BuildInput& input, BuildResult& result, cudaStream_t stream);

} // namespace voronoi::globalsdf