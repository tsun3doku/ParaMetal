#include "VoronoiCellField.cuh"
#include "../../cuda/CudaCheck.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

namespace voronoi {

constexpr auto cudaOk = cudaUtils::cudaCheck;
constexpr uint32_t BlockSize = 256;

__global__ void buildSeedIdGridKernel(
    uint32_t* outSeedIds,
    const uint8_t* devSolidMask,
    DeviceKnn knn,
    float3 gridMin,
    int3 gridDim,
    float cellSize,
    uint32_t totalVoxels)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= totalVoxels) return;

    if (devSolidMask != nullptr && devSolidMask[index] != 0u) {
        outSeedIds[index] = SolidSeed;
        return;
    }

    const int x = int(index % uint32_t(gridDim.x));
    const int y = int((index / uint32_t(gridDim.x)) % uint32_t(gridDim.y));
    const int z = int(index / uint32_t(gridDim.x * gridDim.y));
    const float3 point = make_float3(
        gridMin.x + x * cellSize,
        gridMin.y + y * cellSize,
        gridMin.z + z * cellSize);

    const uint32_t seed = nearestSeed(point, knn);
    outSeedIds[index] = (seed != InvalidNeighborId) ? seed : InvalidSeed;
}

bool buildSeedIdGrid(
    const DeviceKnn& knn,
    const uint8_t* devSolidMask,
    const std::vector<glm::vec4>& seeds,
    glm::vec3 gridMin,
    glm::ivec3 gridDim,
    float cellSize,
    uint32_t* outSeedIds,
    cudaStream_t stream)
{
    if (seeds.empty()) {
        std::cerr << "[VoronoiCellField] Seeds are empty." << std::endl;
        return false;
    }

    // Validate that all seed weights equal each other within tolerance
    const float w0 = seeds[0].w;
    for (size_t i = 1; i < seeds.size(); ++i) {
        if (std::abs(seeds[i].w - w0) > 1e-5f) {
            std::cerr << "[VoronoiCellField] Seeds have non-uniform weights; cannot reconstruct Euclidean Voronoi field." << std::endl;
            return false;
        }
    }

    if (gridDim.x < 2 || gridDim.y < 2 || gridDim.z < 2 || !(cellSize > 0.0f) || !std::isfinite(cellSize)) {
        std::cerr << "[VoronoiCellField] Invalid grid dimensions or cell size." << std::endl;
        return false;
    }

    const size_t totalVoxels64 = size_t(gridDim.x) * gridDim.y * gridDim.z;
    if (totalVoxels64 == 0 || totalVoxels64 > UINT32_MAX || outSeedIds == nullptr) {
        std::cerr << "[VoronoiCellField] Invalid seed ID grid output." << std::endl;
        return false;
    }

    const uint32_t totalVoxels = static_cast<uint32_t>(totalVoxels64);
    const uint32_t gridBlocks = (totalVoxels + BlockSize - 1) / BlockSize;

    const float3 devGridMin = make_float3(gridMin.x, gridMin.y, gridMin.z);
    const int3 devGridDim = make_int3(gridDim.x, gridDim.y, gridDim.z);

    buildSeedIdGridKernel<<<gridBlocks, BlockSize, 0, stream>>>(
        outSeedIds,
        devSolidMask,
        knn,
        devGridMin,
        devGridDim,
        cellSize,
        totalVoxels);

    if (!cudaOk(cudaGetLastError(), "launch buildSeedIdGridKernel")) {
        return false;
    }

    return cudaOk(cudaStreamSynchronize(stream), "synchronize buildSeedIdGrid");
}

} // namespace voronoi
