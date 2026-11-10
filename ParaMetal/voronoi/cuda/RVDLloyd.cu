#include "RVDLloyd.cuh"

#include "../../cuda/CudaCheck.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace voronoi {

namespace {

constexpr uint32_t LloydBlockSize = 256;

__global__ void updateLloydSeedsKernel(
    const float4 *seeds,
    const double4 *baseMoments,
    uint32_t cellCount,
    double invBaseVolumeScale,
    double invBaseMomentScale,
    float damping,
    double minVolumeThreshold,
    float4 *outSeeds) {
  const uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cellCount) return;

  const float4 seed = seeds[cell];
  const glm::dvec3 seedPosition(seed.x, seed.y, seed.z);
  const double4 rawBaseMoment = baseMoments[cell];
  const double baseVolume = (rawBaseMoment.w / 6.0) * invBaseVolumeScale;
  const glm::dvec3 baseMoment = (glm::dvec3(rawBaseMoment.x, rawBaseMoment.y, rawBaseMoment.z) / 6.0) * invBaseMomentScale;
  if (!(baseVolume > minVolumeThreshold) || !isfinite(baseVolume)) {
    outSeeds[cell] = seed;
    return;
  }

  const glm::dvec3 centroid = seedPosition + baseMoment / baseVolume;
  if (!isfinite(centroid.x) || !isfinite(centroid.y) ||
      !isfinite(centroid.z)) {
    outSeeds[cell] = seed;
    return;
  }

  const glm::dvec3 moved = seedPosition + double(damping) * (centroid - seedPosition);
  outSeeds[cell] = make_float4(float(moved.x), float(moved.y), float(moved.z), 0.0f);
}

} // namespace

bool updateLloydSeeds(
    const RVDInput &input,
    const RVDConfig &config,
    const BaseCellDomain &baseDomain,
    std::vector<glm::vec4> &seeds,
    uint32_t &movedCount,
    double &maxDisplacement,
    cudaStream_t stream) {
  movedCount = 0;
  maxDisplacement = 0.0;
  if (input.seeds == nullptr || input.seedSpatialIndex == nullptr ||
      seeds.empty() || seeds.size() != baseDomain.count) {
    return false;
  }
  if (!(config.lloydDamping > 0.0f && config.lloydDamping <= 1.0f) ||
      !std::isfinite(config.lloydDamping) ||
      !(config.lloydVolumeThresholdFraction >= 0.0f) ||
      !std::isfinite(config.lloydVolumeThresholdFraction)) {
    std::cerr << "[RVD-Lloyd] invalid configuration" << std::endl;
    return false;
  }

  const uint32_t seedCount = static_cast<uint32_t>(seeds.size());
  const DeviceKnn knn = input.seedSpatialIndex->deviceKnn();
  if (knn.seeds == nullptr || knn.seedCount != seedCount) {
    std::cerr << "[RVD-Lloyd] seed index mismatch" << std::endl;
    return false;
  }

  const double nominalCellSize = double(
      baseDomain.nominalCellSize > 0.0f ? baseDomain.nominalCellSize
                                        : input.nominalCellSize);
  const double nominalVolume = nominalCellSize * nominalCellSize * nominalCellSize;
  const double minVolumeThreshold = double(config.lloydVolumeThresholdFraction) * nominalVolume;

  cudaUtils::CudaBuffer<float4> outSeeds;
  if (!outSeeds.allocate(seedCount)) return false;

  const uint32_t blocks = (seedCount + LloydBlockSize - 1) / LloydBlockSize;
  updateLloydSeedsKernel<<<blocks, LloydBlockSize, 0, stream>>>(
      knn.seeds,
      baseDomain.devBaseMoments.get(),
      seedCount,
      baseDomain.invBaseVolumeScale,
      baseDomain.invBaseMomentScale,
      config.lloydDamping,
      minVolumeThreshold,
      outSeeds.get());
  if (!cudaUtils::cudaCheck(
          cudaGetLastError(), "launch updateLloydSeedsKernel")) {
    return false;
  }

  std::vector<float4> hostSeeds(seedCount);
  if (!outSeeds.downloadAsync(hostSeeds.data(), seedCount, stream) ||
      !cudaUtils::cudaCheck(
          cudaStreamSynchronize(stream),
          "synchronize Lloyd seed update")) {
    return false;
  }

  double maxDisplacementSquared = 0.0;
  for (uint32_t i = 0; i < seedCount; ++i) {
    const float4 updated = hostSeeds[i];
    if (!std::isfinite(updated.x) || !std::isfinite(updated.y) ||
        !std::isfinite(updated.z)) {
      return false;
    }
    const double dx = double(updated.x) - double(seeds[i].x);
    const double dy = double(updated.y) - double(seeds[i].y);
    const double dz = double(updated.z) - double(seeds[i].z);
    const double displacementSquared = dx * dx + dy * dy + dz * dz;
    if (displacementSquared > 0.0) {
      ++movedCount;
    }
    maxDisplacementSquared =
        std::max(maxDisplacementSquared, displacementSquared);
    seeds[i] = glm::vec4(updated.x, updated.y, updated.z, 1.0f);
  }

  maxDisplacement = std::sqrt(maxDisplacementSquared);
  return true;
}

} // namespace voronoi
