#pragma once

#include "KNNNeighborsKernels.cuh"

#include "../../cuda/CudaBuffer.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace voronoi {

class KNNNeighbors::Implementation {
public:
  Implementation() = default;
  ~Implementation();

  bool initialize(VulkanDevice &vulkanDevice);
  bool setupDevice(int ordinal);
  void cleanup();
  bool buildNeighbors(uint32_t neighborCount, const std::vector<uint32_t> &queryIds);
  bool buildSpatialIndex(const std::vector<glm::vec4> &seeds);
  bool downloadPrefix(uint32_t neighborCount, std::vector<uint32_t> &output) const;

  uint32_t seedCount() const;
  DeviceKnn makeDeviceKnn() const;

private:
  bool buildCellOffsets();
  bool buildGrid(const std::vector<glm::vec4> &seeds);

  int cudaDeviceOrdinal = -1;
  cudaStream_t stream = nullptr;
  bool initialized = false;
  cudaUtils::CudaBuffer<float4> seeds;
  cudaUtils::CudaBuffer<uint32_t> neighbors;
  cudaUtils::CudaBuffer<double> distanceHeap;
  cudaUtils::CudaBuffer<uint32_t> searchFailure;
  cudaUtils::CudaBuffer<BucketEntry> bucketEntries;
  cudaUtils::CudaBuffer<uint32_t> cellStart;
  cudaUtils::CudaBuffer<int3> cellOffsets;
  cudaUtils::CudaBuffer<double> cellOffsetDistances;
  mutable cudaUtils::CudaBuffer<uint32_t> prefixBuffer;
  DeviceGrid grid{};
  uint32_t seedCountValue = 0;
  uint32_t neighborStrideValue = 0;
  uint32_t cellOffsetCount = 0;
};

} // namespace voronoi
