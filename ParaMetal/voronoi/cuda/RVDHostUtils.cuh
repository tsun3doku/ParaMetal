#pragma once

#include "RVDStatus.cuh"
#include "../../cuda/CudaBuffer.cuh"
#include "../../cuda/CudaCheck.cuh"

#include <cub/cub.cuh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace voronoi::rvdHost {

inline constexpr uint32_t BlockSize = 256;

constexpr uint32_t blockCount(uint32_t count, uint32_t blockSize = BlockSize) {
  return (count + blockSize - 1) / blockSize;
}

inline bool checkLaunch(const char *label) {
  return cudaUtils::cudaCheck(cudaGetLastError(), label);
}

inline bool synchronize(cudaStream_t stream, const char *label) {
  return cudaUtils::cudaCheck(cudaStreamSynchronize(stream), label);
}

inline bool ensureCapacity(cudaUtils::CudaBuffer<uint8_t> &scratch, size_t bytes) {
  return scratch.size() >= bytes || scratch.allocate(bytes);
}

template <typename T>
inline bool ensureCapacity(cudaUtils::CudaBuffer<T> &buffer, size_t needed) {
  return buffer.size() >= needed || buffer.allocate(needed);
}

inline bool exclusiveSum(
    const uint32_t *input,
    uint32_t *output,
    int count,
    cudaUtils::CudaBuffer<uint8_t> &scratch,
    cudaStream_t stream) {
  size_t bytes = 0;
  if (cub::DeviceScan::ExclusiveSum(nullptr, bytes, input, output, count, stream) != cudaSuccess) {
    return false;
  }
  if (!ensureCapacity(scratch, bytes)) return false;
  return cub::DeviceScan::ExclusiveSum(scratch.get(), bytes, input, output, count, stream) == cudaSuccess;
}

template <typename KeyT, typename ValueT>
inline bool sortPairs(
    const KeyT *keysIn,
    KeyT *keysOut,
    const ValueT *valuesIn,
    ValueT *valuesOut,
    int count,
    cudaUtils::CudaBuffer<uint8_t> &scratch,
    cudaStream_t stream) {
  size_t bytes = 0;
  if (cub::DeviceRadixSort::SortPairs(
          nullptr, bytes, keysIn, keysOut, valuesIn, valuesOut,
          count, 0, sizeof(KeyT) * 8, stream) != cudaSuccess) {
    return false;
  }
  if (!ensureCapacity(scratch, bytes)) return false;
  return cub::DeviceRadixSort::SortPairs(
             scratch.get(), bytes, keysIn, keysOut, valuesIn, valuesOut,
             count, 0, sizeof(KeyT) * 8, stream) == cudaSuccess;
}

template <typename T, typename FlagT>
inline bool selectFlagged(
    const T *input,
    const FlagT *flags,
    T *output,
    int *selectedCount,
    int count,
    cudaUtils::CudaBuffer<uint8_t> &scratch,
    cudaStream_t stream) {
  size_t bytes = 0;
  if (cub::DeviceSelect::Flagged(
          nullptr, bytes, input, flags, output, selectedCount, count, stream) != cudaSuccess) {
    return false;
  }
  if (!ensureCapacity(scratch, bytes)) return false;
  return cub::DeviceSelect::Flagged(
             scratch.get(), bytes, input, flags, output, selectedCount, count, stream) == cudaSuccess;
}

template <typename KeyT, typename ValueT>
inline bool reduceByKeySum(
    const KeyT *keysIn,
    KeyT *keysOut,
    const ValueT *valuesIn,
    ValueT *valuesOut,
    int *runCount,
    int count,
    cudaUtils::CudaBuffer<uint8_t> &scratch,
    cudaStream_t stream) {
  size_t bytes = 0;
  if (cub::DeviceReduce::ReduceByKey(
          nullptr, bytes, keysIn, keysOut, valuesIn, valuesOut,
          runCount, cub::Sum(), count, stream) != cudaSuccess) {
    return false;
  }
  if (!ensureCapacity(scratch, bytes)) return false;
  return cub::DeviceReduce::ReduceByKey(
             scratch.get(), bytes, keysIn, keysOut, valuesIn, valuesOut,
             runCount, cub::Sum(), count, stream) == cudaSuccess;
}

template <typename T>
inline bool copyDeviceToDeviceAsync(
    T *destination,
    const T *source,
    size_t count,
    cudaStream_t stream,
    const char *label) {
  if (count == 0) return true;
  return cudaUtils::cudaCheck(
      cudaMemcpyAsync(destination, source, count * sizeof(T), cudaMemcpyDeviceToDevice, stream),
      label);
}

template <typename T>
inline bool copyToHostAsync(
    T *destination,
    const T *source,
    size_t count,
    cudaStream_t stream,
    const char *label) {
  if (count == 0) return true;
  return cudaUtils::cudaCheck(
      cudaMemcpyAsync(destination, source, count * sizeof(T), cudaMemcpyDeviceToHost, stream),
      label);
}

template <typename T>
inline bool copyToHost(
    T *destination,
    const T *source,
    size_t count,
    cudaStream_t stream,
    const char *label) {
  return copyToHostAsync(destination, source, count, stream, label) &&
         cudaUtils::cudaCheck(cudaStreamSynchronize(stream), label);
}

template <typename T>
inline bool readAt(
    T &value,
    const T *source,
    size_t index,
    cudaStream_t stream,
    const char *label) {
  return copyToHost(&value, source + index, 1, stream, label);
}

template <typename T>
inline bool readLastExclusiveScanTotal(
    const T *counts,
    const T *offsets,
    uint32_t count,
    cudaStream_t stream,
    T &total,
    const char *label) {
  if (count == 0) {
    total = T(0);
    return true;
  }
  T lastCount = T(0);
  T lastOffset = T(0);
  if (!copyToHostAsync(&lastCount, counts + (count - 1), 1, stream, label) ||
      !copyToHostAsync(&lastOffset, offsets + (count - 1), 1, stream, label) ||
      !cudaUtils::cudaCheck(cudaStreamSynchronize(stream), label)) {
    return false;
  }
  total = lastOffset + lastCount;
  return true;
}

inline bool zero(void *destination, size_t bytes, cudaStream_t stream, const char *label) {
  if (bytes == 0) return true;
  return cudaUtils::cudaCheck(cudaMemsetAsync(destination, 0, bytes, stream), label);
}

inline bool fill(void *destination, int value, size_t bytes, cudaStream_t stream, const char *label) {
  if (bytes == 0) return true;
  return cudaUtils::cudaCheck(cudaMemsetAsync(destination, value, bytes, stream), label);
}

inline bool readPackedError(
    const cudaUtils::CudaBuffer<RvdPackedError> &deviceError,
    cudaStream_t stream,
    RvdPackedError &packedError) {
  return copyToHost(&packedError, deviceError.get(), 1, stream, "read RVD packed error");
}

} // namespace voronoi::rvdHost
