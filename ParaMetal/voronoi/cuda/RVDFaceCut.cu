#include "RVDFaceCut.cuh"
#include "RVDHostUtils.cuh"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace voronoi {

static __device__ inline void faceCutSortedPlanes(
    const FaceCutVertexKey &key, uint32_t planes[3]) {
  planes[0] = key.basePlaneA;
  planes[1] = key.basePlaneB;
  planes[2] = key.basePlaneC;
  for (int i = 1; i < 3; ++i) {
    const uint32_t value = planes[i];
    int j = i - 1;
    while (j >= 0 && planes[j] > value) {
      planes[j + 1] = planes[j];
      --j;
    }
    planes[j + 1] = value;
  }
}

static __device__ inline bool faceCutKeyEqual(
    const FaceCutVertexKey &a, const FaceCutVertexKey &b) {
  if (a.triangleId != b.triangleId || a.cadVertexId != b.cadVertexId ||
      a.cadEdgeVertexA != b.cadEdgeVertexA ||
      a.cadEdgeVertexB != b.cadEdgeVertexB) {
    return false;
  }
  uint32_t aPlanes[3], bPlanes[3];
  faceCutSortedPlanes(a, aPlanes);
  faceCutSortedPlanes(b, bPlanes);
  return aPlanes[0] == bPlanes[0] && aPlanes[1] == bPlanes[1] &&
         aPlanes[2] == bPlanes[2];
}

static __device__ inline bool faceCutKeyLess(
    const FaceCutVertexKey &a, const FaceCutVertexKey &b) {
  uint32_t aPlanes[3], bPlanes[3];
  faceCutSortedPlanes(a, aPlanes);
  faceCutSortedPlanes(b, bPlanes);
  for (int i = 0; i < 3; ++i) {
    if (aPlanes[i] != bPlanes[i]) return aPlanes[i] < bPlanes[i];
  }
  if (a.triangleId != b.triangleId) return a.triangleId < b.triangleId;
  if (a.cadVertexId != b.cadVertexId) return a.cadVertexId < b.cadVertexId;
  if (a.cadEdgeVertexA != b.cadEdgeVertexA)
    return a.cadEdgeVertexA < b.cadEdgeVertexA;
  return a.cadEdgeVertexB < b.cadEdgeVertexB;
}

struct FaceCutEndpointRef {
  FaceCutVertexKey key;
  uint32_t segmentIndex = InvalidId;
  uint32_t endpoint = InvalidId;

  __device__ bool operator<(const FaceCutEndpointRef &other) const {
    if (!faceCutKeyEqual(key, other.key)) {
      return faceCutKeyLess(key, other.key);
    }
    if (segmentIndex != other.segmentIndex) {
      return segmentIndex < other.segmentIndex;
    }
    return endpoint < other.endpoint;
  }
};

static __global__ void fillFaceCutEndpointRefsKernel(
    FaceBoundarySegment *segments,
    uint32_t count,
    FaceCutEndpointRef *refs) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  segments[i].segmentId = i;
  refs[2 * i] = {segments[i].keyA, i, 0u};
  refs[2 * i + 1] = {segments[i].keyB, i, 1u};
}

static __global__ void markFaceCutRunsKernel(
    const FaceCutEndpointRef *refs,
    uint32_t count,
    uint32_t *flags) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  flags[i] = (i == 0 || !faceCutKeyEqual(refs[i].key, refs[i - 1].key))
      ? 1u
      : 0u;
}

static __global__ void writeCanonicalCuts(
    const FaceCutEndpointRef *refs,
    const uint32_t *runOffsets,
    const uint32_t *runStarts,
    uint32_t *canonicalIds,
    uint32_t count,
    FaceBoundarySegment *segments) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint32_t canonicalId = runOffsets[i] + runStarts[i] - 1u;
  canonicalIds[i] = canonicalId;
  FaceBoundarySegment &segment = segments[refs[i].segmentIndex];
  if (refs[i].endpoint == 0u) {
    segment.canonicalA = canonicalId;
  } else {
    segment.canonicalB = canonicalId;
  }
}

static __global__ void verifyRunIds(
    const FaceCutEndpointRef *refs,
    const uint32_t *canonicalIds,
    uint32_t count,
    RvdPackedError *packedError) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0 || i >= count) return;
  const bool sameKey = faceCutKeyEqual(refs[i].key, refs[i - 1].key);
  const bool sameId = canonicalIds[i] == canonicalIds[i - 1];
  if (sameKey != sameId) {
    rvdReportErrorOnce(
        packedError,
        uint32_t(RVDRestrictionError::InvalidFaceCutCanonicalization),
        refs[i].segmentIndex);
  }
}

static __global__ void verifyFaceCuts(
    const FaceBoundarySegment *segments,
    uint32_t count,
    uint32_t offset,
    RvdPackedError *packedError) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const FaceBoundarySegment &segment = segments[i];

  const bool valid =
      segment.canonicalA != InvalidId &&
      segment.canonicalB != InvalidId &&
      segment.canonicalA != segment.canonicalB &&
      segment.runtimeModelId != InvalidId &&
      segment.triangleId != InvalidId &&
      isfinite(segment.a.x) && isfinite(segment.a.y) &&
      isfinite(segment.a.z) && isfinite(segment.b.x) &&
      isfinite(segment.b.y) && isfinite(segment.b.z);
  if (!valid) {
    rvdReportErrorOnce(
        packedError,
        uint32_t(RVDRestrictionError::InvalidFaceCutCanonicalization),
        offset + i);
  }
}

bool canonicalizeFaceCutVertices(
    std::vector<ModelRestriction> &models,
    cudaStream_t stream) {
  std::vector<size_t> offsets(models.size(), 0);
  size_t totalSegments = 0;
  for (size_t i = 0; i < models.size(); ++i) {
    offsets[i] = totalSegments;
    totalSegments += models[i].devFaceSegments.size();
  }
  if (totalSegments == 0) return true;
  if (totalSegments >
      size_t(std::numeric_limits<int>::max()) / 2u) {
    return false;
  }

  cudaUtils::CudaBuffer<FaceBoundarySegment> allSegments;
  if (!allSegments.allocate(totalSegments)) return false;
  for (size_t i = 0; i < models.size(); ++i) {
    const size_t count = models[i].devFaceSegments.size();
    if (count == 0) continue;
    if (!cudaUtils::cudaCheck(
            cudaMemcpyAsync(
                allSegments.get() + offsets[i],
                models[i].devFaceSegments.get(),
                count * sizeof(FaceBoundarySegment),
                cudaMemcpyDeviceToDevice,
                stream),
            "copy face cut segments")) {
      return false;
    }
  }

  const uint32_t segmentCount = uint32_t(totalSegments);
  const uint32_t endpointCount = segmentCount * 2u;
  cudaUtils::CudaBuffer<FaceCutEndpointRef> refs;
  cudaUtils::CudaBuffer<uint32_t> flags;
  cudaUtils::CudaBuffer<uint32_t> canonicalIds;
  cudaUtils::CudaBuffer<RvdPackedError> validationError;
  if (!refs.allocate(endpointCount) || !flags.allocate(endpointCount) ||
      !canonicalIds.allocate(endpointCount) ||
      !validationError.allocate(1) ||
      !rvdHost::zero(
          validationError.get(), sizeof(RvdPackedError), stream,
          "clear face cut run validation error")) {
    return false;
  }

  constexpr uint32_t BlockSize = 256;
  fillFaceCutEndpointRefsKernel<<<
      rvdHost::blockCount(segmentCount, BlockSize), BlockSize, 0, stream>>>(
      allSegments.get(), segmentCount, refs.get());
  if (!rvdHost::checkLaunch("launch fillFaceCutEndpointRefsKernel")) {
    return false;
  }

  thrust::sort(
      thrust::cuda::par.on(stream),
      thrust::device_ptr<FaceCutEndpointRef>(refs.get()),
      thrust::device_ptr<FaceCutEndpointRef>(refs.get()) + endpointCount);

  markFaceCutRunsKernel<<<
      rvdHost::blockCount(endpointCount, BlockSize), BlockSize, 0, stream>>>(
      refs.get(), endpointCount, flags.get());
  if (!rvdHost::checkLaunch("launch markFaceCutRunsKernel")) return false;

  cudaUtils::CudaBuffer<uint8_t> scanScratch;
  if (!rvdHost::exclusiveSum(
          flags.get(),
          canonicalIds.get(),
          int(endpointCount),
          scanScratch,
          stream)) {
    return false;
  }

  writeCanonicalCuts<<<
      rvdHost::blockCount(endpointCount, BlockSize), BlockSize, 0, stream>>>(
      refs.get(), canonicalIds.get(), flags.get(), canonicalIds.get(),
      endpointCount, allSegments.get());
  if (!rvdHost::checkLaunch("launch writeCanonicalCuts")) {
    return false;
  }

  verifyRunIds<<<
      rvdHost::blockCount(endpointCount, BlockSize), BlockSize, 0, stream>>>(
      refs.get(), canonicalIds.get(), endpointCount, validationError.get());
  if (!rvdHost::checkLaunch("launch verifyRunIds")) {
    return false;
  }

  verifyFaceCuts<<<
      rvdHost::blockCount(segmentCount, BlockSize), BlockSize, 0, stream>>>(
      allSegments.get(), segmentCount, 0, validationError.get());
  if (!rvdHost::checkLaunch("launch verifyFaceCuts")) {
    return false;
  }

  RvdPackedError hostValidationError = 0;
  if (!rvdHost::readPackedError(
          validationError, stream, hostValidationError)) {
    return false;
  }
  if (hostValidationError != 0) {
    std::cerr << "[RVD] face cut validation failed code="
              << rvdErrorCode(hostValidationError)
              << " segment=" << rvdErrorId(hostValidationError) << std::endl;
    return false;
  }

  for (size_t i = 0; i < models.size(); ++i) {
    const size_t count = models[i].devFaceSegments.size();
    if (count == 0) continue;
    if (!cudaUtils::cudaCheck(
            cudaMemcpyAsync(
                models[i].devFaceSegments.get(),
                allSegments.get() + offsets[i],
                count * sizeof(FaceBoundarySegment),
                cudaMemcpyDeviceToDevice,
                stream),
            "copy canonical face cut segments")) {
      return false;
    }
  }
  return true;
}

} // namespace voronoi
