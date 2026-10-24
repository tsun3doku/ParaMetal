#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace voronoi {

enum class RVDStatus : uint32_t {
  Success,
  EmptyCell,
  VertexOverflow,
  PlaneOverflow,
  TriangleOverflow,
  InconsistentBoundary,
  NonFiniteGeometry,
  InvalidIndex,
  NegativeVolume,
  SegmentOverflow,
  Count
};

inline const char *rvdStatusName(RVDStatus status) {
  switch (status) {
  case RVDStatus::Success: return "Success";
  case RVDStatus::EmptyCell: return "EmptyCell";
  case RVDStatus::VertexOverflow: return "VertexOverflow";
  case RVDStatus::PlaneOverflow: return "PlaneOverflow";
  case RVDStatus::TriangleOverflow: return "TriangleOverflow";
  case RVDStatus::InconsistentBoundary: return "InconsistentBoundary";
  case RVDStatus::NonFiniteGeometry: return "NonFiniteGeometry";
  case RVDStatus::InvalidIndex: return "InvalidIndex";
  case RVDStatus::NegativeVolume: return "NegativeVolume";
  case RVDStatus::SegmentOverflow: return "SegmentOverflow";
  default: return "Unknown";
  }
}

// Passes publish the first failure as a packed (code, id) pair through a
// single device slot so the host can report it without a full status readback.
using RvdPackedError = unsigned long long;

__host__ __device__ inline RvdPackedError rvdPackError(uint32_t code, uint32_t id) {
  return (RvdPackedError(code) << 32) | RvdPackedError(id);
}

__host__ __device__ inline uint32_t rvdErrorCode(RvdPackedError packed) {
  return uint32_t(packed >> 32);
}

__host__ __device__ inline uint32_t rvdErrorId(RvdPackedError packed) {
  return uint32_t(packed);
}

__device__ inline void rvdReportError(RvdPackedError *packedError, uint32_t code, uint32_t id) {
  atomicCAS(packedError, RvdPackedError(0), rvdPackError(code, id));
}

__device__ inline bool rvdReportErrorOnce(RvdPackedError *packedError, uint32_t code, uint32_t id) {
  return atomicCAS(packedError, RvdPackedError(0), rvdPackError(code, id)) == RvdPackedError(0);
}

} // namespace voronoi
