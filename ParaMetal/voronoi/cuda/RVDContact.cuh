#pragma once

#include "RVD.hpp"
#include "RVDRestriction.cuh"
#include "RVDBaseCells.cuh"
#include "RVDGeometry.cuh"
#include "../../cuda/CudaBuffer.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace voronoi {

// One clipped overlap between two restricted model surfaces. The couplingArea
// is measured in the mortar frame; the per-side areas are its projection back
// onto each surface.
struct MortarOverlap {
  uint32_t surfaceA = 0;
  uint32_t surfaceB = 0;
  double couplingArea = 0.0; // Am
  double areaA = 0.0;        // Am / |nA · m|
  double areaB = 0.0;        // Am / |nB · m|
  double3 momentA{0.0, 0.0, 0.0}; // CA * areaA
  double3 momentB{0.0, 0.0, 0.0}; // CB * areaB
};

bool buildMortarOverlaps(
    const std::vector<ModelRestriction> &models,
    const std::vector<cudaGeometry::DeviceModelGeometryView> &geometries,
    const BaseCellDomain &baseDomain,
    double contactRange,
    cudaUtils::CudaBuffer<MortarOverlap> &devOverlaps,
    uint32_t &overlapCount,
    cudaUtils::CudaBuffer<double4> &devExposure,
    cudaStream_t stream = nullptr);

bool finalizeContactCouplings(
    const cudaUtils::CudaBuffer<MortarOverlap> &devOverlaps,
    uint32_t overlapCount,
    const cudaUtils::CudaBuffer<uint32_t> &devContactFaceBySurface,
    RVDResult &result,
    cudaStream_t stream = nullptr);

} // namespace voronoi
