#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDTopology.hpp"
#include "RVDRestriction.cuh"
#include "../../cuda/CudaBuffer.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace voronoi {

enum class RVDFragmentError : uint32_t {
  None = 0,
  BaseVolumeOvercoverage = 1,
  InvalidBaseFaceMeasure = 4,
  UnmappedModelMaterial = 9,
  InvalidSurfaceExposure = 10,
  MissingFaceRegions = 13,
};

inline const char *rvdFragmentErrorName(uint32_t code) {
  switch (RVDFragmentError(code)) {
  case RVDFragmentError::None: return "None";
  case RVDFragmentError::BaseVolumeOvercoverage: return "BaseVolumeOvercoverage";
  case RVDFragmentError::InvalidBaseFaceMeasure: return "InvalidBaseFaceMeasure";
  case RVDFragmentError::UnmappedModelMaterial: return "UnmappedModelMaterial";
  case RVDFragmentError::InvalidSurfaceExposure: return "InvalidSurfaceExposure";
  case RVDFragmentError::MissingFaceRegions: return "MissingFaceRegions";
  default: return "Unknown";
  }
}

bool buildFragmentsAndFaces(
    const RVDInput &input,
    const BaseCellDomain &baseDomain,
    const BaseFaceTopology &faceTopology,
    const std::vector<ModelRestriction> &models,
    const cudaUtils::CudaBuffer<double4> &devSurfaceExposure,
    cudaUtils::CudaBuffer<uint32_t> &devContactFaceBySurface,
    uint32_t &totalBaseFragmentFaces,
    RVDResult &result,
    cudaStream_t stream = nullptr);

} // namespace voronoi
