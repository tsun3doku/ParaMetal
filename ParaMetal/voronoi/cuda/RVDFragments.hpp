#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDTopology.hpp"
#include "RVDRestriction.cuh"
#include "../../cuda/CudaBuffer.cuh"

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>

namespace voronoi {

bool buildFragmentsAndFaces(
    const RVDInput& input,
    const BaseCellDomain& baseDomain,
    const BaseFaceTopology& faceTopology,
    const std::vector<ModelRestriction>& models,
    const cudaUtils::CudaBuffer<double4>& devSurfaceExposure,
    cudaUtils::CudaBuffer<uint32_t>& devContactFaceBySurface,
    uint32_t& totalBaseFragmentFaces,
    RVDResult& result,
    cudaStream_t stream = nullptr);

bool validateFragmentTopology(
    const RVDResult& result,
    uint32_t totalBaseFragmentFaces,
    double toleranceClosure = 0.005,
    double toleranceGauss = 0.01);

} // namespace voronoi
