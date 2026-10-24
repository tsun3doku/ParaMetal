#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "../../cuda/CudaBuffer.cuh"

#include <vector>
#include <cstdint>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

namespace voronoi {

struct BaseFace {
    uint32_t cellA = InvalidId;      // min(cellA, cellB)
    uint32_t cellB = InvalidId;      // max(cellA, cellB), or InvalidId for domain faces
    uint32_t slotA = InvalidId;      // Plane slot on cellA
    uint32_t slotB = InvalidId;      // Plane slot on cellB (InvalidId for domain faces)
    uint32_t domainFace = InvalidId; // 0..5 for domain boundary
};

struct BaseFaceMeasure {
    double area = 0.0;
    double areaError = 0.0;
    double3 moment = make_double3(0.0, 0.0, 0.0);
    double3 normal = make_double3(0.0, 0.0, 0.0);
    uint32_t valid = 0;
};

struct FaceRegion {
    BaseFaceMeasure measure;
    uint32_t materialA = InvalidId;
    uint32_t materialB = InvalidId;
};

struct BaseFaceTopology {
    uint32_t totalBaseFaces = 0;

    cudaUtils::CudaBuffer<BaseFace> devBaseFaces;
    cudaUtils::CudaBuffer<BaseFaceMeasure> devMeasures;
    cudaUtils::CudaBuffer<uint32_t> devFaceVertexOffsets;
    cudaUtils::CudaBuffer<double3> devFaceVertices;
    cudaUtils::CudaBuffer<uint32_t> devRegionOffsets;
    cudaUtils::CudaBuffer<FaceRegion> devRegions;
};

bool buildBaseFaces(
    const BaseCellDomain& baseDomain,
    BaseFaceTopology& faceTopology,
    cudaStream_t stream = nullptr);

} // namespace voronoi
