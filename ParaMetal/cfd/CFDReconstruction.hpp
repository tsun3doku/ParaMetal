#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "../cuda/CudaBuffer.cuh"
#include "../voronoi/GlobalDomainTopology.hpp"
#include "CFDTopology.hpp"

namespace cfd {

constexpr uint32_t ReconCentralNeighborLimit = 20u;
constexpr uint32_t ReconDirectionalCandidateLimit = 8u;
constexpr uint32_t ReconDirectionalNeighborLimit = 12u;
constexpr uint32_t ReconQuadraticUnknownCount = 9u;
constexpr uint32_t ReconLinearUnknownCount = 3u;

struct ReconCellGeometry {
    double3 centroid{0.0, 0.0, 0.0};
    double volume = 0.0;
    double h = 0.0;
    double s[6] = {};  // unnormalized second moments about the centroid
    double m2[6] = {}; // s / volume
    double q[6] = {};  // m2 / (h * h)
};

struct ReconFaceGeometry {
    double3 centroid{0.0, 0.0, 0.0};
    double area = 0.0;
    double m2[6] = {}; // area-averaged second moments about the centroid
};

struct ReconFinalCoefficients {
    float c[3][ReconQuadraticUnknownCount] = {};
    uint32_t valid = 0u;
    uint32_t pad[3] = {};
};

class CFDReconstruction {
public:
    bool build(
        const topology::GlobalDomain& domain,
        const std::vector<uint32_t>& globalFragmentToFluidNode,
        const std::vector<uint32_t>& fluidNodeToGlobalFragment,
        const CFDTopology& topology,
        cudaStream_t stream = nullptr);

    bool computeCoefficients(
        const glm::vec4* cellAverages,
        const double* referenceVelocity,
        cudaStream_t stream = nullptr);

    cudaError_t appendCoefficientsNode(
        cudaGraph_t graph,
        cudaGraphNode_t& prev,
        const glm::vec4* cellAverages,
        const double* referenceVelocity);

    void reset();

    bool ready() const { return cellCount_ > 0u; }
    uint32_t cellCount() const { return cellCount_; }
    uint32_t faceCount() const { return faceCount_; }

    const cudaUtils::CudaBuffer<ReconCellGeometry>& cellGeometry() const {
        return d_cellGeometry_;
    }
    const cudaUtils::CudaBuffer<ReconFaceGeometry>& faceGeometry() const {
        return d_faceGeometry_;
    }
    const cudaUtils::CudaBuffer<ReconFinalCoefficients>& coefficients() const {
        return d_coefficients_;
    }

    bool downloadInvalidCoefficientCount(uint32_t& count, cudaStream_t stream) const;

private:
    bool verifyReconstruction(
        const topology::GlobalDomain& domain,
        const std::vector<double3>& cellCentroids,
        const std::vector<uint32_t>& faceNodeA,
        const std::vector<uint32_t>& faceNodeB,
        const std::vector<double3>& faceNormals,
        const std::vector<uint32_t>& fineFaceGlobal,
        const std::vector<double3>& loopVertices,
        cudaStream_t stream);

    uint32_t cellCount_ = 0u;
    uint32_t faceCount_ = 0u;
    uint32_t domainFaceCount_ = 0u;

    cudaUtils::CudaBuffer<uint32_t> d_centralCount_;
    cudaUtils::CudaBuffer<uint32_t> d_centralNeighbor_;
    cudaUtils::CudaBuffer<float> d_centralOp_;
    cudaUtils::CudaBuffer<uint32_t> d_directionalCount_;
    cudaUtils::CudaBuffer<uint32_t> d_directionalNeighborCount_;
    cudaUtils::CudaBuffer<uint32_t> d_directionalNeighbor_;
    cudaUtils::CudaBuffer<float> d_directionalOp_;
    cudaUtils::CudaBuffer<ReconCellGeometry> d_cellGeometry_;
    cudaUtils::CudaBuffer<ReconFaceGeometry> d_faceGeometry_;
    cudaUtils::CudaBuffer<ReconFinalCoefficients> d_coefficients_;
    cudaUtils::CudaBuffer<uint32_t> d_diagnostics_;
};

} // namespace cfd
