#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cudaPcg {

constexpr uint32_t BlockSize = 256;

struct PcgStatus {
    float rz;
    float rzNew;
    float pAp;
    float residual0;
    float residual;
    float alpha;
    float beta;
    int converged;
    int failed;
    // Set when maxIterations was reached without convergence or breakdown
    int exhausted;
    int iterationCount;
};

// Device-resident solve parameters (graph-independent)
struct PcgConfig {
    uint32_t maxIterations;
    float    tolerance;
};

// Compute inverse diagonal for Jacobi preconditioning
void launchComputeInvDiag(
    uint32_t nodeCount,
    const int* d_rowOffsets,
    const int* d_columnIndices,
    const float* d_values,
    float* d_invDiag,
    cudaStream_t stream);

// Check solve status on device and set success flag for gated publication
void launchResolveSolutionStatus(
    const PcgStatus* d_status,
    const uint32_t* d_scheduleStatus,
    float* d_solution,
    uint32_t nodeCount,
    int* d_successFlag,
    cudaStream_t stream);

// Build standalone PCG graph: one persistent cooperative kernel (prologue + iteration loop)
cudaGraphExec_t buildPcgGraphExec(
    uint32_t nodeCount,
    const int* d_rowOffsets,
    const int* d_columnIndices,
    const float* d_values,
    const float* d_invDiag,
    float* d_x,
    const float* d_b,
    float* d_Ap,
    float* d_r,
    float* d_z,
    float* d_p,
    PcgStatus* d_status,
    PcgConfig* d_config);

} // namespace cudaPcg
