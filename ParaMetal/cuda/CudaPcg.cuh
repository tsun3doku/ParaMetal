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
    int iterationCount;
};

void recordPcgSolve(
    cudaStream_t stream,
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
    uint32_t maxIterations,
    float tolerance,
    bool computeInitialAp0);

} // namespace cudaPcg
