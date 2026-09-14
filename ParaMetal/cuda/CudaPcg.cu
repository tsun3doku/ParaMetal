#include "CudaPcg.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

namespace cudaPcg {

static __device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

static __device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x / warpSize)) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void csrSpmvKernel(
    uint32_t totalNodes,
    const int* __restrict__ rowOffsets,
    const int* __restrict__ columnIndices,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    const PcgStatus* __restrict__ status) {
    if (status && status->converged) return;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalNodes) return;
    float sum = 0.0f;
    const int rowStart = rowOffsets[i];
    const int rowEnd = rowOffsets[i + 1];
    for (int j = rowStart; j < rowEnd; ++j) {
        sum += values[j] * x[columnIndices[j]];
    }
    y[i] = sum;
}

__global__ void initResidualAndJacobiKernel(
    uint32_t totalNodes,
    const float* __restrict__ b,
    const float* __restrict__ Ap0,
    const float* __restrict__ invDiag,
    float* __restrict__ x,
    float* __restrict__ r,
    float* __restrict__ z,
    float* __restrict__ p,
    PcgStatus* __restrict__ status,
    bool hasAp0) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalNodes) return;
    if (!hasAp0) {
        x[i] = 0.0f;
    }
    const float ri = hasAp0 ? (b[i] - Ap0[i]) : b[i];
    const float zi = ri * invDiag[i];
    r[i] = ri;
    z[i] = zi;
    p[i] = zi;
    if (i == 0 && status) {
        status->converged = 0;
        status->failed = 0;
        status->iterationCount = 0;
        status->rz = 0.0f;
        status->rzNew = 0.0f;
        status->pAp = 0.0f;
        status->residual0 = 0.0f;
        status->residual = 0.0f;
        status->alpha = 0.0f;
        status->beta = 0.0f;
    }
}

__global__ void reduceInitDotProductsKernel(
    uint32_t totalNodes,
    const float* __restrict__ r,
    const float* __restrict__ z,
    PcgStatus* __restrict__ status) {
    float sumRz = 0.0f;
    float sumR2 = 0.0f;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < totalNodes; i += blockDim.x * gridDim.x) {
        const float ri = r[i];
        sumRz += ri * z[i];
        sumR2 += ri * ri;
    }
    sumRz = blockReduceSum(sumRz);
    sumR2 = blockReduceSum(sumR2);
    if (threadIdx.x == 0) {
        atomicAdd(&status->rz, sumRz);
        atomicAdd(&status->residual0, sumR2);
    }
}

__global__ void finalizeInitDotKernel(PcgStatus* __restrict__ status) {
    if (threadIdx.x == 0) {
        status->residual0 = sqrtf(fmaxf(status->residual0, 0.0f));
        status->residual = status->residual0;

        if (status->residual0 <= 1e-12f) {
            status->converged = 1;
            status->failed = 0;
        }
    }
}

__global__ void reducePApKernel(
    uint32_t totalNodes,
    const float* __restrict__ p,
    const float* __restrict__ Ap,
    PcgStatus* __restrict__ status) {
    if (status->converged) return;
    float sumPAp = 0.0f;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < totalNodes; i += blockDim.x * gridDim.x) {
        sumPAp += p[i] * Ap[i];
    }
    sumPAp = blockReduceSum(sumPAp);
    if (threadIdx.x == 0) {
        atomicAdd(&status->pAp, sumPAp);
    }
}

__global__ void computeAlphaKernel(PcgStatus* __restrict__ status) {
    if (threadIdx.x == 0 && !status->converged) {
        const float denom = status->pAp;
        if (denom <= 1e-24f || !isfinite(denom)) {
            status->failed = 1;
            status->converged = 1;
            status->alpha = 0.0f;
            return;
        }
        status->alpha = status->rz / denom;
        status->rzNew = 0.0f;
        status->residual = 0.0f;
        status->pAp = 0.0f;
    }
}

__global__ void updateXAndRKernel(
    uint32_t totalNodes,
    const float* __restrict__ p,
    const float* __restrict__ Ap,
    const float* __restrict__ invDiag,
    float* __restrict__ x,
    float* __restrict__ r,
    float* __restrict__ z,
    PcgStatus* __restrict__ status) {
    if (status->converged) return;
    const float alpha = status->alpha;
    float localRz = 0.0f;
    float localR2 = 0.0f;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < totalNodes; i += blockDim.x * gridDim.x) {
        const float xi = x[i] + alpha * p[i];
        const float ri = r[i] - alpha * Ap[i];
        const float zi = ri * invDiag[i];
        x[i] = xi;
        r[i] = ri;
        z[i] = zi;
        localRz += ri * zi;
        localR2 += ri * ri;
    }
    localRz = blockReduceSum(localRz);
    localR2 = blockReduceSum(localR2);
    if (threadIdx.x == 0) {
        atomicAdd(&status->rzNew, localRz);
        atomicAdd(&status->residual, localR2);
    }
}

__global__ void checkConvergenceAndComputeBetaKernel(
    PcgStatus* __restrict__ status,
    float tolerance,
    int iterIndex) {
    if (threadIdx.x == 0 && !status->converged) {
        status->iterationCount = iterIndex + 1;
        status->residual = sqrtf(fmaxf(status->residual, 0.0f));
        const float relRes = status->residual / fmaxf(status->residual0, 1e-12f);
        if (relRes <= tolerance || status->residual <= 1e-10f) {
            status->converged = 1;
        } else {
            const float rzOld = fmaxf(status->rz, 1e-24f);
            status->beta = status->rzNew / rzOld;
            status->rz = status->rzNew;
        }
    }
}

__global__ void updatePKernel(
    uint32_t totalNodes,
    const float* __restrict__ z,
    float* __restrict__ p,
    const PcgStatus* __restrict__ status) {
    if (status->converged) return;
    const float beta = status->beta;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalNodes) return;
    p[i] = z[i] + beta * p[i];
}

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
    bool computeInitialAp0) {
    if (nodeCount == 0) return;

    const uint32_t gridNodes = (nodeCount + BlockSize - 1) / BlockSize;
    const uint32_t reduceBlocks = std::min(gridNodes, 64u);

    if (computeInitialAp0) {
        csrSpmvKernel<<<gridNodes, BlockSize, 0, stream>>>(
            nodeCount, d_rowOffsets, d_columnIndices, d_values, d_x, d_Ap, nullptr);
        initResidualAndJacobiKernel<<<gridNodes, BlockSize, 0, stream>>>(
            nodeCount, d_b, d_Ap, d_invDiag, d_x, d_r, d_z, d_p, d_status, true);
    } else {
        initResidualAndJacobiKernel<<<gridNodes, BlockSize, 0, stream>>>(
            nodeCount, d_b, nullptr, d_invDiag, d_x, d_r, d_z, d_p, d_status, false);
    }

    reduceInitDotProductsKernel<<<reduceBlocks, BlockSize, 0, stream>>>(
        nodeCount, d_r, d_z, d_status);
    finalizeInitDotKernel<<<1, 1, 0, stream>>>(d_status);

    for (uint32_t iter = 0; iter < maxIterations; ++iter) {
        csrSpmvKernel<<<gridNodes, BlockSize, 0, stream>>>(
            nodeCount, d_rowOffsets, d_columnIndices, d_values, d_p, d_Ap, d_status);

        reducePApKernel<<<reduceBlocks, BlockSize, 0, stream>>>(
            nodeCount, d_p, d_Ap, d_status);

        computeAlphaKernel<<<1, 1, 0, stream>>>(d_status);

        updateXAndRKernel<<<reduceBlocks, BlockSize, 0, stream>>>(
            nodeCount, d_p, d_Ap, d_invDiag, d_x, d_r, d_z, d_status);

        checkConvergenceAndComputeBetaKernel<<<1, 1, 0, stream>>>(
            d_status, tolerance, iter);

        updatePKernel<<<gridNodes, BlockSize, 0, stream>>>(
            nodeCount, d_z, d_p, d_status);
    }
}

} // namespace cudaPcg
