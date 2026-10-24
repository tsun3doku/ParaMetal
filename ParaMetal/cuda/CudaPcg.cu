#include "CudaPcg.cuh"
#include "CudaCheck.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cmath>
#include <algorithm>

namespace cg = cooperative_groups;

namespace cudaPcg {

constexpr float AbsoluteResidualFloor = 1e-7f;

static __device__ __forceinline__ void warpReducePair(float& a, float& b) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        a += __shfl_down_sync(0xffffffff, a, offset);
        b += __shfl_down_sync(0xffffffff, b, offset);
    }
}

static __device__ __forceinline__ void blockReducePair(float& a, float& b) {
    __shared__ float sharedA[32];
    __shared__ float sharedB[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    warpReducePair(a, b);
    if (lane == 0) { sharedA[wid] = a; sharedB[wid] = b; }
    __syncthreads();
    const int nWarps = blockDim.x / warpSize;
    a = (threadIdx.x < nWarps) ? sharedA[lane] : 0.0f;
    b = (threadIdx.x < nWarps) ? sharedB[lane] : 0.0f;
    if (wid == 0) warpReducePair(a, b);
}

constexpr uint32_t SubwarpSize = 8;

__global__ void computeInvDiagKernel(
    uint32_t totalNodes,
    const int* __restrict__ rowOffsets,
    const int* __restrict__ columnIndices,
    const float* __restrict__ values,
    float* __restrict__ invDiag) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalNodes) return;
    float diag = 1.0f;
    const int rowStart = rowOffsets[i];
    const int rowEnd = rowOffsets[i + 1];
    for (int j = rowStart; j < rowEnd; ++j) {
        if (columnIndices[j] == static_cast<int>(i)) {
            diag = values[j];
            break;
        }
    }
    invDiag[i] = (fabsf(diag) > 1e-12f) ? (1.0f / diag) : 1.0f;
}

void launchComputeInvDiag(
    uint32_t nodeCount,
    const int* d_rowOffsets,
    const int* d_columnIndices,
    const float* d_values,
    float* d_invDiag,
    cudaStream_t stream) {
    if (nodeCount == 0 || !d_rowOffsets || !d_columnIndices || !d_values || !d_invDiag) return;
    const uint32_t blocks = (nodeCount + BlockSize - 1) / BlockSize;
    computeInvDiagKernel<<<blocks, BlockSize, 0, stream>>>(
        nodeCount, d_rowOffsets, d_columnIndices, d_values, d_invDiag);
}

__global__ void resolveSolutionStatusKernel(
    const PcgStatus* __restrict__ status,
    const uint32_t* __restrict__ scheduleStatus,
    float* __restrict__ solution,
    uint32_t nodeCount,
    int* __restrict__ successFlag) {
    const int solveOk = (status->converged != 0 && status->failed == 0 && status->exhausted == 0) ? 1 : 0;
    const int scheduleOk = (scheduleStatus == nullptr || *scheduleStatus == 0u) ? 1 : 0;
    const int ok = solveOk && scheduleOk;
    if (blockIdx.x == 0 && threadIdx.x == 0 && successFlag) {
        *successFlag = ok;
    }
    if (ok || !solution) return;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nodeCount; i += blockDim.x * gridDim.x) {
        solution[i] = 0.0f;
    }
}

void launchResolveSolutionStatus(
    const PcgStatus* d_status,
    const uint32_t* d_scheduleStatus,
    float* d_solution,
    uint32_t nodeCount,
    int* d_successFlag,
    cudaStream_t stream) {
    if (!d_status || nodeCount == 0) return;
    const uint32_t blocks = (nodeCount + BlockSize - 1) / BlockSize;
    resolveSolutionStatusKernel<<<blocks, BlockSize, 0, stream>>>(
        d_status, d_scheduleStatus, d_solution, nodeCount, d_successFlag);
}

template <bool AccumulatePAp>
static __device__ __forceinline__ void spmvRowsGridStride(
    uint32_t nodeCount,
    const int* __restrict__ rowOffsets,
    const int* __restrict__ columnIndices,
    const float* __restrict__ values,
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    PcgStatus* __restrict__ status,
    uint32_t lane,
    uint32_t subwarp,
    uint32_t globalSubwarp,
    uint32_t totalSubwarps) {
    __shared__ float sharedPAp[BlockSize / SubwarpSize];
    float localPAp = 0.0f;
    for (uint32_t row = globalSubwarp; row < nodeCount; row += totalSubwarps) {
        const int rowStart = rowOffsets[row];
        const int rowEnd = rowOffsets[row + 1];
        float sum = 0.0f;
        for (int j = rowStart + static_cast<int>(lane); j < rowEnd; j += static_cast<int>(SubwarpSize)) {
            sum += values[j] * input[columnIndices[j]];
        }
        const uint32_t laneInWarp = threadIdx.x % warpSize;
        const unsigned int subwarpMask = 0xFFu << (laneInWarp / SubwarpSize * SubwarpSize);
        #pragma unroll
        for (int offset = static_cast<int>(SubwarpSize) / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(subwarpMask, sum, offset);
        }
        if (lane == 0) {
            output[row] = sum;
            if constexpr (AccumulatePAp) localPAp += weight[row] * sum;
        }
    }
    if (lane == 0) sharedPAp[subwarp] = localPAp;
    __syncthreads();
    if (AccumulatePAp && threadIdx.x == 0) {
        const uint32_t subwarpsPerBlock = blockDim.x / SubwarpSize;
        float totalPAp = 0.0f;
        for (uint32_t i = 0; i < subwarpsPerBlock; ++i) {
            totalPAp += sharedPAp[i];
        }
        atomicAdd(&status->pAp, totalPAp);
    }
}

__global__ void pcgPersistentKernel(
    uint32_t nodeCount,
    const int* __restrict__ rowOffsets,
    const int* __restrict__ columnIndices,
    const float* __restrict__ values,
    const float* __restrict__ invDiag,
    float* __restrict__ x,
    const float* __restrict__ b,
    float* __restrict__ Ap,
    float* __restrict__ r,
    float* __restrict__ z,
    float* __restrict__ p,
    PcgStatus* __restrict__ status,
    const PcgConfig* __restrict__ config) {
    auto grid = cg::this_grid();
    const bool leader = (blockIdx.x == 0 && threadIdx.x == 0);

    const uint32_t lane = threadIdx.x % SubwarpSize;
    const uint32_t subwarp = threadIdx.x / SubwarpSize;
    const uint32_t subwarpsPerBlock = blockDim.x / SubwarpSize;
    const uint32_t totalSubwarps = gridDim.x * subwarpsPerBlock;
    const uint32_t globalSubwarp = blockIdx.x * subwarpsPerBlock + subwarp;
    const uint32_t threadCount = blockDim.x * gridDim.x;
    const uint32_t globalThread = blockIdx.x * blockDim.x + threadIdx.x;

    spmvRowsGridStride<false>(nodeCount, rowOffsets, columnIndices, values, x, Ap, nullptr, nullptr,
                              lane, subwarp, globalSubwarp, totalSubwarps);
    grid.sync();

    if (leader) {
        status->converged = 0;
        status->failed = 0;
        status->exhausted = 0;
        status->iterationCount = 0;
        status->rz = 0.0f;
        status->rzNew = 0.0f;
        status->pAp = 0.0f;
        status->residual0 = 0.0f;
        status->residual = 0.0f;
        status->alpha = 0.0f;
        status->beta = 0.0f;
    }
    grid.sync();

    float localRz = 0.0f;
    float localR2 = 0.0f;
    for (uint32_t i = globalThread; i < nodeCount; i += threadCount) {
        const float ri = b[i] - Ap[i];
        const float zi = ri * invDiag[i];
        r[i] = ri;
        z[i] = zi;
        p[i] = zi;
        localRz += ri * zi;
        localR2 += ri * ri;
    }
    blockReducePair(localRz, localR2);
    if (threadIdx.x == 0) {
        atomicAdd(&status->rz, localRz);
        atomicAdd(&status->residual0, localR2);
    }
    grid.sync();

    if (leader) {
        if (!isfinite(status->residual0) || !isfinite(status->rz)) {
            status->failed = 1;
            status->converged = 1;
        } else {
            status->residual0 = sqrtf(fmaxf(status->residual0, 0.0f));
            status->residual = status->residual0;
            if (status->residual0 <= 1e-12f) {
                status->converged = 1;
                status->failed = 0;
            }
        }
    }
    grid.sync();

    for (;;) {
        if (status->converged) break;
        if (static_cast<uint32_t>(status->iterationCount) >= config->maxIterations) {
            if (leader) status->exhausted = 1;
            break;
        }

        spmvRowsGridStride<true>(nodeCount, rowOffsets, columnIndices, values, p, Ap, p, status,
                                 lane, subwarp, globalSubwarp, totalSubwarps);
        grid.sync();

        if (leader && !status->converged) {
            const float denom = status->pAp;
            if (denom <= 1e-24f || !isfinite(denom) || !isfinite(status->rz)) {
                status->failed = 1;
                status->converged = 1;
                status->alpha = 0.0f;
            } else {
                status->alpha = status->rz / denom;
                status->rzNew = 0.0f;
                status->residual = 0.0f;
                status->pAp = 0.0f;
            }
        }
        grid.sync();

        if (!status->converged) {
            const float alpha = status->alpha;
            float localRzNew = 0.0f;
            float localR2 = 0.0f;
            for (uint32_t i = globalThread; i < nodeCount; i += threadCount) {
                const float xi = x[i] + alpha * p[i];
                const float ri = r[i] - alpha * Ap[i];
                const float zi = ri * invDiag[i];
                x[i] = xi;
                r[i] = ri;
                z[i] = zi;
                localRzNew += ri * zi;
                localR2 += ri * ri;
            }
            blockReducePair(localRzNew, localR2);
            if (threadIdx.x == 0) {
                atomicAdd(&status->rzNew, localRzNew);
                atomicAdd(&status->residual, localR2);
            }
        }
        grid.sync();

        if (leader && !status->converged) {
            status->iterationCount += 1;
            if (!isfinite(status->residual) || !isfinite(status->rzNew)) {
                status->failed = 1;
                status->converged = 1;
            } else {
                status->residual = sqrtf(fmaxf(status->residual, 0.0f));
                const float relRes = status->residual / fmaxf(status->residual0, 1e-12f);
                if (relRes <= config->tolerance || status->residual <= AbsoluteResidualFloor) {
                    status->converged = 1;
                } else {
                    const float rzOld = fmaxf(status->rz, 1e-24f);
                    status->beta = status->rzNew / rzOld;
                    status->rz = status->rzNew;
                }
            }
        }
        grid.sync();

        if (status->converged) break;

        const float beta = status->beta;
        for (uint32_t i = globalThread; i < nodeCount; i += threadCount) {
            p[i] = z[i] + beta * p[i];
        }
        grid.sync();
    }
}

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
    PcgConfig* d_config) {
    if (nodeCount == 0) return nullptr;

    const uint32_t rowsPerBlock = BlockSize / SubwarpSize;
    const uint32_t rowBlocks = (nodeCount + rowsPerBlock - 1) / rowsPerBlock;

    int device = 0;
    int smCount = 0;
    int blocksPerSm = 0;
    if (!cudaUtils::cudaCheck(cudaGetDevice(&device), "CudaPcg get device") ||
        !cudaUtils::cudaCheck(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device), "CudaPcg sm count") ||
        !cudaUtils::cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerSm, pcgPersistentKernel, BlockSize, 0), "CudaPcg occupancy")) {
        return nullptr;
    }
    const uint32_t coResidentBlocks = static_cast<uint32_t>(smCount) * static_cast<uint32_t>(blocksPerSm);
    const uint32_t gridBlocks = std::min(rowBlocks, coResidentBlocks);

    cudaStream_t captureStream = nullptr;
    if (!cudaUtils::cudaCheck(cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking), "CudaPcg capture stream")) {
        return nullptr;
    }

    void* kernelArgs[] = {
        &nodeCount,
        &d_rowOffsets,
        &d_columnIndices,
        &d_values,
        &d_invDiag,
        &d_x,
        &d_b,
        &d_Ap,
        &d_r,
        &d_z,
        &d_p,
        &d_status,
        &d_config,
    };

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool ok =
        cudaUtils::cudaCheck(cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeRelaxed), "CudaPcg begin capture") &&
        cudaUtils::cudaCheck(cudaLaunchCooperativeKernel(
            (const void*)pcgPersistentKernel, dim3(gridBlocks), dim3(BlockSize), kernelArgs, 0, captureStream), "CudaPcg coop launch") &&
        cudaUtils::cudaCheck(cudaStreamEndCapture(captureStream, &graph), "CudaPcg end capture");
    if (ok) {
        ok = cudaUtils::cudaCheck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "CudaPcg instantiate");
    }
    if (graph) cudaGraphDestroy(graph);
    cudaStreamDestroy(captureStream);
    return ok ? exec : nullptr;
}

} // namespace cudaPcg
