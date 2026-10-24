#pragma once

#include "CudaGraphBuilder.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// Shared explicit-substep CFL planning and subcycling loop control
namespace cudaSubcycle {

enum class SubcycleStatus : uint32_t {
    Ok = 0,           // count/dt are valid
    CflOverflow = 1,  // required substeps exceed the caller's cap
    InvalidInput = 2, // non-finite rate/flag, non-positive dt, or bad target
};

struct SubcycleTimestep {
    uint32_t count;   // substeps to perform (valid only when status == Ok)
    float    dt;      // macroDt / count         (valid only when status == Ok)
    SubcycleStatus status;
    uint32_t _pad;
};

// Device kernels and graph helpers visible only in CUDA compilation units
#if defined(__CUDACC__)

// Computes subcycle count and timestep with double-precision ceiling for CFL thresholds
static __global__ void finalizeSubcycleTimestepKernel(
    const float* __restrict__ maxRate,
    const uint32_t* __restrict__ invalidFlag,
    const float* __restrict__ macroDt,
    float targetCfl,
    uint32_t maxCount,
    SubcycleTimestep* __restrict__ out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    SubcycleTimestep ts{};
    ts.status = SubcycleStatus::InvalidInput;

    const float dtMacro = *macroDt;
    if (!(targetCfl > 0.0f) || targetCfl > 1.0f || !isfinite(targetCfl) ||
        !(dtMacro > 0.0f) || !isfinite(dtMacro) || maxCount == 0u) {
        *out = ts;
        return;
    }
    if (invalidFlag != nullptr && *invalidFlag != 0u) {
        *out = ts;
        return;
    }
    const float rate = *maxRate;
    if (!isfinite(rate) || rate < 0.0f) {
        *out = ts;
        return;
    }

    const double required = ceil(static_cast<double>(rate) * static_cast<double>(dtMacro) / static_cast<double>(targetCfl));
    if (!isfinite(required) || required > static_cast<double>(maxCount)) {
        ts.status = SubcycleStatus::CflOverflow;
        *out = ts;
        return;
    }
    const uint32_t requiredCount = (required >= 1.0) ? static_cast<uint32_t>(required) : 1u;

    ts.count = requiredCount;
    ts.dt = dtMacro / static_cast<float>(requiredCount);
    ts.status = SubcycleStatus::Ok;
    *out = ts;
}

// Block max reduction over non-negative rates with invalid-entry flagging
static __global__ void reduceMaxRateKernel(
    uint32_t count,
    const float* __restrict__ rates,
    float* __restrict__ outMax,
    uint32_t* __restrict__ invalidFlag) {
    __shared__ float sharedMax[256];
    float localMax = 0.0f;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        const float rate = rates[i];
        if (!isfinite(rate)) {
            if (invalidFlag != nullptr) atomicOr(invalidFlag, 1u);
            continue;
        }
        localMax = fmaxf(localMax, rate);
    }
    sharedMax[threadIdx.x] = localMax;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedMax[threadIdx.x] = fmaxf(sharedMax[threadIdx.x], sharedMax[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(outMax), __float_as_uint(sharedMax[0]));
    }
}

// Enters subcycle loop only when planner produced a valid timestep
static __global__ void subcycleGateKernel(
    cudaGraphConditionalHandle handle,
    const SubcycleTimestep* __restrict__ timestep) {
    if (threadIdx.x == 0) {
        cudaGraphSetConditional(handle, timestep->status == SubcycleStatus::Ok ? 1u : 0u);
    }
}

static __global__ void subcycleConditionKernel(
    cudaGraphConditionalHandle handle,
    const SubcycleTimestep* __restrict__ timestep,
    uint32_t* __restrict__ iteration) {
    if (threadIdx.x == 0) {
        const uint32_t next = *iteration + 1u;
        *iteration = next;
        cudaGraphSetConditional(handle, next < timestep->count ? 1u : 0u);
    }
}

// Scheduling inputs and outputs for subcycling graph execution
struct SubcycleSchedule {
    const float* maxRate;
    const uint32_t* invalidFlag;
    const float* macroDt;
    float targetCfl;
    uint32_t maxCount;
    SubcycleTimestep* timestep;
    uint32_t* iteration;
};

// Appends planner, iteration reset, gate, and WHILE conditional node
inline cudaError_t beginSubcycleLoop(
    cudaGraph_t graph,
    cudaGraphNode_t& prev,
    const SubcycleSchedule& schedule,
    cudaGraphConditionalHandle& handle,
    cudaGraphBuilder::WhileLoop& loop) {
    cudaError_t result = cudaGraphBuilder::createConditionalHandle(graph, 0u, handle);
    if (result != cudaSuccess) return result;
    result = cudaGraphBuilder::appendKernel(graph, prev, (void*)finalizeSubcycleTimestepKernel, dim3(1), dim3(1),
        schedule.maxRate, schedule.invalidFlag, schedule.macroDt,
        schedule.targetCfl, schedule.maxCount, schedule.timestep);
    if (result != cudaSuccess) return result;
    result = cudaGraphBuilder::appendMemset(graph, prev, schedule.iteration, 0, sizeof(uint32_t));
    if (result != cudaSuccess) return result;
    result = cudaGraphBuilder::appendKernel(graph, prev, (void*)subcycleGateKernel, dim3(1), dim3(1),
        handle, schedule.timestep);
    if (result != cudaSuccess) return result;
    return cudaGraphBuilder::addWhileLoop(graph, &prev, handle, loop);
}

// Appends the continuation check; must be the loop body's last node.
inline cudaError_t endSubcycleLoop(
    cudaGraph_t body,
    cudaGraphNode_t& prev,
    cudaGraphConditionalHandle handle,
    const SubcycleSchedule& schedule) {
    return cudaGraphBuilder::appendKernel(body, prev, (void*)subcycleConditionKernel, dim3(1), dim3(1),
        handle, schedule.timestep, schedule.iteration);
}

#endif // __CUDACC__

} // namespace cudaSubcycle
