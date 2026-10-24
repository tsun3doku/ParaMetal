#pragma once

#include <cuda_runtime.h>

// Host-only mechanical primitives for building CUDA graphs
namespace cudaGraphBuilder {

// Appends a kernel node, threading dependency and advancing prev on success
template <typename... Args>
inline cudaError_t appendKernel(
    cudaGraph_t graph,
    cudaGraphNode_t& prev,
    void* func,
    dim3 grid,
    dim3 block,
    Args... args) {
    static_assert(sizeof...(Args) > 0, "appendKernel requires at least one argument");
    void* params[] = { &args... };

    cudaKernelNodeParams p{};
    p.func = func;
    p.gridDim = grid;
    p.blockDim = block;
    p.sharedMemBytes = 0;
    p.kernelParams = params;
    p.extra = nullptr;

    cudaGraphNode_t node = nullptr;
    const cudaError_t result = cudaGraphAddKernelNode(
        &node, graph, prev ? &prev : nullptr, prev ? 1 : 0, &p);
    if (result == cudaSuccess) prev = node;
    return result;
}

struct WhileLoop {
    cudaGraph_t body = nullptr;
    cudaGraphNode_t node = nullptr;
};

// Graph memset node (byte value). Same threading contract as appendKernel.
inline cudaError_t appendMemset(
    cudaGraph_t graph,
    cudaGraphNode_t& prev,
    void* dst,
    unsigned int value,
    size_t bytes) {
    cudaMemsetParams p{};
    p.dst = dst;
    p.value = value;
    p.elementSize = 1;
    p.pitch = 0;
    p.width = bytes;
    p.height = 1;

    cudaGraphNode_t node = nullptr;
    const cudaError_t result = cudaGraphAddMemsetNode(
        &node, graph, prev ? &prev : nullptr, prev ? 1 : 0, &p);
    if (result == cudaSuccess) prev = node;
    return result;
}

// Creates a WHILE conditional handle initialised to defaultLaunchValue each launch
inline cudaError_t createConditionalHandle(
    cudaGraph_t graph,
    unsigned int defaultLaunchValue,
    cudaGraphConditionalHandle& outHandle) {
    return cudaGraphConditionalHandleCreate(
        &outHandle, graph, defaultLaunchValue, cudaGraphCondAssignDefault);
}

// Adds the WHILE conditional node for handle and exposes its body graph
inline cudaError_t addWhileLoop(
    cudaGraph_t graph,
    const cudaGraphNode_t* dependency,
    cudaGraphConditionalHandle handle,
    WhileLoop& out) {
    cudaGraphNodeParams cParams = {};
    cParams.type = cudaGraphNodeTypeConditional;
    cParams.conditional.handle = handle;
    cParams.conditional.type   = cudaGraphCondTypeWhile;
    cParams.conditional.size   = 1;

    cudaError_t err = cudaGraphAddNode(
        &out.node, graph, dependency, dependency ? 1 : 0, &cParams);
    if (err != cudaSuccess) return err;

    out.body = cParams.conditional.phGraph_out[0];
    return out.body ? cudaSuccess : cudaErrorUnknown;
}

} // namespace cudaGraphBuilder
