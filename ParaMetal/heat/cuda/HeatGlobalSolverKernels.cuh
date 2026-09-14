#pragma once

#include "../HeatGpuStructs.hpp"

#include <cuda_runtime.h>
#include <cstdint>

namespace heat::globalsolver {

constexpr uint32_t BlockSize = 256;
constexpr uint32_t MaxIterations = 100;
constexpr float ConvergenceTolerance = 1e-9f;

__global__ void gatherTemperaturesKernel(
    const float* __restrict__ sourceTemp,
    const uint32_t* __restrict__ localNodeIds,
    uint32_t nodeCount,
    uint32_t solverNodeOffset,
    float* __restrict__ x);

__global__ void buildRhsKernel(
    uint32_t totalNodes,
    const float* __restrict__ thermalMasses,
    const float* __restrict__ x,
    const float* __restrict__ rhsAdditions,
    float* __restrict__ rhs);

__global__ void computeInvDiagKernel(
    uint32_t totalNodes,
    const int* __restrict__ rowOffsets,
    const int* __restrict__ columnIndices,
    const float* __restrict__ values,
    float* __restrict__ invDiag);

__global__ void scatterTemperaturesKernel(
    const float* __restrict__ x,
    uint32_t solverNodeOffset,
    const uint32_t* __restrict__ localNodeIds,
    uint32_t count,
    float* __restrict__ destination);

__global__ void advectFluidKernel(
    uint32_t faceCount,
    const heat::AdvectionFace* __restrict__ faces,
    const float* __restrict__ faceFluxes,
    float* __restrict__ deltaT,
    const float* __restrict__ x,
    float subcycleDt);

__global__ void advectBoundaryKernel(
    uint32_t boundaryFaceCount,
    const heat::AdvectionBoundaryFace* __restrict__ boundaryFaces,
    const float* __restrict__ boundaryFluxes,
    float* __restrict__ deltaT,
    const float* __restrict__ x,
    float inletTemp,
    float subcycleDt);

__global__ void applyAdvectionUpdatesKernel(
    uint32_t totalNodes,
    float* __restrict__ x,
    float* __restrict__ deltaT);

} // namespace heat::globalsolver
