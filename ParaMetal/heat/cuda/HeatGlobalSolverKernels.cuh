#pragma once

#include "../HeatGpuStructs.hpp"
#include "../../cuda/SubcycleTimestep.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace heat::globalsolver {

constexpr uint32_t BlockSize = 256;

__global__ void gatherTemperaturesKernel(
    const float* __restrict__ sourceTemp,
    const uint32_t* __restrict__ gatherOffsets,
    const uint32_t* __restrict__ gatherLocalNodeIds,
    const float* __restrict__ gatherWeights,
    const uint32_t* __restrict__ gatherSolverNodeIds,
    uint32_t groupCount,
    float* __restrict__ cellState);


__global__ void computeInvDiagKernel(
    uint32_t totalNodes,
    const int* __restrict__ rowOffsets,
    const int* __restrict__ columnIndices,
    const float* __restrict__ values,
    float* __restrict__ invDiag);

__global__ void scatterTemperaturesKernel(
    const float* __restrict__ cellState,
    const uint32_t* __restrict__ scatterSolverNodeIds,
    const uint32_t* __restrict__ scatterLocalNodeIds,
    uint32_t scatterCount,
    float* __restrict__ destination);

// Accumulates per-cell advective outflow rate sum(|phi|)/V and flags non-finite fluxes
__global__ void accumulateFineAdvectionOutflowKernel(
    uint32_t faceCount,
    const heat::AdvectionFace* __restrict__ faces,
    const float* __restrict__ faceFlux,
    uint32_t boundaryFaceCount,
    const heat::AdvectionBoundaryFace* __restrict__ boundaryFaces,
    const float* __restrict__ boundaryFlux,
    const float* __restrict__ invVolumes,
    float* __restrict__ outflowRates,
    uint32_t* __restrict__ invalid);

__global__ void advectFineFacesKernel(
    uint32_t faceCount,
    const heat::AdvectionFace* __restrict__ faces,
    const float* __restrict__ faceFlux,
    float* __restrict__ deltaT,
    const float* __restrict__ x,
    const float* __restrict__ invVolumes,
    const cudaSubcycle::SubcycleTimestep* __restrict__ timestep);

__global__ void advectFineBoundaryKernel(
    uint32_t boundaryFaceCount,
    const heat::AdvectionBoundaryFace* __restrict__ boundaryFaces,
    const float* __restrict__ boundaryFlux,
    float* __restrict__ deltaT,
    const float* __restrict__ x,
    const float* __restrict__ inflowTemperatures,
    const float* __restrict__ invVolumes,
    const cudaSubcycle::SubcycleTimestep* __restrict__ timestep);

__global__ void applyAdvectionUpdatesKernel(
    uint32_t cellCount,
    float* __restrict__ x,
    float* __restrict__ deltaT);

__global__ void fillNaNKernel(uint32_t count, float* __restrict__ x);

__global__ void buildCellSourceKernel(
    uint32_t cellCount,
    const float* __restrict__ cellCapacityOverDt,
    const float* __restrict__ cellState,
    const float* __restrict__ cellSourceAdditions,
    float* __restrict__ cellSource);

__global__ void assembleTraceRhsKernel(
    uint32_t traceCount,
    const uint32_t* __restrict__ traceScatterOffsets,
    const uint32_t* __restrict__ traceScatterCells,
    const float* __restrict__ traceScatterWeights,
    const float* __restrict__ cellSource,
    const float* __restrict__ cellRecoveryScale,
    const float* __restrict__ traceBoundaryRhs,
    const uint8_t* __restrict__ traceIsDirichlet,
    const float* __restrict__ traceDirichletValue,
    float* __restrict__ traceRhs);

__global__ void recoverCellTemperaturesKernel(
    uint32_t cellCount,
    const float* __restrict__ cellRecoveryScale,
    const uint32_t* __restrict__ cellTraceOffsets,
    const uint32_t* __restrict__ cellTraceIndices,
    const float* __restrict__ cellTraceWeights,
    const float* __restrict__ traceSolution,
    const float* __restrict__ cellSource,
    const float* __restrict__ fixedCellTemperatures,
    float* __restrict__ cellState);

} // namespace heat::globalsolver
