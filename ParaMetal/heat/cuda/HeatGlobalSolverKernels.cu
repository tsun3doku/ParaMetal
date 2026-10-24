#include "HeatGlobalSolverKernels.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>



namespace heat::globalsolver {

__global__ void gatherTemperaturesKernel(
    const float* __restrict__ sourceTemp,
    const uint32_t* __restrict__ gatherOffsets,
    const uint32_t* __restrict__ gatherLocalNodeIds,
    const float* __restrict__ gatherWeights,
    const uint32_t* __restrict__ gatherSolverNodeIds,
    uint32_t groupCount,
    float* __restrict__ cellState) {
    const uint32_t group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groupCount) return;
    float temperature = 0.0f;
    for (uint32_t i = gatherOffsets[group]; i < gatherOffsets[group + 1]; ++i) {
        temperature += sourceTemp[gatherLocalNodeIds[i]] * gatherWeights[i];
    }
    cellState[gatherSolverNodeIds[group]] = temperature;
}


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
        if (columnIndices[j] == int(i)) {
            diag = values[j];
            break;
        }
    }
    invDiag[i] = (fabsf(diag) > 1e-12f) ? (1.0f / diag) : 1.0f;
}



__global__ void scatterTemperaturesKernel(
    const float* __restrict__ cellState,
    const uint32_t* __restrict__ scatterSolverNodeIds,
    const uint32_t* __restrict__ scatterLocalNodeIds,
    uint32_t scatterCount,
    float* __restrict__ destination) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= scatterCount) return;
    destination[scatterLocalNodeIds[i]] = cellState[scatterSolverNodeIds[i]];
}

__global__ void accumulateFineAdvectionOutflowKernel(
    uint32_t faceCount,
    const heat::AdvectionFace* __restrict__ faces,
    const float* __restrict__ faceFlux,
    uint32_t boundaryFaceCount,
    const heat::AdvectionBoundaryFace* __restrict__ boundaryFaces,
    const float* __restrict__ boundaryFlux,
    const float* __restrict__ invVolumes,
    float* __restrict__ outflowRates,
    uint32_t* __restrict__ invalid) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < faceCount; i += gridDim.x * blockDim.x) {
        if (faces[i].nodeA == faces[i].nodeB) continue;
        const float phi = faceFlux[i];
        if (!isfinite(phi)) { atomicOr(invalid, 1u); continue; }
        if (phi > 0.0f) {
            atomicAdd(&outflowRates[faces[i].nodeA], phi * invVolumes[faces[i].nodeA]);
        } else if (phi < 0.0f) {
            atomicAdd(&outflowRates[faces[i].nodeB], -phi * invVolumes[faces[i].nodeB]);
        }
    }
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < boundaryFaceCount; i += gridDim.x * blockDim.x) {
        const float phi = boundaryFlux[i];
        if (!isfinite(phi)) { atomicOr(invalid, 1u); continue; }
        if (phi > 0.0f) {
            atomicAdd(&outflowRates[boundaryFaces[i].node], phi * invVolumes[boundaryFaces[i].node]);
        }
    }
}

// Conservative upwind transport across shared fluid faces with monotone updates for CFL <= 1
__global__ void advectFineFacesKernel(
    uint32_t faceCount,
    const heat::AdvectionFace* __restrict__ faces,
    const float* __restrict__ faceFlux,
    float* __restrict__ deltaT,
    const float* __restrict__ x,
    const float* __restrict__ invVolumes,
    const cudaSubcycle::SubcycleTimestep* __restrict__ timestep) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const float phi = faceFlux[idx];
    if (phi == 0.0f) return;

    const float dt = timestep->dt;
    const heat::AdvectionFace face = faces[idx];
    if (face.nodeA == face.nodeB) return;
    const float T_up = (phi >= 0.0f) ? x[face.nodeA] : x[face.nodeB];
    const float fluxVal = phi * T_up * dt;

    atomicAdd(&deltaT[face.nodeA], -fluxVal * invVolumes[face.nodeA]);
    atomicAdd(&deltaT[face.nodeB], +fluxVal * invVolumes[face.nodeB]);
}

__global__ void advectFineBoundaryKernel(
    uint32_t boundaryFaceCount,
    const heat::AdvectionBoundaryFace* __restrict__ boundaryFaces,
    const float* __restrict__ boundaryFlux,
    float* __restrict__ deltaT,
    const float* __restrict__ x,
    const float* __restrict__ inflowTemperatures,
    const float* __restrict__ invVolumes,
    const cudaSubcycle::SubcycleTimestep* __restrict__ timestep) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundaryFaceCount) return;

    const float phi = boundaryFlux[idx];
    if (phi == 0.0f) return;

    const float dt = timestep->dt;
    const heat::AdvectionBoundaryFace face = boundaryFaces[idx];
    if (phi < 0.0f) {
        const float faceTemp = (face.domainFace < 6) ? inflowTemperatures[face.domainFace] : 0.0f;
        atomicAdd(&deltaT[face.node], -phi * faceTemp * dt * invVolumes[face.node]);
    } else {
        atomicAdd(&deltaT[face.node], -phi * x[face.node] * dt * invVolumes[face.node]);
    }
}

__global__ void applyAdvectionUpdatesKernel(
    uint32_t cellCount,
    float* __restrict__ x,
    float* __restrict__ deltaT) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cellCount) return;
    x[idx] += deltaT[idx];
}

// Fills device buffer with quiet NaN sentinels for inactive fixed temperature overrides
__global__ void fillNaNKernel(uint32_t count, float* __restrict__ x) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    x[idx] = __int_as_float(0x7fc00000);  // quiet NaN
}

__global__ void buildCellSourceKernel(
    uint32_t cellCount,
    const float* __restrict__ cellCapacityOverDt,
    const float* __restrict__ cellState,
    const float* __restrict__ cellSourceAdditions,
    float* __restrict__ cellSource) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cellCount) return;
    cellSource[c] = cellCapacityOverDt[c] * cellState[c] + cellSourceAdditions[c];
}

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
    float* __restrict__ traceRhs) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= traceCount) return;

    // Constrained Dirichlet traces are held exactly at T_D with zero cell-source scatter
    if (traceIsDirichlet[t]) {
        traceRhs[t] = traceDirichletValue[t];
        return;
    }

    float sum = traceBoundaryRhs[t];
    const uint32_t startIdx = traceScatterOffsets[t];
    const uint32_t endIdx = traceScatterOffsets[t + 1];

    for (uint32_t idx = startIdx; idx < endIdx; ++idx) {
        const uint32_t cell = traceScatterCells[idx];
        const float weight = traceScatterWeights[idx];
        const float S_E = cellSource[cell];
        const float scale_E = cellRecoveryScale[cell];
        sum += S_E * scale_E * weight;
    }

    traceRhs[t] = sum;
}

__global__ void recoverCellTemperaturesKernel(
    uint32_t cellCount,
    const float* __restrict__ cellRecoveryScale,
    const uint32_t* __restrict__ cellTraceOffsets,
    const uint32_t* __restrict__ cellTraceIndices,
    const float* __restrict__ cellTraceWeights,
    const float* __restrict__ traceSolution,
    const float* __restrict__ cellSource,
    const float* __restrict__ fixedCellTemperatures,
    float* __restrict__ cellState) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cellCount) return;

    // If cell belongs to a prescribed body, enforce its recovered cell state to T_fix exactly
    if (!isnan(fixedCellTemperatures[c])) {
        cellState[c] = fixedCellTemperatures[c];
        return;
    }

    float sum = cellSource[c];
    const uint32_t startIdx = cellTraceOffsets[c];
    const uint32_t endIdx = cellTraceOffsets[c + 1];

    for (uint32_t idx = startIdx; idx < endIdx; ++idx) {
        const uint32_t trace = cellTraceIndices[idx];
        const float weight = cellTraceWeights[idx];
        sum += weight * traceSolution[trace];
    }

    cellState[c] = cellRecoveryScale[c] * sum;
}

} // namespace heat::globalsolver
