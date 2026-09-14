#include "HeatGlobalSolverKernels.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>



namespace heat::globalsolver {

__global__ void gatherTemperaturesKernel(
    const float* __restrict__ sourceTemp,
    const uint32_t* __restrict__ localNodeIds,
    uint32_t nodeCount,
    uint32_t solverNodeOffset,
    float* __restrict__ x) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nodeCount) return;
    const uint32_t localId = localNodeIds[i];
    const uint32_t globalId = solverNodeOffset + i;
    x[globalId] = sourceTemp[localId];
}

__global__ void buildRhsKernel(
    uint32_t totalNodes,
    const float* __restrict__ thermalMasses,
    const float* __restrict__ x,
    const float* __restrict__ rhsAdditions,
    float* __restrict__ rhs) {
    const uint32_t node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= totalNodes) return;
    const float addition = rhsAdditions ? rhsAdditions[node] : 0.0f;
    rhs[node] = thermalMasses[node] * x[node] + addition;
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
    const float* __restrict__ x,
    uint32_t solverNodeOffset,
    const uint32_t* __restrict__ localNodeIds,
    uint32_t count,
    float* __restrict__ destination) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    destination[localNodeIds[i]] = x[solverNodeOffset + i];
}

__global__ void advectFluidKernel(
    uint32_t faceCount,
    const heat::AdvectionFace* __restrict__ faces,
    const float* __restrict__ faceFluxes,
    float* __restrict__ deltaT,
    const float* __restrict__ x,
    float subcycleDt) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const float phi = faceFluxes[idx];
    if (phi == 0.0f) return;

    const heat::AdvectionFace face = faces[idx];
    const float T_up = (phi >= 0.0f) ? x[face.nodeA] : x[face.nodeB];
    const float fluxVal = phi * T_up * subcycleDt;

    atomicAdd(&deltaT[face.nodeA], -fluxVal * face.invVolA);
    atomicAdd(&deltaT[face.nodeB], +fluxVal * face.invVolB);
}

__global__ void advectBoundaryKernel(
    uint32_t boundaryFaceCount,
    const heat::AdvectionBoundaryFace* __restrict__ boundaryFaces,
    const float* __restrict__ boundaryFluxes,
    float* __restrict__ deltaT,
    const float* __restrict__ x,
    float inletTemp,
    float subcycleDt) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundaryFaceCount) return;

    const float phi = boundaryFluxes[idx];
    if (phi == 0.0f) return;

    const heat::AdvectionBoundaryFace bf = boundaryFaces[idx];
    if (phi < 0.0f) {
        // Inflow
        const float volIn = -phi * subcycleDt;
        atomicAdd(&deltaT[bf.node], volIn * inletTemp * bf.invVol);
    } else {
        // Outflow
        const float volOut = phi * subcycleDt;
        const float T = x[bf.node];
        atomicAdd(&deltaT[bf.node], -volOut * T * bf.invVol);
    }
}

__global__ void applyAdvectionUpdatesKernel(
    uint32_t totalNodes,
    float* __restrict__ x,
    float* __restrict__ deltaT) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalNodes) return;

    const float d = deltaT[idx];
    if (d != 0.0f) {
        x[idx] += d;
        deltaT[idx] = 0.0f;
    }
}

} // namespace heat::globalsolver
