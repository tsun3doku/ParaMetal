#include "CFDSolverKernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cfd::fluidkernels {

namespace {

__global__ void buildPressureRhsFacesKernel(
    uint32_t faceCount,
    const CFDFaceGpu* __restrict__ faces,
    const float* __restrict__ faceFlux,
    float* __restrict__ pressureRhs) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const CFDFaceGpu face = faces[idx];
    const float phi = faceFlux[idx];

    atomicAdd(&pressureRhs[face.cellA], -phi);
    atomicAdd(&pressureRhs[face.cellB],  phi);
}

__global__ void buildPressureRhsBoundaryKernel(
    uint32_t boundaryFaceCount,
    const CFDBoundaryFaceGpu* __restrict__ boundaryFaces,
    const float* __restrict__ boundaryFlux,
    float* __restrict__ pressureRhs) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundaryFaceCount) return;

    const CFDBoundaryFaceGpu bf = boundaryFaces[idx];
    atomicAdd(&pressureRhs[bf.cell], -boundaryFlux[idx]);
}

__global__ void correctFaceFluxesKernel(
    uint32_t faceCount,
    const CFDFaceGpu* __restrict__ faces,
    const float* __restrict__ psi,
    float* __restrict__ faceFlux) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const CFDFaceGpu face = faces[idx];
    const float psiA = psi[face.cellA];
    const float psiB = psi[face.cellB];

    faceFlux[idx] -= face.projectionWeight * (psiB - psiA);
}

__global__ void correctBoundaryFluxesKernel(
    uint32_t boundaryFaceCount,
    const CFDBoundaryFaceGpu* __restrict__ boundaryFaces,
    const BoundaryConditionGpu* __restrict__ boundaryConditions,
    const float* __restrict__ psi,
    float* __restrict__ boundaryFlux) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundaryFaceCount) return;

    const CFDBoundaryFaceGpu bf = boundaryFaces[idx];
    const BoundaryConditionGpu bc = boundaryConditions[bf.boundaryCondition];
    if (bc.type == 2 /* PressureOutlet */) {
        boundaryFlux[idx] += bf.projectionWeight * psi[bf.cell];
    }
}

__global__ void computeOutflowFacesKernel(
    uint32_t faceCount,
    const CFDFaceGpu* __restrict__ faces,
    const float* __restrict__ faceFlux,
    float* __restrict__ outflowRates,
    uint32_t* __restrict__ invalid) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const CFDFaceGpu face = faces[idx];
    const float phi = faceFlux[idx];

    if (!isfinite(phi)) {
        if (invalid != nullptr) atomicOr(invalid, 1u);
        return;
    }
    if (phi > 0.0f) {
        atomicAdd(&outflowRates[face.cellA], phi);
    } else if (phi < 0.0f) {
        atomicAdd(&outflowRates[face.cellB], -phi);
    }
}

__global__ void computeOutflowBoundaryKernel(
    uint32_t boundaryFaceCount,
    const CFDBoundaryFaceGpu* __restrict__ boundaryFaces,
    const float* __restrict__ boundaryFlux,
    float* __restrict__ outflowRates,
    uint32_t* __restrict__ invalid) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundaryFaceCount) return;

    const CFDBoundaryFaceGpu bf = boundaryFaces[idx];
    const float phi = boundaryFlux[idx];
    if (!isfinite(phi)) {
        if (invalid != nullptr) atomicOr(invalid, 1u);
        return;
    }
    if (phi > 0.0f) {
        atomicAdd(&outflowRates[bf.cell], phi);
    }
}

__global__ void finalizeOutflowRateKernel(
    uint32_t cellCount,
    const float* __restrict__ cellVolumes,
    float* __restrict__ outflowRates,
    uint32_t* __restrict__ invalid) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cellCount) return;

    const float vol = cellVolumes[idx];
    if (vol > 0.0f && isfinite(vol)) {
        outflowRates[idx] /= vol;
    } else {
        outflowRates[idx] = 0.0f;
        if (invalid != nullptr) atomicOr(invalid, 1u);
    }
}

__global__ void zeroMaskedKernel(
    uint32_t cellCount,
    const uint8_t* __restrict__ mask,
    float* __restrict__ buf) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cellCount && mask[idx] != 0) {
        buf[idx] = 0.0f;
    }
}

__global__ void initializeVelocityKernel(
    uint32_t cellCount,
    glm::vec3 initialVelocity,
    glm::vec4* __restrict__ velocity,
    glm::vec3* __restrict__ velocityStar) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < cellCount) {
        velocity[idx] = glm::vec4(initialVelocity, 0.0f);
        if (velocityStar) {
            velocityStar[idx] = initialVelocity;
        }
    }
}

__global__ void momentumInternalFacesKernel(
    uint32_t faceCount,
    const CFDFaceGpu* __restrict__ faces,
    const float* __restrict__ faceFlux,
    const glm::vec4* __restrict__ velocity,
    const float* __restrict__ kinematicViscosity,
    glm::vec3* __restrict__ momentumAccum) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const CFDFaceGpu face = faces[idx];
    const float phi = faceFlux[idx];
    const glm::vec3 uA = glm::vec3(velocity[face.cellA]);
    const glm::vec3 uB = glm::vec3(velocity[face.cellB]);

    const glm::vec3 uUpwind = (phi >= 0.0f) ? uA : uB;
    const glm::vec3 convection = phi * uUpwind;
    const glm::vec3 diffusion = ((*kinematicViscosity) * face.projectionWeight) * (uB - uA);

    const glm::vec3 netA = -convection + diffusion;
    const glm::vec3 netB = convection - diffusion;

    atomicAdd(&momentumAccum[face.cellA].x, netA.x);
    atomicAdd(&momentumAccum[face.cellA].y, netA.y);
    atomicAdd(&momentumAccum[face.cellA].z, netA.z);

    atomicAdd(&momentumAccum[face.cellB].x, netB.x);
    atomicAdd(&momentumAccum[face.cellB].y, netB.y);
    atomicAdd(&momentumAccum[face.cellB].z, netB.z);
}

__global__ void momentumBoundaryFacesKernel(
    uint32_t boundaryFaceCount,
    const CFDBoundaryFaceGpu* __restrict__ boundaryFaces,
    const BoundaryConditionGpu* __restrict__ boundaryConditions,
    const float* __restrict__ boundaryFlux,
    const glm::vec4* __restrict__ velocity,
    const float* __restrict__ kinematicViscosity,
    glm::vec3* __restrict__ momentumAccum) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundaryFaceCount) return;

    const CFDBoundaryFaceGpu bf = boundaryFaces[idx];
    const float phi = boundaryFlux[idx];
    const BoundaryConditionGpu bc = boundaryConditions[bf.boundaryCondition];
    const glm::vec3 uCell = glm::vec3(velocity[bf.cell]);
    const float viscosity = *kinematicViscosity;

    glm::vec3 convection(0.0f);
    glm::vec3 diffusion(0.0f);

    if (bc.type == 1 /* VelocityInlet */) {
        const glm::vec3 uInlet = bc.velocity;
        const glm::vec3 uUpwind = (phi < 0.0f) ? uInlet : uCell;
        convection = phi * uUpwind;
        diffusion = (viscosity * bf.projectionWeight) * (uInlet - uCell);
    } else if (bc.type == 0 /* NoSlipWall / EmbeddedWall */) {
        convection = glm::vec3(0.0f);
        diffusion = (viscosity * bf.projectionWeight) * (-uCell);
    } else if (bc.type == 2 /* PressureOutlet */) {
        convection = phi * uCell;
        diffusion = glm::vec3(0.0f);
    }

    const glm::vec3 net = -convection + diffusion;
    atomicAdd(&momentumAccum[bf.cell].x, net.x);
    atomicAdd(&momentumAccum[bf.cell].y, net.y);
    atomicAdd(&momentumAccum[bf.cell].z, net.z);
}

__global__ void applyMomentumPredictorKernel(
    uint32_t cellCount,
    const float* __restrict__ cellVolumes,
    const float* __restrict__ cellViscousWeights,
    const glm::vec4* __restrict__ velocity,
    const glm::vec3* __restrict__ momentumAccum,
    glm::vec3* __restrict__ velocityStar,
    const float* __restrict__ kinematicViscosity,
    const cudaSubcycle::SubcycleTimestep* __restrict__ timestep) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cellCount) return;

    const float dt = timestep->dt;
    const float vol = cellVolumes[idx];
    const float viscWeight = cellViscousWeights ? cellViscousWeights[idx] : 0.0f;
    const float effVol = vol + (dt * (*kinematicViscosity)) * viscWeight;
    const float invEffVol = (effVol > 0.0f) ? (1.0f / effVol) : 0.0f;
    const glm::vec3 uOld = glm::vec3(velocity[idx]);
    const glm::vec3 net = momentumAccum[idx];

    velocityStar[idx] = uOld + (dt * invEffVol) * net;
}

__global__ void provisionalInternalFluxKernel(
    uint32_t faceCount,
    const CFDFaceGpu* __restrict__ faces,
    const glm::vec3* __restrict__ velocityStar,
    float* __restrict__ faceFlux) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const CFDFaceGpu face = faces[idx];
    const glm::vec3 uA = velocityStar[face.cellA];
    const glm::vec3 uB = velocityStar[face.cellB];

    faceFlux[idx] = glm::dot(uA, face.fluxVectorA) + glm::dot(uB, face.fluxVectorB);
}

__global__ void provisionalBoundaryFluxKernel(
    uint32_t boundaryFaceCount,
    const CFDBoundaryFaceGpu* __restrict__ boundaryFaces,
    const BoundaryConditionGpu* __restrict__ boundaryConditions,
    const glm::vec3* __restrict__ velocityStar,
    float* __restrict__ boundaryFlux) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundaryFaceCount) return;

    const CFDBoundaryFaceGpu bf = boundaryFaces[idx];
    const BoundaryConditionGpu bc = boundaryConditions[bf.boundaryCondition];
    if (bc.type == 1 /* VelocityInlet */) {
        boundaryFlux[idx] = glm::dot(bc.velocity, bf.areaVector);
    } else if (bc.type == 2 /* PressureOutlet */) {
        const glm::vec3 u = velocityStar[bf.cell];
        boundaryFlux[idx] = glm::dot(u, bf.areaVector);
    } else {
        boundaryFlux[idx] = 0.0f;
    }
}

__global__ void accumulateFacePressureImpulseKernel(
    uint32_t faceCount,
    const CFDFaceGpu* __restrict__ faces,
    const float* __restrict__ psi,
    glm::vec3* __restrict__ pressureImpulse) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const CFDFaceGpu face = faces[idx];
    // Difference-based pressure gradient
    const float dp = psi[face.cellB] - psi[face.cellA];
    const glm::vec3 impulseA = dp * face.fluxVectorB;
    const glm::vec3 impulseB = dp * face.fluxVectorA;

    atomicAdd(&pressureImpulse[face.cellA].x, impulseA.x);
    atomicAdd(&pressureImpulse[face.cellA].y, impulseA.y);
    atomicAdd(&pressureImpulse[face.cellA].z, impulseA.z);

    atomicAdd(&pressureImpulse[face.cellB].x, impulseB.x);
    atomicAdd(&pressureImpulse[face.cellB].y, impulseB.y);
    atomicAdd(&pressureImpulse[face.cellB].z, impulseB.z);
}

__global__ void accumulateBoundaryPressureImpulseKernel(
    uint32_t boundaryFaceCount,
    const CFDBoundaryFaceGpu* __restrict__ boundaryFaces,
    const BoundaryConditionGpu* __restrict__ boundaryConditions,
    const float* __restrict__ psi,
    glm::vec3* __restrict__ pressureImpulse) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundaryFaceCount) return;

    const CFDBoundaryFaceGpu bf = boundaryFaces[idx];
    const BoundaryConditionGpu bc = boundaryConditions[bf.boundaryCondition];
    // Dirichlet pressure outlet reference contribution (psi = 0)
    if (bc.type != 2 /* PressureOutlet */) return;

    const float psiVal = psi[bf.cell];
    const glm::vec3 term = -psiVal * bf.areaVector;

    atomicAdd(&pressureImpulse[bf.cell].x, term.x);
    atomicAdd(&pressureImpulse[bf.cell].y, term.y);
    atomicAdd(&pressureImpulse[bf.cell].z, term.z);
}

__global__ void applyVelocityCorrectionKernel(
    uint32_t cellCount,
    const float* __restrict__ cellVolumes,
    const glm::vec3* __restrict__ pressureImpulse,
    const glm::vec3* __restrict__ velocityStar,
    glm::vec4* __restrict__ velocity) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cellCount) return;

    const float vol = cellVolumes[idx];
    const float invVol = (vol > 0.0f) ? (1.0f / vol) : 0.0f;
    const glm::vec3 deltaU = pressureImpulse[idx] * invVol;
    const glm::vec3 uNew = velocityStar[idx] - deltaU;
    velocity[idx] = glm::vec4(uNew, 0.0f);
}


} // anonymous namespace

void launchBuildPressureRhs(
    uint32_t cellCount,
    uint32_t faceCount,
    uint32_t boundaryFaceCount,
    const CFDFaceGpu* d_faces,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const float* d_faceFlux,
    const float* d_boundaryFlux,
    float* d_pressureRhs,
    cudaStream_t stream) {
    if (cellCount == 0 || !d_pressureRhs) return;

    cudaMemsetAsync(d_pressureRhs, 0, cellCount * sizeof(float), stream);

    if (faceCount > 0 && d_faces && d_faceFlux) {
        const uint32_t blocks = (faceCount + BlockSize - 1) / BlockSize;
        buildPressureRhsFacesKernel<<<blocks, BlockSize, 0, stream>>>(
            faceCount, d_faces, d_faceFlux, d_pressureRhs);
    }

    if (boundaryFaceCount > 0 && d_boundaryFaces && d_boundaryFlux) {
        const uint32_t blocks = (boundaryFaceCount + BlockSize - 1) / BlockSize;
        buildPressureRhsBoundaryKernel<<<blocks, BlockSize, 0, stream>>>(
            boundaryFaceCount, d_boundaryFaces, d_boundaryFlux, d_pressureRhs);
    }
}

void launchCorrectFluxes(
    uint32_t faceCount,
    const CFDFaceGpu* d_faces,
    const float* d_psi,
    float* d_faceFlux,
    cudaStream_t stream) {
    if (faceCount == 0 || !d_faces || !d_psi || !d_faceFlux) return;

    const uint32_t blocks = (faceCount + BlockSize - 1) / BlockSize;
    correctFaceFluxesKernel<<<blocks, BlockSize, 0, stream>>>(
        faceCount, d_faces, d_psi, d_faceFlux);
}

void launchCorrectBoundaryFluxes(
    uint32_t boundaryFaceCount,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const BoundaryConditionGpu* d_boundaryConditions,
    const float* d_psi,
    float* d_boundaryFlux,
    cudaStream_t stream) {
    if (boundaryFaceCount == 0 || !d_boundaryFaces || !d_boundaryConditions || !d_psi || !d_boundaryFlux) return;

    const uint32_t blocks = (boundaryFaceCount + BlockSize - 1) / BlockSize;
    correctBoundaryFluxesKernel<<<blocks, BlockSize, 0, stream>>>(
        boundaryFaceCount, d_boundaryFaces, d_boundaryConditions, d_psi, d_boundaryFlux);
}

void launchComputeOutflowRate(
    uint32_t cellCount,
    uint32_t faceCount,
    uint32_t boundaryFaceCount,
    const CFDFaceGpu* d_faces,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const float* d_faceFlux,
    const float* d_boundaryFlux,
    const float* d_cellVolumes,
    float* d_outflowRates,
    cudaStream_t stream) {
    if (cellCount == 0 || !d_outflowRates || !d_cellVolumes) return;

    cudaMemsetAsync(d_outflowRates, 0, cellCount * sizeof(float), stream);

    if (faceCount > 0 && d_faces && d_faceFlux) {
        const uint32_t blocks = (faceCount + BlockSize - 1) / BlockSize;
        computeOutflowFacesKernel<<<blocks, BlockSize, 0, stream>>>(
            faceCount, d_faces, d_faceFlux, d_outflowRates, nullptr);
    }

    if (boundaryFaceCount > 0 && d_boundaryFaces && d_boundaryFlux) {
        const uint32_t blocks = (boundaryFaceCount + BlockSize - 1) / BlockSize;
        computeOutflowBoundaryKernel<<<blocks, BlockSize, 0, stream>>>(
            boundaryFaceCount, d_boundaryFaces, d_boundaryFlux, d_outflowRates, nullptr);
    }

    const uint32_t blocks = (cellCount + BlockSize - 1) / BlockSize;
    finalizeOutflowRateKernel<<<blocks, BlockSize, 0, stream>>>(
        cellCount, d_cellVolumes, d_outflowRates, nullptr);
}

__global__ void argmaxOutflowRateKernel(
    uint32_t cellCount,
    const float* __restrict__ rates,
    unsigned long long* __restrict__ packedMax) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cellCount) return;

    const float rate = rates[idx];
    if (!(rate > 0.0f)) return;

    const unsigned long long packed =
        (static_cast<unsigned long long>(__float_as_uint(rate)) << 32) | static_cast<unsigned long long>(idx);
    atomicMax(packedMax, packed);
}

__global__ void advanceVelocityKernel(
    uint32_t cellCount,
    const glm::vec3* __restrict__ velocityStar,
    glm::vec4* __restrict__ velocity,
    const cudaSubcycle::SubcycleTimestep* __restrict__ timestep,
    const uint32_t* __restrict__ iteration) {
    if (*iteration + 1u >= timestep->count) return;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cellCount) return;
    velocity[idx] = glm::vec4(velocityStar[idx], 0.0f);
}

void launchCflArgmax(
    uint32_t cellCount,
    const float* d_outflowRates,
    unsigned long long* d_packedMax,
    cudaStream_t stream) {
    if (cellCount == 0 || !d_outflowRates || !d_packedMax) return;
    const uint32_t blocks = (cellCount + BlockSize - 1) / BlockSize;
    argmaxOutflowRateKernel<<<blocks, BlockSize, 0, stream>>>(cellCount, d_outflowRates, d_packedMax);
}

void launchZeroMasked(
    uint32_t cellCount,
    const uint8_t* d_isPinned,
    float* d_buf,
    cudaStream_t stream) {
    if (cellCount == 0 || !d_isPinned || !d_buf) return;
    const uint32_t blocks = (cellCount + BlockSize - 1) / BlockSize;
    zeroMaskedKernel<<<blocks, BlockSize, 0, stream>>>(cellCount, d_isPinned, d_buf);
}

void launchInitializeVelocity(
    uint32_t cellCount,
    const glm::vec3& initialVelocity,
    glm::vec4* d_velocity,
    glm::vec3* d_velocityStar,
    cudaStream_t stream) {
    if (cellCount == 0 || !d_velocity) return;
    const uint32_t blocks = (cellCount + BlockSize - 1) / BlockSize;
    initializeVelocityKernel<<<blocks, BlockSize, 0, stream>>>(
        cellCount, initialVelocity, d_velocity, d_velocityStar);
}

void launchProvisionalFlux(
    uint32_t faceCount,
    uint32_t boundaryFaceCount,
    const CFDFaceGpu* d_faces,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const BoundaryConditionGpu* d_boundaryConditions,
    const glm::vec3* d_velocityStar,
    float* d_faceFlux,
    float* d_boundaryFlux,
    cudaStream_t stream) {
    if (faceCount > 0 && d_faces && d_velocityStar && d_faceFlux) {
        const uint32_t blocks = (faceCount + BlockSize - 1) / BlockSize;
        provisionalInternalFluxKernel<<<blocks, BlockSize, 0, stream>>>(
            faceCount, d_faces, d_velocityStar, d_faceFlux);
    }

    if (boundaryFaceCount > 0 && d_boundaryFaces && d_boundaryConditions && d_velocityStar && d_boundaryFlux) {
        const uint32_t blocks = (boundaryFaceCount + BlockSize - 1) / BlockSize;
        provisionalBoundaryFluxKernel<<<blocks, BlockSize, 0, stream>>>(
            boundaryFaceCount, d_boundaryFaces, d_boundaryConditions, d_velocityStar, d_boundaryFlux);
    }
}

void launchCorrectVelocity(
    uint32_t cellCount,
    uint32_t faceCount,
    uint32_t boundaryFaceCount,
    const CFDFaceGpu* d_faces,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const BoundaryConditionGpu* d_boundaryConditions,
    const float* d_cellVolumes,
    const float* d_psi,
    const glm::vec3* d_velocityStar,
    glm::vec4* d_velocity,
    glm::vec3* d_pressureImpulse,
    cudaStream_t stream) {
    if (cellCount == 0 || !d_cellVolumes || !d_psi || !d_velocityStar || !d_velocity || !d_pressureImpulse) return;

    cudaMemsetAsync(d_pressureImpulse, 0, cellCount * sizeof(glm::vec3), stream);

    if (faceCount > 0 && d_faces) {
        const uint32_t blocks = (faceCount + BlockSize - 1) / BlockSize;
        accumulateFacePressureImpulseKernel<<<blocks, BlockSize, 0, stream>>>(
            faceCount, d_faces, d_psi, d_pressureImpulse);
    }

    if (boundaryFaceCount > 0 && d_boundaryFaces && d_boundaryConditions) {
        const uint32_t blocks = (boundaryFaceCount + BlockSize - 1) / BlockSize;
        accumulateBoundaryPressureImpulseKernel<<<blocks, BlockSize, 0, stream>>>(
            boundaryFaceCount, d_boundaryFaces, d_boundaryConditions, d_psi, d_pressureImpulse);
    }

    const uint32_t nodeBlocks = (cellCount + BlockSize - 1) / BlockSize;
    applyVelocityCorrectionKernel<<<nodeBlocks, BlockSize, 0, stream>>>(
        cellCount, d_cellVolumes, d_pressureImpulse, d_velocityStar, d_velocity);
}


__global__ void publishFineFaceFluxKernel(
    uint32_t faceCount,
    const uint32_t* __restrict__ globalIndex,
    const float* __restrict__ sourceFlux,
    float* __restrict__ outFlux,
    const int* __restrict__ publishGate) {
    if (publishGate && *publishGate == 0) return;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= faceCount) return;
    outFlux[globalIndex[i]] = sourceFlux[i];
}

__global__ void publishFineBoundaryFluxKernel(
    uint32_t faceCount,
    const uint32_t* __restrict__ globalIndex,
    const float* __restrict__ sourceFlux,
    float* __restrict__ outFlux,
    const int* __restrict__ publishGate) {
    if (publishGate && *publishGate == 0) return;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= faceCount) return;
    const uint32_t dst = globalIndex[i];
    if (dst == 0xFFFFFFFFu) return; // embedded solid wall: not in the published boundary space
    outFlux[dst] = sourceFlux[i];
}

void publishFineFaceFluxes(
    uint32_t faceCount,
    const uint32_t* d_faceGlobalIndex,
    const float* d_faceFlux,
    float* canonicalFaceFlux,
    const int* d_publishGate,
    cudaStream_t stream) {
    if (faceCount == 0 || !d_faceGlobalIndex || !d_faceFlux || !canonicalFaceFlux) return;
    const uint32_t blocks = (faceCount + BlockSize - 1) / BlockSize;
    publishFineFaceFluxKernel<<<blocks, BlockSize, 0, stream>>>(
        faceCount, d_faceGlobalIndex, d_faceFlux, canonicalFaceFlux, d_publishGate);
}

void publishFineBoundaryFluxes(
    uint32_t boundaryFaceCount,
    const uint32_t* d_boundaryGlobalIndex,
    const float* d_boundaryFlux,
    float* canonicalBoundaryFlux,
    const int* d_publishGate,
    cudaStream_t stream) {
    if (boundaryFaceCount == 0 || !d_boundaryGlobalIndex || !d_boundaryFlux || !canonicalBoundaryFlux) return;
    const uint32_t blocks = (boundaryFaceCount + BlockSize - 1) / BlockSize;
    publishFineBoundaryFluxKernel<<<blocks, BlockSize, 0, stream>>>(
        boundaryFaceCount, d_boundaryGlobalIndex, d_boundaryFlux, canonicalBoundaryFlux, d_publishGate);
}

__global__ void publishVelocityKernel(
    uint32_t nodeCount,
    const glm::vec4* __restrict__ source,
    glm::vec4* __restrict__ destination,
    const int* __restrict__ publishGate) {
    if (publishGate && *publishGate == 0) return;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nodeCount) return;
    destination[i] = source[i];
}

void publishVelocity(
    uint32_t nodeCount,
    const glm::vec4* d_source,
    glm::vec4* d_destination,
    const int* d_publishGate,
    cudaStream_t stream) {
    if (nodeCount == 0 || !d_source || !d_destination) return;
    const uint32_t blocks = (nodeCount + BlockSize - 1) / BlockSize;
    publishVelocityKernel<<<blocks, BlockSize, 0, stream>>>(nodeCount, d_source, d_destination, d_publishGate);
}

cudaError_t appendMomentumOutflowRate(
    cudaGraph_t graph,
    cudaGraphNode_t& prev,
    uint32_t cellCount,
    uint32_t faceCount,
    uint32_t boundaryFaceCount,
    const CFDFaceGpu* d_faces,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const float* d_faceFlux,
    const float* d_boundaryFlux,
    const float* d_cellVolumes,
    float* d_outflowRates,
    float* d_maxRate,
    uint32_t* d_invalid) {
    using cudaGraphBuilder::appendKernel;
    using cudaGraphBuilder::appendMemset;

    cudaError_t result = appendMemset(graph, prev, d_outflowRates, 0, cellCount * sizeof(float));
    if (result != cudaSuccess) return result;
    result = appendMemset(graph, prev, d_maxRate, 0, sizeof(float));
    if (result != cudaSuccess) return result;
    result = appendMemset(graph, prev, d_invalid, 0, sizeof(uint32_t));
    if (result != cudaSuccess) return result;

    if (faceCount > 0 && d_faces && d_faceFlux) {
        result = appendKernel(graph, prev, (void*)computeOutflowFacesKernel, dim3((faceCount + BlockSize - 1) / BlockSize), dim3(BlockSize),
            faceCount, d_faces, d_faceFlux, d_outflowRates, d_invalid);
        if (result != cudaSuccess) return result;
    }
    if (boundaryFaceCount > 0 && d_boundaryFaces && d_boundaryFlux) {
        result = appendKernel(graph, prev, (void*)computeOutflowBoundaryKernel, dim3((boundaryFaceCount + BlockSize - 1) / BlockSize), dim3(BlockSize),
            boundaryFaceCount, d_boundaryFaces, d_boundaryFlux, d_outflowRates, d_invalid);
        if (result != cudaSuccess) return result;
    }
    result = appendKernel(graph, prev, (void*)finalizeOutflowRateKernel, dim3((cellCount + BlockSize - 1) / BlockSize), dim3(BlockSize),
        cellCount, d_cellVolumes, d_outflowRates, d_invalid);
    if (result != cudaSuccess) return result;
    return appendKernel(graph, prev, (void*)cudaSubcycle::reduceMaxRateKernel, dim3(64), dim3(256),
        cellCount, d_outflowRates, d_maxRate, d_invalid);
}

cudaError_t appendMomentumSubcycleBody(
    cudaGraph_t body,
    cudaGraphNode_t& prev,
    uint32_t cellCount,
    uint32_t faceCount,
    uint32_t boundaryFaceCount,
    const CFDFaceGpu* d_faces,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const BoundaryConditionGpu* d_boundaryConditions,
    const float* d_faceFlux,
    const float* d_boundaryFlux,
    const float* d_cellVolumes,
    const float* d_cellViscousWeight,
    const float* d_kinematicViscosity,
    glm::vec3* d_momentumAccum,
    glm::vec4* d_velocity,
    glm::vec3* d_velocityStar,
    const cudaSubcycle::SubcycleTimestep* d_timestep,
    const uint32_t* d_iteration) {
    using cudaGraphBuilder::appendKernel;
    using cudaGraphBuilder::appendMemset;

    cudaError_t result = appendMemset(body, prev, d_momentumAccum, 0, cellCount * sizeof(glm::vec3));
    if (result != cudaSuccess) return result;

    if (faceCount > 0 && d_faces && d_faceFlux) {
        result = appendKernel(body, prev, (void*)momentumInternalFacesKernel, dim3((faceCount + BlockSize - 1) / BlockSize), dim3(BlockSize),
            faceCount, d_faces, d_faceFlux, d_velocity, d_kinematicViscosity, d_momentumAccum);
        if (result != cudaSuccess) return result;
    }
    if (boundaryFaceCount > 0 && d_boundaryFaces && d_boundaryConditions && d_boundaryFlux) {
        result = appendKernel(body, prev, (void*)momentumBoundaryFacesKernel, dim3((boundaryFaceCount + BlockSize - 1) / BlockSize), dim3(BlockSize),
            boundaryFaceCount, d_boundaryFaces, d_boundaryConditions, d_boundaryFlux, d_velocity, d_kinematicViscosity, d_momentumAccum);
        if (result != cudaSuccess) return result;
    }
    result = appendKernel(body, prev, (void*)applyMomentumPredictorKernel, dim3((cellCount + BlockSize - 1) / BlockSize), dim3(BlockSize),
        cellCount, d_cellVolumes, d_cellViscousWeight, d_velocity, d_momentumAccum, d_velocityStar, d_kinematicViscosity, d_timestep);
    if (result != cudaSuccess) return result;
    return appendKernel(body, prev, (void*)advanceVelocityKernel, dim3((cellCount + BlockSize - 1) / BlockSize), dim3(BlockSize),
        cellCount, d_velocityStar, d_velocity, d_timestep, d_iteration);
}

} // namespace cfd::fluidkernels
