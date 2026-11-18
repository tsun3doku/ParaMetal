#include "CFDSolverKernels.cuh"
#include "../CFDReconstruction.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cfd::fluidkernels {

static constexpr float ViscousSafety = 0.5f;
static constexpr float FallbackUpwindWeight = 0.20f;

namespace {

__global__ void buildPressureRhsFacesKernel(
    uint32_t faceCount,
    const FaceTopology* __restrict__ faces,
    const float* __restrict__ faceFlux,
    float* __restrict__ pressureRhs) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const FaceTopology face = faces[idx];
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
    const FaceTopology* __restrict__ faces,
    const float* __restrict__ psi,
    float* __restrict__ faceFlux) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const FaceTopology face = faces[idx];
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
    if (bc.type == BoundaryTypePressureOutlet) {
        boundaryFlux[idx] += bf.projectionWeight * psi[bf.cell];
    }
}

__global__ void computeOutflowFacesKernel(
    uint32_t faceCount,
    const FaceTopology* __restrict__ faces,
    const float* __restrict__ faceFlux,
    float* __restrict__ outflowRates,
    uint32_t* __restrict__ invalid) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const FaceTopology face = faces[idx];
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
    const uint32_t* __restrict__ mergeTarget,
    const float* __restrict__ mergeVolume,
    const float* __restrict__ mergedViscousWeights,
    const float* __restrict__ kinematicViscosity,
    float targetCfl,
    float* __restrict__ outflowRates,
    uint32_t* __restrict__ invalid) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cellCount) return;

    const float vol = cellVolumes[idx];
    if (vol > 0.0f && isfinite(vol)) {
        const float advectiveRate = outflowRates[idx] / vol;
        const uint32_t group = mergeTarget[idx];
        const float groupVol = mergeVolume[group] > 0.0f ? mergeVolume[group] : vol;
        const float viscousRate = (*kinematicViscosity) * mergedViscousWeights[group] / groupVol;
        outflowRates[idx] = advectiveRate / targetCfl + viscousRate / ViscousSafety;
    } else {
        outflowRates[idx] = 0.0f;
        if (invalid != nullptr) atomicOr(invalid, 1u);
    }
}

__global__ void aggregateViscousWeightKernel(uint32_t n,const float* in,const uint32_t* target,float* out) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) atomicAdd(&out[target[i]],in[i]);
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

__device__ inline glm::vec3 evaluateReconstructionFace(
    const ReconFinalCoefficients& reconstruction,
    const ReconCellGeometry& cellGeometry,
    const ReconFaceGeometry& faceGeometry) {
    const double inverseH = 1.0 / cellGeometry.h;
    const double inverseH2 = inverseH * inverseH;
    const double dx = (faceGeometry.centroid.x - cellGeometry.centroid.x) * inverseH;
    const double dy = (faceGeometry.centroid.y - cellGeometry.centroid.y) * inverseH;
    const double dz = (faceGeometry.centroid.z - cellGeometry.centroid.z) * inverseH;

    const float basis[ReconQuadraticUnknownCount] = {
        float(dx), float(dy), float(dz),
        float(faceGeometry.m2[0] * inverseH2) + float(dx) * float(dx) - float(cellGeometry.m2[0] * inverseH2),
        float(faceGeometry.m2[1] * inverseH2) + float(dx) * float(dy) - float(cellGeometry.m2[1] * inverseH2),
        float(faceGeometry.m2[2] * inverseH2) + float(dx) * float(dz) - float(cellGeometry.m2[2] * inverseH2),
        float(faceGeometry.m2[3] * inverseH2) + float(dy) * float(dy) - float(cellGeometry.m2[3] * inverseH2),
        float(faceGeometry.m2[4] * inverseH2) + float(dy) * float(dz) - float(cellGeometry.m2[4] * inverseH2),
        float(faceGeometry.m2[5] * inverseH2) + float(dz) * float(dz) - float(cellGeometry.m2[5] * inverseH2)};

    float value[3] = {};
    for (uint32_t component = 0u; component < 3u; ++component) {
        for (uint32_t unknown = 0u; unknown < ReconQuadraticUnknownCount; ++unknown) {
            value[component] += reconstruction.c[component][unknown] * basis[unknown];
        }
    }
    return glm::vec3(value[0], value[1], value[2]);
}

__global__ void momentumInternalFacesKernel(
    uint32_t faceCount,
    const FaceTopology* __restrict__ faces,
    const float* __restrict__ faceFlux,
    const glm::vec4* __restrict__ velocity,
    const float* __restrict__ kinematicViscosity,
    const ReconFinalCoefficients* __restrict__ reconstruction,
    const ReconCellGeometry* __restrict__ cellGeometry,
    const ReconFaceGeometry* __restrict__ faceGeometry,
    glm::vec3* __restrict__ momentumAccum,
    glm::vec3* __restrict__ viscousAccum) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const FaceTopology face = faces[idx];
    const float phi = faceFlux[idx];
    const glm::vec3 uA = glm::vec3(velocity[face.cellA]);
    const glm::vec3 uB = glm::vec3(velocity[face.cellB]);

    const uint32_t upwind = (phi >= 0.0f) ? face.cellA : face.cellB;
    const glm::vec3 uUpwind = (phi >= 0.0f) ? uA : uB;
    const glm::vec3 uCentral = 0.5f * (uA + uB);

    glm::vec3 uFace;
    bool reconstructed = false;
    if (reconstruction != nullptr && cellGeometry != nullptr &&
        faceGeometry != nullptr && reconstruction[upwind].valid != 0u &&
        cellGeometry[upwind].volume > 0.0 && cellGeometry[upwind].h > 0.0 &&
        faceGeometry[idx].area > 0.0) {
        const glm::vec3 uHighOrder = evaluateReconstructionFace(
            reconstruction[upwind], cellGeometry[upwind], faceGeometry[idx]);
        const glm::vec3 candidate = glm::vec3(velocity[upwind]) + uHighOrder;
        if (isfinite(candidate.x) && isfinite(candidate.y) && isfinite(candidate.z)) {
            uFace = candidate;
            reconstructed = true;
        }
    }
    if (!reconstructed) {
        uFace = FallbackUpwindWeight * uUpwind + (1.0f - FallbackUpwindWeight) * uCentral;
    }

    const glm::vec3 convection = phi * uFace;
    const glm::vec3 diffusion = ((*kinematicViscosity) * face.projectionWeight) * (uB - uA);

    const glm::vec3 netA = -convection;
    const glm::vec3 netB = convection;

    atomicAdd(&momentumAccum[face.cellA].x, netA.x);
    atomicAdd(&momentumAccum[face.cellA].y, netA.y);
    atomicAdd(&momentumAccum[face.cellA].z, netA.z);

    atomicAdd(&momentumAccum[face.cellB].x, netB.x);
    atomicAdd(&momentumAccum[face.cellB].y, netB.y);
    atomicAdd(&momentumAccum[face.cellB].z, netB.z);
    atomicAdd(&viscousAccum[face.cellA].x, diffusion.x);
    atomicAdd(&viscousAccum[face.cellA].y, diffusion.y);
    atomicAdd(&viscousAccum[face.cellA].z, diffusion.z);
    atomicAdd(&viscousAccum[face.cellB].x, -diffusion.x);
    atomicAdd(&viscousAccum[face.cellB].y, -diffusion.y);
    atomicAdd(&viscousAccum[face.cellB].z, -diffusion.z);
}

__global__ void momentumBoundaryFacesKernel(
    uint32_t boundaryFaceCount,
    const CFDBoundaryFaceGpu* __restrict__ boundaryFaces,
    const BoundaryConditionGpu* __restrict__ boundaryConditions,
    const float* __restrict__ boundaryFlux,
    const glm::vec4* __restrict__ velocity,
    const float* __restrict__ kinematicViscosity,
    glm::vec3* __restrict__ momentumAccum,
    glm::vec3* __restrict__ viscousAccum) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundaryFaceCount) return;

    const CFDBoundaryFaceGpu bf = boundaryFaces[idx];
    const float phi = boundaryFlux[idx];
    const BoundaryConditionGpu bc = boundaryConditions[bf.boundaryCondition];
    const glm::vec3 uCell = glm::vec3(velocity[bf.cell]);
    const float viscosity = *kinematicViscosity;

    glm::vec3 convection(0.0f);
    glm::vec3 diffusion(0.0f);

    if (bc.type == BoundaryTypeVelocityInlet) {
        const glm::vec3 uInlet = bc.velocity;
        const glm::vec3 uUpwind = (phi < 0.0f) ? uInlet : uCell;
        convection = phi * uUpwind;
        diffusion = (viscosity * bf.projectionWeight) * (uInlet - uCell);
    } else if (bc.type == BoundaryTypeNoSlipWall) {
        convection = glm::vec3(0.0f);
        diffusion = (viscosity * bf.projectionWeight) * (-uCell);
    } else if (bc.type == BoundaryTypePressureOutlet) {
        convection = phi * uCell;
        diffusion = glm::vec3(0.0f);
    }

    const glm::vec3 net = -convection;
    atomicAdd(&momentumAccum[bf.cell].x, net.x);
    atomicAdd(&momentumAccum[bf.cell].y, net.y);
    atomicAdd(&momentumAccum[bf.cell].z, net.z);
    atomicAdd(&viscousAccum[bf.cell].x, diffusion.x);
    atomicAdd(&viscousAccum[bf.cell].y, diffusion.y);
    atomicAdd(&viscousAccum[bf.cell].z, diffusion.z);
}

__global__ void mergeViscousAccumKernel(uint32_t count, const glm::vec3* src,
    const uint32_t* target, glm::vec3* dst) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t g = target[i];
    atomicAdd(&dst[g].x, src[i].x); atomicAdd(&dst[g].y, src[i].y); atomicAdd(&dst[g].z, src[i].z);
}

__global__ void applyMomentumStageKernel(
    uint32_t cellCount,
    const float* __restrict__ cellVolumes,
    const glm::vec4* __restrict__ baseVelocity,
    const glm::vec4* __restrict__ stageVelocity,
    const glm::vec3* __restrict__ momentumAccum,
    const glm::vec3* __restrict__ mergedViscousAccum,
    const uint32_t* __restrict__ mergeTarget,
    const float* __restrict__ mergeVolume,
    float alphaBase,
    float alphaStage,
    float alphaRhs,
    const cudaSubcycle::SubcycleTimestep* __restrict__ timestep,
    glm::vec4* __restrict__ outVelocity,
    glm::vec3* __restrict__ outVelocityStar) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cellCount) return;

    const float dt = timestep->dt;
    const float vol = cellVolumes[idx];
    const float invVol = (vol > 0.0f) ? (1.0f / vol) : 0.0f;
    const glm::vec3 base = glm::vec3(baseVelocity[idx]);
    const glm::vec3 stage = glm::vec3(stageVelocity[idx]);
    const glm::vec3 net = momentumAccum[idx];
    const uint32_t group = mergeTarget[idx];
    const float groupInvVol = mergeVolume[group] > 0.0f ? 1.0f / mergeVolume[group] : invVol;
    const glm::vec3 viscous = mergedViscousAccum[group];
    const glm::vec3 value = alphaBase * base + alphaStage * stage +
        (alphaRhs * dt * invVol) * net + (alphaRhs * dt * groupInvVol) * viscous;

    if (outVelocity != nullptr) outVelocity[idx] = glm::vec4(value, 0.0f);
    if (outVelocityStar != nullptr) outVelocityStar[idx] = value;
}

__global__ void validateVelocityStarKernel(
    uint32_t cellCount,
    const glm::vec3* __restrict__ velocityStar,
    uint32_t* __restrict__ invalid) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cellCount) return;

    const glm::vec3 value = velocityStar[idx];
    if (!isfinite(value.x) || !isfinite(value.y) || !isfinite(value.z)) {
        if (invalid != nullptr) atomicOr(invalid, 0x80000000u);
    }
}

__global__ void validatePressureRhsKernel(
    uint32_t cellCount,
    const float* __restrict__ pressureRhs,
    uint32_t* __restrict__ invalid) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cellCount) return;

    if (!isfinite(pressureRhs[idx])) {
        if (invalid != nullptr) atomicOr(invalid, 0x80000000u);
    }
}

__global__ void provisionalInternalFluxKernel(
    uint32_t faceCount,
    const FaceTopology* __restrict__ faces,
    const FaceFluxVectors* __restrict__ fluxVectors,
    const glm::vec3* __restrict__ velocityStar,
    float* __restrict__ faceFlux) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const FaceTopology face = faces[idx];
    const FaceFluxVectors vectors = fluxVectors[idx];
    const glm::vec3 uA = velocityStar[face.cellA];
    const glm::vec3 uB = velocityStar[face.cellB];

    faceFlux[idx] = glm::dot(uA, vectors.fluxVectorA) + glm::dot(uB, vectors.fluxVectorB);
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
    if (bc.type == BoundaryTypeVelocityInlet) {
        boundaryFlux[idx] = glm::dot(bc.velocity, bf.areaVector);
    } else if (bc.type == BoundaryTypePressureOutlet) {
        const glm::vec3 u = velocityStar[bf.cell];
        boundaryFlux[idx] = glm::dot(u, bf.areaVector);
    } else {
        boundaryFlux[idx] = 0.0f;
    }
}

__global__ void accumulateFacePressureImpulseKernel(
    uint32_t faceCount,
    const FaceTopology* __restrict__ faces,
    const FaceFluxVectors* __restrict__ fluxVectors,
    const float* __restrict__ psi,
    glm::vec3* __restrict__ pressureImpulse) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const FaceTopology face = faces[idx];
    const FaceFluxVectors vectors = fluxVectors[idx];
    // difference-based pressure gradient
    const float dp = psi[face.cellB] - psi[face.cellA];
    const glm::vec3 impulseA = dp * vectors.fluxVectorB;
    const glm::vec3 impulseB = dp * vectors.fluxVectorA;

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
    if (bc.type != BoundaryTypePressureOutlet) return;

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


} 
void launchAggregateViscousWeights(uint32_t count,const float* weights,const uint32_t* target,
    float* merged,cudaStream_t stream) {
    if (!count || !weights || !target || !merged) return;
    cudaMemsetAsync(merged, 0, size_t(count) * sizeof(float), stream);
    aggregateViscousWeightKernel<<<(count + BlockSize - 1) / BlockSize, BlockSize, 0, stream>>>(count,weights,target,merged);
}

void launchBuildPressureRhs(
    uint32_t cellCount,
    uint32_t faceCount,
    uint32_t boundaryFaceCount,
    const FaceTopology* d_faces,
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

void launchValidatePressureRhs(
    uint32_t cellCount,
    const float* d_pressureRhs,
    uint32_t* d_invalid,
    cudaStream_t stream) {
    if (cellCount == 0 || !d_pressureRhs) return;
    const uint32_t blocks = (cellCount + BlockSize - 1) / BlockSize;
    validatePressureRhsKernel<<<blocks, BlockSize, 0, stream>>>(
        cellCount, d_pressureRhs, d_invalid);
}

void launchCorrectFluxes(
    uint32_t faceCount,
    const FaceTopology* d_faces,
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
    const FaceTopology* d_faces,
    const FaceFluxVectors* d_fluxVectors,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const BoundaryConditionGpu* d_boundaryConditions,
    const glm::vec3* d_velocityStar,
    float* d_faceFlux,
    float* d_boundaryFlux,
    cudaStream_t stream) {
    if (faceCount > 0 && d_faces && d_fluxVectors && d_velocityStar && d_faceFlux) {
        const uint32_t blocks = (faceCount + BlockSize - 1) / BlockSize;
        provisionalInternalFluxKernel<<<blocks, BlockSize, 0, stream>>>(
            faceCount, d_faces, d_fluxVectors, d_velocityStar, d_faceFlux);
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
    const FaceTopology* d_faces,
    const FaceFluxVectors* d_fluxVectors,
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

    if (faceCount > 0 && d_faces && d_fluxVectors) {
        const uint32_t blocks = (faceCount + BlockSize - 1) / BlockSize;
        accumulateFacePressureImpulseKernel<<<blocks, BlockSize, 0, stream>>>(
            faceCount, d_faces, d_fluxVectors, d_psi, d_pressureImpulse);
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
    if (dst == 0xFFFFFFFFu) return;
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

__global__ void computeVelocityGradientsFacesKernel(
    uint32_t faceCount,
    const FaceTopology* __restrict__ faces,
    const FaceFluxVectors* __restrict__ fluxVectors,
    const glm::vec4* __restrict__ velocity,
    glm::vec4* __restrict__ gradients) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    const FaceTopology face = faces[idx];
    const FaceFluxVectors fv = fluxVectors[idx];
    const glm::vec3 areaVec = fv.fluxVectorA + fv.fluxVectorB;
    const glm::vec3 uA = glm::vec3(velocity[face.cellA]);
    const glm::vec3 uB = glm::vec3(velocity[face.cellB]);
    const glm::vec3 uFace = 0.5f * (uA + uB);

    const uint32_t baseA = face.cellA * 3u;
    const uint32_t baseB = face.cellB * 3u;

    const glm::vec3 termX = uFace.x * areaVec;
    const glm::vec3 termY = uFace.y * areaVec;
    const glm::vec3 termZ = uFace.z * areaVec;

    atomicAdd(&gradients[baseA + 0u].x,  termX.x);
    atomicAdd(&gradients[baseA + 0u].y,  termX.y);
    atomicAdd(&gradients[baseA + 0u].z,  termX.z);

    atomicAdd(&gradients[baseA + 1u].x,  termY.x);
    atomicAdd(&gradients[baseA + 1u].y,  termY.y);
    atomicAdd(&gradients[baseA + 1u].z,  termY.z);

    atomicAdd(&gradients[baseA + 2u].x,  termZ.x);
    atomicAdd(&gradients[baseA + 2u].y,  termZ.y);
    atomicAdd(&gradients[baseA + 2u].z,  termZ.z);

    atomicAdd(&gradients[baseB + 0u].x, -termX.x);
    atomicAdd(&gradients[baseB + 0u].y, -termX.y);
    atomicAdd(&gradients[baseB + 0u].z, -termX.z);

    atomicAdd(&gradients[baseB + 1u].x, -termY.x);
    atomicAdd(&gradients[baseB + 1u].y, -termY.y);
    atomicAdd(&gradients[baseB + 1u].z, -termY.z);

    atomicAdd(&gradients[baseB + 2u].x, -termZ.x);
    atomicAdd(&gradients[baseB + 2u].y, -termZ.y);
    atomicAdd(&gradients[baseB + 2u].z, -termZ.z);
}

__global__ void computeVelocityGradientsBoundaryKernel(
    uint32_t boundaryFaceCount,
    const CFDBoundaryFaceGpu* __restrict__ boundaryFaces,
    const BoundaryConditionGpu* __restrict__ boundaryConditions,
    const glm::vec4* __restrict__ velocity,
    glm::vec4* __restrict__ gradients) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundaryFaceCount) return;

    const CFDBoundaryFaceGpu bf = boundaryFaces[idx];
    const BoundaryConditionGpu bc = boundaryConditions[bf.boundaryCondition];
    const glm::vec3 areaVec = bf.areaVector;

    glm::vec3 uFace(0.0f);
    if (bc.type == BoundaryTypeVelocityInlet) {
        uFace = bc.velocity;
    } else if (bc.type == BoundaryTypePressureOutlet) {
        uFace = glm::vec3(velocity[bf.cell]);
    } else {
        return;
    }

    const uint32_t base = bf.cell * 3u;
    const glm::vec3 termX = uFace.x * areaVec;
    const glm::vec3 termY = uFace.y * areaVec;
    const glm::vec3 termZ = uFace.z * areaVec;

    atomicAdd(&gradients[base + 0u].x, termX.x);
    atomicAdd(&gradients[base + 0u].y, termX.y);
    atomicAdd(&gradients[base + 0u].z, termX.z);

    atomicAdd(&gradients[base + 1u].x, termY.x);
    atomicAdd(&gradients[base + 1u].y, termY.y);
    atomicAdd(&gradients[base + 1u].z, termY.z);

    atomicAdd(&gradients[base + 2u].x, termZ.x);
    atomicAdd(&gradients[base + 2u].y, termZ.y);
    atomicAdd(&gradients[base + 2u].z, termZ.z);
}

__global__ void finalizeVelocityGradientsKernel(
    uint32_t cellCount,
    const float* __restrict__ cellVolumes,
    glm::vec4* __restrict__ gradients) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cellCount) return;

    const float vol = cellVolumes[idx];
    const float invVol = (vol > 0.0f) ? (1.0f / vol) : 0.0f;

    const uint32_t base = idx * 3u;
    gradients[base + 0u].x *= invVol;
    gradients[base + 0u].y *= invVol;
    gradients[base + 0u].z *= invVol;
    gradients[base + 0u].w = 0.0f;

    gradients[base + 1u].x *= invVol;
    gradients[base + 1u].y *= invVol;
    gradients[base + 1u].z *= invVol;
    gradients[base + 1u].w = 0.0f;

    gradients[base + 2u].x *= invVol;
    gradients[base + 2u].y *= invVol;
    gradients[base + 2u].z *= invVol;
    gradients[base + 2u].w = 0.0f;
}

void launchReconstructVelocityGradients(
    uint32_t cellCount,
    uint32_t faceCount,
    uint32_t boundaryFaceCount,
    const FaceTopology* d_faces,
    const FaceFluxVectors* d_fluxVectors,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const BoundaryConditionGpu* d_boundaryConditions,
    const float* d_cellVolumes,
    const glm::vec4* d_velocity,
    glm::vec4* d_gradients,
    cudaStream_t stream) {
    if (cellCount == 0 || !d_gradients || !d_cellVolumes || !d_velocity) return;

    cudaMemsetAsync(d_gradients, 0, cellCount * 3u * sizeof(glm::vec4), stream);

    if (faceCount > 0 && d_faces && d_fluxVectors) {
        const uint32_t blocks = (faceCount + BlockSize - 1) / BlockSize;
        computeVelocityGradientsFacesKernel<<<blocks, BlockSize, 0, stream>>>(
            faceCount, d_faces, d_fluxVectors, d_velocity, d_gradients);
    }
    if (boundaryFaceCount > 0 && d_boundaryFaces && d_boundaryConditions) {
        const uint32_t blocks = (boundaryFaceCount + BlockSize - 1) / BlockSize;
        computeVelocityGradientsBoundaryKernel<<<blocks, BlockSize, 0, stream>>>(
            boundaryFaceCount, d_boundaryFaces, d_boundaryConditions, d_velocity, d_gradients);
    }

    const uint32_t cellBlocks = (cellCount + BlockSize - 1) / BlockSize;
    finalizeVelocityGradientsKernel<<<cellBlocks, BlockSize, 0, stream>>>(
        cellCount, d_cellVolumes, d_gradients);
}

cudaError_t appendMomentumOutflowRate(
    cudaGraph_t graph,
    cudaGraphNode_t& prev,
    uint32_t cellCount,
    uint32_t faceCount,
    uint32_t boundaryFaceCount,
    const FaceTopology* d_faces,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const float* d_faceFlux,
    const float* d_boundaryFlux,
    const float* d_cellVolumes,
    const uint32_t* d_viscousMergeTarget,
    const float* d_viscousMergeVolume,
    const float* d_mergedViscousWeight,
    const float* d_kinematicViscosity,
    float targetCfl,
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
        cellCount, d_cellVolumes, d_viscousMergeTarget, d_viscousMergeVolume,
        d_mergedViscousWeight, d_kinematicViscosity, targetCfl,
        d_outflowRates, d_invalid);
    if (result != cudaSuccess) return result;
    result = appendKernel(graph, prev, (void*)cudaSubcycle::reduceMaxRateKernel, dim3(64), dim3(256),
        cellCount, d_outflowRates, d_maxRate, d_invalid);
    return result;
}

__global__ void stageStampKernel(uint64_t* stamps, uint32_t index) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        uint64_t t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        stamps[index] = t;
    }
}

cudaError_t appendMomentumSubcycleBody(
    cudaGraph_t body,
    cudaGraphNode_t& prev,
    uint32_t cellCount,
    uint32_t faceCount,
    uint32_t boundaryFaceCount,
    const FaceTopology* d_faces,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const BoundaryConditionGpu* d_boundaryConditions,
    const float* d_faceFlux,
    const float* d_boundaryFlux,
    const float* d_cellVolumes,
    const float* d_kinematicViscosity,
    const double* d_referenceVelocity,
    glm::vec3* d_momentumAccum,
    glm::vec3* d_viscousAccum,
    glm::vec3* d_mergedViscousAccum,
    const uint32_t* d_viscousMergeTarget,
    const float* d_viscousMergeVolume,
    glm::vec4* d_velocity,
    glm::vec4* d_stageA,
    glm::vec4* d_stageB,
    glm::vec3* d_velocityStar,
    cudaSubcycle::SubcycleTimestep* d_timestep,
    cfd::CFDReconstruction* d_reconstruction,
    uint64_t* d_stageStamps) {
    using cudaGraphBuilder::appendKernel;
    using cudaGraphBuilder::appendMemset;

    const bool hasReconstruction = d_reconstruction != nullptr && d_reconstruction->ready();
    const ReconFinalCoefficients* reconstruction =
        hasReconstruction ? d_reconstruction->coefficients().get() : nullptr;
    const ReconCellGeometry* cellGeometry =
        hasReconstruction ? d_reconstruction->cellGeometry().get() : nullptr;
    const ReconFaceGeometry* faceGeometry =
        hasReconstruction ? d_reconstruction->faceGeometry().get() : nullptr;

    auto appendStage = [&](glm::vec4* state, glm::vec4* out, glm::vec3* outStar,
                           float alphaBase, float alphaStage, float alphaRhs,
                           bool measure) -> cudaError_t {
        cudaError_t result = appendMemset(body, prev, d_momentumAccum, 0, cellCount * sizeof(glm::vec3));
        if (result != cudaSuccess) return result;
        result = appendMemset(body, prev, d_viscousAccum, 0, cellCount * sizeof(glm::vec3));
        if (result != cudaSuccess) return result;
        result = appendMemset(body, prev, d_mergedViscousAccum, 0, cellCount * sizeof(glm::vec3));
        if (result != cudaSuccess) return result;
        if (measure) {
            result = appendKernel(body, prev, (void*)stageStampKernel, dim3(1), dim3(1), d_stageStamps, 0u);
            if (result != cudaSuccess) return result;
        }
        if (hasReconstruction) {
            result = d_reconstruction->appendCoefficientsNode(body, prev, state, d_referenceVelocity);
            if (result != cudaSuccess) return result;
        }
        if (measure) {
            result = appendKernel(body, prev, (void*)stageStampKernel, dim3(1), dim3(1), d_stageStamps, 1u);
            if (result != cudaSuccess) return result;
        }
        if (faceCount > 0 && d_faces && d_faceFlux) {
            result = appendKernel(body, prev, (void*)momentumInternalFacesKernel, dim3((faceCount + BlockSize - 1) / BlockSize), dim3(BlockSize),
                faceCount, d_faces, d_faceFlux, state, d_kinematicViscosity,
                reconstruction, cellGeometry, faceGeometry, d_momentumAccum, d_viscousAccum);
            if (result != cudaSuccess) return result;
        }
        if (measure) {
            result = appendKernel(body, prev, (void*)stageStampKernel, dim3(1), dim3(1), d_stageStamps, 2u);
            if (result != cudaSuccess) return result;
        }
        if (boundaryFaceCount > 0 && d_boundaryFaces && d_boundaryConditions && d_boundaryFlux) {
            result = appendKernel(body, prev, (void*)momentumBoundaryFacesKernel, dim3((boundaryFaceCount + BlockSize - 1) / BlockSize), dim3(BlockSize),
                boundaryFaceCount, d_boundaryFaces, d_boundaryConditions, d_boundaryFlux, state, d_kinematicViscosity, d_momentumAccum, d_viscousAccum);
            if (result != cudaSuccess) return result;
        }
        result = appendKernel(body, prev, (void*)mergeViscousAccumKernel, dim3((cellCount + BlockSize - 1) / BlockSize), dim3(BlockSize),
            cellCount, d_viscousAccum, d_viscousMergeTarget, d_mergedViscousAccum);
        if (result != cudaSuccess) return result;
        result = appendKernel(body, prev, (void*)applyMomentumStageKernel, dim3((cellCount + BlockSize - 1) / BlockSize), dim3(BlockSize),
            cellCount, d_cellVolumes, d_velocity, state, d_momentumAccum, d_mergedViscousAccum,
            d_viscousMergeTarget, d_viscousMergeVolume,
            alphaBase, alphaStage, alphaRhs, d_timestep, out, outStar);
        if (result != cudaSuccess) return result;
        if (measure) {
            result = appendKernel(body, prev, (void*)stageStampKernel, dim3(1), dim3(1), d_stageStamps, 3u);
            if (result != cudaSuccess) return result;
        }
        return result;
    };

    const bool measureStage = d_stageStamps != nullptr;
    cudaError_t result = appendStage(d_velocity, d_stageA, static_cast<glm::vec3*>(nullptr), 1.0f, 0.0f, 1.0f, measureStage);
    if (result != cudaSuccess) return result;
    result = appendStage(d_stageA, d_stageB, static_cast<glm::vec3*>(nullptr), 0.75f, 0.25f, 0.25f, false);
    if (result != cudaSuccess) return result;
    result = appendStage(d_stageB, d_velocity, d_velocityStar, 1.0f / 3.0f, 2.0f / 3.0f, 2.0f / 3.0f, false);
    if (result != cudaSuccess) return result;
    return appendKernel(body, prev, (void*)validateVelocityStarKernel, dim3((cellCount + BlockSize - 1) / BlockSize), dim3(BlockSize),
        cellCount, d_velocityStar, reinterpret_cast<uint32_t*>(&d_timestep->status));
}

} // namespace cfd::fluidkernels
