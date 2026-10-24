#pragma once

#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cstdint>
#include "../CFDTopology.hpp"
#include "../CFDBoundaryConditions.hpp"
#include "../../cuda/SubcycleTimestep.cuh"

namespace cfd::fluidkernels {

constexpr uint32_t BlockSize = 256;

// Find the max per-cell outflow rate for CFL determination
void launchCflArgmax(
    uint32_t cellCount,
    const float* d_outflowRates,
    unsigned long long* d_packedMax,
    cudaStream_t stream);

void launchBuildPressureRhs(
    uint32_t cellCount,
    uint32_t faceCount,
    uint32_t boundaryFaceCount,
    const CFDFaceGpu* d_faces,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const float* d_faceFlux,
    const float* d_boundaryFlux,
    float* d_pressureRhs,
    cudaStream_t stream);

void launchCorrectFluxes(
    uint32_t faceCount,
    const CFDFaceGpu* d_faces,
    const float* d_psi,
    float* d_faceFlux,
    cudaStream_t stream);

void launchCorrectBoundaryFluxes(
    uint32_t boundaryFaceCount,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const BoundaryConditionGpu* d_boundaryConditions,
    const float* d_psi,
    float* d_boundaryFlux,
    cudaStream_t stream);

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
    cudaStream_t stream);

void launchZeroMasked(
    uint32_t cellCount,
    const uint8_t* d_isPinned,
    float* d_buf,
    cudaStream_t stream);

void launchInitializeVelocity(
    uint32_t cellCount,
    const glm::vec3& initialVelocity,
    glm::vec4* d_velocity,
    glm::vec3* d_velocityStar,
    cudaStream_t stream);

void launchProvisionalFlux(
    uint32_t faceCount,
    uint32_t boundaryFaceCount,
    const CFDFaceGpu* d_faces,
    const CFDBoundaryFaceGpu* d_boundaryFaces,
    const BoundaryConditionGpu* d_boundaryConditions,
    const glm::vec3* d_velocityStar,
    float* d_faceFlux,
    float* d_boundaryFlux,
    cudaStream_t stream);

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
    cudaStream_t stream);

// Scatter solver face fluxes into canonical global-face buffers
void publishFineFaceFluxes(
    uint32_t faceCount,
    const uint32_t* d_faceGlobalIndex,
    const float* d_faceFlux,
    float* canonicalFaceFlux,
    const int* d_publishGate,
    cudaStream_t stream);

void publishFineBoundaryFluxes(
    uint32_t boundaryFaceCount,
    const uint32_t* d_boundaryGlobalIndex,
    const float* d_boundaryFlux,
    float* canonicalBoundaryFlux,
    const int* d_publishGate,
    cudaStream_t stream);

// Copy authoritative velocity to external buffer
void publishVelocity(
    uint32_t nodeCount,
    const glm::vec4* d_source,
    glm::vec4* d_destination,
    const int* d_publishGate,
    cudaStream_t stream);

// Append momentum outflow rate calculation to graph
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
    uint32_t* d_invalid);

// Append single momentum subcycle step to graph
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
    const uint32_t* d_iteration);

} // namespace cfd::fluidkernels
