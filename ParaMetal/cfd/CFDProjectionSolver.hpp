#pragma once

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "CFDTopology.hpp"
#include "CFDBoundaryConditions.hpp"
#include "cuda/CFDSolverKernels.cuh"
#include "../cuda/CudaBuffer.cuh"
#include "../cuda/CudaPcg.cuh"

namespace cfd {

class CFDProjectionSolver {
public:
    CFDProjectionSolver() = default;
    ~CFDProjectionSolver();

    CFDProjectionSolver(const CFDProjectionSolver&) = delete;
    CFDProjectionSolver& operator=(const CFDProjectionSolver&) = delete;

    bool initialize(int inCudaDevice);

    bool build(
        uint32_t inCellCount,
        const std::vector<CFDFaceGpu>& faces,
        const std::vector<CFDBoundaryFaceGpu>& boundaryFaces,
        const std::array<BoundaryCondition, 7>& boundaryConditions,
        const std::vector<uint32_t>& component,
        uint32_t componentCount,
        const std::vector<float>& cellVolumes,
        void* flowStreamHandle);

    bool projectVelocity(
        uint32_t inCellCount,
        uint32_t faceCount,
        uint32_t boundaryFaceCount,
        const CFDFaceGpu* d_faces,
        const CFDBoundaryFaceGpu* d_boundaryFaces,
        const BoundaryConditionGpu* d_boundaryConditions,
        const float* d_cellVolumes,
        float* d_faceFlux,
        float* d_boundaryFlux,
        const glm::vec3* d_velocityStar,
        glm::vec4* d_velocity,
        const uint32_t* d_scheduleStatus,
        void* flowStreamHandle);

    void cleanup();

    void setPressureSolverParameters(int maxIterations, float tolerance);

    const int* getPublishGate() const { return d_publishGate.get(); }

    enum class SolvePollResult {
        Pending,
        Succeeded,
        Failed,
        Error,
    };
    // Non-blocking poll for in-flight solve completion
    SolvePollResult pollNativeSolveResult();
    bool hasPendingNativeSolve() const { return pcgStatusPending; }

private:
    bool setupNativePcg(cudaStream_t stream);
    void cleanupNativePcg();

    int cudaDevice = -1;
    uint32_t cellCount = 0;

    cudaGraphExec_t pressurePcgExec = nullptr;
    cudaUtils::CudaBuffer<int> d_publishGate;
    cudaPcg::PcgStatus* pinnedPcgStatus = nullptr;
    cudaEvent_t pcgStatusEvent = nullptr;
    bool pcgStatusPending = false;
    cudaUtils::CudaBuffer<int> d_pressureRowOffsets;
    cudaUtils::CudaBuffer<int> d_pressureColumnIndices;
    cudaUtils::CudaBuffer<float> d_pressureValues;
    cudaUtils::CudaBuffer<float> d_pressureInvDiag;
    cudaUtils::CudaBuffer<float> d_pressureAp;
    cudaUtils::CudaBuffer<float> d_pressureR;
    cudaUtils::CudaBuffer<float> d_pressureZ;
    cudaUtils::CudaBuffer<float> d_pressureP;
    cudaUtils::CudaBuffer<cudaPcg::PcgStatus> d_pcgStatus;
    cudaUtils::CudaBuffer<cudaPcg::PcgConfig> d_pcgConfig;

    std::vector<int> pressureRowOffsets;
    std::vector<int> pressureColumnIndices;
    std::vector<float> pressureValues;
    int pressureMaxIterations = 0;
    float pressureTolerance = 0.0f;

    cudaUtils::CudaBuffer<float> d_pressureRhs;
    cudaUtils::CudaBuffer<float> d_psi;
    cudaUtils::CudaBuffer<uint8_t> d_isPinned;
    cudaUtils::CudaBuffer<glm::vec3> d_workImpulse;
};

} // namespace cfd
