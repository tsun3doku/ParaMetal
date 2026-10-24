#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include "CFDTopology.hpp"
#include "CFDBoundaryConditions.hpp"
#include "CFDProjectionSolver.hpp"
#include "../voronoi/GlobalDomainTopology.hpp"
#include "../cuda/CudaBuffer.cuh"
#include "cuda/CFDSolverKernels.cuh"

class VulkanDevice;

namespace cfd {

class CFDSolver {
public:
    CFDSolver();
    ~CFDSolver();

    bool initialize(VulkanDevice& vulkanDevice);

    bool buildTopology(
        const std::vector<topology::Fragment>& fragments,
        const std::vector<topology::Face>& faces,
        const std::vector<topology::BoundaryFace>& boundaryFaces,
        const std::vector<uint32_t>& globalFragmentToFluidNode,
        const std::vector<uint32_t>& fluidNodeToGlobalFragment);

    bool bindOutput(
        glm::vec4* velocityVec4,
        float* canonicalFaceFlux,
        uint32_t canonicalFaceCount,
        float* canonicalBoundaryFlux,
        uint32_t canonicalBoundaryFaceCount);

    bool bindCompletionSemaphore(
        cudaExternalSemaphore_t semaphore);

    bool bindReleaseSemaphore(
        cudaExternalSemaphore_t semaphore);

    bool initializeVelocity(const glm::vec3& initialVelocity);
    bool setBoundaryConditions(const DomainBoundaryConditions& conditions);

    enum class StepOutcome {
        Enqueued,
        Deferred,
        Failed,
    };

    StepOutcome stepCfd(
        float kinematicViscosity,
        float macroDt,
        float targetCourant = 0.5f,
        uint64_t completionValue = 0,
        uint64_t releaseWaitValue = 0);

    // Drain flowStream before releasing semaphores and buffers
    void waitForIdle();

    void setPressureSolverParameters(int maxIterations, float tolerance);

    int getCudaDevice() const { return cudaDevice; }
    bool hasFailed() const { return failed; }
    float getLastStepGpuMs() const { return lastStepGpuMs; }

    void cleanup();

private:
    void computeAndReduceOutflowRate(cudaStream_t stream);
    void clearTopology();
    void releaseTopologyBuffers();
    bool buildMomentumSubcycleGraph(float targetCourant);
    void destroyMomentumSubcycleGraph();

    int cudaDevice = -1;
    void* flowStreamHandle = nullptr; 
    bool initialized = false;
    // Terminal failure: a failed step halts the solver until reinitialization
    bool failed = false;

    CFDProjectionSolver projectionSolver;
    cudaUtils::CudaBuffer<glm::vec3> d_momentumAccum;
    cudaUtils::CudaBuffer<float> d_cellViscousWeight;
    bool buildMomentumViscousWeights(const std::array<BoundaryCondition, 7>& bcs);

    std::vector<uint32_t> component;
    uint32_t componentCount = 0;

    uint32_t fluidNodeCount = 0;
    uint32_t faceCount = 0;
    uint32_t boundaryFaceCount = 0;

    cudaUtils::CudaBuffer<float> d_cellVolumes;
    cudaUtils::CudaBuffer<glm::vec4> d_velocity; // size fluidNodeCount (authoritative state!)

    cudaUtils::CudaBuffer<float> d_faceFlux;
    cudaUtils::CudaBuffer<float> d_boundaryFlux;

    cudaUtils::CudaBuffer<CFDFaceGpu> d_faces;
    cudaUtils::CudaBuffer<CFDBoundaryFaceGpu> d_boundaryFaces;

    // Local face to canonical global face index mapping
    cudaUtils::CudaBuffer<uint32_t> d_faceGlobalIndex;
    cudaUtils::CudaBuffer<uint32_t> d_boundaryGlobalIndex;

    cudaUtils::CudaBuffer<float> d_outflowRates;
    cudaUtils::CudaBuffer<unsigned long long> d_cflPacked;

    // Device-controlled momentum subcycle graph: planner, gate, conditional loop
    cudaGraphExec_t momentumExec = nullptr;
    float builtTargetCourant = 0.0f;
    cudaUtils::CudaBuffer<float> d_maxOutflowRate;
    cudaUtils::CudaBuffer<uint32_t> d_subcycleInvalid;
    cudaUtils::CudaBuffer<uint32_t> d_subcycleIteration;
    cudaUtils::CudaBuffer<cudaSubcycle::SubcycleTimestep> d_subcycleTimestep;
    cudaUtils::CudaBuffer<float> d_macroDt;
    cudaUtils::CudaBuffer<float> d_kinematicViscosity;
    cudaSubcycle::SubcycleTimestep* pinnedSubcycleTimestep = nullptr;

    // Authoritative velocity pointer & flux scatter outputs
    glm::vec4* externalVelocity = nullptr;
    float* externalCanonicalFaceFlux = nullptr;
    uint32_t externalCanonicalFaceCount = 0;
    float* externalCanonicalBoundaryFlux = nullptr;
    uint32_t externalCanonicalBoundaryFaceCount = 0;
    cudaExternalSemaphore_t completionSemaphore = nullptr;
    cudaExternalSemaphore_t releaseSemaphore = nullptr;

    cudaEvent_t stepStartEvent = nullptr;
    cudaEvent_t stepEndEvent = nullptr;
    float lastStepGpuMs = 0.0f;

    cudaUtils::CudaBuffer<glm::vec3> d_velocityStar;

    DomainBoundaryConditions boundaryConditions = createDefaultDomainBoundaries();
    cudaUtils::CudaBuffer<BoundaryConditionGpu> d_boundaryConditions;

    std::vector<CFDFaceGpu> hostFaces;
    std::vector<CFDBoundaryFaceGpu> hostBoundaryFaces;
    std::vector<float> hostCellVolumes;
};

} // namespace cfd
