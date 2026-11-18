#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include <cmath>
#include "CFDTopology.hpp"
#include "CFDBoundaryConditions.hpp"
#include "CFDProjectionSolver.hpp"
#include "CFDReconstruction.hpp"
#include "../voronoi/GlobalDomainTopology.hpp"
#include "../cuda/CudaBuffer.cuh"
#include "../cuda/CudaEvent.cuh"
#include "cuda/CFDSolverKernels.cuh"

class VulkanDevice;

namespace cfd {

class CFDSolver {
public:
    CFDSolver();
    ~CFDSolver();

    bool initialize(VulkanDevice& vulkanDevice);

    bool buildTopology(
        const topology::GlobalDomain& domain,
        const std::vector<uint32_t>& globalFragmentToFluidNode,
        const std::vector<uint32_t>& fluidNodeToGlobalFragment);

    bool bindOutput(
        glm::vec4* velocityVec4,
        float* canonicalFaceFlux,
        uint32_t canonicalFaceCount,
        float* canonicalBoundaryFlux,
        uint32_t canonicalBoundaryFaceCount,
        glm::vec4* velocityGradients = nullptr);

    const std::vector<glm::vec4>& getCellCentroids() const { return hostCellCentroids; }

    bool bindCompletionSemaphore(cudaExternalSemaphore_t semaphore);

    bool bindReleaseSemaphore(cudaExternalSemaphore_t semaphore);

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

    void waitForIdle();

    bool buildMomentumSubcycleGraph(float targetCourant);
    void destroyMomentumSubcycleGraph();

    void setPressureSolverParameters(int maxIterations, float tolerance);
    void setReferenceVelocity(double value) {
        referenceVelocity = (std::isfinite(value) && value > 0.0) ? value : 1.0;
    }

    bool reconstructVelocityGradients(glm::vec4* externalGradients, cudaStream_t stream = nullptr);

    int getCudaDevice() const { return cudaDevice; }
    bool hasFailed() const { return failed; }
    float getLastStepGpuMs() const { return lastStepGpuMs; }

    void cleanup();

private:
    void clearTopology();
    void releaseTopologyBuffers();

    static constexpr uint32_t MaxMomentumSubsteps = 1024;
    static constexpr uint32_t PhaseMarkerCount = 5;
    static constexpr uint32_t StageStampCount = 4;

    int cudaDevice = -1;
    void* flowStreamHandle = nullptr; 
    bool initialized = false;
    bool failed = false;

    CFDProjectionSolver projectionSolver;
    CFDReconstruction reconstruction;
    cudaUtils::CudaBuffer<glm::vec3> d_momentumAccum;
    cudaUtils::CudaBuffer<float> d_cellViscousWeight;
    bool buildMomentumViscousWeights(const std::array<BoundaryCondition, 7>& bcs);

    std::vector<uint32_t> component;
    uint32_t componentCount = 0;

    uint32_t fluidNodeCount = 0;
    uint32_t faceCount = 0;
    uint32_t boundaryFaceCount = 0;

    cudaUtils::CudaBuffer<float> d_cellVolumes;
    cudaUtils::CudaBuffer<glm::vec4> d_velocity;
    cudaUtils::CudaBuffer<glm::vec4> d_rkStageA;
    cudaUtils::CudaBuffer<glm::vec4> d_rkStageB;

    cudaUtils::CudaBuffer<float> d_faceFlux;
    cudaUtils::CudaBuffer<float> d_boundaryFlux;

    cudaUtils::CudaBuffer<FaceTopology> d_faceTopology;
    cudaUtils::CudaBuffer<FaceFluxVectors> d_faceFluxVectors;
    cudaUtils::CudaBuffer<CFDBoundaryFaceGpu> d_boundaryFaces;

    cudaUtils::CudaBuffer<uint32_t> d_faceGlobalIndex;
    cudaUtils::CudaBuffer<uint32_t> d_boundaryGlobalIndex;

    cudaUtils::CudaBuffer<float> d_outflowRates;
    cudaUtils::CudaBuffer<float> d_mergedViscousWeight;
    cudaUtils::CudaBuffer<uint32_t> d_viscousMergeTarget;
    cudaUtils::CudaBuffer<float> d_viscousMergeVolume;
    cudaUtils::CudaBuffer<glm::vec3> d_viscousAccum;
    cudaUtils::CudaBuffer<glm::vec3> d_mergedViscousAccum;
    cudaGraphExec_t momentumExec = nullptr;
    float builtTargetCourant = 0.0f;
    cudaUtils::CudaBuffer<float> d_maxOutflowRate;
    cudaUtils::CudaBuffer<uint32_t> d_subcycleInvalid;
    cudaUtils::CudaBuffer<uint32_t> d_subcycleIteration;
    cudaUtils::CudaBuffer<cudaSubcycle::SubcycleTimestep> d_subcycleTimestep;
    cudaUtils::CudaBuffer<float> d_macroDt;
    cudaUtils::CudaBuffer<float> d_kinematicViscosity;
    cudaUtils::CudaBuffer<double> d_referenceVelocity;
    double referenceVelocity = 1.0;
    cudaSubcycle::SubcycleTimestep* pinnedSubcycleTimestep = nullptr;

    glm::vec4* externalVelocity = nullptr;
    glm::vec4* externalVelocityGradients = nullptr;
    float* externalCanonicalFaceFlux = nullptr;
    uint32_t externalCanonicalFaceCount = 0;
    float* externalCanonicalBoundaryFlux = nullptr;
    uint32_t externalCanonicalBoundaryFaceCount = 0;
    cudaExternalSemaphore_t completionSemaphore = nullptr;
    cudaExternalSemaphore_t releaseSemaphore = nullptr;

    cudaUtils::CudaEvent phaseMarkers[PhaseMarkerCount];
    cudaUtils::CudaBuffer<uint64_t> d_stageStamps;
    bool phaseMarkersRecorded = false;
    float lastStepGpuMs = 0.0f;

    cudaUtils::CudaBuffer<glm::vec3> d_velocityStar;

    uint64_t stepIndex = 0;

    DomainBoundaryConditions boundaryConditions = createDefaultDomainBoundaries();
    cudaUtils::CudaBuffer<BoundaryConditionGpu> d_boundaryConditions;

    std::vector<FaceTopology> hostFaceTopology;
    std::vector<CFDBoundaryFaceGpu> hostBoundaryFaces;
    std::vector<float> hostCellVolumes;
    std::vector<glm::vec4> hostCellCentroids;
};

} // namespace cfd
