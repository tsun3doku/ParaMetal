#include "../CFDSolver.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cuda_runtime.h>
#include "../../vulkan/VulkanDevice.hpp"
#include "../../cuda/CudaVulkanDevice.hpp"
#include "../cuda/CFDSolverKernels.cuh"
#include "../../cuda/CudaCheck.cuh"

namespace cfd {

CFDSolver::CFDSolver() {
}

CFDSolver::~CFDSolver() {
    cleanup();
}

bool CFDSolver::initialize(VulkanDevice& vulkanDevice) {
    cleanup();

    cudaDevice = cudaVulkan::findDevice(vulkanDevice.getPhysicalDevice());
    if (cudaDevice < 0) {
        std::cerr << "[CFDSolver] Invalid CUDA device ID from VulkanDevice" << std::endl;
        return false;
    }

    if (!cudaUtils::cudaCheck(cudaSetDevice(cudaDevice), "cudaSetDevice")) {
        return false;
    }

    cudaStream_t stream = nullptr;
    if (!cudaUtils::cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create flowStream")) {
        return false;
    }
    flowStreamHandle = reinterpret_cast<void*>(stream);

    for (cudaUtils::CudaEvent& marker : phaseMarkers) {
        if (!marker.create()) {
            cleanup();
            return false;
        }
    }

    if (!projectionSolver.initialize(cudaDevice)) {
        std::cerr << "[CFDSolver] Failed to initialize projectionSolver\n";
        cleanup();
        return false;
    }

    initialized = true;
    return true;
}

bool CFDSolver::bindOutput(
    glm::vec4* velocityVec4,
    float* inCanonicalFaceFlux,
    uint32_t inCanonicalFaceCount,
    float* inCanonicalBoundaryFlux,
    uint32_t inCanonicalBoundaryFaceCount,
    glm::vec4* velocityGradients) {
    externalVelocity = velocityVec4;
    externalCanonicalFaceFlux = inCanonicalFaceFlux;
    externalCanonicalFaceCount = inCanonicalFaceCount;
    externalCanonicalBoundaryFlux = inCanonicalBoundaryFlux;
    externalCanonicalBoundaryFaceCount = inCanonicalBoundaryFaceCount;
    externalVelocityGradients = velocityGradients;
    return true;
}

bool CFDSolver::bindCompletionSemaphore(cudaExternalSemaphore_t semaphore) {
    completionSemaphore = semaphore;
    return true;
}

bool CFDSolver::bindReleaseSemaphore(cudaExternalSemaphore_t semaphore) {
    releaseSemaphore = semaphore;
    return true;
}

void CFDSolver::clearTopology() {
    destroyMomentumSubcycleGraph();
    projectionSolver.cleanup();
    reconstruction.reset();

    d_maxOutflowRate.reset();
    d_mergedViscousWeight.reset();
    d_subcycleInvalid.reset();
    d_subcycleIteration.reset();
    d_subcycleTimestep.reset();
    d_macroDt.reset();
    d_kinematicViscosity.reset();
    d_referenceVelocity.reset();
    if (pinnedSubcycleTimestep != nullptr) {
        cudaFreeHost(pinnedSubcycleTimestep);
        pinnedSubcycleTimestep = nullptr;
    }

    fluidNodeCount = 0;
    faceCount = 0;
    boundaryFaceCount = 0;
    failed = false;
    stepIndex = 0;

    releaseTopologyBuffers();

    hostFaceTopology.clear();
    hostBoundaryFaces.clear();
    hostCellVolumes.clear();
    hostCellCentroids.clear();
    component.clear();
    componentCount = 0;
}

void CFDSolver::releaseTopologyBuffers() {
    d_momentumAccum.reset();
    d_cellViscousWeight.reset();
    d_cellVolumes.reset();
    d_velocity.reset();
    d_rkStageA.reset();
    d_rkStageB.reset();
    d_velocityStar.reset();
    d_faceTopology.reset();
    d_faceFluxVectors.reset();
    d_boundaryFaces.reset();
    d_faceGlobalIndex.reset();
    d_boundaryGlobalIndex.reset();
    d_faceFlux.reset();
    d_boundaryFlux.reset();
    d_outflowRates.reset();
    d_viscousMergeTarget.reset(); d_viscousMergeVolume.reset();
    d_viscousAccum.reset(); d_mergedViscousAccum.reset();
    d_boundaryConditions.reset();
    d_stageStamps.reset();
}

void CFDSolver::cleanup() {
    if (flowStreamHandle != nullptr) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(flowStreamHandle));
    }

    clearTopology();

    for (cudaUtils::CudaEvent& marker : phaseMarkers) {
        marker.reset();
    }
    phaseMarkersRecorded = false;
    lastStepGpuMs = 0.0f;

    if (flowStreamHandle != nullptr) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(flowStreamHandle);
        cudaStreamDestroy(stream);
        flowStreamHandle = nullptr;
    }

    externalVelocity = nullptr;
    externalVelocityGradients = nullptr;
    externalCanonicalFaceFlux = nullptr;
    externalCanonicalFaceCount = 0;
    externalCanonicalBoundaryFlux = nullptr;
    externalCanonicalBoundaryFaceCount = 0;
    completionSemaphore = nullptr;
    releaseSemaphore = nullptr;

    cudaDevice = -1;
    initialized = false;
}

void CFDSolver::waitForIdle() {
    if (flowStreamHandle != nullptr) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(flowStreamHandle));
    }
}

bool CFDSolver::buildTopology(
    const CFDRuntime& runtime,
    const std::vector<glm::dvec3>& fragmentCentroids,
    const std::vector<double>& fragmentVolumes,
    const std::vector<uint32_t>& faceFragmentA,
    const std::vector<uint32_t>& faceFragmentB,
    const std::vector<double>& faceAreasA,
    const std::vector<double>& faceAreasB,
    const std::vector<glm::dvec3>& faceAreaVectorsA,
    const std::vector<glm::dvec3>& faceAreaVectorsB,
    const std::vector<uint32_t>& faceLoopOffsets,
    const std::vector<uint32_t>& loopVertexOffsets,
    const std::vector<glm::dvec3>& loopVertices,
    const std::vector<uint32_t>& globalFragmentToFluidNode,
    const std::vector<uint32_t>& fluidNodeToGlobalFragment) {
    if (!initialized || cudaDevice < 0) {
        std::cerr << "[CFDSolver] buildTopology called before initialize" << std::endl;
        return false;
    }

    clearTopology();
    cudaSetDevice(cudaDevice);

    fluidNodeCount = runtime.getFluidNodeCount();
    faceCount = static_cast<uint32_t>(runtime.faceTopology().size());
    boundaryFaceCount = static_cast<uint32_t>(runtime.boundaryFaces().size());

    if (fluidNodeCount == 0) {
        return true;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(flowStreamHandle);
    if (!reconstruction.build(
            runtime,
            fragmentCentroids,
            fragmentVolumes,
            faceFragmentA,
            faceFragmentB,
            faceAreasA,
            faceAreasB,
            faceAreaVectorsA,
            faceAreaVectorsB,
            faceLoopOffsets,
            loopVertexOffsets,
            loopVertices,
            globalFragmentToFluidNode,
            fluidNodeToGlobalFragment,
            stream)) {
        std::cerr << "[CFDSolver] buildTopology FAILED: CFD reconstruction build failed\n";
        clearTopology();
        return false;
    }

    this->hostFaceTopology = runtime.faceTopology();
    std::vector<FaceFluxVectors> hostFaceFluxVectors = runtime.faceFluxVectors();
    this->hostBoundaryFaces = runtime.boundaryFaces();
    this->hostCellVolumes = runtime.cellVolumes();
    this->hostCellCentroids = runtime.cellCentroids();
    this->component = runtime.pressureComponents();
    this->componentCount = runtime.pressureComponentCount();

    std::vector<uint32_t> mergeTarget(fluidNodeCount);
    std::vector<float> mergeVolume = this->hostCellVolumes;
    std::vector<float> medianWork = this->hostCellVolumes;
    const size_t medianIndex = medianWork.size() / 2u;
    std::nth_element(medianWork.begin(), medianWork.begin() + medianIndex, medianWork.end());
    const float mergeThreshold = medianWork.empty() ? 0.0f : 0.05f * medianWork[medianIndex];
    for (uint32_t i = 0; i < fluidNodeCount; ++i) mergeTarget[i] = i;
    std::vector<uint32_t> bestMergeNeighbor(fluidNodeCount, UINT32_MAX);
    std::vector<float> bestMergeWeight(fluidNodeCount, -1.0f);
    for (const FaceTopology& face : this->hostFaceTopology) {
        const uint32_t a = face.cellA, b = face.cellB;
        if (a >= fluidNodeCount || b >= fluidNodeCount || !std::isfinite(face.projectionWeight)) continue;
        if (mergeVolume[a] < mergeThreshold && mergeVolume[b] >= mergeThreshold &&
            face.projectionWeight > bestMergeWeight[a]) {
            bestMergeNeighbor[a] = b;
            bestMergeWeight[a] = face.projectionWeight;
        }
        if (mergeVolume[b] < mergeThreshold && mergeVolume[a] >= mergeThreshold &&
            face.projectionWeight > bestMergeWeight[b]) {
            bestMergeNeighbor[b] = a;
            bestMergeWeight[b] = face.projectionWeight;
        }
    }
    for (uint32_t cell = 0; cell < fluidNodeCount; ++cell) {
        if (bestMergeNeighbor[cell] != UINT32_MAX) mergeTarget[cell] = bestMergeNeighbor[cell];
    }
    for (uint32_t cell = 0; cell < fluidNodeCount; ++cell) {
        if (mergeTarget[cell] != cell) mergeVolume[mergeTarget[cell]] += mergeVolume[cell];
    }

    if (!d_cellVolumes.allocate(fluidNodeCount) ||
        !d_velocity.allocate(fluidNodeCount) ||
        !d_rkStageA.allocate(fluidNodeCount) ||
        !d_rkStageB.allocate(fluidNodeCount) ||
        !d_velocityStar.allocate(fluidNodeCount) ||
        !d_momentumAccum.allocate(fluidNodeCount) ||
        !d_viscousAccum.allocate(fluidNodeCount) || !d_mergedViscousAccum.allocate(fluidNodeCount) ||
        !d_viscousMergeTarget.allocate(fluidNodeCount) || !d_viscousMergeVolume.allocate(fluidNodeCount) ||
        !d_cellViscousWeight.allocate(fluidNodeCount) ||
        !d_outflowRates.allocate(fluidNodeCount) ||
        !d_mergedViscousWeight.allocate(fluidNodeCount) ||
        !d_boundaryConditions.allocate(7)) {
        clearTopology();
        return false;
    }
    if (!d_viscousMergeTarget.upload(mergeTarget.data(), fluidNodeCount) ||
        !d_viscousMergeVolume.upload(mergeVolume.data(), fluidNodeCount)) {
        clearTopology();
        return false;
    }

    if (faceCount > 0) {
        if (!d_faceTopology.allocate(faceCount) ||
            !d_faceFluxVectors.allocate(faceCount) ||
            !d_faceGlobalIndex.allocate(faceCount) ||
            !d_faceFlux.allocate(faceCount)) {
            clearTopology();
            return false;
        }
    }

    if (boundaryFaceCount > 0) {
        if (!d_boundaryFaces.allocate(boundaryFaceCount) ||
            !d_boundaryGlobalIndex.allocate(boundaryFaceCount) ||
            !d_boundaryFlux.allocate(boundaryFaceCount)) {
            clearTopology();
            return false;
        }
    }

    if (!d_maxOutflowRate.allocate(1) ||
        !d_subcycleInvalid.allocate(1) ||
        !d_subcycleIteration.allocate(1) ||
        !d_subcycleTimestep.allocate(1) ||
        !d_macroDt.allocate(1) ||
        !d_kinematicViscosity.allocate(1) ||
        !d_referenceVelocity.allocate(1) ||
        !d_stageStamps.allocate(StageStampCount)) {
        clearTopology();
        return false;
    }
    if (cudaMallocHost(reinterpret_cast<void**>(&pinnedSubcycleTimestep), sizeof(cudaSubcycle::SubcycleTimestep)) != cudaSuccess) {
        pinnedSubcycleTimestep = nullptr;
        clearTopology();
        return false;
    }
    *pinnedSubcycleTimestep = cudaSubcycle::SubcycleTimestep{};

    auto initialGpuBcs = toGpuArray(boundaryConditions);
    if (!d_cellVolumes.uploadAsync(this->hostCellVolumes.data(), fluidNodeCount, stream) ||
        !d_boundaryConditions.uploadAsync(initialGpuBcs.data(), 7, stream) ||
        !cudaUtils::cudaCheck(cudaMemsetAsync(d_velocity.get(), 0, fluidNodeCount * sizeof(glm::vec4), stream), "zero d_velocity") ||
        !cudaUtils::cudaCheck(cudaMemsetAsync(d_velocityStar.get(), 0, fluidNodeCount * sizeof(glm::vec3), stream), "zero d_velocityStar")) {
        clearTopology();
        return false;
    }

    if (faceCount > 0) {
        if (!d_faceTopology.uploadAsync(this->hostFaceTopology.data(), faceCount, stream) ||
            !d_faceFluxVectors.uploadAsync(hostFaceFluxVectors.data(), faceCount, stream) ||
            !d_faceGlobalIndex.uploadAsync(runtime.faceGlobalIndex().data(), faceCount, stream) ||
            !cudaUtils::cudaCheck(cudaMemsetAsync(d_faceFlux.get(), 0, faceCount * sizeof(float), stream), "zero d_faceFlux")) {
            clearTopology();
            return false;
        }
    }

    if (boundaryFaceCount > 0) {
        if (!d_boundaryFaces.uploadAsync(this->hostBoundaryFaces.data(), boundaryFaceCount, stream) ||
            !d_boundaryGlobalIndex.uploadAsync(runtime.boundaryGlobalIndex().data(), boundaryFaceCount, stream) ||
            !cudaUtils::cudaCheck(cudaMemsetAsync(d_boundaryFlux.get(), 0, boundaryFaceCount * sizeof(float), stream), "zero d_boundaryFlux")) {
            clearTopology();
            return false;
        }
    }

    auto fullBcs = toFullBoundaryArray(boundaryConditions);
    if (!buildMomentumViscousWeights(fullBcs)) {
        std::cerr << "[CFDSolver] buildTopology FAILED: buildMomentumViscousWeights failed\n";
        clearTopology();
        return false;
    }

    if (!projectionSolver.build(fluidNodeCount, this->hostFaceTopology, this->hostBoundaryFaces, fullBcs, this->component, this->componentCount, this->hostCellVolumes, flowStreamHandle)) {
        std::cerr << "[CFDSolver] buildTopology FAILED: projectionSolver.build failed\n";
        clearTopology();
        return false;
    }

    return true;
}

bool CFDSolver::buildMomentumViscousWeights(const std::array<BoundaryCondition, 7>& bcs) {
    if (fluidNodeCount == 0) return true;
    std::vector<float> diagWeights(fluidNodeCount, 0.0f);
    for (const auto& cf : hostFaceTopology) {
        if (cf.cellA < fluidNodeCount && cf.cellB < fluidNodeCount && cf.cellA != cf.cellB) {
            diagWeights[cf.cellA] += cf.projectionWeight;
            diagWeights[cf.cellB] += cf.projectionWeight;
        }
    }
    for (const auto& cbf : hostBoundaryFaces) {
        if (cbf.boundaryCondition < bcs.size()) {
            const auto& bc = bcs[cbf.boundaryCondition];
            if (bc.type == BoundaryType::NoSlipWall || bc.type == BoundaryType::VelocityInlet) {
                if (cbf.cell < fluidNodeCount) {
                    diagWeights[cbf.cell] += cbf.projectionWeight;
                }
            }
        }
    }
    if (!d_cellViscousWeight.upload(diagWeights.data(), fluidNodeCount)) return false;
    fluidkernels::launchAggregateViscousWeights(
        fluidNodeCount, d_cellViscousWeight.get(), d_viscousMergeTarget.get(),
        d_mergedViscousWeight.get(), reinterpret_cast<cudaStream_t>(flowStreamHandle));
    return cudaUtils::cudaCheck(cudaGetLastError(), "aggregate merged viscous weights");
}

bool CFDSolver::buildMomentumSubcycleGraph(float targetCourant) {
    if (momentumExec != nullptr && targetCourant == builtTargetCourant) {
        return true;
    }
    destroyMomentumSubcycleGraph();

    const float zero = 0.0f;
    if (!d_macroDt.upload(&zero, 1) || !d_kinematicViscosity.upload(&zero, 1) ||
        !d_referenceVelocity.upload(&referenceVelocity, 1)) {
        return false;
    }

    cudaGraph_t graph = nullptr;
    if (!cudaUtils::cudaCheck(cudaGraphCreate(&graph, 0), "momentum subcycle graph create")) {
        return false;
    }

    cudaGraphNode_t prev = nullptr;
    if (fluidkernels::appendMomentumOutflowRate(
            graph, prev, fluidNodeCount, faceCount, boundaryFaceCount,
            d_faceTopology.get(), d_boundaryFaces.get(), d_faceFlux.get(), d_boundaryFlux.get(),
            d_cellVolumes.get(), d_viscousMergeTarget.get(),
            d_viscousMergeVolume.get(), d_mergedViscousWeight.get(), d_kinematicViscosity.get(), targetCourant,
            d_outflowRates.get(), d_maxOutflowRate.get(), d_subcycleInvalid.get()) != cudaSuccess) {
        cudaGraphDestroy(graph);
        destroyMomentumSubcycleGraph();
        return false;
    }

    cudaSubcycle::SubcycleSchedule schedule{};
    schedule.maxRate = d_maxOutflowRate.get();
    schedule.invalidFlag = d_subcycleInvalid.get();
    schedule.macroDt = d_macroDt.get();
    schedule.targetCfl = targetCourant;
    schedule.maxCount = MaxMomentumSubsteps;
    schedule.timestep = d_subcycleTimestep.get();
    schedule.iteration = d_subcycleIteration.get();

    cudaGraphConditionalHandle handle = {};
    cudaGraphBuilder::WhileLoop loop;
    if (!cudaUtils::cudaCheck(cudaSubcycle::beginSubcycleLoop(graph, prev, schedule, handle, loop), "momentum subcycle loop")) {
        cudaGraphDestroy(graph);
        destroyMomentumSubcycleGraph();
        return false;
    }

    cudaGraphNode_t body = nullptr;
    if (fluidkernels::appendMomentumSubcycleBody(
            loop.body, body, fluidNodeCount, faceCount, boundaryFaceCount,
            d_faceTopology.get(), d_boundaryFaces.get(), d_boundaryConditions.get(),
            d_faceFlux.get(), d_boundaryFlux.get(), d_cellVolumes.get(),
            d_kinematicViscosity.get(), d_referenceVelocity.get(),
            d_momentumAccum.get(),
            d_viscousAccum.get(), d_mergedViscousAccum.get(), d_viscousMergeTarget.get(), d_viscousMergeVolume.get(),
            d_velocity.get(), d_rkStageA.get(), d_rkStageB.get(), d_velocityStar.get(),
            d_subcycleTimestep.get(), &reconstruction,
            d_stageStamps.get()) != cudaSuccess ||
        cudaSubcycle::endSubcycleLoop(loop.body, body, handle, schedule) != cudaSuccess) {
        cudaGraphDestroy(graph);
        destroyMomentumSubcycleGraph();
        return false;
    }

    if (!cudaUtils::cudaCheck(cudaGraphInstantiate(&momentumExec, graph, nullptr, nullptr, 0), "momentum subcycle instantiate")) {
        cudaGraphDestroy(graph);
        momentumExec = nullptr;
        return false;
    }
    cudaGraphDestroy(graph);
    builtTargetCourant = targetCourant;
    return true;
}

void CFDSolver::destroyMomentumSubcycleGraph() {
    if (momentumExec) {
        cudaGraphExecDestroy(momentumExec);
        momentumExec = nullptr;
    }
    builtTargetCourant = 0.0f;
}

void CFDSolver::setPressureSolverParameters(int maxIterations, float tolerance) {
    projectionSolver.setPressureSolverParameters(maxIterations, tolerance);
}

bool CFDSolver::setBoundaryConditions(const DomainBoundaryConditions& conditions) {
    bool typeChanged = false;
    for (size_t i = 0; i < DomainBoundaryCount; ++i) {
        if (boundaryConditions[i].type != conditions[i].type) {
            typeChanged = true;
            break;
        }
    }

    boundaryConditions = conditions;

    auto gpuBcs = toGpuArray(boundaryConditions);
    if (d_boundaryConditions.get() && flowStreamHandle) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(flowStreamHandle);
        if (!d_boundaryConditions.uploadAsync(gpuBcs.data(), 7, stream)) {
            std::cerr << "[CFDSolver] Failed to upload boundary conditions to GPU\n";
            return false;
        }
    }

    if (typeChanged && fluidNodeCount > 0 && !hostFaceTopology.empty()) {
        waitForIdle();
        auto fullBcs = toFullBoundaryArray(boundaryConditions);
        if (!buildMomentumViscousWeights(fullBcs)) {
            std::cerr << "[CFDSolver] setBoundaryConditions: buildMomentumViscousWeights failed\n";
            return false;
        }
        if (!projectionSolver.build(fluidNodeCount, hostFaceTopology, hostBoundaryFaces, fullBcs, component, componentCount, hostCellVolumes, flowStreamHandle)) {
            std::cerr << "[CFDSolver] setBoundaryConditions: projectionSolver.build failed\n";
            return false;
        }
    }

    return true;
}

bool CFDSolver::initializeVelocity(const glm::vec3& initialVelocity) {
    if (!std::isfinite(initialVelocity.x) ||
        !std::isfinite(initialVelocity.y) ||
        !std::isfinite(initialVelocity.z)) {
        std::cerr << "[CFDSolver] initializeVelocity FAILED: non-finite initial velocity" << std::endl;
        return false;
    }

    if (!initialized || fluidNodeCount == 0 || !flowStreamHandle) {
        return true;
    }

    failed = false;

    // Reinitialization discards stale subcycle plan before next run
    if (pinnedSubcycleTimestep != nullptr) {
        *pinnedSubcycleTimestep = cudaSubcycle::SubcycleTimestep{};
        pinnedSubcycleTimestep->status = cudaSubcycle::SubcycleStatus::Ok;
    }

    cudaSetDevice(cudaDevice);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(flowStreamHandle);

    fluidkernels::launchInitializeVelocity(
        fluidNodeCount, initialVelocity, d_velocity.get(), d_velocityStar.get(), stream);

    if (!projectionSolver.projectVelocity(
            fluidNodeCount, faceCount, boundaryFaceCount,
            d_faceTopology.get(), d_faceFluxVectors.get(), d_boundaryFaces.get(),
            d_boundaryConditions.get(),
            d_cellVolumes.get(),
            d_faceFlux.get(), d_boundaryFlux.get(),
            d_velocityStar.get(), d_velocity.get(),
            nullptr,
            flowStreamHandle)) {
        std::cerr << "[CFDSolver] initializeVelocity: projectVelocity failed" << std::endl;
        return false;
    }

    if (externalVelocity && fluidNodeCount > 0) {
        fluidkernels::publishVelocity(fluidNodeCount, d_velocity.get(), externalVelocity, projectionSolver.getPublishGate(), stream);
    }
    if (externalVelocityGradients && fluidNodeCount > 0) {
        fluidkernels::launchReconstructVelocityGradients(
            fluidNodeCount, faceCount, boundaryFaceCount,
            d_faceTopology.get(), d_faceFluxVectors.get(),
            d_boundaryFaces.get(), d_boundaryConditions.get(),
            d_cellVolumes.get(), d_velocity.get(),
            externalVelocityGradients, stream);
    }

    if (externalCanonicalFaceFlux && externalCanonicalFaceCount > 0) {
        if (!cudaUtils::cudaCheck(cudaMemsetAsync(externalCanonicalFaceFlux, 0, externalCanonicalFaceCount * sizeof(float), stream), "zero canonical face flux")) {
            return false;
        }
        fluidkernels::publishFineFaceFluxes(faceCount, d_faceGlobalIndex.get(), d_faceFlux.get(), externalCanonicalFaceFlux, projectionSolver.getPublishGate(), stream);
    }
    if (externalCanonicalBoundaryFlux && externalCanonicalBoundaryFaceCount > 0) {
        if (!cudaUtils::cudaCheck(cudaMemsetAsync(externalCanonicalBoundaryFlux, 0, externalCanonicalBoundaryFaceCount * sizeof(float), stream), "zero canonical boundary flux")) {
            return false;
        }
        fluidkernels::publishFineBoundaryFluxes(boundaryFaceCount, d_boundaryGlobalIndex.get(), d_boundaryFlux.get(), externalCanonicalBoundaryFlux, projectionSolver.getPublishGate(), stream);
    }

    if (!cudaUtils::cudaCheck(cudaStreamSynchronize(stream), "sync initializeVelocity")) {
        return false;
    }

    const auto pressureResult = projectionSolver.pollNativeSolveResult();
    if (pressureResult == CFDProjectionSolver::SolvePollResult::Failed ||
        pressureResult == CFDProjectionSolver::SolvePollResult::Error) {
        std::cerr << "[CFDSolver] initializeVelocity: native pressure solve failed" << std::endl;
        return false;
    }

    return true;
}

CFDSolver::StepOutcome CFDSolver::stepCfd(
    float kinematicViscosity,
    float macroDt,
    float targetCourant,
    uint64_t completionValue,
    uint64_t releaseWaitValue) {
    if (failed) {
        return StepOutcome::Failed;
    }

    switch (projectionSolver.pollNativeSolveResult()) {
    case CFDProjectionSolver::SolvePollResult::Error:
        std::cerr << "[CFDSolver] native pressure status query failed; halting until reinitialization" << std::endl;
        failed = true;
        return StepOutcome::Failed;
    case CFDProjectionSolver::SolvePollResult::Failed:
        std::cerr << "[CFDSolver] native pressure solve failed; halting until reinitialization" << std::endl;
        failed = true;
        return StepOutcome::Failed;
    case CFDProjectionSolver::SolvePollResult::Pending:
        if (projectionSolver.hasPendingNativeSolve()) {
                    return StepOutcome::Deferred;
        }
        break;
    case CFDProjectionSolver::SolvePollResult::Succeeded:
        if (pinnedSubcycleTimestep != nullptr &&
            pinnedSubcycleTimestep->status != cudaSubcycle::SubcycleStatus::Ok) {
            std::cerr << "[CFDSolver] subcycle plan rejected: status="
                      << static_cast<uint32_t>(pinnedSubcycleTimestep->status)
                      << " (cap " << MaxMomentumSubsteps << " substeps)"
                      << "; halting until reinitialization" << std::endl;
            failed = true;
            return StepOutcome::Failed;
        }
        break;
    }

    if (!std::isfinite(macroDt) || !(macroDt > 0.0f) ||
        !std::isfinite(targetCourant) || !(targetCourant > 0.0f) ||
        !std::isfinite(kinematicViscosity) || kinematicViscosity < 0.0f) {
        std::cerr << "[CFDSolver] stepCfd FAILED: invalid parameters\n";
        failed = true;
        return StepOutcome::Failed;
    }

    if (!initialized || fluidNodeCount == 0 || !flowStreamHandle) {
        return StepOutcome::Deferred;
    }

    cudaSetDevice(cudaDevice);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(flowStreamHandle);

    if (phaseMarkersRecorded) {
        const cudaError_t markerQuery = cudaEventQuery(phaseMarkers[PhaseMarkerCount - 1].handle());
        if (markerQuery == cudaSuccess) {
            phaseMarkersRecorded = false;
            float waitMs = 0.0f;
            float momentumMs = 0.0f;
            float projectionMs = 0.0f;
            float publishMs = 0.0f;
            float totalMs = 0.0f;
            phaseMarkers[1].elapsedSince(phaseMarkers[0], waitMs);
            phaseMarkers[2].elapsedSince(phaseMarkers[1], momentumMs);
            phaseMarkers[3].elapsedSince(phaseMarkers[2], projectionMs);
            phaseMarkers[4].elapsedSince(phaseMarkers[3], publishMs);
            phaseMarkers[4].elapsedSince(phaseMarkers[0], totalMs);
            if (totalMs > 0.0f) {
                lastStepGpuMs = totalMs;
            }
        } else if (markerQuery != cudaErrorNotReady) {
            (void)cudaGetLastError();
            phaseMarkersRecorded = false;
        }
    }
    phaseMarkers[0].record(stream);

    if (releaseSemaphore != nullptr && releaseWaitValue > 0) {
        cudaExternalSemaphoreWaitParams waitParams{};
        waitParams.flags = 0;
        waitParams.params.fence.value = releaseWaitValue;
        if (!cudaUtils::cudaCheck(cudaWaitExternalSemaphoresAsync(&releaseSemaphore, &waitParams, 1, stream),
                                  "cudaWaitExternalSemaphoresAsync flowStream")) {
            failed = true;
            return StepOutcome::Failed;
        }
    }
    phaseMarkers[1].record(stream);

    if (momentumExec == nullptr || !(targetCourant == builtTargetCourant)) {
        std::cerr << "[CFDSolver] momentum subcycle graph was not prepared for this step" << std::endl;
        failed = true;
        return StepOutcome::Failed;
    }

    if (!d_macroDt.uploadAsync(&macroDt, 1, stream) ||
        !d_kinematicViscosity.uploadAsync(&kinematicViscosity, 1, stream) ||
        !d_referenceVelocity.uploadAsync(&referenceVelocity, 1, stream)) {
        failed = true;
        return StepOutcome::Failed;
    }

    if (!cudaUtils::cudaCheck(cudaGraphLaunch(momentumExec, stream), "launch momentum subcycle graph")) {
        failed = true;
        return StepOutcome::Failed;
    }

    if (!d_subcycleTimestep.downloadAsync(pinnedSubcycleTimestep, 1, stream)) {
        failed = true;
        return StepOutcome::Failed;
    }
    phaseMarkers[2].record(stream);

    if (!projectionSolver.projectVelocity(
            fluidNodeCount, faceCount, boundaryFaceCount,
            d_faceTopology.get(), d_faceFluxVectors.get(), d_boundaryFaces.get(),
            d_boundaryConditions.get(),
            d_cellVolumes.get(),
            d_faceFlux.get(), d_boundaryFlux.get(),
            d_velocityStar.get(), d_velocity.get(),
            reinterpret_cast<uint32_t*>(&d_subcycleTimestep.get()->status),
            flowStreamHandle)) {
        std::cerr << "[CFDSolver] stepCfd: projectionSolver.projectVelocity failed (single projection)\n";
        cudaStreamSynchronize(stream);
        failed = true;
        return StepOutcome::Failed;
    }
    phaseMarkers[3].record(stream);

    if (externalVelocity && fluidNodeCount > 0) {
        fluidkernels::publishVelocity(fluidNodeCount, d_velocity.get(), externalVelocity, projectionSolver.getPublishGate(), stream);
    }
    if (externalVelocityGradients && fluidNodeCount > 0) {
        fluidkernels::launchReconstructVelocityGradients(
            fluidNodeCount, faceCount, boundaryFaceCount,
            d_faceTopology.get(), d_faceFluxVectors.get(),
            d_boundaryFaces.get(), d_boundaryConditions.get(),
            d_cellVolumes.get(), d_velocity.get(),
            externalVelocityGradients, stream);
    }

    if (externalCanonicalFaceFlux && externalCanonicalFaceCount > 0) {
        fluidkernels::publishFineFaceFluxes(faceCount, d_faceGlobalIndex.get(), d_faceFlux.get(), externalCanonicalFaceFlux, projectionSolver.getPublishGate(), stream);
    }
    if (externalCanonicalBoundaryFlux && externalCanonicalBoundaryFaceCount > 0) {
        fluidkernels::publishFineBoundaryFluxes(boundaryFaceCount, d_boundaryGlobalIndex.get(), d_boundaryFlux.get(), externalCanonicalBoundaryFlux, projectionSolver.getPublishGate(), stream);
    }

    if (completionSemaphore != nullptr && completionValue > 0) {
        cudaExternalSemaphoreSignalParams signalParams{};
        signalParams.flags = 0;
        signalParams.params.fence.value = completionValue;
        if (!cudaUtils::cudaCheck(cudaSignalExternalSemaphoresAsync(&completionSemaphore, &signalParams, 1, stream),
                                  "cudaSignalExternalSemaphoresAsync flowStream")) {
            failed = true;
            return StepOutcome::Failed;
        }
    }

    phaseMarkers[4].record(stream);
    phaseMarkersRecorded = true;
    ++stepIndex;

    return StepOutcome::Enqueued;
}

bool CFDSolver::reconstructVelocityGradients(glm::vec4* externalGradients, cudaStream_t inStream) {
    if (!initialized || fluidNodeCount == 0 || !externalGradients) {
        return false;
    }
    cudaStream_t stream = inStream ? inStream : reinterpret_cast<cudaStream_t>(flowStreamHandle);
    cudaSetDevice(cudaDevice);
    fluidkernels::launchReconstructVelocityGradients(
        fluidNodeCount, faceCount, boundaryFaceCount,
        d_faceTopology.get(), d_faceFluxVectors.get(),
        d_boundaryFaces.get(), d_boundaryConditions.get(),
        d_cellVolumes.get(), d_velocity.get(),
        externalGradients, stream);
    return true;
}

} // namespace cfd
