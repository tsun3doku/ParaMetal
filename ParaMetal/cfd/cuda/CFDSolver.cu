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

    if (!cudaUtils::cudaCheck(cudaEventCreate(&stepStartEvent), "create stepStartEvent") ||
        !cudaUtils::cudaCheck(cudaEventCreate(&stepEndEvent), "create stepEndEvent")) {
        cleanup();
        return false;
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
    uint32_t inCanonicalBoundaryFaceCount) {
    externalVelocity = velocityVec4;
    externalCanonicalFaceFlux = inCanonicalFaceFlux;
    externalCanonicalFaceCount = inCanonicalFaceCount;
    externalCanonicalBoundaryFlux = inCanonicalBoundaryFlux;
    externalCanonicalBoundaryFaceCount = inCanonicalBoundaryFaceCount;
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
    projectionSolver.cleanup();
    destroyMomentumSubcycleGraph();

    fluidNodeCount = 0;
    faceCount = 0;
    boundaryFaceCount = 0;
    failed = false;

    releaseTopologyBuffers();

    hostFaces.clear();
    hostBoundaryFaces.clear();
    hostCellVolumes.clear();
    component.clear();
    componentCount = 0;
}

void CFDSolver::releaseTopologyBuffers() {
    d_momentumAccum.reset();
    d_cellViscousWeight.reset();
    d_cellVolumes.reset();
    d_velocity.reset();
    d_velocityStar.reset();
    d_faces.reset();
    d_boundaryFaces.reset();
    d_faceGlobalIndex.reset();
    d_boundaryGlobalIndex.reset();
    d_faceFlux.reset();
    d_boundaryFlux.reset();
    d_outflowRates.reset();
    d_cflPacked.reset();
    d_boundaryConditions.reset();
}

void CFDSolver::cleanup() {
    if (flowStreamHandle != nullptr) {
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(flowStreamHandle));
    }

    clearTopology();

    if (stepStartEvent != nullptr) {
        cudaEventDestroy(stepStartEvent);
        stepStartEvent = nullptr;
    }
    if (stepEndEvent != nullptr) {
        cudaEventDestroy(stepEndEvent);
        stepEndEvent = nullptr;
    }
    lastStepGpuMs = 0.0f;

    if (flowStreamHandle != nullptr) {
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(flowStreamHandle);
        cudaStreamDestroy(stream);
        flowStreamHandle = nullptr;
    }

    externalVelocity = nullptr;
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
    const std::vector<topology::Fragment>& fragments,
    const std::vector<topology::Face>& faces,
    const std::vector<topology::BoundaryFace>& boundaryFaces,
    const std::vector<uint32_t>& globalFragmentToFluidNode,
    const std::vector<uint32_t>& fluidNodeToGlobalFragment) {
    if (!initialized || cudaDevice < 0) {
        std::cerr << "[CFDSolver] buildTopology called before initialize" << std::endl;
        return false;
    }

    clearTopology();
    cudaSetDevice(cudaDevice);

    CFDTopology topology;
    if (!buildCFDTopology(fragments, faces, boundaryFaces, globalFragmentToFluidNode, fluidNodeToGlobalFragment, topology)) {
        std::cerr << "[CFDSolver] buildTopology FAILED: buildCFDTopology failed\n";
        clearTopology();
        return false;
    }

    fluidNodeCount = topology.fluidNodeCount;
    faceCount = static_cast<uint32_t>(topology.faces.size());
    boundaryFaceCount = static_cast<uint32_t>(topology.boundaryFaces.size());

    if (fluidNodeCount == 0) {
        return true;
    }

    this->hostFaces = std::move(topology.faces);
    this->hostBoundaryFaces = std::move(topology.boundaryFaces);
    this->hostCellVolumes = std::move(topology.cellVolumes);
    this->component = std::move(topology.component);
    this->componentCount = topology.componentCount;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(flowStreamHandle);

    // Allocate GPU buffers via CudaBuffer
    if (!d_cellVolumes.allocate(fluidNodeCount) ||
        !d_velocity.allocate(fluidNodeCount) ||
        !d_velocityStar.allocate(fluidNodeCount) ||
        !d_outflowRates.allocate(fluidNodeCount) ||
        !d_cflPacked.allocate(1) ||
        !d_boundaryConditions.allocate(7)) {
        clearTopology();
        return false;
    }

    if (faceCount > 0) {
        if (!d_faces.allocate(faceCount) ||
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

    // Upload topology and boundary buffers
    auto initialGpuBcs = toGpuArray(boundaryConditions);
    if (!d_cellVolumes.uploadAsync(this->hostCellVolumes.data(), fluidNodeCount, stream) ||
        !d_boundaryConditions.uploadAsync(initialGpuBcs.data(), 7, stream) ||
        !cudaUtils::cudaCheck(cudaMemsetAsync(d_velocity.get(), 0, fluidNodeCount * sizeof(glm::vec4), stream), "zero d_velocity") ||
        !cudaUtils::cudaCheck(cudaMemsetAsync(d_velocityStar.get(), 0, fluidNodeCount * sizeof(glm::vec3), stream), "zero d_velocityStar")) {
        clearTopology();
        return false;
    }

    if (faceCount > 0) {
        if (!d_faces.uploadAsync(this->hostFaces.data(), faceCount, stream) ||
            !d_faceGlobalIndex.uploadAsync(topology.faceGlobalIndex.data(), faceCount, stream) ||
            !cudaUtils::cudaCheck(cudaMemsetAsync(d_faceFlux.get(), 0, faceCount * sizeof(float), stream), "zero d_faceFlux")) {
            clearTopology();
            return false;
        }
    }

    if (boundaryFaceCount > 0) {
        if (!d_boundaryFaces.uploadAsync(this->hostBoundaryFaces.data(), boundaryFaceCount, stream) ||
            !d_boundaryGlobalIndex.uploadAsync(topology.boundaryGlobalIndex.data(), boundaryFaceCount, stream) ||
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

    if (!projectionSolver.build(fluidNodeCount, this->hostFaces, this->hostBoundaryFaces, fullBcs, this->component, this->componentCount, this->hostCellVolumes, flowStreamHandle)) {
        std::cerr << "[CFDSolver] buildTopology FAILED: projectionSolver.build failed\n";
        clearTopology();
        return false;
    }

    return true;
}

bool CFDSolver::buildMomentumViscousWeights(const std::array<BoundaryCondition, 7>& bcs) {
    if (fluidNodeCount == 0) return true;
    if (!d_momentumAccum.allocate(fluidNodeCount) || !d_cellViscousWeight.allocate(fluidNodeCount)) {
        return false;
    }
    std::vector<float> diagWeights(fluidNodeCount, 0.0f);
    for (const auto& cf : hostFaces) {
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
    return d_cellViscousWeight.upload(diagWeights.data(), fluidNodeCount);
}

bool CFDSolver::buildMomentumSubcycleGraph(float targetCourant) {
    destroyMomentumSubcycleGraph();

    if (!d_maxOutflowRate.allocate(1) ||
        !d_subcycleInvalid.allocate(1) ||
        !d_subcycleIteration.allocate(1) ||
        !d_subcycleTimestep.allocate(1) ||
        !d_macroDt.allocate(1) ||
        !d_kinematicViscosity.allocate(1)) {
        destroyMomentumSubcycleGraph();
        return false;
    }
    if (cudaMallocHost(reinterpret_cast<void**>(&pinnedSubcycleTimestep), sizeof(cudaSubcycle::SubcycleTimestep)) != cudaSuccess) {
        pinnedSubcycleTimestep = nullptr;
        destroyMomentumSubcycleGraph();
        return false;
    }
    *pinnedSubcycleTimestep = cudaSubcycle::SubcycleTimestep{};

    const float zero = 0.0f;
    if (!d_macroDt.upload(&zero, 1) || !d_kinematicViscosity.upload(&zero, 1)) {
        destroyMomentumSubcycleGraph();
        return false;
    }

    constexpr uint32_t MaxSubsteps = 128;

    cudaGraph_t graph = nullptr;
    if (!cudaUtils::cudaCheck(cudaGraphCreate(&graph, 0), "momentum subcycle graph create")) {
        destroyMomentumSubcycleGraph();
        return false;
    }

    cudaGraphNode_t prev = nullptr;
    if (fluidkernels::appendMomentumOutflowRate(
            graph, prev, fluidNodeCount, faceCount, boundaryFaceCount,
            d_faces.get(), d_boundaryFaces.get(), d_faceFlux.get(), d_boundaryFlux.get(),
            d_cellVolumes.get(), d_outflowRates.get(), d_maxOutflowRate.get(), d_subcycleInvalid.get()) != cudaSuccess) {
        cudaGraphDestroy(graph);
        destroyMomentumSubcycleGraph();
        return false;
    }

    cudaSubcycle::SubcycleSchedule schedule{};
    schedule.maxRate = d_maxOutflowRate.get();
    schedule.invalidFlag = d_subcycleInvalid.get();
    schedule.macroDt = d_macroDt.get();
    schedule.targetCfl = targetCourant;
    schedule.maxCount = MaxSubsteps;
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
            d_faces.get(), d_boundaryFaces.get(), d_boundaryConditions.get(),
            d_faceFlux.get(), d_boundaryFlux.get(), d_cellVolumes.get(), d_cellViscousWeight.get(),
            d_kinematicViscosity.get(), d_momentumAccum.get(), d_velocity.get(), d_velocityStar.get(),
            d_subcycleTimestep.get(), d_subcycleIteration.get()) != cudaSuccess ||
        cudaSubcycle::endSubcycleLoop(loop.body, body, handle, schedule) != cudaSuccess) {
        cudaGraphDestroy(graph);
        destroyMomentumSubcycleGraph();
        return false;
    }

    if (!cudaUtils::cudaCheck(cudaGraphInstantiate(&momentumExec, graph, nullptr, nullptr, 0), "momentum subcycle instantiate")) {
        cudaGraphDestroy(graph);
        momentumExec = nullptr;
        destroyMomentumSubcycleGraph();
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
    if (pinnedSubcycleTimestep) {
        cudaFreeHost(pinnedSubcycleTimestep);
        pinnedSubcycleTimestep = nullptr;
    }
    d_maxOutflowRate.reset();
    d_subcycleInvalid.reset();
    d_subcycleIteration.reset();
    d_subcycleTimestep.reset();
    d_macroDt.reset();
    d_kinematicViscosity.reset();
    builtTargetCourant = 0.0f;
}

void CFDSolver::setPressureSolverParameters(int maxIterations, float tolerance) {
    projectionSolver.setPressureSolverParameters(maxIterations, tolerance);
}

void CFDSolver::computeAndReduceOutflowRate(cudaStream_t stream) {
    if (fluidNodeCount == 0) return;

    // Compute cell outflow rates for CFL determination
    fluidkernels::launchComputeOutflowRate(
        fluidNodeCount, faceCount, boundaryFaceCount,
        d_faces.get(), d_boundaryFaces.get(),
        d_faceFlux.get(), d_boundaryFlux.get(),
        d_cellVolumes.get(), d_outflowRates.get(), stream);

    // Argmax over the per-cell rates
    if (!cudaUtils::cudaCheck(cudaMemsetAsync(d_cflPacked.get(), 0, sizeof(unsigned long long), stream), "zero d_cflPacked")) {
        return;
    }
    fluidkernels::launchCflArgmax(
        fluidNodeCount, d_outflowRates.get(), d_cflPacked.get(), stream);
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

    if (typeChanged && fluidNodeCount > 0 && !hostFaces.empty()) {
        auto fullBcs = toFullBoundaryArray(boundaryConditions);
        if (!buildMomentumViscousWeights(fullBcs)) {
            std::cerr << "[CFDSolver] setBoundaryConditions: buildMomentumViscousWeights failed\n";
            return false;
        }
        if (!projectionSolver.build(fluidNodeCount, hostFaces, hostBoundaryFaces, fullBcs, component, componentCount, hostCellVolumes, flowStreamHandle)) {
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

    failed = false;   // (re)initialization clears any terminal failure
    cudaSetDevice(cudaDevice);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(flowStreamHandle);

    fluidkernels::launchInitializeVelocity(
        fluidNodeCount, initialVelocity, d_velocity.get(), d_velocityStar.get(), stream);

    if (!projectionSolver.projectVelocity(
            fluidNodeCount, faceCount, boundaryFaceCount,
            d_faces.get(), d_boundaryFaces.get(),
            d_boundaryConditions.get(),
            d_cellVolumes.get(),
            d_faceFlux.get(), d_boundaryFlux.get(),
            d_velocityStar.get(), d_velocity.get(),
            nullptr,
            flowStreamHandle)) {
        std::cerr << "[CFDSolver] initializeVelocity: projectVelocity failed" << std::endl;
        return false;
    }

    // Publish velocity and per-face fluxes (cells == fluid nodes)
    if (externalVelocity && fluidNodeCount > 0) {
        fluidkernels::publishVelocity(fluidNodeCount, d_velocity.get(), externalVelocity, projectionSolver.getPublishGate(), stream);
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

    computeAndReduceOutflowRate(stream);

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
        return StepOutcome::Failed;   // terminal: awaiting reinitialization
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
            // Backpressure: previous solve still in flight
            return StepOutcome::Deferred;
        }
        break;
    case CFDProjectionSolver::SolvePollResult::Succeeded:
        if (pinnedSubcycleTimestep != nullptr &&
            pinnedSubcycleTimestep->status != cudaSubcycle::SubcycleStatus::Ok) {
            std::cerr << "[CFDSolver] subcycle plan rejected: status="
                      << static_cast<uint32_t>(pinnedSubcycleTimestep->status)
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

    if (stepEndEvent != nullptr && cudaEventQuery(stepEndEvent) == cudaSuccess) {
        float elapsedMs = 0.0f;
        if (cudaEventElapsedTime(&elapsedMs, stepStartEvent, stepEndEvent) == cudaSuccess && elapsedMs > 0.0f) {
            lastStepGpuMs = elapsedMs;
        }
    }
    if (stepStartEvent) {
        cudaEventRecord(stepStartEvent, stream);
    }

    // Wait on Vulkan release timeline semaphore before writing to published buffers
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

    if (momentumExec == nullptr || !(targetCourant == builtTargetCourant)) {
        if (!buildMomentumSubcycleGraph(targetCourant)) {
            std::cerr << "[CFDSolver] momentum subcycle graph build failed\n";
            failed = true;
            return StepOutcome::Failed;
        }
    }

    if (!d_macroDt.uploadAsync(&macroDt, 1, stream) ||
        !d_kinematicViscosity.uploadAsync(&kinematicViscosity, 1, stream)) {
        failed = true;
        return StepOutcome::Failed;
    }

    // Launch device-planned momentum subcycling graph
    if (!cudaUtils::cudaCheck(cudaGraphLaunch(momentumExec, stream), "launch momentum subcycle graph")) {
        failed = true;
        return StepOutcome::Failed;
    }

    // Asynchronous subcycle status readback
    if (!d_subcycleTimestep.downloadAsync(pinnedSubcycleTimestep, 1, stream)) {
        failed = true;
        return StepOutcome::Failed;
    }

    if (!projectionSolver.projectVelocity(
            fluidNodeCount, faceCount, boundaryFaceCount,
            d_faces.get(), d_boundaryFaces.get(),
            d_boundaryConditions.get(),
            d_cellVolumes.get(),
            d_faceFlux.get(), d_boundaryFlux.get(),
            d_velocityStar.get(), d_velocity.get(),
            reinterpret_cast<const uint32_t*>(&d_subcycleTimestep.get()->status),
            flowStreamHandle)) {
        std::cerr << "[CFDSolver] stepCfd: projectionSolver.projectVelocity failed (single projection)\n";
        failed = true;
        return StepOutcome::Failed;
    }

    computeAndReduceOutflowRate(stream);

    if (externalVelocity && fluidNodeCount > 0) {
        fluidkernels::publishVelocity(fluidNodeCount, d_velocity.get(), externalVelocity, projectionSolver.getPublishGate(), stream);
    }

    // Publish face fluxes into canonical global-face buffers
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

    if (stepEndEvent) {
        cudaEventRecord(stepEndEvent, stream);
    }

    return StepOutcome::Enqueued;
}

} // namespace cfd
