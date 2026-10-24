#include "CFDProjectionSolver.hpp"

#include <iostream>
#include <map>
#include <limits>
#include <cmath>
#include <algorithm>
#include "../cuda/CudaCheck.cuh"

namespace cfd {

namespace {

uint32_t nativeIterationLimit(int authoredIterations) {
    return (authoredIterations > 0) ? static_cast<uint32_t>(authoredIterations) : 1000u;
}

float nativeTolerance(float authoredTolerance) {
    return (authoredTolerance > 0.0f) ? authoredTolerance : 1e-4f;
}

} // anonymous namespace

CFDProjectionSolver::~CFDProjectionSolver() {
    cleanup();
}

bool CFDProjectionSolver::initialize(int inCudaDevice) {
    cudaDevice = inCudaDevice;
    return true;
}

bool CFDProjectionSolver::build(
    uint32_t inCellCount,
    const std::vector<CFDFaceGpu>& faces,
    const std::vector<CFDBoundaryFaceGpu>& boundaryFaces,
    const std::array<BoundaryCondition, 7>& boundaryConditions,
    const std::vector<uint32_t>& component,
    uint32_t componentCount,
    const std::vector<float>& cellVolumes,
    void* flowStreamHandle) {
    cleanup();

    cellCount = inCellCount;
    if (cellCount == 0) {
        return true;
    }

    cudaSetDevice(cudaDevice);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(flowStreamHandle);

    // Identify connected components that reach a PressureOutlet (bc.type == BoundaryType::PressureOutlet)
    std::vector<bool> componentHasOutlet(componentCount, false);
    for (const auto& cbf : boundaryFaces) {
        if (cbf.boundaryCondition < boundaryConditions.size()) {
            if (boundaryConditions[cbf.boundaryCondition].type == BoundaryType::PressureOutlet && cbf.cell < cellCount) {
                const uint32_t comp = (cbf.cell < component.size()) ? component[cbf.cell] : 0;
                if (comp < componentCount) {
                    componentHasOutlet[comp] = true;
                }
            }
        }
    }

    // Only pin a reference cell for components that have NO outlet (pure Neumann boundaries)
    std::vector<uint32_t> componentPinnedCell(componentCount, UINT32_MAX);
    std::vector<float> componentMaxVol(componentCount, -1.0f);

    for (uint32_t i = 0; i < cellCount; ++i) {
        const uint32_t comp = (i < component.size()) ? component[i] : 0;
        if (comp < componentCount && !componentHasOutlet[comp]) {
            if (i < cellVolumes.size() && cellVolumes[i] > componentMaxVol[comp]) {
                componentMaxVol[comp] = cellVolumes[i];
                componentPinnedCell[comp] = i;
            }
        }
    }

    std::vector<uint8_t> hostIsPinned(cellCount, 0);
    for (uint32_t comp = 0; comp < componentCount; ++comp) {
        if (!componentHasOutlet[comp] && componentPinnedCell[comp] != UINT32_MAX) {
            hostIsPinned[componentPinnedCell[comp]] = 1;
        }
    }

    // Build face adjacency via flat CSR
    std::vector<uint32_t> adjDeg(cellCount, 0);
    for (const auto& cf : faces) {
        if (cf.cellA < cellCount && cf.cellB < cellCount && cf.cellA != cf.cellB) {
            adjDeg[cf.cellA]++;
            adjDeg[cf.cellB]++;
        }
    }

    std::vector<uint32_t> adjOffsets(cellCount + 1, 0);
    for (uint32_t i = 0; i < cellCount; ++i) {
        adjOffsets[i + 1] = adjOffsets[i] + adjDeg[i];
    }

    struct AdjEntry {
        uint32_t neighbor;
        float weight;
    };
    std::vector<AdjEntry> adjEntries(adjOffsets.back());
    std::vector<uint32_t> adjCurPos = adjOffsets;
    for (const auto& cf : faces) {
        if (cf.cellA < cellCount && cf.cellB < cellCount && cf.cellA != cf.cellB) {
            adjEntries[adjCurPos[cf.cellA]++] = {cf.cellB, cf.projectionWeight};
            adjEntries[adjCurPos[cf.cellB]++] = {cf.cellA, cf.projectionWeight};
        }
    }

    // Accumulate outlet Dirichlet boundary weights (PressureOutlet: psi = 0 at outlet face)
    std::vector<float> outletWeights(cellCount, 0.0f);
    for (const auto& cbf : boundaryFaces) {
        if (cbf.boundaryCondition < boundaryConditions.size()) {
            if (boundaryConditions[cbf.boundaryCondition].type == BoundaryType::PressureOutlet && cbf.cell < cellCount) {
                outletWeights[cbf.cell] += cbf.projectionWeight;
            }
        }
    }

    // Assemble Sparse Symmetric CSR Laplacian for Pressure Poisson equation
    std::vector<int> rowOffsets;
    std::vector<int> columnIndices;
    std::vector<float> values;

    rowOffsets.reserve(cellCount + 1);
    rowOffsets.push_back(0);

    std::vector<std::pair<int, float>> rowBuffer;
    rowBuffer.reserve(32);

    for (uint32_t i = 0; i < cellCount; ++i) {
        if (hostIsPinned[i] != 0) {
            columnIndices.push_back(static_cast<int>(i));
            values.push_back(1.0f);
            rowOffsets.push_back(static_cast<int>(columnIndices.size()));
            continue;
        }

        rowBuffer.clear();
        float diagSum = outletWeights[i];

        const uint32_t aStart = adjOffsets[i];
        const uint32_t aEnd = adjOffsets[i + 1];
        for (uint32_t a = aStart; a < aEnd; ++a) {
            const auto& entry = adjEntries[a];
            diagSum += entry.weight;
            if (hostIsPinned[entry.neighbor] == 0) {
                rowBuffer.push_back({static_cast<int>(entry.neighbor), -entry.weight});
            }
        }

        rowBuffer.push_back({static_cast<int>(i), diagSum});

        std::sort(rowBuffer.begin(), rowBuffer.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        for (const auto& [col, val] : rowBuffer) {
            columnIndices.push_back(col);
            values.push_back(val);
        }
        rowOffsets.push_back(static_cast<int>(columnIndices.size()));
    }

    pressureRowOffsets = std::move(rowOffsets);
    pressureColumnIndices = std::move(columnIndices);
    pressureValues = std::move(values);

    if (!d_pressureRhs.allocate(cellCount) ||
        !d_psi.allocate(cellCount) ||
        !d_isPinned.allocate(cellCount) ||
        !d_workImpulse.allocate(cellCount)) {
        std::cerr << "[CFDProjectionSolver] buffer allocation failed\n";
        cleanup();
        return false;
    }

    if (!d_isPinned.uploadAsync(hostIsPinned.data(), cellCount, stream) ||
        !cudaUtils::cudaCheck(cudaStreamSynchronize(stream), "sync hostIsPinned upload")) {
        cleanup();
        return false;
    }

    if (!cudaUtils::cudaCheck(
            cudaMemsetAsync(d_psi.get(), 0, cellCount * sizeof(float), stream),
            "zero pressure psi")) {
        cleanup();
        return false;
    }

    if (!setupNativePcg(stream)) {
        std::cerr << "[CFDProjectionSolver] pressure PCG setup failed\n";
        cleanup();
        return false;
    }

    std::cerr << "[CFDProjectionSolver] pressure solver=native-pcg"
              << " cells=" << cellCount
              << " nnz=" << pressureValues.size()
              << " maxIterations=" << nativeIterationLimit(pressureMaxIterations)
              << " tolerance=" << nativeTolerance(pressureTolerance)
              << std::endl;
    return true;
}

bool CFDProjectionSolver::setupNativePcg(cudaStream_t stream) {
    cleanupNativePcg();

    const size_t nnz = pressureValues.size();
    if (!d_pressureRowOffsets.allocate(pressureRowOffsets.size()) ||
        !d_pressureColumnIndices.allocate(nnz) ||
        !d_pressureValues.allocate(nnz) ||
        !d_pressureInvDiag.allocate(cellCount) ||
        !d_pressureAp.allocate(cellCount) ||
        !d_pressureR.allocate(cellCount) ||
        !d_pressureZ.allocate(cellCount) ||
        !d_pressureP.allocate(cellCount) ||
        !d_pcgStatus.allocate(1) ||
        !d_pcgConfig.allocate(1) ||
        !d_publishGate.allocate(1)) {
        cleanupNativePcg();
        return false;
    }

    if (cudaMallocHost(reinterpret_cast<void**>(&pinnedPcgStatus), sizeof(cudaPcg::PcgStatus)) != cudaSuccess ||
        cudaEventCreateWithFlags(&pcgStatusEvent, cudaEventDisableTiming) != cudaSuccess) {
        cleanupNativePcg();
        return false;
    }

    if (!d_pressureRowOffsets.uploadAsync(pressureRowOffsets.data(), pressureRowOffsets.size(), stream) ||
        !d_pressureColumnIndices.uploadAsync(pressureColumnIndices.data(), nnz, stream) ||
        !d_pressureValues.uploadAsync(pressureValues.data(), nnz, stream)) {
        cleanupNativePcg();
        return false;
    }

    cudaPcg::launchComputeInvDiag(
        cellCount,
        d_pressureRowOffsets.get(),
        d_pressureColumnIndices.get(),
        d_pressureValues.get(),
        d_pressureInvDiag.get(),
        stream);

    cudaPcg::PcgConfig config{};
    config.maxIterations = nativeIterationLimit(pressureMaxIterations);
    config.tolerance = nativeTolerance(pressureTolerance);
    if (!d_pcgConfig.uploadAsync(&config, 1, stream) ||
        !cudaUtils::cudaCheck(cudaStreamSynchronize(stream), "sync native PCG setup")) {
        cleanupNativePcg();
        return false;
    }

    pressurePcgExec = cudaPcg::buildPcgGraphExec(
        cellCount,
        d_pressureRowOffsets.get(),
        d_pressureColumnIndices.get(),
        d_pressureValues.get(),
        d_pressureInvDiag.get(),
        d_psi.get(),
        d_pressureRhs.get(),
        d_pressureAp.get(),
        d_pressureR.get(),
        d_pressureZ.get(),
        d_pressureP.get(),
        d_pcgStatus.get(),
        d_pcgConfig.get());
    if (!pressurePcgExec) {
        cleanupNativePcg();
        return false;
    }

    if (cudaMemsetAsync(d_publishGate.get(), 0, sizeof(int), stream) != cudaSuccess) {
        cleanupNativePcg();
        return false;
    }

    return true;
}

void CFDProjectionSolver::cleanupNativePcg() {
    if (pressurePcgExec) {
        cudaGraphExecDestroy(pressurePcgExec);
        pressurePcgExec = nullptr;
    }
    if (pcgStatusEvent) {
        cudaEventDestroy(pcgStatusEvent);
        pcgStatusEvent = nullptr;
    }
    if (pinnedPcgStatus) {
        cudaFreeHost(pinnedPcgStatus);
        pinnedPcgStatus = nullptr;
    }
    d_pressureRowOffsets.reset();
    d_pressureColumnIndices.reset();
    d_pressureValues.reset();
    d_pressureInvDiag.reset();
    d_pressureAp.reset();
    d_pressureR.reset();
    d_pressureZ.reset();
    d_pressureP.reset();
    d_pcgStatus.reset();
    d_pcgConfig.reset();
    d_publishGate.reset();
    pcgStatusPending = false;
}

CFDProjectionSolver::SolvePollResult CFDProjectionSolver::pollNativeSolveResult() {
    if (!pcgStatusPending) {
        return SolvePollResult::Pending;
    }
    if (!pcgStatusEvent) {
        pcgStatusPending = false;
        return SolvePollResult::Error;
    }
    const cudaError_t query = cudaEventQuery(pcgStatusEvent);
    if (query == cudaErrorNotReady) {
        return SolvePollResult::Pending;
    }
    if (query != cudaSuccess) {
        cudaGetLastError();
        pcgStatusPending = false;
        return SolvePollResult::Error;
    }
    pcgStatusPending = false;
    const bool success = pinnedPcgStatus->converged != 0 &&
                         pinnedPcgStatus->failed == 0 &&
                         pinnedPcgStatus->exhausted == 0;
    if (!success) {
        std::cerr << "[CFDProjectionSolver] native PCG result: converged="
                  << pinnedPcgStatus->converged
                  << " failed=" << pinnedPcgStatus->failed
                  << " exhausted=" << pinnedPcgStatus->exhausted
                  << " iterations=" << pinnedPcgStatus->iterationCount
                  << " residual=" << pinnedPcgStatus->residual
                  << " residual0=" << pinnedPcgStatus->residual0 << std::endl;
    }
    return success ? SolvePollResult::Succeeded : SolvePollResult::Failed;
}

bool CFDProjectionSolver::projectVelocity(
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
    void* flowStreamHandle) {
    if (inCellCount == 0 || !flowStreamHandle) {
        return true;
    }

    cudaSetDevice(cudaDevice);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(flowStreamHandle);

    // 1. Interpolate provisional face fluxes phi* from velocityStar directly into published flux buffers
    fluidkernels::launchProvisionalFlux(
        faceCount, boundaryFaceCount,
        d_faces, d_boundaryFaces,
        d_boundaryConditions,
        d_velocityStar,
        d_faceFlux, d_boundaryFlux,
        stream);

    // 2. Accumulate pressure RHS b = -div(phi*) into d_pressureRhs on flowStream
    fluidkernels::launchBuildPressureRhs(
        inCellCount, faceCount, boundaryFaceCount,
        d_faces, d_boundaryFaces,
        d_faceFlux, d_boundaryFlux,
        d_pressureRhs.get(), stream);

    // 3. Set RHS = 0 on pinned reference nodes (for pure Neumann closed components)
    fluidkernels::launchZeroMasked(inCellCount, d_isPinned.get(), d_pressureRhs.get(), stream);

    if (!pressurePcgExec) {
        std::cerr << "[CFDProjectionSolver] pressure PCG graph missing\n";
        cudaMemsetAsync(d_psi.get(), 0, inCellCount * sizeof(float), stream);
        return false;
    }

    cudaPcg::PcgConfig config{};
    config.maxIterations = nativeIterationLimit(pressureMaxIterations);
    config.tolerance = nativeTolerance(pressureTolerance);
    if (!d_pcgConfig.uploadAsync(&config, 1, stream)) {
        cudaMemsetAsync(d_psi.get(), 0, inCellCount * sizeof(float), stream);
        return false;
    }

    if (!cudaUtils::cudaCheck(cudaGraphLaunch(pressurePcgExec, stream), "launch CFD pressure PCG graph")) {
        cudaMemsetAsync(d_psi.get(), 0, inCellCount * sizeof(float), stream);
        return false;
    }
    cudaPcg::launchResolveSolutionStatus(
        d_pcgStatus.get(), d_scheduleStatus, d_psi.get(), inCellCount, d_publishGate.get(), stream);
    if (!d_pcgStatus.downloadAsync(pinnedPcgStatus, 1, stream) ||
        !cudaUtils::cudaCheck(cudaEventRecord(pcgStatusEvent, stream), "record PCG status event")) {
        cudaMemsetAsync(d_psi.get(), 0, inCellCount * sizeof(float), stream);
        return false;
    }
    pcgStatusPending = true;

    // 7. Ensure pinned nodes have zero potential
    fluidkernels::launchZeroMasked(inCellCount, d_isPinned.get(), d_psi.get(), stream);

    // 8. Correct internal face fluxes: phi_corr = phi - W * (psi_B - psi_A)
    fluidkernels::launchCorrectFluxes(faceCount, d_faces, d_psi.get(), d_faceFlux, stream);

    // 9. Correct outlet boundary face fluxes: phi_corr = phi + W * psi_cell
    if (boundaryFaceCount > 0 && d_boundaryConditions) {
        fluidkernels::launchCorrectBoundaryFluxes(
            boundaryFaceCount, d_boundaryFaces, d_boundaryConditions, d_psi.get(), d_boundaryFlux, stream);
    }

    // 10. Direct velocity correction: u^{n+1} = u* - impulse / V_cell
    fluidkernels::launchCorrectVelocity(
        inCellCount, faceCount, boundaryFaceCount,
        d_faces, d_boundaryFaces,
        d_boundaryConditions,
        d_cellVolumes,
        d_psi.get(),
        d_velocityStar, d_velocity,
        d_workImpulse.get(), stream);

    return true;
}

void CFDProjectionSolver::cleanup() {
    cleanupNativePcg();
    d_pressureRhs.reset();
    d_psi.reset();
    d_isPinned.reset();
    d_workImpulse.reset();
    cellCount = 0;
}

void CFDProjectionSolver::setPressureSolverParameters(int maxIterations, float tolerance) {
    if (maxIterations < 0) maxIterations = 0;
    if (tolerance < 0.0f) tolerance = 0.0f;
    if (maxIterations == pressureMaxIterations && tolerance == pressureTolerance) {
        return;
    }
    pressureMaxIterations = maxIterations;
    pressureTolerance = tolerance;
}

} // namespace cfd
