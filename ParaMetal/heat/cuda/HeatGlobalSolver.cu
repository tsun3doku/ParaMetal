#define NOMINMAX
#include <Windows.h>
#include "../HeatGlobalSolver.hpp"
#include "HeatGlobalSolverKernels.cuh"
#include "../../cuda/CudaBuffer.cuh"
#include "../../cuda/CudaCheck.cuh"
#include "../../cuda/CudaPcg.cuh"
#include "../../cuda/CudaGraphBuilder.cuh"
#include "../../cuda/CudaExternalBuffer.hpp"
#include "../../cuda/CudaVulkanDevice.hpp"
#include "../../vulkan/VulkanDevice.hpp"
#include "../../vulkan/VulkanExternalBuffer.hpp"

#include <cuda_runtime.h>
#include <vulkan/vulkan_win32.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

class HeatGlobalSolver::Implementation {
public:
    struct DeviceTemperatureNodes {
        float* d_temperatureA = nullptr;
        float* d_temperatureB = nullptr;
        cudaUtils::CudaBuffer<uint32_t> gatherOffsets;
        cudaUtils::CudaBuffer<uint32_t> gatherLocalNodeIds;
        cudaUtils::CudaBuffer<float> gatherWeights;
        cudaUtils::CudaBuffer<uint32_t> gatherSolverNodeIds;
        cudaUtils::CudaBuffer<uint32_t> scatterLocalNodeIds;
        cudaUtils::CudaBuffer<uint32_t> scatterSolverNodeIds;
        uint32_t gatherGroupCount = 0;
        uint32_t scatterCount = 0;
    };

    ~Implementation() { cleanup(); }

    bool initialize(
        VulkanDevice& vulkanDevice,
        const vem::VEMTraceSystem& traceSystem,
        const std::vector<float>& cellCapacityOverDt,
        const std::vector<float>& cellSourceAdditions,
        const std::vector<TemperatureNodes>& inputTemperatureNodes,
        const std::vector<float>& fixedCellTemperatures = {}) {
        cleanup();
        if (traceSystem.traceCount == 0 || traceSystem.cellCount == 0 ||
            cellCapacityOverDt.size() != traceSystem.cellCount ||
            traceSystem.rowOffsets.size() != traceSystem.traceCount + 1 ||
            traceSystem.columnIndices.size() != traceSystem.values.size() ||
            inputTemperatureNodes.empty()) {
            return false;
        }

        device = vulkanDevice.getDevice();
        cellCount = traceSystem.cellCount;
        traceCount = traceSystem.traceCount;
        matrixValueCount = static_cast<uint32_t>(traceSystem.values.size());

        cudaDevice = cudaVulkan::findDevice(vulkanDevice.getPhysicalDevice());
        if (cudaDevice < 0 || !cudaUtils::cudaCheck(cudaSetDevice(cudaDevice), "cudaSetDevice")) return false;

        if (!cudaUtils::cudaCheck(cudaStreamCreateWithFlags(&solveStream, cudaStreamNonBlocking), "create solveStream") ||
            !cudaUtils::cudaCheck(cudaEventCreate(&solveStartEvent), "create solveStartEvent") ||
            !cudaUtils::cudaCheck(cudaEventCreate(&solveEndEvent), "create solveEndEvent")) {
            cleanup();
            return false;
        }

        if (!cudaUtils::cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&pinnedPcgStatus), sizeof(cudaPcg::PcgStatus)), "alloc pinned PCG status") ||
            !cudaUtils::cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&pinnedAdvectionTimestep), sizeof(cudaSubcycle::SubcycleTimestep)), "alloc pinned advection timestep")) {
            cleanup();
            return false;
        }

        if (!createTimelineSemaphore(cudaReadyVkSemaphore) ||
            !importTimelineSemaphore(cudaReadyVkSemaphore, cudaReadySemaphore) ||
            !createTimelineSemaphore(vulkanReleaseVkSemaphore) ||
            !importTimelineSemaphore(vulkanReleaseVkSemaphore, vulkanReleaseSemaphore)) {
            cleanup();
            return false;
        }

        // Import temperature CUDA external memory
        temperatureNodes.reserve(inputTemperatureNodes.size());
        for (const auto& inputNode : inputTemperatureNodes) {
            if (!inputNode.externalA || !inputNode.externalB || !inputNode.temperatureA || !inputNode.temperatureB) {
                cleanup();
                return false;
            }
            if (!inputNode.temperatureA->ensureMapped(*inputNode.externalA, cudaDevice) ||
                !inputNode.temperatureB->ensureMapped(*inputNode.externalB, cudaDevice)) {
                cleanup();
                return false;
            }

            temperatureNodes.emplace_back();
            auto& devNode = temperatureNodes.back();
            devNode.d_temperatureA = inputNode.temperatureA->getPointer();
            devNode.d_temperatureB = inputNode.temperatureB->getPointer();
            devNode.gatherGroupCount = static_cast<uint32_t>(inputNode.gatherSolverNodeIds.size());
            devNode.scatterCount = static_cast<uint32_t>(inputNode.scatterLocalNodeIds.size());
            if (inputNode.gatherOffsets.size() != size_t(devNode.gatherGroupCount) + 1 ||
                inputNode.gatherLocalNodeIds.size() != inputNode.gatherWeights.size() ||
                inputNode.gatherLocalNodeIds.empty() ||
                inputNode.scatterLocalNodeIds.size() != inputNode.scatterSolverNodeIds.size() ||
                inputNode.gatherOffsets.back() != inputNode.gatherLocalNodeIds.size()) {
                cleanup();
                return false;
            }

            if (devNode.gatherGroupCount > 0 && devNode.scatterCount > 0) {
                if (!devNode.gatherOffsets.allocate(static_cast<uint32_t>(inputNode.gatherOffsets.size())) ||
                    !devNode.gatherOffsets.upload(inputNode.gatherOffsets.data(), static_cast<uint32_t>(inputNode.gatherOffsets.size())) ||
                    !devNode.gatherLocalNodeIds.allocate(static_cast<uint32_t>(inputNode.gatherLocalNodeIds.size())) ||
                    !devNode.gatherLocalNodeIds.upload(inputNode.gatherLocalNodeIds.data(), static_cast<uint32_t>(inputNode.gatherLocalNodeIds.size())) ||
                    !devNode.gatherWeights.allocate(static_cast<uint32_t>(inputNode.gatherWeights.size())) ||
                    !devNode.gatherWeights.upload(inputNode.gatherWeights.data(), static_cast<uint32_t>(inputNode.gatherWeights.size())) ||
                    !devNode.gatherSolverNodeIds.allocate(devNode.gatherGroupCount) ||
                    !devNode.gatherSolverNodeIds.upload(inputNode.gatherSolverNodeIds.data(), devNode.gatherGroupCount) ||
                    !devNode.scatterLocalNodeIds.allocate(devNode.scatterCount) ||
                    !devNode.scatterLocalNodeIds.upload(inputNode.scatterLocalNodeIds.data(), devNode.scatterCount) ||
                    !devNode.scatterSolverNodeIds.allocate(devNode.scatterCount) ||
                    !devNode.scatterSolverNodeIds.upload(inputNode.scatterSolverNodeIds.data(), devNode.scatterCount)) {
                    cleanup();
                    return false;
                }
            } else if (devNode.gatherGroupCount != 0 || devNode.scatterCount != 0) {
                cleanup();
                return false;
            }
        }

        // Allocate CSR matrix and trace vector memory
        if (!d_rowOffsets.allocate(static_cast<uint32_t>(traceSystem.rowOffsets.size())) ||
            !d_columnIndices.allocate(matrixValueCount) ||
            !d_values.allocate(matrixValueCount) ||
            !d_invDiag.allocate(traceCount) ||
            !d_traceRhs.allocate(traceCount) ||
            !d_traceSolution.allocate(traceCount) ||
            !d_r.allocate(traceCount) ||
            !d_z.allocate(traceCount) ||
            !d_p.allocate(traceCount) ||
            !d_Ap.allocate(traceCount) ||
            !d_status.allocate(1) ||
            !d_config.allocate(1) ||
            !d_traceIsDirichlet.allocate(traceCount) ||
            !d_traceDirichletValue.allocate(traceCount) ||
            !d_fixedCellTemperatures.allocate(cellCount)) {
            cleanup();
            return false;
        }

        // Zero PCG warm-start vector while remaining scratch is overwritten before use
        if (!cudaUtils::cudaCheck(cudaMemset(d_traceSolution.get(), 0, traceCount * sizeof(float)), "zero d_traceSolution")) {
            cleanup();
            return false;
        }

        // Allocate Cell memory
        if (!d_cellState.allocate(cellCount) ||
            !d_cellCapacityOverDt.allocate(cellCount) ||
            !d_cellSourceAdditions.allocate(cellCount) ||
            !d_cellSource.allocate(cellCount)) {
            cleanup();
            return false;
        }

        // Allocate VEM Recovery & Scatter memory
        const uint32_t totalCellTraceEntries = static_cast<uint32_t>(traceSystem.cellTraceIndices.size());
        const uint32_t totalScatterEntries = static_cast<uint32_t>(traceSystem.traceScatterCells.size());

        if (!d_cellRecoveryScale.allocate(cellCount) ||
            !d_cellTraceOffsets.allocate(static_cast<uint32_t>(traceSystem.cellTraceOffsets.size())) ||
            !d_cellTraceIndices.allocate(totalCellTraceEntries) ||
            !d_cellTraceWeights.allocate(totalCellTraceEntries) ||
            !d_traceScatterOffsets.allocate(static_cast<uint32_t>(traceSystem.traceScatterOffsets.size())) ||
            !d_traceScatterCells.allocate(totalScatterEntries) ||
            !d_traceScatterWeights.allocate(totalScatterEntries) ||
            !d_traceBoundaryRhs.allocate(traceCount)) {
            cleanup();
            return false;
        }

        // Upload CSR data
        if (!d_rowOffsets.upload(traceSystem.rowOffsets.data(), static_cast<uint32_t>(traceSystem.rowOffsets.size())) ||
            !d_columnIndices.upload(traceSystem.columnIndices.data(), matrixValueCount) ||
            !d_values.upload(traceSystem.values.data(), matrixValueCount)) {
            cleanup();
            return false;
        }

        // Upload Cell data
        if (!d_cellCapacityOverDt.upload(cellCapacityOverDt.data(), cellCount)) {
            cleanup();
            return false;
        }
        if (!cellSourceAdditions.empty()) {
            if (!d_cellSourceAdditions.upload(cellSourceAdditions.data(), cellCount)) {
                cleanup();
                return false;
            }
        } else if (!cudaUtils::cudaCheck(cudaMemset(d_cellSourceAdditions.get(), 0, cellCount * sizeof(float)), "zero d_cellSourceAdditions")) {
            cleanup();
            return false;
        }

        // Upload VEM Recovery & Scatter data
        if (!d_cellRecoveryScale.upload(traceSystem.cellRecoveryScale.data(), cellCount) ||
            !d_cellTraceOffsets.upload(traceSystem.cellTraceOffsets.data(), static_cast<uint32_t>(traceSystem.cellTraceOffsets.size())) ||
            !d_cellTraceIndices.upload(traceSystem.cellTraceIndices.data(), totalCellTraceEntries) ||
            !d_cellTraceWeights.upload(traceSystem.cellTraceWeights.data(), totalCellTraceEntries) ||
            !d_traceScatterOffsets.upload(traceSystem.traceScatterOffsets.data(), static_cast<uint32_t>(traceSystem.traceScatterOffsets.size())) ||
            !d_traceScatterCells.upload(traceSystem.traceScatterCells.data(), totalScatterEntries) ||
            !d_traceScatterWeights.upload(traceSystem.traceScatterWeights.data(), totalScatterEntries) ||
            !d_traceBoundaryRhs.upload(traceSystem.traceBoundaryRhs.data(), traceCount)) {
            cleanup();
            return false;
        }

        // Dirichlet buffers preallocated before graph capture; sentinels represent inactive entries
        if (!traceSystem.traceIsDirichlet.empty()) {
            if (!d_traceIsDirichlet.upload(traceSystem.traceIsDirichlet.data(), traceCount) ||
                !d_traceDirichletValue.upload(traceSystem.traceDirichletValue.data(), traceCount)) {
                cleanup();
                return false;
            }
        } else if (!cudaUtils::cudaCheck(cudaMemset(d_traceIsDirichlet.get(), 0, traceCount * sizeof(uint8_t)), "zero d_traceIsDirichlet") ||
                   !cudaUtils::cudaCheck(cudaMemset(d_traceDirichletValue.get(), 0, traceCount * sizeof(float)), "zero d_traceDirichletValue")) {
            cleanup();
            return false;
        }
        if (!fixedCellTemperatures.empty()) {
            if (!d_fixedCellTemperatures.upload(fixedCellTemperatures.data(), cellCount)) {
                cleanup();
                return false;
            }
        } else {
            const uint32_t cellBlocks = (cellCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;
            heat::globalsolver::fillNaNKernel<<<cellBlocks, heat::globalsolver::BlockSize, 0, solveStream>>>(cellCount, d_fixedCellTemperatures.get());
        }

        // Compute initial Jacobi inverse diagonal over traceCount
        const uint32_t gridTraces = (traceCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;
        heat::globalsolver::computeInvDiagKernel<<<gridTraces, heat::globalsolver::BlockSize, 0, solveStream>>>(
            traceCount, d_rowOffsets.get(), d_columnIndices.get(), d_values.get(), d_invDiag.get());
        if (!cudaUtils::cudaCheck(cudaStreamSynchronize(solveStream), "computeInvDiagKernel")) {
            cleanup();
            return false;
        }

        // Upload the graph-independent PCG parameters before graph construction.
        cudaPcg::PcgConfig initialConfig{ solverMaxIterations, solverConvergenceTolerance };
        if (!d_config.upload(&initialConfig, 1)) {
            cleanup();
            return false;
        }

        initialized = true;

        // Build execution graphs
        if (!buildGraphs()) {
            std::cerr << "[HeatGlobalSolver] Graph construction failed." << std::endl;
            cleanup();
            return false;
        }

        return true;
    }

    bool updateValues(
        const vem::VEMTraceSystem& traceSystem,
        const std::vector<float>& cellCapacityOverDt,
        const std::vector<float>& cellSourceAdditions,
        const std::vector<float>& fixedCellTemperatures = {}) {
        if (!initialized) return false;
        waitIdle();
        cudaSetDevice(cudaDevice);

        if (traceSystem.traceCount != traceCount || traceSystem.cellCount != cellCount ||
            traceSystem.values.size() != matrixValueCount || cellCapacityOverDt.size() != cellCount) {
            return false;
        }

        if (!d_values.uploadAsync(traceSystem.values.data(), matrixValueCount, solveStream)) {
            return false;
        }

        if (!d_cellCapacityOverDt.uploadAsync(cellCapacityOverDt.data(), cellCount, solveStream)) {
            return false;
        }

        if (!cellSourceAdditions.empty()) {
            if (!d_cellSourceAdditions.uploadAsync(cellSourceAdditions.data(), cellCount, solveStream)) {
                return false;
            }
        } else if (!cudaUtils::cudaCheck(cudaMemsetAsync(d_cellSourceAdditions.get(), 0, cellCount * sizeof(float), solveStream), "clear d_cellSourceAdditions")) {
            return false;
        }

        if (!d_cellRecoveryScale.uploadAsync(traceSystem.cellRecoveryScale.data(), cellCount, solveStream)) {
            return false;
        }
        if (!d_cellTraceWeights.uploadAsync(traceSystem.cellTraceWeights.data(), static_cast<uint32_t>(traceSystem.cellTraceWeights.size()), solveStream)) {
            return false;
        }
        if (!d_traceScatterWeights.uploadAsync(traceSystem.traceScatterWeights.data(), static_cast<uint32_t>(traceSystem.traceScatterWeights.size()), solveStream)) {
            return false;
        }
        if (!d_traceBoundaryRhs.uploadAsync(traceSystem.traceBoundaryRhs.data(), traceCount, solveStream)) {
            return false;
        }

        // Refresh constraint buffers via sentinels without reallocating captured memory
        if (!traceSystem.traceIsDirichlet.empty()) {
            if (!d_traceIsDirichlet.uploadAsync(traceSystem.traceIsDirichlet.data(), traceCount, solveStream) ||
                !d_traceDirichletValue.uploadAsync(traceSystem.traceDirichletValue.data(), traceCount, solveStream)) {
                return false;
            }
        } else if (!cudaUtils::cudaCheck(cudaMemsetAsync(d_traceIsDirichlet.get(), 0, traceCount * sizeof(uint8_t), solveStream), "clear d_traceIsDirichlet") ||
                   !cudaUtils::cudaCheck(cudaMemsetAsync(d_traceDirichletValue.get(), 0, traceCount * sizeof(float), solveStream), "clear d_traceDirichletValue")) {
            return false;
        }

        if (!fixedCellTemperatures.empty()) {
            if (!d_fixedCellTemperatures.uploadAsync(fixedCellTemperatures.data(), cellCount, solveStream)) return false;
        } else {
            const uint32_t cellBlocks = (cellCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;
            heat::globalsolver::fillNaNKernel<<<cellBlocks, heat::globalsolver::BlockSize, 0, solveStream>>>(cellCount, d_fixedCellTemperatures.get());
        }

        const uint32_t gridTraces = (traceCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;
        heat::globalsolver::computeInvDiagKernel<<<gridTraces, heat::globalsolver::BlockSize, 0, solveStream>>>(
            traceCount, d_rowOffsets.get(), d_columnIndices.get(), d_values.get(), d_invDiag.get());

        return cudaUtils::cudaCheck(cudaStreamSynchronize(solveStream), "sync updateValues");
    }

    bool configureAdvectionTopology(
        uint32_t inFaceCount,
        uint32_t inBoundaryFaceCount,
        const std::vector<heat::AdvectionFace>& faces,
        const std::vector<heat::AdvectionBoundaryFace>& boundaryFaces,
        const float* deviceFaceFlux,
        const float* deviceBoundaryFlux,
        const std::vector<float>& cellVolumes) {
        if (!initialized) return false;
        if (inFaceCount == 0 || !deviceFaceFlux) return false;
        if (faces.size() != inFaceCount || boundaryFaces.size() != inBoundaryFaceCount) return false;
        if (cellVolumes.size() != cellCount) return false;
        waitIdle();
        cudaSetDevice(cudaDevice);

        disableAdvection();

        // Reject non-positive or non-finite cell volumes immediately
        std::vector<float> invVolumes(cellCount, 0.0f);
        for (uint32_t c = 0; c < cellCount; ++c) {
            const float volume = cellVolumes[c];
            if (!(volume > 0.0f) || !std::isfinite(volume)) {
                std::cerr << "[HeatGlobalSolver] configureAdvectionTopology: invalid cell volume at " << c
                          << ": " << volume << std::endl;
                disableAdvection();
                return false;
            }
            invVolumes[c] = 1.0f / volume;
        }

        advectionFaceCount = inFaceCount;
        advectionBoundaryFaceCount = inBoundaryFaceCount;
        d_advectionFaceFlux = deviceFaceFlux;
        d_advectionBoundaryFlux = deviceBoundaryFlux;

        d_advectionFaces.reset();
        d_advectionBoundaryFaces.reset();
        d_cellInvVolumes.reset();
        d_advectionDeltaT.reset();
        d_advectionOutflow.reset();
        d_advectionMaxRate.reset();
        d_advectionInvalid.reset();
        d_advectionTimestep.reset();
        d_advectionMacroDt.reset();
        d_advectionIter.reset();
        d_inflowTemperatures.reset();
        if (!d_advectionFaces.allocate(inFaceCount) ||
            !d_advectionFaces.uploadAsync(faces.data(), inFaceCount, solveStream)) {
            disableAdvection();
            return false;
        }
        if (inBoundaryFaceCount > 0) {
            if (!d_advectionBoundaryFaces.allocate(inBoundaryFaceCount) ||
                !d_advectionBoundaryFaces.uploadAsync(boundaryFaces.data(), inBoundaryFaceCount, solveStream)) {
                disableAdvection();
                return false;
            }
        }
        if (!d_cellInvVolumes.allocate(cellCount) ||
            !d_cellInvVolumes.uploadAsync(invVolumes.data(), cellCount, solveStream) ||
            !d_advectionDeltaT.allocate(cellCount) ||
            !d_advectionOutflow.allocate(cellCount) ||
            !d_advectionMaxRate.allocate(1) ||
            !d_advectionInvalid.allocate(1) ||
            !d_advectionTimestep.allocate(1) ||
            !d_advectionMacroDt.allocate(1) ||
            !d_advectionIter.allocate(1) ||
            !d_inflowTemperatures.allocate(6) ||
            !cudaUtils::cudaCheck(cudaMemsetAsync(d_advectionMacroDt.get(), 0, sizeof(float), solveStream), "zero d_advectionMacroDt")) {
            disableAdvection();
            return false;
        }

        if (!cudaUtils::cudaCheck(cudaStreamSynchronize(solveStream), "sync configureAdvectionTopology")) {
            disableAdvection();
            return false;
        }

        if (!buildAdvectionGraph()) {
            disableAdvection();
            return false;
        }
        return true;
    }

    bool updateAdvectionStep(
        const float* inflowTemperaturesC,
        float stepDt) {
        if (inflowTemperaturesC && d_inflowTemperatures.get() &&
            !d_inflowTemperatures.uploadAsync(inflowTemperaturesC, 6, solveStream)) {
            std::cerr << "[HeatGlobalSolver] inflow temperature upload failed" << std::endl;
            return false;
        }
        if (d_advectionMacroDt.get() && !d_advectionMacroDt.uploadAsync(&stepDt, 1, solveStream)) {
            std::cerr << "[HeatGlobalSolver] advection macro dt upload failed" << std::endl;
            return false;
        }
        return true;
    }

    bool launch(
        bool sourceIsA,
        uint64_t destinationVulkanReleaseValue,
        uint64_t cudaReadyValue) {
        if (!initialized) return false;
        cudaSetDevice(cudaDevice);

        if (solveStartEvent) {
            cudaEventRecord(solveStartEvent, solveStream);
        }

        // Wait for destination buffer last reader or current frame flux staging copy
        if (destinationVulkanReleaseValue > 0) {
            cudaExternalSemaphoreWaitParams waitParams{};
            waitParams.params.fence.value = destinationVulkanReleaseValue;
            if (!cudaUtils::cudaCheck(cudaWaitExternalSemaphoresAsync(&vulkanReleaseSemaphore, &waitParams, 1, solveStream), "wait Vulkan release")) {
                return false;
            }
        }

        // Everything from here on is GPU compute enqueued on the solve stream.

        // Gather source temperature into d_cellState
        for (const auto& node : temperatureNodes) {
            const uint32_t blocks = (node.gatherGroupCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;
            heat::globalsolver::gatherTemperaturesKernel<<<blocks, heat::globalsolver::BlockSize, 0, solveStream>>>(
                sourceIsA ? node.d_temperatureA : node.d_temperatureB,
                node.gatherOffsets.get(),
                node.gatherLocalNodeIds.get(),
                node.gatherWeights.get(),
                node.gatherSolverNodeIds.get(),
                node.gatherGroupCount,
                d_cellState.get());
        }

        // Launch device-controlled conditional subcycle graph for fine conservative advection
        if (advectionExec) {
            if (!cudaUtils::cudaCheck(cudaGraphLaunch(advectionExec, solveStream), "launch advection graph")) {
                return false;
            }
        }

        // Three ordered launches on solveStream: prologue, PCG loop, recovery + scatter. Stream order supplies the dependencies.
        if (!preExec || !pcgExec || !postExecAToB || !postExecBToA) {
            std::cerr << "[HeatGlobalSolver] solve graphs are not built" << std::endl;
            return false;
        }
        if (!cudaUtils::cudaCheck(cudaGraphLaunch(preExec, solveStream), "launch pre graph")) {
            return false;
        }
        if (!cudaUtils::cudaCheck(cudaGraphLaunch(pcgExec, solveStream), "launch pcg graph")) {
            return false;
        }
        if (!cudaUtils::cudaCheck(cudaGraphLaunch(sourceIsA ? postExecAToB : postExecBToA, solveStream), "launch post graph")) {
            return false;
        }

        // Signal CUDA ready semaphore
        if (cudaReadyValue > 0) {
            cudaExternalSemaphoreSignalParams signalParams{};
            signalParams.params.fence.value = cudaReadyValue;
            if (!cudaUtils::cudaCheck(cudaSignalExternalSemaphoresAsync(&cudaReadySemaphore, &signalParams, 1, solveStream), "signal CUDA ready")) {
                return false;
            }
        }

        // Asynchronously download solve status to pinned memory on solve stream
        if (advectionFaceCount > 0 && d_advectionTimestep.get() && pinnedAdvectionTimestep &&
            !d_advectionTimestep.downloadAsync(pinnedAdvectionTimestep, 1, solveStream)) {
            return false;
        }
        if (d_status.get() && pinnedPcgStatus && !d_status.downloadAsync(pinnedPcgStatus, 1, solveStream)) {
            return false;
        }

        // Record solve completion event after all graph launches and staging copies
        if (solveEndEvent) {
            cudaEventRecord(solveEndEvent, solveStream);
        }

        return true;
    }

        SolvePollResult poll() const {
            const cudaError_t status = cudaEventQuery(solveEndEvent);
            if (status == cudaSuccess) {
                float elapsedMs = 0.0f;
                if (cudaEventElapsedTime(&elapsedMs, solveStartEvent, solveEndEvent) == cudaSuccess) {
                    lastSolveMs = double(elapsedMs);
                }
                return SolvePollResult::Complete;
            }
            if (status == cudaErrorNotReady) {
                return SolvePollResult::Pending;
            }
            std::cerr << "[HeatGlobalSolver] async solve failed: "
                      << cudaGetErrorString(status) << std::endl;
            return SolvePollResult::Failed;
        }

    void waitIdle() {
        if (solveStream) {
            cudaStreamSynchronize(solveStream);
        }
    }

    bool hasSolveFailed() {
        // Check pinned advection and PCG completion status to reject failed steps
        if (advectionFaceCount > 0 && d_advectionTimestep.get() && pinnedAdvectionTimestep) {
            const cudaSubcycle::SubcycleTimestep& timestep = *pinnedAdvectionTimestep;
            if (timestep.status != cudaSubcycle::SubcycleStatus::Ok) {
                std::cerr << "[HeatGlobalSolver] advection step rejected: status="
                          << static_cast<uint32_t>(timestep.status) << std::endl;
                return true;
            }
        }

        if (!d_status.get() || !pinnedPcgStatus) return false;
        const cudaPcg::PcgStatus& hostStatus = *pinnedPcgStatus;

        const bool breakdown = hostStatus.failed != 0;
        const bool nonConverged = hostStatus.converged == 0 &&
                                  hostStatus.iterationCount >= int(solverMaxIterations);
        if (!breakdown && !nonConverged) return false;

        const float relRes = hostStatus.residual0 > 1e-12f
            ? hostStatus.residual / hostStatus.residual0
            : hostStatus.residual;
        std::cerr << "[HeatGlobalSolver] solve did not succeed: "
                  << (breakdown ? "numerical breakdown (pAp non-finite or <= 0)"
                                : "no convergence at max iterations")
                  << ", iterations=" << hostStatus.iterationCount
                  << ", residual0=" << hostStatus.residual0
                  << ", relResidual=" << relRes
                  << ", pAp=" << hostStatus.pAp
                  << ", alpha=" << hostStatus.alpha
                  << std::endl;

        return true;
    }

    void setSolverParameters(uint32_t maxIterations, float convergenceTolerance) {
        // 0 / non-positive means "keep the current value" (authoring default).
        if (maxIterations > 0u) solverMaxIterations = maxIterations;
        if (convergenceTolerance > 0.0f) solverConvergenceTolerance = convergenceTolerance;
        if (!initialized || !d_config.get()) return;
        // Solver parameters reside in device memory read by graph kernels without rebuilds
        waitIdle();
        cudaSetDevice(cudaDevice);
        cudaPcg::PcgConfig cfg{ solverMaxIterations, solverConvergenceTolerance };
        d_config.upload(&cfg, 1);
    }

    float getLastSolveMs() const { return static_cast<float>(lastSolveMs); }

    VkSemaphore getCudaReadySemaphore() const { return cudaReadyVkSemaphore; }
    VkSemaphore getVulkanReleaseSemaphore() const { return vulkanReleaseVkSemaphore; }

    void destroyGraphs() {
        if (pcgExec) { cudaGraphExecDestroy(pcgExec); pcgExec = nullptr; }
        if (preExec) { cudaGraphExecDestroy(preExec); preExec = nullptr; }
        if (postExecAToB) { cudaGraphExecDestroy(postExecAToB); postExecAToB = nullptr; }
        if (postExecBToA) { cudaGraphExecDestroy(postExecBToA); postExecBToA = nullptr; }
        disableAdvection();
    }

    void cleanup() {
        if (cudaDevice >= 0) {
            cudaSetDevice(cudaDevice);
        }
        waitIdle();
        destroyGraphs();
        if (solveStartEvent) { cudaEventDestroy(solveStartEvent); solveStartEvent = nullptr; }
        if (solveEndEvent) { cudaEventDestroy(solveEndEvent); solveEndEvent = nullptr; }
        if (pinnedPcgStatus) { cudaFreeHost(pinnedPcgStatus); pinnedPcgStatus = nullptr; }
        if (pinnedAdvectionTimestep) { cudaFreeHost(pinnedAdvectionTimestep); pinnedAdvectionTimestep = nullptr; }
        if (solveStream) { cudaStreamDestroy(solveStream); solveStream = nullptr; }

        d_rowOffsets.reset();
        d_columnIndices.reset();
        d_values.reset();
        d_invDiag.reset();
        d_traceRhs.reset();
        d_traceSolution.reset();
        d_r.reset();
        d_z.reset();
        d_p.reset();
        d_Ap.reset();
        d_status.reset();
        d_config.reset();

        d_cellState.reset();
        d_cellCapacityOverDt.reset();
        d_cellSourceAdditions.reset();
        d_cellSource.reset();

        d_cellRecoveryScale.reset();
        d_cellTraceOffsets.reset();
        d_cellTraceIndices.reset();
        d_cellTraceWeights.reset();
        d_traceScatterOffsets.reset();
        d_traceScatterCells.reset();
        d_traceScatterWeights.reset();
        d_traceBoundaryRhs.reset();
        d_traceIsDirichlet.reset();
        d_traceDirichletValue.reset();
        d_fixedCellTemperatures.reset();

        d_advectionFaces.reset();
        d_advectionBoundaryFaces.reset();
        d_cellInvVolumes.reset();
        d_advectionDeltaT.reset();
        d_advectionOutflow.reset();
        d_advectionMaxRate.reset();
        d_advectionInvalid.reset();
        d_advectionTimestep.reset();
        d_advectionMacroDt.reset();
        d_advectionIter.reset();
        d_inflowTemperatures.reset();

        advectionFaceCount = 0;
        advectionBoundaryFaceCount = 0;
        d_advectionFaceFlux = nullptr;
        d_advectionBoundaryFlux = nullptr;

        temperatureNodes.clear();

        if (cudaReadySemaphore) { cudaDestroyExternalSemaphore(cudaReadySemaphore); cudaReadySemaphore = nullptr; }
        if (vulkanReleaseSemaphore) { cudaDestroyExternalSemaphore(vulkanReleaseSemaphore); vulkanReleaseSemaphore = nullptr; }

        if (device != VK_NULL_HANDLE) {
            if (cudaReadyVkSemaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device, cudaReadyVkSemaphore, nullptr);
                cudaReadyVkSemaphore = VK_NULL_HANDLE;
            }
            if (vulkanReleaseVkSemaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(device, vulkanReleaseVkSemaphore, nullptr);
                vulkanReleaseVkSemaphore = VK_NULL_HANDLE;
            }
        }

        device = VK_NULL_HANDLE;
        cudaDevice = -1;
        cellCount = 0;
        traceCount = 0;
        matrixValueCount = 0;
        initialized = false;
    }

private:
    // Direction-independent prologue of the solve sequence.
    void recordPrePcgSequence(cudaStream_t stream) {
        const uint32_t cellBlocks = (cellCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;
        const uint32_t traceBlocks = (traceCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;

        // 1. Build authoritative cell source: S_E = (C_E / dt) * u_E^adv + S_E^add
        heat::globalsolver::buildCellSourceKernel<<<cellBlocks, heat::globalsolver::BlockSize, 0, stream>>>(
            cellCount,
            d_cellCapacityOverDt.get(),
            d_cellState.get(),
            d_cellSourceAdditions.get(),
            d_cellSource.get());

        // 2. Assemble trace RHS: b_t = sum_E (w_Et * S_E / d_E) + b_bc, with Dirichlet override
        heat::globalsolver::assembleTraceRhsKernel<<<traceBlocks, heat::globalsolver::BlockSize, 0, stream>>>(
            traceCount,
            d_traceScatterOffsets.get(),
            d_traceScatterCells.get(),
            d_traceScatterWeights.get(),
            d_cellSource.get(),
            d_cellRecoveryScale.get(),
            d_traceBoundaryRhs.get(),
            d_traceIsDirichlet.get(),
            d_traceDirichletValue.get(),
            d_traceRhs.get());
    }

    // Recovery and scatter; direction only selects the destination buffer.
    void recordPostPcgSequence(bool sourceIsA, cudaStream_t stream) {
        const uint32_t cellBlocks = (cellCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;

        // 3. Recover cell temperatures: u_E = (1 / d_E) * (sum_f w_Ef * lambda_f + S_E), with fixed temperature override
        heat::globalsolver::recoverCellTemperaturesKernel<<<cellBlocks, heat::globalsolver::BlockSize, 0, stream>>>(
            cellCount,
            d_cellRecoveryScale.get(),
            d_cellTraceOffsets.get(),
            d_cellTraceIndices.get(),
            d_cellTraceWeights.get(),
            d_traceSolution.get(),
            d_cellSource.get(),
            d_fixedCellTemperatures.get(),
            d_cellState.get());

        // 4. Scatter solution into the alternate destination buffer
        for (const auto& node : temperatureNodes) {
            const uint32_t blocks = (node.scatterCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;
            heat::globalsolver::scatterTemperaturesKernel<<<blocks, heat::globalsolver::BlockSize, 0, stream>>>(
                d_cellState.get(),
                node.scatterSolverNodeIds.get(),
                node.scatterLocalNodeIds.get(),
                node.scatterCount,
                sourceIsA ? node.d_temperatureB : node.d_temperatureA);
        }
    }

    bool buildPcgGraph() {
        pcgExec = cudaPcg::buildPcgGraphExec(
            traceCount,
            d_rowOffsets.get(),
            d_columnIndices.get(),
            d_values.get(),
            d_invDiag.get(),
            d_traceSolution.get(),
            d_traceRhs.get(),
            d_Ap.get(),
            d_r.get(),
            d_z.get(),
            d_p.get(),
            d_status.get(),
            d_config.get());
        return pcgExec != nullptr;
    }

    bool buildGraphs() {
        if (!solveStream) return false;

        // Standalone PCG graph executable because CUDA disallows conditional nodes in child graphs
        if (!buildPcgGraph()) return false;

        // Capture the direction-independent prologue.
        {
            cudaGraph_t graph = nullptr;
            if (!cudaUtils::cudaCheck(cudaStreamBeginCapture(solveStream, cudaStreamCaptureModeGlobal), "begin capture pre")) return false;
            recordPrePcgSequence(solveStream);
            if (!cudaUtils::cudaCheck(cudaStreamEndCapture(solveStream, &graph), "end capture pre")) return false;
            if (!cudaUtils::cudaCheck(cudaGraphInstantiate(&preExec, graph, nullptr, nullptr, 0), "instantiate pre")) return false;
            cudaGraphDestroy(graph);
        }

        // Capture recovery + scatter to B.
        {
            cudaGraph_t graph = nullptr;
            if (!cudaUtils::cudaCheck(cudaStreamBeginCapture(solveStream, cudaStreamCaptureModeGlobal), "begin capture post AB")) return false;
            recordPostPcgSequence(true, solveStream);
            if (!cudaUtils::cudaCheck(cudaStreamEndCapture(solveStream, &graph), "end capture post AB")) return false;
            if (!cudaUtils::cudaCheck(cudaGraphInstantiate(&postExecAToB, graph, nullptr, nullptr, 0), "instantiate post AB")) return false;
            cudaGraphDestroy(graph);
        }

        // Capture recovery + scatter to A.
        {
            cudaGraph_t graph = nullptr;
            if (!cudaUtils::cudaCheck(cudaStreamBeginCapture(solveStream, cudaStreamCaptureModeGlobal), "begin capture post BA")) return false;
            recordPostPcgSequence(false, solveStream);
            if (!cudaUtils::cudaCheck(cudaStreamEndCapture(solveStream, &graph), "end capture post BA")) return false;
            if (!cudaUtils::cudaCheck(cudaGraphInstantiate(&postExecBToA, graph, nullptr, nullptr, 0), "instantiate post BA")) return false;
            cudaGraphDestroy(graph);
        }

        return true;
    }

    // Tears down advection so the solver degrades to heat-only on any failure.
    void disableAdvection() {
        if (advectionExec) { cudaGraphExecDestroy(advectionExec); advectionExec = nullptr; }
        advectionFaceCount = 0;
        advectionBoundaryFaceCount = 0;
        d_advectionFaceFlux = nullptr;
        d_advectionBoundaryFlux = nullptr;
    }

    bool buildAdvectionGraph() {
        if (!solveStream || advectionFaceCount == 0) return false;

        using cudaGraphBuilder::appendKernel;
        using cudaGraphBuilder::appendMemset;

        cudaGraph_t graph = nullptr;
        if (!cudaUtils::cudaCheck(cudaGraphCreate(&graph, 0), "advection graph create")) {
            return false;
        }

        const uint32_t faceBlocks = (advectionFaceCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;
        const uint32_t bfaceBlocks = std::max(1u, (advectionBoundaryFaceCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize);
        const uint32_t cellBlocks = (cellCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;

        cudaGraphNode_t prev = nullptr;
        const bool prologueOk =
            appendMemset(graph, prev, d_advectionMaxRate.get(), 0, sizeof(float)) == cudaSuccess &&
            appendMemset(graph, prev, d_advectionOutflow.get(), 0, cellCount * sizeof(float)) == cudaSuccess &&
            appendMemset(graph, prev, d_advectionInvalid.get(), 0, sizeof(uint32_t)) == cudaSuccess &&
            appendKernel(graph, prev, (void*)heat::globalsolver::accumulateFineAdvectionOutflowKernel, dim3(faceBlocks), dim3(heat::globalsolver::BlockSize),
                         advectionFaceCount, d_advectionFaces.get(), d_advectionFaceFlux,
                         advectionBoundaryFaceCount, d_advectionBoundaryFaces.get(), d_advectionBoundaryFlux,
                         d_cellInvVolumes.get(), d_advectionOutflow.get(), d_advectionInvalid.get()) == cudaSuccess &&
            appendKernel(graph, prev, (void*)cudaSubcycle::reduceMaxRateKernel, dim3(64), dim3(256),
                         cellCount, d_advectionOutflow.get(), d_advectionMaxRate.get(), d_advectionInvalid.get()) == cudaSuccess;
        if (!prologueOk) {
            cudaGraphDestroy(graph);
            return false;
        }

        cudaSubcycle::SubcycleSchedule schedule{};
        schedule.maxRate = d_advectionMaxRate.get();
        schedule.invalidFlag = d_advectionInvalid.get();
        schedule.macroDt = d_advectionMacroDt.get();
        schedule.targetCfl = AdvectionTargetCfl;
        schedule.maxCount = MaxAdvectionSubsteps;
        schedule.timestep = d_advectionTimestep.get();
        schedule.iteration = d_advectionIter.get();

        cudaGraphConditionalHandle handle = {};
        cudaGraphBuilder::WhileLoop loop;
        if (!cudaUtils::cudaCheck(cudaSubcycle::beginSubcycleLoop(graph, prev, schedule, handle, loop), "advection subcycle loop")) {
            cudaGraphDestroy(graph);
            return false;
        }
        cudaGraph_t body = loop.body;

        cudaGraphNode_t b = nullptr;
        const bool bodyOk =
            appendMemset(body, b, d_advectionDeltaT.get(), 0, cellCount * sizeof(float)) == cudaSuccess &&
            appendKernel(body, b, (void*)heat::globalsolver::advectFineFacesKernel, dim3(faceBlocks), dim3(heat::globalsolver::BlockSize),
                         advectionFaceCount, d_advectionFaces.get(), d_advectionFaceFlux, d_advectionDeltaT.get(),
                         d_cellState.get(), d_cellInvVolumes.get(), d_advectionTimestep.get()) == cudaSuccess;
        bool bodyComplete = bodyOk;
        if (bodyComplete && advectionBoundaryFaceCount > 0) {
            bodyComplete = appendKernel(body, b, (void*)heat::globalsolver::advectFineBoundaryKernel, dim3(bfaceBlocks), dim3(heat::globalsolver::BlockSize),
                                        advectionBoundaryFaceCount, d_advectionBoundaryFaces.get(), d_advectionBoundaryFlux,
                                        d_advectionDeltaT.get(), d_cellState.get(), d_inflowTemperatures.get(),
                                        d_cellInvVolumes.get(), d_advectionTimestep.get()) == cudaSuccess;
        }
        if (bodyComplete) {
            bodyComplete =
                appendKernel(body, b, (void*)heat::globalsolver::applyAdvectionUpdatesKernel, dim3(cellBlocks), dim3(heat::globalsolver::BlockSize),
                             cellCount, d_cellState.get(), d_advectionDeltaT.get()) == cudaSuccess &&
                cudaSubcycle::endSubcycleLoop(body, b, handle, schedule) == cudaSuccess;
        }
        if (!bodyComplete) {
            cudaGraphDestroy(graph);
            return false;
        }

        if (!cudaUtils::cudaCheck(cudaGraphInstantiate(&advectionExec, graph, nullptr, nullptr, 0), "advection graph instantiate")) {
            cudaGraphDestroy(graph);
            advectionExec = nullptr;
            return false;
        }
        cudaGraphDestroy(graph);
        return true;
    }

    bool createTimelineSemaphore(VkSemaphore& sem) {
        VkExportSemaphoreCreateInfo exportInfo{};
        exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
        exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        VkSemaphoreTypeCreateInfo typeInfo{};
        typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        typeInfo.pNext = &exportInfo;
        typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        typeInfo.initialValue = 0;
        VkSemaphoreCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        createInfo.pNext = &typeInfo;
        return vkCreateSemaphore(device, &createInfo, nullptr, &sem) == VK_SUCCESS;
    }

    bool importTimelineSemaphore(VkSemaphore vkSem, cudaExternalSemaphore_t& cudaSem) {
        const auto getHandle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
        if (!getHandle) return false;
        VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
        handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
        handleInfo.semaphore = vkSem;
        handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        HANDLE handle = nullptr;
        if (getHandle(device, &handleInfo, &handle) != VK_SUCCESS) return false;
        cudaExternalSemaphoreHandleDesc descriptor{};
        descriptor.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
        descriptor.handle.win32.handle = handle;
        const cudaError_t result = cudaImportExternalSemaphore(&cudaSem, &descriptor);
        if (handle) CloseHandle(handle);
        return cudaUtils::cudaCheck(result, "import timeline semaphore");
    }

    VkDevice device = VK_NULL_HANDLE;
    int cudaDevice = -1;
    cudaStream_t solveStream = nullptr;
    cudaEvent_t solveStartEvent = nullptr;
    cudaEvent_t solveEndEvent = nullptr;
    mutable double lastSolveMs = 0.0;
    cudaSubcycle::SubcycleTimestep* pinnedAdvectionTimestep = nullptr;
    cudaPcg::PcgStatus* pinnedPcgStatus = nullptr;

    VkSemaphore cudaReadyVkSemaphore = VK_NULL_HANDLE;
    cudaExternalSemaphore_t cudaReadySemaphore = nullptr;

    VkSemaphore vulkanReleaseVkSemaphore = VK_NULL_HANDLE;
    cudaExternalSemaphore_t vulkanReleaseSemaphore = nullptr;

    cudaGraphExec_t pcgExec = nullptr;

    cudaGraphExec_t preExec = nullptr;
    cudaGraphExec_t postExecAToB = nullptr;
    cudaGraphExec_t postExecBToA = nullptr;

    cudaGraphExec_t advectionExec = nullptr;

    cudaUtils::CudaBuffer<int> d_rowOffsets;
    cudaUtils::CudaBuffer<int> d_columnIndices;
    cudaUtils::CudaBuffer<float> d_values;
    cudaUtils::CudaBuffer<float> d_invDiag;
    cudaUtils::CudaBuffer<float> d_traceRhs;
    cudaUtils::CudaBuffer<float> d_traceSolution;
    cudaUtils::CudaBuffer<float> d_r;
    cudaUtils::CudaBuffer<float> d_z;
    cudaUtils::CudaBuffer<float> d_p;
    cudaUtils::CudaBuffer<float> d_Ap;
    cudaUtils::CudaBuffer<cudaPcg::PcgStatus> d_status;
    cudaUtils::CudaBuffer<cudaPcg::PcgConfig> d_config;

    cudaUtils::CudaBuffer<float> d_cellState;
    cudaUtils::CudaBuffer<float> d_cellCapacityOverDt;
    cudaUtils::CudaBuffer<float> d_cellSourceAdditions;
    cudaUtils::CudaBuffer<float> d_cellSource;

    cudaUtils::CudaBuffer<float> d_cellRecoveryScale;
    cudaUtils::CudaBuffer<uint32_t> d_cellTraceOffsets;
    cudaUtils::CudaBuffer<uint32_t> d_cellTraceIndices;
    cudaUtils::CudaBuffer<float> d_cellTraceWeights;
    cudaUtils::CudaBuffer<uint32_t> d_traceScatterOffsets;
    cudaUtils::CudaBuffer<uint32_t> d_traceScatterCells;
    cudaUtils::CudaBuffer<float> d_traceScatterWeights;
    cudaUtils::CudaBuffer<float> d_traceBoundaryRhs;
    cudaUtils::CudaBuffer<uint8_t> d_traceIsDirichlet;
    cudaUtils::CudaBuffer<float> d_traceDirichletValue;
    cudaUtils::CudaBuffer<float> d_fixedCellTemperatures;

    uint32_t advectionFaceCount = 0;
    uint32_t advectionBoundaryFaceCount = 0;
    const float* d_advectionFaceFlux = nullptr;
    const float* d_advectionBoundaryFlux = nullptr;
    cudaUtils::CudaBuffer<heat::AdvectionFace> d_advectionFaces;
    cudaUtils::CudaBuffer<heat::AdvectionBoundaryFace> d_advectionBoundaryFaces;
    cudaUtils::CudaBuffer<float> d_cellInvVolumes;
    cudaUtils::CudaBuffer<float> d_advectionDeltaT;
    cudaUtils::CudaBuffer<float> d_advectionOutflow;
    cudaUtils::CudaBuffer<float> d_advectionMaxRate;
    cudaUtils::CudaBuffer<uint32_t> d_advectionInvalid;
    cudaUtils::CudaBuffer<cudaSubcycle::SubcycleTimestep> d_advectionTimestep;
    cudaUtils::CudaBuffer<float> d_advectionMacroDt;
    cudaUtils::CudaBuffer<uint32_t> d_advectionIter;
    cudaUtils::CudaBuffer<float> d_inflowTemperatures;

    // Explicit fine-advection CFL target and substep cap safety ceiling
    static constexpr float AdvectionTargetCfl = 1.0f;
    static constexpr uint32_t MaxAdvectionSubsteps = 1024;

    std::vector<DeviceTemperatureNodes> temperatureNodes;

    uint32_t cellCount = 0;
    uint32_t traceCount = 0;
    uint32_t matrixValueCount = 0;
    bool initialized = false;

    // Runtime PCG knobs uploaded to device config without graph rebuilds
    uint32_t solverMaxIterations = 200;
    float solverConvergenceTolerance = 3e-3f;
};

HeatGlobalSolver::HeatGlobalSolver() : implementation(std::make_unique<Implementation>()) {}
HeatGlobalSolver::~HeatGlobalSolver() = default;

bool HeatGlobalSolver::initialize(
    VulkanDevice& vulkanDevice,
    const vem::VEMTraceSystem& traceSystem,
    const std::vector<float>& cellCapacityOverDt,
    const std::vector<float>& cellSourceAdditions,
    const std::vector<TemperatureNodes>& temperatureNodes,
    const std::vector<float>& fixedCellTemperatures) {
    return implementation->initialize(
        vulkanDevice, traceSystem, cellCapacityOverDt, cellSourceAdditions, temperatureNodes, fixedCellTemperatures);
}

bool HeatGlobalSolver::launch(
    bool sourceIsA,
    uint64_t destinationVulkanReleaseValue,
    uint64_t cudaReadyValue) {
    return implementation->launch(
        sourceIsA, destinationVulkanReleaseValue, cudaReadyValue);
}

HeatGlobalSolver::SolvePollResult HeatGlobalSolver::poll() const {
    return implementation->poll();
}

void HeatGlobalSolver::waitIdle() {
    implementation->waitIdle();
}

bool HeatGlobalSolver::updateValues(
    const vem::VEMTraceSystem& traceSystem,
    const std::vector<float>& cellCapacityOverDt,
    const std::vector<float>& cellSourceAdditions,
    const std::vector<float>& fixedCellTemperatures) {
    return implementation->updateValues(traceSystem, cellCapacityOverDt, cellSourceAdditions, fixedCellTemperatures);
}

VkSemaphore HeatGlobalSolver::getCudaReadySemaphore() const {
    return implementation->getCudaReadySemaphore();
}

VkSemaphore HeatGlobalSolver::getVulkanReleaseSemaphore() const {
    return implementation->getVulkanReleaseSemaphore();
}

bool HeatGlobalSolver::hasSolveFailed() const {
    return implementation->hasSolveFailed();
}

void HeatGlobalSolver::setSolverParameters(uint32_t maxIterations, float convergenceTolerance) {
    implementation->setSolverParameters(maxIterations, convergenceTolerance);
}

float HeatGlobalSolver::getLastSolveMs() const {
    return implementation ? implementation->getLastSolveMs() : 0.0f;
}

bool HeatGlobalSolver::configureAdvectionTopology(
    uint32_t inFaceCount,
    uint32_t inBoundaryFaceCount,
    const std::vector<heat::AdvectionFace>& faces,
    const std::vector<heat::AdvectionBoundaryFace>& boundaryFaces,
    const float* deviceFaceFlux,
    const float* deviceBoundaryFlux,
    const std::vector<float>& cellVolumes) {
    return implementation ? implementation->configureAdvectionTopology(
        inFaceCount, inBoundaryFaceCount, faces, boundaryFaces,
        deviceFaceFlux, deviceBoundaryFlux, cellVolumes) : false;
}

bool HeatGlobalSolver::updateAdvectionStep(
    const float* inflowTemperaturesC,
    float stepDt) {
    return implementation ? implementation->updateAdvectionStep(inflowTemperaturesC, stepDt) : false;
}
