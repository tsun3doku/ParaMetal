#define NOMINMAX
#include <Windows.h>
#include "../HeatGlobalSolver.hpp"
#include "HeatGlobalSolverKernels.cuh"
#include "../../cuda/CudaBuffer.cuh"
#include "../../cuda/CudaCheck.cuh"
#include "../../cuda/CudaPcg.cuh"
#include "../../cuda/CudaExternalBuffer.hpp"
#include "../../cuda/CudaVulkanDevice.hpp"
#include "../../vulkan/VulkanDevice.hpp"
#include "../../vulkan/VulkanExternalBuffer.hpp"

#include <cuda_runtime.h>
#include <vulkan/vulkan_win32.h>

#include <algorithm>
#include <iostream>
#include <vector>

class HeatGlobalSolver::Implementation {
public:
    struct DeviceTemperatureNodes {
        float* d_temperatureA = nullptr;
        float* d_temperatureB = nullptr;
        cudaUtils::CudaBuffer<uint32_t> localNodeIds;
        uint32_t solverNodeOffset = 0;
        uint32_t nodeCount = 0;
    };

    ~Implementation() { cleanup(); }

    bool initialize(
        VulkanDevice& vulkanDevice,
        const std::vector<int>& rowOffsets,
        const std::vector<int>& columnIndices,
        const std::vector<float>& values,
        const std::vector<float>& masses,
        const std::vector<TemperatureNodes>& inputTemperatureNodes,
        const std::vector<float>& rhsAdditions) {
        cleanup();
        if (masses.empty() || rowOffsets.size() != masses.size() + 1 ||
            columnIndices.size() != values.size() || inputTemperatureNodes.empty() ||
            (!rhsAdditions.empty() && rhsAdditions.size() != masses.size())) {
            return false;
        }

        device = vulkanDevice.getDevice();
        nodeCount = static_cast<uint32_t>(masses.size());

        cudaDevice = cudaVulkan::findDevice(vulkanDevice.getPhysicalDevice());
        if (cudaDevice < 0 || !cudaUtils::cudaCheck(cudaSetDevice(cudaDevice), "cudaSetDevice")) return false;

        if (!cudaUtils::cudaCheck(cudaStreamCreateWithFlags(&solveStream, cudaStreamNonBlocking), "create solveStream") ||
            !cudaUtils::cudaCheck(cudaEventCreateWithFlags(&solveCompleteEvent, cudaEventDisableTiming), "create completeEvent")) {
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
            devNode.solverNodeOffset = inputNode.solverNodeOffset;
            devNode.nodeCount = static_cast<uint32_t>(inputNode.localNodeIds.size());

            if (devNode.nodeCount > 0) {
                if (!devNode.localNodeIds.allocate(devNode.nodeCount) ||
                    !devNode.localNodeIds.upload(inputNode.localNodeIds.data(), devNode.nodeCount)) {
                    cleanup();
                    return false;
                }
            }
        }

        // Allocate CSR matrix and vector memory
        matrixValueCount = static_cast<uint32_t>(values.size());

        if (!d_rowOffsets.allocate(static_cast<uint32_t>(rowOffsets.size())) ||
            !d_columnIndices.allocate(matrixValueCount) ||
            !d_values.allocate(matrixValueCount) ||
            !d_invDiag.allocate(nodeCount) ||
            !d_thermalMasses.allocate(nodeCount) ||
            !d_rhsAdditions.allocate(nodeCount) ||
            !d_rhs.allocate(nodeCount) ||
            !d_x.allocate(nodeCount) ||
            !d_r.allocate(nodeCount) ||
            !d_z.allocate(nodeCount) ||
            !d_p.allocate(nodeCount) ||
            !d_Ap.allocate(nodeCount) ||
            !d_status.allocate(1)) {
            cleanup();
            return false;
        }

        if (!d_rowOffsets.upload(rowOffsets.data(), static_cast<uint32_t>(rowOffsets.size())) ||
            !d_columnIndices.upload(columnIndices.data(), matrixValueCount) ||
            !d_values.upload(values.data(), matrixValueCount) ||
            !d_thermalMasses.upload(masses.data(), nodeCount)) {
            cleanup();
            return false;
        }

        if (!rhsAdditions.empty()) {
            if (!d_rhsAdditions.upload(rhsAdditions.data(), nodeCount)) {
                cleanup();
                return false;
            }
        } else {
            if (!cudaUtils::cudaCheck(cudaMemset(d_rhsAdditions.get(), 0, nodeCount * sizeof(float)), "zero d_rhsAdditions")) {
                cleanup();
                return false;
            }
        }

        // Compute initial Jacobi inverse diagonal
        const uint32_t gridNodes = (nodeCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;
        heat::globalsolver::computeInvDiagKernel<<<gridNodes, heat::globalsolver::BlockSize, 0, solveStream>>>(
            nodeCount, d_rowOffsets.get(), d_columnIndices.get(), d_values.get(), d_invDiag.get());
        if (!cudaUtils::cudaCheck(cudaStreamSynchronize(solveStream), "computeInvDiagKernel")) {
            cleanup();
            return false;
        }

        initialized = true;

        // Build execution graphs
        if (!buildGraphs()) {
            std::cerr << "[HeatGlobalSolver] Graph construction failed; using direct stream launch." << std::endl;
        }

        return true;
    }

    bool updateValues(
        const std::vector<float>& values,
        const std::vector<float>& masses,
        const std::vector<float>& rhsAdditions) {
        if (!initialized) return false;
        waitIdle();
        cudaSetDevice(cudaDevice);

        if (values.size() != matrixValueCount || masses.size() != nodeCount ||
            (!rhsAdditions.empty() && rhsAdditions.size() != nodeCount)) {
            return false;
        }

        if (!d_values.upload(values.data(), matrixValueCount)) {
            return false;
        }

        if (!d_thermalMasses.uploadAsync(masses.data(), nodeCount, solveStream)) {
            return false;
        }

        if (!rhsAdditions.empty()) {
            if (!d_rhsAdditions.uploadAsync(rhsAdditions.data(), nodeCount, solveStream)) {
                return false;
            }
        } else {
            if (!cudaUtils::cudaCheck(cudaMemsetAsync(d_rhsAdditions.get(), 0, nodeCount * sizeof(float), solveStream), "zero d_rhsAdditions")) {
                return false;
            }
        }

        const uint32_t gridNodes = (nodeCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;
        heat::globalsolver::computeInvDiagKernel<<<gridNodes, heat::globalsolver::BlockSize, 0, solveStream>>>(
            nodeCount, d_rowOffsets.get(), d_columnIndices.get(), d_values.get(), d_invDiag.get());

        return cudaUtils::cudaCheck(cudaStreamSynchronize(solveStream), "sync updateValues");
    }

    bool configureAdvectionTopology(
        const std::vector<heat::AdvectionFace>& faces,
        const std::vector<heat::AdvectionBoundaryFace>& boundaryFaces,
        const float* deviceFaceFlux,
        const float* deviceBoundaryFlux) {
        if (!initialized) return false;
        waitIdle();
        cudaSetDevice(cudaDevice);

        advectionFaceCount = static_cast<uint32_t>(faces.size());
        advectionBoundaryFaceCount = static_cast<uint32_t>(boundaryFaces.size());
        d_advectionFaceFlux = deviceFaceFlux;
        d_advectionBoundaryFlux = deviceBoundaryFlux;

        d_advectionFaces.reset();
        if (advectionFaceCount > 0) {
            if (!d_advectionFaces.allocate(advectionFaceCount) ||
                !d_advectionFaces.uploadAsync(faces.data(), advectionFaceCount, solveStream)) {
                return false;
            }
        }

        d_advectionBoundaryFaces.reset();
        if (advectionBoundaryFaceCount > 0) {
            if (!d_advectionBoundaryFaces.allocate(advectionBoundaryFaceCount) ||
                !d_advectionBoundaryFaces.uploadAsync(boundaryFaces.data(), advectionBoundaryFaceCount, solveStream)) {
                return false;
            }
        }

        if (nodeCount > 0) {
            if (d_advectionDeltaT.size() < nodeCount && !d_advectionDeltaT.allocate(nodeCount)) {
                return false;
            }
            if (!cudaUtils::cudaCheck(cudaMemsetAsync(d_advectionDeltaT.get(), 0, nodeCount * sizeof(float), solveStream), "zero d_advectionDeltaT")) {
                return false;
            }
        }

        return cudaUtils::cudaCheck(cudaStreamSynchronize(solveStream), "sync configureAdvectionTopology");
    }

    void updateAdvectionStep(
        float inletTemperatureC,
        uint32_t subcycleCount,
        float subcycleDt) {
        advectionInletTemperatureC = inletTemperatureC;
        advectionSubcycleCount = std::max(1u, subcycleCount);
        advectionSubcycleDt = subcycleDt;
    }

    bool launch(
        bool sourceIsA,
        uint64_t destinationVulkanReleaseValue,
        uint64_t cudaReadyValue) {
        if (!initialized) return false;
        cudaSetDevice(cudaDevice);

        // Wait for Vulkan to finish reading the destination buffer before overwriting
        if (destinationVulkanReleaseValue > 0) {
            cudaExternalSemaphoreWaitParams waitParams{};
            waitParams.params.fence.value = destinationVulkanReleaseValue;
            if (!cudaUtils::cudaCheck(cudaWaitExternalSemaphoresAsync(&vulkanReleaseSemaphore, &waitParams, 1, solveStream), "wait Vulkan release")) {
                return false;
            }
        }

        const uint32_t gridNodes = (nodeCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;

        // Gather source temperature into d_x
        for (const auto& node : temperatureNodes) {
            const uint32_t blocks = (node.nodeCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;
            heat::globalsolver::gatherTemperaturesKernel<<<blocks, heat::globalsolver::BlockSize, 0, solveStream>>>(
                sourceIsA ? node.d_temperatureA : node.d_temperatureB,
                node.localNodeIds.get(), node.nodeCount, node.solverNodeOffset,
                d_x.get());
        }

        // Subcycled explicit upwind advection on fluid temperature components of d_x
        if ((advectionFaceCount > 0 || advectionBoundaryFaceCount > 0) && d_advectionDeltaT.get()) {
            const uint32_t faceBlocks = (advectionFaceCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;
            const uint32_t bfaceBlocks = (advectionBoundaryFaceCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;

            for (uint32_t sub = 0; sub < advectionSubcycleCount; ++sub) {
                if (advectionFaceCount > 0 && d_advectionFaceFlux) {
                    heat::globalsolver::advectFluidKernel<<<faceBlocks, heat::globalsolver::BlockSize, 0, solveStream>>>(
                        advectionFaceCount, d_advectionFaces.get(), d_advectionFaceFlux, d_advectionDeltaT.get(), d_x.get(), advectionSubcycleDt);
                }
                if (advectionBoundaryFaceCount > 0 && d_advectionBoundaryFlux) {
                    heat::globalsolver::advectBoundaryKernel<<<bfaceBlocks, heat::globalsolver::BlockSize, 0, solveStream>>>(
                        advectionBoundaryFaceCount, d_advectionBoundaryFaces.get(), d_advectionBoundaryFlux, d_advectionDeltaT.get(), d_x.get(), advectionInletTemperatureC, advectionSubcycleDt);
                }
                heat::globalsolver::applyAdvectionUpdatesKernel<<<gridNodes, heat::globalsolver::BlockSize, 0, solveStream>>>(
                    nodeCount, d_x.get(), d_advectionDeltaT.get());
            }
        }

        // Build authoritative RHS: rhs = thermalMasses * x + rhsAdditions
        heat::globalsolver::buildRhsKernel<<<gridNodes, heat::globalsolver::BlockSize, 0, solveStream>>>(
            nodeCount, d_thermalMasses.get(), d_x.get(), d_rhsAdditions.get(), d_rhs.get());

        // Launch the PCG solver graph
        cudaGraphExec_t graphExec = sourceIsA ? graphExecAToB : graphExecBToA;
        if (graphExec) {
            if (!cudaUtils::cudaCheck(cudaGraphLaunch(graphExec, solveStream), "launch CUDA graph")) {
                return false;
            }
        } else {
            recordPcgSequence(sourceIsA, solveStream);
        }

        // Record completion polling event
        cudaEventRecord(solveCompleteEvent, solveStream);

        // Signal CUDA ready semaphore
        if (cudaReadyValue > 0) {
            cudaExternalSemaphoreSignalParams signalParams{};
            signalParams.params.fence.value = cudaReadyValue;
            if (!cudaUtils::cudaCheck(cudaSignalExternalSemaphoresAsync(&cudaReadySemaphore, &signalParams, 1, solveStream), "signal CUDA ready")) {
                return false;
            }
        }

        return true;
    }

    SolvePollResult poll() const {
        if (!initialized || !solveCompleteEvent) return SolvePollResult::Complete;
        const cudaError_t status = cudaEventQuery(solveCompleteEvent);
        if (status == cudaSuccess) {
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

    bool hasSolveFailed() const {
        if (!d_status.get()) return false;
        cudaPcg::PcgStatus hostStatus{};
        if (cudaMemcpy(&hostStatus, d_status.get(), sizeof(cudaPcg::PcgStatus), cudaMemcpyDeviceToHost) == cudaSuccess) {
            return hostStatus.failed != 0 ||
                   (hostStatus.converged == 0 &&
                    hostStatus.iterationCount >= heat::globalsolver::MaxIterations);
        }
        return false;
    }

    VkSemaphore getCudaReadySemaphore() const { return cudaReadyVkSemaphore; }
    VkSemaphore getVulkanReleaseSemaphore() const { return vulkanReleaseVkSemaphore; }

    void destroyGraphs() {
        if (graphExecAToB) { cudaGraphExecDestroy(graphExecAToB); graphExecAToB = nullptr; }
        if (graphExecBToA) { cudaGraphExecDestroy(graphExecBToA); graphExecBToA = nullptr; }
        if (graphAToB) { cudaGraphDestroy(graphAToB); graphAToB = nullptr; }
        if (graphBToA) { cudaGraphDestroy(graphBToA); graphBToA = nullptr; }
    }

    void cleanup() {
        if (cudaDevice >= 0) {
            cudaSetDevice(cudaDevice);
        }
        waitIdle();
        destroyGraphs();
        if (solveCompleteEvent) { cudaEventDestroy(solveCompleteEvent); solveCompleteEvent = nullptr; }
        if (solveStream) { cudaStreamDestroy(solveStream); solveStream = nullptr; }

        d_rowOffsets.reset();
        d_columnIndices.reset();
        d_values.reset();
        d_invDiag.reset();
        d_thermalMasses.reset();
        d_rhs.reset();
        d_x.reset();
        d_r.reset();
        d_z.reset();
        d_p.reset();
        d_Ap.reset();
        d_status.reset();
        d_rhsAdditions.reset();
        d_advectionFaces.reset();
        d_advectionBoundaryFaces.reset();
        d_advectionDeltaT.reset();

        advectionFaceCount = 0;
        advectionBoundaryFaceCount = 0;
        d_advectionFaceFlux = nullptr;
        d_advectionBoundaryFlux = nullptr;
        advectionInletTemperatureC = 0.0f;
        advectionSubcycleCount = 1;
        advectionSubcycleDt = 0.0f;

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
        nodeCount = 0;
        matrixValueCount = 0;
        initialized = false;
    }

private:
    void recordPcgSequence(bool sourceIsA, cudaStream_t stream) {
        cudaPcg::recordPcgSolve(
            stream,
            nodeCount,
            d_rowOffsets.get(),
            d_columnIndices.get(),
            d_values.get(),
            d_invDiag.get(),
            d_x.get(),
            d_rhs.get(),
            d_Ap.get(),
            d_r.get(),
            d_z.get(),
            d_p.get(),
            d_status.get(),
            heat::globalsolver::MaxIterations,
            heat::globalsolver::ConvergenceTolerance,
            true);

        // Scatter solution into alternate destination buffer
        for (const auto& node : temperatureNodes) {
            const uint32_t blocks = (node.nodeCount + heat::globalsolver::BlockSize - 1) / heat::globalsolver::BlockSize;
            heat::globalsolver::scatterTemperaturesKernel<<<blocks, heat::globalsolver::BlockSize, 0, stream>>>(
                d_x.get(), node.solverNodeOffset, node.localNodeIds.get(), node.nodeCount,
                sourceIsA ? node.d_temperatureB : node.d_temperatureA);
        }
    }

    bool buildGraphs() {
        if (!solveStream) return false;

        // Capture Graph A -> B
        if (!cudaUtils::cudaCheck(cudaStreamBeginCapture(solveStream, cudaStreamCaptureModeGlobal), "begin capture A->B")) return false;
        recordPcgSequence(true, solveStream);
        if (!cudaUtils::cudaCheck(cudaStreamEndCapture(solveStream, &graphAToB), "end capture A->B")) return false;
        if (!cudaUtils::cudaCheck(cudaGraphInstantiate(&graphExecAToB, graphAToB, nullptr, nullptr, 0), "instantiate graph A->B")) return false;

        // Capture Graph B -> A
        if (!cudaUtils::cudaCheck(cudaStreamBeginCapture(solveStream, cudaStreamCaptureModeGlobal), "begin capture B->A")) return false;
        recordPcgSequence(false, solveStream);
        if (!cudaUtils::cudaCheck(cudaStreamEndCapture(solveStream, &graphBToA), "end capture B->A")) return false;
        if (!cudaUtils::cudaCheck(cudaGraphInstantiate(&graphExecBToA, graphBToA, nullptr, nullptr, 0), "instantiate graph B->A")) return false;

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
    cudaEvent_t solveCompleteEvent = nullptr;

    VkSemaphore cudaReadyVkSemaphore = VK_NULL_HANDLE;
    cudaExternalSemaphore_t cudaReadySemaphore = nullptr;

    VkSemaphore vulkanReleaseVkSemaphore = VK_NULL_HANDLE;
    cudaExternalSemaphore_t vulkanReleaseSemaphore = nullptr;

    cudaGraph_t graphAToB = nullptr;
    cudaGraphExec_t graphExecAToB = nullptr;
    cudaGraph_t graphBToA = nullptr;
    cudaGraphExec_t graphExecBToA = nullptr;

    cudaUtils::CudaBuffer<int> d_rowOffsets;
    cudaUtils::CudaBuffer<int> d_columnIndices;
    cudaUtils::CudaBuffer<float> d_values;
    cudaUtils::CudaBuffer<float> d_invDiag;
    cudaUtils::CudaBuffer<float> d_thermalMasses;
    cudaUtils::CudaBuffer<float> d_rhsAdditions;
    cudaUtils::CudaBuffer<float> d_rhs;
    cudaUtils::CudaBuffer<float> d_x;
    cudaUtils::CudaBuffer<float> d_r;
    cudaUtils::CudaBuffer<float> d_z;
    cudaUtils::CudaBuffer<float> d_p;
    cudaUtils::CudaBuffer<float> d_Ap;
    cudaUtils::CudaBuffer<cudaPcg::PcgStatus> d_status;

    cudaUtils::CudaBuffer<heat::AdvectionFace> d_advectionFaces;
    uint32_t advectionFaceCount = 0;
    cudaUtils::CudaBuffer<heat::AdvectionBoundaryFace> d_advectionBoundaryFaces;
    uint32_t advectionBoundaryFaceCount = 0;
    const float* d_advectionFaceFlux = nullptr;
    const float* d_advectionBoundaryFlux = nullptr;
    float advectionInletTemperatureC = 0.0f;
    cudaUtils::CudaBuffer<float> d_advectionDeltaT;
    uint32_t advectionSubcycleCount = 1;
    float advectionSubcycleDt = 0.0f;

    std::vector<DeviceTemperatureNodes> temperatureNodes;

    uint32_t nodeCount = 0;
    uint32_t matrixValueCount = 0;
    bool initialized = false;
};

HeatGlobalSolver::HeatGlobalSolver() : implementation(std::make_unique<Implementation>()) {}
HeatGlobalSolver::~HeatGlobalSolver() = default;

bool HeatGlobalSolver::initialize(
    VulkanDevice& vulkanDevice,
    const std::vector<int>& rowOffsets,
    const std::vector<int>& columnIndices,
    const std::vector<float>& values,
    const std::vector<float>& thermalMasses,
    const std::vector<TemperatureNodes>& temperatureNodes,
    const std::vector<float>& rhsAdditions) {
    return implementation->initialize(
        vulkanDevice, rowOffsets, columnIndices, values, thermalMasses,
        temperatureNodes, rhsAdditions);
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
    const std::vector<float>& values,
    const std::vector<float>& thermalMasses,
    const std::vector<float>& rhsAdditions) {
    return implementation->updateValues(values, thermalMasses, rhsAdditions);
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

void HeatGlobalSolver::cleanup() {
    implementation->cleanup();
}

bool HeatGlobalSolver::configureAdvectionTopology(
    const std::vector<heat::AdvectionFace>& faces,
    const std::vector<heat::AdvectionBoundaryFace>& boundaryFaces,
    const float* deviceFaceFlux,
    const float* deviceBoundaryFlux) {
    return implementation ? implementation->configureAdvectionTopology(faces, boundaryFaces, deviceFaceFlux, deviceBoundaryFlux) : false;
}

void HeatGlobalSolver::updateAdvectionStep(
    float inletTemperatureC,
    uint32_t subcycleCount,
    float subcycleDt) {
    if (implementation) {
        implementation->updateAdvectionStep(inletTemperatureC, subcycleCount, subcycleDt);
    }
}
