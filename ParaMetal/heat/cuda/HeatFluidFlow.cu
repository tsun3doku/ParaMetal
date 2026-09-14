#include "../HeatFluidFlow.hpp"
#include "../../cuda/CudaCheck.cuh"
#include "../../cuda/CudaPcg.cuh"
#include "../../cuda/CudaVulkanDevice.hpp"
#include "../../vulkan/VulkanDevice.hpp"

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <queue>
#include <vector>
#include <map>
#include <utility>
#include <iostream>

namespace heat {

constexpr uint32_t BlockSize = 256;
constexpr uint32_t MaxPcgIterations = 300;
constexpr float PcgConvergenceTolerance = 1e-7f;

struct GpuFluidProjectionFace {
    uint32_t nodeA;
    uint32_t nodeB;
    float weight;
    float area;
    glm::vec3 normal;
};

struct GpuFluidBoundaryFace {
    uint32_t node;
    float area;
    glm::vec3 outwardNormal;
};

__global__ void buildRawFluxKernel(
    uint32_t faceCount,
    uint32_t boundaryFaceCount,
    const GpuFluidProjectionFace* __restrict__ faces,
    const GpuFluidBoundaryFace* __restrict__ boundaryFaces,
    glm::vec3 velocity,
    float* __restrict__ faceRawFlux,
    float* __restrict__ boundaryFlux) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < faceCount) {
        const GpuFluidProjectionFace f = faces[idx];
        faceRawFlux[idx] = (velocity.x * f.normal.x + velocity.y * f.normal.y + velocity.z * f.normal.z) * f.area;
    }
    if (idx < boundaryFaceCount) {
        const GpuFluidBoundaryFace bf = boundaryFaces[idx];
        boundaryFlux[idx] = (velocity.x * bf.outwardNormal.x + velocity.y * bf.outwardNormal.y + velocity.z * bf.outwardNormal.z) * bf.area;
    }
}

__global__ void accumulateFaceDivergenceKernel(
    uint32_t faceCount,
    const GpuFluidProjectionFace* __restrict__ faces,
    const float* __restrict__ rawFlux,
    float* __restrict__ divergence) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;
    const GpuFluidProjectionFace f = faces[idx];
    const float fl = rawFlux[idx];
    atomicAdd(&divergence[f.nodeA], fl);
    atomicAdd(&divergence[f.nodeB], -fl);
}

__global__ void accumulateBoundaryDivergenceKernel(
    uint32_t boundaryFaceCount,
    const GpuFluidBoundaryFace* __restrict__ bfaces,
    const float* __restrict__ boundaryFlux,
    float* __restrict__ divergence) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundaryFaceCount) return;
    const GpuFluidBoundaryFace bf = bfaces[idx];
    atomicAdd(&divergence[bf.node], boundaryFlux[idx]);
}

__global__ void finalizeDivergenceRhsKernel(
    uint32_t nodeCount,
    const uint8_t* __restrict__ isPinned,
    float* __restrict__ divergence) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nodeCount) {
        if (isPinned[idx] != 0) {
            divergence[idx] = 0.0f;
        } else {
            divergence[idx] = -divergence[idx];
        }
    }
}

__global__ void correctFluxKernel(
    uint32_t faceCount,
    const GpuFluidProjectionFace* __restrict__ faces,
    const float* __restrict__ rawFlux,
    const float* __restrict__ psi,
    float* __restrict__ correctedFlux) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;
    const GpuFluidProjectionFace f = faces[idx];
    // Conservative Hodge-Helmholtz correction: flux = rawFlux - w * (psi_B - psi_A)
    correctedFlux[idx] = rawFlux[idx] - f.weight * (psi[f.nodeB] - psi[f.nodeA]);
}

__global__ void accumulateFaceOutflowKernel(
    uint32_t faceCount,
    const GpuFluidProjectionFace* __restrict__ faces,
    const float* __restrict__ faceFlux,
    float* __restrict__ outflow) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;
    const GpuFluidProjectionFace f = faces[idx];
    const float fl = faceFlux[idx];
    if (fl > 0.0f) {
        atomicAdd(&outflow[f.nodeA], fl);
    } else if (fl < 0.0f) {
        atomicAdd(&outflow[f.nodeB], -fl);
    }
}

__global__ void accumulateBoundaryOutflowKernel(
    uint32_t boundaryFaceCount,
    const GpuFluidBoundaryFace* __restrict__ bfaces,
    const float* __restrict__ boundaryFlux,
    float* __restrict__ outflow) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= boundaryFaceCount) return;
    const GpuFluidBoundaryFace bf = bfaces[idx];
    const float fl = boundaryFlux[idx];
    if (fl > 0.0f) {
        atomicAdd(&outflow[bf.node], fl);
    }
}

__global__ void computeNodeOutflowRateKernel(
    uint32_t nodeCount,
    const float* __restrict__ volumes,
    float* __restrict__ outflowAndRate) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nodeCount) return;
    outflowAndRate[idx] = outflowAndRate[idx] / volumes[idx];
}

HeatFluidFlow::HeatFluidFlow() = default;

HeatFluidFlow::~HeatFluidFlow() {
    cleanup();
}

bool HeatFluidFlow::initialize(VulkanDevice& vulkanDevice) {
    cleanup();

    cudaDevice = cudaVulkan::findDevice(vulkanDevice.getPhysicalDevice());
    if (cudaDevice < 0 || !cudaUtils::cudaCheck(cudaSetDevice(cudaDevice), "cudaSetDevice")) {
        return false;
    }

    cudaStream_t stream = nullptr;
    if (!cudaUtils::cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create flowStream")) {
        return false;
    }
    flowStreamHandle = reinterpret_cast<void*>(stream);
    initialized = true;
    return true;
}

void HeatFluidFlow::releaseTopologyBuffers() {
    if (d_rowOffsets) { cudaFree(d_rowOffsets); d_rowOffsets = nullptr; }
    if (d_columnIndices) { cudaFree(d_columnIndices); d_columnIndices = nullptr; }
    if (d_values) { cudaFree(d_values); d_values = nullptr; }
    if (d_invDiag) { cudaFree(d_invDiag); d_invDiag = nullptr; }
    if (d_isPinned) { cudaFree(d_isPinned); d_isPinned = nullptr; }
    if (d_faces) { cudaFree(d_faces); d_faces = nullptr; }
    if (d_boundaryFaces) { cudaFree(d_boundaryFaces); d_boundaryFaces = nullptr; }
    if (d_faceFlux) { cudaFree(d_faceFlux); d_faceFlux = nullptr; }
    if (d_boundaryFlux) { cudaFree(d_boundaryFlux); d_boundaryFlux = nullptr; }
    if (d_faceRawFlux) { cudaFree(d_faceRawFlux); d_faceRawFlux = nullptr; }
    if (d_divergence) { cudaFree(d_divergence); d_divergence = nullptr; }
    if (d_psi) { cudaFree(d_psi); d_psi = nullptr; }
    if (d_p) { cudaFree(d_p); d_p = nullptr; }
    if (d_Ap) { cudaFree(d_Ap); d_Ap = nullptr; }
    if (d_r) { cudaFree(d_r); d_r = nullptr; }
    if (d_z) { cudaFree(d_z); d_z = nullptr; }
    if (d_status) { cudaFree(d_status); d_status = nullptr; }
    if (d_fluidVolumes) { cudaFree(d_fluidVolumes); d_fluidVolumes = nullptr; }
    if (d_maxOutflowRate) { cudaFree(d_maxOutflowRate); d_maxOutflowRate = nullptr; }
    if (d_cubScratch) { cudaFree(d_cubScratch); d_cubScratch = nullptr; }
    cubScratchBytes = 0;
}

void HeatFluidFlow::clearTopology() {
    releaseTopologyBuffers();

    nodeCount = 0;
    faceCount = 0;
    boundaryFaceCount = 0;
    componentCount = 0;

    fluidFaces.clear();
    fluidBoundaryFaces.clear();
    nodeComponent.clear();

    timeStep = {};
}

bool HeatFluidFlow::buildTopology(
    const std::vector<topology::Fragment>& fragments,
    const std::vector<topology::Face>& faces,
    const std::vector<topology::BoundaryFace>& boundaryFaces,
    const std::vector<uint32_t>& globalFragmentToFluidNode,
    const std::vector<uint32_t>& fluidNodeToGlobalFragment) {
    if (!initialized) {
        std::cerr << "[HeatFluidFlow] buildTopology called before initialize" << std::endl;
        return false;
    }

    if (!cudaUtils::cudaCheck(cudaSetDevice(cudaDevice), "cudaSetDevice")) {
        return false;
    }

    clearTopology();

    if (globalFragmentToFluidNode.size() != fragments.size()) {
        std::cerr << "[HeatFluidFlow] buildTopology FAILED: globalFragmentToFluidNode size ("
                  << globalFragmentToFluidNode.size() << ") != fragments size ("
                  << fragments.size() << ")" << std::endl;
        clearTopology();
        return false;
    }

    std::vector<float> fluidVolumes(fluidNodeToGlobalFragment.size(), 0.0f);
    for (uint32_t node = 0; node < fluidNodeToGlobalFragment.size(); ++node) {
        const uint32_t frag = fluidNodeToGlobalFragment[node];
        if (frag >= fragments.size() ||
            globalFragmentToFluidNode[frag] != node ||
            !std::isfinite(fragments[frag].volume) ||
            !(fragments[frag].volume > 0.0f)) {
            std::cerr << "[HeatFluidFlow] buildTopology FAILED: invalid mapping or non-positive volume on fluid node "
                      << node << " (frag " << frag << ")" << std::endl;
            clearTopology();
            return false;
        }
        fluidVolumes[node] = fragments[frag].volume;
    }

    nodeCount = static_cast<uint32_t>(fluidVolumes.size());
    if (nodeCount == 0) {
        timeStep = FluidTimeStep{};
        return true;
    }

    // Extract fluid-fluid faces and calculate projection weight
    for (uint32_t faceIdx = 0; faceIdx < faces.size(); ++faceIdx) {
        const auto& face = faces[faceIdx];
        if (face.fragmentA >= fragments.size() || face.fragmentB >= fragments.size()) {
            continue;
        }

        const uint32_t nodeA = globalFragmentToFluidNode[face.fragmentA];
        const uint32_t nodeB = globalFragmentToFluidNode[face.fragmentB];

        if (nodeA != UINT32_MAX && nodeB != UINT32_MAX && nodeA != nodeB) {
            const float dA = face.orthogonalDistanceA;
            const float dB = face.orthogonalDistanceB;
            const float dist = dA + dB;

            if (!std::isfinite(face.area) || !(face.area > 0.0f) ||
                !std::isfinite(face.normal.x) || !std::isfinite(face.normal.y) || !std::isfinite(face.normal.z) ||
                !std::isfinite(dA) || !std::isfinite(dB) ||
                dA < 0.0f || dB < 0.0f ||
                !std::isfinite(dist) || !(dist > 0.0f)) {
                std::cerr << "[HeatFluidFlow] buildTopology FAILED: invalid face geometry ("
                          << "area=" << face.area << ", dA=" << dA << ", dB=" << dB << ", dist=" << dist
                          << ") on face " << faceIdx << std::endl;
                clearTopology();
                return false;
            }

            const float weight = face.area / dist;
            if (!std::isfinite(weight) || !(weight > 0.0f)) {
                std::cerr << "[HeatFluidFlow] buildTopology FAILED: non-finite or non-positive weight on face "
                          << faceIdx << std::endl;
                clearTopology();
                return false;
            }

            HeatFluidFace ff{};
            ff.faceIndex = faceIdx;
            ff.fluidNodeA = nodeA;
            ff.fluidNodeB = nodeB;
            ff.normal = face.normal;
            ff.area = face.area;
            ff.projectionWeight = weight;
            fluidFaces.push_back(ff);
        }
    }

    // Extract fluid domain boundary faces
    for (uint32_t bfIdx = 0; bfIdx < boundaryFaces.size(); ++bfIdx) {
        const auto& bf = boundaryFaces[bfIdx];
        if (bf.fragment >= fragments.size()) {
            continue;
        }

        const uint32_t node = globalFragmentToFluidNode[bf.fragment];
        if (node != UINT32_MAX && bf.domainFace < 6) {
            if (!std::isfinite(bf.area) || !(bf.area > 0.0f)) {
                std::cerr << "[HeatFluidFlow] buildTopology FAILED: non-finite or non-positive boundary face area on boundary face "
                          << bfIdx << std::endl;
                clearTopology();
                return false;
            }

            HeatFluidBoundaryFace fbf{};
            fbf.boundaryFaceIndex = bfIdx;
            fbf.fluidNode = node;
            fbf.outwardNormal = getDomainBoxNormal(bf.domainFace);
            fbf.area = bf.area;
            fluidBoundaryFaces.push_back(fbf);
        }
    }

    faceCount = static_cast<uint32_t>(fluidFaces.size());
    boundaryFaceCount = static_cast<uint32_t>(fluidBoundaryFaces.size());

    // Build weighted adjacency graph on fluid nodes
    std::vector<std::vector<std::pair<uint32_t, float>>> adjacency(nodeCount);
    for (const auto& ff : fluidFaces) {
        adjacency[ff.fluidNodeA].emplace_back(ff.fluidNodeB, ff.projectionWeight);
        adjacency[ff.fluidNodeB].emplace_back(ff.fluidNodeA, ff.projectionWeight);
    }

    // BFS Connected Component Detection on fluid graph
    nodeComponent.assign(nodeCount, UINT32_MAX);
    std::vector<uint32_t> componentPinnedNode;
    uint32_t curComp = 0;
    for (uint32_t i = 0; i < nodeCount; ++i) {
        if (nodeComponent[i] != UINT32_MAX) continue;

        std::queue<uint32_t> q;
        q.push(i);
        nodeComponent[i] = curComp;

        uint32_t bestPinNode = i;
        float maxVol = fluidVolumes[i];

        while (!q.empty()) {
            const uint32_t u = q.front();
            q.pop();

            if (fluidVolumes[u] > maxVol) {
                maxVol = fluidVolumes[u];
                bestPinNode = u;
            }

            for (const auto& [v, weight] : adjacency[u]) {
                if (nodeComponent[v] == UINT32_MAX) {
                    nodeComponent[v] = curComp;
                    q.push(v);
                }
            }
        }

        componentPinnedNode.push_back(bestPinNode);
        ++curComp;
    }
    componentCount = curComp;

    std::vector<uint8_t> hostIsPinned(nodeCount, 0);
    for (uint32_t pinNode : componentPinnedNode) {
        hostIsPinned[pinNode] = 1;
    }

    // Assemble Sparse Symmetric CSR Laplacian
    std::vector<int> rowOffsets;
    std::vector<int> columnIndices;
    std::vector<float> values;
    std::vector<float> invDiag(nodeCount, 1.0f);

    rowOffsets.reserve(nodeCount + 1);
    rowOffsets.push_back(0);

    for (uint32_t i = 0; i < nodeCount; ++i) {
        if (hostIsPinned[i] != 0) {
            // Pinned reference node
            columnIndices.push_back(static_cast<int>(i));
            values.push_back(1.0f);
            invDiag[i] = 1.0f;
            rowOffsets.push_back(static_cast<int>(columnIndices.size()));
            continue;
        }

        float diagSum = 0.0f;
        std::map<uint32_t, float> rowMap;

        for (const auto& [j, weight] : adjacency[i]) {
            diagSum += weight;
            if (hostIsPinned[j] == 0) {
                rowMap[j] -= weight;
            }
        }

        rowMap[i] += diagSum;

        for (const auto& [col, val] : rowMap) {
            columnIndices.push_back(static_cast<int>(col));
            values.push_back(val);
            if (col == i) {
                if (!std::isfinite(val) || !(val > 0.0f)) {
                    std::cerr << "[HeatFluidFlow] buildTopology FAILED: non-positive diagonal on unpinned node "
                              << i << " (val=" << val << ")" << std::endl;
                    clearTopology();
                    return false;
                }
                invDiag[i] = 1.0f / val;
            }
        }
        rowOffsets.push_back(static_cast<int>(columnIndices.size()));
    }

    std::vector<GpuFluidProjectionFace> gpuFaces(faceCount);
    for (uint32_t f = 0; f < faceCount; ++f) {
        gpuFaces[f].nodeA = fluidFaces[f].fluidNodeA;
        gpuFaces[f].nodeB = fluidFaces[f].fluidNodeB;
        gpuFaces[f].weight = fluidFaces[f].projectionWeight;
        gpuFaces[f].area = fluidFaces[f].area;
        gpuFaces[f].normal = fluidFaces[f].normal;
    }

    std::vector<GpuFluidBoundaryFace> gpuBfaces(boundaryFaceCount);
    for (uint32_t bf = 0; bf < boundaryFaceCount; ++bf) {
        gpuBfaces[bf].node = fluidBoundaryFaces[bf].fluidNode;
        gpuBfaces[bf].area = fluidBoundaryFaces[bf].area;
        gpuBfaces[bf].outwardNormal = fluidBoundaryFaces[bf].outwardNormal;
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(flowStreamHandle);

    const size_t rowOffsetBytes = rowOffsets.size() * sizeof(int);
    const size_t colIndexBytes = columnIndices.size() * sizeof(int);
    const size_t valueBytes = values.size() * sizeof(float);
    const size_t nodeBytes = nodeCount * sizeof(float);

    if (!cudaUtils::cudaCheck(cudaMalloc(&d_rowOffsets, rowOffsetBytes), "alloc d_rowOffsets") ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_columnIndices, colIndexBytes), "alloc d_columnIndices") ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_values, valueBytes), "alloc d_values") ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_invDiag, nodeBytes), "alloc d_invDiag") ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_isPinned, nodeCount * sizeof(uint8_t)), "alloc d_isPinned") ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_divergence, nodeBytes), "alloc d_divergence") ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_psi, nodeBytes), "alloc d_psi") ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_p, nodeBytes), "alloc d_p") ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_Ap, nodeBytes), "alloc d_Ap") ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_r, nodeBytes), "alloc d_r") ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_z, nodeBytes), "alloc d_z") ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_status, sizeof(cudaPcg::PcgStatus)), "alloc d_status") ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_fluidVolumes, nodeBytes), "alloc d_fluidVolumes") ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_maxOutflowRate, sizeof(float)), "alloc d_maxOutflowRate")) {
        clearTopology();
        return false;
    }

    if (faceCount > 0) {
        const size_t faceBytes = faceCount * sizeof(float);
        if (!cudaUtils::cudaCheck(cudaMalloc(&d_faces, faceCount * sizeof(GpuFluidProjectionFace)), "alloc d_faces") ||
            !cudaUtils::cudaCheck(cudaMalloc(&d_faceFlux, faceBytes), "alloc d_faceFlux") ||
            !cudaUtils::cudaCheck(cudaMalloc(&d_faceRawFlux, faceBytes), "alloc d_faceRawFlux")) {
            clearTopology();
            return false;
        }
    }

    if (boundaryFaceCount > 0) {
        const size_t bfaceBytes = boundaryFaceCount * sizeof(float);
        if (!cudaUtils::cudaCheck(cudaMalloc(&d_boundaryFaces, boundaryFaceCount * sizeof(GpuFluidBoundaryFace)), "alloc d_boundaryFaces") ||
            !cudaUtils::cudaCheck(cudaMalloc(&d_boundaryFlux, bfaceBytes), "alloc d_boundaryFlux")) {
            clearTopology();
            return false;
        }
    }

    // Upload topology to GPU
    if (!cudaUtils::cudaCheck(cudaMemcpyAsync(d_rowOffsets, rowOffsets.data(), rowOffsetBytes, cudaMemcpyHostToDevice, stream), "upload d_rowOffsets") ||
        !cudaUtils::cudaCheck(cudaMemcpyAsync(d_columnIndices, columnIndices.data(), colIndexBytes, cudaMemcpyHostToDevice, stream), "upload d_columnIndices") ||
        !cudaUtils::cudaCheck(cudaMemcpyAsync(d_values, values.data(), valueBytes, cudaMemcpyHostToDevice, stream), "upload d_values") ||
        !cudaUtils::cudaCheck(cudaMemcpyAsync(d_invDiag, invDiag.data(), nodeBytes, cudaMemcpyHostToDevice, stream), "upload d_invDiag") ||
        !cudaUtils::cudaCheck(cudaMemcpyAsync(d_isPinned, hostIsPinned.data(), nodeCount * sizeof(uint8_t), cudaMemcpyHostToDevice, stream), "upload d_isPinned") ||
        !cudaUtils::cudaCheck(cudaMemcpyAsync(d_fluidVolumes, fluidVolumes.data(), nodeBytes, cudaMemcpyHostToDevice, stream), "upload d_fluidVolumes")) {
        clearTopology();
        return false;
    }

    if (faceCount > 0) {
        const size_t faceBytes = faceCount * sizeof(float);
        if (!cudaUtils::cudaCheck(cudaMemcpyAsync(d_faces, gpuFaces.data(), faceCount * sizeof(GpuFluidProjectionFace), cudaMemcpyHostToDevice, stream), "upload d_faces") ||
            !cudaUtils::cudaCheck(cudaMemsetAsync(d_faceFlux, 0, faceBytes, stream), "zero d_faceFlux")) {
            clearTopology();
            return false;
        }
    }

    if (boundaryFaceCount > 0) {
        const size_t bfaceBytes = boundaryFaceCount * sizeof(float);
        if (!cudaUtils::cudaCheck(cudaMemcpyAsync(d_boundaryFaces, gpuBfaces.data(), boundaryFaceCount * sizeof(GpuFluidBoundaryFace), cudaMemcpyHostToDevice, stream), "upload d_boundaryFaces") ||
            !cudaUtils::cudaCheck(cudaMemsetAsync(d_boundaryFlux, 0, bfaceBytes, stream), "zero d_boundaryFlux")) {
            clearTopology();
            return false;
        }
    }

    // Allocate persistent CUB reduction scratch buffer
    cubScratchBytes = 0;
    if (cub::DeviceReduce::Max(nullptr, cubScratchBytes, d_divergence, d_maxOutflowRate, nodeCount, stream) != cudaSuccess ||
        !cudaUtils::cudaCheck(cudaMalloc(&d_cubScratch, cubScratchBytes), "alloc d_cubScratch")) {
        clearTopology();
        return false;
    }

    if (!cudaUtils::cudaCheck(cudaStreamSynchronize(stream), "sync buildTopology")) {
        clearTopology();
        return false;
    }

    return true;
}

bool HeatFluidFlow::updateTimeStep(float macroDt, float targetCourant, float maxOutflowRate) {
    if (!std::isfinite(macroDt) || !(macroDt > 0.0f) ||
        !std::isfinite(targetCourant) || !(targetCourant > 0.0f) ||
        !std::isfinite(maxOutflowRate) || maxOutflowRate < 0.0f) {
        return false;
    }

    timeStep.maxCourant = macroDt * maxOutflowRate;
    if (!std::isfinite(timeStep.maxCourant)) {
        std::cerr << "[HeatFluidFlow] Non-finite max Courant number detected: "
                  << timeStep.maxCourant << " (maxOutflowRate=" << maxOutflowRate << ")" << std::endl;
        return false;
    }

    const double required = std::ceil(double(timeStep.maxCourant) / double(targetCourant));
    if (!std::isfinite(required) || required > double(UINT32_MAX)) {
        std::cerr << "[HeatFluidFlow] Invalid subcycle requirement: " << required << std::endl;
        return false;
    }

    timeStep.subcycleCount = std::max(1u, static_cast<uint32_t>(required));
    timeStep.subcycleDt = macroDt / static_cast<float>(timeStep.subcycleCount);
    return true;
}

bool HeatFluidFlow::updatePrescribedFlow(
    const glm::vec3& inPrescribedVelocity,
    float macroDt,
    float inTargetCourant) {
    if (!std::isfinite(macroDt) || !(macroDt > 0.0f) ||
        !std::isfinite(inTargetCourant) || !(inTargetCourant > 0.0f) ||
        !std::isfinite(inPrescribedVelocity.x) ||
        !std::isfinite(inPrescribedVelocity.y) ||
        !std::isfinite(inPrescribedVelocity.z)) {
        std::cerr << "[HeatFluidFlow] updatePrescribedFlow FAILED: invalid parameters (macroDt="
                  << macroDt << ", targetCourant=" << inTargetCourant << ")" << std::endl;
        return false;
    }

    if (!initialized || nodeCount == 0 || !flowStreamHandle) {
        timeStep = FluidTimeStep{};
        return true;
    }

    cudaSetDevice(cudaDevice);

    std::vector<float> compNetFlux(componentCount, 0.0f);
    std::vector<float> compSumAbsFlux(componentCount, 0.0f);

    for (const auto& bf : fluidBoundaryFaces) {
        const float flux = glm::dot(inPrescribedVelocity, bf.outwardNormal) * bf.area;
        const uint32_t comp = nodeComponent[bf.fluidNode];
        if (comp < componentCount) {
            compNetFlux[comp] += flux;
            compSumAbsFlux[comp] += std::abs(flux);
        }
    }

    constexpr float absTol = 1e-7f;
    constexpr float relTol = 1e-4f;
    for (uint32_t c = 0; c < componentCount; ++c) {
        const float allowed = absTol + relTol * compSumAbsFlux[c];
        if (std::abs(compNetFlux[c]) > allowed) {
            std::cerr << "[HeatFluidFlow] Component " << c << " violates Neumann boundary compatibility: netFlux="
                      << compNetFlux[c] << " m^3/s, allowed=" << allowed << " m^3/s (sumAbs="
                      << compSumAbsFlux[c] << " m^3/s)" << std::endl;
            return false;
        }
    }

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(flowStreamHandle);

    // Evaluate raw face and boundary fluxes on GPU
    if (faceCount > 0 || boundaryFaceCount > 0) {
        const uint32_t maxFaceBlocks = (std::max(faceCount, boundaryFaceCount) + BlockSize - 1) / BlockSize;
        buildRawFluxKernel<<<maxFaceBlocks, BlockSize, 0, stream>>>(
            faceCount, boundaryFaceCount,
            reinterpret_cast<const GpuFluidProjectionFace*>(d_faces),
            reinterpret_cast<const GpuFluidBoundaryFace*>(d_boundaryFaces),
            inPrescribedVelocity,
            d_faceRawFlux, d_boundaryFlux);
    }

    // Calculate cell divergence on GPU
    const size_t nodeBytes = nodeCount * sizeof(float);
    cudaMemsetAsync(d_divergence, 0, nodeBytes, stream);

    if (faceCount > 0) {
        const uint32_t faceBlocks = (faceCount + BlockSize - 1) / BlockSize;
        accumulateFaceDivergenceKernel<<<faceBlocks, BlockSize, 0, stream>>>(
            faceCount, reinterpret_cast<const GpuFluidProjectionFace*>(d_faces), d_faceRawFlux, d_divergence);
    }

    if (boundaryFaceCount > 0) {
        const uint32_t bfaceBlocks = (boundaryFaceCount + BlockSize - 1) / BlockSize;
        accumulateBoundaryDivergenceKernel<<<bfaceBlocks, BlockSize, 0, stream>>>(
            boundaryFaceCount, reinterpret_cast<const GpuFluidBoundaryFace*>(d_boundaryFaces), d_boundaryFlux, d_divergence);
    }

    const uint32_t nodeBlocks = (nodeCount + BlockSize - 1) / BlockSize;
    finalizeDivergenceRhsKernel<<<nodeBlocks, BlockSize, 0, stream>>>(nodeCount, d_isPinned, d_divergence);

    cudaPcg::recordPcgSolve(
        stream,
        nodeCount,
        d_rowOffsets,
        d_columnIndices,
        d_values,
        d_invDiag,
        d_psi,
        d_divergence,
        d_Ap,
        d_r,
        d_z,
        d_p,
        reinterpret_cast<cudaPcg::PcgStatus*>(d_status),
        MaxPcgIterations,
        PcgConvergenceTolerance,
        false);

    // Apply conservative flux correction
    if (faceCount > 0) {
        const uint32_t faceBlocks = (faceCount + BlockSize - 1) / BlockSize;
        correctFluxKernel<<<faceBlocks, BlockSize, 0, stream>>>(
            faceCount, reinterpret_cast<const GpuFluidProjectionFace*>(d_faces), d_faceRawFlux, d_psi, d_faceFlux);
    }

    // Outflow rate calculation
    cudaMemsetAsync(d_divergence, 0, nodeBytes, stream);

    if (faceCount > 0) {
        const uint32_t faceBlocks = (faceCount + BlockSize - 1) / BlockSize;
        accumulateFaceOutflowKernel<<<faceBlocks, BlockSize, 0, stream>>>(
            faceCount, reinterpret_cast<const GpuFluidProjectionFace*>(d_faces), d_faceFlux, d_divergence);
    }

    if (boundaryFaceCount > 0) {
        const uint32_t bfaceBlocks = (boundaryFaceCount + BlockSize - 1) / BlockSize;
        accumulateBoundaryOutflowKernel<<<bfaceBlocks, BlockSize, 0, stream>>>(
            boundaryFaceCount, reinterpret_cast<const GpuFluidBoundaryFace*>(d_boundaryFaces), d_boundaryFlux, d_divergence);
    }

    // Calculate outflow rate per node
    computeNodeOutflowRateKernel<<<nodeBlocks, BlockSize, 0, stream>>>(
        nodeCount, d_fluidVolumes, d_divergence);

    // GPU max reduction of outflow rate
    cub::DeviceReduce::Max(
        d_cubScratch, cubScratchBytes,
        d_divergence, d_maxOutflowRate,
        nodeCount, stream);

    // Asynchronously copy PCG status and maxOutflowRate to host
    cudaPcg::PcgStatus status{};
    float hostMaxOutflowRate = 0.0f;

    if (!cudaUtils::cudaCheck(cudaMemcpyAsync(&status, d_status, sizeof(cudaPcg::PcgStatus), cudaMemcpyDeviceToHost, stream), "copy status") ||
        !cudaUtils::cudaCheck(cudaMemcpyAsync(&hostMaxOutflowRate, d_maxOutflowRate, sizeof(float), cudaMemcpyDeviceToHost, stream), "copy maxOutflowRate") ||
        !cudaUtils::cudaCheck(cudaStreamSynchronize(stream), "sync updatePrescribedFlow")) {
        return false;
    }

    // Verify PCG convergence
    if (status.failed || !status.converged) {
        const float relative =
            status.residual / std::max(status.residual0, 1e-30f);

        std::cerr
            << "[HeatFluidFlow] projection solve failed:"
            << " iterations=" << status.iterationCount
            << " residual0=" << status.residual0
            << " residual=" << status.residual
            << " relative=" << relative
            << " failed=" << status.failed
            << '\n';
        return false;
    }

    // Derive timestep
    if (!updateTimeStep(macroDt, inTargetCourant, hostMaxOutflowRate)) {
        return false;
    }

    return true;
}

void HeatFluidFlow::cleanup() {
    if (initialized && cudaDevice >= 0) {
        cudaSetDevice(cudaDevice);
    }

    if (flowStreamHandle) {
        auto stream = reinterpret_cast<cudaStream_t>(flowStreamHandle);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        flowStreamHandle = nullptr;
    }

    clearTopology();
    initialized = false;
    cudaDevice = -1;
}

} // namespace heat
