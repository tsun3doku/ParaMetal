#pragma once

#include "../vulkan/VulkanExternalBuffer.hpp"
#include "../cuda/CudaExternalBuffer.hpp"
#include "CFDGpuStructs.hpp"
#include <glm/glm.hpp>
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

class VulkanDevice;

namespace cfd {

class CFDRuntime {
public:
    CFDRuntime();
    ~CFDRuntime();

    bool initialize(
        VulkanDevice& device,
        int cudaDevice,
        uint32_t fluidNodeCount,
        uint32_t globalFaceCount,
        uint32_t globalBoundaryFaceCount);

    bool buildTopology(
        const std::vector<uint32_t>& fragmentInstanceIds,
        const std::vector<uint32_t>& fragmentSeedIds,
        const std::vector<glm::dvec3>& fragmentCentroids,
        const std::vector<double>& fragmentVolumes,
        const std::vector<uint32_t>& faceFragmentA,
        const std::vector<uint32_t>& faceFragmentB,
        const std::vector<double>& faceAreasA,
        const std::vector<double>& faceAreasB,
        const std::vector<glm::dvec3>& faceAreaVectorsA,
        const std::vector<glm::dvec3>& faceAreaVectorsB,
        const std::vector<glm::dvec3>& faceMomentsA,
        const std::vector<glm::dvec3>& faceMomentsB,
        const std::vector<uint8_t>& faceTypes,
        const std::vector<uint32_t>& faceDomainFaces,
        const std::vector<uint32_t>& boundaryFaceIndices,
        const std::vector<uint32_t>& faceLoopOffsets,
        const std::vector<uint32_t>& loopVertexOffsets,
        const std::vector<glm::dvec3>& loopVertices,
        const std::vector<uint32_t>& globalFragmentToFluidNode,
        const std::vector<uint32_t>& fluidNodeToGlobalFragment);

    void cleanup();

    const std::vector<FaceTopology>& faceTopology() const { return outFaceTopology; }
    const std::vector<FaceFluxVectors>& faceFluxVectors() const { return outFaceFluxVectors; }
    const std::vector<uint32_t>& faceGlobalIndex() const { return outFaceGlobalIndex; }
    const std::vector<CFDBoundaryFaceGpu>& boundaryFaces() const { return outBoundaryFaces; }
    const std::vector<uint32_t>& boundaryGlobalIndex() const { return outBoundaryGlobalIndex; }
    const std::vector<float>& cellVolumes() const { return outCellVolumes; }
    const std::vector<glm::vec4>& cellCentroids() const { return outCellCentroids; }
    const std::vector<uint32_t>& pressureComponents() const { return outPressureComponents; }
    uint32_t pressureComponentCount() const { return outPressureComponentCount; }
    void clearTopology();

    void releaseTopologyConstructionData();

    glm::vec4* velocityCuda() {
        return reinterpret_cast<glm::vec4*>(outVelocityMapping.getPointer());
    }

    float* faceFluxCuda() {
        return outFaceFluxMapping.getPointer();
    }

    float* boundaryFluxCuda() {
        return outBoundaryFluxMapping.getPointer();
    }

    VkBuffer velocityBuffer() const { return outVelocity.getBuffer(); }
    VkBuffer faceFluxBuffer() const { return outFaceFlux.getBuffer(); }
    VkBuffer boundaryFluxBuffer() const { return outBoundaryFlux.getBuffer(); }
    VkBuffer velocityGradientsBuffer() const { return outVelocityGradients.getBuffer(); }

    glm::vec4* velocityGradientsCuda() {
        return reinterpret_cast<glm::vec4*>(outVelocityGradientsMapping.getPointer());
    }

    const VulkanExternalBuffer& getVelocityExternalBuffer() const { return outVelocity; }
    const VulkanExternalBuffer& getFaceFluxExternalBuffer() const { return outFaceFlux; }
    const VulkanExternalBuffer& getBoundaryFluxExternalBuffer() const { return outBoundaryFlux; }
    const VulkanExternalBuffer& getVelocityGradientsExternalBuffer() const { return outVelocityGradients; }

    uint32_t getFluidNodeCount() const { return inFluidNodeCount; }
    uint32_t getGlobalFaceCount() const { return inGlobalFaceCount; }
    uint32_t getGlobalBoundaryFaceCount() const { return inGlobalBoundaryFaceCount; }

private:
    VulkanExternalBuffer outVelocity;
    CudaExternalBuffer outVelocityMapping;

    VulkanExternalBuffer outVelocityGradients;
    CudaExternalBuffer outVelocityGradientsMapping;

    VulkanExternalBuffer outFaceFlux;
    CudaExternalBuffer outFaceFluxMapping;

    VulkanExternalBuffer outBoundaryFlux;
    CudaExternalBuffer outBoundaryFluxMapping;

    uint32_t inFluidNodeCount = 0;
    uint32_t inGlobalFaceCount = 0;
    uint32_t inGlobalBoundaryFaceCount = 0;

    std::vector<FaceTopology> outFaceTopology;
    std::vector<FaceFluxVectors> outFaceFluxVectors;
    std::vector<uint32_t> outFaceGlobalIndex;
    std::vector<CFDBoundaryFaceGpu> outBoundaryFaces;
    std::vector<uint32_t> outBoundaryGlobalIndex;
    std::vector<float> outCellVolumes;
    std::vector<glm::vec4> outCellCentroids;
    std::vector<uint32_t> outPressureComponents;
    uint32_t outPressureComponentCount = 0;
};

} // namespace cfd
