#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include "../voronoi/GlobalDomainTopology.hpp"

class VulkanDevice;

namespace heat {

// Fluid-fluid internal face 
struct HeatFluidFace {
    uint32_t faceIndex = 0;
    uint32_t fluidNodeA = 0;
    uint32_t fluidNodeB = 0;
    glm::vec3 normal{0.0f};
    float area = 0.0f;
    float projectionWeight = 0.0f;
};

// Fluid domain boundary face 
struct HeatFluidBoundaryFace {
    uint32_t boundaryFaceIndex = 0;
    uint32_t fluidNode = 0;
    glm::vec3 outwardNormal{0.0f};
    float area = 0.0f;
};

struct FluidTimeStep {
    uint32_t subcycleCount = 1;
    float subcycleDt = 0.0f;
    float maxCourant = 0.0f;
};

class HeatFluidFlow {
public:
    HeatFluidFlow();
    ~HeatFluidFlow();

    static glm::vec3 getDomainBoxNormal(uint32_t domainFace) {
        switch (domainFace) {
            case 0: return glm::vec3(-1.0f, 0.0f, 0.0f);
            case 1: return glm::vec3(+1.0f, 0.0f, 0.0f);
            case 2: return glm::vec3(0.0f, -1.0f, 0.0f);
            case 3: return glm::vec3(0.0f, +1.0f, 0.0f);
            case 4: return glm::vec3(0.0f, 0.0f, -1.0f);
            case 5: return glm::vec3(0.0f, 0.0f, +1.0f);
            default: return glm::vec3(0.0f);
        }
    }

    bool initialize(VulkanDevice& vulkanDevice);

    bool buildTopology(
        const std::vector<topology::Fragment>& fragments,
        const std::vector<topology::Face>& faces,
        const std::vector<topology::BoundaryFace>& boundaryFaces,
        const std::vector<uint32_t>& globalFragmentToFluidNode,
        const std::vector<uint32_t>& fluidNodeToGlobalFragment);

    bool updatePrescribedFlow(
        const glm::vec3& prescribedVelocity,
        float macroDt,
        float targetCourant = 0.5f);

    bool updateTimeStep(float macroDt, float targetCourant, float maxOutflowRate);

    const std::vector<HeatFluidFace>& getFaces() const { return fluidFaces; }
    const std::vector<HeatFluidBoundaryFace>& getBoundaryFaces() const { return fluidBoundaryFaces; }
    const FluidTimeStep& getTimeStep() const { return timeStep; }

    const float* getDeviceFaceFlux() const { return d_faceFlux; }
    const float* getDeviceBoundaryFlux() const { return d_boundaryFlux; }

    void cleanup();

private:
    void clearTopology();
    void releaseTopologyBuffers();

    int cudaDevice = -1;
    void* flowStreamHandle = nullptr; 
    bool initialized = false;

    std::vector<HeatFluidFace> fluidFaces;
    std::vector<HeatFluidBoundaryFace> fluidBoundaryFaces;

    std::vector<uint32_t> nodeComponent;
    uint32_t componentCount = 0;

    FluidTimeStep timeStep;

    float* d_faceFlux = nullptr;
    float* d_boundaryFlux = nullptr;
    float* d_faceRawFlux = nullptr;
    float* d_divergence = nullptr;
    float* d_psi = nullptr;

    void* d_faces = nullptr;           
    void* d_boundaryFaces = nullptr;  

    int* d_rowOffsets = nullptr;
    int* d_columnIndices = nullptr;
    float* d_values = nullptr;
    float* d_invDiag = nullptr;
    uint8_t* d_isPinned = nullptr;

    float* d_p = nullptr;
    float* d_Ap = nullptr;
    float* d_r = nullptr;
    float* d_z = nullptr;
    void* d_status = nullptr;

    float* d_fluidVolumes = nullptr;
    float* d_maxOutflowRate = nullptr;
    void* d_cubScratch = nullptr;
    size_t cubScratchBytes = 0;          

    uint32_t nodeCount = 0;
    uint32_t faceCount = 0;
    uint32_t boundaryFaceCount = 0;
};

} // namespace heat
