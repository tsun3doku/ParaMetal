#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan.h>

namespace topology {

inline constexpr uint32_t InvalidInstanceId = 0xffffffffu;
inline constexpr uint32_t DomainFaceCount = 6;

enum class FaceType : uint8_t {
    Shared = 0,          // Two-sided physical material interface
    DomainBoundary = 1,  // Domain boundary face
    MaterialBoundary = 2 // One-sided solid surface facing contact / exterior
};

// Thermal geometry is strictly fragment based
struct Fragment {
    uint32_t instanceId = InvalidInstanceId;
    uint32_t seedId = 0;
    glm::dvec3 centroid{0.0};
    double volume = 0.0;
};

struct Face {
    uint32_t fragmentA = 0xffffffffu;
    uint32_t fragmentB = 0xffffffffu;
    double areaA = 0.0;
    glm::dvec3 areaVectorA{0.0}; // Outward from fragmentA
    glm::dvec3 momentA{0.0};     
    double areaB = 0.0;
    glm::dvec3 areaVectorB{0.0}; // Outward from fragmentB
    glm::dvec3 momentB{0.0};     
    FaceType type = FaceType::Shared;
    uint32_t domainFace = 0xffffffffu;
};

struct ContactCoupling {
    uint32_t faceA = 0xffffffffu;
    uint32_t fragmentA = 0xffffffffu;
    uint32_t faceB = 0xffffffffu;
    uint32_t fragmentB = 0xffffffffu;
    double couplingArea = 0.0;
};

struct GlobalDomain {
    std::vector<Fragment> fragments;
    std::vector<Face> faces;
    std::vector<uint32_t> faceLoopOffsets;
    std::vector<uint32_t> loopVertexOffsets;
    std::vector<glm::dvec3> loopVertices;
    std::vector<ContactCoupling> contactCouplings;
    std::vector<uint32_t> boundaryFaceIndices;
    std::vector<uint32_t> instanceFragmentCounts;
    std::vector<glm::vec3> seedPositions;
    std::vector<uint32_t> cellFragmentOffsets;
    std::vector<uint32_t> cellFragments;
    glm::vec3 domainMin{0.0f, 0.0f, 0.0f}; // AABB of the authored domain corners
    glm::vec3 domainMax{0.0f, 0.0f, 0.0f};
    glm::vec3 volumeGridMin{0.0f, 0.0f, 0.0f};
    glm::ivec3 volumeGridDim{0, 0, 0};
    float volumeGridCellSize = 0.0f;
    VkBuffer seedIdGridBuffer = VK_NULL_HANDLE;
    VkDeviceSize seedIdGridBufferOffset = 0;

    bool isValid() const {
        return !fragments.empty() && !faces.empty() &&
            faceLoopOffsets.size() == faces.size() + 1 &&
            !faceLoopOffsets.empty() &&
            loopVertexOffsets.size() == faceLoopOffsets.back() + 1 &&
            !loopVertexOffsets.empty() &&
            loopVertices.size() == loopVertexOffsets.back() &&
            cellFragmentOffsets.size() == seedPositions.size() + 1 &&
            (!cellFragmentOffsets.empty() &&
             cellFragmentOffsets.back() == cellFragments.size()) &&
            std::isfinite(domainMin.x) && std::isfinite(domainMin.y) && std::isfinite(domainMin.z) &&
            std::isfinite(domainMax.x) && std::isfinite(domainMax.y) && std::isfinite(domainMax.z) &&
            domainMax.x > domainMin.x && domainMax.y > domainMin.y && domainMax.z > domainMin.z &&
            volumeGridDim.x > 0 && volumeGridDim.y > 0 && volumeGridDim.z > 0 &&
            volumeGridCellSize > 0.0f;
    }
};

bool validateBoundaryFaceIndices(const GlobalDomain& domain);

} // namespace topology
