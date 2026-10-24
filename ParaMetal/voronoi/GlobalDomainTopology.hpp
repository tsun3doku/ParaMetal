#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan.h>

namespace topology {

inline constexpr uint32_t InvalidInstanceId = 0xffffffffu;

enum class FaceType : uint8_t {
    Shared,          // Internal Voronoi interface
    ContactSide,     // Solid contact face or fluid closure face 
    DomainBoundary   // Domain boundary face 
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
    uint32_t faceA = 0xffffffffu; // Index into GlobalDomain::faces
    uint32_t faceB = 0xffffffffu; // Index into GlobalDomain::faces
    double couplingArea = 0.0;
};

struct BoundaryFace {
    uint32_t faceIndex = 0xffffffffu;
    uint32_t fragment = 0xffffffffu;
    uint32_t domainFace = 0;
};

struct GlobalDomain {
    std::vector<Fragment> fragments;
    std::vector<Face> faces;
    std::vector<ContactCoupling> contactCouplings;
    std::vector<BoundaryFace> boundaryFaces;
    std::vector<uint32_t> instanceFragmentCounts;
    std::vector<glm::vec3> seedPositions;
    std::vector<uint32_t> seedToFragment;
    glm::vec3 volumeGridMin{0.0f, 0.0f, 0.0f};
    glm::ivec3 volumeGridDim{0, 0, 0};
    float volumeGridCellSize = 0.0f;
    float nominalCellSize = 0.0f;
    VkBuffer seedIdGridBuffer = VK_NULL_HANDLE;
    VkDeviceSize seedIdGridBufferOffset = 0;
    uint32_t seedIdGridCount = 0;

    bool isValid() const {
        return !fragments.empty() && !faces.empty() &&
            nominalCellSize > 0.0f &&
            volumeGridDim.x > 0 && volumeGridDim.y > 0 && volumeGridDim.z > 0 &&
            volumeGridCellSize > 0.0f;
    }
};

} // namespace topology
