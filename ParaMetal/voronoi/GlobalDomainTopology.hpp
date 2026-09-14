#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan.h>

namespace topology {

inline constexpr uint32_t InvalidInstanceId = 0xffffffffu;

    // Thermal geometry is strictly fragment based
struct Fragment {
    uint32_t instanceId = InvalidInstanceId;
    uint32_t seedId = 0;
    glm::vec3 centroid{0.0f};
    float volume = 0.0f;
};

struct Face {
    uint32_t fragmentA = 0;
    uint32_t fragmentB = 0;
    // Consistently oriented fragmentA -> fragmentB
    glm::vec3 normal{0.0f};
    float area = 0.0f;
    float orthogonalDistanceA = 0.0f;
    float orthogonalDistanceB = 0.0f;
};

struct ContactFace {
    uint32_t fragmentA = 0;
    uint32_t fragmentB = 0;
    float area = 0.0f;
    float orthogonalDistanceA = 0.0f;
    float orthogonalDistanceB = 0.0f;
    float gap = 0.0f;
};

struct BoundaryFace {
    uint32_t fragment = 0;
    uint32_t domainFace = 0;
    float area = 0.0f;
    // Orthogonal distance from fragment centroid to the domain boundary plane
    float orthogonalDistance = 0.0f;
};

struct GlobalDomain {
    std::vector<Fragment> fragments;
    std::vector<Face> faces;
    std::vector<ContactFace> contactFaces;
    std::vector<BoundaryFace> boundaryFaces;
    std::vector<uint32_t> instanceFragmentCounts;
    std::vector<uint32_t> sdfRuntimeModelIds;
    std::vector<glm::vec3> seedPositions;
    glm::vec3 sdfGridMin{0.0f, 0.0f, 0.0f};
    glm::ivec3 sdfGridDim{0, 0, 0};
    float sdfCellSize = 0.0f;
    float nominalCellSize = 0.0f;
    std::vector<float> sdfValues;
    VkBuffer seedIdGridBuffer = VK_NULL_HANDLE;
    VkDeviceSize seedIdGridBufferOffset = 0;
    uint32_t seedIdGridCount = 0;

    bool isValid() const {
        return !fragments.empty() && !faces.empty() &&
            nominalCellSize > 0.0f &&
            sdfGridDim.x > 0 && sdfGridDim.y > 0 && sdfGridDim.z > 0 && sdfCellSize > 0.0f &&
            sdfValues.size() ==
                static_cast<size_t>(sdfGridDim.x) * static_cast<size_t>(sdfGridDim.y) *
                    static_cast<size_t>(sdfGridDim.z) * sdfRuntimeModelIds.size();
    }
};

} // namespace topology
