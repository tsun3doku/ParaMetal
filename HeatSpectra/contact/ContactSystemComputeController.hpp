#pragma once

#include "contact/ContactTypes.hpp"
#include "mesh/remesher/SupportingHalfedge.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "vulkan/CommandBufferManager.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

class MemoryAllocator;
class VulkanDevice;
class ContactSystem;

class ContactSystemComputeController {
public:
    struct Config {
        float minNormalDot = -0.65f;
        float contactRadius = 0.01f;
        uint32_t modelAId = 0;
        std::array<float, 16> modelALocalToWorld{
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
        SupportingHalfedge::IntrinsicMesh modelAIntrinsicMesh;
        uint32_t modelBId = 0;
        std::array<float, 16> modelBLocalToWorld{
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
        SupportingHalfedge::IntrinsicMesh modelBIntrinsicMesh;
        uint32_t modelARuntimeModelId = 0;
        uint32_t modelBRuntimeModelId = 0;
        std::vector<uint32_t> modelBTriangleIndices;
        uint64_t computeHash = 0;

        bool isValid() const {
            return modelARuntimeModelId != 0 &&
                modelBRuntimeModelId != 0 &&
                modelARuntimeModelId != modelBRuntimeModelId &&
                modelAId != 0 &&
                modelBId != 0 &&
                !modelAIntrinsicMesh.vertices.empty() &&
                !modelBIntrinsicMesh.vertices.empty() &&
                !modelBTriangleIndices.empty();
        }
    };

    ContactSystemComputeController(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& renderCommandPool);
    ~ContactSystemComputeController();

    void configure(uint64_t socketKey, const Config& config);
    void disable(uint64_t socketKey);
    void disableAll();
    bool exportProduct(uint64_t socketKey, ContactProduct& outProduct);

private:
    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& renderCommandPool;
    std::unordered_map<uint64_t, std::unique_ptr<ContactSystem>> activeSystems;
    std::unordered_map<uint64_t, Config> configuredConfigs;
};

inline uint64_t buildComputeHash(const ContactSystemComputeController::Config& config) {
    uint64_t hash = 1469598103934665603ull;
    hash = RuntimeProductHash::mixPod(hash, config.minNormalDot);
    hash = RuntimeProductHash::mixPod(hash, config.contactRadius);
    hash = RuntimeProductHash::mixPod(hash, config.modelAId);
    hash = RuntimeProductHash::mixPod(hash, config.modelALocalToWorld);
    hash = RuntimeProductHash::mixPodVector(hash, config.modelAIntrinsicMesh.vertices);
    hash = RuntimeProductHash::mixPodVector(hash, config.modelAIntrinsicMesh.indices);
    hash = RuntimeProductHash::mixPodVector(hash, config.modelAIntrinsicMesh.faceIds);
    hash = RuntimeProductHash::mixPodVector(hash, config.modelAIntrinsicMesh.triangles);
    hash = RuntimeProductHash::mixPod(hash, config.modelBId);
    hash = RuntimeProductHash::mixPod(hash, config.modelBLocalToWorld);
    hash = RuntimeProductHash::mixPodVector(hash, config.modelBIntrinsicMesh.vertices);
    hash = RuntimeProductHash::mixPodVector(hash, config.modelBIntrinsicMesh.indices);
    hash = RuntimeProductHash::mixPodVector(hash, config.modelBIntrinsicMesh.faceIds);
    hash = RuntimeProductHash::mixPodVector(hash, config.modelBIntrinsicMesh.triangles);
    hash = RuntimeProductHash::mixPod(hash, config.modelARuntimeModelId);
    hash = RuntimeProductHash::mixPod(hash, config.modelBRuntimeModelId);
    hash = RuntimeProductHash::mixPodVector(hash, config.modelBTriangleIndices);
    return hash;
}