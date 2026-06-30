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
        std::array<float, 16> modelALocalToWorld{
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
        SupportingHalfedge::IntrinsicMesh modelAIntrinsicMesh;
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
                !modelAIntrinsicMesh.vertices.empty() &&
                !modelBIntrinsicMesh.vertices.empty() &&
                !modelBTriangleIndices.empty();
        }
    };

    ContactSystemComputeController(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        CommandPool& commandPool);
    ~ContactSystemComputeController();

    void apply(uint64_t socketKey, const Config& config);
    void retain(uint64_t socketKey);
    bool buildProduct(uint64_t socketKey, ContactProduct& product) const;
    void remove(uint64_t socketKey);
    void disableAll();
    ContactSystem* getSystem(uint64_t socketKey) const;
    const Config* getConfig(uint64_t socketKey) const;

private:
    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;
    std::unordered_map<uint64_t, std::unique_ptr<ContactSystem>> systemsBySocket;
    std::unordered_map<uint64_t, Config> configuredConfigs;
};
