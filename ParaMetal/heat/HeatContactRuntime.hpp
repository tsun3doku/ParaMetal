#pragma once

#include "contact/ContactGpuStructs.hpp"
#include "contact/ContactTypes.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

class VulkanDevice;
class MemoryAllocator;
class HeatModelRuntime;
class ContactSystemComputeStage;
struct StencilKDTree;

class HeatContactRuntime {
public:
    HeatContactRuntime() = default;
    ~HeatContactRuntime();

    HeatContactRuntime(const HeatContactRuntime&) = delete;
    HeatContactRuntime& operator=(const HeatContactRuntime&) = delete;

    uint32_t getModelARuntimeModelId() const { return modelARuntimeModelId; }
    uint32_t getModelBRuntimeModelId() const { return modelBRuntimeModelId; }

    VkDescriptorSet getSetAA() const { return setAA; }
    VkDescriptorSet getSetAB() const { return setAB; }
    VkDescriptorSet getSetBA() const { return setBA; }
    VkDescriptorSet getSetBB() const { return setBB; }

    bool build(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        const ContactCoupling& coupling,
        const HeatModelRuntime& modelA,
        const HeatModelRuntime& modelB,
        float contactThermalConductance);

    bool createDescriptorSets(
        const ContactSystemComputeStage& contactStage,
        const HeatModelRuntime& modelA,
        const HeatModelRuntime& modelB);

    void cleanup(MemoryAllocator& memoryAllocator);

private:
    struct BakedContact {
        float totalConductance = 0.0f;
        std::unordered_map<uint32_t, float> neighborWeights;
    };

    static bool buildPointStencil(
        const glm::vec3& point,
        const StencilKDTree& kdTree,
        std::vector<contact::ContactSampleWeight>& valueWeightsOut,
        uint32_t& valueWeightOffsetOut,
        uint32_t& valueWeightCountOut,
        std::vector<contact::ContactSampleWeight>& scatterWeightsOut,
        uint32_t& scatterWeightOffsetOut,
        uint32_t& scatterWeightCountOut);

    static void remapWeightsToSimNodes(
        const HeatModelRuntime& model,
        std::vector<contact::ContactSampleWeight>& weights);

    static void thresholdContactEdges(
        const std::vector<BakedContact>& baked,
        uint32_t nodeCount,
        std::vector<contact::ContactSampleWeight>& outEdges,
        std::vector<contact::ContactIndex>& outIndex);

    uint32_t modelARuntimeModelId = 0;
    uint32_t modelBRuntimeModelId = 0;

    VkBuffer edgesAToB = VK_NULL_HANDLE;
    VkDeviceSize edgesAToBOffset = 0;
    uint32_t edgeCountAToB = 0;
    VkBuffer edgeIndexAToB = VK_NULL_HANDLE;
    VkDeviceSize edgeIndexAToBOffset = 0;

    VkBuffer edgesBToA = VK_NULL_HANDLE;
    VkDeviceSize edgesBToAOffset = 0;
    uint32_t edgeCountBToA = 0;
    VkBuffer edgeIndexBToA = VK_NULL_HANDLE;
    VkDeviceSize edgeIndexBToAOffset = 0;

    VkDescriptorSet setAA = VK_NULL_HANDLE;
    VkDescriptorSet setAB = VK_NULL_HANDLE;
    VkDescriptorSet setBA = VK_NULL_HANDLE;
    VkDescriptorSet setBB = VK_NULL_HANDLE;
};
