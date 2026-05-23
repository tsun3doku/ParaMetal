#pragma once

#include "contact/ContactTypes.hpp"

#include <cstdint>

#include <vulkan/vulkan.h>

class VulkanDevice;
class MemoryAllocator;
class HeatModelRuntime;
class ContactSystemComputeStage;

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
