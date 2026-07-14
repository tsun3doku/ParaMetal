#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>

class VulkanDevice;
class CommandPool;

class VoronoiGeoCompute {
public:
    struct PushConstants {
        uint32_t debugEnable = 0;
        uint32_t nodeOffset = 0;
        uint32_t nodeCount = 0;
        uint32_t _padding = 0;
    };

    struct Bindings {
        VkBuffer candidateNodeBuffer = VK_NULL_HANDLE;
        VkDeviceSize candidateNodeBufferOffset = 0;
        VkDeviceSize candidateNodeBufferRange = 0;

        VkBuffer meshTriangleBuffer = VK_NULL_HANDLE;
        VkDeviceSize meshTriangleBufferOffset = 0;

        VkBuffer seedPositionBuffer = VK_NULL_HANDLE;
        VkDeviceSize seedPositionBufferOffset = 0;

        VkBuffer voxelGridParamsBuffer = VK_NULL_HANDLE;
        VkDeviceSize voxelGridParamsBufferOffset = 0;
        VkDeviceSize voxelGridParamsBufferRange = 0;

        VkBuffer voxelOccupancyBuffer = VK_NULL_HANDLE;
        VkDeviceSize voxelOccupancyBufferOffset = 0;

        VkBuffer voxelTrianglesListBuffer = VK_NULL_HANDLE;
        VkDeviceSize voxelTrianglesListBufferOffset = 0;

        VkBuffer voxelOffsetsBuffer = VK_NULL_HANDLE;
        VkDeviceSize voxelOffsetsBufferOffset = 0;

        VkBuffer candidateNeighborIndicesBuffer = VK_NULL_HANDLE;
        VkDeviceSize candidateNeighborIndicesBufferOffset = 0;

        VkBuffer candidateInterfaceAreasBuffer = VK_NULL_HANDLE;
        VkDeviceSize candidateInterfaceAreasBufferOffset = 0;

        VkBuffer candidateInterfaceNeighborIdsBuffer = VK_NULL_HANDLE;
        VkDeviceSize candidateInterfaceNeighborIdsBufferOffset = 0;

        VkBuffer debugCellGeometryBuffer = VK_NULL_HANDLE;
        VkDeviceSize debugCellGeometryBufferOffset = 0;

        VkBuffer nodeFlagsBuffer = VK_NULL_HANDLE;
        VkDeviceSize nodeFlagsBufferOffset = 0;

        VkBuffer voronoiDumpBuffer = VK_NULL_HANDLE;
        VkDeviceSize voronoiDumpBufferOffset = 0;

        VkBuffer surfacePatchAreasBuffer = VK_NULL_HANDLE;
        VkDeviceSize surfacePatchAreasBufferOffset = 0;
    };

    VoronoiGeoCompute(VulkanDevice& vulkanDevice, CommandPool& commandPool);
    ~VoronoiGeoCompute();

    void initialize(uint32_t nodeCount);
    void updateDescriptors(const Bindings& bindings);
    bool dispatch(const PushConstants& pushConstants);

    void cleanupResources();
    void cleanup();

private:
    bool createDescriptorSetLayout();
    bool createDescriptorPool();
    bool createDescriptorSet();
    bool createPipeline();

    VulkanDevice& vulkanDevice;
    CommandPool& commandPool;

    uint32_t nodeCount = 0;
    bool initialized = false;

    Bindings currentBindings;

    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};
