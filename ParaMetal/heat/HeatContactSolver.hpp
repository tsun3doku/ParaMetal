#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

class VulkanDevice;
class VulkanExternalBuffer;
class CudaExternalBuffer;

class HeatContactSolver {
public:
    struct ModelNodes {
        uint32_t solverNodeOffset = 0;
        std::vector<uint32_t> localNodeIds;
        const VulkanExternalBuffer* externalA = nullptr;
        const VulkanExternalBuffer* externalB = nullptr;
        CudaExternalBuffer* temperatureA = nullptr;
        CudaExternalBuffer* temperatureB = nullptr;
    };

    struct FixedContribution {
        uint32_t boundaryValueIndex = 0;
        float coefficient = 0.0f;
    };

    struct FixedRow {
        uint32_t contributionOffset = 0;
        uint32_t contributionCount = 0;
    };

    HeatContactSolver();
    ~HeatContactSolver();

    HeatContactSolver(const HeatContactSolver&) = delete;
    HeatContactSolver& operator=(const HeatContactSolver&) = delete;

    bool initialize(
        VulkanDevice& vulkanDevice,
        const std::vector<int>& rowOffsets,
        const std::vector<int>& columnIndices,
        const std::vector<float>& values,
        const std::vector<float>& thermalMasses,
        const std::vector<ModelNodes>& models,
        const std::vector<FixedRow>& fixedRows,
        const std::vector<FixedContribution>& fixedContributions,
        uint32_t boundaryValueCount);

    bool solve(
        bool temperatureBufferAIsCurrent,
        const std::vector<float>& boundaryTemperaturesC,
        uint64_t waitValue,
        uint64_t signalValue);

    VkSemaphore getTimelineSemaphore() const;
    bool isInitialized() const;
    void cleanup();

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation;
};
