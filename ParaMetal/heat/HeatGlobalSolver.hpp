#pragma once

#include "HeatGpuStructs.hpp"
#include "../vem/VEMDiffusion.hpp"

#include <cstdint>
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

class VulkanDevice;
class VulkanExternalBuffer;
class CudaExternalBuffer;

class HeatGlobalSolver {
public:
    struct TemperatureNodes {
        std::vector<uint32_t> gatherOffsets;
        std::vector<uint32_t> gatherLocalNodeIds;
        std::vector<float> gatherWeights;
        std::vector<uint32_t> gatherSolverNodeIds;
        std::vector<uint32_t> scatterLocalNodeIds;
        std::vector<uint32_t> scatterSolverNodeIds;
        const VulkanExternalBuffer* externalA = nullptr;
        const VulkanExternalBuffer* externalB = nullptr;
        CudaExternalBuffer* temperatureA = nullptr;
        CudaExternalBuffer* temperatureB = nullptr;
    };

    HeatGlobalSolver();
    ~HeatGlobalSolver();

    HeatGlobalSolver(const HeatGlobalSolver&) = delete;
    HeatGlobalSolver& operator=(const HeatGlobalSolver&) = delete;

    bool initialize(
        VulkanDevice& vulkanDevice,
        const vem::VEMTraceSystem& traceSystem,
        const std::vector<float>& cellCapacityOverDt,
        const std::vector<float>& cellSourceAdditions,
        const std::vector<TemperatureNodes>& temperatureNodes,
        const std::vector<float>& fixedCellTemperatures = {});

    enum class SolvePollResult {
        Pending,
        Complete,
        Failed
    };

    bool launch(
        bool sourceIsA,
        uint64_t destinationVulkanReleaseValue,
        uint64_t cudaReadyValue);

    SolvePollResult poll() const;
    void waitIdle();

    bool updateValues(
        const vem::VEMTraceSystem& traceSystem,
        const std::vector<float>& cellCapacityOverDt,
        const std::vector<float>& cellSourceAdditions,
        const std::vector<float>& fixedCellTemperatures = {});

    bool configureAdvectionTopology(
        uint32_t faceCount,
        uint32_t boundaryFaceCount,
        const std::vector<heat::AdvectionFace>& faces,
        const std::vector<heat::AdvectionBoundaryFace>& boundaryFaces,
        const float* deviceFaceFlux,
        const float* deviceBoundaryFlux,
        const std::vector<float>& cellVolumes);

    bool updateAdvectionStep(
        const float* inflowTemperaturesC,
        float stepDt);

    VkSemaphore getCudaReadySemaphore() const;
    VkSemaphore getVulkanReleaseSemaphore() const;
    bool hasSolveFailed() const;
    void setSolverParameters(uint32_t maxIterations, float convergenceTolerance);
    float getLastSolveMs() const;

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation;
};
