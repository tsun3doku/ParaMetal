#include "HeatSystemSimStage.hpp"

#include "heat/HeatModelRuntime.hpp"
#include "HeatSystemResources.hpp"
#include "HeatSystemSurfaceStage.hpp"
#include "HeatSystemVoronoiStage.hpp"
#include "contact/ContactSystemComputeStage.hpp"
#include "heat/HeatContactRuntime.hpp"
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <iostream>
#include <limits>

HeatSystemSimStage::HeatSystemSimStage(const HeatSystemStageContext& stageContext)
    : context(stageContext) {
}

void HeatSystemSimStage::recordComputeCommands(
    VkCommandBuffer commandBuffer,
    uint32_t currentFrame,
    const HeatSystemSimRuntime& simRuntime,
    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
    const HeatSystemVoronoiStage& voronoiStage,
    const HeatSystemSurfaceStage& surfaceStage,
    const ContactSystemComputeStage& contactStage,
    const std::vector<std::unique_ptr<HeatContactRuntime>>& contactRuntimes,
    uint32_t maxNodeNeighbors,
    uint32_t numSubsteps) const {
    if (activeModels.empty() || numSubsteps == 0) return;

    const uint32_t workGroupSize = 256;

    for (uint32_t substepIndex = 0; substepIndex < numSubsteps; ++substepIndex) {
        const bool isEven = (substepIndex % 2 == 0);

        if (!contactRuntimes.empty()) {
            for (const auto& couplingPtr : contactRuntimes) {
                if (!couplingPtr) continue;
                const HeatContactRuntime& coupling = *couplingPtr;

                const VkDescriptorSet setA = isEven ? coupling.getSetAA() : coupling.getSetAB();
                const VkDescriptorSet setB = isEven ? coupling.getSetBA() : coupling.getSetBB();

                auto dispatch = [&](uint32_t id, VkDescriptorSet set) {
                    if (set == VK_NULL_HANDLE) return;
                    auto it = activeModels.find(id);
                    if (it != activeModels.end() && it->second) {
                        uint32_t n = it->second->getSimNodeCount();
                        contactStage.dispatchGather(commandBuffer, set, n);
                    }
                };

                dispatch(coupling.getModelARuntimeModelId(), setA);
                dispatch(coupling.getModelBRuntimeModelId(), setB);
            }

            // Barrier for coupling accumulator buffers
            std::vector<VkBufferMemoryBarrier> fluxBarriers;
            fluxBarriers.reserve(activeModels.size());
            for (const auto& [modelId, modelPtr] : activeModels) {
                if (!modelPtr || modelPtr->getContactAccumulatorBuffer() == VK_NULL_HANDLE) continue;
                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                barrier.buffer = modelPtr->getContactAccumulatorBuffer();
                barrier.offset = modelPtr->getContactAccumulatorBufferOffset();
                barrier.size = modelPtr->getSimNodeCount() * sizeof(float) * 2;
                fluxBarriers.push_back(barrier);
            }
            if (!fluxBarriers.empty()) {
                vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, static_cast<uint32_t>(fluxBarriers.size()), fluxBarriers.data(), 0, nullptr);
            }
        }

        for (const auto& [modelId, modelPtr] : activeModels) {
            if (!modelPtr || modelPtr->getSimNodeCount() == 0) continue;

            heat::HeatModelPushConstant pushConstant{modelPtr->getSimNodeCount()};
            VkDescriptorSet modelSet = isEven ? modelPtr->getVoronoiDescriptorSetA() : modelPtr->getVoronoiDescriptorSetB();

            if (modelSet != VK_NULL_HANDLE) {
                voronoiStage.dispatchDiffusionSubstep(commandBuffer, modelSet, pushConstant, (modelPtr->getSimNodeCount() + workGroupSize - 1) / workGroupSize);
            }
        }

        voronoiStage.insertInterSubstepBarrier(commandBuffer, static_cast<int>(substepIndex), numSubsteps);
    }

    const bool finalWritesBufferB = voronoiStage.finalSubstepWritesBufferB(numSubsteps);
    for (const auto& [modelId, heatModel] : activeModels) {
        if (!heatModel || heatModel->getSimNodeCount() == 0) continue;

        voronoiStage.insertFinalTemperatureBarrier(commandBuffer, numSubsteps, heatModel->getTempBufferA(), heatModel->getTempBufferAOffset(), heatModel->getTempBufferB(), heatModel->getTempBufferBOffset());

        if (heatModel->getBoundaryCondition() == 1u) {
            heatModel->injectFixedTemperature(finalWritesBufferB ? heatModel->getTempBufferB() : heatModel->getTempBufferA(), (finalWritesBufferB ? heatModel->getTempBufferBOffset() : heatModel->getTempBufferAOffset()), heatModel->getSimNodeCount(), heatModel->getFixedTemperatureValue());
        }
    }

    surfaceStage.dispatchSurfaceTemperatureUpdates(commandBuffer, activeModels, finalWritesBufferB);
    if (currentFrame % 4 == 0) {
        surfaceStage.dispatchSurfaceGradientUpdates(commandBuffer, activeModels, finalWritesBufferB);
    }
    
}
