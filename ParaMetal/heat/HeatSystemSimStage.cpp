#include "HeatSystemSimStage.hpp"

#include "heat/HeatModelRuntime.hpp"
#include "heat/HeatSystemPlayback.hpp"
#include "HeatSystemDiffusionStage.hpp"
#include "contact/ContactSystemComputeStage.hpp"
#include "heat/HeatContactRuntime.hpp"

void HeatSystemSimStage::recordComputeCommands(
    VkCommandBuffer commandBuffer,
    uint32_t currentFrame,
    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
    const HeatSystemDiffusionStage& diffusionStage,
    const ContactSystemComputeStage& contactStage,
    const std::vector<std::unique_ptr<HeatContactRuntime>>& contactRuntimes,
    bool steppingPhysics,
    bool captureFrame,
    uint32_t numSubsteps) const {
    if (!steppingPhysics || activeModels.empty() || numSubsteps == 0) return;

    recordSim(
        commandBuffer,
        currentFrame,
        activeModels,
        diffusionStage,
        contactStage,
        contactRuntimes,
        numSubsteps);

    if (captureFrame) {
        recordHistoryCapture(commandBuffer, activeModels, diffusionStage, numSubsteps);
    }
}

void HeatSystemSimStage::recordSim(
    VkCommandBuffer commandBuffer,
    uint32_t currentFrame,
    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
    const HeatSystemDiffusionStage& diffusionStage,
    const ContactSystemComputeStage& contactStage,
    const std::vector<std::unique_ptr<HeatContactRuntime>>& contactRuntimes,
    uint32_t numSubsteps) const {
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
                diffusionStage.dispatchDiffusionSubstep(commandBuffer, modelSet, pushConstant, (modelPtr->getSimNodeCount() + workGroupSize - 1) / workGroupSize);
            }
        }

        if (substepIndex + 1 < numSubsteps) {
            std::vector<VkBufferMemoryBarrier> substepBarriers;
            substepBarriers.reserve(activeModels.size() * (contactRuntimes.empty() ? 1u : 2u));

            for (const auto& [modelId, modelPtr] : activeModels) {
                if (!modelPtr || modelPtr->getSimNodeCount() == 0) continue;

                VkBufferMemoryBarrier tempBarrier{};
                tempBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                tempBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                tempBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                tempBarrier.buffer = isEven ? modelPtr->getTempBufferB() : modelPtr->getTempBufferA();
                tempBarrier.offset = isEven ? modelPtr->getTempBufferBOffset() : modelPtr->getTempBufferAOffset();
                tempBarrier.size = modelPtr->getSimNodeCount() * sizeof(float);
                substepBarriers.push_back(tempBarrier);

                if (!contactRuntimes.empty() && modelPtr->getContactAccumulatorBuffer() != VK_NULL_HANDLE) {
                    VkBufferMemoryBarrier contactBarrier{};
                    contactBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    contactBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    contactBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    contactBarrier.buffer = modelPtr->getContactAccumulatorBuffer();
                    contactBarrier.offset = modelPtr->getContactAccumulatorBufferOffset();
                    contactBarrier.size = modelPtr->getSimNodeCount() * sizeof(float) * 2;
                    substepBarriers.push_back(contactBarrier);
                }
            }

            if (!substepBarriers.empty()) {
                vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    0,
                    nullptr,
                    static_cast<uint32_t>(substepBarriers.size()),
                    substepBarriers.data(),
                    0,
                    nullptr);
            }
        }
    }

    const bool finalWritesBufferB = diffusionStage.finalSubstepWritesBufferB(numSubsteps);
    for (const auto& [modelId, heatModel] : activeModels) {
        if (!heatModel || heatModel->getSimNodeCount() == 0) continue;

        diffusionStage.insertFinalTemperatureBarrier(commandBuffer, numSubsteps, heatModel->getTempBufferA(), heatModel->getTempBufferAOffset(), heatModel->getTempBufferB(), heatModel->getTempBufferBOffset(), heatModel->getSimNodeCount() * sizeof(float));

    }

}

void HeatSystemSimStage::recordHistoryCapture(
    VkCommandBuffer commandBuffer,
    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& activeModels,
    const HeatSystemDiffusionStage& diffusionStage,
    uint32_t numSubsteps) const {
    const bool finalWritesBufferB = diffusionStage.finalSubstepWritesBufferB(numSubsteps);

    std::vector<VkBufferMemoryBarrier> snapshotBarriers;
    for (const auto& [modelId, heatModel] : activeModels) {
        if (!heatModel || heatModel->getSimNodeCount() == 0) continue;
        auto* playback = heatModel->getPlayback();
        if (!playback) continue;

        VkBuffer finalBuf = finalWritesBufferB ? heatModel->getTempBufferB() : heatModel->getTempBufferA();
        VkDeviceSize finalOff = finalWritesBufferB ? heatModel->getTempBufferBOffset() : heatModel->getTempBufferAOffset();

        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.buffer = finalBuf;
        barrier.offset = finalOff;
        barrier.size = heatModel->getSimNodeCount() * sizeof(float);
        snapshotBarriers.push_back(barrier);
    }
    if (!snapshotBarriers.empty()) {
        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            static_cast<uint32_t>(snapshotBarriers.size()), snapshotBarriers.data(),
            0, nullptr);
    }

    for (const auto& [modelId, heatModel] : activeModels) {
        if (!heatModel || heatModel->getSimNodeCount() == 0) continue;
        auto* playback = heatModel->getPlayback();
        if (!playback) continue;

        VkBuffer finalBuf = finalWritesBufferB ? heatModel->getTempBufferB() : heatModel->getTempBufferA();
        VkDeviceSize finalOff = finalWritesBufferB ? heatModel->getTempBufferBOffset() : heatModel->getTempBufferAOffset();
        playback->recordFrame(commandBuffer, finalBuf, finalOff);
    }
}
