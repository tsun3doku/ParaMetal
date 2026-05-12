#include "HeatOverlayRenderer.hpp"

#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "domain/HeatModelData.hpp"

#include <iostream>

namespace render {

HeatOverlayRenderer::HeatOverlayRenderer(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    UniformBufferManager& uniformBufferManager)
    : memoryAllocator(allocator),
      surfaceRenderer(std::make_unique<HeatSurfaceRenderer>(device, uniformBufferManager)),
      vectorArrowRenderer(std::make_unique<VectorArrowRenderer>(device, allocator, uniformBufferManager)) {
}

HeatOverlayRenderer::~HeatOverlayRenderer() {
    cleanup();
}

void HeatOverlayRenderer::initialize(VkRenderPass renderPass, uint32_t updatedMaxFramesInFlight) {
    if (!surfaceRenderer || !vectorArrowRenderer || initialized) {
        return;
    }

    maxFramesInFlight = updatedMaxFramesInFlight;
    surfaceRenderer->initialize(renderPass, maxFramesInFlight);
    vectorArrowRenderer->initialize(renderPass, maxFramesInFlight);
    initialized = true;
}

void HeatOverlayRenderer::apply(uint64_t socketKey, const HeatDisplayController::Config& config) {
    if (socketKey == 0 || !config.anyVisible() || !config.isValid()) {
        remove(socketKey);
        return;
    }

    configsBySocket[socketKey] = config;
    rebuildBindings();
}

void HeatOverlayRenderer::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    configsBySocket.erase(socketKey);
    rebuildBindings();
}

void HeatOverlayRenderer::rebuildBindings() {
    if (!initialized || !surfaceRenderer || !vectorArrowRenderer) {
        return;
    }

    surfaceBindings.clear();
    fluxVectorBindings.clear();

    for (const auto& [socketKey, config] : configsBySocket) {
        (void)socketKey;
        if (!config.active && !config.paused) {
            continue;
        }

        if (config.showHeatOverlay) {
            surfaceBindings.reserve(surfaceBindings.size() + config.models.size());
            for (size_t index = 0; index < config.models.size(); ++index) {
                const auto& modelProduct = config.models[index];
                if (!modelProduct.isValid()) {
                    continue;
                }

                HeatSurfaceRenderer::SurfaceRenderBinding binding{};
                binding.runtimeModelId = modelProduct.runtimeModelId;
                binding.vertexBuffer = modelProduct.renderVertexBuffer;
                binding.vertexBufferOffset = modelProduct.renderVertexBufferOffset;
                binding.indexBuffer = modelProduct.renderIndexBuffer;
                binding.indexBufferOffset = modelProduct.renderIndexBufferOffset;
                binding.indexCount = modelProduct.renderIndexCount;
                binding.modelMatrix = modelProduct.modelMatrix;
                binding.bufferViews = config.modelBufferViews[index];
                binding.surfaceBuffer = config.modelSurfaceBuffers[index];
                binding.surfaceBufferOffset = config.modelSurfaceBufferOffsets[index];
                surfaceBindings.push_back(binding);
            }
        }

        if (config.showFluxVectors) {
            for (size_t index = 0; index < config.models.size(); ++index) {
                if (config.models[index].isValid() &&
                    index < config.modelSurfaceBuffers.size() &&
                    index < config.modelSurfaceBufferOffsets.size() &&
                    index < config.modelSurfaceGradientBuffers.size() &&
                    index < config.modelSurfaceGradientBufferOffsets.size() &&
                    index < config.modelSurfacePointCounts.size() &&
                    config.modelSurfaceGradientBuffers[index] != VK_NULL_HANDLE &&
                    config.modelSurfacePointCounts[index] != 0) {
                    VectorArrowRenderer::VectorRenderBinding vectorBinding{};
                    vectorBinding.bindingKey = config.models[index].runtimeModelId;
                    vectorBinding.surfaceBuffer = config.modelSurfaceBuffers[index];
                    vectorBinding.surfaceBufferOffset = config.modelSurfaceBufferOffsets[index];
                    vectorBinding.gradientBuffer = config.modelSurfaceGradientBuffers[index];
                    vectorBinding.gradientBufferOffset = config.modelSurfaceGradientBufferOffsets[index];
                    vectorBinding.sampleCount = config.modelSurfacePointCounts[index];
                    vectorBinding.modelMatrix = config.models[index].modelMatrix;
                    vectorBinding.scale = config.fluxVectorScale;
                    fluxVectorBindings.push_back(vectorBinding);
                }
            }
        }
    }

    surfaceRenderer->updateDescriptors(surfaceBindings, maxFramesInFlight, false);
    vectorArrowRenderer->updateDescriptors(fluxVectorBindings, maxFramesInFlight, false);
}

void HeatOverlayRenderer::render(VkCommandBuffer commandBuffer, uint32_t frameIndex) {
    if (!initialized || !surfaceRenderer || !vectorArrowRenderer) {
        return;
    }

    surfaceRenderer->render(commandBuffer, frameIndex, surfaceBindings);
    vectorArrowRenderer->render(commandBuffer, frameIndex, fluxVectorBindings);
}

void HeatOverlayRenderer::cleanup() {
    configsBySocket.clear();
    surfaceBindings.clear();
    fluxVectorBindings.clear();
    maxFramesInFlight = 0;
    initialized = false;
    if (surfaceRenderer) {
        surfaceRenderer->cleanup();
    }
    if (vectorArrowRenderer) {
        vectorArrowRenderer->cleanup();
    }
}

} // namespace render