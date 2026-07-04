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
    UniformBufferManager& uniformBufferManager,
    CommandPool& commandPool)
    : vulkanDevice(device),
      commandPool(commandPool),
      memoryAllocator(allocator),
      surfaceRenderer(std::make_unique<HeatSurfaceRenderer>(device, uniformBufferManager)),
      vectorArrowRenderer(std::make_unique<VectorArrowRenderer>(device, allocator, uniformBufferManager, commandPool)),
      heatPaletteRenderer(std::make_unique<HeatPaletteRenderer>(device, allocator, commandPool)) {
}

HeatOverlayRenderer::~HeatOverlayRenderer() {
    cleanup();
}

void HeatOverlayRenderer::initializeSurface(VkRenderPass renderPass, uint32_t surfaceSubpass, uint32_t updatedMaxFramesInFlight) {
    if (!surfaceRenderer || surfaceInitialized) {
        return;
    }

    maxFramesInFlight = updatedMaxFramesInFlight;
    surfaceRenderer->initialize(renderPass, surfaceSubpass, maxFramesInFlight);
    surfaceInitialized = true;
    rebuildBindings();
}

void HeatOverlayRenderer::initializeOverlay(VkRenderPass renderPass, uint32_t overlaySubpass, uint32_t updatedMaxFramesInFlight) {
    if (!vectorArrowRenderer || overlayInitialized) {
        return;
    }

    maxFramesInFlight = updatedMaxFramesInFlight;
    vectorArrowRenderer->initialize(renderPass, overlaySubpass, maxFramesInFlight);
    if (heatPaletteRenderer) {
        heatPaletteRenderer->initialize(renderPass, overlaySubpass, maxFramesInFlight);
    }
    overlayInitialized = true;
    rebuildBindings();
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
    if (!surfaceRenderer || !vectorArrowRenderer) {
        return;
    }

    surfaceBindings.clear();
    fluxVectorBindings.clear();

    bool heatPaletteVisible = false;
    float heatPaletteMin = 0.0f;
    float heatPaletteMax = 100.0f;

    for (const auto& [socketKey, config] : configsBySocket) {
        if (!config.active && !config.paused) {

            continue;
        }

        if (config.showHeatPalette) {
            heatPaletteVisible = true;
            heatPaletteMin = config.heatPaletteMinTemp;
            heatPaletteMax = config.heatPaletteMaxTemp;
        }

        if (config.showHeatOverlay) {
            surfaceBindings.reserve(surfaceBindings.size() + config.models.size());
            for (size_t index = 0; index < config.models.size(); ++index) {
                const auto& modelProduct = config.models[index];
                if (!modelProduct.isValid() ||
                    modelProduct.runtimeModelId == 0 ||
                    modelProduct.renderVertexBuffer == VK_NULL_HANDLE ||
                    modelProduct.renderIndexBuffer == VK_NULL_HANDLE ||
                    modelProduct.renderIndexCount == 0) {
                    continue;
                }

                if (index >= config.modelSurfaceBuffers.size() ||
                    config.modelSurfaceBuffers[index] == VK_NULL_HANDLE ||
                    index >= config.modelBufferViews.size()) {
                    continue;
                }

                bool viewsValid = true;
                for (int j = 0; j < 10; ++j) {
                    if (config.modelBufferViews[index][j] == VK_NULL_HANDLE) {
                        viewsValid = false;
                        break;
                    }
                }
                if (!viewsValid) {
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
                    config.modelSurfaceBuffers[index] != VK_NULL_HANDLE &&
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



    if (heatPaletteRenderer) {
        heatPaletteRenderer->setVisible(heatPaletteVisible);
        if (heatPaletteVisible) {
            heatPaletteRenderer->setRange(heatPaletteMin, heatPaletteMax);
        }
    }
}

void HeatOverlayRenderer::renderSurface(VkCommandBuffer commandBuffer, uint32_t frameIndex) {
    if (!surfaceInitialized || !surfaceRenderer) {
        return;
    }

    surfaceRenderer->render(commandBuffer, frameIndex, surfaceBindings);
}

void HeatOverlayRenderer::renderOverlay(VkCommandBuffer commandBuffer, uint32_t frameIndex) {
    if (!overlayInitialized || !vectorArrowRenderer) {
        return;
    }

    vectorArrowRenderer->render(commandBuffer, frameIndex, fluxVectorBindings);
}

void HeatOverlayRenderer::renderScreen(VkCommandBuffer commandBuffer, uint32_t frameIndex, VkExtent2D extent) {
    if (!overlayInitialized || !heatPaletteRenderer) {
        return;
    }

    heatPaletteRenderer->render(commandBuffer, frameIndex, extent);
}

void HeatOverlayRenderer::cleanup() {
    configsBySocket.clear();
    surfaceBindings.clear();
    fluxVectorBindings.clear();
    maxFramesInFlight = 0;
    surfaceInitialized = false;
    overlayInitialized = false;
    if (surfaceRenderer) {
        surfaceRenderer->cleanup();
    }
    if (vectorArrowRenderer) {
        vectorArrowRenderer->cleanup();
    }
    if (heatPaletteRenderer) {
        heatPaletteRenderer->cleanup();
    }
}

} // namespace render
