#include "HeatOverlayRenderer.hpp"

#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/UniformBufferManager.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "domain/HeatModelData.hpp"
#include "ui/UiPalette.hpp"

#include <iostream>
#include <algorithm>
#include <cmath>

namespace render {

HeatOverlayRenderer::HeatOverlayRenderer(
    VulkanDevice& device,
    MemoryAllocator& allocator,
    UniformBufferManager& uniformBufferManager,
    CommandPool& commandPool)
    : vulkanDevice(device),
      commandPool(commandPool),
      memoryAllocator(allocator),
      surfaceRenderer(std::make_unique<HeatSurfaceRenderer>(device, allocator, uniformBufferManager, commandPool)),
      surfaceVectorRenderer(std::make_unique<SurfaceVectorRenderer>(device, allocator, uniformBufferManager, commandPool)),
      boxRenderer(std::make_unique<HeatBoxRenderer>(device, allocator, uniformBufferManager)),
      probeRenderer(std::make_unique<ProbeMarkerRenderer>(device, allocator, uniformBufferManager, commandPool)) {
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
    if ((!surfaceVectorRenderer && !boxRenderer && !probeRenderer) || overlayInitialized) {
        return;
    }

    maxFramesInFlight = updatedMaxFramesInFlight;
    if (surfaceVectorRenderer) {
        surfaceVectorRenderer->initialize(renderPass, overlaySubpass, maxFramesInFlight);
    }
    if (boxRenderer) {
        boxRenderer->initialize(renderPass, overlaySubpass, maxFramesInFlight);
    }
    if (probeRenderer) {
        probeRenderer->initialize(renderPass, overlaySubpass, maxFramesInFlight);
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
    surfaceBindings.clear();
    fluxVectorBindings.clear();
    boxBindings.clear();
    streamlineBindings.clear();
    probeBindings.clear();

    paletteVisible = false;
    cfdVelocityPaletteVisible = false;
    bool anyIsocontours = false;
    float isocontourSpacingC = 0.0f;

    for (const auto& [socketKey, config] : configsBySocket) {
        if (!config.active) continue;

        if (config.showHeatPalette) {
            paletteVisible = true;
        }

        if (config.showCfdVelocityPalette) {
            cfdVelocityPaletteVisible = true;
        }

        if (config.showIsocontours) {
            anyIsocontours = true;
            isocontourSpacingC = config.isocontourSpacingC;
        }

        if (config.showHeatOverlay) {
            surfaceBindings.reserve(surfaceBindings.size() + config.modelRuntimeIds.size());
            for (size_t index = 0; index < config.modelRuntimeIds.size(); ++index) {
                if (config.modelRuntimeIds[index] == 0 ||
                    config.modelRenderVertexBuffers[index] == VK_NULL_HANDLE ||
                    config.modelRenderIndexBuffers[index] == VK_NULL_HANDLE ||
                    config.modelRenderIndexCounts[index] == 0) {
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
                binding.runtimeModelId = config.modelRuntimeIds[index];
                binding.vertexBuffer = config.modelRenderVertexBuffers[index];
                binding.vertexBufferOffset = config.modelRenderVertexBufferOffsets[index];
                binding.indexBuffer = config.modelRenderIndexBuffers[index];
                binding.indexBufferOffset = config.modelRenderIndexBufferOffsets[index];
                binding.indexCount = config.modelRenderIndexCounts[index];
                binding.modelMatrix = config.modelMatrices[index];
                binding.bufferViews = config.modelBufferViews[index];
                binding.surfaceBuffer = config.modelSurfaceBuffers[index];
                binding.surfaceBufferOffset = config.modelSurfaceBufferOffsets[index];


                surfaceBindings.push_back(binding);
            }
        }

        if (config.showFluxVectors) {
            for (size_t index = 0; index < config.modelRuntimeIds.size(); ++index) {
                if (config.modelRuntimeIds[index] != 0 &&
                    index < config.modelSurfaceBuffers.size() &&
                    config.modelSurfaceBuffers[index] != VK_NULL_HANDLE &&
                    index < config.modelSurfaceBufferOffsets.size() &&
                    index < config.modelSurfaceGradientBuffers.size() &&
                    index < config.modelSurfaceGradientBufferOffsets.size() &&
                    index < config.modelSurfacePointCounts.size() &&
                    config.modelSurfaceGradientBuffers[index] != VK_NULL_HANDLE &&
                    config.modelSurfacePointCounts[index] != 0) {
                    SurfaceVectorRenderer::RenderBinding vectorBinding{};
                    vectorBinding.bindingKey = config.modelRuntimeIds[index];
                    vectorBinding.surfaceBuffer = config.modelSurfaceBuffers[index];
                    vectorBinding.surfaceBufferOffset = config.modelSurfaceBufferOffsets[index];
                    vectorBinding.gradientBuffer = config.modelSurfaceGradientBuffers[index];
                    vectorBinding.gradientBufferOffset = config.modelSurfaceGradientBufferOffsets[index];
                    vectorBinding.sampleCount = config.modelSurfacePointCounts[index];
                    vectorBinding.modelMatrix = config.modelMatrices[index];
                    vectorBinding.scale = config.fluxVectorScale;
                    vectorBinding.canonicalToWorldScale = config.canonicalToWorldScale;

                    fluxVectorBindings.push_back(vectorBinding);
                }
            }
        }

        if (config.showDomainBox && config.hasGlobalDomain) {
            HeatBoxRenderer::RenderBinding boxBinding{};
            boxBinding.corners = config.globalDomainCorners;

            boxBindings.push_back(boxBinding);
        }

        if (config.showCfdStreamlines &&
            config.cfdStreamlineVertexBuffer != VK_NULL_HANDLE &&
            config.cfdStreamlineVertexCount != 0) {
            StreamlineRenderer::RenderBinding streamlineBinding{};
            streamlineBinding.bindingKey = socketKey;
            streamlineBinding.vertexBuffer = config.cfdStreamlineVertexBuffer;
            streamlineBinding.vertexBufferOffset = config.cfdStreamlineVertexBufferOffset;
            streamlineBinding.vertexCount = config.cfdStreamlineVertexCount;

            streamlineBindings.push_back(streamlineBinding);
        }

        for (size_t i = 0; i < config.probePositions.size(); ++i) {
            ProbeMarkerRenderer::RenderBinding probeBinding{};
            probeBinding.worldPosition = config.probePositions[i];
            probeBinding.scale = 1.0f;
            probeBinding.probeId = (i < config.probeIds.size()) ? config.probeIds[i] : 0;
            probeBinding.nodeId = (i < config.probeNodeIds.size()) ? config.probeNodeIds[i] : 0;
            probeBinding.probeName = (i < config.probeNames.size()) ? config.probeNames[i] : ("Probe_" + std::to_string(i + 1));
            probeBinding.showTemperature = (i < config.probeShowTemperatures.size()) ? config.probeShowTemperatures[i] : true;
            probeBinding.showName = (i < config.probeShowNames.size()) ? config.probeShowNames[i] : true;

            // Preserve current temperature if probe was already bound
            float prevTemp = 20.0f;
            for (const auto& prev : probeBindings) {
                if (prev.nodeId == probeBinding.nodeId && prev.probeId == probeBinding.probeId) {
                    prevTemp = prev.currentTemperatureC;
                    break;
                }
            }
            probeBinding.currentTemperatureC = prevTemp;

            const bool isSelected = (selectedProbeId != 0 && probeBinding.probeId == selectedProbeId)
                                 || (selectedProbeIndex >= 0 && static_cast<int>(i) == selectedProbeIndex);
            probeBinding.isSelected = isSelected;
            probeBinding.color = isSelected
                ? ui::UiPalette::selectionHighlightSrgb()
                : ui::UiPalette::probeMarkerRedSrgb();
            probeBindings.push_back(probeBinding);
        }
    }

    if (surfaceRenderer) {
        const float spacing = anyIsocontours ? isocontourSpacingC : 0.0f;
        surfaceRenderer->setIsocontourSpacing(spacing);
    }
}

void HeatOverlayRenderer::renderSurface(VkCommandBuffer commandBuffer, uint32_t frameIndex, VkExtent2D extent) {
    (void)extent;
    if (!surfaceInitialized || !surfaceRenderer) {
        return;
    }

    surfaceRenderer->render(commandBuffer, frameIndex, surfaceBindings);
}

void HeatOverlayRenderer::setPaletteRange(float minimum, float maximum) {
    if (std::isfinite(minimum) && std::isfinite(maximum) && maximum > minimum) {
        if (surfaceRenderer) surfaceRenderer->setRange(minimum, maximum);
    }
}
void HeatOverlayRenderer::setPalette(int palette) {
    if (surfaceRenderer) surfaceRenderer->setPalette(palette);
}

uint32_t HeatOverlayRenderer::getPalette() const {
    return surfaceRenderer ? surfaceRenderer->getPalette() : 0u;
}

float HeatOverlayRenderer::getMinTemperature() const {
    return surfaceRenderer ? surfaceRenderer->getMinTemperature() : 0.0f;
}

float HeatOverlayRenderer::getMaxTemperature() const {
    return surfaceRenderer ? surfaceRenderer->getMaxTemperature() : 100.0f;
}

void HeatOverlayRenderer::setTextRenderer(ScreenTextRenderer* renderer) {
    if (probeRenderer) {
        probeRenderer->setTextRenderer(renderer);
    }
}

void HeatOverlayRenderer::renderOverlay(
    VkCommandBuffer commandBuffer,
    uint32_t frameIndex,
    VkExtent2D extent,
    const glm::mat4& viewProj) {
    if (!overlayInitialized) {
        return;
    }

    if (surfaceVectorRenderer) {
        surfaceVectorRenderer->render(commandBuffer, frameIndex, fluxVectorBindings);
    }
    if (probeRenderer) {
        probeRenderer->render(commandBuffer, frameIndex, extent, viewProj, probeBindings);
    }
    if (boxRenderer) {
        boxRenderer->render(commandBuffer, frameIndex, boxBindings);
    }
}

void HeatOverlayRenderer::cleanup() {
    configsBySocket.clear();
    surfaceBindings.clear();
    fluxVectorBindings.clear();
    boxBindings.clear();
    streamlineBindings.clear();
    probeBindings.clear();
    maxFramesInFlight = 0;
    surfaceInitialized = false;
    overlayInitialized = false;
    if (surfaceRenderer) {
        surfaceRenderer->cleanup();
    }
    if (surfaceVectorRenderer) {
        surfaceVectorRenderer->cleanup();
    }
    if (boxRenderer) {
        boxRenderer->cleanup();
    }
    if (probeRenderer) {
        probeRenderer->cleanup();
    }
}

void HeatOverlayRenderer::setSelectedProbe(uint32_t probeId, int probeIndex) {
    if (selectedProbeId == probeId && selectedProbeIndex == probeIndex) {
        return;
    }
    selectedProbeId = probeId;
    selectedProbeIndex = probeIndex;
    rebuildBindings();
}

void HeatOverlayRenderer::updateProbePosition(uint32_t probeId, int probeIndex, const glm::vec3& newPos) {
    if (probeId != 0) {
        for (auto& binding : probeBindings) {
            if (binding.probeId == probeId) {
                binding.worldPosition = newPos;
                return;
            }
        }
    }
    if (probeIndex >= 0 && probeIndex < static_cast<int>(probeBindings.size())) {
        probeBindings[probeIndex].worldPosition = newPos;
    }
}

void HeatOverlayRenderer::updateProbeTemperatures(uint64_t nodeId, const std::vector<float>& temps) {
    size_t tempIdx = 0;
    for (auto& binding : probeBindings) {
        if (binding.nodeId == nodeId || binding.nodeId == 0) {
            if (tempIdx < temps.size()) {
                binding.currentTemperatureC = temps[tempIdx++];
            }
        }
    }
}

} // namespace render
