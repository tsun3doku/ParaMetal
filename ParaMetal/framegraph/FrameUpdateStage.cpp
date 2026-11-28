#include "FrameUpdateStage.hpp"

#include "interaction/InteractionContext.hpp"
#include "scene/LightingSystem.hpp"
#include "scene/MaterialSystem.hpp"
#include "scene/ModelSelection.hpp"
#include "scene/SubselectionPickService.hpp"
#include "vulkan/ModelRegistry.hpp"
#include "render/SceneRenderer.hpp"
#include "util/Structs.hpp"
#include "vulkan/UniformBufferManager.hpp"

FrameUpdateStage::FrameUpdateStage(
    InteractionContext& interaction,
    UniformBufferManager& uniformBufferManager,
    ModelRegistry& modelRegistry,
    LightingSystem& lightingSystem,
    MaterialSystem& materialSystem,
    SceneRenderer& sceneRenderer,
    ModelSelection& modelSelection,
    SubselectionPickService& subselectionPickService)
    : interaction(interaction),
      uniformBufferManager(uniformBufferManager),
      modelRegistry(modelRegistry),
      lightingSystem(lightingSystem),
      materialSystem(materialSystem),
      sceneRenderer(sceneRenderer),
      modelSelection(modelSelection),
      subselectionPickService(subselectionPickService) {
}

void FrameUpdateStage::setViewportExtent(uint32_t width, uint32_t height) {
    subselectionPickService.setViewportExtent(width, height);
}

void FrameUpdateStage::processPicking(uint32_t frameIndex) {
    modelSelection.processPickingRequests(frameIndex);
    subselectionPickService.processPickingRequests(frameIndex);
}

void FrameUpdateStage::updateFrameState(uint32_t frameIndex, const render::SceneView& sceneView) {
    interaction.resolvePick(modelSelection, modelRegistry);

    UniformBufferObject ubo{};
    uniformBufferManager.updateUniformBuffer(frameIndex, sceneView, ubo);

    sceneRenderer.updateGrid(frameIndex, sceneView, modelRegistry.sceneWorldExtent());

    lightingSystem.update(frameIndex);
    materialSystem.update(frameIndex);
}
