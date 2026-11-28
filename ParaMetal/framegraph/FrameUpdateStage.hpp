#pragma once

#include <cstdint>

#include "scene/SceneView.hpp"

class InteractionContext;
class UniformBufferManager;
class ModelRegistry;
class LightingSystem;
class MaterialSystem;
class SceneRenderer;
class ModelSelection;
class SubselectionPickService;

class FrameUpdateStage {
public:
    FrameUpdateStage(
        InteractionContext& interaction,
        UniformBufferManager& uniformBufferManager,
        ModelRegistry& modelRegistry,
        LightingSystem& lightingSystem,
        MaterialSystem& materialSystem,
        SceneRenderer& sceneRenderer,
        ModelSelection& modelSelection,
        SubselectionPickService& subselectionPickService);

    void setViewportExtent(uint32_t width, uint32_t height);
    void processPicking(uint32_t frameIndex);
    void updateFrameState(uint32_t frameIndex, const render::SceneView& sceneView);

private:
    InteractionContext& interaction;
    UniformBufferManager& uniformBufferManager;
    ModelRegistry& modelRegistry;
    LightingSystem& lightingSystem;
    MaterialSystem& materialSystem;
    SceneRenderer& sceneRenderer;
    ModelSelection& modelSelection;
    SubselectionPickService& subselectionPickService;
};

