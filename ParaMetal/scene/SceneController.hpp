#pragma once

#include <cstdint>
#include <string>

#include <glm/vec3.hpp>

class CameraController;
class ModelComputeController;
class ModelRegistry;

class SceneController {
public:
    SceneController(
        ModelRegistry& modelRegistry,
        CameraController& cameraController);

    void setModelComputeController(ModelComputeController* updatedModelComputeController);
    bool tryGetRuntimeModelSocketKey(uint32_t runtimeModelId, uint64_t& outSocketKey) const;
    void focusOnVisibleModel();
    void focusCameraOn(const glm::vec3& worldCenter);

private:
    ModelRegistry& modelRegistry;
    CameraController& cameraController;
    ModelComputeController* modelComputeController = nullptr;
};


