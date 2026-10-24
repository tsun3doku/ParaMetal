#include "SceneController.hpp"

#include "CameraController.hpp"
#include "runtime/ModelComputeController.hpp"
#include "vulkan/ModelRegistry.hpp"

SceneController::SceneController(
    ModelRegistry& modelRegistry,
    CameraController& cameraController)
    : modelRegistry(modelRegistry),
      cameraController(cameraController) {
}

void SceneController::setModelComputeController(ModelComputeController* updatedModelComputeController) {
    modelComputeController = updatedModelComputeController;
}

bool SceneController::tryGetRuntimeModelSocketKey(uint32_t runtimeModelId, uint64_t& outSocketKey) const {
    outSocketKey = 0;
    if (!modelComputeController) {
        return false;
    }

    return modelComputeController->tryGetSocketKey(runtimeModelId, outSocketKey);
}

void SceneController::focusOnVisibleModel() {
    for (uint32_t modelId : modelRegistry.getRenderableModelIds()) {
        glm::vec3 worldMin(0.0f);
        glm::vec3 worldMax(0.0f);
        if (modelRegistry.tryGetWorldBounds(modelId, worldMin, worldMax)) {
            cameraController.focusOn((worldMin + worldMax) * 0.5f);
            return;
        }
    }
}

void SceneController::focusCameraOn(const glm::vec3& worldCenter) {
    cameraController.focusOn(worldCenter);
}

