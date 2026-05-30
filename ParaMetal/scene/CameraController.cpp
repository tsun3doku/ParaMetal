#include "CameraController.hpp"

#include "Camera.hpp"
#include "SceneView.hpp"

CameraController::CameraController(Camera& camera)
    : camera(camera) {
}

void CameraController::setPanSensitivity(float sensitivity) {
    camera.panSensitivity = sensitivity;
}

void CameraController::focusOn(const glm::vec3& target) {
    camera.setLookAt(target);
}

void CameraController::resetFocusToOrigin() {
    camera.setLookAt(glm::vec3(0.0f));
    camera.resetRadius();
}

void CameraController::setCameraState(const glm::vec3& lookAt, const glm::quat& orientation, float radius, float fov) {
    camera.setLookAt(lookAt);
    camera.setOrientation(orientation);
    camera.setRadius(radius);
    camera.update(0.0f);
    camera.setFov(fov);
}

void CameraController::tick(float deltaTime) {
    camera.update(deltaTime);
}

render::SceneView CameraController::buildSceneView(VkExtent2D extent) const {
    render::SceneView sceneView{};
    sceneView.view = camera.getViewMatrix();
    sceneView.proj = camera.getProjectionMatrix(static_cast<float>(extent.width) / static_cast<float>(extent.height));
    sceneView.proj[1][1] *= -1.0f;
    sceneView.cameraPosition = camera.getPosition();
    sceneView.cameraFov = camera.getFov();
    return sceneView;
}
