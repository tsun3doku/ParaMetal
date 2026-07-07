#include "LightingSystem.hpp"

#include "Camera.hpp"
#include "vulkan/UniformBufferManager.hpp"

#include <algorithm>
#include <cstring>

LightingSystem::LightingSystem(Camera& camera, UniformBufferManager& uniformBufferManager)
    : camera(camera), uniformBufferManager(uniformBufferManager) {
}

void LightingSystem::update(uint32_t frameIndex) {
    LightUniformBufferObject lightUbo{};
    lightUbo.iblParams = glm::vec4(iblIntensity, iblDiffuseScale, iblSpecularScale, iblMaxReflectionLod);
    lightUbo.cameraPos = camera.getPosition();

    if (!enabled) {
        lightUbo.iblParams = glm::vec4(0.0f);
        lightUbo.cameraPos = glm::vec3(0.0f);
    }

    auto& mapped = uniformBufferManager.getLightBuffersMapped();
    if (frameIndex < mapped.size() && mapped[frameIndex]) {
        std::memcpy(mapped[frameIndex], &lightUbo, sizeof(lightUbo));
    }
}

void LightingSystem::setIBLIntensity(float intensity) {
    iblIntensity = std::max(intensity, 0.0f);
}

void LightingSystem::setIBLDiffuseScale(float diffuseScale) {
    iblDiffuseScale = std::max(diffuseScale, 0.0f);
}

void LightingSystem::setIBLSpecularScale(float specularScale) {
    iblSpecularScale = std::max(specularScale, 0.0f);
}

void LightingSystem::setIBLMaxReflectionLod(float maxReflectionLod) {
    iblMaxReflectionLod = std::max(maxReflectionLod, 0.0f);
}

void LightingSystem::setEnabled(bool enabled) {
    this->enabled = enabled;
}
