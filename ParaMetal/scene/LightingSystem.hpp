#pragma once

#include <glm/glm.hpp>

#include "util/Structs.hpp"

class Camera;
class UniformBufferManager;

class LightingSystem {
public:
    LightingSystem(Camera& camera, UniformBufferManager& uniformBufferManager);

    void update(uint32_t frameIndex);

    void setIBLIntensity(float intensity);
    void setIBLDiffuseScale(float diffuseScale);
    void setIBLSpecularScale(float specularScale);
    void setIBLMaxReflectionLod(float maxReflectionLod);
    void setEnabled(bool enabled);

private:
    Camera& camera;
    UniformBufferManager& uniformBufferManager;
    float iblIntensity = 0.12f;
    float iblDiffuseScale = 1.0f;
    float iblSpecularScale = 1.0f;
    float iblMaxReflectionLod = 6.0f;
    bool enabled = true;
};
