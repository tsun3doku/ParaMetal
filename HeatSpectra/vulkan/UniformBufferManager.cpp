#include <vulkan/vulkan.h>

#include "MemoryAllocator.hpp"
#include "VulkanDevice.hpp"
#include "UniformBufferManager.hpp"
#include "VulkanBuffer.hpp"
#include <array>

UniformBufferManager::UniformBufferManager(VulkanDevice& vulkanDevice, MemoryAllocator& memoryAllocator, uint32_t maxFramesInFlight)
: vulkanDevice(vulkanDevice), memoryAllocator(memoryAllocator) {
    createUniformBuffers(maxFramesInFlight);
    createGridUniformBuffers(maxFramesInFlight);
    createLightUniformBuffers(maxFramesInFlight);
    createMaterialUniformBuffers(maxFramesInFlight);
    createSSAOKernelBuffers(maxFramesInFlight);
}

UniformBufferManager::~UniformBufferManager() {
}

void UniformBufferManager::createUniformBuffers(uint32_t maxFramesInFlight) {
    VkDeviceSize bufferSize = sizeof(UniformBufferObject);
    
    uniformBuffers.resize(maxFramesInFlight);
    uniformBuffersMapped.resize(maxFramesInFlight);
    uniformBufferOffsets_.resize(maxFramesInFlight);

    for (size_t i = 0; i < maxFramesInFlight; i++) {
        createUniformBuffer(
            memoryAllocator,
            vulkanDevice,
            bufferSize,
            uniformBuffers[i],
            uniformBufferOffsets_[i],
            &uniformBuffersMapped[i]
        );
    }
}

void UniformBufferManager::createGridUniformBuffers(uint32_t maxFramesInFlight) {
    VkDeviceSize bufferSize = sizeof(GridUniformBufferObject);

    gridUniformBuffers.resize(maxFramesInFlight);
    gridUniformBuffersMapped.resize(maxFramesInFlight);
    gridUniformBufferOffsets_.resize(maxFramesInFlight);

    for (size_t i = 0; i < maxFramesInFlight; i++) {
        createUniformBuffer(
            memoryAllocator,
            vulkanDevice,
            bufferSize,
            gridUniformBuffers[i],
            gridUniformBufferOffsets_[i],
            &gridUniformBuffersMapped[i]
        );
    }
}

void UniformBufferManager::createLightUniformBuffers(uint32_t maxFramesInFlight) {
    VkDeviceSize bufferSize = sizeof(LightUniformBufferObject); 
    
    lightBuffers.resize(maxFramesInFlight);
    lightBuffersMapped.resize(maxFramesInFlight);
    lightBufferOffsets_.resize(maxFramesInFlight);

    for (size_t i = 0; i < maxFramesInFlight; i++) {
        createUniformBuffer(
            memoryAllocator,
            vulkanDevice,
            bufferSize,
            lightBuffers[i],
            lightBufferOffsets_[i],
            &lightBuffersMapped[i]
        );
    }
}

void UniformBufferManager::createMaterialUniformBuffers(uint32_t maxFramesInFlight) {
    VkDeviceSize bufferSize = sizeof(MaterialUniformBufferObject);

    materialBuffers.resize(maxFramesInFlight);
    materialBuffersMapped.resize(maxFramesInFlight);
    materialBufferOffsets_.resize(maxFramesInFlight);

    for (size_t i = 0; i < maxFramesInFlight; i++) {
        createUniformBuffer(
            memoryAllocator,
            vulkanDevice,
            bufferSize,
            materialBuffers[i],
            materialBufferOffsets_[i],
            &materialBuffersMapped[i]
        );
    }
}

void UniformBufferManager::createSSAOKernelBuffers(uint32_t maxFramesInFlight) {
    VkDeviceSize bufferSize = sizeof(SSAOKernelBufferObject);

    SSAOKernelBuffers.resize(maxFramesInFlight);
    SSAOKernelBuffersMapped.resize(maxFramesInFlight);
    SSAOKernelBufferOffsets_.resize(maxFramesInFlight);

    // Generate SSAO kernel samples once
    SSAOKernelBufferObject ssaoKernel;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> randomFloats(0.0, 1.0);

    for (int j = 0; j < 16; ++j) {
        glm::vec3 sample(
            randomFloats(generator) * 2.0f - 1.0f,
            randomFloats(generator) * 2.0f - 1.0f,
            randomFloats(generator)
        );
        sample = glm::normalize(sample);
        sample *= randomFloats(generator);

        float scale = float(j) / 16.0f;
        scale = glm::mix(0.1f, 1.0f, scale * scale);
        sample *= scale;

        ssaoKernel.SSAOKernel[j] = glm::vec4(sample, 0.0f);
    }

    // Create buffers and copy kernel data
    for (size_t i = 0; i < maxFramesInFlight; i++) {
        createUniformBuffer(
            memoryAllocator,
            vulkanDevice,
            bufferSize,
            SSAOKernelBuffers[i],
            SSAOKernelBufferOffsets_[i],
            &SSAOKernelBuffersMapped[i]
        );

        memcpy(SSAOKernelBuffersMapped[i], &ssaoKernel, sizeof(ssaoKernel));
    }
}

void UniformBufferManager::updateUniformBuffer(uint32_t currentImage, const render::SceneView& sceneView, UniformBufferObject& ubo) {
    ubo.model = glm::mat4(1.0f);
    ubo.view = sceneView.view;
    ubo.proj = sceneView.proj;

    ubo.color = glm::vec3(1.0f, 1.0f, 1.0f);

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

void UniformBufferManager::setColor(glm::vec3 newColor, UniformBufferObject& ubo) {
    ubo.color = newColor;
    for (size_t i = 0; i < uniformBuffersMapped.size(); i++) {
        UniformBufferObject* mappedUbo = (UniformBufferObject*)uniformBuffersMapped[i];
        mappedUbo->color = newColor;
    }
}

void UniformBufferManager::updateGridUniformBuffer(uint32_t currentImage, const render::SceneView& sceneView, GridUniformBufferObject& gridUbo, const glm::vec3& gridSize) {
    
    gridUbo.view = sceneView.view;
    gridUbo.proj = sceneView.proj;
    gridUbo.pos = sceneView.cameraPosition;
    gridUbo.gridSize = gridSize;

    memcpy(gridUniformBuffersMapped[currentImage], &gridUbo, sizeof(gridUbo));
}

void UniformBufferManager::updateSSAOKernelBuffer(uint32_t currentImage, SSAOKernelBufferObject& ssaoKernel) {
    memcpy(SSAOKernelBuffersMapped[currentImage], &ssaoKernel, sizeof(ssaoKernel));
}

void UniformBufferManager::cleanup(uint32_t maxFramesInFlight) {
    for (size_t i = 0; i < maxFramesInFlight; i++) {
        freeBuffer(memoryAllocator, uniformBuffers[i], uniformBufferOffsets_[i]);
        freeBuffer(memoryAllocator, gridUniformBuffers[i], gridUniformBufferOffsets_[i]);
        freeBuffer(memoryAllocator, lightBuffers[i], lightBufferOffsets_[i]);
        freeBuffer(memoryAllocator, materialBuffers[i], materialBufferOffsets_[i]);
        freeBuffer(memoryAllocator, SSAOKernelBuffers[i], SSAOKernelBufferOffsets_[i]);
    }
}
