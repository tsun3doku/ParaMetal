#pragma once

#include <cstdint>
#include <vector>

#include <glm/vec3.hpp>

#include "mesh/remesher/Remesher.hpp"

class ModelRegistry;
class VulkanDevice;

class RemeshSystem {
public:
    RemeshSystem(
        Remesher& remesher,
        VulkanDevice& vulkanDevice,
        ModelRegistry& resourceManager);
    ~RemeshSystem();

    void setSourceGeometry(const std::vector<float>& pointPositions, const std::vector<uint32_t>& triangleIndices);
    void setParams(int iterations, float minAngleDegrees, float maxEdgeLength, float stepSize);
    void setRuntimeModelId(uint32_t runtimeModelId);
    bool ensureConfigured();
    void disable();
    bool isReady() const { return ready; }
    uint32_t getRuntimeModelId() const { return runtimeModelId; }
    const std::vector<glm::vec3>& getGeometryPositions() const { return geometryPositions; }
    const std::vector<uint32_t>& getGeometryTriangleIndices() const { return geometryTriangleIndices; }
    const SupportingHalfedge::IntrinsicMesh& getIntrinsicMesh() const { return intrinsicMesh; }
    const SupportingHalfedge::GPUResources& getIntrinsicGpuResources() const { return intrinsicGpuResources; }

private:
    void resetState(bool waitForDevice);

    VulkanDevice& vulkanDevice;
    ModelRegistry& resourceManager;
    Remesher& remesher;

    std::vector<float> pointPositions;
    std::vector<uint32_t> triangleIndices;
    int iterations = 1;
    float minAngleDegrees = 20.0f;
    float maxEdgeLength = 0.1f;
    float stepSize = 0.25f;
    uint32_t runtimeModelId = 0;
    std::vector<glm::vec3> geometryPositions;
    std::vector<uint32_t> geometryTriangleIndices;
    SupportingHalfedge::IntrinsicMesh intrinsicMesh;
    SupportingHalfedge::GPUResources intrinsicGpuResources;
    bool ready = false;
};
