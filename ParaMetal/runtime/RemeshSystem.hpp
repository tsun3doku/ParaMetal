#pragma once

#include <cstdint>
#include <vector>

#include <glm/vec3.hpp>

#include "mesh/remesher/Remesher.hpp"
#include "mesh/remesher/SupportingHalfedge.hpp"

struct RemeshProduct;

class RemeshSystem {
public:
    explicit RemeshSystem(Remesher& remesher);
    ~RemeshSystem();

    void setSourceGeometry(const std::vector<float>& pointPositions, const std::vector<uint32_t>& triangleIndices);
    void setParams(int iterations, float minAngleDegrees, float maxEdgeLength, float stepSize);
    void setRuntimeModelId(uint32_t runtimeModelId);

    bool remesh();

    const SupportingHalfedge::IntrinsicMesh& intrinsicMesh() const { return mesh; }
    const std::vector<float>& sourcePositions() const { return pointPositions; }
    const std::vector<uint32_t>& sourceTriangles() const { return triangleIndices; }
    uint32_t runtimeModelId() const { return modelRuntimeId; }

    void copyGpu(RemeshProduct& product) const;

private:
    void freeGpu();

    Remesher& remesher;

    std::vector<float> pointPositions;
    std::vector<uint32_t> triangleIndices;
    int iterations = 1;
    float minAngleDegrees = 20.0f;
    float maxEdgeLength = 0.1f;
    float stepSize = 0.25f;
    uint32_t modelRuntimeId = 0;

    SupportingHalfedge::IntrinsicMesh mesh;
    SupportingHalfedge::GPUResources gpuResources{};
};
