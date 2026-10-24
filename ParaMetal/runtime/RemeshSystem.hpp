#pragma once

#include <cstdint>
#include <vector>

#include <glm/vec3.hpp>

#include "mesh/remesher/Remesher.hpp"
#include "mesh/remesher/SupportingHalfedge.hpp"

struct RemeshProduct;

// Product-agnostic backend owning CPU mesh and GPU resources until controller handoff
class RemeshSystem {
public:
    explicit RemeshSystem(Remesher& remesher);
    ~RemeshSystem();

    void setSourceGeometry(const std::vector<float>& pointPositions, const std::vector<uint32_t>& triangleIndices);
    void setParams(int iterations, float minAngleDegrees, float maxEdgeLength, float stepSize);
    void setRuntimeModelId(uint32_t runtimeModelId);

    // Run remesh into internal state, freeing untaken GPU resources from prior runs
    bool remesh();

    // CPU data: borrowed by const ref (controller copies by value into product).
    const SupportingHalfedge::IntrinsicMesh& intrinsicMesh() const { return mesh; }
    const std::vector<float>& sourcePositions() const { return pointPositions; }
    const std::vector<uint32_t>& sourceTriangles() const { return triangleIndices; }
    uint32_t runtimeModelId() const { return modelRuntimeId; }

    // Non-detaching copy of this generation's handles into a product record.
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
