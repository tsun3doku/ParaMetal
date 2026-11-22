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

    void setSourceGeometry(const std::vector<float>& inPointPositions, const std::vector<uint32_t>& inTriangleIndices);
    void setParams(int inIterations, float inMinAngleDegrees, float inMaxEdgeLength, float inStepSize);
    void setRuntimeModelId(uint32_t runtimeModelId);

    bool remesh();

    const SupportingHalfedge::IntrinsicMesh& intrinsicMesh() const { return outMesh; }
    const std::vector<float>& sourcePositions() const { return inPointPositions; }
    const std::vector<uint32_t>& sourceTriangles() const { return inTriangleIndices; }
    uint32_t runtimeModelId() const { return inRuntimeModelId; }

    void copyGpu(RemeshProduct& product) const;

private:
    void freeGpu();

    Remesher& remesher;

    std::vector<float> inPointPositions;
    std::vector<uint32_t> inTriangleIndices;
    int inIterations = 1;
    float inMinAngleDegrees = 20.0f;
    float inMaxEdgeLength = 0.1f;
    float inStepSize = 0.25f;
    uint32_t inRuntimeModelId = 0;

    SupportingHalfedge::IntrinsicMesh outMesh;
    SupportingHalfedge::GPUResources outGpuResources{};
};
