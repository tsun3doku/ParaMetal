#pragma once

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>
#include "../voronoi/GlobalDomainTopology.hpp"

namespace cfd {

struct CFDFaceGpu {
    uint32_t cellA = 0;
    uint32_t cellB = 0;
    glm::vec3 fluxVectorA{0.0f};
    glm::vec3 fluxVectorB{0.0f};
    float projectionWeight = 0.0f; // sum of constituent fine projectionWeights
};

struct CFDBoundaryFaceGpu {
    uint32_t cell = 0;
    uint32_t boundaryCondition = UINT32_MAX; // 0..5 domain boundary, 6 embedded solid wall
    glm::vec3 areaVector{0.0f};
    float projectionWeight = 0.0f; // sum of constituent fine (area / distance)
};

struct CFDTopology {
    uint32_t fluidNodeCount = 0;

    // Fluid fragment cells and fluid-fluid faces
    std::vector<CFDFaceGpu> faces;
    std::vector<uint32_t> faceGlobalIndex;
    std::vector<CFDBoundaryFaceGpu> boundaryFaces;
    std::vector<uint32_t> boundaryGlobalIndex;
    std::vector<float> cellVolumes;
    std::vector<uint32_t> component;
    uint32_t componentCount = 0;
};

bool buildCFDTopology(
    const std::vector<topology::Fragment>& fragments,
    const std::vector<topology::Face>& faces,
    const std::vector<topology::BoundaryFace>& boundaryFaces,
    const std::vector<uint32_t>& globalFragmentToFluidNode,
    const std::vector<uint32_t>& fluidNodeToGlobalFragment,
    CFDTopology& outTopology);

} // namespace cfd
