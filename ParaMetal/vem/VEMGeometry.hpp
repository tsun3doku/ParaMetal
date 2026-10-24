#pragma once

#include <cstdint>
#include <glm/glm.hpp>

namespace vem {

struct FaceGeometry {
    uint32_t traceIndex = 0;
    uint32_t neighborCellId = UINT32_MAX;
    uint32_t neighborOwnerModelId = UINT32_MAX;
    uint32_t neighborSeedId = UINT32_MAX;
    uint32_t boundaryCondition = UINT32_MAX;
    double area = 0.0;
    glm::dvec3 areaVector{0.0};
    glm::dvec3 moment{0.0};
};

struct CellGeometry {
    uint32_t faceOffset = 0;
    uint32_t faceCount = 0;
    uint32_t ownerModelId = UINT32_MAX;
    uint32_t seedId = UINT32_MAX;
    double volume = 0.0;
    glm::dvec3 centroid{0.0};
};

} // namespace vem
