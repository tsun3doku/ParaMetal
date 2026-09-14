#pragma once

#include "RVD.hpp"
#include "RVDBaseCells.cuh"

#include <vector>
#include <cstdint>
#include <glm/glm.hpp>

namespace voronoi {

struct BaseFace {
    uint32_t cellA = InvalidId;     // min(cellA, cellB)
    uint32_t cellB = InvalidId;     // max(cellA, cellB) 
    uint32_t domainFace = InvalidId; 

    uint32_t slotA = InvalidId;     // Plane slot on cellA

    glm::dvec3 normal{0.0};
    glm::dvec3 point{0.0};
    double area = 0.0;
};

struct TopologyContext {
    std::vector<BaseFace> baseFaces;
    std::vector<uint32_t> voronoiFaceIdByPlane;
};

bool buildBaseFaces(
    const RVDInput& input,
    const BaseCellsContext& baseCtx,
    TopologyContext& topoCtx);

} // namespace voronoi
