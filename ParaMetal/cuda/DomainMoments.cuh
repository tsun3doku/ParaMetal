#pragma once

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

#include "../voronoi/GlobalDomainTopology.hpp"

namespace domainMoments {

bool buildCellMoments(
    const topology::GlobalDomain& domain,
    uint32_t cellCount,
    const std::vector<uint32_t>& cellFaceOffsets,
    const std::vector<uint32_t>& cellFaces,
    const std::vector<double>& cellFaceSides,
    const std::vector<glm::dvec3>& cellOrigins,
    std::vector<double>& outMoments);

} // namespace domainMoments
