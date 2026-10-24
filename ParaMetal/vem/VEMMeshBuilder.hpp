#pragma once

#include "VEMGeometry.hpp"
#include "VEMDiffusion.hpp"
#include "../voronoi/GlobalDomainTopology.hpp"

#include <vector>
#include <cstdint>

namespace vem {

struct VEMMesh {
    std::vector<FaceGeometry> faceIncidences;
    std::vector<CellGeometry> cells;
    std::vector<TraceCoupling> traceCouplings;
    uint32_t totalTraceCount = 0;
};

class VEMMeshBuilder {
public:
    static VEMMesh buildFromGlobalDomain(const topology::GlobalDomain& domain);
};

} // namespace vem
