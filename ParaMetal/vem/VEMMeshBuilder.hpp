#pragma once

#include "VEMGeometry.hpp"
#include "VEMDiffusion.hpp"
#include "../voronoi/GlobalDomainTopology.hpp"

#include <vector>
#include <cstdint>
#include <string>

namespace vem {

struct VEMMesh {
    std::vector<FaceGeometry> faceIncidences;
    std::vector<CellGeometry> cells;
    std::vector<TraceCoupling> traceCouplings;
    std::vector<uint32_t> faceSideToTrace;
    std::vector<uint64_t> perfectContinuityPairs;
    uint32_t totalTraceCount = 0;
};

bool validateMesh(
    const VEMMesh& mesh,
    const std::vector<float>& cellConductivity,
    std::string* reason = nullptr);

class VEMMeshBuilder {
public:
    static VEMMesh buildFromGlobalDomain(const topology::GlobalDomain& domain);
};

} // namespace vem
