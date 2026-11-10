#pragma once

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>
#include "../voronoi/GlobalDomainTopology.hpp"
#include "CFDGpuStructs.hpp"

namespace cfd {

struct CFDTopology {
    uint32_t fluidNodeCount = 0;

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
