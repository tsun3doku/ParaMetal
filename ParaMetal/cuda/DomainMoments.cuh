#pragma once

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

namespace domainMoments {

bool buildCellMoments(
    uint32_t faceCount,
    const std::vector<double>& faceAreasA,
    const std::vector<double>& faceAreasB,
    const std::vector<glm::dvec3>& faceAreaVectorsA,
    const std::vector<glm::dvec3>& faceAreaVectorsB,
    const std::vector<uint32_t>& faceLoopOffsets,
    const std::vector<uint32_t>& loopVertexOffsets,
    const std::vector<glm::dvec3>& loopVertices,
    uint32_t cellCount,
    const std::vector<uint32_t>& cellFaceOffsets,
    const std::vector<uint32_t>& cellFaces,
    const std::vector<double>& cellFaceSides,
    const std::vector<glm::dvec3>& cellOrigins,
    std::vector<double>& outMoments);

} // namespace domainMoments
