#pragma once

#include "VEMGeometry.hpp"
#include "VEMDiffusion.hpp"

#include <vector>
#include <cstdint>
#include <string>
#include <glm/glm.hpp>

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
    static VEMMesh buildFromArrays(
        const std::vector<uint32_t>& fragmentInstanceIds,
        const std::vector<uint32_t>& fragmentSeedIds,
        const std::vector<glm::dvec3>& fragmentCentroids,
        const std::vector<double>& fragmentVolumes,
        const std::vector<uint32_t>& faceFragmentA,
        const std::vector<uint32_t>& faceFragmentB,
        const std::vector<double>& faceAreasA,
        const std::vector<double>& faceAreasB,
        const std::vector<glm::dvec3>& faceAreaVectorsA,
        const std::vector<glm::dvec3>& faceAreaVectorsB,
        const std::vector<glm::dvec3>& faceMomentsA,
        const std::vector<glm::dvec3>& faceMomentsB,
        const std::vector<uint8_t>& faceTypes,
        const std::vector<uint32_t>& faceDomainFaces,
        const std::vector<uint32_t>& contactFaceA,
        const std::vector<uint32_t>& contactFragmentA,
        const std::vector<uint32_t>& contactFaceB,
        const std::vector<uint32_t>& contactFragmentB,
        const std::vector<double>& contactAreas);
};

} // namespace vem
