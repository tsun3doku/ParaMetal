#include "VEMMeshBuilder.hpp"

#include <cmath>
#include <iostream>

namespace vem {

bool validateMesh(
    const VEMMesh& mesh,
    const std::vector<float>& cellConductivity,
    std::string* reason) {
    if (mesh.cells.empty() || mesh.faceIncidences.empty() ||
        cellConductivity.size() != mesh.cells.size()) {
        if (reason) *reason = "mesh arrays have inconsistent sizes";
        return false;
    }

    for (uint32_t cellIndex = 0; cellIndex < mesh.cells.size(); ++cellIndex) {
        const CellGeometry& cell = mesh.cells[cellIndex];
        if (VEMDiffusion::validateElementGeometry(
                mesh.faceIncidences, cell, cellConductivity[cellIndex])) {
            continue;
        }

        std::cerr << "[VEMMesh] invalid cell=" << cellIndex
                  << " owner=" << cell.ownerModelId
                  << " seed=" << cell.seedId
                  << " volume=" << cell.volume
                  << " faces=" << cell.faceCount << std::endl;
        if (reason) {
            *reason = "VEM operator validation failed for cell " +
                std::to_string(cellIndex);
        }
        return false;
    }
    return true;
}

VEMMesh VEMMeshBuilder::buildFromArrays(
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
    const std::vector<double>& contactAreas) {
    VEMMesh mesh;
    const size_t numFrags = fragmentInstanceIds.size();
    const size_t numFaces = faceFragmentA.size();
    if (fragmentSeedIds.size() != numFrags || fragmentCentroids.size() != numFrags ||
        fragmentVolumes.size() != numFrags || faceFragmentB.size() != numFaces ||
        faceAreasA.size() != numFaces || faceAreasB.size() != numFaces ||
        faceAreaVectorsA.size() != numFaces || faceAreaVectorsB.size() != numFaces ||
        faceMomentsA.size() != numFaces || faceMomentsB.size() != numFaces ||
        faceTypes.size() != numFaces || faceDomainFaces.size() != numFaces ||
        contactFaceA.size() != contactFaceB.size() ||
        contactFaceA.size() != contactFragmentA.size() ||
        contactFaceA.size() != contactFragmentB.size() ||
        contactAreas.size() != contactFaceA.size()) {
        std::cerr << "[VEMMeshBuilder] flat topology input sizes are inconsistent" << std::endl;
        return {};
    }
    for (size_t coupling = 0; coupling < contactFaceA.size(); ++coupling) {
        if (!std::isfinite(contactAreas[coupling]) || !(contactAreas[coupling] > 0.0)) {
            std::cerr << "[VEMMeshBuilder] invalid contact area on coupling "
                      << coupling << std::endl;
            return {};
        }
    }

    mesh.totalTraceCount = static_cast<uint32_t>(numFaces);
    mesh.faceSideToTrace.assign(numFaces * 2, UINT32_MAX);
    for (uint32_t face = 0; face < numFaces; ++face) {
        if (faceFragmentA[face] < numFrags) {
            mesh.faceSideToTrace[face * 2] = face;
        }
        if (faceFragmentB[face] < numFrags) {
            mesh.faceSideToTrace[face * 2 + 1] = face;
        }
    }

    std::vector<uint32_t> contactSolidFragment(numFaces, UINT32_MAX);
    const size_t couplingCount = contactFaceA.size();
    for (size_t coupling = 0; coupling < couplingCount; ++coupling) {
        const uint32_t endpoints[2] = { contactFaceA[coupling], contactFaceB[coupling] };
        const uint32_t fragments[2] = { contactFragmentA[coupling], contactFragmentB[coupling] };
        for (int endpoint = 0; endpoint < 2; ++endpoint) {
            const uint32_t faceId = endpoints[endpoint];
            const uint32_t fragmentId = fragments[endpoint];
            if (faceId >= numFaces || fragmentId >= numFrags ||
                (faceFragmentA[faceId] != fragmentId && faceFragmentB[faceId] != fragmentId)) {
                std::cerr << "[VEMMeshBuilder] invalid contact incidence "
                          << faceId << ':' << fragmentId << std::endl;
                return {};
            }
            if (contactSolidFragment[faceId] != UINT32_MAX &&
                contactSolidFragment[faceId] != fragmentId) {
                std::cerr << "[VEMMeshBuilder] contact references both sides "
                          << "of physical face " << faceId << std::endl;
                return {};
            }
            contactSolidFragment[faceId] = fragmentId;
        }
    }

    for (uint32_t faceId = 0; faceId < numFaces; ++faceId) {
        const uint32_t contactFragment = contactSolidFragment[faceId];
        if (contactFragment == UINT32_MAX) continue;
        if (faceFragmentA[faceId] >= numFrags || faceFragmentB[faceId] >= numFrags) continue;

        const uint32_t solidSide = faceFragmentA[faceId] == contactFragment ? 0u : 1u;
        const uint32_t fluidSide = 1u - solidSide;
        const uint32_t solidTrace = mesh.faceSideToTrace[faceId * 2 + solidSide];
        const uint32_t otherTrace = mesh.totalTraceCount++;
        mesh.faceSideToTrace[faceId * 2 + fluidSide] = otherTrace;
        mesh.perfectContinuityPairs.push_back(
            (uint64_t(solidTrace) << 32) | uint64_t(otherTrace));
    }

    std::vector<uint32_t> fragFaceCount(numFrags, 0);
    for (uint32_t face = 0; face < numFaces; ++face) {
        if (faceFragmentA[face] < numFrags) fragFaceCount[faceFragmentA[face]]++;
        if (faceFragmentB[face] < numFrags) fragFaceCount[faceFragmentB[face]]++;
    }

    std::vector<uint32_t> fragFaceOffsets(numFrags + 1, 0);
    size_t totalFaces = 0;
    for (size_t i = 0; i < numFrags; ++i) {
        fragFaceOffsets[i] = static_cast<uint32_t>(totalFaces);
        totalFaces += fragFaceCount[i];
    }
    fragFaceOffsets[numFrags] = static_cast<uint32_t>(totalFaces);

    mesh.faceIncidences.resize(totalFaces);
    std::vector<uint32_t> currentFragOffset = fragFaceOffsets;

    for (uint32_t fIdx = 0; fIdx < numFaces; ++fIdx) {
        if (faceFragmentA[fIdx] < numFrags) {
            const uint32_t dst = currentFragOffset[faceFragmentA[fIdx]]++;
            FaceGeometry& geometry = mesh.faceIncidences[dst];
            geometry.traceIndex = mesh.faceSideToTrace[fIdx * 2];
            geometry.neighborCellId = faceFragmentB[fIdx] < numFrags ? faceFragmentB[fIdx] : UINT32_MAX;
            if (geometry.neighborCellId < numFrags) {
                geometry.neighborOwnerModelId = fragmentInstanceIds[geometry.neighborCellId];
                geometry.neighborSeedId = fragmentSeedIds[geometry.neighborCellId];
            }
            geometry.boundaryCondition = (faceTypes[fIdx] == 1u) ? faceDomainFaces[fIdx] : UINT32_MAX;
            geometry.area = faceAreasA[fIdx];
            geometry.areaVector = faceAreaVectorsA[fIdx];
            geometry.moment = faceMomentsA[fIdx];
        }

        if (faceFragmentB[fIdx] < numFrags) {
            const uint32_t dst = currentFragOffset[faceFragmentB[fIdx]]++;
            FaceGeometry& geometry = mesh.faceIncidences[dst];
            geometry.traceIndex = mesh.faceSideToTrace[fIdx * 2 + 1];
            geometry.neighborCellId = faceFragmentA[fIdx] < numFrags ? faceFragmentA[fIdx] : UINT32_MAX;
            if (geometry.neighborCellId < numFrags) {
                geometry.neighborOwnerModelId = fragmentInstanceIds[geometry.neighborCellId];
                geometry.neighborSeedId = fragmentSeedIds[geometry.neighborCellId];
            }
            geometry.boundaryCondition = UINT32_MAX;
            geometry.area = faceAreasB[fIdx];
            geometry.areaVector = faceAreaVectorsB[fIdx];
            geometry.moment = faceMomentsB[fIdx];
        }
    }

    mesh.traceCouplings.reserve(couplingCount);
    for (size_t coupling = 0; coupling < couplingCount; ++coupling) {
        uint32_t traceA = UINT32_MAX;
        uint32_t traceB = UINT32_MAX;
        if (contactFaceA[coupling] < numFaces) {
            const uint32_t faceId = contactFaceA[coupling];
            const uint32_t fragmentId = contactFragmentA[coupling];
            if (faceFragmentA[faceId] == fragmentId || faceFragmentB[faceId] == fragmentId) {
                const uint32_t side = faceFragmentA[faceId] == fragmentId ? 0u : 1u;
                traceA = mesh.faceSideToTrace[faceId * 2 + side];
            }
        }
        if (contactFaceB[coupling] < numFaces) {
            const uint32_t faceId = contactFaceB[coupling];
            const uint32_t fragmentId = contactFragmentB[coupling];
            if (faceFragmentA[faceId] == fragmentId || faceFragmentB[faceId] == fragmentId) {
                const uint32_t side = faceFragmentA[faceId] == fragmentId ? 0u : 1u;
                traceB = mesh.faceSideToTrace[faceId * 2 + side];
            }
        }
        if (traceA == UINT32_MAX || traceB == UINT32_MAX) {
            std::cerr << "[VEMMeshBuilder] invalid contact side reference "
                      << contactFaceA[coupling] << ':' << contactFragmentA[coupling]
                      << " -> " << contactFaceB[coupling] << ':' << contactFragmentB[coupling]
                      << std::endl;
            return {};
        }
        TraceCoupling term{};
        term.traceA = traceA;
        term.traceB = traceB;
        term.area = static_cast<float>(contactAreas[coupling]);
        term.coefficientPerArea = 0.0f;
        mesh.traceCouplings.push_back(term);
    }

    mesh.cells.resize(numFrags);
    for (uint32_t c = 0; c < numFrags; ++c) {
        CellGeometry& cell = mesh.cells[c];
        cell.ownerModelId = fragmentInstanceIds[c];
        cell.seedId = fragmentSeedIds[c];
        cell.volume = fragmentVolumes[c];
        cell.centroid = fragmentCentroids[c];
        cell.faceOffset = fragFaceOffsets[c];
        cell.faceCount = fragFaceCount[c];
    }

    return mesh;
}

} // namespace vem
