#include "VEMMeshBuilder.hpp"

#include <iostream>

namespace vem {

VEMMesh VEMMeshBuilder::buildFromGlobalDomain(const topology::GlobalDomain& domain) {
    VEMMesh mesh;
    const size_t numFrags = domain.fragments.size();
    const size_t numFaces = domain.faces.size();

    mesh.totalTraceCount = static_cast<uint32_t>(numFaces);

    std::vector<uint32_t> fragFaceCount(numFrags, 0);
    uint32_t zeroIncidenceTraces = 0;
    uint32_t oneSidedTraces = 0;
    for (const auto& face : domain.faces) {
        const bool hasA = face.fragmentA < numFrags;
        const bool hasB = face.fragmentB < numFrags;
        if (hasA) fragFaceCount[face.fragmentA]++;
        if (hasB) fragFaceCount[face.fragmentB]++;
        if (!hasA && !hasB) ++zeroIncidenceTraces;
        else if (!hasA || !hasB) ++oneSidedTraces;
    }

    if (zeroIncidenceTraces > 0) {
        std::cerr << "[VEMMeshBuilder] zero-incidence traces=" << zeroIncidenceTraces
                  << " one-sided=" << oneSidedTraces
                  << " fragments=" << numFrags
                  << " traces=" << numFaces << std::endl;
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
        const auto& face = domain.faces[fIdx];
        const uint32_t trace = fIdx;

        if (face.fragmentA < numFrags) {
            const uint32_t dst = currentFragOffset[face.fragmentA]++;
            FaceGeometry& geometry = mesh.faceIncidences[dst];
            geometry.traceIndex = trace;
            geometry.neighborCellId = face.fragmentB < numFrags ? face.fragmentB : UINT32_MAX;
            if (geometry.neighborCellId < numFrags) {
                geometry.neighborOwnerModelId = domain.fragments[geometry.neighborCellId].instanceId;
                geometry.neighborSeedId = domain.fragments[geometry.neighborCellId].seedId;
            }
            geometry.boundaryCondition = (face.type == topology::FaceType::DomainBoundary) ? face.domainFace : UINT32_MAX;
            geometry.area = face.areaA;
            geometry.areaVector = face.areaVectorA;
            geometry.moment = face.momentA;
        }

        if (face.fragmentB < numFrags) {
            const uint32_t dst = currentFragOffset[face.fragmentB]++;
            FaceGeometry& geometry = mesh.faceIncidences[dst];
            geometry.traceIndex = trace;
            geometry.neighborCellId = face.fragmentA < numFrags ? face.fragmentA : UINT32_MAX;
            if (geometry.neighborCellId < numFrags) {
                geometry.neighborOwnerModelId = domain.fragments[geometry.neighborCellId].instanceId;
                geometry.neighborSeedId = domain.fragments[geometry.neighborCellId].seedId;
            }
            geometry.boundaryCondition = UINT32_MAX;
            geometry.area = face.areaB;
            geometry.areaVector = face.areaVectorB;
            geometry.moment = face.momentB;
        }
    }

    mesh.traceCouplings.reserve(domain.contactCouplings.size());
    for (const auto& coupling : domain.contactCouplings) {
        TraceCoupling term{};
        term.traceA = coupling.faceA;
        term.traceB = coupling.faceB;
        term.area = static_cast<float>(coupling.couplingArea);
        term.coefficientPerArea = 0.0f;
        mesh.traceCouplings.push_back(term);
    }

    mesh.cells.resize(numFrags);
    for (uint32_t c = 0; c < numFrags; ++c) {
        CellGeometry& cell = mesh.cells[c];
        cell.ownerModelId = domain.fragments[c].instanceId;
        cell.seedId = domain.fragments[c].seedId;
        cell.volume = domain.fragments[c].volume;
        cell.centroid = domain.fragments[c].centroid;
        cell.faceOffset = fragFaceOffsets[c];
        cell.faceCount = fragFaceCount[c];
    }

    return mesh;
}

} // namespace vem
