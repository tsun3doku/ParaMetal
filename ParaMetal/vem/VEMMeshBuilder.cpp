#include "VEMMeshBuilder.hpp"

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

VEMMesh VEMMeshBuilder::buildFromGlobalDomain(const topology::GlobalDomain& domain) {
    VEMMesh mesh;
    const size_t numFrags = domain.fragments.size();
    const size_t numFaces = domain.faces.size();

    mesh.totalTraceCount = static_cast<uint32_t>(numFaces);
    mesh.faceSideToTrace.assign(numFaces * 2, UINT32_MAX);
    for (uint32_t face = 0; face < numFaces; ++face) {
        if (domain.faces[face].fragmentA < numFrags) {
            mesh.faceSideToTrace[face * 2] = face;
        }
        if (domain.faces[face].fragmentB < numFrags) {
            mesh.faceSideToTrace[face * 2 + 1] = face;
        }
    }

    std::vector<uint32_t> contactSolidFragment(numFaces, UINT32_MAX);
    for (const auto& coupling : domain.contactCouplings) {
        const uint32_t endpoints[2] = {coupling.faceA, coupling.faceB};
        const uint32_t fragments[2] = {coupling.fragmentA, coupling.fragmentB};
        for (int endpoint = 0; endpoint < 2; ++endpoint) {
            const uint32_t faceId = endpoints[endpoint];
            const uint32_t fragmentId = fragments[endpoint];
            if (faceId >= numFaces || fragmentId >= numFrags ||
                (domain.faces[faceId].fragmentA != fragmentId &&
                 domain.faces[faceId].fragmentB != fragmentId)) {
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
        const auto& face = domain.faces[faceId];
        if (face.fragmentA >= numFrags || face.fragmentB >= numFrags) continue;

        const uint32_t solidSide = face.fragmentA == contactFragment ? 0u : 1u;
        const uint32_t fluidSide = 1u - solidSide;
        const uint32_t solidTrace = mesh.faceSideToTrace[faceId * 2 + solidSide];
        const uint32_t otherTrace = mesh.totalTraceCount++;
        mesh.faceSideToTrace[faceId * 2 + fluidSide] = otherTrace;
        mesh.perfectContinuityPairs.push_back(
            (uint64_t(solidTrace) << 32) | uint64_t(otherTrace));
    }

    std::vector<uint32_t> fragFaceCount(numFrags, 0);
    for (const auto& face : domain.faces) {
        const bool hasA = face.fragmentA < numFrags;
        const bool hasB = face.fragmentB < numFrags;
        if (hasA) fragFaceCount[face.fragmentA]++;
        if (hasB) fragFaceCount[face.fragmentB]++;
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

        if (face.fragmentA < numFrags) {
            const uint32_t dst = currentFragOffset[face.fragmentA]++;
            FaceGeometry& geometry = mesh.faceIncidences[dst];
            geometry.traceIndex = mesh.faceSideToTrace[fIdx * 2];
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
            geometry.traceIndex = mesh.faceSideToTrace[fIdx * 2 + 1];
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
        uint32_t traceA = UINT32_MAX;
        uint32_t traceB = UINT32_MAX;
        if (coupling.faceA < numFaces) {
            const auto& face = domain.faces[coupling.faceA];
            if (face.fragmentA == coupling.fragmentA ||
                face.fragmentB == coupling.fragmentA) {
                const uint32_t side = face.fragmentA == coupling.fragmentA ? 0u : 1u;
                traceA = mesh.faceSideToTrace[coupling.faceA * 2 + side];
            }
        }
        if (coupling.faceB < numFaces) {
            const auto& face = domain.faces[coupling.faceB];
            if (face.fragmentA == coupling.fragmentB ||
                face.fragmentB == coupling.fragmentB) {
                const uint32_t side = face.fragmentA == coupling.fragmentB ? 0u : 1u;
                traceB = mesh.faceSideToTrace[coupling.faceB * 2 + side];
            }
        }
        if (traceA == UINT32_MAX || traceB == UINT32_MAX) {
            std::cerr << "[VEMMeshBuilder] invalid contact side reference "
                      << coupling.faceA << ':' << coupling.fragmentA
                      << " -> " << coupling.faceB << ':' << coupling.fragmentB
                      << std::endl;
            continue;
        }
        TraceCoupling term{};
        term.traceA = traceA;
        term.traceB = traceB;
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
