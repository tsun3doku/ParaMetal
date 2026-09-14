#include "RVDFragments.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>

namespace voronoi {

constexpr double FaceAreaClosureFraction = 1e-3;

static bool appendFace(
    RVDResult& result,
    const BaseFace& base,
    uint32_t fragA,
    uint32_t fragB,
    double area) {

    if (fragA == InvalidId || (base.cellB != InvalidId && fragB == InvalidId)) return false;

    FragmentFace face{};
    face.fragmentA = fragA;
    face.fragmentB = fragB;
    face.area = area;

    const glm::dvec3 n = base.normal;
    const glm::dvec3 facePt = base.point;

    const glm::dvec3 cA(result.fragments[fragA].centroid);
    face.orthogonalDistanceA = std::abs(glm::dot(n, cA - facePt));

    if (base.cellB != InvalidId) {
        face.source = FragmentFaceSource::VoronoiInterface;
        const glm::dvec3 cB(result.fragments[fragB].centroid);
        face.orthogonalDistanceB = std::abs(glm::dot(n, cB - facePt));
        face.normal = glm::vec3(-n); 
    } else {
        face.source = FragmentFaceSource::DomainBoundary;
        face.domainFace = base.domainFace;
        face.orthogonalDistanceB = 0.0;
        face.normal = glm::vec3(n); 
    }

    result.fragmentFaces.push_back(face);
    return true;
}

bool buildFragments(
    const RVDInput& input,
    const BaseCellsContext& baseCtx,
    const std::vector<ModelRestrictionData>& models,
    std::vector<std::vector<uint32_t>>& fragmentIdByModelCell,
    std::vector<uint32_t>& unownedFragmentIdByCell,
    RVDResult& result) {

    const size_t numGeometries = models.size();
    fragmentIdByModelCell.assign(numGeometries, std::vector<uint32_t>(baseCtx.count, InvalidId));
    unownedFragmentIdByCell.assign(baseCtx.count, InvalidId);

    const double volumeTolerance = std::max(1e-18, double(input.nominalCellSize * input.nominalCellSize * input.nominalCellSize) * 1e-9);

    for (uint32_t cell = 0; cell < baseCtx.count; ++cell) {
        const auto& baseCell = baseCtx.baseCells[cell];
        const glm::dvec3 seedPos((*input.seeds)[cell]);

        double totalModelVol = 0.0;
        glm::dvec3 totalModelMom(0.0);

        for (size_t g = 0; g < numGeometries; ++g) {
            const auto& mCell = models[g].cells[cell];
            if (mCell.volume > volumeTolerance) {
                Fragment frag{};
                frag.cellId = cell;
                frag.ownerModelId = (*input.geometries)[g].runtimeModelId;
                frag.volume = mCell.volume;
                frag.exposedSurfaceArea = 0.0;
                frag.centroid = glm::vec3(seedPos + mCell.localMoment / mCell.volume);

                const uint32_t fragId = uint32_t(result.fragments.size());
                fragmentIdByModelCell[g][cell] = fragId;
                result.fragments.push_back(frag);

                totalModelVol += mCell.volume;
                totalModelMom += mCell.localMoment;
            }
        }

        const double volumeClosureTolerance = std::max(
            double(input.nominalCellSize * input.nominalCellSize * input.nominalCellSize) * 1e-6,
            baseCell.volume * 1e-4);

        const double rawUnownedVol = baseCell.volume - totalModelVol;
        if (rawUnownedVol < -volumeClosureTolerance) {
            std::cerr << "[RVD-Invariant] model volume exceeds base cell cell=" << cell
                      << " modelVolume=" << totalModelVol << " baseVolume=" << baseCell.volume
                      << " remainder=" << rawUnownedVol << std::endl;
            return false;
        }

        const double unownedVol = std::max(0.0, rawUnownedVol);
        if (unownedVol > volumeClosureTolerance) {
            const glm::dvec3 remainderMoment = baseCell.localMoment - totalModelMom;
            const glm::dvec3 remainderCentroid = seedPos + remainderMoment / unownedVol;

            Fragment frag{};
            frag.cellId = cell;
            frag.ownerModelId = InvalidModelId;
            frag.volume = unownedVol;
            frag.exposedSurfaceArea = 0.0;
            frag.centroid = glm::vec3(remainderCentroid);

            const uint32_t fragId = uint32_t(result.fragments.size());
            unownedFragmentIdByCell[cell] = fragId;
            result.fragments.push_back(frag);
        }
    }

    for (size_t g = 0; g < numGeometries; ++g) {
        const auto& model = models[g];
        for (uint32_t local = 0; local < model.totalPatches; ++local) {
            const uint32_t surfIdx = model.surfaceBase + local;
            if (surfIdx >= result.surfaces.size() || surfIdx >= result.surfaceExposure.size()) continue;
            const auto& surface = result.surfaces[surfIdx];
            const uint32_t cell = surface.cellId;
            if (cell >= baseCtx.count) continue;
            const uint32_t solidFrag = fragmentIdByModelCell[g][cell];
            if (solidFrag != InvalidId && solidFrag < result.fragments.size()) {
                result.fragments[solidFrag].exposedSurfaceArea += result.surfaceExposure[surfIdx].area;
            }
        }
    }

    return true;
}

bool buildFragmentFaces(
    const BaseCellsContext& baseCtx,
    const TopologyContext& topoCtx,
    const std::vector<ModelRestrictionData>& models,
    const std::vector<std::vector<uint32_t>>& fragmentIdByModelCell,
    const std::vector<uint32_t>& unownedFragmentIdByCell,
    RVDResult& result) {

    for (size_t faceId = 0; faceId < topoCtx.baseFaces.size(); ++faceId) {
        const BaseFace& face = topoCtx.baseFaces[faceId];
        double solidAreaSum = 0.0;

        for (size_t g = 0; g < models.size(); ++g) {
            const double area = models[g].faceAreas[faceId];
            if (area <= baseCtx.areaTolerance) continue;

            const uint32_t fragA = fragmentIdByModelCell[g][face.cellA];
            const uint32_t fragB = face.cellB == InvalidId
                ? InvalidId : fragmentIdByModelCell[g][face.cellB];

            const double closureTolerance = std::max(baseCtx.areaTolerance, face.area * FaceAreaClosureFraction);

            if (fragA == InvalidId || (face.cellB != InvalidId && fragB == InvalidId)) {
                if (area <= closureTolerance) continue;
                std::cerr << "[RVD-Invariant] solid face missing fragment face=" << faceId
                          << " modelId=" << models[g].runtimeModelId
                          << " area=" << area << " tol=" << closureTolerance
                          << " baseFaceArea=" << face.area
                          << " cellA=" << face.cellA << " fragA=" << fragA << " volA=" << models[g].cells[face.cellA].volume
                          << " cellB=" << face.cellB << " fragB=" << fragB << " volB=" << (face.cellB != InvalidId ? models[g].cells[face.cellB].volume : -1.0)
                          << std::endl;
                return false;
            }

            if (!appendFace(result, face, fragA, fragB, area)) {
                return false;
            }
            solidAreaSum += area;
        }

        const double closureTolerance = std::max(baseCtx.areaTolerance, face.area * FaceAreaClosureFraction);
        if (solidAreaSum > face.area + closureTolerance) {
            std::cerr << "[RVD-Invariant] solid areas exceed base face=" << faceId
                      << " solidSum=" << solidAreaSum << " baseArea=" << face.area << std::endl;
            return false;
        }

        const uint32_t fragA = unownedFragmentIdByCell[face.cellA];
        const uint32_t fragB = face.cellB == InvalidId
            ? InvalidId : unownedFragmentIdByCell[face.cellB];

        if (fragA == InvalidId || (face.cellB != InvalidId && fragB == InvalidId)) {
            continue;
        }

        const double fluidArea = std::max(0.0, face.area - solidAreaSum);
        if (fluidArea <= baseCtx.areaTolerance) continue;

        if (!appendFace(result, face, fragA, fragB, fluidArea)) {
            return false;
        }
    }

    const double invScale2 = baseCtx.invBaseScale * baseCtx.invBaseScale;

    for (size_t g = 0; g < models.size(); ++g) {
        const auto& model = models[g];
        for (uint32_t local = 0; local < model.totalPatches; ++local) {
            const uint32_t surfIdx = model.surfaceBase + local;
            if (surfIdx >= result.surfaces.size()) continue;

            const auto& surface = result.surfaces[surfIdx];
            if (surface.vertexCount < 3) continue;

            const uint32_t cell = surface.cellId;
            if (cell >= baseCtx.count) continue;

            const uint32_t fluidFrag = unownedFragmentIdByCell[cell];
            if (fluidFrag == InvalidId) continue;

            const uint32_t solidFrag = fragmentIdByModelCell[g][cell];
            if (solidFrag == InvalidId) continue;

            const auto& geom = model.surfaceGeometry[local];
            const double canonicalArea = double(geom.centroidArea.w);
            const double totalPolygonArea = canonicalArea * invScale2;
            if (totalPolygonArea <= baseCtx.areaTolerance) continue;

            glm::dvec3 normal(geom.plane.x, geom.plane.y, geom.plane.z);
            const double normalLen = glm::length(normal);
            if (normalLen <= 1e-14 || !std::isfinite(normalLen)) continue;
            normal /= normalLen;

            const auto& exposure = (surfIdx < result.surfaceExposure.size())
                ? result.surfaceExposure[surfIdx]
                : SurfaceMeasure{};

            const double Aexposed = exposure.area;
            if (Aexposed <= baseCtx.areaTolerance) continue;

            const glm::dvec3 Cexposed = exposure.moment / Aexposed;

            const glm::dvec3 cSolid(result.fragments[solidFrag].centroid);
            const glm::dvec3 cFluid(result.fragments[fluidFrag].centroid);

            if (glm::dot(normal, cFluid - cSolid) < 0.0) {
                normal = -normal;
            }

            const double dSolid = std::abs(glm::dot(normal, cSolid - Cexposed));
            const double dFluid = std::abs(glm::dot(normal, cFluid - Cexposed));

            FragmentFace face{};
            face.fragmentA = solidFrag;
            face.fragmentB = fluidFrag;
            face.normal = glm::vec3(normal);
            face.area = Aexposed;
            face.orthogonalDistanceA = dSolid;
            face.orthogonalDistanceB = dFluid;
            face.source = FragmentFaceSource::ModelSurface;
            face.domainFace = InvalidId;

            result.fragmentFaces.push_back(face);
        }
    }

    return true;
}

} // namespace voronoi
