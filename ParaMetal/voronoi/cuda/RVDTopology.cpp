#include "RVDTopology.hpp"
#include "ConvexCell.cuh"

#include <iostream>
#include <cmath>

namespace voronoi {

bool buildBaseFaces(
    const RVDInput& input,
    const BaseCellsContext& baseCtx,
    TopologyContext& topoCtx) {

    const uint32_t count = baseCtx.count;
    if (baseCtx.cellPlaneOffsets.empty() || baseCtx.cellPlaneOffsets.back() == 0) return true;

    topoCtx.voronoiFaceIdByPlane.assign(baseCtx.cellPlaneOffsets[count], InvalidId);
    topoCtx.baseFaces.clear();

    // Process Domain Faces 
    for (uint32_t cellA = 0; cellA < count; ++cellA) {
        const uint32_t countA = baseCtx.planeCount(cellA);

        for (uint32_t localPlaneA = 0; localPlaneA < countA; ++localPlaneA) {
            const int tag = baseCtx.planeNeighbor(cellA, localPlaneA);
            if (tag >= 0) continue;

            const int domainFace = rvdDevice::domainFaceFromTag(tag);
            if (domainFace >= 0 && domainFace < 6) {
                const double area = baseCtx.baseFaceArea(cellA, localPlaneA);
                if (!std::isfinite(area)) {
                    std::cerr << "[RVD-Invariant] non-finite domain face area cell=" << cellA
                              << " domainFace=" << domainFace << std::endl;
                    return false;
                }
                if (area <= baseCtx.areaTolerance) continue;

                const uint32_t slotA = baseCtx.cellPlaneOffsets[cellA] + localPlaneA;
                const glm::dvec4 plane = baseCtx.outwardDomainPlanes[domainFace];
                const glm::dvec3 n(plane.x, plane.y, plane.z);
                const glm::dvec3 seedPos((*input.seeds)[cellA]);
                const double planeDist = -(glm::dot(n, seedPos) + plane.w);
                const glm::dvec3 planePoint = seedPos + n * planeDist;

                BaseFace face{};
                face.cellA = cellA;
                face.cellB = InvalidId;
                face.domainFace = uint32_t(domainFace);
                face.slotA = slotA;
                face.normal = n;
                face.point = planePoint;
                face.area = area;

                const uint32_t faceId = uint32_t(topoCtx.baseFaces.size());
                topoCtx.baseFaces.push_back(face);
                topoCtx.voronoiFaceIdByPlane[slotA] = faceId;
                continue;
            }

            std::cerr << "[RVD-Invariant] base plane has no valid neighbor cell=" << cellA << " tag=" << tag << std::endl;
            return false;
        }
    }

    // Process voronoi interfaces 
    for (uint32_t cellA = 0; cellA < count; ++cellA) {
        const uint32_t countA = baseCtx.planeCount(cellA);

        for (uint32_t localPlaneA = 0; localPlaneA < countA; ++localPlaneA) {
            const int tag = baseCtx.planeNeighbor(cellA, localPlaneA);
            if (tag < 0) continue;

            const uint32_t cellB = uint32_t(tag);
            if (cellB >= count || cellB == cellA) {
                std::cerr << "[RVD-Invariant] invalid Voronoi neighbor cells=("
                          << cellA << "," << cellB << ")" << std::endl;
                return false;
            }

            const uint32_t slotA = baseCtx.cellPlaneOffsets[cellA] + localPlaneA;
            if (topoCtx.voronoiFaceIdByPlane[slotA] != InvalidId) {
                continue;
            }

            const double areaA = baseCtx.baseFaceArea(cellA, localPlaneA);
            if (!std::isfinite(areaA)) {
                std::cerr << "[RVD-Invariant] non-finite Voronoi face area cells=("
                          << cellA << "," << cellB << ")" << std::endl;
                return false;
            }

            if (areaA <= baseCtx.areaTolerance) {
                continue;
            }

            // Canonical pair (u, v) with u < v
            const uint32_t u = std::min(cellA, cellB);
            const uint32_t v = std::max(cellA, cellB);

            // Find matching plane in cellB pointing back to cellA
            uint32_t slotB = InvalidId;
            double areaB = 0.0;
            const uint32_t countB = baseCtx.planeCount(cellB);

            for (uint32_t localPlaneB = 0; localPlaneB < countB; ++localPlaneB) {
                if (baseCtx.planeNeighbor(cellB, localPlaneB) == int(cellA)) {
                    slotB = baseCtx.cellPlaneOffsets[cellB] + localPlaneB;
                    areaB = baseCtx.baseFaceArea(cellB, localPlaneB);
                    break;
                }
            }

            // Require valid reciprocal face on both sides
            if (slotB == InvalidId || !std::isfinite(areaB) || areaB <= baseCtx.areaTolerance) {
                continue;
            }

            const double areaDiff = std::abs(areaA - areaB);
            const double maxArea = std::max(areaA, areaB);
            const double closureTolerance = std::max(baseCtx.areaTolerance * 10.0, maxArea * 0.10);

            if (areaDiff > closureTolerance) {
                continue;
            }

            const double faceArea = 0.5 * (areaA + areaB);

            const glm::dvec3 pA((*input.seeds)[u]);
            const glm::dvec3 pB((*input.seeds)[v]);
            const glm::dvec3 dir = pA - pB;
            const double dist = glm::length(dir);
            if (dist <= 1e-12 || !std::isfinite(dist)) {
                std::cerr << "[RVD-Invariant] degenerate seed distance cells=("
                          << u << "," << v << ")" << std::endl;
                return false;
            }

            const glm::dvec3 n = dir / dist;
            const glm::dvec3 midpoint = 0.5 * (pA + pB);
            const glm::dvec4 plane(n, -glm::dot(n, midpoint));

            const uint32_t slotU = (cellA == u) ? slotA : slotB;
            const uint32_t slotV = (cellA == u) ? slotB : slotA;

            BaseFace face{};
            face.cellA = u;
            face.cellB = v;
            face.domainFace = InvalidId;
            face.slotA = slotU;
            face.normal = n;
            face.point = midpoint;
            face.area = faceArea;

            const uint32_t faceId = uint32_t(topoCtx.baseFaces.size());
            topoCtx.baseFaces.push_back(face);

            if (slotU != InvalidId) topoCtx.voronoiFaceIdByPlane[slotU] = faceId;
            if (slotV != InvalidId) topoCtx.voronoiFaceIdByPlane[slotV] = faceId;
        }
    }

    return true;
}

} // namespace voronoi
