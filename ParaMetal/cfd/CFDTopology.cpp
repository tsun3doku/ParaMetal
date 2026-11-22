#include "CFDRuntime.hpp"
#include "CFDBoundaryConditions.hpp"
#include "util/GeometryUtils.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace cfd {

static constexpr uint32_t InvalidIndex = 0xffffffffu;
static constexpr uint8_t DomainBoundaryFaceType = 1u;
static constexpr uint8_t MaterialBoundaryFaceType = 2u;

static bool isMonotonicOffsets(const std::vector<uint32_t>& offsets, uint32_t lastValue) {
    if (offsets.empty() || offsets.front() != 0u || offsets.back() != lastValue) {
        return false;
    }
    for (size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] < offsets[i - 1]) {
            return false;
        }
    }
    return true;
}

void CFDRuntime::clearTopology() {
    outFaceTopology.clear();
    outFaceFluxVectors.clear();
    outFaceGlobalIndex.clear();
    outBoundaryFaces.clear();
    outBoundaryGlobalIndex.clear();
    outCellVolumes.clear();
    outCellCentroids.clear();
    outPressureComponents.clear();
    outPressureComponentCount = 0;
}

void CFDRuntime::releaseTopologyConstructionData() {
    std::vector<FaceTopology>().swap(outFaceTopology);
    std::vector<FaceFluxVectors>().swap(outFaceFluxVectors);
    std::vector<uint32_t>().swap(outFaceGlobalIndex);
    std::vector<CFDBoundaryFaceGpu>().swap(outBoundaryFaces);
    std::vector<uint32_t>().swap(outBoundaryGlobalIndex);
    std::vector<float>().swap(outCellVolumes);
    std::vector<glm::vec4>().swap(outCellCentroids);
    std::vector<uint32_t>().swap(outPressureComponents);
    outPressureComponentCount = 0;
}

bool CFDRuntime::buildTopology(
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
    const std::vector<uint32_t>& boundaryFaceIndices,
    const std::vector<uint32_t>& faceLoopOffsets,
    const std::vector<uint32_t>& loopVertexOffsets,
    const std::vector<glm::dvec3>& loopVertices,
    const std::vector<uint32_t>& globalFragmentToFluidNode,
    const std::vector<uint32_t>& fluidNodeToGlobalFragment) {

    clearTopology();

    struct RawFineFace {
        uint32_t globalFaceIndex = 0;
        uint32_t fineA = 0;
        uint32_t fineB = 0;
        glm::vec3 normal{0.0f};
        float area = 0.0f;
        float weight = 0.0f;
        float distA = 0.0f;
        float distB = 0.0f;
    };

    struct RawFineBoundaryFace {
        uint32_t globalBoundaryFaceIndex = UINT32_MAX;
        uint32_t fineNode = 0;
        uint32_t boundaryCondition = cfd::EmbeddedWallBoundary;
        glm::vec3 outwardNormal{0.0f};
        float area = 0.0f;
        float weight = 0.0f;
    };

    const size_t fragmentCount = fragmentInstanceIds.size();
    const size_t faceCount = faceFragmentA.size();
    if (fragmentCount == 0 || faceCount == 0) {
        return false;
    }
    if (fragmentSeedIds.size() != fragmentCount ||
        fragmentCentroids.size() != fragmentCount ||
        fragmentVolumes.size() != fragmentCount) {
        return false;
    }
    if (faceFragmentB.size() != faceCount ||
        faceAreasA.size() != faceCount ||
        faceAreasB.size() != faceCount ||
        faceAreaVectorsA.size() != faceCount ||
        faceAreaVectorsB.size() != faceCount ||
        faceMomentsA.size() != faceCount ||
        faceMomentsB.size() != faceCount ||
        faceTypes.size() != faceCount ||
        faceDomainFaces.size() != faceCount) {
        return false;
    }

    if (faceLoopOffsets.size() != faceCount + 1u ||
        !isMonotonicOffsets(faceLoopOffsets, faceLoopOffsets.back())) {
        return false;
    }
    const uint32_t loopCount = faceLoopOffsets.back();
    if (loopVertexOffsets.size() != size_t(loopCount) + 1u ||
        !isMonotonicOffsets(loopVertexOffsets, static_cast<uint32_t>(loopVertices.size()))) {
        return false;
    }
    for (uint32_t loop = 0; loop < loopCount; ++loop) {
        if (loopVertexOffsets[loop + 1u] < loopVertexOffsets[loop] + 3u) {
            return false;
        }
    }
    for (const glm::dvec3& vertex : loopVertices) {
        if (!GeometryUtils::isFinite(vertex)) {
            return false;
        }
    }

    for (size_t face = 0; face < faceCount; ++face) {
        if (faceFragmentA[face] != InvalidIndex && faceFragmentA[face] >= fragmentCount) {
            return false;
        }
        if (faceFragmentB[face] != InvalidIndex && faceFragmentB[face] >= fragmentCount) {
            return false;
        }
        if (!std::isfinite(faceAreasA[face]) ||
            !std::isfinite(faceAreasB[face]) ||
            !GeometryUtils::isFinite(faceAreaVectorsA[face]) ||
            !GeometryUtils::isFinite(faceAreaVectorsB[face]) ||
            !GeometryUtils::isFinite(faceMomentsA[face]) ||
            !GeometryUtils::isFinite(faceMomentsB[face])) {
            return false;
        }
        if (faceTypes[face] > MaterialBoundaryFaceType) {
            return false;
        }
        if (faceTypes[face] == DomainBoundaryFaceType &&
            faceDomainFaces[face] >= DomainBoundaryCount) {
            return false;
        }
    }

    size_t boundarySlot = 0;
    for (size_t face = 0; face < faceCount; ++face) {
        if (faceTypes[face] != DomainBoundaryFaceType) {
            continue;
        }
        if (boundarySlot >= boundaryFaceIndices.size() ||
            boundaryFaceIndices[boundarySlot] != face) {
            return false;
        }
        ++boundarySlot;
    }
    if (boundarySlot != boundaryFaceIndices.size()) {
        return false;
    }

    if (globalFragmentToFluidNode.size() != fragmentCount) {
        return false;
    }

    const uint32_t fineNodeCount = static_cast<uint32_t>(fluidNodeToGlobalFragment.size());
    if (fineNodeCount == 0) {
        return true;
    }

    std::vector<float> fineVolumes(fineNodeCount, 0.0f);
    std::unordered_map<uint32_t, float> seedVolume;
    for (uint32_t fragment = 0; fragment < fragmentCount; ++fragment) {
        const double volume = fragmentVolumes[fragment];
        if (std::isfinite(volume) && volume > 0.0) {
            seedVolume[fragmentSeedIds[fragment]] += static_cast<float>(volume);
        }
    }

    std::vector<float> refVolume(fineNodeCount, 0.0f);
    std::vector<glm::vec4> fineCentroids(fineNodeCount, glm::vec4(0.0f));
    for (uint32_t i = 0; i < fineNodeCount; ++i) {
        const uint32_t fragment = fluidNodeToGlobalFragment[i];
        if (fragment >= fragmentCount ||
            globalFragmentToFluidNode[fragment] != i ||
            !std::isfinite(fragmentVolumes[fragment]) ||
            !(fragmentVolumes[fragment] > 0.0)) {
            return false;
        }
        fineVolumes[i] = static_cast<float>(fragmentVolumes[fragment]);
        fineCentroids[i] = glm::vec4(glm::vec3(fragmentCentroids[fragment]), 1.0f);
        const auto it = seedVolume.find(fragmentSeedIds[fragment]);
        refVolume[i] = (it != seedVolume.end() && it->second > 0.0f)
            ? it->second
            : fineVolumes[i];
    }

    std::vector<RawFineFace> fineFluidFaces;
    std::vector<RawFineBoundaryFace> fineBoundaryFaces;

    for (uint32_t faceIdx = 0; faceIdx < faceCount; ++faceIdx) {
        if (faceTypes[faceIdx] == DomainBoundaryFaceType) continue;
        const uint32_t fragmentA = faceFragmentA[faceIdx];
        const uint32_t fragmentB = faceFragmentB[faceIdx];
        if (fragmentA == InvalidIndex || fragmentA >= fragmentCount) {
            return false;
        }

        const bool hasB = fragmentB != InvalidIndex && fragmentB < fragmentCount;
        const uint32_t nodeA = globalFragmentToFluidNode[fragmentA];
        const uint32_t nodeB = hasB ? globalFragmentToFluidNode[fragmentB] : UINT32_MAX;

        if (nodeA == UINT32_MAX && nodeB == UINT32_MAX) continue;
        if (nodeA != UINT32_MAX && nodeB != UINT32_MAX && nodeA == nodeB) continue;

        if (nodeA != UINT32_MAX && nodeB != UINT32_MAX) {
            const double areaA = faceAreasA[faceIdx];
            const double areaB = faceAreasB[faceIdx];
            if (!std::isfinite(areaA) || !(areaA > 0.0) ||
                !std::isfinite(areaB) || !(areaB > 0.0)) {
                return false;
            }

            const glm::dvec3 faceCentroid = faceMomentsA[faceIdx] / areaA;
            const glm::vec3 normal = glm::vec3(faceAreaVectorsA[faceIdx] / areaA);
            const float rawDA = static_cast<float>(glm::dot(faceCentroid - fragmentCentroids[fragmentA], glm::dvec3(normal)));
            const float rawDB = static_cast<float>(glm::dot(fragmentCentroids[fragmentB] - faceCentroid, glm::dvec3(normal)));
            if (!std::isfinite(rawDA) || !std::isfinite(rawDB)) {
                return false;
            }

            const float minDA = 0.10f * std::cbrt(std::max(refVolume[nodeA], 1e-12f));
            const float minDB = 0.10f * std::cbrt(std::max(refVolume[nodeB], 1e-12f));
            const float dA = std::max(rawDA, minDA);
            const float dB = std::max(rawDB, minDB);
            const float dist = dA + dB;
            if (!std::isfinite(dA) || !std::isfinite(dB) || dA < 0.0f || dB < 0.0f ||
                !std::isfinite(dist) || !(dist > 0.0f)) {
                return false;
            }
            RawFineFace raw{};
            raw.globalFaceIndex = faceIdx;
            raw.fineA = nodeA;
            raw.fineB = nodeB;
            raw.normal = normal;
            raw.area = static_cast<float>(areaA);
            raw.weight = static_cast<float>(areaA) / dist;
            raw.distA = dA;
            raw.distB = dB;
            fineFluidFaces.push_back(raw);
        } else if (nodeA != UINT32_MAX) {
            const double area = faceAreasA[faceIdx];
            if (!std::isfinite(area) || !(area > 0.0)) {
                return false;
            }
            const glm::dvec3 faceCentroid = faceMomentsA[faceIdx] / area;
            const glm::vec3 normal = glm::vec3(faceAreaVectorsA[faceIdx] / area);
            const float rawD = static_cast<float>(glm::dot(faceCentroid - fragmentCentroids[fragmentA], glm::dvec3(normal)));
            if (!std::isfinite(rawD)) {
                return false;
            }
            const float minD = 0.25f * std::cbrt(std::max(refVolume[nodeA], 1e-12f));
            const float effD = std::max(rawD, minD);
            if (!std::isfinite(effD) || !(effD > 0.0f)) {
                return false;
            }
            RawFineBoundaryFace wall{};
            wall.globalBoundaryFaceIndex = UINT32_MAX;
            wall.fineNode = nodeA;
            wall.boundaryCondition = cfd::EmbeddedWallBoundary;
            wall.outwardNormal = normal;
            wall.area = static_cast<float>(area);
            wall.weight = static_cast<float>(area) / effD;
            fineBoundaryFaces.push_back(wall);
        } else {
            const double area = faceAreasB[faceIdx];
            if (!std::isfinite(area) || !(area > 0.0)) {
                return false;
            }
            const glm::dvec3 faceCentroid = faceMomentsB[faceIdx] / area;
            const glm::vec3 outwardNormal = glm::vec3(-faceAreaVectorsB[faceIdx] / area);
            const float rawD = static_cast<float>(glm::dot(
                faceCentroid - fragmentCentroids[fragmentB], glm::dvec3(outwardNormal)));
            if (!std::isfinite(rawD)) {
                return false;
            }
            const float minD = 0.25f * std::cbrt(std::max(refVolume[nodeB], 1e-12f));
            const float effD = std::max(rawD, minD);
            if (!std::isfinite(effD) || !(effD > 0.0f)) {
                return false;
            }
            RawFineBoundaryFace wall{};
            wall.globalBoundaryFaceIndex = UINT32_MAX;
            wall.fineNode = nodeB;
            wall.boundaryCondition = cfd::EmbeddedWallBoundary;
            wall.outwardNormal = outwardNormal;
            wall.area = static_cast<float>(area);
            wall.weight = static_cast<float>(area) / effD;
            fineBoundaryFaces.push_back(wall);
        }
    }

    for (uint32_t boundaryIdx = 0; boundaryIdx < boundaryFaceIndices.size(); ++boundaryIdx) {
        const uint32_t faceIdx = boundaryFaceIndices[boundaryIdx];
        if (faceIdx >= faceCount) {
            return false;
        }
        const uint32_t fragmentA = faceFragmentA[faceIdx];
        const uint32_t fragmentB = faceFragmentB[faceIdx];
        const bool hasA = fragmentA != InvalidIndex && fragmentA < fragmentCount;
        const bool hasB = fragmentB != InvalidIndex && fragmentB < fragmentCount;
        const uint32_t nodeA = hasA ? globalFragmentToFluidNode[fragmentA] : UINT32_MAX;
        const uint32_t nodeB = hasB ? globalFragmentToFluidNode[fragmentB] : UINT32_MAX;
        if (nodeA == UINT32_MAX && nodeB == UINT32_MAX) continue;

        uint32_t node = nodeA;
        double faceArea = faceAreasA[faceIdx];
        glm::vec3 normal = (faceArea > 0.0)
            ? glm::vec3(faceAreaVectorsA[faceIdx] / faceArea)
            : glm::vec3(0.0f);
        glm::dvec3 centroid = (faceArea > 0.0)
            ? (faceMomentsA[faceIdx] / faceArea)
            : glm::dvec3(0.0);
        uint32_t fragment = fragmentA;
        if (nodeA == UINT32_MAX) {
            node = nodeB;
            fragment = fragmentB;
            faceArea = faceAreasB[faceIdx];
            if (!std::isfinite(faceArea) || !(faceArea > 0.0)) {
                return false;
            }
            normal = glm::vec3(-faceAreaVectorsB[faceIdx] / faceArea);
            centroid = faceMomentsB[faceIdx] / faceArea;
        } else if (!std::isfinite(faceArea) || !(faceArea > 0.0)) {
            return false;
        }

        const float rawD = static_cast<float>(glm::dot(centroid - fragmentCentroids[fragment], glm::dvec3(normal)));
        if (!std::isfinite(rawD)) {
            return false;
        }
        const float minD = 0.25f * std::cbrt(std::max(refVolume[node], 1e-12f));
        const float effD = std::max(rawD, minD);
        if (!std::isfinite(effD) || !(effD > 0.0f)) {
            return false;
        }
        RawFineBoundaryFace raw{};
        raw.globalBoundaryFaceIndex = boundaryIdx;
        raw.fineNode = node;
        raw.boundaryCondition = faceDomainFaces[faceIdx];
        raw.outwardNormal = normal;
        raw.area = static_cast<float>(faceArea);
        raw.weight = static_cast<float>(faceArea) / effD;
        fineBoundaryFaces.push_back(raw);
    }

    outFaceTopology.reserve(fineFluidFaces.size());
    outFaceFluxVectors.reserve(fineFluidFaces.size());
    outFaceGlobalIndex.reserve(fineFluidFaces.size());
    for (const RawFineFace& raw : fineFluidFaces) {
        const float dist = raw.distA + raw.distB;
        const float weightA = (dist > 0.0f) ? (raw.distB / dist) : 0.5f;
        const float weightB = (dist > 0.0f) ? (raw.distA / dist) : 0.5f;

        FaceTopology face{};
        face.cellA = raw.fineA;
        face.cellB = raw.fineB;
        face.projectionWeight = raw.weight;
        outFaceTopology.push_back(face);

        FaceFluxVectors flux{};
        flux.fluxVectorA = raw.normal * (raw.area * weightA);
        flux.fluxVectorB = raw.normal * (raw.area * weightB);
        outFaceFluxVectors.push_back(flux);
        outFaceGlobalIndex.push_back(raw.globalFaceIndex);
    }

    outBoundaryFaces.reserve(fineBoundaryFaces.size());
    outBoundaryGlobalIndex.reserve(fineBoundaryFaces.size());
    for (const RawFineBoundaryFace& raw : fineBoundaryFaces) {
        CFDBoundaryFaceGpu face{};
        face.cell = raw.fineNode;
        face.boundaryCondition = raw.boundaryCondition;
        face.areaVector = raw.outwardNormal * raw.area;
        face.projectionWeight = raw.weight;
        outBoundaryFaces.push_back(face);
        outBoundaryGlobalIndex.push_back(raw.globalBoundaryFaceIndex);
    }

    outPressureComponents.assign(fineNodeCount, UINT32_MAX);
    outPressureComponentCount = 0;
    {
        std::vector<std::vector<uint32_t>> adjacency(fineNodeCount);
        for (const FaceTopology& face : outFaceTopology) {
            adjacency[face.cellA].push_back(face.cellB);
            adjacency[face.cellB].push_back(face.cellA);
        }
        std::vector<uint32_t> stack;
        for (uint32_t i = 0; i < fineNodeCount; ++i) {
            if (outPressureComponents[i] != UINT32_MAX) continue;
            const uint32_t component = outPressureComponentCount++;
            outPressureComponents[i] = component;
            stack.push_back(i);
            while (!stack.empty()) {
                const uint32_t u = stack.back();
                stack.pop_back();
                for (uint32_t v : adjacency[u]) {
                    if (outPressureComponents[v] == UINT32_MAX) {
                        outPressureComponents[v] = component;
                        stack.push_back(v);
                    }
                }
            }
        }
    }

    outCellVolumes = std::move(fineVolumes);
    outCellCentroids = std::move(fineCentroids);
    return true;
}

} // namespace cfd
