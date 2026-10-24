#include "CFDTopology.hpp"
#include "CFDBoundaryConditions.hpp"

#include <iostream>
#include <cmath>
#include <limits>
#include <queue>
#include <algorithm>
#include <unordered_map>
#include <map>

namespace cfd {

namespace {

glm::vec3 getDomainBoxNormal(uint32_t domainFace) {
    switch (domainFace) {
        case 0: return {-1.0f,  0.0f,  0.0f}; // -X
        case 1: return { 1.0f,  0.0f,  0.0f}; // +X
        case 2: return { 0.0f, -1.0f,  0.0f}; // -Y
        case 3: return { 0.0f,  1.0f,  0.0f}; // +Y
        case 4: return { 0.0f,  0.0f, -1.0f}; // -Z
        case 5: return { 0.0f,  0.0f,  1.0f}; // +Z
        default: return {0.0f,  0.0f,  0.0f};
    }
}

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

} // anonymous namespace

bool buildCFDTopology(
    const std::vector<topology::Fragment>& fragments,
    const std::vector<topology::Face>& faces,
    const std::vector<topology::BoundaryFace>& boundaryFaces,
    const std::vector<uint32_t>& globalFragmentToFluidNode,
    const std::vector<uint32_t>& fluidNodeToGlobalFragment,
    CFDTopology& outTopology) {

    outTopology = CFDTopology{};

    if (fragments.empty() || globalFragmentToFluidNode.size() != fragments.size()) {
        std::cerr << "[CFDTopology] build FAILED: globalFragmentToFluidNode size mismatch\n";
        return false;
    }

    const uint32_t fineNodeCount = static_cast<uint32_t>(fluidNodeToGlobalFragment.size());
    if (fineNodeCount == 0) {
        return true;
    }

    // 1. Recover fine fragment volumes and parent base-cell volumes
    std::vector<float> fineVolumes(fineNodeCount, 0.0f);
    std::unordered_map<uint32_t, float> seedVolume;
    for (const auto& frag : fragments) {
        if (std::isfinite(frag.volume) && frag.volume > 0.0f) {
            seedVolume[frag.seedId] += frag.volume;
        }
    }

    std::vector<float> refVolume(fineNodeCount, 0.0f);
    for (uint32_t i = 0; i < fineNodeCount; ++i) {
        const uint32_t fragIdx = fluidNodeToGlobalFragment[i];
        if (fragIdx >= fragments.size() || globalFragmentToFluidNode[fragIdx] != i ||
            !std::isfinite(fragments[fragIdx].volume) || !(fragments[fragIdx].volume > 0.0f)) {
            std::cerr << "[CFDTopology] build FAILED: invalid mapping or volume on fine node " << i << "\n";
            return false;
        }
        fineVolumes[i] = fragments[fragIdx].volume;
        auto it = seedVolume.find(fragments[fragIdx].seedId);
        refVolume[i] = (it != seedVolume.end() && it->second > 0.0f) ? it->second : fineVolumes[i];
    }

    // 2. Classify raw fine fluid-fluid faces
    std::vector<RawFineFace> fineFluidFaces;
    std::vector<RawFineBoundaryFace> fineBoundaryFaces;

    // Geometry-floor census for raw centroid distance clamping
    uint64_t internalFloorFaces = 0;
    uint64_t internalFaceTotal = 0;
    uint64_t wallFloorFaces = 0;
    uint64_t wallFaceTotal = 0;
    uint64_t domainFloorFaces = 0;
    uint64_t domainFaceTotal = 0;
    float worstFloorRatio = 1.0f;

    for (uint32_t faceIdx = 0; faceIdx < faces.size(); ++faceIdx) {
        const auto& face = faces[faceIdx];
        if (face.fragmentA >= fragments.size()) continue;

        const uint32_t nodeA = globalFragmentToFluidNode[face.fragmentA];
        const uint32_t nodeB = (face.fragmentB < fragments.size())
            ? globalFragmentToFluidNode[face.fragmentB]
            : UINT32_MAX;

        if (nodeA == UINT32_MAX && nodeB == UINT32_MAX) continue;

        const double area = (face.areaA > 0.0) ? face.areaA : face.areaB;
        if (!std::isfinite(area) || !(area > 0.0)) continue;

        const glm::dvec3 faceCentroid = (face.areaA > 0.0) ? (face.momentA / face.areaA) : (face.momentB / face.areaB);
        glm::vec3 normal(0.0f);
        const double areaVecLenA = glm::length(face.areaVectorA);
        const double areaVecLenB = glm::length(face.areaVectorB);
        if (areaVecLenA > 1e-14 && std::isfinite(areaVecLenA)) {
            normal = glm::vec3(face.areaVectorA / areaVecLenA);
        } else if (areaVecLenB > 1e-14 && std::isfinite(areaVecLenB)) {
            normal = glm::vec3(-face.areaVectorB / areaVecLenB);
        } else if (face.fragmentB < fragments.size()) {
            const glm::dvec3 diff = fragments[face.fragmentB].centroid - fragments[face.fragmentA].centroid;
            const double diffLen = glm::length(diff);
            if (diffLen > 1e-14) normal = glm::vec3(diff / diffLen);
        }

        const float rawDA = static_cast<float>(glm::dot(faceCentroid - fragments[face.fragmentA].centroid, glm::dvec3(normal)));
        float rawDB = 0.0f;
        if (face.fragmentB < fragments.size()) {
            rawDB = static_cast<float>(glm::dot(fragments[face.fragmentB].centroid - faceCentroid, glm::dvec3(normal)));
        }

        if (nodeA != UINT32_MAX && nodeB != UINT32_MAX && nodeA != nodeB) {
            const float minDA = 0.10f * std::cbrt(std::max(refVolume[nodeA], 1e-12f));
            const float minDB = 0.10f * std::cbrt(std::max(refVolume[nodeB], 1e-12f));
            const float dA = std::max(rawDA, minDA);
            const float dB = std::max(rawDB, minDB);
            const float dist = dA + dB;

            ++internalFaceTotal;
            if (rawDA < minDA || rawDB < minDB) {
                ++internalFloorFaces;
                if (rawDA < minDA) worstFloorRatio = std::min(worstFloorRatio, rawDA / minDA);
                if (rawDB < minDB) worstFloorRatio = std::min(worstFloorRatio, rawDB / minDB);
            }

            if (!std::isfinite(dA) || !std::isfinite(dB) || dA < 0.0f || dB < 0.0f ||
                !std::isfinite(dist) || !(dist > 0.0f)) {
                std::cerr << "[CFDTopology] build FAILED: invalid internal face geometry on face " << faceIdx << "\n";
                return false;
            }

            const float weight = static_cast<float>(area) / dist;
            RawFineFace rff{};
            rff.globalFaceIndex = faceIdx;
            rff.fineA = nodeA;
            rff.fineB = nodeB;
            rff.normal = normal;
            rff.area = static_cast<float>(area);
            rff.weight = weight;
            rff.distA = dA;
            rff.distB = dB;
            fineFluidFaces.push_back(rff);
        } else if (nodeA != UINT32_MAX && nodeB == UINT32_MAX) {
            // Embedded solid wall: fluid nodeA -> solid
            const float minDA = 0.25f * std::cbrt(std::max(refVolume[nodeA], 1e-12f));
            const float effDA = std::max(rawDA, minDA);
            ++wallFaceTotal;
            if (rawDA < minDA) {
                ++wallFloorFaces;
                worstFloorRatio = std::min(worstFloorRatio, rawDA / minDA);
            }
            if (!std::isfinite(effDA) || !(effDA > 0.0f)) {
                std::cerr << "[CFDTopology] build FAILED: invalid wall geometry on face " << faceIdx << "\n";
                return false;
            }
            RawFineBoundaryFace wall{};
            wall.globalBoundaryFaceIndex = UINT32_MAX;
            wall.fineNode = nodeA;
            wall.boundaryCondition = cfd::EmbeddedWallBoundary;
            wall.outwardNormal = normal;
            wall.area = static_cast<float>(area);
            wall.weight = static_cast<float>(area) / effDA;
            fineBoundaryFaces.push_back(wall);
        } else if (nodeA == UINT32_MAX && nodeB != UINT32_MAX) {
            // Embedded solid wall: solid -> fluid nodeB
            const float minDB = 0.25f * std::cbrt(std::max(refVolume[nodeB], 1e-12f));
            const float effDB = std::max(rawDB, minDB);
            ++wallFaceTotal;
            if (rawDB < minDB) {
                ++wallFloorFaces;
                worstFloorRatio = std::min(worstFloorRatio, rawDB / minDB);
            }
            if (!std::isfinite(effDB) || !(effDB > 0.0f)) {
                std::cerr << "[CFDTopology] build FAILED: invalid wall geometry on face " << faceIdx << "\n";
                return false;
            }
            RawFineBoundaryFace wall{};
            wall.globalBoundaryFaceIndex = UINT32_MAX;
            wall.fineNode = nodeB;
            wall.boundaryCondition = cfd::EmbeddedWallBoundary;
            wall.outwardNormal = -normal;
            wall.area = static_cast<float>(area);
            wall.weight = static_cast<float>(area) / effDB;
            fineBoundaryFaces.push_back(wall);
        }
    }

    // Domain boundary faces (domainFace < 6)
    for (uint32_t bfIdx = 0; bfIdx < boundaryFaces.size(); ++bfIdx) {
        const auto& bf = boundaryFaces[bfIdx];
        if (bf.fragment >= fragments.size()) continue;

        const uint32_t node = globalFragmentToFluidNode[bf.fragment];
        if (node != UINT32_MAX && bf.domainFace < 6) {
            const topology::Face* boundaryFace =
                (bf.faceIndex < faces.size()) ? &faces[bf.faceIndex] : nullptr;
            const double faceArea = boundaryFace ? boundaryFace->areaA : 0.0;
            const glm::dvec3 faceMoment = boundaryFace ? boundaryFace->momentA : glm::dvec3(0.0);
            const glm::vec3 boxNormal = getDomainBoxNormal(bf.domainFace);
            float rawD = 0.0f;
            if (faceArea > 0.0 && std::isfinite(faceArea)) {
                const glm::dvec3 bfCentroid = faceMoment / faceArea;
                rawD = static_cast<float>(glm::dot(bfCentroid - fragments[bf.fragment].centroid, glm::dvec3(boxNormal)));
            }
            const float minD = 0.25f * std::cbrt(std::max(refVolume[node], 1e-12f));
            const float effD = std::max(rawD, minD);
            ++domainFaceTotal;
            if (rawD < minD) {
                ++domainFloorFaces;
                worstFloorRatio = std::min(worstFloorRatio, rawD / minD);
            }
            if (!std::isfinite(faceArea) || !(faceArea > 0.0) ||
                !std::isfinite(effD) || !(effD > 0.0f)) {
                std::cerr << "[CFDTopology] build FAILED: invalid domain boundary geometry on face " << bfIdx << "\n";
                return false;
            }
            RawFineBoundaryFace fbf{};
            fbf.globalBoundaryFaceIndex = bfIdx;
            fbf.fineNode = node;
            fbf.boundaryCondition = bf.domainFace;
            fbf.outwardNormal = boxNormal;
            fbf.area = static_cast<float>(faceArea);
            fbf.weight = static_cast<float>(faceArea) / effD;
            fineBoundaryFaces.push_back(fbf);
        }
    }

    // Build execution topology and map global face indices
    std::vector<CFDFaceGpu> fineFaceGpu;
    std::vector<uint32_t> fineFaceGlobalIndex;
    fineFaceGpu.reserve(fineFluidFaces.size());
    fineFaceGlobalIndex.reserve(fineFluidFaces.size());
    for (const RawFineFace& raw : fineFluidFaces) {
        CFDFaceGpu gpu{};
        gpu.cellA = raw.fineA;
        gpu.cellB = raw.fineB;
        // Distance-weighted interpolation: phi* = dot(uA, fvA) + dot(uB, fvB)
        const float dist = raw.distA + raw.distB;
        const float weightA = (dist > 0.0f) ? (raw.distB / dist) : 0.5f;
        const float weightB = (dist > 0.0f) ? (raw.distA / dist) : 0.5f;
        gpu.fluxVectorA = raw.normal * (raw.area * weightA);
        gpu.fluxVectorB = raw.normal * (raw.area * weightB);
        gpu.projectionWeight = raw.weight;
        fineFaceGpu.push_back(gpu);
        fineFaceGlobalIndex.push_back(raw.globalFaceIndex);
    }

    std::vector<CFDBoundaryFaceGpu> fineBoundaryFaceGpu;
    std::vector<uint32_t> fineBoundaryGlobalIndex;
    fineBoundaryFaceGpu.reserve(fineBoundaryFaces.size());
    fineBoundaryGlobalIndex.reserve(fineBoundaryFaces.size());
    for (const RawFineBoundaryFace& raw : fineBoundaryFaces) {
        CFDBoundaryFaceGpu gpu{};
        gpu.cell = raw.fineNode;
        gpu.boundaryCondition = raw.boundaryCondition;
        gpu.areaVector = raw.outwardNormal * raw.area;
        gpu.projectionWeight = raw.weight;
        fineBoundaryFaceGpu.push_back(gpu);
        fineBoundaryGlobalIndex.push_back(raw.globalBoundaryFaceIndex);
    }

    // Fine pressure components (disjoint fluid regions)
    std::vector<uint32_t> fineComponent(fineNodeCount, UINT32_MAX);
    uint32_t fineComponentCount = 0;
    {
        std::vector<std::vector<uint32_t>> fineAdjacency(fineNodeCount);
        for (const CFDFaceGpu& face : fineFaceGpu) {
            fineAdjacency[face.cellA].push_back(face.cellB);
            fineAdjacency[face.cellB].push_back(face.cellA);
        }
        std::vector<uint32_t> stack;
        for (uint32_t i = 0; i < fineNodeCount; ++i) {
            if (fineComponent[i] != UINT32_MAX) continue;
            const uint32_t component = fineComponentCount++;
            fineComponent[i] = component;
            stack.push_back(i);
            while (!stack.empty()) {
                const uint32_t u = stack.back();
                stack.pop_back();
                for (uint32_t v : fineAdjacency[u]) {
                    if (fineComponent[v] == UINT32_MAX) {
                        fineComponent[v] = component;
                        stack.push_back(v);
                    }
                }
            }
        }
    }

    outTopology.fluidNodeCount = fineNodeCount;
    outTopology.faces = std::move(fineFaceGpu);
    outTopology.faceGlobalIndex = std::move(fineFaceGlobalIndex);
    outTopology.boundaryFaces = std::move(fineBoundaryFaceGpu);
    outTopology.boundaryGlobalIndex = std::move(fineBoundaryGlobalIndex);
    outTopology.cellVolumes = fineVolumes;
    outTopology.component = std::move(fineComponent);
    outTopology.componentCount = fineComponentCount;

    std::cout << "[CFDTopology] Topology generated successfully: "
              << outTopology.faces.size() << " faces, "
              << outTopology.boundaryFaces.size() << " boundary faces.\n";
    std::cout << "[CFDTopology] geometry floors: internal=" << internalFloorFaces << "/" << internalFaceTotal
              << " wall=" << wallFloorFaces << "/" << wallFaceTotal
              << " domain=" << domainFloorFaces << "/" << domainFaceTotal
              << " worstRawFloorRatio=" << worstFloorRatio << "\n";

    return true;
}

} // namespace cfd
