#include "VoronoiRuntime.hpp"

#include "cuda/RVD.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "util/GeometryUtils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <utility>

static constexpr uint32_t InvalidIndex = std::numeric_limits<uint32_t>::max();
static constexpr uint32_t InvalidModelId = voronoi::InvalidModelId;
static constexpr uint8_t SharedFaceType = 0u;
static constexpr uint8_t DomainBoundaryFaceType = 1u;
static constexpr uint8_t MaterialBoundaryFaceType = 2u;
static constexpr uint32_t DomainBoundaryCount = 6u;

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

static bool positiveFinite(double value) {
    return std::isfinite(value) && value > 0.0;
}

void VoronoiRuntime::clearMaterializedTopology() {
    outFragmentInstanceIds.clear();
    outFragmentSeedIds.clear();
    outFragmentCentroids.clear();
    outFragmentVolumes.clear();
    outFaceFragmentA.clear();
    outFaceFragmentB.clear();
    outFaceAreasA.clear();
    outFaceAreasB.clear();
    outFaceAreaVectorsA.clear();
    outFaceAreaVectorsB.clear();
    outFaceMomentsA.clear();
    outFaceMomentsB.clear();
    outFaceTypes.clear();
    outFaceDomainFaces.clear();
    outBoundaryFaceIndices.clear();
    outFaceLoopOffsets.clear();
    outLoopVertexOffsets.clear();
    outLoopVertices.clear();
    outContactCouplingFaceA.clear();
    outContactCouplingFragmentA.clear();
    outContactCouplingFaceB.clear();
    outContactCouplingFragmentB.clear();
    outContactCouplingAreas.clear();
    outSeedPositions.clear();
    outCellFragmentOffsets.clear();
    outCellFragments.clear();
    outInstanceFragmentCounts.clear();
    outFragmentToSpaceIndex.clear();
    outSpaceToFragmentIndex.clear();
    outSeedToSpaceIndex.clear();
    outSpaceFragmentCount = 0;
    outHasUniqueSeedToSpaceMapping = false;
    outDomainMin = glm::vec3(0.0f);
    outDomainMax = glm::vec3(0.0f);
    outVolumeGridMin = glm::vec3(0.0f);
    outVolumeGridDim = glm::ivec3(0);
    outVolumeGridCellSize = 0.0f;
    outDomainCorners = {};
}

bool VoronoiRuntime::materializeTopology(
    voronoi::RVDResult&& result,
    const std::vector<uint32_t>& runtimeModelIds,
    const glm::vec3& domainMin,
    const glm::vec3& domainMax,
    const glm::vec3& gridMin,
    const glm::ivec3& gridDim,
    float gridSpacing) {
    clearMaterializedTopology();

    outFragmentInstanceIds.reserve(result.fragments.size());
    outFragmentSeedIds.reserve(result.fragments.size());
    outFragmentCentroids.reserve(result.fragments.size());
    outFragmentVolumes.reserve(result.fragments.size());
    for (const voronoi::Fragment& fragment : result.fragments) {
        outFragmentInstanceIds.push_back(fragment.ownerModelId);
        outFragmentSeedIds.push_back(fragment.cellId);
        outFragmentCentroids.push_back(fragment.centroid);
        outFragmentVolumes.push_back(fragment.volume);
    }

    outFaceFragmentA.reserve(result.fragmentFaces.size());
    outFaceFragmentB.reserve(result.fragmentFaces.size());
    outFaceAreasA.reserve(result.fragmentFaces.size());
    outFaceAreasB.reserve(result.fragmentFaces.size());
    outFaceAreaVectorsA.reserve(result.fragmentFaces.size());
    outFaceAreaVectorsB.reserve(result.fragmentFaces.size());
    outFaceMomentsA.reserve(result.fragmentFaces.size());
    outFaceMomentsB.reserve(result.fragmentFaces.size());
    outFaceTypes.reserve(result.fragmentFaces.size());
    outFaceDomainFaces.reserve(result.fragmentFaces.size());
    for (uint32_t faceIndex = 0; faceIndex < result.fragmentFaces.size(); ++faceIndex) {
        const voronoi::FragmentFace& face = result.fragmentFaces[faceIndex];
        outFaceFragmentA.push_back(face.fragmentA);
        outFaceFragmentB.push_back(face.fragmentB);
        outFaceAreasA.push_back(face.areaA);
        outFaceAreasB.push_back(face.areaB);
        outFaceAreaVectorsA.push_back(face.areaVectorA);
        outFaceAreaVectorsB.push_back(face.areaVectorB);
        outFaceMomentsA.push_back(face.momentA);
        outFaceMomentsB.push_back(face.momentB);
        outFaceTypes.push_back(static_cast<uint8_t>(face.type));
        outFaceDomainFaces.push_back(face.domainFace);
        if (face.type == voronoi::FaceType::DomainBoundary) {
            outBoundaryFaceIndices.push_back(faceIndex);
        }
    }

    outFaceLoopOffsets = std::move(result.faceLoopOffsets);
    outLoopVertexOffsets = std::move(result.loopVertexOffsets);
    outLoopVertices = std::move(result.loopVertices);

    outContactCouplingFaceA.reserve(result.contactCouplings.size());
    outContactCouplingFragmentA.reserve(result.contactCouplings.size());
    outContactCouplingFaceB.reserve(result.contactCouplings.size());
    outContactCouplingFragmentB.reserve(result.contactCouplings.size());
    outContactCouplingAreas.reserve(result.contactCouplings.size());
    for (const voronoi::ContactCoupling& coupling : result.contactCouplings) {
        outContactCouplingFaceA.push_back(coupling.faceA);
        outContactCouplingFragmentA.push_back(coupling.fragmentA);
        outContactCouplingFaceB.push_back(coupling.faceB);
        outContactCouplingFragmentB.push_back(coupling.fragmentB);
        outContactCouplingAreas.push_back(coupling.couplingArea);
    }

    if (result.seedPositions.empty()) {
        std::cerr << "[VoronoiRuntime] materialization requires resolved seed positions"
                  << std::endl;
        clearMaterializedTopology();
        return false;
    }
    outSeedPositions.reserve(result.seedPositions.size());
    for (const glm::vec4& seed : result.seedPositions) {
        outSeedPositions.push_back(glm::vec3(seed));
    }

    outCellFragmentOffsets = std::move(result.cellFragmentOffsets);
    outCellFragments = std::move(result.cellFragments);

    std::unordered_map<uint32_t, uint32_t> runtimeModelIdToInstanceIndex;
    for (size_t i = 0; i < runtimeModelIds.size(); ++i) {
        runtimeModelIdToInstanceIndex[runtimeModelIds[i]] = static_cast<uint32_t>(i);
    }
    outInstanceFragmentCounts.assign(runtimeModelIds.size(), 0);
    for (uint32_t fragment = 0; fragment < outFragmentInstanceIds.size(); ++fragment) {
        const uint32_t ownerModelId = outFragmentInstanceIds[fragment];
        if (ownerModelId == InvalidModelId) continue;
        const auto it = runtimeModelIdToInstanceIndex.find(ownerModelId);
        if (it == runtimeModelIdToInstanceIndex.end()) {
            std::cerr << "[VoronoiRuntime] fragment " << fragment
                      << " references unknown model " << ownerModelId << std::endl;
            clearMaterializedTopology();
            return false;
        }
        outInstanceFragmentCounts[it->second]++;
    }

    outDomainMin = domainMin;
    outDomainMax = domainMax;
    outVolumeGridMin = gridMin;
    outVolumeGridDim = gridDim;
    outVolumeGridCellSize = gridSpacing;
    for (uint32_t corner = 0; corner < 8; ++corner) {
        outDomainCorners[corner] = glm::vec3(
            (corner & 1u) ? domainMax.x : domainMin.x,
            (corner & 2u) ? domainMax.y : domainMin.y,
            (corner & 4u) ? domainMax.z : domainMin.z);
    }

    if (!buildSpacePartition(outSeedPositions.size())) {
        std::cerr << "[VoronoiRuntime] Space partition construction failed" << std::endl;
        clearMaterializedTopology();
        return false;
    }
    if (!validateTopology()) {
        std::cerr << "[VoronoiRuntime] Runtime topology validation failed" << std::endl;
        clearMaterializedTopology();
        return false;
    }
    return true;
}

void VoronoiRuntime::publishTopology(VoronoiProduct& product) const {
    product.fragmentInstanceIds = &outFragmentInstanceIds;
    product.fragmentSeedIds = &outFragmentSeedIds;
    product.fragmentCentroids = &outFragmentCentroids;
    product.fragmentVolumes = &outFragmentVolumes;
    product.faceFragmentA = &outFaceFragmentA;
    product.faceFragmentB = &outFaceFragmentB;
    product.faceAreasA = &outFaceAreasA;
    product.faceAreasB = &outFaceAreasB;
    product.faceAreaVectorsA = &outFaceAreaVectorsA;
    product.faceAreaVectorsB = &outFaceAreaVectorsB;
    product.faceMomentsA = &outFaceMomentsA;
    product.faceMomentsB = &outFaceMomentsB;
    product.faceTypes = &outFaceTypes;
    product.faceDomainFaces = &outFaceDomainFaces;
    product.boundaryFaceIndices = &outBoundaryFaceIndices;
    product.faceLoopOffsets = &outFaceLoopOffsets;
    product.loopVertexOffsets = &outLoopVertexOffsets;
    product.loopVertices = &outLoopVertices;
    product.contactCouplingFaceA = &outContactCouplingFaceA;
    product.contactCouplingFragmentA = &outContactCouplingFragmentA;
    product.contactCouplingFaceB = &outContactCouplingFaceB;
    product.contactCouplingFragmentB = &outContactCouplingFragmentB;
    product.contactCouplingAreas = &outContactCouplingAreas;
    product.seedPositions = &outSeedPositions;
    product.cellFragmentOffsets = &outCellFragmentOffsets;
    product.cellFragments = &outCellFragments;
    product.instanceFragmentCounts = &outInstanceFragmentCounts;
    product.fragmentToSpaceIndex = &outFragmentToSpaceIndex;
    product.spaceToFragmentIndex = &outSpaceToFragmentIndex;
    product.seedToSpaceIndex = outHasUniqueSeedToSpaceMapping
        ? &outSeedToSpaceIndex
        : nullptr;
    product.spaceFragmentCount = outSpaceFragmentCount;
    product.hasUniqueSeedToSpaceMapping = outHasUniqueSeedToSpaceMapping;
    product.domainMin = outDomainMin;
    product.domainMax = outDomainMax;
    product.volumeGridMin = outVolumeGridMin;
    product.volumeGridDim = outVolumeGridDim;
    product.volumeGridCellSize = outVolumeGridCellSize;
    product.globalDomainCorners = outDomainCorners;
}

bool VoronoiRuntime::buildSpacePartition(size_t seedCount) {
    outFragmentToSpaceIndex.assign(outFragmentInstanceIds.size(), InvalidIndex);
    outSpaceToFragmentIndex.clear();
    outSpaceToFragmentIndex.reserve(outFragmentInstanceIds.size());
    for (uint32_t fragment = 0; fragment < outFragmentInstanceIds.size(); ++fragment) {
        if (outFragmentInstanceIds[fragment] != InvalidModelId) continue;
        outFragmentToSpaceIndex[fragment] = static_cast<uint32_t>(outSpaceToFragmentIndex.size());
        outSpaceToFragmentIndex.push_back(fragment);
    }
    outSpaceFragmentCount = static_cast<uint32_t>(outSpaceToFragmentIndex.size());

    outSeedToSpaceIndex.assign(seedCount, InvalidIndex);
    outHasUniqueSeedToSpaceMapping = true;
    if (outCellFragmentOffsets.size() != seedCount + 1u ||
        !isMonotonicOffsets(outCellFragmentOffsets, static_cast<uint32_t>(outCellFragments.size()))) {
        return false;
    }

    for (uint32_t cell = 0; cell < seedCount; ++cell) {
        for (uint32_t entry = outCellFragmentOffsets[cell];
             entry < outCellFragmentOffsets[cell + 1u]; ++entry) {
            const uint32_t fragment = outCellFragments[entry];
            if (fragment >= outFragmentInstanceIds.size()) {
                std::cerr << "[VoronoiRuntime] seed cell " << cell << " references out-of-range fragment " << fragment << std::endl;
                return false;
            }
            if (outFragmentInstanceIds[fragment] != InvalidModelId) continue;
            const uint32_t spaceIndex = outFragmentToSpaceIndex[fragment];
            if (spaceIndex == InvalidIndex) {
                std::cerr << "[VoronoiRuntime] space fragment " << fragment << " is missing from the space partition" << std::endl;
                return false;
            }
            if (outSeedToSpaceIndex[cell] != InvalidIndex) {
                outHasUniqueSeedToSpaceMapping = false;
            } else {
                outSeedToSpaceIndex[cell] = spaceIndex;
            }
        }
    }

    if (!outHasUniqueSeedToSpaceMapping) {
        outSeedToSpaceIndex.clear();
    }
    return true;
}

bool VoronoiRuntime::validateTopology() const {
    const size_t fragmentCount = outFragmentInstanceIds.size();
    const size_t faceCount = outFaceFragmentA.size();
    if (fragmentCount == 0 || faceCount == 0 || outSeedPositions.empty() ||
        outFragmentSeedIds.size() != fragmentCount ||
        outFragmentCentroids.size() != fragmentCount ||
        outFragmentVolumes.size() != fragmentCount ||
        outFaceFragmentB.size() != faceCount ||
        outFaceAreasA.size() != faceCount ||
        outFaceAreasB.size() != faceCount ||
        outFaceAreaVectorsA.size() != faceCount ||
        outFaceAreaVectorsB.size() != faceCount ||
        outFaceMomentsA.size() != faceCount ||
        outFaceMomentsB.size() != faceCount ||
        outFaceTypes.size() != faceCount ||
        outFaceDomainFaces.size() != faceCount) {
        return false;
    }
    for (size_t fragment = 0; fragment < fragmentCount; ++fragment) {
        if (outFragmentSeedIds[fragment] >= outSeedPositions.size()) {
            return false;
        }
        if (!positiveFinite(outFragmentVolumes[fragment])) {
            return false;
        }
        if (!GeometryUtils::isFinite(outFragmentCentroids[fragment])) {
            return false;
        }
    }
    if (outFaceLoopOffsets.size() != faceCount + 1u ||
        !isMonotonicOffsets(outFaceLoopOffsets, outFaceLoopOffsets.back())) {
        return false;
    }
    const uint32_t loopCount = outFaceLoopOffsets.back();
    if (outLoopVertexOffsets.size() != size_t(loopCount) + 1u ||
        !isMonotonicOffsets(outLoopVertexOffsets, static_cast<uint32_t>(outLoopVertices.size()))) {
        return false;
    }
    for (uint32_t loop = 0; loop < loopCount; ++loop) {
        if (outLoopVertexOffsets[loop + 1u] < outLoopVertexOffsets[loop] + 3u) {
            return false;
        }
    }
    for (size_t vertex = 0; vertex < outLoopVertices.size(); ++vertex) {
        if (!GeometryUtils::isFinite(outLoopVertices[vertex])) {
            return false;
        }
    }
    for (size_t face = 0; face < faceCount; ++face) {
        const uint32_t fragmentA = outFaceFragmentA[face];
        const uint32_t fragmentB = outFaceFragmentB[face];
        const bool hasA = fragmentA != InvalidIndex && fragmentA < fragmentCount;
        const bool hasB = fragmentB != InvalidIndex && fragmentB < fragmentCount;
        if (!hasA ||
            (fragmentB != InvalidIndex && !hasB) ||
            !std::isfinite(outFaceAreasA[face]) || !std::isfinite(outFaceAreasB[face]) ||
            !GeometryUtils::isFinite(outFaceAreaVectorsA[face]) || !GeometryUtils::isFinite(outFaceAreaVectorsB[face]) ||
            !GeometryUtils::isFinite(outFaceMomentsA[face]) || !GeometryUtils::isFinite(outFaceMomentsB[face])) {
            return false;
        }
        if (!positiveFinite(outFaceAreasA[face])) {
            return false;
        }

        const uint8_t type = outFaceTypes[face];
        if (type == SharedFaceType) {
            if (!hasB || fragmentA == fragmentB || !positiveFinite(outFaceAreasB[face])) {
                return false;
            }
        } else if (type == DomainBoundaryFaceType) {
            if (hasB || outFaceDomainFaces[face] >= DomainBoundaryCount) {
                return false;
            }
        } else if (type == MaterialBoundaryFaceType) {
            if (hasB || outFaceDomainFaces[face] != InvalidIndex) {
                return false;
            }
        } else {
            return false;
        }
    }

    size_t boundarySlot = 0;
    for (size_t face = 0; face < faceCount; ++face) {
        if (outFaceTypes[face] != 1u) continue;
        if (boundarySlot >= outBoundaryFaceIndices.size() ||
            outBoundaryFaceIndices[boundarySlot] != face) {
            return false;
        }
        ++boundarySlot;
    }
    if (boundarySlot != outBoundaryFaceIndices.size()) {
        return false;
    }

    const size_t couplingCount = outContactCouplingFaceA.size();
    if (outContactCouplingFaceB.size() != couplingCount ||
        outContactCouplingFragmentA.size() != couplingCount ||
        outContactCouplingFragmentB.size() != couplingCount ||
        outContactCouplingAreas.size() != couplingCount) {
        return false;
    }
    for (size_t coupling = 0; coupling < couplingCount; ++coupling) {
        const uint32_t faceA = outContactCouplingFaceA[coupling];
        const uint32_t faceB = outContactCouplingFaceB[coupling];
        const uint32_t fragmentA = outContactCouplingFragmentA[coupling];
        const uint32_t fragmentB = outContactCouplingFragmentB[coupling];
        if (faceA >= faceCount || faceB >= faceCount ||
            fragmentA >= fragmentCount || fragmentB >= fragmentCount ||
            !positiveFinite(outContactCouplingAreas[coupling]) ||
            (fragmentA != outFaceFragmentA[faceA] && fragmentA != outFaceFragmentB[faceA]) ||
            (fragmentB != outFaceFragmentA[faceB] && fragmentB != outFaceFragmentB[faceB])) {
            return false;
        }
    }

    if (outCellFragmentOffsets.size() != outSeedPositions.size() + 1u ||
        !isMonotonicOffsets(outCellFragmentOffsets, static_cast<uint32_t>(outCellFragments.size()))) {
        return false;
    }
    std::vector<uint32_t> lastSeenCell(fragmentCount, InvalidIndex);
    std::vector<uint8_t> representativeCovered(fragmentCount, 0u);
    for (uint32_t cell = 0; cell < outSeedPositions.size(); ++cell) {
        for (uint32_t entry = outCellFragmentOffsets[cell];
             entry < outCellFragmentOffsets[cell + 1u]; ++entry) {
            const uint32_t fragment = outCellFragments[entry];
            if (fragment >= fragmentCount) {
                return false;
            }
            if (lastSeenCell[fragment] == cell) {
                return false;
            }
            lastSeenCell[fragment] = cell;
            if (outFragmentSeedIds[fragment] == cell) {
                representativeCovered[fragment] = 1u;
            }
        }
    }
    for (uint32_t fragment = 0; fragment < fragmentCount; ++fragment) {
        if (!representativeCovered[fragment]) {
            return false;
        }
    }

    if (outFragmentToSpaceIndex.size() != fragmentCount ||
        outSpaceToFragmentIndex.size() != outSpaceFragmentCount) {
        return false;
    }
    std::vector<uint8_t> seenFragments(fragmentCount, 0u);
    for (uint32_t space = 0; space < outSpaceFragmentCount; ++space) {
        const uint32_t fragment = outSpaceToFragmentIndex[space];
        if (fragment >= fragmentCount || seenFragments[fragment] ||
            outFragmentInstanceIds[fragment] != InvalidModelId ||
            outFragmentToSpaceIndex[fragment] != space) {
            return false;
        }
        seenFragments[fragment] = 1u;
    }
    for (uint32_t fragment = 0; fragment < fragmentCount; ++fragment) {
        if (seenFragments[fragment]) continue;
        if (outFragmentInstanceIds[fragment] == InvalidModelId ||
            outFragmentToSpaceIndex[fragment] != InvalidIndex) {
            return false;
        }
    }
    if (outHasUniqueSeedToSpaceMapping) {
        if (outSeedToSpaceIndex.size() != outSeedPositions.size()) {
            return false;
        }
        for (uint32_t cell = 0; cell < outSeedToSpaceIndex.size(); ++cell) {
            const uint32_t space = outSeedToSpaceIndex[cell];
            if (space != InvalidIndex && space >= outSpaceFragmentCount) {
                return false;
            }
        }
        for (uint32_t cell = 0; cell < outSeedPositions.size(); ++cell) {
            uint32_t expectedSpace = InvalidIndex;
            for (uint32_t entry = outCellFragmentOffsets[cell];
                 entry < outCellFragmentOffsets[cell + 1u]; ++entry) {
                const uint32_t fragment = outCellFragments[entry];
                if (outFragmentInstanceIds[fragment] == InvalidModelId) {
                    expectedSpace = outFragmentToSpaceIndex[fragment];
                    break;
                }
            }
            if (outSeedToSpaceIndex[cell] != expectedSpace) {
                return false;
            }
        }
    } else if (!outSeedToSpaceIndex.empty()) {
        return false;
    }

    if (!std::isfinite(outDomainMin.x) || !std::isfinite(outDomainMin.y) ||
        !std::isfinite(outDomainMin.z) || !std::isfinite(outDomainMax.x) ||
        !std::isfinite(outDomainMax.y) || !std::isfinite(outDomainMax.z) ||
        outDomainMax.x <= outDomainMin.x || outDomainMax.y <= outDomainMin.y ||
        outDomainMax.z <= outDomainMin.z ||
        outVolumeGridDim.x < 2 || outVolumeGridDim.y < 2 || outVolumeGridDim.z < 2 ||
        !(outVolumeGridCellSize > 0.0f) || !std::isfinite(outVolumeGridCellSize)) {
        return false;
    }

    size_t occupiedFragmentCount = 0;
    for (uint32_t instanceId : outFragmentInstanceIds) {
        if (instanceId != InvalidModelId) ++occupiedFragmentCount;
    }
    size_t countedFragments = 0;
    for (uint32_t count : outInstanceFragmentCounts) {
        countedFragments += count;
    }
    if (countedFragments != occupiedFragmentCount) {
        return false;
    }
    return true;
}

void VoronoiRuntime::invalidateMaterialization() {
    voronoiReady = false;
    inSeedPositions.clear();
    inGlobalRemeshRuntimeModelIds.clear();
    inGlobalRemeshPositions.clear();
    inGlobalRemeshTriangleIndices.clear();
    inGlobalRemeshSurfacePositions.clear();
    inGlobalRemeshSurfaceTriangleIndices.clear();
    inGlobalVolumePadding = 0.0f;
    clearMaterializedTopology();
}

void VoronoiRuntime::setSeedPositions(
    const std::vector<glm::vec4>& positions,
    const std::array<glm::vec3, 8>& domainCorners) {
    inSeedPositions = positions;
    inPointDomainCorners = domainCorners;
}

void VoronoiRuntime::setGlobalGeometry(
    const std::vector<uint32_t>& runtimeModelIds,
    const std::vector<std::vector<glm::vec3>>& positions,
    const std::vector<std::vector<uint32_t>>& triangleIndices,
    const std::vector<std::vector<glm::vec3>>& surfacePositions,
    const std::vector<std::vector<uint32_t>>& surfaceTriangleIndices,
    float volumePadding) {
    inGlobalRemeshRuntimeModelIds = runtimeModelIds;
    inGlobalRemeshPositions = positions;
    inGlobalRemeshTriangleIndices = triangleIndices;
    inGlobalRemeshSurfacePositions = surfacePositions;
    inGlobalRemeshSurfaceTriangleIndices = surfaceTriangleIndices;
    inGlobalVolumePadding = volumePadding;
    voronoiReady = false;
}

void VoronoiRuntime::clearGeometry() {
    invalidateMaterialization();
}

void VoronoiRuntime::setParams(int updatedVoxelResolution) {
    if (inVoxelResolution == updatedVoxelResolution) {
        return;
    }

    inVoxelResolution = updatedVoxelResolution;
    invalidateMaterialization();
}

void VoronoiRuntime::setLloydParams(
    int iterations,
    float volumeThreshold,
    float damping) {
    const int nextIterations = iterations < 0 ? 0 : iterations;
    const float nextThreshold =
        volumeThreshold > 0.0f && std::isfinite(volumeThreshold)
            ? volumeThreshold
            : voronoi::DefaultLloydVolumeThresholdFraction;
    const float nextDamping =
        damping > 0.0f && damping <= 1.0f && std::isfinite(damping)
            ? damping
            : voronoi::DefaultLloydDamping;
    if (inLloydIterations == nextIterations &&
        inLloydVolumeThreshold == nextThreshold &&
        inLloydDamping == nextDamping) {
        return;
    }

    inLloydIterations = nextIterations;
    inLloydVolumeThreshold = nextThreshold;
    inLloydDamping = nextDamping;
    invalidateMaterialization();
}

void VoronoiRuntime::setContactRange(float range) {
    const float nextRange =
        range > 0.0f && std::isfinite(range)
            ? range
            : voronoi::DefaultContactRange;
    if (inContactRange == nextRange) {
        return;
    }

    inContactRange = nextRange;
    invalidateMaterialization();
}

void VoronoiRuntime::markReady() {
    voronoiReady = true;
}

void VoronoiRuntime::cleanup() {
    invalidateMaterialization();
}


