#include "HashProduct.hpp"
#include "HashBuilder.hpp"

#include "runtime/RuntimeProducts.hpp"

static void combineVkBuffer(uint64_t& hash, VkBuffer handle) {
    HashBuilder::combine(hash, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle)));
}

static void combineVkBufferView(uint64_t& hash, VkBufferView handle) {
    HashBuilder::combine(hash, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle)));
}

static void combineVkDeviceSize(uint64_t& hash, VkDeviceSize handle) {
    HashBuilder::combine(hash, static_cast<uint64_t>(handle));
}

static void combineVkImageView(uint64_t& hash, VkImageView handle) {
    HashBuilder::combine(hash, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle)));
}

static void combineVkSampler(uint64_t& hash, VkSampler handle) {
    HashBuilder::combine(hash, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle)));
}

static void combineVec3(uint64_t& hash, const glm::vec3& value) {
    HashBuilder::combinePod(hash, value.x);
    HashBuilder::combinePod(hash, value.y);
    HashBuilder::combinePod(hash, value.z);
}

static void combineDVec3(uint64_t& hash, const glm::dvec3& value) {
    HashBuilder::combinePod(hash, value.x);
    HashBuilder::combinePod(hash, value.y);
    HashBuilder::combinePod(hash, value.z);
}

static void combineIVec3(uint64_t& hash, const glm::ivec3& value) {
    HashBuilder::combine(hash, static_cast<uint64_t>(value.x));
    HashBuilder::combine(hash, static_cast<uint64_t>(value.y));
    HashBuilder::combine(hash, static_cast<uint64_t>(value.z));
}

void HashProduct::seal(ModelProduct& p) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, p.runtimeModelId);
    combineVkBuffer(geometryHash, p.vertexBuffer);
    HashBuilder::combine(geometryHash, p.vertexBufferOffset);
    combineVkBuffer(geometryHash, p.indexBuffer);
    HashBuilder::combine(geometryHash, p.indexBufferOffset);
    HashBuilder::combine(geometryHash, p.indexCount);
    combineVkBuffer(geometryHash, p.renderVertexBuffer);
    HashBuilder::combine(geometryHash, p.renderVertexBufferOffset);
    combineVkBuffer(geometryHash, p.renderIndexBuffer);
    HashBuilder::combine(geometryHash, p.renderIndexBufferOffset);
    HashBuilder::combine(geometryHash, p.renderIndexCount);

    p.hashes.full = geometryHash;
    p.hashes.geometry = geometryHash;
    p.hashes.simulation = geometryHash;
    p.hashes.display = geometryHash;
    p.hashes.properties = 0;
}

void HashProduct::seal(RemeshProduct& p) {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combine(hash, p.runtimeModelId);
    HashBuilder::combinePodVector(hash, p.geometryPositions);
    HashBuilder::combinePodVector(hash, p.geometryTriangleIndices);
    HashBuilder::combinePodVector(hash, p.surfacePositions);
    HashBuilder::combinePodVector(hash, p.surfaceNormals);
    HashBuilder::combinePodVector(hash, p.surfaceTriangleIndices);
    HashBuilder::combinePodVector(hash, p.surfaceFaceIds);
    HashBuilder::combinePodVector(hash, p.surfaceVertexIds);
    HashBuilder::combinePodVector(hash, p.surfaceEdgeIds);
    combineVkBuffer(hash, p.intrinsicTriangleBuffer);
    HashBuilder::combine(hash, p.intrinsicTriangleBufferOffset);
    combineVkBuffer(hash, p.intrinsicVertexBuffer);
    HashBuilder::combine(hash, p.intrinsicVertexBufferOffset);
    HashBuilder::combine(hash, p.intrinsicTriangleCount);
    HashBuilder::combine(hash, p.intrinsicVertexCount);
    HashBuilder::combinePod(hash, p.averageTriangleArea);
    combineVkBuffer(hash, p.faceIdBuffer);
    HashBuilder::combine(hash, p.faceIdOffset);
    combineVkBufferView(hash, p.faceIdView);

    combineVkBuffer(hash, p.supportingHalfedgeBuffer);
    HashBuilder::combine(hash, p.supportingHalfedgeOffset);
    combineVkBufferView(hash, p.supportingHalfedgeView);
    combineVkBuffer(hash, p.supportingAngleBuffer);
    HashBuilder::combine(hash, p.supportingAngleOffset);
    combineVkBufferView(hash, p.supportingAngleView);
    combineVkBuffer(hash, p.halfedgeBuffer);
    HashBuilder::combine(hash, p.halfedgeOffset);
    combineVkBufferView(hash, p.halfedgeView);
    combineVkBuffer(hash, p.edgeBuffer);
    HashBuilder::combine(hash, p.edgeOffset);
    combineVkBufferView(hash, p.edgeView);
    combineVkBuffer(hash, p.triangleBuffer);
    HashBuilder::combine(hash, p.triangleOffset);
    combineVkBufferView(hash, p.triangleView);
    combineVkBuffer(hash, p.lengthBuffer);
    HashBuilder::combine(hash, p.lengthOffset);
    combineVkBufferView(hash, p.lengthView);
    combineVkBuffer(hash, p.inputHalfedgeBuffer);
    HashBuilder::combine(hash, p.inputHalfedgeOffset);
    combineVkBufferView(hash, p.inputHalfedgeView);
    combineVkBuffer(hash, p.inputEdgeBuffer);
    HashBuilder::combine(hash, p.inputEdgeOffset);
    combineVkBufferView(hash, p.inputEdgeView);
    combineVkBuffer(hash, p.inputTriangleBuffer);
    HashBuilder::combine(hash, p.inputTriangleOffset);
    combineVkBufferView(hash, p.inputTriangleView);
    combineVkBuffer(hash, p.inputLengthBuffer);
    HashBuilder::combine(hash, p.inputLengthOffset);
    combineVkBufferView(hash, p.inputLengthView);

    p.hashes.full = hash;
    p.hashes.geometry = hash;
    p.hashes.simulation = hash;
    p.hashes.display = hash;
    p.hashes.properties = 0;
}

void HashProduct::seal(VoronoiProduct& p) {
    uint64_t geometryHash = HashBuilder::start();

    if (p.fragmentInstanceIds != nullptr) {
        const size_t fragmentCount = p.fragmentInstanceIds->size();
        HashBuilder::combine(geometryHash, static_cast<uint64_t>(fragmentCount));
        for (size_t i = 0; i < fragmentCount; ++i) {
            HashBuilder::combine(geometryHash, (*p.fragmentInstanceIds)[i]);
            HashBuilder::combine(geometryHash, (*p.fragmentSeedIds)[i]);
            combineDVec3(geometryHash, (*p.fragmentCentroids)[i]);
            HashBuilder::combinePod(geometryHash, (*p.fragmentVolumes)[i]);
        }
        const size_t faceCount = p.faceFragmentA->size();
        HashBuilder::combine(geometryHash, static_cast<uint64_t>(faceCount));
        for (size_t i = 0; i < faceCount; ++i) {
            HashBuilder::combine(geometryHash, (*p.faceFragmentA)[i]);
            HashBuilder::combine(geometryHash, (*p.faceFragmentB)[i]);
            HashBuilder::combinePod(geometryHash, (*p.faceAreasA)[i]);
            combineDVec3(geometryHash, (*p.faceAreaVectorsA)[i]);
            combineDVec3(geometryHash, (*p.faceMomentsA)[i]);
            HashBuilder::combinePod(geometryHash, (*p.faceAreasB)[i]);
            combineDVec3(geometryHash, (*p.faceAreaVectorsB)[i]);
            combineDVec3(geometryHash, (*p.faceMomentsB)[i]);
            HashBuilder::combine(geometryHash, (*p.faceTypes)[i]);
            HashBuilder::combine(geometryHash, (*p.faceDomainFaces)[i]);
        }
        HashBuilder::combinePodVector(geometryHash, *p.faceLoopOffsets);
        HashBuilder::combinePodVector(geometryHash, *p.loopVertexOffsets);
        HashBuilder::combine(geometryHash, static_cast<uint64_t>(p.loopVertices->size()));
        for (const glm::dvec3& vertex : *p.loopVertices) {
            combineDVec3(geometryHash, vertex);
        }
        HashBuilder::combine(
            geometryHash, static_cast<uint64_t>(p.contactCouplingFaceA->size()));
        for (size_t i = 0; i < p.contactCouplingFaceA->size(); ++i) {
            HashBuilder::combine(geometryHash, (*p.contactCouplingFaceA)[i]);
            HashBuilder::combine(geometryHash, (*p.contactCouplingFragmentA)[i]);
            HashBuilder::combine(geometryHash, (*p.contactCouplingFaceB)[i]);
            HashBuilder::combine(geometryHash, (*p.contactCouplingFragmentB)[i]);
            HashBuilder::combinePod(geometryHash, (*p.contactCouplingAreas)[i]);
        }
        HashBuilder::combinePodVector(geometryHash, *p.boundaryFaceIndices);
        HashBuilder::combinePodVector(geometryHash, *p.instanceFragmentCounts);
        HashBuilder::combine(geometryHash, static_cast<uint64_t>(p.seedPositions->size()));
        for (const glm::vec3& seed : *p.seedPositions) {
            combineVec3(geometryHash, seed);
        }
        HashBuilder::combinePodVector(geometryHash, *p.cellFragmentOffsets);
        HashBuilder::combinePodVector(geometryHash, *p.cellFragments);
        combineVec3(geometryHash, p.domainMin);
        combineVec3(geometryHash, p.domainMax);
        combineVec3(geometryHash, p.volumeGridMin);
        combineIVec3(geometryHash, p.volumeGridDim);
        HashBuilder::combineFloat(geometryHash, p.volumeGridCellSize);
    }

    HashBuilder::combine(geometryHash, p.spaceFragmentCount);
    HashBuilder::combine(geometryHash,
        static_cast<uint64_t>(p.hasUniqueSeedToSpaceMapping ? 1u : 0u));
    if (p.fragmentToSpaceIndex != nullptr) {
        HashBuilder::combinePodVector(geometryHash, *p.fragmentToSpaceIndex);
    }
    if (p.spaceToFragmentIndex != nullptr) {
        HashBuilder::combinePodVector(geometryHash, *p.spaceToFragmentIndex);
    }
    if (p.seedToSpaceIndex != nullptr) {
        HashBuilder::combinePodVector(geometryHash, *p.seedToSpaceIndex);
    }

    uint64_t displayHash = HashBuilder::start();
    HashBuilder::combine(displayHash, p.candidateNodeCount);
    combineVkBuffer(displayHash, p.candidateNeighborIndicesBuffer);
    HashBuilder::combine(displayHash, p.candidateNeighborIndicesBufferOffset);
    combineVkBuffer(displayHash, p.seedPositionBuffer);
    HashBuilder::combine(displayHash, p.seedPositionBufferOffset);
    combineVkBuffer(displayHash, p.candidatePointVertexBuffer);
    HashBuilder::combine(displayHash, p.candidatePointVertexBufferOffset);
    HashBuilder::combine(
        displayHash, static_cast<uint64_t>(p.globalDisplayRuntimeModelIds.size()));
    for (size_t i = 0; i < p.globalDisplayRuntimeModelIds.size(); ++i) {
        HashBuilder::combine(displayHash, p.globalDisplayRuntimeModelIds[i]);
        combineVkBuffer(displayHash, p.globalDisplayVertexBuffers[i]);
        HashBuilder::combine(displayHash, p.globalDisplayVertexBufferOffsets[i]);
        combineVkBuffer(displayHash, p.globalDisplayFaceIndexBuffers[i]);
        HashBuilder::combine(displayHash, p.globalDisplayFaceIndexBufferOffsets[i]);
        combineVkBuffer(displayHash, p.globalDisplayCandidateBuffers[i]);
        HashBuilder::combine(displayHash, p.globalDisplayCandidateBufferOffsets[i]);
    }

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, geometryHash);
    HashBuilder::combine(fullHash, displayHash);

    p.hashes.full = fullHash;
    p.hashes.geometry = geometryHash;
    p.hashes.simulation = geometryHash;
    p.hashes.display = displayHash;
    p.hashes.properties = 0;
}

void HashProduct::seal(PointProduct& p) {
    uint64_t hash = HashBuilder::start();
    combineVkBuffer(hash, p.positionBuffer);
    HashBuilder::combine(hash, p.positionBufferOffset);
    HashBuilder::combine(hash, p.pointCount);
    HashBuilder::combinePod(hash, p.modelMatrix);

    p.hashes.full = hash;
    p.hashes.geometry = hash;
    p.hashes.simulation = hash;
    p.hashes.display = hash;
}

void HashProduct::seal(CFDProduct& p, uint64_t geometryHash, uint64_t simulationHash) {
    uint64_t displayHash = HashBuilder::start();
    combineVkBuffer(displayHash, p.streamlineVertexBuffer);
    HashBuilder::combine(displayHash, static_cast<uint64_t>(p.streamlineVertexBufferOffset));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(p.streamlineVertexCount));

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, geometryHash);
    HashBuilder::combine(fullHash, simulationHash);
    HashBuilder::combine(fullHash, displayHash);

    p.hashes.full = fullHash;
    p.hashes.geometry = geometryHash;
    p.hashes.simulation = simulationHash;
    p.hashes.display = displayHash;
    p.hashes.properties = 0;
}

void HashProduct::seal(
    HeatProduct& p,
    uint64_t geometryHash,
    uint64_t simulationHash) {
    uint64_t displayHash = HashBuilder::start();
    HashBuilder::combinePodVector(displayHash, p.modelRuntimeModelIds);
    HashBuilder::combine(displayHash, static_cast<uint64_t>(p.modelSurfaceBuffers.size()));
    for (size_t i = 0; i < p.modelSurfaceBuffers.size(); ++i) {
        combineVkBuffer(displayHash, p.modelSurfaceBuffers[i]);
        HashBuilder::combine(displayHash, p.modelSurfaceBufferOffsets[i]);
    }
    HashBuilder::combinePodVector(displayHash, p.modelSurfacePointCounts);
    HashBuilder::combine(displayHash, static_cast<uint64_t>(p.modelSurfaceGradientBuffers.size()));
    for (size_t i = 0; i < p.modelSurfaceGradientBuffers.size(); ++i) {
        combineVkBuffer(displayHash, p.modelSurfaceGradientBuffers[i]);
        HashBuilder::combine(displayHash, p.modelSurfaceGradientBufferOffsets[i]);
    }
    HashBuilder::combine(displayHash, static_cast<uint64_t>(p.hasFluidVolume ? 1u : 0u));
    HashBuilder::combine(
        displayHash, reinterpret_cast<uint64_t>(p.fluidTemperatureVolumeView));

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, geometryHash);
    HashBuilder::combine(fullHash, simulationHash);
    HashBuilder::combine(fullHash, displayHash);

    p.hashes.full = fullHash;
    p.hashes.geometry = geometryHash;
    p.hashes.simulation = simulationHash;
    p.hashes.display = displayHash;
    p.hashes.properties = 0;
}
