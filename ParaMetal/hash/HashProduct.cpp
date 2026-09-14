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
    p.hashes.thermal = 0;
}

void HashProduct::seal(RemeshProduct& p) {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combine(hash, p.runtimeModelId);
    HashBuilder::combinePodVector(hash, p.geometryPositions);
    HashBuilder::combinePodVector(hash, p.geometryTriangleIndices);
    HashBuilder::combinePodVector(hash, p.surfacePositions);
    HashBuilder::combinePodVector(hash, p.surfaceNormals);
    HashBuilder::combinePodVector(hash, p.surfaceTriangleIndices);
    combineVkBuffer(hash, p.intrinsicTriangleBuffer);
    HashBuilder::combine(hash, p.intrinsicTriangleBufferOffset);
    combineVkBuffer(hash, p.intrinsicVertexBuffer);
    HashBuilder::combine(hash, p.intrinsicVertexBufferOffset);
    HashBuilder::combine(hash, p.intrinsicTriangleCount);
    HashBuilder::combine(hash, p.intrinsicVertexCount);
    HashBuilder::combinePod(hash, p.averageTriangleArea);

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
    p.hashes.thermal = 0;
}

void HashProduct::seal(VoronoiProduct& p) {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combine(hash, p.candidateNodeCount);
    combineVkBuffer(hash, p.candidateNeighborIndicesBuffer);
    HashBuilder::combine(hash, p.candidateNeighborIndicesBufferOffset);
    combineVkBuffer(hash, p.seedPositionBuffer);
    HashBuilder::combine(hash, p.seedPositionBufferOffset);
    combineVkBuffer(hash, p.candidatePointVertexBuffer);
    HashBuilder::combine(hash, p.candidatePointVertexBufferOffset);
    combineVkBuffer(hash, p.contactVertexBuffer);
    HashBuilder::combine(hash, p.contactVertexBufferOffset);
    HashBuilder::combine(hash, uint64_t(p.contactVertexCount));
    HashBuilder::combinePodVector(hash, p.globalSeedPositions);
    HashBuilder::combinePodVector(hash, p.fragmentInstanceIds);
    HashBuilder::combinePodVector(hash, p.fragmentSeedIds);
    HashBuilder::combinePodVector(hash, p.fragmentCentroids);
    HashBuilder::combinePodVector(hash, p.fragmentVolumes);
    HashBuilder::combineFloat(hash, p.nominalCellSize);
    HashBuilder::combinePodVector(hash, p.faceFragmentA);
    HashBuilder::combinePodVector(hash, p.faceFragmentB);
    HashBuilder::combinePodVector(hash, p.faceAreas);
    HashBuilder::combinePodVector(hash, p.faceOrthogonalDistancesA);
    HashBuilder::combinePodVector(hash, p.faceOrthogonalDistancesB);
    HashBuilder::combinePodVector(hash, p.faceNormals);
    HashBuilder::combinePodVector(hash, p.contactFaceFragmentA);
    HashBuilder::combinePodVector(hash, p.contactFaceFragmentB);
    HashBuilder::combinePodVector(hash, p.contactFaceAreas);
    HashBuilder::combinePodVector(hash, p.contactFaceOrthogonalDistancesA);
    HashBuilder::combinePodVector(hash, p.contactFaceOrthogonalDistancesB);
    HashBuilder::combinePodVector(hash, p.contactFaceGaps);
    HashBuilder::combinePodVector(hash, p.boundaryFaceFragments);
    HashBuilder::combinePodVector(hash, p.boundaryFaceDomainFaces);
    HashBuilder::combinePodVector(hash, p.boundaryFaceAreas);
    HashBuilder::combinePodVector(hash, p.boundaryFaceOrthogonalDistances);
    HashBuilder::combinePodVector(hash, p.instanceFragmentCounts);
    HashBuilder::combinePodVector(hash, p.globalSdfValues);
    HashBuilder::combine(hash, uint64_t(p.globalFragmentCount));
    HashBuilder::combine(hash, uint64_t(p.globalFaceCount));
    HashBuilder::combine(hash, uint64_t(p.globalContactFaceCount));
    HashBuilder::combine(hash, uint64_t(p.globalBoundaryFaceCount));
    HashBuilder::combinePod(hash, p.globalSdfGridMin);
    HashBuilder::combinePod(hash, p.globalSdfGridDim);
    HashBuilder::combineFloat(hash, p.globalSdfCellSize);
    HashBuilder::combine(hash, static_cast<uint64_t>(p.globalDisplayRuntimeModelIds.size()));
    for (size_t i = 0; i < p.globalDisplayRuntimeModelIds.size(); ++i) {
        HashBuilder::combine(hash, p.globalDisplayRuntimeModelIds[i]);
        combineVkBuffer(hash, p.globalDisplayVertexBuffers[i]);
        HashBuilder::combine(hash, p.globalDisplayVertexBufferOffsets[i]);
        combineVkBuffer(hash, p.globalDisplayFaceIndexBuffers[i]);
        HashBuilder::combine(hash, p.globalDisplayFaceIndexBufferOffsets[i]);
        combineVkBuffer(hash, p.globalDisplayCandidateBuffers[i]);
        HashBuilder::combine(hash, p.globalDisplayCandidateBufferOffsets[i]);
    }

    p.hashes.full = hash;
    p.hashes.geometry = hash;
    p.hashes.simulation = hash;
    p.hashes.display = hash;
    p.hashes.thermal = 0;
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

void HashProduct::seal(HeatProduct& p) {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combinePodVector(hash, p.modelRuntimeModelIds);
    HashBuilder::combine(hash, static_cast<uint64_t>(p.modelSurfaceBuffers.size()));
    for (size_t i = 0; i < p.modelSurfaceBuffers.size(); ++i) {
        combineVkBuffer(hash, p.modelSurfaceBuffers[i]);
        HashBuilder::combine(hash, p.modelSurfaceBufferOffsets[i]);
    }
    HashBuilder::combinePodVector(hash, p.modelSurfacePointCounts);
    HashBuilder::combine(hash, static_cast<uint64_t>(p.modelSurfaceGradientBuffers.size()));
    for (size_t i = 0; i < p.modelSurfaceGradientBuffers.size(); ++i) {
        combineVkBuffer(hash, p.modelSurfaceGradientBuffers[i]);
        HashBuilder::combine(hash, p.modelSurfaceGradientBufferOffsets[i]);
    }
    HashBuilder::combine(hash, static_cast<uint64_t>(p.hasFluidVolume ? 1u : 0u));
    HashBuilder::combine(hash, reinterpret_cast<uint64_t>(p.fluidTemperatureVolumeView));

    p.hashes.full = hash;
    p.hashes.simulation = hash;
    p.hashes.geometry = hash;
    p.hashes.display = hash;
    p.hashes.thermal = 0;
}
