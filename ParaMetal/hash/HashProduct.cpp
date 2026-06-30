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

void HashProduct::seal(ModelProduct& p) {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combine(hash, p.runtimeModelId);
    combineVkBuffer(hash, p.vertexBuffer);
    HashBuilder::combine(hash, p.vertexBufferOffset);
    combineVkBuffer(hash, p.indexBuffer);
    HashBuilder::combine(hash, p.indexBufferOffset);
    HashBuilder::combine(hash, p.indexCount);
    combineVkBuffer(hash, p.renderVertexBuffer);
    HashBuilder::combine(hash, p.renderVertexBufferOffset);
    combineVkBuffer(hash, p.renderIndexBuffer);
    HashBuilder::combine(hash, p.renderIndexBufferOffset);
    HashBuilder::combine(hash, p.renderIndexCount);
    HashBuilder::combinePod(hash, p.modelMatrix);

    p.hashes.full = hash;
    p.hashes.geometry = hash;
    p.hashes.simulation = hash;
    p.hashes.display = hash;
    p.hashes.thermal = 0;
}

void HashProduct::seal(RemeshProduct& p) {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combine(hash, p.runtimeModelId);
    HashBuilder::combinePodVector(hash, p.geometryPositions);
    HashBuilder::combinePodVector(hash, p.geometryTriangleIndices);
    HashBuilder::combinePodVector(hash, p.intrinsicMesh.vertices);
    HashBuilder::combinePodVector(hash, p.intrinsicMesh.indices);
    HashBuilder::combinePodVector(hash, p.intrinsicMesh.faceIds);
    HashBuilder::combinePodVector(hash, p.intrinsicMesh.triangles);
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
    HashBuilder::combine(hash, p.nodeCount);
    HashBuilder::combine(hash, p.simNodeCount);
    HashBuilder::combine(hash, static_cast<uint64_t>(p.isPointDomain ? 1u : 0u));
    HashBuilder::combine(hash, p.runtimeModelId);
    HashBuilder::combinePodVector(hash, p.seedFlags);
    HashBuilder::combinePodVector(hash, p.seedPositions);
    HashBuilder::combine(hash, p.occupancyPointCount);
    HashBuilder::combinePodVector(hash, p.voronoiToSim);
    HashBuilder::combinePodVector(hash, p.simToVoronoi);
    HashBuilder::combinePodVector(hash, p.simNodeVolumes);
    combineVkBuffer(hash, p.nodeBuffer);
    HashBuilder::combine(hash, p.nodeBufferOffset);
    combineVkBuffer(hash, p.voronoiNeighborBuffer);
    HashBuilder::combine(hash, p.voronoiNeighborBufferOffset);
    combineVkBuffer(hash, p.voronoiNeighborIndicesBuffer);
    HashBuilder::combine(hash, p.voronoiNeighborIndicesBufferOffset);
    combineVkBuffer(hash, p.voronoiInterfaceAreasBuffer);
    HashBuilder::combine(hash, p.voronoiInterfaceAreasBufferOffset);
    combineVkBuffer(hash, p.voronoiInterfaceNeighborIdsBuffer);
    HashBuilder::combine(hash, p.voronoiInterfaceNeighborIdsBufferOffset);
    combineVkBuffer(hash, p.voronoiGMLSInterfaceBuffer);
    HashBuilder::combine(hash, p.voronoiGMLSInterfaceBufferOffset);
    combineVkBuffer(hash, p.simNodeBuffer);
    HashBuilder::combine(hash, p.simNodeBufferOffset);
    combineVkBuffer(hash, p.simGMLSInterfaceBuffer);
    HashBuilder::combine(hash, p.simGMLSInterfaceBufferOffset);
    HashBuilder::combine(hash, p.simGMLSInterfaceCount);
    combineVkBuffer(hash, p.voronoiSeedFlagsBuffer);
    HashBuilder::combine(hash, p.voronoiSeedFlagsBufferOffset);
    combineVkBuffer(hash, p.seedPositionBuffer);
    HashBuilder::combine(hash, p.seedPositionBufferOffset);
    combineVkBuffer(hash, p.occupancyPointBuffer);
    HashBuilder::combine(hash, p.occupancyPointBufferOffset);
    combineVkBuffer(hash, p.candidateBuffer);
    HashBuilder::combine(hash, p.candidateBufferOffset);
    combineVkBuffer(hash, p.gmlsSurfaceStencilBuffer);
    HashBuilder::combine(hash, p.gmlsSurfaceStencilBufferOffset);
    combineVkBuffer(hash, p.gmlsSurfaceWeightBuffer);
    HashBuilder::combine(hash, p.gmlsSurfaceWeightBufferOffset);
    HashBuilder::combine(hash, p.gmlsSurfaceWeightCount);
    combineVkBuffer(hash, p.gmlsSurfaceGradientWeightBuffer);
    HashBuilder::combine(hash, p.gmlsSurfaceGradientWeightBufferOffset);
    HashBuilder::combine(hash, p.gmlsSurfaceGradientWeightCount);

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
    p.hashes.thermal = 0;
}

void HashProduct::seal(ContactProduct& p) {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combine(hash, p.coupling.modelARuntimeModelId);
    HashBuilder::combine(hash, p.coupling.modelBRuntimeModelId);
    HashBuilder::combinePodVector(hash, p.coupling.modelBTriangleIndices);
    HashBuilder::combine(hash, p.coupling.contactPairCount);
    HashBuilder::combinePodVector(hash, p.coupling.contactPairs);
    HashBuilder::combine(hash, p.modelARuntimeModelId);
    HashBuilder::combine(hash, p.modelBRuntimeModelId);
    HashBuilder::combinePodVector(hash, p.outlineVertices);
    HashBuilder::combinePodVector(hash, p.correspondenceVertices);
    combineVkBuffer(hash, p.contactPairBuffer);
    HashBuilder::combine(hash, p.contactPairBufferOffset);

    p.hashes.full = hash;
    p.hashes.geometry = hash;
    p.hashes.simulation = hash;
    p.hashes.display = hash;
    p.hashes.thermal = 0;
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

    p.hashes.full = hash;
    p.hashes.simulation = hash;
    p.hashes.geometry = hash;
    p.hashes.display = hash;
    p.hashes.thermal = 0;
}
