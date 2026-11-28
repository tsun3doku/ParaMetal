#include "RemeshSystem.hpp"

#include "runtime/RuntimeProducts.hpp"

RemeshSystem::RemeshSystem(Remesher& remesher)
    : remesher(remesher) {
}

RemeshSystem::~RemeshSystem() {
    freeGpu();
}

void RemeshSystem::setSourceGeometry(const std::vector<float>& updatedPointPositions, const std::vector<uint32_t>& updatedTriangleIndices) {
    inPointPositions = updatedPointPositions;
    inTriangleIndices = updatedTriangleIndices;
}

void RemeshSystem::setParams(int nextIterations, float nextMinAngleDegrees, float nextMaxEdgeLength, float nextStepSize) {
    inIterations = nextIterations;
    inMinAngleDegrees = nextMinAngleDegrees;
    inMaxEdgeLength = nextMaxEdgeLength;
    inStepSize = nextStepSize;
}

void RemeshSystem::setRuntimeModelId(uint32_t nextRuntimeModelId) {
    inRuntimeModelId = nextRuntimeModelId;
}

bool RemeshSystem::remesh() {
    outMesh = {};

    if (!remesher.remesh(
            inPointPositions,
            inTriangleIndices,
            inIterations,
            inMinAngleDegrees,
            inMaxEdgeLength,
            inStepSize,
            outMesh,
            outGpuResources)) {
        remesher.cleanupGpuResources(outGpuResources);
        outMesh = {};
        return false;
    }

    if (outGpuResources.viewS == VK_NULL_HANDLE ||
        outGpuResources.intrinsicTriangleBuffer == VK_NULL_HANDLE ||
        outGpuResources.intrinsicVertexBuffer == VK_NULL_HANDLE ||
        outGpuResources.faceIdBuffer == VK_NULL_HANDLE ||
        outGpuResources.faceIdView == VK_NULL_HANDLE) {
        remesher.cleanupGpuResources(outGpuResources);
        outMesh = {};
        return false;
    }

    return true;
}

void RemeshSystem::copyGpu(RemeshProduct& product) const {
    product.intrinsicTriangleCount = outGpuResources.triangleCount;
    product.intrinsicVertexCount = outGpuResources.vertexCount;
    product.averageTriangleArea = outGpuResources.averageTriangleArea;
    product.intrinsicTriangleBuffer = outGpuResources.intrinsicTriangleBuffer;
    product.intrinsicTriangleBufferOffset = outGpuResources.triangleGeometryOffset;
    product.intrinsicVertexBuffer = outGpuResources.intrinsicVertexBuffer;
    product.intrinsicVertexBufferOffset = outGpuResources.vertexGeometryOffset;
    product.faceIdBuffer = outGpuResources.faceIdBuffer;
    product.faceIdOffset = outGpuResources.faceIdOffset;
    product.faceIdView = outGpuResources.faceIdView;
    product.supportingHalfedgeBuffer = outGpuResources.bufferS;
    product.supportingHalfedgeOffset = outGpuResources.offsetS;
    product.supportingHalfedgeView = outGpuResources.viewS;
    product.supportingAngleBuffer = outGpuResources.bufferA;
    product.supportingAngleOffset = outGpuResources.offsetA;
    product.supportingAngleView = outGpuResources.viewA;
    product.halfedgeBuffer = outGpuResources.bufferH;
    product.halfedgeOffset = outGpuResources.offsetH;
    product.halfedgeView = outGpuResources.viewH;
    product.edgeBuffer = outGpuResources.bufferE;
    product.edgeOffset = outGpuResources.offsetE;
    product.edgeView = outGpuResources.viewE;
    product.triangleBuffer = outGpuResources.bufferT;
    product.triangleOffset = outGpuResources.offsetT;
    product.triangleView = outGpuResources.viewT;
    product.lengthBuffer = outGpuResources.bufferL;
    product.lengthOffset = outGpuResources.offsetL;
    product.lengthView = outGpuResources.viewL;
    product.inputHalfedgeBuffer = outGpuResources.bufferHInput;
    product.inputHalfedgeOffset = outGpuResources.offsetHInput;
    product.inputHalfedgeView = outGpuResources.viewHInput;
    product.inputEdgeBuffer = outGpuResources.bufferEInput;
    product.inputEdgeOffset = outGpuResources.offsetEInput;
    product.inputEdgeView = outGpuResources.viewEInput;
    product.inputTriangleBuffer = outGpuResources.bufferTInput;
    product.inputTriangleOffset = outGpuResources.offsetTInput;
    product.inputTriangleView = outGpuResources.viewTInput;
    product.inputLengthBuffer = outGpuResources.bufferLInput;
    product.inputLengthOffset = outGpuResources.offsetLInput;
    product.inputLengthView = outGpuResources.viewLInput;
}

void RemeshSystem::freeGpu() {
    remesher.cleanupGpuResources(outGpuResources);
}
