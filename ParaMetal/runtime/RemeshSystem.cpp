#include "RemeshSystem.hpp"

#include "runtime/RuntimeProducts.hpp"

RemeshSystem::RemeshSystem(Remesher& remesher)
    : remesher(remesher) {
}

RemeshSystem::~RemeshSystem() {
    freeGpu();
}

void RemeshSystem::setSourceGeometry(const std::vector<float>& updatedPointPositions, const std::vector<uint32_t>& updatedTriangleIndices) {
    pointPositions = updatedPointPositions;
    triangleIndices = updatedTriangleIndices;
}

void RemeshSystem::setParams(int nextIterations, float nextMinAngleDegrees, float nextMaxEdgeLength, float nextStepSize) {
    iterations = nextIterations;
    minAngleDegrees = nextMinAngleDegrees;
    maxEdgeLength = nextMaxEdgeLength;
    stepSize = nextStepSize;
}

void RemeshSystem::setRuntimeModelId(uint32_t nextRuntimeModelId) {
    modelRuntimeId = nextRuntimeModelId;
}

bool RemeshSystem::remesh() {
    mesh = {};

    if (!remesher.remesh(
            pointPositions,
            triangleIndices,
            iterations,
            minAngleDegrees,
            maxEdgeLength,
            stepSize,
            mesh,
            gpuResources)) {
        remesher.cleanupGpuResources(gpuResources);
        mesh = {};
        return false;
    }

    if (gpuResources.viewS == VK_NULL_HANDLE ||
        gpuResources.intrinsicTriangleBuffer == VK_NULL_HANDLE ||
        gpuResources.intrinsicVertexBuffer == VK_NULL_HANDLE) {
        remesher.cleanupGpuResources(gpuResources);
        mesh = {};
        return false;
    }

    return true;
}

void RemeshSystem::copyGpu(RemeshProduct& product) const {
    product.intrinsicTriangleCount = gpuResources.triangleCount;
    product.intrinsicVertexCount = gpuResources.vertexCount;
    product.averageTriangleArea = gpuResources.averageTriangleArea;

    product.intrinsicTriangleBuffer = gpuResources.intrinsicTriangleBuffer;
    product.intrinsicTriangleBufferOffset = gpuResources.triangleGeometryOffset;
    product.intrinsicVertexBuffer = gpuResources.intrinsicVertexBuffer;
    product.intrinsicVertexBufferOffset = gpuResources.vertexGeometryOffset;

    product.supportingHalfedgeBuffer = gpuResources.bufferS;
    product.supportingHalfedgeOffset = gpuResources.offsetS;
    product.supportingHalfedgeView = gpuResources.viewS;
    product.supportingAngleBuffer = gpuResources.bufferA;
    product.supportingAngleOffset = gpuResources.offsetA;
    product.supportingAngleView = gpuResources.viewA;
    product.halfedgeBuffer = gpuResources.bufferH;
    product.halfedgeOffset = gpuResources.offsetH;
    product.halfedgeView = gpuResources.viewH;
    product.edgeBuffer = gpuResources.bufferE;
    product.edgeOffset = gpuResources.offsetE;
    product.edgeView = gpuResources.viewE;
    product.triangleBuffer = gpuResources.bufferT;
    product.triangleOffset = gpuResources.offsetT;
    product.triangleView = gpuResources.viewT;
    product.lengthBuffer = gpuResources.bufferL;
    product.lengthOffset = gpuResources.offsetL;
    product.lengthView = gpuResources.viewL;
    product.inputHalfedgeBuffer = gpuResources.bufferHInput;
    product.inputHalfedgeOffset = gpuResources.offsetHInput;
    product.inputHalfedgeView = gpuResources.viewHInput;
    product.inputEdgeBuffer = gpuResources.bufferEInput;
    product.inputEdgeOffset = gpuResources.offsetEInput;
    product.inputEdgeView = gpuResources.viewEInput;
    product.inputTriangleBuffer = gpuResources.bufferTInput;
    product.inputTriangleOffset = gpuResources.offsetTInput;
    product.inputTriangleView = gpuResources.viewTInput;
    product.inputLengthBuffer = gpuResources.bufferLInput;
    product.inputLengthOffset = gpuResources.offsetLInput;
    product.inputLengthView = gpuResources.viewLInput;
}

void RemeshSystem::freeGpu() {
    remesher.cleanupGpuResources(gpuResources);
}
