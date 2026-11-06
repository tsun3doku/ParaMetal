#include "RuntimeVoronoiComputeTransport.hpp"
#include "runtime/RuntimeProductManager.hpp"
#include "util/GeometryUtils.hpp"

ComputeApplyResult RuntimeVoronoiComputeTransport::apply(
    uint64_t socketKey,
    const VoronoiPackage& package,
    VoronoiProduct& outProduct) {
    if (!controller || !products || socketKey == 0) {
        return ComputeApplyResult::Failed;
    }

    const uint64_t computeHash = package.hashes.simulation;

    if (package.pointPositions.empty() ||
        package.globalRemeshProducts.size() != package.globalRemeshLocalToWorld.size() ||
        package.globalRemeshProducts.size() != package.globalModelProducts.size()) {
        return ComputeApplyResult::Failed;
    }

    if (controller->getCurrentComputeHash(socketKey) == computeHash) {
        return ComputeApplyResult::Unchanged;
    }

    VoronoiSystemComputeController::Config config{};
    config.cellSize = package.authored.cellSize;
    config.voxelResolution = package.authored.voxelResolution;
    config.lloydIterations = package.authored.lloydIterations;
    config.lloydVolumeThreshold = package.authored.lloydVolumeThreshold;
    config.lloydDamping = package.authored.lloydDamping;
    config.contactRange = package.authored.contactRange;
    config.pointDomainCorners = package.pointDomainCorners;

    config.volumePadding = package.authored.volumePadding;
    config.pointPositions = package.pointPositions;
    config.globalRemeshRuntimeModelIds.reserve(package.globalRemeshProducts.size());
    config.globalRemeshPositions.reserve(package.globalRemeshProducts.size());
    config.globalRemeshTriangleIndices.reserve(package.globalRemeshProducts.size());
    config.globalRemeshSurfacePositions.reserve(package.globalRemeshProducts.size());
    config.globalRemeshSurfaceTriangleIndices.reserve(package.globalRemeshProducts.size());
    config.globalRemeshWorldToModel.reserve(package.globalRemeshProducts.size());
    for (size_t i = 0; i < package.globalRemeshProducts.size(); ++i) {
        if (!products->isCurrent<RemeshProduct>(package.globalRemeshProducts[i])) {
            return ComputeApplyResult::Failed;
        }
        const RemeshProduct* remeshProduct = products->resolve<RemeshProduct>(package.globalRemeshProducts[i]);
        if (!remeshProduct || remeshProduct->runtimeModelId == 0) {
            return ComputeApplyResult::Failed;
        }
        const glm::mat4 transform = toMat4(package.globalRemeshLocalToWorld[i]);
        std::vector<glm::vec3> positions;
        positions.reserve(remeshProduct->geometryPositions.size());
        for (const glm::vec3& position : remeshProduct->geometryPositions) {
            positions.push_back(glm::vec3(transform * glm::vec4(position, 1.0f)));
        }
        std::vector<glm::vec3> surfacePositions;
        surfacePositions.reserve(remeshProduct->surfacePositions.size());
        for (const glm::vec3& position : remeshProduct->surfacePositions) {
            surfacePositions.push_back(glm::vec3(transform * glm::vec4(position, 1.0f)));
        }
        config.globalRemeshRuntimeModelIds.push_back(remeshProduct->runtimeModelId);
        config.globalRemeshPositions.push_back(std::move(positions));
        config.globalRemeshTriangleIndices.push_back(remeshProduct->geometryTriangleIndices);
        config.globalRemeshSurfacePositions.push_back(std::move(surfacePositions));
        config.globalRemeshSurfaceTriangleIndices.push_back(remeshProduct->surfaceTriangleIndices);
        config.globalRemeshWorldToModel.push_back(toMatrixArray(glm::inverse(transform)));
    }

    config.computeHash = computeHash;

    if (!controller->apply(socketKey, config)) {
        return ComputeApplyResult::Failed;
    }

    outProduct = {};
    if (!controller->buildProduct(socketKey, outProduct)) {
        controller->cancelCandidate(socketKey);
        return ComputeApplyResult::Failed;
    }
    return ComputeApplyResult::Prepared;
}

void RuntimeVoronoiComputeTransport::commitPending(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->commitPending(socketKey);
}

void RuntimeVoronoiComputeTransport::cancelPending(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->cancelCandidate(socketKey);
}

void RuntimeVoronoiComputeTransport::remove(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->remove(socketKey);
}
