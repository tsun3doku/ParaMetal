#include "RuntimeRemeshComputeTransport.hpp"

#include "runtime/RuntimeProductManager.hpp"

ComputeApplyResult RuntimeRemeshComputeTransport::apply(
    uint64_t socketKey,
    const RemeshPackage& package,
    RemeshProduct& outProduct) {
    if (!controller || !products || socketKey == 0) {
        return ComputeApplyResult::Failed;
    }

    if (!products->isCurrent<ModelProduct>(package.sourceModelProduct)) {
        return ComputeApplyResult::Failed;
    }
    const ModelProduct* modelProduct = products->resolve<ModelProduct>(package.sourceModelProduct);
    if (!modelProduct || modelProduct->runtimeModelId == 0) {
        return ComputeApplyResult::Failed;
    }

    const uint64_t computeHash = package.hashes.geometry;
    if (controller->getCurrentComputeHash(socketKey) == computeHash) {
        return ComputeApplyResult::Unchanged;
    }

    RemeshController::Config config{};
    config.pointPositions = package.sourceGeometry.pointPositions;
    config.triangleIndices = package.sourceGeometry.triangleIndices;
    config.iterations = package.iterations;
    config.minAngleDegrees = package.minAngleDegrees;
    config.maxEdgeLength = package.maxEdgeLength;
    config.stepSize = package.stepSize;
    config.runtimeModelId = modelProduct->runtimeModelId;
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

void RuntimeRemeshComputeTransport::commitPending(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->commitPending(socketKey);
}

void RuntimeRemeshComputeTransport::cancelPending(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->cancelCandidate(socketKey);
}

void RuntimeRemeshComputeTransport::remove(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->remove(socketKey);
}
