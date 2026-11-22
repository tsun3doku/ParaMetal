#include "RuntimeModelComputeTransport.hpp"

#include "runtime/ModelComputeController.hpp"

ComputeApplyResult RuntimeModelComputeTransport::apply(
    uint64_t socketKey,
    const ModelPackage& package,
    ModelProduct& outProduct) {
    if (!controller || socketKey == 0) {
        return ComputeApplyResult::Failed;
    }

    if (controller->getCurrentComputeHash(socketKey) == package.hashes.geometry) {
        return ComputeApplyResult::Unchanged;
    }

    ModelComputeController::Config config{};
    config.vertexPositions = package.vertexPositions;
    config.triangleIndices = package.triangleIndices;
    config.renderPositions = package.renderPositions;
    config.renderNormals = package.renderNormals;
    config.renderTexcoords = package.renderTexcoords;
    config.renderIndices = package.renderIndices;
    config.computeHash = package.hashes.geometry;
    if (!controller->apply(socketKey, config)) {
        return ComputeApplyResult::Failed;
    }

    outProduct = {};
    if (!controller->buildProduct(socketKey, outProduct)) {
        return ComputeApplyResult::Failed;
    }
    return ComputeApplyResult::Prepared;
}

void RuntimeModelComputeTransport::remove(uint64_t socketKey) {
    if (!controller || socketKey == 0) {
        return;
    }
    controller->remove(socketKey);
}
