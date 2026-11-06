#include "RuntimePointComputeTransport.hpp"

#include "runtime/PointComputeRuntime.hpp"
#include "util/GeometryUtils.hpp"

ComputeApplyResult RuntimePointComputeTransport::apply(
    uint64_t socketKey,
    const PointPackage& package,
    PointProduct& outProduct) {
    if (!runtime || socketKey == 0) {
        return ComputeApplyResult::Failed;
    }

    const uint64_t inputHash = package.hashes.geometry;
    if (runtime->currentHashMatches(socketKey, inputHash)) {
        return ComputeApplyResult::Unchanged;
    }

    PointComputeRuntime::Config config{};
    config.socketKey = socketKey;
    config.positions = package.positions;
    config.modelMatrix = toMat4(package.localToWorld);
    config.computeHash = inputHash;

    if (!runtime->prepareCandidate(config)) {
        return ComputeApplyResult::Failed;
    }

    outProduct = {};
    if (!runtime->buildProduct(socketKey, outProduct)) {
        runtime->cancelCandidate(socketKey);
        return ComputeApplyResult::Failed;
    }
    return ComputeApplyResult::Prepared;
}

void RuntimePointComputeTransport::commitPending(uint64_t socketKey) {
    if (!runtime || socketKey == 0) {
        return;
    }
    runtime->commitPending(socketKey);
}

void RuntimePointComputeTransport::cancelPending(uint64_t socketKey) {
    if (!runtime || socketKey == 0) {
        return;
    }
    runtime->cancelCandidate(socketKey);
}

void RuntimePointComputeTransport::remove(uint64_t socketKey) {
    if (!runtime || socketKey == 0) {
        return;
    }
    runtime->disable(socketKey);
}
