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
    config.positions = package.positions.data();
    config.positionCount = package.positions.size();
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

bool RuntimePointComputeTransport::commitPending(uint64_t socketKey) {
    return runtime && socketKey != 0 && runtime->commitPending(socketKey);
}

void RuntimePointComputeTransport::cancelPending(uint64_t socketKey) {
    if (!runtime || socketKey == 0) {
        return;
    }
    runtime->cancelCandidate(socketKey);
}

bool RuntimePointComputeTransport::remove(uint64_t socketKey) {
    if (!runtime) {
        return false;
    }
    if (socketKey == 0) {
        return true;
    }
    return runtime->disable(socketKey);
}

bool RuntimePointComputeTransport::hasAnyRetiring() const {
    return runtime && runtime->hasAnyRetiring();
}

void RuntimePointComputeTransport::quiescedClear() {
    if (runtime) {
        runtime->quiescedClear();
    }
}
