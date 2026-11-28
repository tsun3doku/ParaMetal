#include "RuntimePackageManager.hpp"

void RuntimePackageManager::beginCompile() {
    staleSockets.clear();
    retainedSockets.clear();
    clearPending();
    collectKeys(modelPackages);
    collectKeys(groupPackages);
    collectKeys(pointPackages);
    collectKeys(remeshPackages);
    collectKeys(voronoiPackages);
    collectKeys(heatPackages);
    collectKeys(cfdPackages);
}

void RuntimePackageManager::erasePackagesForSocket(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }
    eraseFrom(modelPackages, pendingModelPackages, socketKey);
    eraseFrom(groupPackages, pendingGroupPackages, socketKey);
    eraseFrom(pointPackages, pendingPointPackages, socketKey);
    eraseFrom(remeshPackages, pendingRemeshPackages, socketKey);
    eraseFrom(voronoiPackages, pendingVoronoiPackages, socketKey);
    eraseFrom(heatPackages, pendingHeatPackages, socketKey);
    eraseFrom(cfdPackages, pendingCfdPackages, socketKey);
    staleSockets.erase(socketKey);
    retainedSockets.erase(socketKey);
}

void RuntimePackageManager::retainCommitted(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }
    pendingModelPackages.erase(socketKey);
    pendingGroupPackages.erase(socketKey);
    pendingPointPackages.erase(socketKey);
    pendingRemeshPackages.erase(socketKey);
    pendingVoronoiPackages.erase(socketKey);
    pendingHeatPackages.erase(socketKey);
    pendingCfdPackages.erase(socketKey);
    staleSockets.erase(socketKey);
    retainedSockets.insert(socketKey);
}

void RuntimePackageManager::setProductHandle(uint64_t socketKey, const ProductHandle& handle) {
    setProductHandle(socketKey, handle, modelPackages, pendingModelPackages);
    setProductHandle(socketKey, handle, pointPackages, pendingPointPackages);
    setProductHandle(socketKey, handle, remeshPackages, pendingRemeshPackages);
    setProductHandle(socketKey, handle, voronoiPackages, pendingVoronoiPackages);
    setProductHandle(socketKey, handle, heatPackages, pendingHeatPackages);
    setProductHandle(socketKey, handle, cfdPackages, pendingCfdPackages);
}

    const std::vector<uint64_t>* RuntimePackageManager::displayDependencyKeysFor(uint64_t socketKey) const {
    if (const GroupPackage* package = findAny<GroupPackage>(socketKey)) {
        return &package->displayDependencyKeys;
    }
    if (const ModelPackage* package = findAny<ModelPackage>(socketKey)) {
        return &package->displayDependencyKeys;
    }
    if (const PointPackage* package = findAny<PointPackage>(socketKey)) {
        return &package->displayDependencyKeys;
    }
    if (const RemeshPackage* package = findAny<RemeshPackage>(socketKey)) {
        return &package->displayDependencyKeys;
    }
    if (const VoronoiPackage* package = findAny<VoronoiPackage>(socketKey)) {
        return &package->displayDependencyKeys;
    }
    if (const HeatPackage* package = findAny<HeatPackage>(socketKey)) {
        return &package->displayDependencyKeys;
    }
    if (const CFDPackage* package = findAny<CFDPackage>(socketKey)) {
        return &package->displayDependencyKeys;
    }
    return nullptr;
}

void RuntimePackageManager::destroyStale() {
    eraseStale(modelPackages);
    eraseStale(groupPackages);
    eraseStale(pointPackages);
    eraseStale(remeshPackages);
    eraseStale(voronoiPackages);
    eraseStale(heatPackages);
    eraseStale(cfdPackages);
    staleSockets.clear();
}

void RuntimePackageManager::preserveModelComputeData(
    ModelPackage& incoming,
    ModelPackage& previous) {
    if (previous.hashes.geometry != incoming.hashes.geometry ||
        !previous.hasValidProduct() ||
        !incoming.hasValidProduct() ||
        !isSamePublication(previous.productHandle, incoming.productHandle)) {
        return;
    }
    if (!incoming.vertexPositions.empty() ||
        !incoming.triangleIndices.empty() ||
        !incoming.renderPositions.empty() ||
        !incoming.renderNormals.empty() ||
        !incoming.renderTexcoords.empty() ||
        !incoming.renderIndices.empty()) {
        return;
    }
    incoming.vertexPositions = std::move(previous.vertexPositions);
    incoming.triangleIndices = std::move(previous.triangleIndices);
    incoming.renderPositions = std::move(previous.renderPositions);
    incoming.renderNormals = std::move(previous.renderNormals);
    incoming.renderTexcoords = std::move(previous.renderTexcoords);
    incoming.renderIndices = std::move(previous.renderIndices);
}

void RuntimePackageManager::commitModelPendingInto(
    std::unordered_map<uint64_t, ModelPackage>& committed,
    std::unordered_map<uint64_t, ModelPackage>& pending) {
    for (auto& [socketKey, package] : pending) {
        const auto previous = committed.find(socketKey);
        if (previous != committed.end()) {
            preserveModelComputeData(package, previous->second);
        }
        committed.insert_or_assign(socketKey, std::move(package));
    }
}

void RuntimePackageManager::commitProcessedModelInto(
    std::unordered_map<uint64_t, ModelPackage>& committed,
    std::unordered_map<uint64_t, ModelPackage>& pending,
    const std::unordered_set<uint64_t>& processedSockets) {
    for (auto& [socketKey, package] : pending) {
        if (processedSockets.find(socketKey) == processedSockets.end()) {
            continue;
        }
        const auto previous = committed.find(socketKey);
        if (previous != committed.end()) {
            preserveModelComputeData(package, previous->second);
        }
        committed.insert_or_assign(socketKey, std::move(package));
    }
}

void RuntimePackageManager::commitStaged() {
    for (const auto& [socketKey, package] : pendingGroupPackages) {
        (void)package;
        eraseNonGroupPackagesForSocket(socketKey);
    }
    commitModelPendingInto(modelPackages, pendingModelPackages);
    commitPendingInto(groupPackages, pendingGroupPackages);
    commitPendingInto(pointPackages, pendingPointPackages);
    commitPendingInto(remeshPackages, pendingRemeshPackages);
    commitPendingInto(voronoiPackages, pendingVoronoiPackages);
    commitPendingInto(heatPackages, pendingHeatPackages);
    commitPendingInto(cfdPackages, pendingCfdPackages);
    clearPending();
    retainedSockets.clear();
}

void RuntimePackageManager::commitProcessed(const std::unordered_set<uint64_t>& processedSockets) {
    for (const auto& [socketKey, package] : pendingGroupPackages) {
        (void)package;
        if (processedSockets.find(socketKey) != processedSockets.end()) {
            eraseNonGroupPackagesForSocket(socketKey);
        }
    }
    commitProcessedModelInto(modelPackages, pendingModelPackages, processedSockets);
    commitProcessedInto(groupPackages, pendingGroupPackages, processedSockets);
    commitProcessedInto(pointPackages, pendingPointPackages, processedSockets);
    commitProcessedInto(remeshPackages, pendingRemeshPackages, processedSockets);
    commitProcessedInto(voronoiPackages, pendingVoronoiPackages, processedSockets);
    commitProcessedInto(heatPackages, pendingHeatPackages, processedSockets);
    commitProcessedInto(cfdPackages, pendingCfdPackages, processedSockets);
    clearPending();
    staleSockets.clear();
    retainedSockets.clear();
}

void RuntimePackageManager::eraseNonGroupPackagesForSocket(
    uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }
    eraseFrom(modelPackages, pendingModelPackages, socketKey);
    eraseFrom(pointPackages, pendingPointPackages, socketKey);
    eraseFrom(remeshPackages, pendingRemeshPackages, socketKey);
    eraseFrom(voronoiPackages, pendingVoronoiPackages, socketKey);
    eraseFrom(heatPackages, pendingHeatPackages, socketKey);
    eraseFrom(cfdPackages, pendingCfdPackages, socketKey);
}

void RuntimePackageManager::clearPending() {
    pendingModelPackages.clear();
    pendingGroupPackages.clear();
    pendingPointPackages.clear();
    pendingRemeshPackages.clear();
    pendingVoronoiPackages.clear();
    pendingHeatPackages.clear();
    pendingCfdPackages.clear();
}

void RuntimePackageManager::clearAll() {
    clearPending();
    modelPackages.clear();
    groupPackages.clear();
    pointPackages.clear();
    remeshPackages.clear();
    voronoiPackages.clear();
    heatPackages.clear();
    cfdPackages.clear();
    staleSockets.clear();
    retainedSockets.clear();
}
