#include "RuntimePackageManager.hpp"

void RuntimePackageManager::beginCompile() {
    staleSockets.clear();
    retainedSockets.clear();
    clearPending();
    collectKeys(modelPackages);
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
    eraseFrom(pointPackages, pendingPointPackages, socketKey);
    eraseFrom(remeshPackages, pendingRemeshPackages, socketKey);
    eraseFrom(voronoiPackages, pendingVoronoiPackages, socketKey);
    eraseFrom(heatPackages, pendingHeatPackages, socketKey);
    eraseFrom(cfdPackages, pendingCfdPackages, socketKey);
    staleSockets.erase(socketKey);
    retainedSockets.erase(socketKey);
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
    eraseStale(pointPackages);
    eraseStale(remeshPackages);
    eraseStale(voronoiPackages);
    eraseStale(heatPackages);
    eraseStale(cfdPackages);
    staleSockets.clear();
}

void RuntimePackageManager::commitStaged() {
    commitPendingInto(modelPackages, pendingModelPackages);
    commitPendingInto(pointPackages, pendingPointPackages);
    commitPendingInto(remeshPackages, pendingRemeshPackages);
    commitPendingInto(voronoiPackages, pendingVoronoiPackages);
    commitPendingInto(heatPackages, pendingHeatPackages);
    commitPendingInto(cfdPackages, pendingCfdPackages);
    clearPending();
}

void RuntimePackageManager::commitProcessed(const std::unordered_set<uint64_t>& processedSockets) {
    commitProcessedInto(modelPackages, pendingModelPackages, processedSockets);
    commitProcessedInto(pointPackages, pendingPointPackages, processedSockets);
    commitProcessedInto(remeshPackages, pendingRemeshPackages, processedSockets);
    commitProcessedInto(voronoiPackages, pendingVoronoiPackages, processedSockets);
    commitProcessedInto(heatPackages, pendingHeatPackages, processedSockets);
    commitProcessedInto(cfdPackages, pendingCfdPackages, processedSockets);
    clearPending();
    staleSockets.clear();
    retainedSockets.clear();
}

void RuntimePackageManager::clearPending() {
    pendingModelPackages.clear();
    pendingPointPackages.clear();
    pendingRemeshPackages.clear();
    pendingVoronoiPackages.clear();
    pendingHeatPackages.clear();
    pendingCfdPackages.clear();
}

void RuntimePackageManager::clearAll() {
    clearPending();
    modelPackages.clear();
    pointPackages.clear();
    remeshPackages.clear();
    voronoiPackages.clear();
    heatPackages.clear();
    cfdPackages.clear();
    staleSockets.clear();
    retainedSockets.clear();
}
