#include "RuntimePackageController.hpp"

#include "nodegraph/NodeGraphState.hpp"
#include "runtime/ComputeTransportResult.hpp"
#include "runtime/RuntimeHeatComputeTransport.hpp"
#include "runtime/RuntimeCFDComputeTransport.hpp"
#include "runtime/RuntimeHeatDisplayTransport.hpp"
#include "runtime/RuntimeCFDDisplayTransport.hpp"
#include "runtime/RuntimeModelComputeTransport.hpp"
#include "runtime/RuntimeModelDisplayTransport.hpp"
#include "runtime/RuntimePointComputeTransport.hpp"
#include "runtime/RuntimePointDisplayTransport.hpp"
#include "runtime/RuntimeProductManager.hpp"
#include "runtime/RuntimeRemeshComputeTransport.hpp"
#include "runtime/RuntimeRemeshDisplayTransport.hpp"
#include "runtime/RuntimeVoronoiComputeTransport.hpp"
#include "runtime/RuntimeVoronoiDisplayTransport.hpp"

#include <algorithm>
#include <unordered_map>

RuntimePackageController::~RuntimePackageController() {
    clear();
}

void RuntimePackageController::clear() {
    quiescedClear();
}

RuntimePackageController::RuntimePackageController(const RuntimeTransports& runtimeTransports)
    : transports(runtimeTransports),
      productManager(std::make_unique<RuntimeProductManager>()) {
    RuntimeProductManager* products = productManager.get();
    if (transports.remeshComputeTransport) transports.remeshComputeTransport->setProducts(products);
    if (transports.voronoiComputeTransport) transports.voronoiComputeTransport->setProducts(products);
    if (transports.heatComputeTransport) transports.heatComputeTransport->setProducts(products);
    if (transports.cfdComputeTransport) transports.cfdComputeTransport->setProducts(products);
    if (transports.modelDisplayTransport) transports.modelDisplayTransport->setProducts(products);
    if (transports.pointDisplayTransport) transports.pointDisplayTransport->setProducts(products);
    if (transports.remeshDisplayTransport) transports.remeshDisplayTransport->setProducts(products);
    if (transports.voronoiDisplayTransport) transports.voronoiDisplayTransport->setProducts(products);
    if (transports.heatDisplayTransport) transports.heatDisplayTransport->setProducts(products);
    if (transports.cfdDisplayTransport) transports.cfdDisplayTransport->setProducts(products);
}

void RuntimePackageController::beginCompilation() {
    packageManager.beginCompile();
}

uint64_t RuntimePackageController::publicationRevision() const {
    const RuntimeProductManager* manager = products();
    return manager ? manager->getRevision() : 0;
}

void RuntimePackageController::clearRetiringProducts() {
    if (RuntimeProductManager* manager = products()) {
        manager->clearAllRetiring();
    }
}

static void appendRuntimeModelId(
    std::vector<uint32_t>& outIds,
    uint32_t id) {
    if (id == 0) {
        return;
    }
    for (uint32_t existing : outIds) {
        if (existing == id) {
            return;
        }
    }
    outIds.push_back(id);
}

bool RuntimePackageController::appendRuntimeModelIdsForSocket(
    uint64_t socketKey,
    std::vector<uint32_t>& outIds) const {    const RuntimeProductManager* products = this->products();
    const RuntimePackageManager& packages = packageManager;
    if (socketKey == 0 || !products) {
        return false;
    }

    if (const GroupPackage* package = packages.findAny<GroupPackage>(socketKey)) {
        if (package->surfaceProduct.type == ProductType::Model) {
            if (!products->isCurrent<ModelProduct>(package->surfaceProduct)) {
                return false;
            }
            if (const ModelProduct* product =
                    products->resolve<ModelProduct>(package->surfaceProduct)) {
                appendRuntimeModelId(outIds, product->runtimeModelId);
            }
            return true;
        }
        if (package->surfaceProduct.type == ProductType::Remesh) {
            if (!products->isCurrent<RemeshProduct>(package->surfaceProduct)) {
                return false;
            }
            if (const RemeshProduct* product =
                    products->resolve<RemeshProduct>(package->surfaceProduct)) {
                appendRuntimeModelId(outIds, product->runtimeModelId);
            }
            return true;
        }
        return false;
    }
    if (const ModelPackage* package = packages.findAny<ModelPackage>(socketKey)) {
        if (!products->isCurrent<ModelProduct>(package->productHandle)) {
            return false;
        }
        if (const ModelProduct* product =
                products->resolve<ModelProduct>(package->productHandle)) {
            appendRuntimeModelId(outIds, product->runtimeModelId);
        }
        return true;
    }
    if (const RemeshPackage* package = packages.findAny<RemeshPackage>(socketKey)) {
        if (!products->isCurrent<RemeshProduct>(package->productHandle)) {
            return false;
        }
        if (const RemeshProduct* product =
                products->resolve<RemeshProduct>(package->productHandle)) {
            appendRuntimeModelId(outIds, product->runtimeModelId);
        }
        return true;
    }
    if (const VoronoiPackage* package = packages.findAny<VoronoiPackage>(socketKey)) {
        if (!products->isCurrent<VoronoiProduct>(package->productHandle)) {
            return false;
        }
        if (const VoronoiProduct* product =
                products->resolve<VoronoiProduct>(package->productHandle)) {
            for (uint32_t modelId : product->globalDisplayRuntimeModelIds) {
                appendRuntimeModelId(outIds, modelId);
            }
        }
        return true;
    }
    if (const HeatPackage* package = packages.findAny<HeatPackage>(socketKey)) {
        if (!products->isCurrent<HeatProduct>(package->productHandle)) {
            return false;
        }
        for (const HeatModelPackage& model : package->models) {
            if (products->isCurrent<ModelProduct>(model.modelProduct)) {
                if (const ModelProduct* product =
                        products->resolve<ModelProduct>(model.modelProduct)) {
                    appendRuntimeModelId(outIds, product->runtimeModelId);
                }
            }
        }
        return true;
    }
    return packages.findAny<PointPackage>(socketKey) != nullptr;
}

bool RuntimePackageController::syncDisplay(
    const std::unordered_set<uint64_t>& displayKeys) {
    lastDisplayError_.clear();
    std::unordered_map<uint32_t, uint64_t> displayedRemeshByRuntimeModelId;
    std::unordered_set<uint64_t> sceneKeys;
    for (const auto& [socketKey, remesh] : packageManager.remeshes()) {
        if (!packageManager.isPackageCurrent(socketKey)) {
            continue;
        }
        if (displayKeys.find(socketKey) == displayKeys.end()) {
            continue;
        }
        if (remesh.sourceModelProduct.isValid() &&
            displayKeys.find(remesh.sourceModelProduct.outputSocketKey) !=
                displayKeys.end()) {
            lastDisplayError_ =
                "Conflicting display dependencies: a Remesh and its source "
                "model are both displayed.";
            return false;
        }

        std::vector<uint32_t> modelIds;
        if (!appendRuntimeModelIdsForSocket(socketKey, modelIds) ||
            modelIds.size() != 1) {
            lastDisplayError_ =
                "Displayed Remesh has no unique runtime model id.";
            return false;
        }
        const auto [it, inserted] =
            displayedRemeshByRuntimeModelId.emplace(modelIds.front(), socketKey);
        if (!inserted && it->second != socketKey) {
            lastDisplayError_ =
                "Conflicting displayed Remesh surfaces for one runtime model "
                "id.";
            return false;
        }
        if (remesh.sourceModelProduct.isValid()) {
            sceneKeys.insert(remesh.sourceModelProduct.outputSocketKey);
        }
    }

    if (transports.modelDisplayTransport &&
        !transports.modelDisplayTransport->sync(
            packageManager, displayKeys, sceneKeys)) {
        lastDisplayError_ =
            "A displayed source model could not be registered.";
        return false;
    }
    if (transports.remeshDisplayTransport) {
        if (!transports.remeshDisplayTransport->sync(
                packageManager, displayKeys)) {
            lastDisplayError_ =
                "A displayed Remesh surface is not ready.";
            return false;
        }
    }
    if (transports.voronoiDisplayTransport) {
        transports.voronoiDisplayTransport->sync(packageManager, displayKeys);
        transports.voronoiDisplayTransport->finalizeSync();
    }
    if (transports.heatDisplayTransport) {
        transports.heatDisplayTransport->sync(packageManager, displayKeys);
        transports.heatDisplayTransport->finalizeSync();
    }
    if (transports.cfdDisplayTransport) {
        transports.cfdDisplayTransport->sync(packageManager, displayKeys);
        transports.cfdDisplayTransport->finalizeSync();
    }
    if (transports.pointDisplayTransport) {
        transports.pointDisplayTransport->sync(packageManager, displayKeys);
        transports.pointDisplayTransport->finalizeSync();
    }
    return true;
}

bool RuntimePackageController::findDisplayedSurface(
    uint32_t runtimeModelId,
    DisplayedSurface& out) const {
    out = {};
    if (runtimeModelId == 0) {
        return false;
    }

    DisplayedSurface remeshSurface{};
    const bool hasRemesh = transports.remeshDisplayTransport &&
        transports.remeshDisplayTransport->findByRuntimeModelId(
            runtimeModelId,
            remeshSurface.socketKey,
            remeshSurface.surfaceIdentity);

    uint64_t modelSocketKey = 0;
    const bool hasModel = transports.modelDisplayTransport &&
        transports.modelDisplayTransport->findByRuntimeModelId(
            runtimeModelId, modelSocketKey);

    if (hasRemesh && hasModel) {
        return false;
    }
    if (hasRemesh) {
        remeshSurface.intrinsic = true;
        out = remeshSurface;
        return true;
    }
    if (hasModel) {
        out.socketKey = modelSocketKey;
        out.intrinsic = false;
        return true;
    }
    return false;
}

bool RuntimePackageController::isDisplayActive(
    uint64_t socketKey,
    uint64_t surfaceIdentity) const {
    if (socketKey == 0) {
        return false;
    }
    if (transports.remeshDisplayTransport &&
        transports.remeshDisplayTransport->surfaceMatches(
            socketKey, surfaceIdentity)) {
        return true;
    }
    return transports.modelDisplayTransport &&
        transports.modelDisplayTransport->isDisplayed(socketKey);
}

std::vector<uint32_t>
RuntimePackageController::displayedSourceModelIds() const {
    std::vector<uint32_t> ids;
    const auto append = [&ids](uint32_t runtimeModelId) {
        if (runtimeModelId == 0) {
            return;
        }
        if (std::find(ids.begin(), ids.end(), runtimeModelId) == ids.end()) {
            ids.push_back(runtimeModelId);
        }
    };
    if (transports.modelDisplayTransport) {
        for (uint32_t runtimeModelId :
             transports.modelDisplayTransport->registeredRuntimeModelIds()) {
            append(runtimeModelId);
        }
    }
    if (transports.remeshDisplayTransport) {
        for (uint32_t runtimeModelId :
             transports.remeshDisplayTransport->registeredRuntimeModelIds()) {
            append(runtimeModelId);
        }
    }
    return ids;
}

template <typename ProductT, typename PackageT, typename TransportT>
static bool applyComputePackage(
    uint64_t socketKey,
    const PackageT& package,
    TransportT* transport,
    RuntimeProductManager& products,
    RuntimePackageManager& packages,
    std::unordered_set<uint64_t>& processedSockets) {
    if (!transport || !products.canPublish<ProductT>(socketKey)) {
        return false;
    }

    ProductT product{};
    const ComputeApplyResult result =
        transport->apply(socketKey, package, product);
    if (result == ComputeApplyResult::Unchanged) {
        const ProductHandle handle =
            products.getCurrentHandle<ProductT>(socketKey);
        if (!handle.isValid()) {
            return false;
        }
        packages.setProductHandle(socketKey, handle);
        processedSockets.insert(socketKey);
        return true;
    }
    if (result != ComputeApplyResult::Prepared) {
        return false;
    }
    if (!product.isValid()) {
        transport->cancelPending(socketKey);
        return false;
    }
    if (!transport->commitPending(socketKey)) {
        transport->cancelPending(socketKey);
        return false;
    }
    const ProductHandle handle =
        products.publish<ProductT>(socketKey, product);
    if (!handle.isValid()) {
        return false;
    }
    packages.setProductHandle(socketKey, handle);
    processedSockets.insert(socketKey);
    return true;
}

bool RuntimePackageController::applyNode(const NodeGraphNode& node) {
    if (!productManager) return false;
    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t socketKey = NodeSocketKey(node.id, output.id).value;
        if (packageManager.find<GroupPackage>(socketKey)) {
            if (!productManager->unpublish<ModelProduct>(socketKey)) {
                return false;
            }
            processedSockets.insert(socketKey);
            continue;
        }
        if (const ModelPackage* package = packageManager.find<ModelPackage>(socketKey)) {
            if (!applyComputePackage<ModelProduct>(
                    socketKey, *package,
                    transports.modelComputeTransport,
                    *productManager, packageManager, processedSockets)) {
                return false;
            }
            continue;
        }
        if (const PointPackage* package = packageManager.find<PointPackage>(socketKey)) {
            if (package->pointCount == 0 || package->positions.empty()) {
                requestRemoveRuntime(socketKey);
                continue;
            }
            if (!applyComputePackage<PointProduct>(
                    socketKey, *package,
                    transports.pointComputeTransport,
                    *productManager, packageManager, processedSockets)) {
                return false;
            }
            continue;
        }
        if (const RemeshPackage* remeshPackage = packageManager.find<RemeshPackage>(socketKey)) {
            if (!applyComputePackage<RemeshProduct>(
                    socketKey, *remeshPackage,
                    transports.remeshComputeTransport,
                    *productManager, packageManager, processedSockets)) {
                return false;
            }
            continue;
        }

        if (const VoronoiPackage* voronoiPackage = packageManager.find<VoronoiPackage>(socketKey)) {
            if (!voronoiPackage->active) {
                requestRemoveRuntime(socketKey);
                continue;
            }
            if (!applyComputePackage<VoronoiProduct>(
                    socketKey, *voronoiPackage,
                    transports.voronoiComputeTransport,
                    *productManager, packageManager, processedSockets)) {
                return false;
            }
            continue;
        }

        if (const HeatPackage* heatPackage = packageManager.find<HeatPackage>(socketKey)) {
            if (!heatPackage->active) {
                requestRemoveRuntime(socketKey);
                continue;
            }
            if (!applyComputePackage<HeatProduct>(
                    socketKey, *heatPackage,
                    transports.heatComputeTransport,
                    *productManager, packageManager, processedSockets)) {
                return false;
            }
            continue;
        }

        if (const CFDPackage* cfdPackage = packageManager.find<CFDPackage>(socketKey)) {
            if (!cfdPackage->active) {
                requestRemoveRuntime(socketKey);
                continue;
            }
            if (!applyComputePackage<CFDProduct>(
                    socketKey, *cfdPackage,
                    transports.cfdComputeTransport,
                    *productManager, packageManager, processedSockets)) {
                return false;
            }
        }
    }
    return true;
}

void RuntimePackageController::finishCompilation(bool success) {
    if (success) {
        removeStalePackages();
        packageManager.commitStaged();
    } else {
        // Frozen outputs must not keep a generation that references an upstream
        // publication retired by the packages this pass did commit.
        removeStalePackages();
        packageManager.commitProcessed(processedSockets);
    }
    processedSockets.clear();
}

void RuntimePackageController::abortCompilationPreservingUnprocessed() {
    packageManager.commitProcessed(processedSockets);
    processedSockets.clear();
}

void RuntimePackageController::detachAllDisplayBindings() {
    if (transports.modelDisplayTransport) transports.modelDisplayTransport->detachAll();
    if (transports.pointDisplayTransport) transports.pointDisplayTransport->detachAll();
    if (transports.remeshDisplayTransport) transports.remeshDisplayTransport->detachAll();
    if (transports.voronoiDisplayTransport) transports.voronoiDisplayTransport->detachAll();
    if (transports.heatDisplayTransport) transports.heatDisplayTransport->detachAll();
    if (transports.cfdDisplayTransport) transports.cfdDisplayTransport->detachAll();
}

bool RuntimePackageController::hasRetiringGenerations() const {
    if (productManager && productManager->hasAnyRetiring()) {
        return true;
    }
    if (transports.modelComputeTransport && transports.modelComputeTransport->hasAnyRetiring()) return true;
    if (transports.pointComputeTransport && transports.pointComputeTransport->hasAnyRetiring()) return true;
    if (transports.remeshComputeTransport && transports.remeshComputeTransport->hasAnyRetiring()) return true;
    if (transports.voronoiComputeTransport && transports.voronoiComputeTransport->hasAnyRetiring()) return true;
    if (transports.heatComputeTransport && transports.heatComputeTransport->hasAnyRetiring()) return true;
    if (transports.cfdComputeTransport && transports.cfdComputeTransport->hasAnyRetiring()) return true;
    return false;
}

bool RuntimePackageController::hasPendingRemovals() const {
    return !pendingRemovals.empty();
}

bool RuntimePackageController::removeRuntimeNow(uint64_t socketKey) {
    if (socketKey == 0) {
        return true;
    }

    detachDisplayBindings(socketKey);

    bool productsRetired = productManager != nullptr;
    if (productManager) {
        productsRetired = productManager->unpublish<ModelProduct>(socketKey) && productsRetired;
        productsRetired = productManager->unpublish<PointProduct>(socketKey) && productsRetired;
        productsRetired = productManager->unpublish<RemeshProduct>(socketKey) && productsRetired;
        productsRetired = productManager->unpublish<VoronoiProduct>(socketKey) && productsRetired;
        productsRetired = productManager->unpublish<HeatProduct>(socketKey) && productsRetired;
        productsRetired = productManager->unpublish<CFDProduct>(socketKey) && productsRetired;
    }
    if (!productsRetired) {
        return false;
    }

    bool backendsRemoved = true;
    if (transports.modelComputeTransport) backendsRemoved = transports.modelComputeTransport->remove(socketKey) && backendsRemoved;
    if (transports.pointComputeTransport) backendsRemoved = transports.pointComputeTransport->remove(socketKey) && backendsRemoved;
    if (transports.remeshComputeTransport) backendsRemoved = transports.remeshComputeTransport->remove(socketKey) && backendsRemoved;
    if (transports.voronoiComputeTransport) backendsRemoved = transports.voronoiComputeTransport->remove(socketKey) && backendsRemoved;
    if (transports.heatComputeTransport) backendsRemoved = transports.heatComputeTransport->remove(socketKey) && backendsRemoved;
    if (transports.cfdComputeTransport) backendsRemoved = transports.cfdComputeTransport->remove(socketKey) && backendsRemoved;

    packageManager.erasePackagesForSocket(socketKey);
    processedSockets.erase(socketKey);
    return backendsRemoved;
}

bool RuntimePackageController::requestRemoveRuntime(uint64_t socketKey) {
    if (socketKey == 0) {
        return true;
    }

    if (removeRuntimeNow(socketKey)) {
        pendingRemovals.erase(socketKey);
        return true;
    }

    packageManager.erasePackagesForSocket(socketKey);
    processedSockets.erase(socketKey);
    pendingRemovals.insert(socketKey);
    return false;
}

void RuntimePackageController::retryPendingRemovals() {
    if (hasRetiringGenerations()) {
        return;
    }

    for (auto it = pendingRemovals.begin(); it != pendingRemovals.end();) {
        const uint64_t socketKey = *it;
        if (productManager && !productManager->canUnpublishForSocket(socketKey)) {
            ++it;
            continue;
        }
        if (removeRuntimeNow(socketKey)) {
            it = pendingRemovals.erase(it);
        } else {
            ++it;
        }
    }
}

void RuntimePackageController::quiescedClear() {
    detachAllDisplayBindings();
    if (productManager) {
        productManager->quiescedClear();
    }
    if (transports.modelComputeTransport) transports.modelComputeTransport->quiescedClear();
    if (transports.pointComputeTransport) transports.pointComputeTransport->quiescedClear();
    if (transports.remeshComputeTransport) transports.remeshComputeTransport->quiescedClear();
    if (transports.voronoiComputeTransport) transports.voronoiComputeTransport->quiescedClear();
    if (transports.heatComputeTransport) transports.heatComputeTransport->quiescedClear();
    if (transports.cfdComputeTransport) transports.cfdComputeTransport->quiescedClear();
    packageManager.clearAll();
    processedSockets.clear();
    pendingRemovals.clear();
}

void RuntimePackageController::detachDisplayBindings(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }
    if (transports.modelDisplayTransport) transports.modelDisplayTransport->detach(socketKey);
    if (transports.pointDisplayTransport) transports.pointDisplayTransport->detach(socketKey);
    if (transports.remeshDisplayTransport) transports.remeshDisplayTransport->detach(socketKey);
    if (transports.voronoiDisplayTransport) transports.voronoiDisplayTransport->detach(socketKey);
    if (transports.heatDisplayTransport) transports.heatDisplayTransport->detach(socketKey);
    if (transports.cfdDisplayTransport) transports.cfdDisplayTransport->detach(socketKey);
}

void RuntimePackageController::invalidateOutputsForNode(const NodeGraphNode& node) {
    std::vector<uint64_t> socketKeys;
    socketKeys.reserve(node.outputs.size());
    for (const NodeGraphSocket& output : node.outputs) {
        socketKeys.push_back(NodeSocketKey(node.id, output.id).value);
    }
    invalidateOutputsForSockets(socketKeys);
}

void RuntimePackageController::invalidateOutputsForSockets(const std::vector<uint64_t>& socketKeys) {
    for (uint64_t socketKey : socketKeys) {
        if (socketKey == 0) {
            continue;
        }
        requestRemoveRuntime(socketKey);
    }
}

void RuntimePackageController::removeStalePackages() {
    for (uint64_t key : packageManager.staleSocketKeys()) {
        requestRemoveRuntime(key);
    }
    packageManager.destroyStale();
}
