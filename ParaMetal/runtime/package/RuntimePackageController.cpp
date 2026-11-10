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

RuntimePackageController::~RuntimePackageController() {
    clear();
}

void RuntimePackageController::clear() {
    for (const auto& [key, _] : packageManager.models()) {
        if (transports.modelComputeTransport) transports.modelComputeTransport->remove(key);
    }
    for (const auto& [key, _] : packageManager.points()) {
        if (transports.pointComputeTransport) transports.pointComputeTransport->remove(key);
    }
    for (const auto& [key, _] : packageManager.remeshes()) {
        if (transports.remeshComputeTransport) transports.remeshComputeTransport->remove(key);
    }
    for (const auto& [key, _] : packageManager.voronois()) {
        if (transports.voronoiComputeTransport) transports.voronoiComputeTransport->remove(key);
    }
    for (const auto& [key, _] : packageManager.heatSystems()) {
        if (transports.heatComputeTransport) transports.heatComputeTransport->remove(key);
    }
    for (const auto& [key, _] : packageManager.cfdSystems()) {
        if (transports.cfdComputeTransport) transports.cfdComputeTransport->remove(key);
    }
    packageManager.clearAll();
    processedSockets.clear();
    if (productManager) {
        productManager->clear();
    }
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

bool RuntimePackageController::applyNode(const NodeGraphNode& node) {
    if (!productManager) return false;
    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t socketKey = NodeSocketKey(node.id, output.id).value;
        if (const ModelPackage* package = packageManager.find<ModelPackage>(socketKey)) {
            if (!applyPackage(socketKey, *package)) return false;
            continue;
        }
        if (const PointPackage* package = packageManager.find<PointPackage>(socketKey)) {
            if (package->pointCount == 0 || package->positions.empty()) {
                removeNodeRuntime(socketKey);
                processedSockets.insert(socketKey);
                continue;
            }
            if (!applyPackage(socketKey, *package)) return false;
            continue;
        }
        if (const RemeshPackage* remeshPackage = packageManager.find<RemeshPackage>(socketKey)) {
            if (!applyPackage(socketKey, *remeshPackage)) return false;
            continue;
        }

        if (const VoronoiPackage* voronoiPackage = packageManager.find<VoronoiPackage>(socketKey)) {
            if (!voronoiPackage->authored.active) {
                removeNodeRuntime(socketKey);
                processedSockets.insert(socketKey);
                continue;
            }
            if (!applyPackage(socketKey, *voronoiPackage)) return false;
            continue;
        }

        if (const HeatPackage* heatPackage = packageManager.findAny<HeatPackage>(socketKey)) {
            if (!heatPackage->authored.active) {
                removeNodeRuntime(socketKey);
                processedSockets.insert(socketKey);
                continue;
            }
            if (!applyPackage(socketKey, *heatPackage)) return false;
            continue;
        }

        if (const CFDPackage* cfdPackage = packageManager.find<CFDPackage>(socketKey)) {
            if (!cfdPackage->active) {
                removeNodeRuntime(socketKey);
                processedSockets.insert(socketKey);
                continue;
            }
            if (!applyPackage(socketKey, *cfdPackage)) return false;
        }
    }
    return true;
}

bool RuntimePackageController::applyPackage(uint64_t socketKey, const ModelPackage& package) {
    if (!transports.modelComputeTransport) return true;
    ModelProduct product{};
    const ComputeApplyResult result = transports.modelComputeTransport->apply(socketKey, package, product);
    if (result == ComputeApplyResult::Unchanged) {
        const ProductHandle handle = productManager->getCurrentHandle<ModelProduct>(socketKey);
        if (handle.isValid()) {
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
    } else if (result == ComputeApplyResult::Prepared) {
        const ProductHandle handle = productManager->publish<ModelProduct>(socketKey, product);
        if (handle.isValid()) {
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
    }
    return false;
}

bool RuntimePackageController::applyPackage(uint64_t socketKey, const PointPackage& package) {
    if (!transports.pointComputeTransport) return true;
    PointProduct product{};
    const ComputeApplyResult result = transports.pointComputeTransport->apply(socketKey, package, product);
    if (result == ComputeApplyResult::Unchanged) {
        const ProductHandle handle = productManager->getCurrentHandle<PointProduct>(socketKey);
        if (handle.isValid()) {
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
    } else if (result == ComputeApplyResult::Prepared) {
        const ProductHandle handle = productManager->publish<PointProduct>(socketKey, product);
        if (handle.isValid()) {
            transports.pointComputeTransport->commitPending(socketKey);
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
        transports.pointComputeTransport->cancelPending(socketKey);
    }
    return false;
}

bool RuntimePackageController::applyPackage(uint64_t socketKey, const RemeshPackage& package) {
    if (!transports.remeshComputeTransport) return true;
    RemeshProduct product{};
    const ComputeApplyResult result = transports.remeshComputeTransport->apply(socketKey, package, product);
    if (result == ComputeApplyResult::Unchanged) {
        const ProductHandle handle = productManager->getCurrentHandle<RemeshProduct>(socketKey);
        if (handle.isValid()) {
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
    } else if (result == ComputeApplyResult::Prepared) {
        const ProductHandle handle = productManager->publish<RemeshProduct>(socketKey, product);
        if (handle.isValid()) {
            transports.remeshComputeTransport->commitPending(socketKey);
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
        transports.remeshComputeTransport->cancelPending(socketKey);
    }
    return false;
}

bool RuntimePackageController::applyPackage(uint64_t socketKey, const VoronoiPackage& package) {
    if (!transports.voronoiComputeTransport) return true;
    VoronoiProduct product{};
    const ComputeApplyResult result = transports.voronoiComputeTransport->apply(socketKey, package, product);
    if (result == ComputeApplyResult::Unchanged) {
        const ProductHandle handle = productManager->getCurrentHandle<VoronoiProduct>(socketKey);
        if (handle.isValid()) {
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
    } else if (result == ComputeApplyResult::Prepared) {
        const ProductHandle handle = productManager->publish<VoronoiProduct>(socketKey, product);
        if (handle.isValid()) {
            transports.voronoiComputeTransport->commitPending(socketKey);
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
        transports.voronoiComputeTransport->cancelPending(socketKey);
    }
    return false;
}

bool RuntimePackageController::applyPackage(uint64_t socketKey, const HeatPackage& package) {
    if (!transports.heatComputeTransport) return true;
    HeatProduct product{};
    const ComputeApplyResult result = transports.heatComputeTransport->apply(socketKey, package, product);
    if (result == ComputeApplyResult::Unchanged) {
        const ProductHandle handle = productManager->getCurrentHandle<HeatProduct>(socketKey);
        if (handle.isValid()) {
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
    } else if (result == ComputeApplyResult::Prepared) {
        const ProductHandle handle = productManager->publish<HeatProduct>(socketKey, product);
        if (handle.isValid()) {
            transports.heatComputeTransport->commitPending(socketKey);
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
        transports.heatComputeTransport->cancelPending(socketKey);
    }
    return false;
}

bool RuntimePackageController::applyPackage(uint64_t socketKey, const CFDPackage& package) {
    if (!transports.cfdComputeTransport) return true;
    CFDProduct product{};
    const ComputeApplyResult result = transports.cfdComputeTransport->apply(socketKey, package, product);
    if (result == ComputeApplyResult::Unchanged) {
        const ProductHandle handle = productManager->getCurrentHandle<CFDProduct>(socketKey);
        if (handle.isValid()) {
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
    } else if (result == ComputeApplyResult::Prepared) {
        const ProductHandle handle = productManager->publish<CFDProduct>(socketKey, product);
        if (handle.isValid()) {
            transports.cfdComputeTransport->commitPending(socketKey);
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
        transports.cfdComputeTransport->cancelPending(socketKey);
    }
    return false;
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

void RuntimePackageController::detachAllDisplayBindings() {
    if (transports.modelDisplayTransport) transports.modelDisplayTransport->detachAll();
    if (transports.pointDisplayTransport) transports.pointDisplayTransport->detachAll();
    if (transports.remeshDisplayTransport) transports.remeshDisplayTransport->detachAll();
    if (transports.voronoiDisplayTransport) transports.voronoiDisplayTransport->detachAll();
    if (transports.heatDisplayTransport) transports.heatDisplayTransport->detachAll();
    if (transports.cfdDisplayTransport) transports.cfdDisplayTransport->detachAll();
}

void RuntimePackageController::quiescedClear() {
    packageManager.clearAll();
    processedSockets.clear();
    if (productManager) {
        productManager->quiescedClear();
    }
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

void RuntimePackageController::removeNodeRuntime(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }
    // Detach display bindings before the product and backend enter retirement.
    detachDisplayBindings(socketKey);
    if (productManager) {
        productManager->unpublish<ModelProduct>(socketKey);
        productManager->unpublish<PointProduct>(socketKey);
        productManager->unpublish<RemeshProduct>(socketKey);
        productManager->unpublish<VoronoiProduct>(socketKey);
        productManager->unpublish<HeatProduct>(socketKey);
        productManager->unpublish<CFDProduct>(socketKey);
    }
    if (transports.modelComputeTransport) transports.modelComputeTransport->remove(socketKey);
    if (transports.pointComputeTransport) transports.pointComputeTransport->remove(socketKey);
    if (transports.remeshComputeTransport) transports.remeshComputeTransport->remove(socketKey);
    if (transports.voronoiComputeTransport) transports.voronoiComputeTransport->remove(socketKey);
    if (transports.heatComputeTransport) transports.heatComputeTransport->remove(socketKey);
    if (transports.cfdComputeTransport) transports.cfdComputeTransport->remove(socketKey);
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
        removeNodeRuntime(socketKey);
        packageManager.erasePackagesForSocket(socketKey);
        processedSockets.erase(socketKey);
    }
}

void RuntimePackageController::removeStalePackages() {
    for (uint64_t key : packageManager.staleSocketKeys()) {
        removeNodeRuntime(key);
    }
    packageManager.destroyStale();
}
