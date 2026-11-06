#include "RuntimePackageController.hpp"

#include "nodegraph/NodeGraphState.hpp"
#include "runtime/ComputeTransportResult.hpp"
#include "runtime/RuntimeHeatComputeTransport.hpp"
#include "runtime/RuntimeCFDComputeTransport.hpp"
#include "runtime/RuntimeHeatDisplayTransport.hpp"
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
        if (connections.modelComputeTransport) connections.modelComputeTransport->remove(key);
    }
    for (const auto& [key, _] : packageManager.points()) {
        if (connections.pointComputeTransport) connections.pointComputeTransport->remove(key);
    }
    for (const auto& [key, _] : packageManager.remeshes()) {
        if (connections.remeshComputeTransport) connections.remeshComputeTransport->remove(key);
    }
    for (const auto& [key, _] : packageManager.voronois()) {
        if (connections.voronoiComputeTransport) connections.voronoiComputeTransport->remove(key);
    }
    for (const auto& [key, _] : packageManager.heatSystems()) {
        if (connections.heatComputeTransport) connections.heatComputeTransport->remove(key);
    }
    for (const auto& [key, _] : packageManager.cfdSystems()) {
        if (connections.cfdComputeTransport) connections.cfdComputeTransport->remove(key);
    }
    packageManager.clearAll();
    processedSockets.clear();
    if (productManager) {
        productManager->clear();
    }
}

RuntimePackageController::RuntimePackageController(const RuntimeConnections& runtimeConnections)
    : connections(runtimeConnections),
      productManager(std::make_unique<RuntimeProductManager>()) {
    RuntimeProductManager* products = productManager.get();
    if (connections.remeshComputeTransport) connections.remeshComputeTransport->setProducts(products);
    if (connections.voronoiComputeTransport) connections.voronoiComputeTransport->setProducts(products);
    if (connections.heatComputeTransport) connections.heatComputeTransport->setProducts(products);
    if (connections.cfdComputeTransport) connections.cfdComputeTransport->setProducts(products);
    if (connections.modelDisplayTransport) connections.modelDisplayTransport->setProducts(products);
    if (connections.pointDisplayTransport) connections.pointDisplayTransport->setProducts(products);
    if (connections.remeshDisplayTransport) connections.remeshDisplayTransport->setProducts(products);
    if (connections.voronoiDisplayTransport) connections.voronoiDisplayTransport->setProducts(products);
    if (connections.heatDisplayTransport) connections.heatDisplayTransport->setProducts(products);
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
    if (!connections.modelComputeTransport) return true;
    ModelProduct product{};
    const ComputeApplyResult result = connections.modelComputeTransport->apply(socketKey, package, product);
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
    if (!connections.pointComputeTransport) return true;
    PointProduct product{};
    const ComputeApplyResult result = connections.pointComputeTransport->apply(socketKey, package, product);
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
            connections.pointComputeTransport->commitPending(socketKey);
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
        connections.pointComputeTransport->cancelPending(socketKey);
    }
    return false;
}

bool RuntimePackageController::applyPackage(uint64_t socketKey, const RemeshPackage& package) {
    if (!connections.remeshComputeTransport) return true;
    RemeshProduct product{};
    const ComputeApplyResult result = connections.remeshComputeTransport->apply(socketKey, package, product);
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
            connections.remeshComputeTransport->commitPending(socketKey);
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
        connections.remeshComputeTransport->cancelPending(socketKey);
    }
    return false;
}

bool RuntimePackageController::applyPackage(uint64_t socketKey, const VoronoiPackage& package) {
    if (!connections.voronoiComputeTransport) return true;
    VoronoiProduct product{};
    const ComputeApplyResult result = connections.voronoiComputeTransport->apply(socketKey, package, product);
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
            connections.voronoiComputeTransport->commitPending(socketKey);
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
        connections.voronoiComputeTransport->cancelPending(socketKey);
    }
    return false;
}

bool RuntimePackageController::applyPackage(uint64_t socketKey, const HeatPackage& package) {
    if (!connections.heatComputeTransport) return true;
    HeatProduct product{};
    const ComputeApplyResult result = connections.heatComputeTransport->apply(socketKey, package, product);
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
            connections.heatComputeTransport->commitPending(socketKey);
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
        connections.heatComputeTransport->cancelPending(socketKey);
    }
    return false;
}

bool RuntimePackageController::applyPackage(uint64_t socketKey, const CFDPackage& package) {
    if (!connections.cfdComputeTransport) return true;
    CFDProduct product{};
    const ComputeApplyResult result = connections.cfdComputeTransport->apply(socketKey, package, product);
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
            connections.cfdComputeTransport->commitPending(socketKey);
            packageManager.setProductHandle(socketKey, handle);
            processedSockets.insert(socketKey);
            return true;
        }
        connections.cfdComputeTransport->cancelPending(socketKey);
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
    if (connections.modelDisplayTransport) connections.modelDisplayTransport->detachAll();
    if (connections.pointDisplayTransport) connections.pointDisplayTransport->detachAll();
    if (connections.remeshDisplayTransport) connections.remeshDisplayTransport->detachAll();
    if (connections.voronoiDisplayTransport) connections.voronoiDisplayTransport->detachAll();
    if (connections.heatDisplayTransport) connections.heatDisplayTransport->detachAll();
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
    if (connections.modelDisplayTransport) connections.modelDisplayTransport->detach(socketKey);
    if (connections.pointDisplayTransport) connections.pointDisplayTransport->detach(socketKey);
    if (connections.remeshDisplayTransport) connections.remeshDisplayTransport->detach(socketKey);
    if (connections.voronoiDisplayTransport) connections.voronoiDisplayTransport->detach(socketKey);
    if (connections.heatDisplayTransport) connections.heatDisplayTransport->detach(socketKey);
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
    if (connections.modelComputeTransport) connections.modelComputeTransport->remove(socketKey);
    if (connections.pointComputeTransport) connections.pointComputeTransport->remove(socketKey);
    if (connections.remeshComputeTransport) connections.remeshComputeTransport->remove(socketKey);
    if (connections.voronoiComputeTransport) connections.voronoiComputeTransport->remove(socketKey);
    if (connections.heatComputeTransport) connections.heatComputeTransport->remove(socketKey);
    if (connections.cfdComputeTransport) connections.cfdComputeTransport->remove(socketKey);
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
