#pragma once

#include "hash/HashBuilder.hpp"
#include "runtime/CFDDisplayController.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/RuntimeProductManager.hpp"

#include <unordered_set>
#include <vector>

class RuntimeCFDDisplayTransport {
public:
    void setController(CFDDisplayController* updatedController) {
        controller = updatedController;
    }

    void setProducts(RuntimeProductManager* updatedProducts) {
        products = updatedProducts;
    }

    void sync(const RuntimePackageManager& registry, const std::unordered_set<uint64_t>& visibleKeys) {
        if (!controller) {
            return;
        }

        std::unordered_set<uint64_t> nextSocketKeys;
        for (const auto& [socketKey, package] : registry.cfdSystems()) {
            if (!registry.isPackageCurrent(socketKey)) continue;
            if (visibleKeys.find(socketKey) == visibleKeys.end()) {
                continue;
            }

            CFDDisplayController::Config config{};
            if (!tryBuildConfig(socketKey, package, config)) {
                controller->remove(socketKey);
                continue;
            }

            controller->apply(socketKey, config);
            nextSocketKeys.insert(socketKey);
        }

        for (uint64_t socketKey : activeSocketKeys) {
            if (nextSocketKeys.find(socketKey) == nextSocketKeys.end()) {
                controller->remove(socketKey);
            }
        }
        activeSocketKeys = nextSocketKeys;
    }

    void finalizeSync() {
        if (!controller) {
            return;
        }

        controller->finalizeSync();
    }

    void detach(uint64_t socketKey) {
        if (!controller || socketKey == 0) {
            return;
        }
        controller->remove(socketKey);
        activeSocketKeys.erase(socketKey);
    }

    void detachAll() {
        for (uint64_t socketKey : activeSocketKeys) {
            if (controller) {
                controller->remove(socketKey);
            }
        }
        activeSocketKeys.clear();
    }

private:
    bool tryBuildConfig(
        uint64_t socketKey,
        const CFDPackage& package,
        CFDDisplayController::Config& outConfig) const {
        if (!controller || !products || socketKey == 0) {
            return false;
        }

        if (!package.display.anyVisible()) {
            return false;
        }

        if (!products->isCurrent<CFDProduct>(package.productHandle)) {
            return false;
        }
        const CFDProduct* product = products->resolve<CFDProduct>(package.productHandle);
        if (!product || !product->isValid()) {
            return false;
        }

        outConfig = {};
        outConfig.showStreamlines = package.display.showStreamlines;
        outConfig.showDomainBox = package.display.showDomainBox;
        outConfig.showVelocityPalette = package.display.showVelocityPalette;
        outConfig.authoredActive = package.active;
        outConfig.active = package.active;
        outConfig.canonicalToWorldScale = package.canonicalToWorldScale;
        outConfig.streamlineVertexBuffer = product->streamlineVertexBuffer;
        outConfig.streamlineVertexBufferOffset = product->streamlineVertexBufferOffset;
        outConfig.streamlineVertexCount = product->streamlineVertexCount;

        uint64_t productDisplayHash = product->hashes.display;

        if (package.domainVoronoiProduct.isValid()) {
            if (!products->isCurrent<VoronoiProduct>(package.domainVoronoiProduct)) {
                return false;
            }
            const VoronoiProduct* voronoi = products->resolve<VoronoiProduct>(package.domainVoronoiProduct);
            if (voronoi && voronoi->isValid()) {
                outConfig.globalDomainCorners = voronoi->globalDomainCorners;
                outConfig.hasGlobalDomain = true;
                HashBuilder::combine(productDisplayHash, voronoi->hashes.geometry);
                HashBuilder::combine(productDisplayHash, voronoi->hashes.display);
            }
        }

        outConfig.displayHash = buildCFDDisplayHash(outConfig, productDisplayHash);
        HashBuilder::combine(outConfig.displayHash, package.productHandle.publicationId);
        return true;
    }

    CFDDisplayController* controller = nullptr;
    RuntimeProductManager* products = nullptr;
    std::unordered_set<uint64_t> activeSocketKeys;
};
