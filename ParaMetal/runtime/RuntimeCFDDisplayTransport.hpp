#pragma once

#include "hash/HashBuilder.hpp"
#include "runtime/CFDDisplayController.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/RuntimeProductManager.hpp"

#include <unordered_set>
#include <memory>
#include <unordered_map>
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
        outConfig.showSlice = package.display.showSlice;
        outConfig.sliceAxis = package.display.sliceAxis;
        outConfig.slicePosition = package.display.slicePosition;
        outConfig.showSliceWireframe = package.display.showSliceWireframe;
        outConfig.authoredActive = package.active;
        outConfig.active = package.active;
        outConfig.canonicalToWorldScale = package.canonicalToWorldScale;
        outConfig.streamlineVertexBuffer = product->streamlineVertexBuffer;
        outConfig.streamlineVertexBufferOffset = product->streamlineVertexBufferOffset;
        outConfig.streamlineVertexCount = product->streamlineVertexCount;
        outConfig.fluidNodeCount = product->fluidNodeCount;
        outConfig.velocityBuffer = product->velocityBuffer;
        outConfig.velocityBufferOffset = product->velocityBufferOffset;
        outConfig.gradientBuffer = product->gradientBuffer;
        outConfig.gradientBufferOffset = product->gradientBufferOffset;
        outConfig.centroidBuffer = product->centroidBuffer;
        outConfig.centroidBufferOffset = product->centroidBufferOffset;

        uint64_t productDisplayHash = product->hashes.display;

        if (package.domainVoronoiProduct.isValid()) {
            if (!products->isCurrent<VoronoiProduct>(package.domainVoronoiProduct)) {
                return false;
            }
            const VoronoiProduct* voronoi = products->resolve<VoronoiProduct>(package.domainVoronoiProduct);
            if (voronoi && voronoi->isValid()) {
                if (product->hashes.geometry != voronoi->hashes.geometry) {
                    return false;
                }
                outConfig.globalDomainCorners = voronoi->globalDomainCorners;
                outConfig.hasGlobalDomain = true;
                HashBuilder::combine(productDisplayHash, voronoi->hashes.geometry);
                HashBuilder::combine(productDisplayHash, voronoi->hashes.display);
                if (outConfig.showSlice) {
                    const uint64_t geometryKey = voronoi->hashes.geometry;
                    std::shared_ptr<const cfd::CFDSliceGeometrySource> source;
                    auto cached = sliceGeometryCache.find(geometryKey);
                    if (cached != sliceGeometryCache.end()) source = cached->second.lock();
                    if (!source) {
                        for (auto entry = sliceGeometryCache.begin(); entry != sliceGeometryCache.end();) {
                            if (entry->second.expired()) entry = sliceGeometryCache.erase(entry);
                            else ++entry;
                        }
                        auto created = std::make_shared<cfd::CFDSliceGeometrySource>();
                        created->faceFragmentA = voronoi->faceFragmentA;
                        created->faceFragmentB = voronoi->faceFragmentB;
                        created->faceAreaVectorA = voronoi->faceAreaVectorsA;
                        created->faceAreaVectorB = voronoi->faceAreaVectorsB;
                        created->faceLoopOffsets = voronoi->faceLoopOffsets;
                        created->loopVertexOffsets = voronoi->loopVertexOffsets;
                        created->loopVertices = voronoi->loopVertices;
                        created->globalFragmentToFluidNode = product->globalFragmentToFluidNode;
                        created->fragmentCount = voronoi->globalFragmentCount;
                        created->fluidNodeCount = product->fluidNodeCount;
                        created->geometryHash = geometryKey;
                        source = std::move(created);
                        sliceGeometryCache[geometryKey] = source;
                    }
                    outConfig.sliceGeometry = std::move(source);
                }
            }
        }

        outConfig.displayHash = buildCFDDisplayHash(outConfig, productDisplayHash);
        HashBuilder::combine(outConfig.displayHash, package.productHandle.publicationId);
        return true;
    }

    CFDDisplayController* controller = nullptr;
    RuntimeProductManager* products = nullptr;
    std::unordered_set<uint64_t> activeSocketKeys;
    mutable std::unordered_map<uint64_t, std::weak_ptr<const cfd::CFDSliceGeometrySource>> sliceGeometryCache;
};
