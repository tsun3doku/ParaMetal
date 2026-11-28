#pragma once

#include "runtime/RuntimeTransports.hpp"
#include "runtime/package/RuntimePackageManager.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

class RuntimeProductManager;
struct NodeGraphNode;

class RuntimePackageController {
public:
    struct DisplayedSurface {
        uint64_t socketKey = 0;
        uint64_t surfaceIdentity = 0;
        bool intrinsic = false;
    };

    explicit RuntimePackageController(const RuntimeTransports& transports);
    ~RuntimePackageController();

    void beginCompilation();
    bool applyNode(const NodeGraphNode& node);
    // Commits staged generation on success or commits only processed packages on failure
    void finishCompilation(bool success);
    void abortCompilationPreservingUnprocessed();

    uint64_t publicationRevision() const;
    void clearRetiringProducts();
    bool hasRetiringGenerations() const;
    bool hasPendingRemovals() const;
    bool requestRemoveRuntime(uint64_t socketKey);
    bool appendRuntimeModelIdsForSocket(
        uint64_t socketKey,
        std::vector<uint32_t>& outIds) const;
    bool syncDisplay(const std::unordered_set<uint64_t>& displayKeys);
    bool findDisplayedSurface(
        uint32_t runtimeModelId,
        DisplayedSurface& out) const;
    bool isDisplayActive(
        uint64_t socketKey,
        uint64_t surfaceIdentity) const;
    std::vector<uint32_t> displayedSourceModelIds() const;
    const std::string& lastDisplayError() const { return lastDisplayError_; }
    void retryPendingRemovals();
    void invalidateOutputsForNode(const NodeGraphNode& node);
    void invalidateOutputsForSockets(const std::vector<uint64_t>& socketKeys);
    void clear();
    // Drops every display binding without retiring any backend generation.
    void detachAllDisplayBindings();
    // Drops packages, products, and compute state; the caller must have quiesced the GPU.
    void quiescedClear();

    RuntimePackageManager& packages() { return packageManager; }
    const RuntimePackageManager& packages() const { return packageManager; }
    RuntimeProductManager* products() { return productManager.get(); }
    const RuntimeProductManager* products() const { return productManager.get(); }

private:
    bool removeRuntimeNow(uint64_t socketKey);
    void removeStalePackages();
    void detachDisplayBindings(uint64_t socketKey);

    RuntimeTransports transports{};
    RuntimePackageManager packageManager{};
    std::unordered_set<uint64_t> processedSockets;
    std::unordered_set<uint64_t> pendingRemovals;
    std::string lastDisplayError_;
    std::unique_ptr<RuntimeProductManager> productManager;
};
