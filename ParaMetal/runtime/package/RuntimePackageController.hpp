#pragma once

#include "runtime/RuntimeTransports.hpp"
#include "runtime/package/RuntimePackageManager.hpp"

#include <memory>
#include <unordered_set>
#include <vector>

class RuntimeProductManager;
struct NodeGraphNode;

class RuntimePackageController {
public:
    explicit RuntimePackageController(const RuntimeTransports& transports);
    ~RuntimePackageController();

    void beginCompilation();
    bool applyNode(const NodeGraphNode& node);
    // Commits staged generation on success or commits only processed packages on failure
    void finishCompilation(bool success);

    void removeNodeRuntime(uint64_t socketKey);
    void invalidateOutputsForNode(const NodeGraphNode& node);
    void invalidateOutputsForSockets(const std::vector<uint64_t>& socketKeys);
    void clear();
    // Drops every display binding without retiring any backend generation.
    void detachAllDisplayBindings();
    // Drops packages and published products; the caller must have quiesced the GPU.
    void quiescedClear();

    RuntimePackageManager& packages() { return packageManager; }
    const RuntimePackageManager& packages() const { return packageManager; }
    RuntimeProductManager* products() { return productManager.get(); }
    const RuntimeProductManager* products() const { return productManager.get(); }

private:
    bool applyPackage(uint64_t socketKey, const ModelPackage& package);
    bool applyPackage(uint64_t socketKey, const PointPackage& package);
    bool applyPackage(uint64_t socketKey, const RemeshPackage& package);
    bool applyPackage(uint64_t socketKey, const VoronoiPackage& package);
    bool applyPackage(uint64_t socketKey, const HeatPackage& package);
    bool applyPackage(uint64_t socketKey, const CFDPackage& package);

    void removeStalePackages();
    void detachDisplayBindings(uint64_t socketKey);

    RuntimeTransports transports{};
    RuntimePackageManager packageManager{};
    std::unordered_set<uint64_t> processedSockets;
    std::unique_ptr<RuntimeProductManager> productManager;
};
