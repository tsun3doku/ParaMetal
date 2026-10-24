#pragma once

#include "runtime/RuntimeConnections.hpp"
#include "runtime/package/RuntimePackageManager.hpp"

#include <memory>
#include <unordered_set>

class RuntimeProductManager;
struct NodeGraphNode;

class RuntimePackageController {
public:
    explicit RuntimePackageController(const RuntimeConnections& connections);

    void beginCompilation();
    bool applyNode(const NodeGraphNode& node);
    // Commits staged generation on success or commits only successful packages on failure
    void finishCompilation(bool success);

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

    RuntimeConnections connections{};
    RuntimePackageManager packageManager{};
    std::unordered_set<uint64_t> appliedSockets;
    std::unique_ptr<RuntimeProductManager> productManager;
};
