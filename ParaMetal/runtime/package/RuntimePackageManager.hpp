#pragma once

#include "runtime/package/RuntimePackages.hpp"

#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

class RuntimePackageManager {
public:
    void beginCompile();

    template <typename PackageT>
    void apply(uint64_t socketKey, const PackageT& package) {
        if (socketKey == 0 || package.hashes.full == 0) {
            return;
        }

        const auto& committed = packagesFor<PackageT>();
        auto& pending = pendingFor<PackageT>();
        const auto it = committed.find(socketKey);

        const bool computeChanged =
            it == committed.end() ||
            !package.hasValidProduct() ||
            it->second.computeHash() != package.computeHash();

        pending.insert_or_assign(socketKey, package);
        staleSockets.erase(socketKey);

        if (computeChanged) {
            retainedSockets.erase(socketKey);
        } else {
            retainedSockets.insert(socketKey);
        }
    }

    void erasePackagesForSocket(uint64_t socketKey);
    void setProductHandle(uint64_t socketKey, const ProductHandle& handle);

    template <typename PackageT>
    const PackageT* find(uint64_t socketKey) const {
        if (staleSockets.find(socketKey) != staleSockets.end()) {
            return nullptr;
        }
        if (retainedSockets.find(socketKey) != retainedSockets.end()) {
            return nullptr;
        }
        const auto& pending = pendingFor<PackageT>();
        const auto pendingIt = pending.find(socketKey);
        if (pendingIt != pending.end()) {
            return &pendingIt->second;
        }
        const auto& committed = packagesFor<PackageT>();
        const auto it = committed.find(socketKey);
        return it != committed.end() ? &it->second : nullptr;
    }

    template <typename PackageT>
    const PackageT* findAny(uint64_t socketKey) const {
        if (staleSockets.find(socketKey) != staleSockets.end()) {
            return nullptr;
        }
        const auto& pending = pendingFor<PackageT>();
        const auto pendingIt = pending.find(socketKey);
        if (pendingIt != pending.end()) {
            return &pendingIt->second;
        }
        const auto& committed = packagesFor<PackageT>();
        const auto it = committed.find(socketKey);
        return it != committed.end() ? &it->second : nullptr;
    }

    template <typename PackageT>
    const PackageT* findStored(uint64_t socketKey) const {
        const auto& committed = packagesFor<PackageT>();
        const auto it = committed.find(socketKey);
        return it != committed.end() ? &it->second : nullptr;
    }

    const std::unordered_map<uint64_t, ModelPackage>& models() const { return modelPackages; }
    const std::unordered_map<uint64_t, PointPackage>& points() const { return pointPackages; }
    const std::unordered_map<uint64_t, RemeshPackage>& remeshes() const { return remeshPackages; }
    const std::unordered_map<uint64_t, VoronoiPackage>& voronois() const { return voronoiPackages; }
    const std::unordered_map<uint64_t, HeatPackage>& heatSystems() const { return heatPackages; }
    const std::unordered_map<uint64_t, CFDPackage>& cfdSystems() const { return cfdPackages; }
    const std::unordered_set<uint64_t>& staleSocketKeys() const { return staleSockets; }

    bool isPackageCurrent(uint64_t socketKey) const {
        return staleSockets.find(socketKey) == staleSockets.end();
    }

    void clearAll();
    void destroyStale();
    // Success finalize: staged packages become the committed generation
    void commitStaged();
    // Abort finalize commits successful packages and retains previous packages for failed nodes
    void commitProcessed(const std::unordered_set<uint64_t>& processedSockets);

private:
    template <typename PackageT>
    static void commitPendingInto(
        std::unordered_map<uint64_t, PackageT>& committed,
        std::unordered_map<uint64_t, PackageT>& pending) {
        for (auto& [socketKey, package] : pending) {
            committed.insert_or_assign(socketKey, std::move(package));
        }
    }

    template <typename PackageT>
    static void commitProcessedInto(
        std::unordered_map<uint64_t, PackageT>& committed,
        const std::unordered_map<uint64_t, PackageT>& pending,
        const std::unordered_set<uint64_t>& processedSockets) {
        for (const auto& [socketKey, package] : pending) {
            if (processedSockets.find(socketKey) != processedSockets.end()) {
                committed.insert_or_assign(socketKey, package);
            }
        }
    }

    template <typename PackageT>
    static void eraseFrom(
        std::unordered_map<uint64_t, PackageT>& committed,
        std::unordered_map<uint64_t, PackageT>& pending,
        uint64_t socketKey) {
        committed.erase(socketKey);
        pending.erase(socketKey);
    }

    template <typename PackageT>
    std::unordered_map<uint64_t, PackageT>& packagesFor() {
        if constexpr (std::is_same_v<PackageT, ModelPackage>) return modelPackages;
        else if constexpr (std::is_same_v<PackageT, PointPackage>) return pointPackages;
        else if constexpr (std::is_same_v<PackageT, RemeshPackage>) return remeshPackages;
        else if constexpr (std::is_same_v<PackageT, VoronoiPackage>) return voronoiPackages;
        else if constexpr (std::is_same_v<PackageT, HeatPackage>) return heatPackages;
        else if constexpr (std::is_same_v<PackageT, CFDPackage>) return cfdPackages;
    }

    template <typename PackageT>
    const std::unordered_map<uint64_t, PackageT>& packagesFor() const {
        if constexpr (std::is_same_v<PackageT, ModelPackage>) return modelPackages;
        else if constexpr (std::is_same_v<PackageT, PointPackage>) return pointPackages;
        else if constexpr (std::is_same_v<PackageT, RemeshPackage>) return remeshPackages;
        else if constexpr (std::is_same_v<PackageT, VoronoiPackage>) return voronoiPackages;
        else if constexpr (std::is_same_v<PackageT, HeatPackage>) return heatPackages;
        else if constexpr (std::is_same_v<PackageT, CFDPackage>) return cfdPackages;
    }

    template <typename PackageT>
    std::unordered_map<uint64_t, PackageT>& pendingFor() {
        if constexpr (std::is_same_v<PackageT, ModelPackage>) return pendingModelPackages;
        else if constexpr (std::is_same_v<PackageT, PointPackage>) return pendingPointPackages;
        else if constexpr (std::is_same_v<PackageT, RemeshPackage>) return pendingRemeshPackages;
        else if constexpr (std::is_same_v<PackageT, VoronoiPackage>) return pendingVoronoiPackages;
        else if constexpr (std::is_same_v<PackageT, HeatPackage>) return pendingHeatPackages;
        else if constexpr (std::is_same_v<PackageT, CFDPackage>) return pendingCfdPackages;
    }

    template <typename PackageT>
    const std::unordered_map<uint64_t, PackageT>& pendingFor() const {
        if constexpr (std::is_same_v<PackageT, ModelPackage>) return pendingModelPackages;
        else if constexpr (std::is_same_v<PackageT, PointPackage>) return pendingPointPackages;
        else if constexpr (std::is_same_v<PackageT, RemeshPackage>) return pendingRemeshPackages;
        else if constexpr (std::is_same_v<PackageT, VoronoiPackage>) return pendingVoronoiPackages;
        else if constexpr (std::is_same_v<PackageT, HeatPackage>) return pendingHeatPackages;
        else if constexpr (std::is_same_v<PackageT, CFDPackage>) return pendingCfdPackages;
    }

    void clearPending();

    template <typename PackageT>
    static void setProductHandle(
        uint64_t socketKey,
        const ProductHandle& handle,
        std::unordered_map<uint64_t, PackageT>& committed,
        std::unordered_map<uint64_t, PackageT>& pending) {
        const auto pendingIt = pending.find(socketKey);
        if (pendingIt != pending.end()) {
            pendingIt->second.productHandle = handle;
            return;
        }
        const auto committedIt = committed.find(socketKey);
        if (committedIt != committed.end()) {
            committedIt->second.productHandle = handle;
        }
    }

    template <typename PackageT>
    void collectKeys(const std::unordered_map<uint64_t, PackageT>& packages) {
        for (const auto& [socketKey, package] : packages) {
            (void)package;
            staleSockets.insert(socketKey);
        }
    }

    template <typename PackageT>
    void eraseStale(std::unordered_map<uint64_t, PackageT>& packages) {
        for (auto it = packages.begin(); it != packages.end();) {
            if (staleSockets.find(it->first) != staleSockets.end()) {
                it = packages.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::unordered_map<uint64_t, ModelPackage> modelPackages;
    std::unordered_map<uint64_t, PointPackage> pointPackages;
    std::unordered_map<uint64_t, RemeshPackage> remeshPackages;
    std::unordered_map<uint64_t, VoronoiPackage> voronoiPackages;
    std::unordered_map<uint64_t, HeatPackage> heatPackages;
    std::unordered_map<uint64_t, CFDPackage> cfdPackages;

    std::unordered_map<uint64_t, ModelPackage> pendingModelPackages;
    std::unordered_map<uint64_t, PointPackage> pendingPointPackages;
    std::unordered_map<uint64_t, RemeshPackage> pendingRemeshPackages;
    std::unordered_map<uint64_t, VoronoiPackage> pendingVoronoiPackages;
    std::unordered_map<uint64_t, HeatPackage> pendingHeatPackages;
    std::unordered_map<uint64_t, CFDPackage> pendingCfdPackages;

    std::unordered_set<uint64_t> staleSockets;
    std::unordered_set<uint64_t> retainedSockets;
};
