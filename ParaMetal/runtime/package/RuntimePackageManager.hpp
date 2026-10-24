#pragma once

#include "runtime/package/RuntimePackages.hpp"

#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

class RuntimePackageManager {
public:
    void beginCompile() {
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

    void retain(uint64_t socketKey) {
        if (socketKey == 0 || !hasPackage(socketKey)) {
            return;
        }
        staleSockets.erase(socketKey);
        retainedSockets.insert(socketKey);
    }

    void setProductHandle(uint64_t socketKey, const ProductHandle& handle) {
        setProductHandle(socketKey, handle, modelPackages, pendingModelPackages);
        setProductHandle(socketKey, handle, pointPackages, pendingPointPackages);
        setProductHandle(socketKey, handle, remeshPackages, pendingRemeshPackages);
        setProductHandle(socketKey, handle, voronoiPackages, pendingVoronoiPackages);
        setProductHandle(socketKey, handle, heatPackages, pendingHeatPackages);
        setProductHandle(socketKey, handle, cfdPackages, pendingCfdPackages);
    }

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

    bool isCurrent(uint64_t socketKey) const {
        return staleSockets.find(socketKey) == staleSockets.end();
    }

    void destroyStale() {
        eraseStale(modelPackages);
        eraseStale(pointPackages);
        eraseStale(remeshPackages);
        eraseStale(voronoiPackages);
        eraseStale(heatPackages);
        eraseStale(cfdPackages);
        staleSockets.clear();
    }

    // Success finalize: staged packages become the committed generation
    void commitStaged() {
        commitPendingInto(modelPackages, pendingModelPackages);
        commitPendingInto(pointPackages, pendingPointPackages);
        commitPendingInto(remeshPackages, pendingRemeshPackages);
        commitPendingInto(voronoiPackages, pendingVoronoiPackages);
        commitPendingInto(heatPackages, pendingHeatPackages);
        commitPendingInto(cfdPackages, pendingCfdPackages);
        clearPending();
    }

    // Abort finalize commits successful packages and retains previous packages for failed nodes
    void commitApplied(const std::unordered_set<uint64_t>& appliedSockets) {
        commitAppliedInto(modelPackages, pendingModelPackages, appliedSockets);
        commitAppliedInto(pointPackages, pendingPointPackages, appliedSockets);
        commitAppliedInto(remeshPackages, pendingRemeshPackages, appliedSockets);
        commitAppliedInto(voronoiPackages, pendingVoronoiPackages, appliedSockets);
        commitAppliedInto(heatPackages, pendingHeatPackages, appliedSockets);
        commitAppliedInto(cfdPackages, pendingCfdPackages, appliedSockets);
        clearPending();
        staleSockets.clear();
        retainedSockets.clear();
    }

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
    static void commitAppliedInto(
        std::unordered_map<uint64_t, PackageT>& committed,
        const std::unordered_map<uint64_t, PackageT>& pending,
        const std::unordered_set<uint64_t>& appliedSockets) {
        for (const auto& [socketKey, package] : pending) {
            if (appliedSockets.find(socketKey) != appliedSockets.end()) {
                committed.insert_or_assign(socketKey, package);
            }
        }
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

    void clearPending() {
        pendingModelPackages.clear();
        pendingPointPackages.clear();
        pendingRemeshPackages.clear();
        pendingVoronoiPackages.clear();
        pendingHeatPackages.clear();
        pendingCfdPackages.clear();
    }

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

    bool hasPackage(uint64_t socketKey) const {
        return modelPackages.find(socketKey) != modelPackages.end() ||
            pendingModelPackages.find(socketKey) != pendingModelPackages.end() ||
            pointPackages.find(socketKey) != pointPackages.end() ||
            pendingPointPackages.find(socketKey) != pendingPointPackages.end() ||
            remeshPackages.find(socketKey) != remeshPackages.end() ||
            pendingRemeshPackages.find(socketKey) != pendingRemeshPackages.end() ||
            voronoiPackages.find(socketKey) != voronoiPackages.end() ||
            pendingVoronoiPackages.find(socketKey) != pendingVoronoiPackages.end() ||
            heatPackages.find(socketKey) != heatPackages.end() ||
            pendingHeatPackages.find(socketKey) != pendingHeatPackages.end() ||
            cfdPackages.find(socketKey) != cfdPackages.end() ||
            pendingCfdPackages.find(socketKey) != pendingCfdPackages.end();
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
