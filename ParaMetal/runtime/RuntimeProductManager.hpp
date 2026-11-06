#pragma once

#include <cstdint>
#include <iostream>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "nodegraph/NodeGraphProductTypes.hpp"
#include "runtime/RuntimeProducts.hpp"

template <typename ProductT>
struct PublishedProduct {
    ProductHandle handle;
    ProductT product;
};

// A socket owns exactly one published generation plus at most one retiring
// generation awaiting GPU completion.
template <typename ProductT>
struct ProductSlot {
    std::optional<PublishedProduct<ProductT>> current;
    std::optional<PublishedProduct<ProductT>> retiring;
};

// ProductHandle::operator== is content-based, so lifetime decisions must match
// the publication identity explicitly.
inline bool isSamePublication(const ProductHandle& a, const ProductHandle& b) {
    return a.type == b.type &&
           a.outputSocketKey == b.outputSocketKey &&
           a.publicationId == b.publicationId;
}

class RuntimeProductManager {
public:
    RuntimeProductManager() = default;
    ~RuntimeProductManager();

    // A socket publishes only after its retiring generation has been reclaimed.
    template <typename ProductT>
    ProductHandle publish(uint64_t socketKey, ProductT& product) {
        if (socketKey == 0 || product.hashes.full == 0) {
            return {};
        }
        auto& map = productsFor<ProductT>();
        auto [it, _] = map.try_emplace(socketKey);
        if (it->second.retiring.has_value()) {
            std::cerr << "[RuntimeProductManager] publish rejected: retiring slot occupied for socket "
                      << socketKey << std::endl;
            return {};
        }

        ProductHandle handle = makeHandle<ProductT>(socketKey, product);
        handle.publicationId = nextPublicationId++;
        PublishedProduct<ProductT> prepared{handle, std::move(product)};
        product = {};

        it->second.retiring = std::move(it->second.current);
        it->second.current = std::move(prepared);
        ++revision;
        return handle;
    }

    template <typename ProductT>
    const ProductHandle& getCurrentHandle(uint64_t socketKey) const {
        static const ProductHandle emptyHandle{};
        const auto it = productsFor<ProductT>().find(socketKey);
        return (it != productsFor<ProductT>().end() && it->second.current.has_value())
            ? it->second.current->handle
            : emptyHandle;
    }

    // Socket-keyed removal for lifecycle orchestration; the manager owns identity
    template <typename ProductT>
    bool unpublish(uint64_t socketKey) {
        if (socketKey == 0) {
            return false;
        }
        auto it = productsFor<ProductT>().find(socketKey);
        if (it == productsFor<ProductT>().end() || !it->second.current.has_value()) {
            return false;
        }
        auto& slot = it->second;
        if (slot.retiring.has_value()) {
            std::cerr << "[RuntimeProductManager] Error: retiring slot occupied during unpublish for socket "
                      << socketKey << std::endl;
            return false;
        }
        slot.retiring = std::move(slot.current);
        slot.current.reset();
        ++revision;
        return true;
    }

    template <typename ProductT>
    const ProductT* resolve(const ProductHandle& handle) const {
        if (handle.type != type<ProductT>() || !handle.isValid()) {
            return nullptr;
        }
        const auto it = productsFor<ProductT>().find(handle.outputSocketKey);
        if (it == productsFor<ProductT>().end()) {
            return nullptr;
        }
        const auto& slot = it->second;
        if (slot.current.has_value() &&
            isSamePublication(slot.current->handle, handle) &&
            slot.current->handle.hashes.full == handle.hashes.full) {
            return &slot.current->product;
        }
        if (slot.retiring.has_value() &&
            isSamePublication(slot.retiring->handle, handle) &&
            slot.retiring->handle.hashes.full == handle.hashes.full) {
            return &slot.retiring->product;
        }
        return nullptr;
    }

    template <typename ProductT>
    bool isCurrent(const ProductHandle& handle) const {
        if (handle.type != type<ProductT>() || !handle.isValid()) {
            return false;
        }
        const auto it = productsFor<ProductT>().find(handle.outputSocketKey);
        if (it == productsFor<ProductT>().end() || !it->second.current.has_value()) {
            return false;
        }
        return isSamePublication(it->second.current->handle, handle);
    }

    uint64_t getRevision() const { return revision; }

    bool hasAnyRetiring() const {
        const auto anyRetiring = [](const auto& map) {
            for (const auto& [socketKey, slot] : map) {
                (void)socketKey;
                if (slot.retiring.has_value()) return true;
            }
            return false;
        };
        return anyRetiring(modelProducts) || anyRetiring(remeshProducts) ||
               anyRetiring(voronoiProducts) || anyRetiring(pointProducts) ||
               anyRetiring(heatProducts) || anyRetiring(cfdProducts);
    }

    // Called only after GPU completion has been established for every retiring generation.
    void clearAllRetiring() noexcept {
        const auto clearMap = [](auto& map) {
            for (auto& [socketKey, slot] : map) {
                (void)socketKey;
                slot.retiring.reset();
            }
        };
        clearMap(modelProducts);
        clearMap(remeshProducts);
        clearMap(voronoiProducts);
        clearMap(pointProducts);
        clearMap(heatProducts);
        clearMap(cfdProducts);
        ++revision;
    }

    // Unconditional reset used during quiesced project teardown
    void quiescedClear() noexcept {
        modelProducts.clear();
        remeshProducts.clear();
        voronoiProducts.clear();
        pointProducts.clear();
        heatProducts.clear();
        cfdProducts.clear();
        ++revision;
    }

    void clear() { quiescedClear(); }

private:
    template <typename ProductT>
    static NodeProductType type() {
        if constexpr (std::is_same_v<ProductT, ModelProduct>) return NodeProductType::Model;
        else if constexpr (std::is_same_v<ProductT, RemeshProduct>) return NodeProductType::Remesh;
        else if constexpr (std::is_same_v<ProductT, VoronoiProduct>) return NodeProductType::Voronoi;
        else if constexpr (std::is_same_v<ProductT, PointProduct>) return NodeProductType::Point;
        else if constexpr (std::is_same_v<ProductT, HeatProduct>) return NodeProductType::Heat;
        else if constexpr (std::is_same_v<ProductT, CFDProduct>) return NodeProductType::CFD;
        else return NodeProductType::None;
    }

    template <typename ProductT>
    static ProductHandle makeHandle(uint64_t socketKey, const ProductT& product) {
        ProductHandle handle{};
        handle.type = type<ProductT>();
        handle.outputSocketKey = socketKey;
        handle.hashes = product.hashes;
        return handle;
    }

    template <typename ProductT>
    std::unordered_map<uint64_t, ProductSlot<ProductT>>& productsFor() {
        if constexpr (std::is_same_v<ProductT, ModelProduct>) return modelProducts;
        else if constexpr (std::is_same_v<ProductT, RemeshProduct>) return remeshProducts;
        else if constexpr (std::is_same_v<ProductT, VoronoiProduct>) return voronoiProducts;
        else if constexpr (std::is_same_v<ProductT, PointProduct>) return pointProducts;
        else if constexpr (std::is_same_v<ProductT, CFDProduct>) return cfdProducts;
        else return heatProducts;
    }

    template <typename ProductT>
    const std::unordered_map<uint64_t, ProductSlot<ProductT>>& productsFor() const {
        if constexpr (std::is_same_v<ProductT, ModelProduct>) return modelProducts;
        else if constexpr (std::is_same_v<ProductT, RemeshProduct>) return remeshProducts;
        else if constexpr (std::is_same_v<ProductT, VoronoiProduct>) return voronoiProducts;
        else if constexpr (std::is_same_v<ProductT, PointProduct>) return pointProducts;
        else if constexpr (std::is_same_v<ProductT, CFDProduct>) return cfdProducts;
        else return heatProducts;
    }

    std::unordered_map<uint64_t, ProductSlot<ModelProduct>> modelProducts;
    std::unordered_map<uint64_t, ProductSlot<RemeshProduct>> remeshProducts;
    std::unordered_map<uint64_t, ProductSlot<VoronoiProduct>> voronoiProducts;
    std::unordered_map<uint64_t, ProductSlot<PointProduct>> pointProducts;
    std::unordered_map<uint64_t, ProductSlot<HeatProduct>> heatProducts;
    std::unordered_map<uint64_t, ProductSlot<CFDProduct>> cfdProducts;

    uint64_t nextPublicationId = 1;
    uint64_t revision = 0;
};
