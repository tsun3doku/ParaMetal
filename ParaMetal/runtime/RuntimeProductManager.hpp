#pragma once

#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "runtime/package/RuntimePackages.hpp"
#include "runtime/RuntimeProducts.hpp"

class RuntimeProductManager {
public:
    RuntimeProductManager() = default;
    ~RuntimeProductManager();

    template <typename ProductT>
    ProductHandle publish(uint64_t socketKey, ProductT& product) {
        if (socketKey == 0 || product.hashes.full == 0) {
            return {};
        }

        ProductHandle handle = makeHandle<ProductT>(socketKey, product);
        if (!handle.isValid()) {
            return {};
        }

        handle.publicationId = nextPublicationId;
        productsFor<ProductT>()[socketKey].emplace_back(handle, std::move(product));
        product = {};
        ++nextPublicationId;
        return handle;
    }

    template <typename ProductT>
    const ProductT* resolve(const ProductHandle& handle) const {
        if (handle.type != type<ProductT>() || !handle.isValid()) {
            return nullptr;
        }

        const auto productsIt = productsFor<ProductT>().find(handle.outputSocketKey);
        if (productsIt == productsFor<ProductT>().end()) {
            return nullptr;
        }

        for (const auto& [storedHandle, storedProduct] : productsIt->second) {
            if (storedHandle.publicationId == handle.publicationId &&
                storedHandle.hashes.full == handle.hashes.full) {
                return &storedProduct;
            }
        }
        return nullptr;
    }

    // Clears publication records, buffers are owned and freed by producing runtimes
    void clear();

private:
    template <typename ProductT>
    using ProductStorage = std::vector<std::pair<ProductHandle, ProductT>>;

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
    std::unordered_map<uint64_t, ProductStorage<ProductT>>& productsFor() {
        if constexpr (std::is_same_v<ProductT, ModelProduct>) return modelProducts;
        else if constexpr (std::is_same_v<ProductT, RemeshProduct>) return remeshProducts;
        else if constexpr (std::is_same_v<ProductT, VoronoiProduct>) return voronoiProducts;
        else if constexpr (std::is_same_v<ProductT, PointProduct>) return pointProducts;
        else if constexpr (std::is_same_v<ProductT, CFDProduct>) return cfdProducts;
        else return heatProducts;
    }

    template <typename ProductT>
    const std::unordered_map<uint64_t, ProductStorage<ProductT>>& productsFor() const {
        if constexpr (std::is_same_v<ProductT, ModelProduct>) return modelProducts;
        else if constexpr (std::is_same_v<ProductT, RemeshProduct>) return remeshProducts;
        else if constexpr (std::is_same_v<ProductT, VoronoiProduct>) return voronoiProducts;
        else if constexpr (std::is_same_v<ProductT, PointProduct>) return pointProducts;
        else if constexpr (std::is_same_v<ProductT, CFDProduct>) return cfdProducts;
        else return heatProducts;
    }

    std::unordered_map<uint64_t, ProductStorage<ModelProduct>> modelProducts;
    std::unordered_map<uint64_t, ProductStorage<RemeshProduct>> remeshProducts;
    std::unordered_map<uint64_t, ProductStorage<VoronoiProduct>> voronoiProducts;
    std::unordered_map<uint64_t, ProductStorage<PointProduct>> pointProducts;
    std::unordered_map<uint64_t, ProductStorage<HeatProduct>> heatProducts;
    std::unordered_map<uint64_t, ProductStorage<CFDProduct>> cfdProducts;

    uint64_t nextPublicationId = 1;
};
