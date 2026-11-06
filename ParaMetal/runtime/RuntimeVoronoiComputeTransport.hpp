#pragma once

#include <cstdint>

#include "runtime/ComputeTransportResult.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "runtime/package/RuntimePackages.hpp"
#include "voronoi/VoronoiSystemComputeController.hpp"

class RuntimeProductManager;

class RuntimeVoronoiComputeTransport {
public:
    void setController(VoronoiSystemComputeController* updatedController) {
        controller = updatedController;
    }

    void setProducts(RuntimeProductManager* updatedProducts) {
        products = updatedProducts;
    }

    ComputeApplyResult apply(
        uint64_t socketKey,
        const VoronoiPackage& package,
        VoronoiProduct& outProduct);
    void commitPending(uint64_t socketKey);
    void cancelPending(uint64_t socketKey);
    void remove(uint64_t socketKey);

private:
    VoronoiSystemComputeController* controller = nullptr;
    RuntimeProductManager* products = nullptr;
};
