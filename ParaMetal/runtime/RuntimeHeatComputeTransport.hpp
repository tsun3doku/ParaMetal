#pragma once

#include <cstdint>

#include "heat/HeatSystemComputeController.hpp"
#include "runtime/ComputeTransportResult.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "runtime/package/RuntimePackages.hpp"

class RuntimeProductManager;

class RuntimeHeatComputeTransport {
public:
    void setController(HeatSystemComputeController* updatedController) {
        controller = updatedController;
    }

    void setProducts(RuntimeProductManager* updatedProducts) {
        products = updatedProducts;
    }

    uint64_t getDisplayRevision() const {
        return controller ? controller->getDisplayRevision() : 0;
    }

    ComputeApplyResult apply(
        uint64_t socketKey,
        const HeatPackage& package,
        HeatProduct& outProduct);
    void commitPending(uint64_t socketKey);
    void cancelPending(uint64_t socketKey);
    void remove(uint64_t socketKey);

private:
    uint64_t advectionUpstreamSocket(const HeatPackage& package) const;
    bool tryBuildConfig(uint64_t socketKey, const HeatPackage& package, HeatSystemComputeController::Config& outConfig) const;

    HeatSystemComputeController* controller = nullptr;
    RuntimeProductManager* products = nullptr;
};
