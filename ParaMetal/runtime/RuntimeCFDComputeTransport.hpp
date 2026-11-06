#pragma once

#include <cstdint>

#include "cfd/CFDSystemComputeController.hpp"
#include "runtime/ComputeTransportResult.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "runtime/package/RuntimePackages.hpp"

class RuntimeProductManager;

class RuntimeCFDComputeTransport {
public:
    void setController(cfd::CFDSystemComputeController* updatedController) {
        controller = updatedController;
    }

    void setProducts(RuntimeProductManager* updatedProducts) {
        products = updatedProducts;
    }

    ComputeApplyResult apply(
        uint64_t socketKey,
        const CFDPackage& package,
        CFDProduct& outProduct);
    void commitPending(uint64_t socketKey);
    void cancelPending(uint64_t socketKey);
    void remove(uint64_t socketKey);

private:
    bool tryBuildConfig(
        uint64_t socketKey,
        const CFDPackage& package,
        cfd::CFDSystemComputeController::Config& outConfig,
        topology::GlobalDomain& outDomain) const;

    cfd::CFDSystemComputeController* controller = nullptr;
    RuntimeProductManager* products = nullptr;
};
