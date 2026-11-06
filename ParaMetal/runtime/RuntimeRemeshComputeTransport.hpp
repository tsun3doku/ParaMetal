#pragma once

#include <cstdint>

#include "runtime/ComputeTransportResult.hpp"
#include "runtime/RemeshController.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "runtime/package/RuntimePackages.hpp"

class RuntimeProductManager;

class RuntimeRemeshComputeTransport {
public:
    void setController(RemeshController* updatedController) {
        controller = updatedController;
    }

    void setProducts(RuntimeProductManager* updatedProducts) {
        products = updatedProducts;
    }

    ComputeApplyResult apply(uint64_t socketKey, const RemeshPackage& package, RemeshProduct& outProduct);
    void commitPending(uint64_t socketKey);
    void cancelPending(uint64_t socketKey);
    void remove(uint64_t socketKey);

private:
    RemeshController* controller = nullptr;
    RuntimeProductManager* products = nullptr;
};
