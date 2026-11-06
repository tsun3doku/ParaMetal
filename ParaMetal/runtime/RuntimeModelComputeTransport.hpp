#pragma once

#include <cstdint>

#include "runtime/ComputeTransportResult.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "runtime/package/RuntimePackages.hpp"

class ModelComputeController;

class RuntimeModelComputeTransport {
public:
    void setController(ModelComputeController* updatedController) {
        controller = updatedController;
    }

    ComputeApplyResult apply(uint64_t socketKey, const ModelPackage& package, ModelProduct& outProduct);
    void remove(uint64_t socketKey);

private:
    ModelComputeController* controller = nullptr;
};
