#pragma once

#include <cstdint>
#include <string>

#include "runtime/RuntimePackages.hpp"
#include "runtime/RuntimeProductManager.hpp"

class ModelComputeRuntime;

class RuntimeModelComputeTransport {
public:
    void setRuntime(ModelComputeRuntime* updatedRuntime) {
        modelRuntime = updatedRuntime;
    }

    void setProducts(RuntimeProductManager* updatedProducts) {
        products = updatedProducts;
    }

    ProductHandle apply(uint64_t socketKey, const ModelPackage& package);
    void remove(uint64_t socketKey);
    void flush();

private:
    ModelComputeRuntime* modelRuntime = nullptr;
    RuntimeProductManager* products = nullptr;
};
