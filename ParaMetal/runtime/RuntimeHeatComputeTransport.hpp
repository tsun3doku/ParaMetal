#pragma once


#include "heat/HeatSystemComputeController.hpp"
#include "runtime/package/RuntimePackages.hpp"
#include "runtime/RuntimeProductManager.hpp"
#include "simgraph/SimGraphRuntime.hpp"

class RuntimeHeatComputeTransport {
public:
    void setController(HeatSystemComputeController* updatedController) {
        controller = updatedController;
    }

    void setProducts(RuntimeProductManager* updatedProducts) {
        products = updatedProducts;
    }

    void setSimGraph(SimGraphRuntime* updatedSimGraph) {
        simGraph = updatedSimGraph;
    }

    ProductHandle apply(uint64_t socketKey, const HeatPackage& package);
    void remove(uint64_t socketKey);

private:
    bool tryBuildConfig(uint64_t socketKey, const HeatPackage& package, HeatSystemComputeController::Config& outConfig) const;

    HeatSystemComputeController* controller = nullptr;

    SimGraphRuntime* simGraph = nullptr;
    RuntimeProductManager* products = nullptr;
};
