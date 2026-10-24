#pragma once

#include "cfd/CFDSystemComputeController.hpp"
#include "runtime/package/RuntimePackages.hpp"
#include "runtime/RuntimeProductManager.hpp"
#include "simgraph/SimGraphRuntime.hpp"

class RuntimeCFDComputeTransport {
public:
    void setController(cfd::CFDSystemComputeController* updatedController) {
        controller = updatedController;
    }

    void setSimGraph(SimGraphRuntime* updatedSimGraph) {
        simGraph = updatedSimGraph;
    }

    void setProducts(RuntimeProductManager* updatedProducts) {
        products = updatedProducts;
    }

    ProductHandle apply(uint64_t socketKey, const CFDPackage& package);
    void remove(uint64_t socketKey);

private:
    bool tryBuildConfig(
        uint64_t socketKey,
        const CFDPackage& package,
        cfd::CFDSystemComputeController::Config& outConfig,
        topology::GlobalDomain& outDomain) const;

    cfd::CFDSystemComputeController* controller = nullptr;

    SimGraphRuntime* simGraph = nullptr;
    RuntimeProductManager* products = nullptr;
};
