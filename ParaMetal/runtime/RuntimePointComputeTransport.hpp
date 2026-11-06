#pragma once

#include <cstdint>

#include "runtime/ComputeTransportResult.hpp"
#include "runtime/PointComputeRuntime.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "runtime/package/RuntimePackages.hpp"

class RuntimePointComputeTransport {
public:
    void setRuntime(PointComputeRuntime* updatedRuntime) {
        runtime = updatedRuntime;
    }

    ComputeApplyResult apply(uint64_t socketKey, const PointPackage& package, PointProduct& outProduct);
    void commitPending(uint64_t socketKey);
    void cancelPending(uint64_t socketKey);
    void remove(uint64_t socketKey);

private:
    PointComputeRuntime* runtime = nullptr;
};
