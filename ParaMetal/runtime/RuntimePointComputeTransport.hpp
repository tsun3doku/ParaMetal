#pragma once

#include <cstdint>
#include <unordered_set>
#include <unordered_map>

#include "runtime/RuntimeECS.hpp"
#include "runtime/RuntimePackages.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "runtime/PointComputeRuntime.hpp"

class RuntimePointComputeTransport {
public:
    void setRuntime(PointComputeRuntime* updatedRuntime) {
        runtime = updatedRuntime;
    }

    void setECSRegistry(ECSRegistry* updatedRegistry) {
        ecsRegistry = updatedRegistry;
    }

    void sync(const ECSRegistry& registry);
    void finalizeSync();

private:
    bool tryBuildConfig(uint64_t socketKey, const PointPackage& package, PointComputeRuntime::Config& outConfig) const;
    void removePublishedProduct(uint64_t socketKey);
    void publishProduct(uint64_t socketKey);
    uint64_t buildConfigInputHash(uint64_t socketKey, const PointPackage& package) const;

    PointComputeRuntime* runtime = nullptr;
    ECSRegistry* ecsRegistry = nullptr;
    std::unordered_set<uint64_t> activeSocketKeys;
    std::unordered_map<uint64_t, uint64_t> appliedConfigInputHash;
};
