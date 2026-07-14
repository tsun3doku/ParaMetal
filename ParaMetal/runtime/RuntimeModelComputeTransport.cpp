#include "RuntimeModelComputeTransport.hpp"

#include "util/GeometryUtils.hpp"
#include "runtime/ModelComputeRuntime.hpp"

#include <iostream>

ProductHandle RuntimeModelComputeTransport::apply(uint64_t socketKey, const ModelPackage& package) {
    if (!modelRuntime || !products || socketKey == 0) {
        return {};
    }

    if (package.geometry.baseModelPath.empty()) {
        return {};
    }

    modelRuntime->queueAcquireSocket(socketKey, package.geometry.baseModelPath);
    modelRuntime->flush();

    uint32_t runtimeModelId = 0;
    if (!modelRuntime->tryGetRuntimeModelId(socketKey, runtimeModelId) || runtimeModelId == 0) {
        std::cerr << "[ModelCompute] No runtime model ID for socketKey=" << socketKey
                  << ", baseModelPath='" << package.geometry.baseModelPath << "'" << std::endl;
        return {};
    }

    modelRuntime->setModelMatrix(runtimeModelId, toMat4(package.geometry.localToWorld));

    ModelProduct product{};
    if (!modelRuntime->exportProduct(runtimeModelId, product)) {
        return {};
    }

    ProductHandle handle = products->publish<ModelProduct>(socketKey, product);
    return handle;
}


void RuntimeModelComputeTransport::remove(uint64_t socketKey) {
    if (!modelRuntime || socketKey == 0) {
        return;
    }
    modelRuntime->queueReleaseSocket(socketKey);
    modelRuntime->flush();
}

void RuntimeModelComputeTransport::flush() {
    if (modelRuntime) {
        modelRuntime->flush();
    }
}
