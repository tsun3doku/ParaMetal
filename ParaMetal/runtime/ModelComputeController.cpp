#include "ModelComputeController.hpp"

#include "hash/HashProduct.hpp"
#include "render/RenderConfig.hpp"
#include "scene/Model.hpp"
#include "scene/ModelUploader.hpp"
#include "vulkan/ModelRegistry.hpp"

#include <chrono>
#include <thread>

ModelComputeController::ModelComputeController(
    ModelRegistry& modelRegistry,
    ModelUploader& modelUploader,
    std::atomic<bool>& isOperating)
    : modelRegistry(modelRegistry),
      modelUploader(modelUploader),
      isOperating(isOperating) {
}

bool ModelComputeController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0 || config.vertexPositions.empty() ||
        config.triangleIndices.empty() || config.computeHash == 0) {
        return false;
    }

    std::lock_guard<std::mutex> lock(executionMutex);
    if (pendingModels.find(socketKey) != pendingModels.end()) {
        return false;
    }

    const auto runtimeIt = runtimeModelIdBySocketKey.find(socketKey);
    if (runtimeIt != runtimeModelIdBySocketKey.end() &&
        modelRegistry.isRetiring(runtimeIt->second)) {
        return false;
    }

    const auto hashesIt = configuredComputeHashes.find(socketKey);
    if (hashesIt != configuredComputeHashes.end() &&
        hashesIt->second == config.computeHash) {
        return true;
    }

    return prepareModel(socketKey, config);
}

bool ModelComputeController::buildProduct(uint64_t socketKey, ModelProduct& product) const {
    std::lock_guard<std::mutex> lock(executionMutex);
    product = {};
    uint32_t runtimeModelId = 0;
    const auto pendingIt = pendingModels.find(socketKey);
    if (pendingIt != pendingModels.end()) {
        runtimeModelId = pendingIt->second.runtimeModelId;
    } else {
        const auto runtimeIt = runtimeModelIdBySocketKey.find(socketKey);
        if (runtimeIt != runtimeModelIdBySocketKey.end()) {
            runtimeModelId = runtimeIt->second;
        }
    }
    if (runtimeModelId == 0 || !modelRegistry.exportProduct(runtimeModelId, product)) {
        return false;
    }
    HashProduct::seal(product);
    return product.isValid();
}

bool ModelComputeController::commitPending(uint64_t socketKey) {
    if (socketKey == 0) return false;

    std::lock_guard<std::mutex> lock(executionMutex);
    const auto pendingIt = pendingModels.find(socketKey);
    if (pendingIt == pendingModels.end() || pendingIt->second.runtimeModelId == 0) {
        return false;
    }

    const auto runtimeIt = runtimeModelIdBySocketKey.find(socketKey);
    if (runtimeIt != runtimeModelIdBySocketKey.end() &&
        modelRegistry.isRetiring(runtimeIt->second)) {
        return false;
    }

    const uint32_t previousModelId =
        runtimeIt != runtimeModelIdBySocketKey.end() ? runtimeIt->second : 0;
    if (previousModelId != 0 &&
        modelRegistry.hasModel(previousModelId) &&
        !modelRegistry.removeModelByID(previousModelId)) {
        return false;
    }

    const uint32_t committedModelId = pendingIt->second.runtimeModelId;
    const uint64_t computeHash = pendingIt->second.computeHash;

    runtimeModelIdBySocketKey[socketKey] = committedModelId;
    configuredComputeHashes[socketKey] = computeHash;
    pendingModels.erase(pendingIt);
    return true;
}

void ModelComputeController::cancelPending(uint64_t socketKey) {
    if (socketKey == 0) return;

    std::lock_guard<std::mutex> lock(executionMutex);
    const auto it = pendingModels.find(socketKey);
    if (it == pendingModels.end()) {
        return;
    }
    const uint32_t pendingModelId = it->second.runtimeModelId;
    if (pendingModelId != 0 && modelRegistry.hasModel(pendingModelId)) {
        modelRegistry.removeModelByID(pendingModelId);
    }
    pendingModels.erase(it);
}

bool ModelComputeController::remove(uint64_t socketKey) {
    if (socketKey == 0) return true;
    std::lock_guard<std::mutex> lock(executionMutex);
    return removeModel(socketKey);
}

void ModelComputeController::disableAll() {
    std::lock_guard<std::mutex> lock(executionMutex);
    if (runtimeModelIdBySocketKey.empty() && pendingModels.empty()) return;

    OperatingScope operatingScope(isOperating);
    for (const auto& [socketKey, pending] : pendingModels) {
        (void)socketKey;
        if (pending.runtimeModelId != 0 && modelRegistry.hasModel(pending.runtimeModelId)) {
            modelRegistry.removeModelByID(pending.runtimeModelId);
        }
    }
    pendingModels.clear();
    // Removed models are retained whole, no in-place destruction needs a wait
    for (const auto& [socketKey, runtimeModelId] : runtimeModelIdBySocketKey) {
        (void)socketKey;
        modelRegistry.removeModelByID(runtimeModelId);
    }
    runtimeModelIdBySocketKey.clear();
    configuredComputeHashes.clear();
}

uint64_t ModelComputeController::getCurrentComputeHash(uint64_t socketKey) const {
    std::lock_guard<std::mutex> lock(executionMutex);
    const auto it = configuredComputeHashes.find(socketKey);
    return it != configuredComputeHashes.end() ? it->second : 0;
}

bool ModelComputeController::hasAnyRetiring() const {
    std::lock_guard<std::mutex> lock(executionMutex);
    return modelRegistry.hasAnyRetiring();
}

void ModelComputeController::clearAllRetiring() {
    std::lock_guard<std::mutex> lock(executionMutex);
    modelRegistry.clearAllRetiring();
}

bool ModelComputeController::prepareModel(uint64_t socketKey, const Config& config) {
    OperatingScope operatingScope(isOperating);
    std::this_thread::sleep_for(std::chrono::milliseconds(renderconfig::ModelLoadPauseMs));

    std::unique_ptr<Model> model = modelUploader.createModel();
    if (!model) return false;

    const bool hasRenderGeometry =
        !config.renderPositions.empty() && !config.renderIndices.empty();
    model->setGeometry(config.vertexPositions, config.triangleIndices);
    if (hasRenderGeometry) {
        model->setRenderGeometry(
            config.renderPositions,
            config.renderNormals,
            config.renderTexcoords,
            config.renderIndices);
    }
    model->setGeometryIdentity(config.computeHash);
    if (!model->init()) return false;

    const uint32_t runtimeModelId = modelRegistry.addModel(std::move(model));
    if (runtimeModelId == 0) return false;
    modelRegistry.setModelVisible(runtimeModelId, false);

    pendingModels[socketKey] = PendingModel{runtimeModelId, config.computeHash};
    return true;
}

bool ModelComputeController::removeModel(uint64_t socketKey) {
    bool success = true;

    const auto pendingIt = pendingModels.find(socketKey);
    if (pendingIt != pendingModels.end()) {
        const uint32_t pendingModelId = pendingIt->second.runtimeModelId;
        if (pendingModelId != 0 && modelRegistry.hasModel(pendingModelId)) {
            success = modelRegistry.removeModelByID(pendingModelId) && success;
        }
        pendingModels.erase(pendingIt);
    }

    const auto runtimeIt = runtimeModelIdBySocketKey.find(socketKey);
    if (runtimeIt == runtimeModelIdBySocketKey.end()) {
        return success;
    }
    const uint32_t runtimeModelId = runtimeIt->second;

    OperatingScope operatingScope(isOperating);
    if (modelRegistry.hasModel(runtimeModelId) &&
        !modelRegistry.removeModelByID(runtimeModelId)) {
        return false;
    }
    runtimeModelIdBySocketKey.erase(runtimeIt);
    configuredComputeHashes.erase(socketKey);
    return success;
}

ModelComputeController::OperatingScope::OperatingScope(std::atomic<bool>& isOperating)
    : isOperating(isOperating) {
    previousState = isOperating.exchange(true, std::memory_order_acq_rel);
}

ModelComputeController::OperatingScope::~OperatingScope() {
    isOperating.store(previousState, std::memory_order_release);
}
