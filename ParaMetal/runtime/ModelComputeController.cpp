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

void ModelComputeController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0 || config.pointPositions.empty() ||
        config.triangleIndices.empty() || config.computeHash == 0) {
        remove(socketKey);
        return;
    }

    std::lock_guard<std::mutex> lock(executionMutex);
    const auto runtimeIt = runtimeModelIdBySocketKey.find(socketKey);
    const uint32_t runtimeModelId = runtimeIt != runtimeModelIdBySocketKey.end()
        ? runtimeIt->second : 0;
    const auto hashesIt = configuredComputeHashes.find(socketKey);
    if (runtimeModelId != 0 && hashesIt != configuredComputeHashes.end() &&
        hashesIt->second == config.computeHash) {
        return;
    }

    if (!rebuildModel(socketKey, config, runtimeModelId)) {
        removeModel(socketKey);
    }
}

bool ModelComputeController::buildProduct(uint64_t socketKey, ModelProduct& product) const {
    std::lock_guard<std::mutex> lock(executionMutex);
    product = {};
    const auto runtimeIt = runtimeModelIdBySocketKey.find(socketKey);
    if (runtimeIt == runtimeModelIdBySocketKey.end() || runtimeIt->second == 0 ||
        !modelRegistry.exportProduct(runtimeIt->second, product)) {
        return false;
    }
    HashProduct::seal(product);
    return product.isValid();
}

void ModelComputeController::remove(uint64_t socketKey) {
    if (socketKey == 0) return;
    std::lock_guard<std::mutex> lock(executionMutex);
    removeModel(socketKey);
}

void ModelComputeController::disableAll() {
    std::lock_guard<std::mutex> lock(executionMutex);
    if (runtimeModelIdBySocketKey.empty()) return;

    OperatingScope operatingScope(isOperating);
    // Removed models are retained whole, no in-place destruction needs a wait
    for (const auto& [socketKey, runtimeModelId] : runtimeModelIdBySocketKey) {
        (void)socketKey;
        modelRegistry.removeModelByID(runtimeModelId);
    }
    runtimeModelIdBySocketKey.clear();
    socketKeyByRuntimeModelId.clear();
    configuredComputeHashes.clear();
}

bool ModelComputeController::tryGetSocketKey(uint32_t runtimeModelId, uint64_t& outSocketKey) const {
    std::lock_guard<std::mutex> lock(executionMutex);
    outSocketKey = 0;
    const auto it = socketKeyByRuntimeModelId.find(runtimeModelId);
    if (it == socketKeyByRuntimeModelId.end()) return false;
    outSocketKey = it->second;
    return outSocketKey != 0;
}

bool ModelComputeController::rebuildModel(uint64_t socketKey, const Config& config, uint32_t preferredModelId) {
    OperatingScope operatingScope(isOperating);
    std::this_thread::sleep_for(std::chrono::milliseconds(renderconfig::ModelLoadPauseMs));

    std::unique_ptr<Model> model = modelUploader.createModel();
    if (!model) return false;

    const bool hasRenderGeometry =
        !config.renderPositions.empty() && !config.renderIndices.empty();
    model->setGeometry(config.pointPositions, config.triangleIndices);
    if (hasRenderGeometry) {
        model->setRenderGeometry(
            config.renderPositions,
            config.renderNormals,
            config.renderTexcoords,
            config.renderIndices);
    }
    if (!model->init()) return false;

    if (preferredModelId != 0) {
        modelRegistry.removeModelByID(preferredModelId);
        socketKeyByRuntimeModelId.erase(preferredModelId);
    }
    const uint32_t runtimeModelId = modelRegistry.addModel(std::move(model), preferredModelId);
    if (runtimeModelId == 0) return false;
    modelRegistry.setModelVisible(runtimeModelId, false);

    runtimeModelIdBySocketKey[socketKey] = runtimeModelId;
    socketKeyByRuntimeModelId[runtimeModelId] = socketKey;
    configuredComputeHashes[socketKey] = config.computeHash;
    return true;
}

bool ModelComputeController::removeModel(uint64_t socketKey) {
    const auto runtimeIt = runtimeModelIdBySocketKey.find(socketKey);
    if (runtimeIt == runtimeModelIdBySocketKey.end()) return false;
    const uint32_t runtimeModelId = runtimeIt->second;

    OperatingScope operatingScope(isOperating);
    const bool removed = modelRegistry.removeModelByID(runtimeModelId);
    runtimeModelIdBySocketKey.erase(runtimeIt);
    socketKeyByRuntimeModelId.erase(runtimeModelId);
    configuredComputeHashes.erase(socketKey);
    return removed;
}

ModelComputeController::OperatingScope::OperatingScope(std::atomic<bool>& isOperating)
    : isOperating(isOperating) {
    previousState = isOperating.exchange(true, std::memory_order_acq_rel);
}

ModelComputeController::OperatingScope::~OperatingScope() {
    isOperating.store(previousState, std::memory_order_release);
}
