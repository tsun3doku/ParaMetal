#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
#include <stdexcept>

#include "MemoryAllocator.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "hash/HashProduct.hpp"
#include "scene/Model.hpp"
#include "ModelRegistry.hpp"

ModelRegistry::ModelRegistry(MemoryAllocator& memoryAllocator)
    : memoryAllocator(memoryAllocator) {
}

ModelRegistry::~ModelRegistry() {
}

void ModelRegistry::clearModels() {
    for (auto& [modelId, model] : models) {
        (void)modelId;
        if (model) {
            model->cleanup();
        }
    }
    models.clear();
    visibleModelIds.clear();
    recycledModelIds.clear();
    nextModelId = 1;
}

uint32_t ModelRegistry::addModel(std::unique_ptr<Model> model, uint32_t preferredModelId) {
    if (!model) {
        return 0;
    }

    const uint32_t modelId = acquireModelId(preferredModelId);
    model->setRuntimeModelId(modelId);
    models.emplace(modelId, std::move(model));
    visibleModelIds.insert(modelId);
    return modelId;
}

bool ModelRegistry::removeModelByID(uint32_t modelID) {
    if (modelID == 0) {
        return false;
    }

    const auto it = models.find(modelID);
    if (it == models.end()) {
        return false;
    }

    if (it->second) {
        it->second->cleanup();
    }
    
    unregisterModel(modelID);
    models.erase(it);
    return true;
}

std::vector<uint32_t> ModelRegistry::getRenderableModelIds() const {
    std::vector<uint32_t> modelIds;
    modelIds.reserve(visibleModelIds.size());
    for (uint32_t modelId : visibleModelIds) {
        if (models.find(modelId) == models.end()) {
            continue;
        }
        modelIds.push_back(modelId);
    }

    std::sort(modelIds.begin(), modelIds.end());
    return modelIds;
}

bool ModelRegistry::hasModel(uint32_t modelID) const {
    return models.find(modelID) != models.end();
}

bool ModelRegistry::setModelVisible(uint32_t modelID, bool visible) {
    if (modelID == 0 || !hasModel(modelID)) {
        return false;
    }

    if (visible) {
        visibleModelIds.insert(modelID);
    } else {
        visibleModelIds.erase(modelID);
    }

    return true;
}

bool ModelRegistry::isModelVisible(uint32_t modelID) const {
    return visibleModelIds.find(modelID) != visibleModelIds.end();
}

bool ModelRegistry::exportProduct(uint32_t modelID, ModelProduct& outProduct) const {
    outProduct = {};

    Model* model = const_cast<Model*>(findModel(modelID));
    if (!model) {
        return false;
    }

    outProduct.runtimeModelId = modelID;
    outProduct.vertexBuffer = model->getVertexBuffer();
    outProduct.vertexBufferOffset = model->getVertexBufferOffset();
    outProduct.indexBuffer = model->getIndexBuffer();
    outProduct.indexBufferOffset = model->getIndexBufferOffset();
    outProduct.indexCount = static_cast<uint32_t>(model->getIndices().size());
    outProduct.renderVertexBuffer = model->getRenderVertexBuffer();
    outProduct.renderVertexBufferOffset = model->getRenderVertexBufferOffset();
    outProduct.renderIndexBuffer = model->getRenderIndexBuffer();
    outProduct.renderIndexBufferOffset = model->getRenderIndexBufferOffset();
    outProduct.renderIndexCount = static_cast<uint32_t>(model->getRenderIndices().size());
    outProduct.modelMatrix = model->getModelMatrix();
    HashProduct::seal(outProduct);
    return outProduct.isValid();
}

bool ModelRegistry::setModelMatrix(uint32_t modelID, const glm::mat4& matrix) {
    Model* model = findModel(modelID);
    if (!model) {
        return false;
    }

    model->setModelMatrix(matrix);
    return true;
}

bool ModelRegistry::tryGetModelMatrix(uint32_t modelID, glm::mat4& outMatrix) const {
    Model* model = const_cast<Model*>(findModel(modelID));
    if (!model) {
        return false;
    }

    outMatrix = model->getModelMatrix();
    return true;
}

bool ModelRegistry::tryGetBoundingBoxCenter(uint32_t modelID, glm::vec3& outCenter) const {
    Model* model = const_cast<Model*>(findModel(modelID));
    if (!model) {
        return false;
    }

    outCenter = model->getBoundingBoxCenter();
    return true;
}

bool ModelRegistry::tryGetBoundingBoxMinMax(uint32_t modelID, glm::vec3& outMin, glm::vec3& outMax) const {
    Model* model = const_cast<Model*>(findModel(modelID));
    if (!model) {
        return false;
    }

    outMin = model->getBoundingBoxMin();
    outMax = model->getBoundingBoxMax();
    return true;
}

bool ModelRegistry::tryGetWorldBoundingBoxCenter(uint32_t modelID, glm::vec3& outCenter) const {
    glm::vec3 localCenter(0.0f);
    glm::mat4 modelMatrix(1.0f);
    if (!tryGetBoundingBoxCenter(modelID, localCenter) || !tryGetModelMatrix(modelID, modelMatrix)) {
        return false;
    }

    outCenter = glm::vec3(modelMatrix * glm::vec4(localCenter, 1.0f));
    return true;
}

Model* ModelRegistry::findModel(uint32_t modelID) {
    const auto it = models.find(modelID);
    if (it == models.end()) {
        return nullptr;
    }
    return it->second.get();
}

const Model* ModelRegistry::findModel(uint32_t modelID) const {
    const auto it = models.find(modelID);
    if (it == models.end()) {
        return nullptr;
    }
    return it->second.get();
}

void ModelRegistry::cleanup() {
    clearModels();
}

bool ModelRegistry::isReservedModelId(uint32_t modelId) {
    return modelId == 0;
}

uint32_t ModelRegistry::acquireModelId(uint32_t preferredModelId) {
    if (preferredModelId != 0 &&
        preferredModelId <= MaxStencilModelId &&
        !isReservedModelId(preferredModelId) &&
        models.find(preferredModelId) == models.end()) {
        return preferredModelId;
    }

    while (!recycledModelIds.empty()) {
        const uint32_t candidate = recycledModelIds.back();
        recycledModelIds.pop_back();
        if (candidate == 0 || candidate > MaxStencilModelId || isReservedModelId(candidate)) {
            continue;
        }
        if (models.find(candidate) == models.end()) {
            return candidate;
        }
    }

    while (nextModelId <= MaxStencilModelId &&
           (isReservedModelId(nextModelId) ||
            models.find(nextModelId) != models.end())) {
        ++nextModelId;
    }

    if (nextModelId > MaxStencilModelId) {
        std::cerr << "[ModelRegistry] Exhausted stencil model IDs" << std::endl;
        return 0;
    }

    return nextModelId++;
}

void ModelRegistry::recycleModelId(uint32_t modelId) {
    if (modelId == 0 || modelId > MaxStencilModelId || isReservedModelId(modelId)) {
        return;
    }

    if (std::find(recycledModelIds.begin(), recycledModelIds.end(), modelId) == recycledModelIds.end()) {
        recycledModelIds.push_back(modelId);
    }
}


void ModelRegistry::unregisterModel(uint32_t modelId) {
    if (modelId != 0) {
        visibleModelIds.erase(modelId);
        recycleModelId(modelId);
        
        auto it = models.find(modelId);
        if (it != models.end() && it->second) {
            it->second->setRuntimeModelId(0);
        }
    }
}

glm::vec3 ModelRegistry::calculateMaxBoundingBoxSize() const {
    glm::vec3 globalMin = glm::vec3(std::numeric_limits<float>::max());
    glm::vec3 globalMax = glm::vec3(std::numeric_limits<float>::lowest());
    bool hasAnyBounds = false;

    for (uint32_t modelId : getRenderableModelIds()) {
        Model* model = const_cast<Model*>(findModel(modelId));
        if (!model || model->getVertexCount() == 0) {
            continue;
        }

        globalMin = glm::min(globalMin, model->getBoundingBoxMin());
        globalMax = glm::max(globalMax, model->getBoundingBoxMax());
        hasAnyBounds = true;
    }

    if (!hasAnyBounds) {
        return glm::vec3(1.0f);
    }

    glm::vec3 size = globalMax - globalMin;

    float greaterLineInterval = 1.0f;
    float snappedWidth = std::ceil(size.x / greaterLineInterval) * greaterLineInterval;
    float snappedDepth = std::ceil(size.z / greaterLineInterval) * greaterLineInterval;
    float snappedHeight = std::ceil(size.y / greaterLineInterval) * greaterLineInterval;

    float minSize = 1.0f;
    snappedWidth = glm::max(snappedWidth, minSize);
    snappedDepth = glm::max(snappedDepth, minSize);
    snappedHeight = glm::max(snappedHeight, minSize);

    float padding = 1.0f;
    if (size.x >= snappedWidth) snappedWidth += padding;
    if (size.z >= snappedDepth) snappedDepth += padding;
    if (size.y >= snappedHeight) snappedHeight += padding;

    return glm::vec3(snappedWidth, snappedDepth, snappedHeight);
}

