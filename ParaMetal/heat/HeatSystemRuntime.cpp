#include "HeatSystemRuntime.hpp"

#include "vulkan/CommandBufferManager.hpp"
#include "heat/HeatModelRuntime.hpp"
#include "heat/HeatSystemPresets.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanDevice.hpp"

#include <iostream>
#include <algorithm>
#include <limits>
#include <unordered_set>

void HeatSystemRuntime::setHeatModels(
    const std::vector<SupportingHalfedge::IntrinsicMesh>& modelIntrinsicMeshes,
    const std::vector<uint32_t>& modelRuntimeModelIds,
    const std::unordered_map<uint32_t, float>& modelTemperatureByRuntimeId,
    const std::unordered_map<uint32_t, uint32_t>& modelBoundaryConditions,
    const std::unordered_map<uint32_t, float>& modelFixedTemperatureValues,
    const std::unordered_map<uint32_t, float>& modelDensity,
    const std::unordered_map<uint32_t, float>& modelSpecificHeat,
    const std::unordered_map<uint32_t, float>& modelConductivity) {
    activeModelIntrinsicMeshes = modelIntrinsicMeshes;
    activeModelRuntimeModelIds = modelRuntimeModelIds;
    activeModelTemperatureByRuntimeId = modelTemperatureByRuntimeId;
    activeModelBoundaryConditions = modelBoundaryConditions;
    activeModelFixedTemperatureValues = modelFixedTemperatureValues;
    activeModelDensity = modelDensity;
    activeModelSpecificHeat = modelSpecificHeat;
    activeModelConductivity = modelConductivity;
    modelsDirty = true;
}


HeatModelRuntime* HeatSystemRuntime::getModelByRuntimeId(uint32_t runtimeModelId) const {
    auto it = activeModels.find(runtimeModelId);
    return (it != activeModels.end()) ? it->second.get() : nullptr;
}

float HeatSystemRuntime::getTemperature(uint32_t modelId) const {
    auto it = activeModelTemperatureByRuntimeId.find(modelId);
    return (it != activeModelTemperatureByRuntimeId.end()) ? it->second : HeatSimDefaults::ambientTemperature;
}

void HeatSystemRuntime::configureModelProperties(HeatModelRuntime* model, uint32_t modelId) const {
    if (!model) return;

    const auto bcIt = activeModelBoundaryConditions.find(modelId);
    const uint32_t boundaryCondition = (bcIt != activeModelBoundaryConditions.end()) ? bcIt->second : 0u;

    const auto fixedTempIt = activeModelFixedTemperatureValues.find(modelId);
    const float fixedTemperatureValue = (fixedTempIt != activeModelFixedTemperatureValues.end()) ? fixedTempIt->second : HeatSimDefaults::ambientTemperature;

    const auto densityIt = activeModelDensity.find(modelId);
    const float density = (densityIt != activeModelDensity.end()) ? densityIt->second : HeatSimDefaults::density;

    const auto specificHeatIt = activeModelSpecificHeat.find(modelId);
    const float specificHeat = (specificHeatIt != activeModelSpecificHeat.end()) ? specificHeatIt->second : HeatSimDefaults::specificHeat;

    const auto conductivityIt = activeModelConductivity.find(modelId);
    const float conductivity = (conductivityIt != activeModelConductivity.end()) ? conductivityIt->second : HeatSimDefaults::conductivity;

    model->setBoundaryCondition(boundaryCondition);
    model->setFixedTemperatureValue(fixedTemperatureValue);
    model->setMaterialProperties(density, specificHeat, conductivity);
}

bool HeatSystemRuntime::ensureModelBindings(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& renderCommandPool) {
    if (!modelsDirty) {
        return true;
    }

    // Determine incoming model IDs
    std::unordered_set<uint32_t> incomingIds;
    const std::size_t pairCount = std::min(
        activeModelRuntimeModelIds.size(),
        activeModelIntrinsicMeshes.size()
    );
    for (std::size_t i = 0; i < pairCount; ++i) {
        uint32_t id = activeModelRuntimeModelIds[i];
        if (id != 0) incomingIds.insert(id);
    }

    // Remove models no longer in config
    for (auto it = activeModels.begin(); it != activeModels.end(); ) {
        if (incomingIds.find(it->first) == incomingIds.end()) {
            if (it->second) it->second->cleanup();
            it = activeModels.erase(it);
        } else {
            ++it;
        }
    }

    // Create, recreate or update each incoming model
    std::unordered_set<uint32_t> seenModelIds;
    for (std::size_t index = 0; index < pairCount; ++index) {
        const uint32_t modelId = activeModelRuntimeModelIds[index];
        if (modelId == 0 || !seenModelIds.insert(modelId).second) {
            continue;
        }

        const auto& incomingMesh = activeModelIntrinsicMeshes[index];
        size_t incomingVertexCount = incomingMesh.vertices.size();

        auto it = activeModels.find(modelId);
        if (it == activeModels.end()) {
            // NEW MODEL
            auto heatModel = std::make_unique<HeatModelRuntime>(
                vulkanDevice, memoryAllocator, incomingMesh, renderCommandPool,
                getTemperature(modelId));
            configureModelProperties(heatModel.get(), modelId);
            if (heatModel && heatModel->isInitialized()) {
                activeModels.emplace(modelId, std::move(heatModel));
            } else {
                std::cerr << "[HeatSystemRuntime] Failed to initialize heat model for model " << modelId << std::endl;
            }
        } else if (it->second->getIntrinsicVertexCount() != incomingVertexCount) {
            // GEOMETRY CHANGED 
            it->second->cleanup();
            auto heatModel = std::make_unique<HeatModelRuntime>(
                vulkanDevice, memoryAllocator, incomingMesh, renderCommandPool,
                getTemperature(modelId));
            configureModelProperties(heatModel.get(), modelId);
            if (heatModel && heatModel->isInitialized()) {
                it->second = std::move(heatModel);
            } else {
                std::cerr << "[HeatSystemRuntime] Failed to recreate heat model for model " << modelId << std::endl;
                activeModels.erase(it);
            }
        } else {
            // EXISTING MODEL AND SAME GEOMETRY UPDATE PARAMS ONLY
            configureModelProperties(it->second.get(), modelId);
        }
    }

    modelsDirty = false;
    return true;
}

void HeatSystemRuntime::cleanup() {
    for (auto& [modelId, heatModel] : activeModels) {
        if (heatModel) {
            heatModel->cleanup();
        }
    }
    activeModels.clear();
    modelsDirty = true;
}
