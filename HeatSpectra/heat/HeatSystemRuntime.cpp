#include "HeatSystemRuntime.hpp"

#include "vulkan/CommandBufferManager.hpp"
#include "heat/HeatModelRuntime.hpp"
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

bool HeatSystemRuntime::ensureModelBindings(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    CommandPool& renderCommandPool) {
    if (!modelsDirty) {
        return true;
    }

    cleanup();

    const std::size_t pairCount = std::min(
        activeModelRuntimeModelIds.size(),
        activeModelIntrinsicMeshes.size()
    );

    std::unordered_set<uint32_t> seenModelIds;
    for (std::size_t index = 0; index < pairCount; ++index) {
        const uint32_t modelId = activeModelRuntimeModelIds[index];
        if (modelId == 0 || !seenModelIds.insert(modelId).second) {
            continue;
        }


        const auto tempIt = activeModelTemperatureByRuntimeId.find(modelId);
        const float initialTemperature =
            (tempIt != activeModelTemperatureByRuntimeId.end()) ? tempIt->second : 1.0f;

        const auto bcIt = activeModelBoundaryConditions.find(modelId);
        const uint32_t boundaryCondition =
            (bcIt != activeModelBoundaryConditions.end()) ? bcIt->second : 0u;

        const auto fixedTempIt = activeModelFixedTemperatureValues.find(modelId);
        const float fixedTemperatureValue =
            (fixedTempIt != activeModelFixedTemperatureValues.end()) ? fixedTempIt->second : 1.0f;

        const auto densityIt = activeModelDensity.find(modelId);
        const float density = (densityIt != activeModelDensity.end()) ? densityIt->second : 1000.0f;

        const auto specificHeatIt = activeModelSpecificHeat.find(modelId);
        const float specificHeat = (specificHeatIt != activeModelSpecificHeat.end()) ? specificHeatIt->second : 1000.0f;

        const auto conductivityIt = activeModelConductivity.find(modelId);
        const float conductivity = (conductivityIt != activeModelConductivity.end()) ? conductivityIt->second : 1.0f;

        auto heatModel = std::make_unique<HeatModelRuntime>(
            vulkanDevice,
            memoryAllocator,
            activeModelIntrinsicMeshes[index],
            renderCommandPool,
            initialTemperature);

        if (heatModel) {
            heatModel->setBoundaryCondition(boundaryCondition);
            heatModel->setFixedTemperatureValue(fixedTemperatureValue);
            heatModel->setMaterialProperties(density, specificHeat, conductivity);
        }

        if (!heatModel || !heatModel->isInitialized()) {
            std::cerr << "[HeatSystemRuntime] Failed to initialize heat model for model " << modelId << std::endl;
            if (heatModel) {
                heatModel->cleanup();
            }
            continue;
        }

        activeModels.emplace(modelId, std::move(heatModel));
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
