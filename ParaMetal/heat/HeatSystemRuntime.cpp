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
    const std::vector<std::vector<glm::vec3>>& modelSurfacePositions,
    const std::vector<std::vector<glm::vec3>>& modelSurfaceNormals,
    const std::vector<std::vector<uint32_t>>& modelSurfaceTriangleIndices,
    const std::vector<uint32_t>& modelRuntimeModelIds,
    const std::unordered_map<uint32_t, float>& modelInitialTemperaturesCByRuntimeId,
    const std::unordered_map<uint32_t, uint32_t>& modelBoundaryConditionTypesByRuntimeId,
    const std::unordered_map<uint32_t, float>& modelBoundaryTemperaturesCByRuntimeId,
    const std::unordered_map<uint32_t, float>& modelBoundaryHeatFluxesByRuntimeId,
    const std::unordered_map<uint32_t, float>& modelBoundaryHeatTransferCoefficientsByRuntimeId,
    const std::unordered_map<uint32_t, float>& modelVolumetricPowerDensitiesByRuntimeId,
    const std::unordered_map<uint32_t, float>& modelDensity,
    const std::unordered_map<uint32_t, float>& modelSpecificHeat,
    const std::unordered_map<uint32_t, float>& modelConductivity) {
    activeModelSurfacePositions = modelSurfacePositions;
    activeModelSurfaceNormals = modelSurfaceNormals;
    activeModelSurfaceTriangleIndices = modelSurfaceTriangleIndices;
    activeModelRuntimeModelIds = modelRuntimeModelIds;
    activeModelInitialTemperaturesCByRuntimeId = modelInitialTemperaturesCByRuntimeId;
    activeModelBoundaryConditionTypesByRuntimeId = modelBoundaryConditionTypesByRuntimeId;
    activeModelBoundaryTemperaturesCByRuntimeId = modelBoundaryTemperaturesCByRuntimeId;
    activeModelBoundaryHeatFluxesByRuntimeId = modelBoundaryHeatFluxesByRuntimeId;
    activeModelBoundaryHeatTransferCoefficientsByRuntimeId = modelBoundaryHeatTransferCoefficientsByRuntimeId;
    activeModelVolumetricPowerDensitiesByRuntimeId = modelVolumetricPowerDensitiesByRuntimeId;
    activeModelDensity = modelDensity;
    activeModelSpecificHeat = modelSpecificHeat;
    activeModelConductivity = modelConductivity;
    modelsDirty = true;
}


HeatModelRuntime* HeatSystemRuntime::getModelByRuntimeId(uint32_t runtimeModelId) const {
    auto it = activeModels.find(runtimeModelId);
    return (it != activeModels.end()) ? it->second.get() : nullptr;
}

float HeatSystemRuntime::getInitialTemperatureC(uint32_t modelId) const {
    auto it = activeModelInitialTemperaturesCByRuntimeId.find(modelId);
    return (it != activeModelInitialTemperaturesCByRuntimeId.end()) ? it->second : HeatSimDefaults::ambientTemperatureC;
}

void HeatSystemRuntime::configureModelProperties(HeatModelRuntime* model, uint32_t modelId) const {
    if (!model) return;

    const auto densityIt = activeModelDensity.find(modelId);
    const float density = (densityIt != activeModelDensity.end()) ? densityIt->second : HeatSimDefaults::density;

    const auto specificHeatIt = activeModelSpecificHeat.find(modelId);
    const float specificHeat = (specificHeatIt != activeModelSpecificHeat.end()) ? specificHeatIt->second : HeatSimDefaults::specificHeat;

    const auto conductivityIt = activeModelConductivity.find(modelId);
    const float conductivity = (conductivityIt != activeModelConductivity.end()) ? conductivityIt->second : HeatSimDefaults::conductivity;

    const auto initialTemperatureIt = activeModelInitialTemperaturesCByRuntimeId.find(modelId);
    const float initialTemperatureC = initialTemperatureIt != activeModelInitialTemperaturesCByRuntimeId.end()
        ? initialTemperatureIt->second : HeatSimDefaults::ambientTemperatureC;
    const auto boundaryTypeIt = activeModelBoundaryConditionTypesByRuntimeId.find(modelId);
    const uint32_t boundaryConditionType = boundaryTypeIt != activeModelBoundaryConditionTypesByRuntimeId.end()
        ? boundaryTypeIt->second : 0u;
    const auto boundaryTemperatureIt = activeModelBoundaryTemperaturesCByRuntimeId.find(modelId);
    const float boundaryTemperatureC = boundaryTemperatureIt != activeModelBoundaryTemperaturesCByRuntimeId.end()
        ? boundaryTemperatureIt->second : HeatSimDefaults::ambientTemperatureC;
    const auto heatFluxIt = activeModelBoundaryHeatFluxesByRuntimeId.find(modelId);
    const float heatFlux = heatFluxIt != activeModelBoundaryHeatFluxesByRuntimeId.end() ? heatFluxIt->second : 0.0f;
    const auto htcIt = activeModelBoundaryHeatTransferCoefficientsByRuntimeId.find(modelId);
    const float heatTransferCoefficient = htcIt != activeModelBoundaryHeatTransferCoefficientsByRuntimeId.end() ? htcIt->second : 0.0f;
    const auto powerDensityIt = activeModelVolumetricPowerDensitiesByRuntimeId.find(modelId);
    const float volumetricPowerDensity = powerDensityIt != activeModelVolumetricPowerDensitiesByRuntimeId.end() ? powerDensityIt->second : 0.0f;

    model->setMaterialProperties(density, specificHeat, conductivity);
    model->setInitialTemperatureC(initialTemperatureC);
    model->setBoundaryInputs(boundaryConditionType, boundaryTemperatureC, heatFlux,
        heatTransferCoefficient, volumetricPowerDensity);
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
    const std::size_t pairCount = std::min({
        activeModelRuntimeModelIds.size(),
        activeModelSurfacePositions.size(),
        activeModelSurfaceNormals.size(),
        activeModelSurfaceTriangleIndices.size()});
    for (std::size_t i = 0; i < pairCount; ++i) {
        uint32_t id = activeModelRuntimeModelIds[i];
        if (id != 0) incomingIds.insert(id);
    }

    // Remove models no longer in config
    for (auto it = activeModels.begin(); it != activeModels.end(); ) {
        if (incomingIds.find(it->first) == incomingIds.end()) {
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

        const auto& incomingPositions = activeModelSurfacePositions[index];
        const auto& incomingNormals = activeModelSurfaceNormals[index];
        const auto& incomingTriangleIndices = activeModelSurfaceTriangleIndices[index];
        auto it = activeModels.find(modelId);
        const bool geometryChanged = it != activeModels.end() &&
            (it->second->getSurfacePositions() != incomingPositions ||
             it->second->getSurfaceNormals() != incomingNormals ||
             it->second->getSurfaceTriangleIndices() != incomingTriangleIndices);

        if (it == activeModels.end() || geometryChanged) {
            auto heatModel = std::make_unique<HeatModelRuntime>(
                vulkanDevice, memoryAllocator, incomingPositions, incomingNormals,
                incomingTriangleIndices, renderCommandPool,
                getInitialTemperatureC(modelId));
            configureModelProperties(heatModel.get(), modelId);
            if (heatModel->isInitialized()) {
                activeModels[modelId] = std::move(heatModel);
            } else {
                std::cerr << "[HeatSystemRuntime] Failed to initialize heat model for model " << modelId << std::endl;
                activeModels.erase(modelId);
            }
        } else {
            configureModelProperties(it->second.get(), modelId);
        }
    }

    modelsDirty = false;
    return true;
}

void HeatSystemRuntime::cleanup() {
    activeModels.clear();
    modelsDirty = true;
}
