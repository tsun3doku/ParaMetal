#pragma once

#include "hash/HashBuilder.hpp"
#include "runtime/RuntimePackages.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "domain/HeatModelData.hpp"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

namespace render {
class HeatOverlayRenderer;
}

class HeatDisplayController {
public:
    struct Config {
        bool showHeatOverlay = false;
        bool showFluxVectors = false;
        bool showHeatPalette = false;
        float fluxVectorScale = 1.0f;
        bool authoredActive = false;
        bool active = false;
        std::vector<ModelProduct> models;
        std::vector<float> modelInitialTemperaturesC;
        std::vector<float> modelBoundaryTemperaturesC;
        std::vector<uint32_t> modelBoundaryConditionTypes;
        std::vector<VkBuffer> modelSurfaceBuffers;
        std::vector<VkDeviceSize> modelSurfaceBufferOffsets;
        std::vector<uint32_t> modelSurfacePointCounts;
        std::vector<std::array<VkBufferView, 11>> modelBufferViews;
        std::vector<VkBuffer> modelSurfaceGradientBuffers;
        std::vector<VkDeviceSize> modelSurfaceGradientBufferOffsets;
        uint64_t displayHash = 0;

        bool anyVisible() const {
            return showHeatOverlay || showFluxVectors || showHeatPalette;
        }

        bool isValid() const {
            return !models.empty() &&
                models.size() == modelInitialTemperaturesC.size() &&
                models.size() == modelBoundaryTemperaturesC.size() &&
                models.size() == modelBoundaryConditionTypes.size() &&
                models.size() == modelSurfaceBuffers.size() &&
                models.size() == modelSurfaceBufferOffsets.size() &&
                models.size() == modelSurfacePointCounts.size() &&
                models.size() == modelBufferViews.size() &&
                models.size() == modelSurfaceGradientBuffers.size() &&
                models.size() == modelSurfaceGradientBufferOffsets.size();
        }

    };

    void setOverlayRenderer(render::HeatOverlayRenderer* updatedOverlayRenderer);
    void apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);
    void finalizeSync();

private:
    render::HeatOverlayRenderer* overlayRenderer = nullptr;
    std::unordered_map<uint64_t, Config> configsBySocket;
    std::unordered_set<uint64_t> syncedSockets;
};

inline uint64_t buildDisplayHash(const HeatDisplayController::Config& config, uint64_t productDisplayHash) {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.showHeatOverlay ? 1u : 0u));
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.showFluxVectors ? 1u : 0u));
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.showHeatPalette ? 1u : 0u));
    HashBuilder::combinePod(hash, config.fluxVectorScale);
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.authoredActive ? 1u : 0u));
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.active ? 1u : 0u));
    HashBuilder::combine(hash, productDisplayHash);
    return hash;
}
