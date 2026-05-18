#pragma once

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
        float fluxVectorScale = 1.0f;
        bool authoredActive = false;
        bool active = false;
        bool paused = false;
        std::vector<ModelProduct> models;
        std::vector<float> modelTemperatures;
        std::vector<float> modelFixedTemperatures;
        std::vector<HeatBoundaryCondition> modelBoundaryConditions;
        std::vector<VkBuffer> modelSurfaceBuffers;
        std::vector<VkDeviceSize> modelSurfaceBufferOffsets;
        std::vector<uint32_t> modelSurfacePointCounts;
        std::vector<std::array<VkBufferView, 11>> modelBufferViews;
        std::vector<VkBuffer> modelSurfaceGradientBuffers;
        std::vector<VkDeviceSize> modelSurfaceGradientBufferOffsets;
        uint64_t displayHash = 0;

        bool anyVisible() const {
            return showHeatOverlay || showFluxVectors;
        }

        bool isValid() const {
            return !models.empty() &&
                models.size() == modelTemperatures.size() &&
                models.size() == modelFixedTemperatures.size() &&
                models.size() == modelBoundaryConditions.size() &&
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

inline uint64_t buildDisplayHash(const HeatDisplayController::Config& config, uint64_t productHash) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combinePod(hash, static_cast<uint64_t>(config.showHeatOverlay ? 1u : 0u));
    NodeGraphHash::combinePod(hash, static_cast<uint64_t>(config.showFluxVectors ? 1u : 0u));
    NodeGraphHash::combinePod(hash, config.fluxVectorScale);
    NodeGraphHash::combinePod(hash, static_cast<uint64_t>(config.authoredActive ? 1u : 0u));
    NodeGraphHash::combinePod(hash, static_cast<uint64_t>(config.active ? 1u : 0u));
    NodeGraphHash::combinePod(hash, static_cast<uint64_t>(config.paused ? 1u : 0u));
    NodeGraphHash::combine(hash, productHash);
    return hash;
}