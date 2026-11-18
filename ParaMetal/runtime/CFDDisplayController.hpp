#pragma once

#include "hash/HashBuilder.hpp"
#include "cfd/CFDSliceGeometry.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <glm/vec3.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vulkan/vulkan.h>

namespace render {
class CFDOverlayRenderer;
}

class CFDDisplayController {
public:
    struct Config {
        bool showStreamlines = false;
        bool showDomainBox = false;
        bool showVelocityPalette = false;
        bool showSlice = false;
        int sliceAxis = 0;
        float slicePosition = 0.5f;
        bool showSliceWireframe = false;
        std::shared_ptr<const cfd::CFDSliceGeometrySource> sliceGeometry;
        bool authoredActive = false;
        bool active = false;
        float canonicalToWorldScale = 1.0f;

        std::array<glm::vec3, 8> globalDomainCorners{};
        bool hasGlobalDomain = false;

        VkBuffer streamlineVertexBuffer = VK_NULL_HANDLE;
        VkDeviceSize streamlineVertexBufferOffset = 0;
        uint32_t streamlineVertexCount = 0;

        uint32_t fluidNodeCount = 0;
        VkBuffer velocityBuffer = VK_NULL_HANDLE;
        VkDeviceSize velocityBufferOffset = 0;
        VkBuffer gradientBuffer = VK_NULL_HANDLE;
        VkDeviceSize gradientBufferOffset = 0;
        VkBuffer centroidBuffer = VK_NULL_HANDLE;
        VkDeviceSize centroidBufferOffset = 0;

        uint64_t displayHash = 0;

        bool anyVisible() const {
            return showStreamlines || showDomainBox || showVelocityPalette || showSlice;
        }

        bool isValid() const {
            return hasGlobalDomain || streamlineVertexBuffer != VK_NULL_HANDLE;
        }
    };

    void setOverlayRenderer(render::CFDOverlayRenderer* updatedOverlayRenderer);
    void apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);
    void finalizeSync();

private:
    render::CFDOverlayRenderer* overlayRenderer = nullptr;
    std::unordered_map<uint64_t, Config> configsBySocket;
    std::unordered_set<uint64_t> syncedSockets;
};

inline uint64_t buildCFDDisplayHash(const CFDDisplayController::Config& config, uint64_t productDisplayHash) {
    uint64_t hash = HashBuilder::start();
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.showStreamlines ? 1u : 0u));
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.showDomainBox ? 1u : 0u));
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.showVelocityPalette ? 1u : 0u));
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.showSlice ? 1u : 0u));
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.sliceAxis));
    HashBuilder::combinePod(hash, config.slicePosition);
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.showSliceWireframe ? 1u : 0u));
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.authoredActive ? 1u : 0u));
    HashBuilder::combinePod(hash, static_cast<uint64_t>(config.active ? 1u : 0u));
    HashBuilder::combinePod(hash, config.canonicalToWorldScale);
    HashBuilder::combine(hash, productDisplayHash);
    return hash;
}
