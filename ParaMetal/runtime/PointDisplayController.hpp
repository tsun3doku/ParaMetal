#pragma once

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

namespace render {
class PointOverlayRenderer;
}

class PointDisplayController {
public:
    struct Config {
        VkBuffer vertexBuffer = VK_NULL_HANDLE;
        VkDeviceSize vertexBufferOffset = 0;
        uint32_t pointCount = 0;
        glm::mat4 modelMatrix = glm::mat4(1.0f);
        uint64_t displayHash = 0;

        bool isValid() const {
            return vertexBuffer != VK_NULL_HANDLE && pointCount != 0;
        }
    };

    void setOverlayRenderer(render::PointOverlayRenderer* updatedOverlayRenderer);
    void apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);
    void finalizeSync();

private:
    render::PointOverlayRenderer* overlayRenderer = nullptr;
    std::unordered_map<uint64_t, Config> configsBySocket;
    std::unordered_set<uint64_t> syncedSockets;
};
