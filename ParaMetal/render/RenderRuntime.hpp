#pragma once

#include <vulkan/vulkan.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "framegraph/FrameStats.hpp"
#include "framegraph/FramePass.hpp"
#include "WindowRuntimeState.hpp"

class VulkanDevice;
class ViewportTarget;
class CommandPool;
class FrameSync;
class MemoryAllocator;
class ModelRegistry;
class UniformBufferManager;
class IBLSystem;
class LightingSystem;
class MaterialSystem;
class FrameGraph;
class SceneRenderer;
class ModelSelection;
class WireframeRenderer;
class FrameController;
class VkFrameGraphBackend;
class VkFrameGraphRuntime;

class SceneContext;
class InteractionContext;

class RenderRuntime {
public:
    RenderRuntime(
        const WindowRuntimeState& windowState,
        VulkanDevice& vulkanDevice,
        ViewportTarget& viewportTarget,
        CommandPool& renderCommandPool,
        FrameSync& frameSync,
        std::atomic<bool>& isOperating,
        std::atomic<bool>& isShuttingDown);
    ~RenderRuntime();

    bool initializeBase(VkFormat swapChainFormat, VkExtent2D extent, MemoryAllocator& allocator, ModelRegistry& modelRegistry, UniformBufferManager& ubo, IBLSystem& iblSystem);
    bool initializeFrameController(SceneContext& scene, InteractionContext& interaction);

    bool renderFrame(
        VkCommandBuffer commandBuffer,
        uint32_t frameIndex,
        const render::RenderFlags& flags);
    bool updateViewportTarget(VkImage image, VkFormat format, VkExtent2D extent);
    void cleanup();

    VulkanDevice& getVulkanDevice();
    const VulkanDevice& getVulkanDevice() const;
    FrameStats& getFrameStats();
    const FrameStats& getFrameStats() const;
    FrameGraph& getFrameGraph();
    const FrameGraph& getFrameGraph() const;
    VkFrameGraphRuntime& getFrameGraphRuntime();
    const VkFrameGraphRuntime& getFrameGraphRuntime() const;
    SceneRenderer& getSceneRenderer();
    const SceneRenderer& getSceneRenderer() const;
    ModelSelection& getModelSelection();
    const ModelSelection& getModelSelection() const;
    WireframeRenderer& getWireframeRenderer();
    const WireframeRenderer& getWireframeRenderer() const;

private:
    const WindowRuntimeState& windowState;
    VulkanDevice& vulkanDevice;
    ViewportTarget& viewportTarget;
    CommandPool& renderCommandPool;
    FrameSync& frameSync;
    std::atomic<bool>& isOperating;
    std::atomic<bool>& isShuttingDown;

    FrameStats frameStats;

    std::unique_ptr<FrameGraph> frameGraph;
    std::unique_ptr<VkFrameGraphBackend> frameGraphBackend;
    std::unique_ptr<SceneRenderer> sceneRenderer;
    std::unique_ptr<ModelSelection> modelSelection;
    std::unique_ptr<WireframeRenderer> wireframeRenderer;
    std::unique_ptr<FrameController> frameController;
};
