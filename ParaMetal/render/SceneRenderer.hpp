#pragma once

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include "framegraph/FramePass.hpp"

#include <memory>
#include <string>
#include <vector>

#include "nodegraph/NodeGraphCoreTypes.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "util/Units.hpp"

class UniformBufferManager;
class MemoryAllocator;
class ModelRegistry;
class FrameGraph;
class VulkanDevice;
class CommandPool;
class VkFrameGraphRuntime;
class IBLSystem;
class IntrinsicRenderer;
class SubselectionRenderer;
class GizmoRenderer;
class NavigationGizmoRenderer;
class ScreenTextRenderer;
class ModelSelection;
class GizmoController;
class NavigationGizmoController;
class SubselectionPickService;
class WireframeRenderer;

namespace render { class CFDOverlayRenderer; }
namespace render { class HeatOverlayRenderer; }
namespace render { class PointOverlayRenderer; }
namespace render { class VoronoiOverlayRenderer; }
namespace render { class SimVolumeRenderer; }

namespace render {
class GeometryPass;
class SurfacePass;
class LightingPass;
class VolumePass;
class OverlayPass;
class PickPass;
class BlendPass;

struct RenderFrameRequest {
    uint32_t frameIndex = 0;
    uint32_t imageIndex = 0;
    VkExtent2D extent{};
    SceneView sceneView{};
    RenderFlags flags{};

};
}

struct GpuPassTiming {
    std::string name;
    float ms = 0.0f;
};

struct GpuTimingStats {
    float totalMs = 0.0f;
    std::vector<GpuPassTiming> passTimings;
};

class SceneRenderer {
public:
    SceneRenderer(VulkanDevice& device, MemoryAllocator& allocator, FrameGraph& graph, VkFrameGraphRuntime& frameGraphRuntime, ModelRegistry& manager, UniformBufferManager& ubo, uint32_t framesInFlight, CommandPool& commandPool,
                  IBLSystem& iblSystem);
    ~SceneRenderer();

    void resize(VkExtent2D extent);
    void updateDescriptorSets();
    IntrinsicRenderer* getIntrinsicRenderer() const;
    SubselectionRenderer* getSubselectionRenderer() const;
    render::HeatOverlayRenderer* getHeatOverlayRenderer() const;
    render::CFDOverlayRenderer* getCFDOverlayRenderer() const;
    render::PointOverlayRenderer* getPointOverlayRenderer() const;
    render::VoronoiOverlayRenderer* getVoronoiOverlayRenderer() const;
    render::SimVolumeRenderer* getHeatVolumeRenderer() const;
    void setHeatPaletteRange(float minimum, float maximum);
    void setHeatPalette(int palette);
    void setVelocityPaletteRange(float minimum, float maximum);
    void setVelocityPalette(int palette);
    void setStreamlineCull(float worldDistance);
    bool isHeatPaletteVisible() const;
    bool isVelocityPaletteVisible() const;

    void setTimingOverlayLines(const std::vector<std::string>& lines);
    void updateGrid(uint32_t frameIndex, const render::SceneView& sceneView, const glm::vec3& sceneExtent);
    void setWorldUnit(units::LengthUnit unit);
    void setRenderServices(
        ModelSelection& modelSelection,
        GizmoController& gizmoController,
        NavigationGizmoController& navigationGizmoController,
        WireframeRenderer& wireframeRenderer);
    void setDisplayedModelIds(const std::vector<uint32_t>& modelIds);
    void setSubselectionPickState(const SubselectionPickState& state);
    void setSubselectionPickService(SubselectionPickService* service);
    void prepareFrame(uint32_t frameIndex, VkExtent2D extent);

    bool recordExternalCommandBuffer(
        VkCommandBuffer commandBuffer,
        const render::RenderFrameRequest& frameRequest,
        bool insertComputeToGraphicsBarrier,
        VkPipelineStageFlags computeToGraphicsDstStageMask);

    void cleanup();
    bool getGpuTimingStats(uint32_t frameIndex, GpuTimingStats& outStats) const;

    VkDescriptorSetLayout getGbufferDescriptorSetLayout() const;

    bool isReady() const {
        return ready;
    }

private:
    bool initializePasses();
    void addPass(std::unique_ptr<render::Pass> pass);
    void rebuildOrderedPasses();
    bool createPasses();
    void resizePasses(VkExtent2D extent);
    void updatePassDescriptors();
    void recordPasses(
        const render::FrameContext& frameContext,
        const render::SceneView& sceneView,
        const render::RenderFlags& flags,
        VkQueryPool passTimingQueryPool = VK_NULL_HANDLE,
        uint32_t passTimingQueryBase = 0);
    void destroyPasses();
    void createGpuTimingQueryPool();
    void destroyGpuTimingQueryPool();
    void recordComputeToGraphicsBarrier(VkCommandBuffer commandBuffer, VkPipelineStageFlags dstStageMask);
    bool recordCommands(
        VkCommandBuffer commandBuffer,
        const render::RenderFrameRequest& frameRequest,
        bool insertComputeToGraphicsBarrier,
        VkPipelineStageFlags computeToGraphicsDstStageMask);
    
    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    FrameGraph& frameGraph;
    VkFrameGraphRuntime& frameGraphRuntime;
    ModelRegistry& modelRegistry;
    UniformBufferManager& uniformBufferManager;
    CommandPool& renderCommandPool;
    IBLSystem& iblSystem;
    render::RenderServices renderServices;
    std::vector<uint32_t> displayedModelIds;
    SubselectionPickState subselectionPickState;
    SubselectionPickService* subselectionPickService = nullptr;

    render::GeometryPass* geometryPass = nullptr;
    render::SurfacePass* surfacePass = nullptr;
    render::LightingPass* lightingPass = nullptr;
    render::VolumePass* volumePass = nullptr;
    render::OverlayPass* overlayPass = nullptr;
    render::PickPass* pickPass = nullptr;
    render::BlendPass* blendPass = nullptr;
    std::unique_ptr<render::HeatOverlayRenderer> heatOverlayRenderer;
    std::unique_ptr<render::CFDOverlayRenderer> cfdOverlayRenderer;
    std::unique_ptr<render::VoronoiOverlayRenderer> voronoiOverlayRenderer;
    std::unique_ptr<IntrinsicRenderer> intrinsicRenderer;
    std::unique_ptr<GizmoRenderer> gizmoRenderer;
    std::unique_ptr<NavigationGizmoRenderer> navigationGizmoRenderer;
    std::unique_ptr<ScreenTextRenderer> screenTextRenderer;

    uint32_t maxFramesInFlight = 0;
    VkQueryPool gpuTimingQueryPool = VK_NULL_HANDLE;
    float gpuTimestampPeriod = 0.0f;
    uint32_t gpuTimingQueriesPerFrame = 0;
    std::vector<std::string> gpuTimingPassNames;
    std::vector<uint8_t> gpuTimingValidFrames;
    std::vector<std::unique_ptr<render::Pass>> passes;
    std::vector<render::Pass*> orderedPasses;

    bool ready = false;
};
