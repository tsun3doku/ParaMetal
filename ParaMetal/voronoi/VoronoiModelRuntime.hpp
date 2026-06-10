#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>
#include <memory>

#include "mesh/remesher/SupportingHalfedge.hpp"
#include "voronoi/VoronoiDomainRuntime.hpp"
#include "voronoi/VoronoiGpuStructs.hpp"

struct StencilKDTree;

class VulkanDevice;
class MemoryAllocator;
class CommandPool;

class VoronoiModelRuntime : public VoronoiDomainRuntime {
public:
    struct SurfaceVertex {
        glm::vec4 position{0.0f, 0.0f, 0.0f, 1.0f};
        glm::vec4 normal{0.0f, 0.0f, 1.0f, 0.0f};
    };

    struct CpuData {
        uint32_t nodeModelId = 0;
        SupportingHalfedge::IntrinsicMesh intrinsicMesh;
        std::vector<glm::vec3> geometryPositions;
        std::vector<uint32_t> geometryTriangleIndices;
        std::vector<SurfaceVertex> surfaceVertices;
        std::vector<uint32_t> intrinsicTriangleIndices;
    };

    VoronoiModelRuntime(
        VulkanDevice& vulkanDevice,
        MemoryAllocator& memoryAllocator,
        uint32_t runtimeModelId,
        const glm::mat4& modelMatrix,
        CpuData cpuData,
        CommandPool& renderCommandPool);
    ~VoronoiModelRuntime();

    bool isPointDomain() const override { return false; }
    bool createVoronoiBuffers() override;
    bool createSurfaceBuffers();
    bool resetSurfaceState();

    void stageGMLSSurfaceData(
        const std::vector<voronoi::GMLSSurfaceStencil>& stencils,
        const std::vector<voronoi::GMLSSurfaceWeight>& valueWeights,
        const std::vector<voronoi::GMLSSurfaceGradientWeight>& gradientWeights);
    void cleanup() override;

    void setStencilKDTree(std::unique_ptr<StencilKDTree> kdTree);
    StencilKDTree* getStencilKDTree() const { return stencilKDTree.get(); }

    uint32_t getNodeModelId() const override { return nodeModelId; }
    uint32_t getRuntimeModelId() const override { return runtimeModelId; }
    const glm::mat4& getModelMatrix() const { return modelMatrix; }

    size_t getIntrinsicVertexCount() const { return intrinsicVertexCount; }
    size_t getIntrinsicTriangleCount() const { return intrinsicTriangleCount; }
    const SupportingHalfedge::IntrinsicMesh& getIntrinsicMesh() const { return intrinsicMesh; }
    const std::vector<glm::vec3>& getGeometryPositions() const { return geometryPositions; }
    const std::vector<uint32_t>& getGeometryTriangleIndices() const { return geometryTriangleIndices; }
    const std::vector<SurfaceVertex>& getSurfaceVertices() const { return surfaceVertices; }
    std::vector<glm::vec3> getIntrinsicSurfacePositions() const;
    const std::vector<uint32_t>& getIntrinsicTriangleIndices() const { return intrinsicTriangleIndices; }
    VkBuffer getTriangleIndicesBuffer() const override { return triangleIndicesBuffer; }
    VkDeviceSize getTriangleIndicesBufferOffset() const override { return triangleIndicesBufferOffset; }
    VkBuffer getCandidateBuffer() const override { return voronoiCandidateBuffer; }
    VkDeviceSize getCandidateBufferOffset() const override { return voronoiCandidateBufferOffset; }
    VkBuffer getSurfaceBuffer() const override { return surfaceBuffer; }
    VkDeviceSize getSurfaceBufferOffset() const override { return surfaceBufferOffset; }
    VkBuffer getGMLSSurfaceStencilBuffer() const override { return gmlsSurfaceStencilBuffer; }
    VkDeviceSize getGMLSSurfaceStencilBufferOffset() const override { return gmlsSurfaceStencilBufferOffset; }
    VkBuffer getGMLSSurfaceWeightBuffer() const override { return gmlsSurfaceWeightBuffer; }
    VkDeviceSize getGMLSSurfaceWeightBufferOffset() const override { return gmlsSurfaceWeightBufferOffset; }
    size_t getGMLSSurfaceWeightCount() const override { return valueWeightCount; }
    VkBuffer getGMLSSurfaceGradientWeightBuffer() const override { return gmlsSurfaceGradientWeightBuffer; }
    VkDeviceSize getGMLSSurfaceGradientWeightBufferOffset() const override { return gmlsSurfaceGradientWeightBufferOffset; }
    size_t getGMLSSurfaceGradientWeightCount() const override { return gradientWeightCount; }

private:
    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    uint32_t nodeModelId = 0;
    uint32_t runtimeModelId = 0;
    glm::mat4 modelMatrix{1.0f};
    SupportingHalfedge::IntrinsicMesh intrinsicMesh;
    std::vector<glm::vec3> geometryPositions;
    std::vector<uint32_t> geometryTriangleIndices;
    std::vector<SurfaceVertex> surfaceVertices;
    CommandPool& renderCommandPool;

    size_t intrinsicVertexCount = 0;
    size_t intrinsicTriangleCount = 0;
    size_t valueWeightCount = 0;
    size_t gradientWeightCount = 0;
    std::vector<uint32_t> intrinsicTriangleIndices;

    VkBuffer triangleIndicesBuffer = VK_NULL_HANDLE;
    VkDeviceSize triangleIndicesBufferOffset = 0;

    VkBuffer voronoiCandidateBuffer = VK_NULL_HANDLE;
    VkDeviceSize voronoiCandidateBufferOffset = 0;

    VkBuffer gmlsSurfaceStencilBuffer = VK_NULL_HANDLE;
    VkDeviceSize gmlsSurfaceStencilBufferOffset = 0;

    VkBuffer gmlsSurfaceWeightBuffer = VK_NULL_HANDLE;
    VkDeviceSize gmlsSurfaceWeightBufferOffset = 0;

    VkBuffer gmlsSurfaceGradientWeightBuffer = VK_NULL_HANDLE;
    VkDeviceSize gmlsSurfaceGradientWeightBufferOffset = 0;

    VkBuffer surfaceBuffer = VK_NULL_HANDLE;
    VkDeviceSize surfaceBufferOffset = 0;

    std::unique_ptr<StencilKDTree> stencilKDTree;
};
