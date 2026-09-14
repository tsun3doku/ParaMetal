#pragma once

#include "../VoronoiConfig.hpp"
#include "../VoronoiGpuStructs.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

class VoxelGrid;
class VulkanDevice;

namespace voronoi {

class KNNNeighbors;

constexpr uint32_t RVDRetryPlaneCount = 256;
constexpr uint32_t RVDRetryVertexCount = 256;
constexpr uint32_t RVDRetryTriangleCount = 768;

struct RVDConfig {
    uint32_t baseChunkSize = 8192;
    uint32_t restrictedChunkSize = 1024;
    float contactRange = DefaultContactRange;
};

struct ModelGeometry {
    uint32_t runtimeModelId = 0;
    const VoxelGrid* voxelGrid = nullptr;
    const std::vector<std::array<glm::vec4, 3>>* triangles = nullptr;
};

struct RVDInput {
    const std::vector<glm::vec4>* seeds = nullptr;
    KNNNeighbors* seedSpatialIndex = nullptr;
    const std::array<glm::vec3, 8>* domainCorners = nullptr;
    const std::vector<ModelGeometry>* geometries = nullptr;
    float nominalCellSize = 0.0f;
};

inline constexpr uint32_t InvalidId = 0xffffffffu;

struct RestrictedSurface {
    uint32_t cellId = 0;
    uint32_t firstVertex = 0;
    uint32_t vertexCount = 0;
};

enum class FragmentFaceSource : uint8_t {
    VoronoiInterface,
    ModelSurface,
    DomainBoundary
};

struct FragmentFace {
    uint32_t fragmentA = InvalidId;
    uint32_t fragmentB = InvalidId;

    glm::vec3 normal{0.0f}; // Consistently oriented fragmentA -> fragmentB

    double area = 0.0;
    double orthogonalDistanceA = 0.0;
    double orthogonalDistanceB = 0.0;

    FragmentFaceSource source = FragmentFaceSource::VoronoiInterface;
    uint32_t domainFace = InvalidId;
};

struct SurfaceMeasure {
    double area = 0.0;
    glm::dvec3 moment{0.0};
};

struct ContactFace {
    uint32_t fragmentA = InvalidId;
    uint32_t fragmentB = InvalidId;

    double area = 0.0;
    double distanceA = 0.0;
    double distanceB = 0.0;
    double gap = 0.0;
};

struct RVDResult {
    std::vector<RestrictedSurface> surfaces;
    std::vector<SurfaceMeasure> surfaceExposure;

    std::vector<Fragment> fragments;
    std::vector<FragmentFace> fragmentFaces;
    std::vector<ContactFace> contactFaces;
    std::vector<ContactMeshVertex> contactMesh;
};

class RVD {
public:
    RVD();
    ~RVD();

    RVD(const RVD&) = delete;
    RVD& operator=(const RVD&) = delete;

    bool initialize(VulkanDevice& vulkanDevice);
    bool build(const RVDInput& input, const RVDConfig& config, RVDResult& result);
    void cleanup();

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation;
};

} // namespace voronoi
