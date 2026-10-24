#pragma once

#include "../GlobalDomainTopology.hpp"
#include "../VoronoiConfig.hpp"
#include "../VoronoiGpuStructs.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "../../cuda/CudaBuffer.cuh"
#include <glm/glm.hpp>

class VoxelGrid;
class VulkanDevice;

namespace voronoi {

class KNNNeighbors;

constexpr uint32_t RVDRetryPlaneCount = 256;
constexpr uint32_t RVDRetryVertexCount = 256;
constexpr uint32_t RVDRetryTriangleCount = 768;

// Overflow tier: failing base cells are re-run once at full capacity
constexpr uint32_t RVDOverflowPlaneCount = 512;
constexpr uint32_t RVDOverflowVertexCount = 512;

struct RVDConfig {
  uint32_t baseChunkSize = 8192;
  uint32_t restrictedChunkSize = 1024;
  float contactRange = DefaultContactRange;
};

struct ModelGeometry {
  uint32_t runtimeModelId = 0;
  const VoxelGrid *voxelGrid = nullptr;
  const std::vector<std::array<glm::vec4, 3>> *triangles = nullptr;
};

struct RVDInput {
  const std::vector<glm::vec4> *seeds = nullptr;
  KNNNeighbors *seedSpatialIndex = nullptr;
  const std::array<glm::vec3, 8> *domainCorners = nullptr;
  const std::vector<ModelGeometry> *geometries = nullptr;
  float nominalCellSize = 0.0f;

  uint8_t *outDevSolidMask = nullptr;
  glm::vec3 solidMaskGridMin{};
  glm::ivec3 solidMaskGridDim{};
  float solidMaskCellSize = 0.0f;
};

inline constexpr uint32_t InvalidId = 0xffffffffu;

struct RestrictedSurface {
  uint32_t cellId = 0;
  uint32_t firstVertex = 0;
  uint32_t vertexCount = 0;
  uint32_t triangleId = InvalidId;
};

using FaceType = topology::FaceType;
using FragmentFace = topology::Face;
using ContactCoupling = topology::ContactCoupling;

struct RVDResult {
  std::vector<Fragment> fragments;
  std::vector<FragmentFace> fragmentFaces;
  std::vector<ContactCoupling> contactCouplings;
  std::vector<ContactMeshVertex> contactMesh;

  // Seed-to-fragment provenance for fluid visualization
  std::vector<uint32_t> seedToFragment;
};

class RVD {
public:
  RVD();
  ~RVD();

  RVD(const RVD &) = delete;
  RVD &operator=(const RVD &) = delete;

  bool initialize(VulkanDevice &vulkanDevice);
  bool build(const RVDInput &input, const RVDConfig &config, RVDResult &result);
  void cleanup();

private:
  class Implementation;
  std::unique_ptr<Implementation> implementation;
};

} // namespace voronoi
