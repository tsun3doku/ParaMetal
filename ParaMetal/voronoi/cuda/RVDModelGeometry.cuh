#pragma once

#include "../../cuda/CudaBuffer.cuh"
#include "../../cuda/CudaMath.cuh"
#include "../../cuda/Triangle.cuh"
#include "../../cuda/VoxelGrid.cuh"
#include "RVD.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

namespace voronoi::cudaGeometry {

using ::cudaGeometry::DeviceTriangle;
using ::cudaGeometry::DeviceVoxelGrid;

constexpr uint32_t LeafBit = 0x80000000u;
constexpr uint32_t InvalidNodeId = 0xffffffffu;

struct alignas(16) TriangleBvhNode {
  float3 minimum;
  uint32_t left;
  float3 maximum;
  uint32_t right;

  __device__ __host__ inline bool isLeaf() const { return (left & LeafBit) != 0; }
  __device__ __host__ inline uint32_t triangleId() const { return left & ~LeafBit; }
  __device__ __host__ inline uint32_t leftChild() const { return left; }
  __device__ __host__ inline uint32_t rightChild() const { return right; }
};

struct TriangleBvh {
  cudaUtils::CudaBuffer<TriangleBvhNode> nodes;
  cudaUtils::CudaBuffer<uint32_t> parents;
  uint32_t root = InvalidNodeId;

  void reset() {
    nodes.reset();
    parents.reset();
    root = InvalidNodeId;
  }

  bool valid() const {
    return root != InvalidNodeId && nodes.size() > 0 && parents.size() > 0;
  }
};

__host__ __device__ inline float pointAabbDistanceSquared(float3 p, float3 bMin, float3 bMax) {
  const float dx = fmaxf(0.0f, fmaxf(bMin.x - p.x, p.x - bMax.x));
  const float dy = fmaxf(0.0f, fmaxf(bMin.y - p.y, p.y - bMax.y));
  const float dz = fmaxf(0.0f, fmaxf(bMin.z - p.z, p.z - bMax.z));
  return dx * dx + dy * dy + dz * dz;
}

bool buildTriangleBvh(
    const cudaUtils::CudaBuffer<DeviceTriangle>& triangles,
    TriangleBvh& bvh,
    cudaStream_t stream);

// Winding number computation
__host__ __device__ inline double triangleSolidAngle(double3 a, double3 b, double3 c) {
  using cudaMath::dot3;
  using cudaMath::cross3;
  const double la2 = dot3(a, a);
  const double lb2 = dot3(b, b);
  const double lc2 = dot3(c, c);
  if (!(la2 > 0.0) || !(lb2 > 0.0) || !(lc2 > 0.0)) {
    return 0.0;
  }
  const double la = sqrt(la2);
  const double lb = sqrt(lb2);
  const double lc = sqrt(lc2);
  const double numerator = dot3(a, cross3(b, c));
  const double denominator =
      la * lb * lc +
      dot3(a, b) * lc +
      dot3(b, c) * la +
      dot3(c, a) * lb;
  return 2.0 * atan2(numerator, denominator);
}

__device__ inline bool windingNodeIsFar(float3 point, const TriangleBvhNode& node) {
  const float3 extent = make_float3(
      node.maximum.x - node.minimum.x,
      node.maximum.y - node.minimum.y,
      node.maximum.z - node.minimum.z);
  const float radiusSquared =
      0.25f * (extent.x * extent.x + extent.y * extent.y + extent.z * extent.z);
  return pointAabbDistanceSquared(point, node.minimum, node.maximum) > 9.0f * radiusSquared;
}

__device__ inline double computeWinding(
    const DeviceTriangle* triangles,
    const TriangleBvhNode* nodes,
    const uint32_t* parents,
    uint32_t root,
    const float4* moments,
    double3 corner,
    float3 cornerF) {
  using cudaMath::toDouble3;
  using cudaMath::sub3;
  if (root == InvalidNodeId) return 0.0;
  double winding = 0.0;

  uint32_t stack[64];
  int stackTop = 0;
  stack[stackTop++] = root;

  while (stackTop > 0) {
    const uint32_t curr = stack[--stackTop];
    const TriangleBvhNode node = nodes[curr];

    if (node.isLeaf()) {
      const DeviceTriangle tri = triangles[node.triangleId()];
      winding += triangleSolidAngle(
          sub3(toDouble3(tri.v0), corner),
          sub3(toDouble3(tri.v1), corner),
          sub3(toDouble3(tri.v2), corner));
    } else if (windingNodeIsFar(cornerF, node)) {
      const float4 moment = moments[curr];
      const double cx = 0.5 * (double(node.minimum.x) + double(node.maximum.x));
      const double cy = 0.5 * (double(node.minimum.y) + double(node.maximum.y));
      const double cz = 0.5 * (double(node.minimum.z) + double(node.maximum.z));
      const double dx = cx - corner.x;
      const double dy = cy - corner.y;
      const double dz = cz - corner.z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > 0.0) {
        const double inverseR3 = 1.0 / (r2 * sqrt(r2));
        winding +=
            (double(moment.x) * dx + double(moment.y) * dy + double(moment.z) * dz) *
            inverseR3;
      }
    } else {
      if (stackTop < 62) {
        stack[stackTop++] = node.rightChild();
        stack[stackTop++] = node.leftChild();
      }
    }
  }
  return winding * (1.0 / (4.0 * 3.14159265358979323846));
}

inline constexpr float CanonicalDomainSize = 1000.0f;
inline constexpr float RVDVoxelScale = 990.222f;
inline constexpr float RVDVoxelBias = 4.998f;

inline DeviceVoxelGrid makeRVDVoxelGrid(
    const glm::vec3& meshMin,
    const glm::vec3& meshMax,
    uint32_t voxelResolution) {
  DeviceVoxelGrid grid{};
  grid.scale = 0.0f;
  grid.voxelSize = 0.0f;
  grid.gridDim = make_int3(0, 0, 0);
  const glm::vec3 extent = meshMax - meshMin;
  const float maxExtent = std::max(std::max(extent.x, extent.y), extent.z);
  if (!(maxExtent > 0.0f) || !std::isfinite(maxExtent) || voxelResolution == 0) {
    return grid;
  }

  grid.scale = RVDVoxelScale / maxExtent;
  grid.voxelSize = CanonicalDomainSize / float(voxelResolution);
  grid.gridMin = make_float3(
      meshMin.x - RVDVoxelBias / grid.scale,
      meshMin.y - RVDVoxelBias / grid.scale,
      meshMin.z - RVDVoxelBias / grid.scale);
  grid.gridDim = make_int3(
      int(voxelResolution), int(voxelResolution), int(voxelResolution));
  return grid;
}

struct DeviceModelGeometryView {
  const DeviceTriangle *triangles = nullptr;
  const uint3 *triangleVertexIds = nullptr;
  const TriangleBvhNode *nodes = nullptr;
  const uint32_t *parents = nullptr;
  const float4 *moments = nullptr;
  uint32_t root = InvalidNodeId;
  uint32_t runtimeModelId = 0;
  float3 boundsMin{0.0f};
  float3 boundsMax{0.0f};
};

struct DeviceModelGeometry {
  cudaUtils::CudaBuffer<DeviceTriangle> triangles;
  cudaUtils::CudaBuffer<uint3> triangleVertexIds;
  cudaUtils::CudaBuffer<float4> bvhMoments;
  cudaUtils::CudaBuffer<int32_t> voxelTriangleIds;
  cudaUtils::CudaBuffer<int32_t> voxelOffsets;
  DeviceVoxelGrid voxelGrid{};
  uint32_t runtimeModelId = 0;
  float3 boundsMin{0.0f};
  float3 boundsMax{0.0f};
  TriangleBvh bvh;

  void reset() {
    triangles.reset();
    triangleVertexIds.reset();
    bvhMoments.reset();
    voxelTriangleIds.reset();
    voxelOffsets.reset();
    bvh.reset();
    voxelGrid = {};
    runtimeModelId = 0;
    boundsMin = make_float3(0.0f, 0.0f, 0.0f);
    boundsMax = make_float3(0.0f, 0.0f, 0.0f);
  }

  size_t voxelCellCount() const {
    return size_t(voxelGrid.gridDim.x) * size_t(voxelGrid.gridDim.y) *
           size_t(voxelGrid.gridDim.z);
  }

  bool valid() const {
    return triangles.size() != 0 &&
           triangleVertexIds.size() == triangles.size() &&
           bvhMoments.size() != 0 &&
           voxelOffsets.size() == voxelCellCount() + 1 &&
           voxelGrid.gridDim.x > 0 && voxelGrid.gridDim.y > 0 &&
           voxelGrid.gridDim.z > 0 &&
           voxelGrid.scale > 0.0f &&
           bvh.valid();
  }

  DeviceModelGeometryView view() const {
    DeviceModelGeometryView v{};
    v.triangles = triangles.get();
    v.triangleVertexIds = triangleVertexIds.get();
    v.nodes = bvh.nodes.get();
    v.parents = bvh.parents.get();
    v.moments = bvhMoments.get();
    v.root = bvh.root;
    v.runtimeModelId = runtimeModelId;
    v.boundsMin = boundsMin;
    v.boundsMax = boundsMax;
    return v;
  }
};

bool buildBvhMoments(
    DeviceModelGeometry& geometry,
    cudaStream_t stream = nullptr);

bool buildWindingSolidMask(
    const DeviceModelGeometryView* views,
    uint32_t viewCount,
    float3 gridMin,
    int3 gridDim,
    float cellSize,
    uint8_t* outMask,
    cudaStream_t stream = nullptr);

bool uploadModelGeometry(
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>& indices,
    uint32_t runtimeModelId,
    uint32_t voxelResolution,
    DeviceModelGeometry& geometry,
    cudaStream_t stream = nullptr);

bool uploadGeometries(
    const RVDInput& input,
    std::vector<DeviceModelGeometry>& geometries,
    cudaStream_t stream = nullptr);

__host__ __device__ inline float3 canonicalFromWorld(const DeviceVoxelGrid &grid, float3 point) {
  return make_float3(
      (point.x - grid.gridMin.x) * grid.scale,
      (point.y - grid.gridMin.y) * grid.scale,
      (point.z - grid.gridMin.z) * grid.scale);
}

__host__ __device__ inline double3 canonicalToWorld(
    double3 point,
    double3 domainMin,
    double invBaseScale) {
  return make_double3(
      domainMin.x + point.x * invBaseScale,
      domainMin.y + point.y * invBaseScale,
      domainMin.z + point.z * invBaseScale);
}

} // namespace voronoi::cudaGeometry
