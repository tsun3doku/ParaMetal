#pragma once

#include "../../cuda/CudaBuffer.cuh"
#include "../../cuda/CudaMath.cuh"
#include "../../cuda/Triangle.cuh"
#include "../../cuda/VoxelGrid.cuh"
#include "RVD.hpp"
#include "RVDPredicates.cuh"

#include <cuda_runtime.h>
#include <cfloat>
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
inline constexpr float CanonicalDomainSize = 1000.0f;

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
  uint32_t root = InvalidNodeId;

  void reset();
  bool valid() const;
};

struct PointRay {
  double crossings = 0.0;
  bool onSurface = false;
  bool ambiguous = false;
};

struct SideWinding {
  double value = 0.0;
  bool ambiguous = false;
};

enum PointClass : uint32_t {
  PointOutside = 0,
  PointInside = 1,
  PointOnSurface = 2,
  PointUnresolved = 3,
};

struct DeviceModelGeometryView {
  const DeviceTriangle *triangles = nullptr;
  uint32_t triangleCount = 0;
  const uint3 *triangleVertexIds = nullptr;
  const TriangleBvhNode *nodes = nullptr;
  uint32_t root = InvalidNodeId;
  uint32_t runtimeModelId = 0;
  float3 boundsMin{0.0f};
  float3 boundsMax{0.0f};
};

struct DeviceModelGeometry {
  cudaUtils::CudaBuffer<DeviceTriangle> triangles;
  cudaUtils::CudaBuffer<uint3> triangleVertexIds;
  cudaUtils::CudaBuffer<int32_t> voxelTriangleIds;
  cudaUtils::CudaBuffer<int32_t> voxelOffsets;
  DeviceVoxelGrid voxelGrid{};
  uint32_t runtimeModelId = 0;
  float3 boundsMin{0.0f};
  float3 boundsMax{0.0f};
  TriangleBvh bvh;

  void reset();
  size_t voxelCellCount() const;
  bool valid() const;
  DeviceModelGeometryView view() const;
};

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
  const double denominator = la * lb * lc + dot3(a, b) * lc + dot3(b, c) * la + dot3(c, a) * lb;
  return 2.0 * atan2(numerator, denominator);
}

__device__ inline double computeWinding(
    const DeviceTriangle* triangles,
    const TriangleBvhNode* nodes,
    uint32_t root,
    double3 corner) {
  using cudaMath::toDouble3;
  using cudaMath::sub3;
  if (root == InvalidNodeId) return 0.0;
  double winding = 0.0;

  uint32_t stack[64];
  int stackTop = 0;
  uint32_t current = root;

  while (true) {
    const TriangleBvhNode node = nodes[current];
    if (node.isLeaf()) {
      const DeviceTriangle tri = triangles[node.triangleId()];
      winding += triangleSolidAngle(
          sub3(toDouble3(tri.v0), corner),
          sub3(toDouble3(tri.v1), corner),
          sub3(toDouble3(tri.v2), corner));
      if (stackTop == 0) break;
      current = stack[--stackTop];
      continue;
    }
    if (stackTop == 64) return __longlong_as_double(0x7ff8000000000000ull);
    stack[stackTop++] = node.rightChild();
    current = node.leftChild();
  }
  return winding * (1.0 / (4.0 * 3.14159265358979323846));
}

__device__ inline bool rayHitsBounds(
    const TriangleBvhNode &node,
    double3 origin,
    double3 direction) {
  const double minimum[3] = {double(node.minimum.x), double(node.minimum.y), double(node.minimum.z)};
  const double maximum[3] = {double(node.maximum.x), double(node.maximum.y), double(node.maximum.z)};
  const double position[3] = {origin.x, origin.y, origin.z};
  const double travel[3] = {direction.x, direction.y, direction.z};
  double tNear = -DBL_MAX;
  double tFar = DBL_MAX;
  for (int axis = 0; axis < 3; ++axis) {
    if (travel[axis] == 0.0) {
      if (position[axis] < minimum[axis] || position[axis] > maximum[axis]) {
        return false;
      }
      continue;
    }
    const double t0 = (minimum[axis] - position[axis]) / travel[axis];
    const double t1 = (maximum[axis] - position[axis]) / travel[axis];
    const double enter = t0 < t1 ? t0 : t1;
    const double exit = t0 < t1 ? t1 : t0;
    if (enter > tNear) tNear = enter;
    if (exit < tFar) tFar = exit;
  }
  tNear = nextafter(tNear, -DBL_MAX);
  tFar = nextafter(tFar, DBL_MAX);
  return tFar >= tNear && tFar >= 0.0;
}

__device__ inline int classifyRayCrossing(int edgeAB, int edgeBC, int edgeCA) {
  if (edgeAB == 0 || edgeBC == 0 || edgeCA == 0) {
    if (edgeAB == 0 && edgeBC != 0 && edgeCA != 0 && edgeBC != edgeCA) return 0;
    if (edgeBC == 0 && edgeAB != 0 && edgeCA != 0 && edgeAB != edgeCA) return 0;
    if (edgeCA == 0 && edgeAB != 0 && edgeBC != 0 && edgeAB != edgeBC) return 0;
    return -1;
  }
  if (edgeAB == edgeBC && edgeBC == edgeCA) return 1;
  return 0;
}

__device__ inline PointRay castPointRay(
    const DeviceTriangle* triangles,
    const TriangleBvhNode* nodes,
    uint32_t root,
    double3 origin,
    double3 direction,
    uint32_t excludedTriangle,
    bool detectSurface) {
  using cudaMath::toDouble3;
  using cudaMath::sub3;
  PointRay result{};
  if (triangles == nullptr || nodes == nullptr || root == InvalidNodeId) {
    return result;
  }

  uint32_t stack[64];
  int stackTop = 0;
  uint32_t current = root;

  while (true) {
    const TriangleBvhNode node = nodes[current];
    if (!rayHitsBounds(node, origin, direction)) {
      if (stackTop == 0) break;
      current = stack[--stackTop];
      continue;
    }
    if (node.isLeaf()) {
      const uint32_t triangleId = node.triangleId();
      if (triangleId != excludedTriangle) {
        const DeviceTriangle tri = triangles[triangleId];
        const double3 a = toDouble3(tri.v0);
        const double3 b = toDouble3(tri.v1);
        const double3 c = toDouble3(tri.v2);
        const double3 ab = sub3(b, a);
        const double3 ac = sub3(c, a);
        const double3 ao = sub3(origin, a);
        const int denominatorSign =
            rvdDevice::det3x3ExactSign(ab, direction, ac);
        const int tSign = rvdDevice::det3x3ExactSign(ac, ao, ab);
        if (denominatorSign != 0) {
          const bool ahead =
              tSign != 0 && ((tSign > 0) == (denominatorSign > 0));
          const bool inPlane = detectSurface && tSign == 0;
          if (ahead || inPlane) {
            const int crossing = classifyRayCrossing(
                rvdDevice::det3x3ExactSign(ab, ao, direction),
                rvdDevice::det3x3ExactSign(
                    sub3(c, b), sub3(origin, b), direction),
                rvdDevice::det3x3ExactSign(
                    sub3(a, c), sub3(origin, c), direction));
            if (inPlane) {
              if (crossing != 0) {
                result.onSurface = true;
                return result;
              }
            } else if (crossing < 0) {
              result.ambiguous = true;
              return result;
            } else if (crossing > 0) {
              result.crossings += denominatorSign < 0 ? 1.0 : -1.0;
            }
          }
        } else if (detectSurface && tSign == 0) {
          result.ambiguous = true;
          return result;
        }
      }
      if (stackTop == 0) break;
      current = stack[--stackTop];
      continue;
    }
    if (stackTop == 64) {
      result.ambiguous = true;
      return result;
    }
    stack[stackTop++] = node.rightChild();
    current = node.leftChild();
  }
  return result;
}

__device__ inline uint32_t classifyPoint(
    const DeviceTriangle* triangles,
    const TriangleBvhNode* nodes,
    uint32_t root,
    double3 point) {
  if (triangles == nullptr || nodes == nullptr || root == InvalidNodeId) {
    return PointOutside;
  }
  const double3 directions[9] = {
      make_double3(1.0, 0.0, 0.0), make_double3(0.0, 1.0, 0.0),
      make_double3(0.0, 0.0, 1.0), make_double3(1.0, 1.0, 0.0),
      make_double3(1.0, 0.0, 1.0), make_double3(0.0, 1.0, 1.0),
      make_double3(1.0, 1.0, 1.0), make_double3(1.0, -1.0, 1.0),
      make_double3(-1.0, 1.0, 1.0)};
  for (const double3 &direction : directions) {
    const PointRay ray = castPointRay(
        triangles, nodes, root, point, direction, InvalidNodeId, true);
    if (ray.onSurface) return PointOnSurface;
    if (!ray.ambiguous) {
      return fabs(ray.crossings) > 0.5 ? PointInside : PointOutside;
    }
  }
  const double winding = computeWinding(triangles, nodes, root, point);
  if (!isfinite(winding)) return PointUnresolved;
  return fabs(winding) > 0.5 ? PointInside : PointOutside;
}

__device__ inline SideWinding computeSideWinding(
    const DeviceTriangle* triangles,
    const TriangleBvhNode* nodes,
    uint32_t root,
    double3 origin,
    double3 normal,
    uint32_t excludedTriangle) {
  SideWinding result{};
  if (triangles == nullptr || nodes == nullptr || root == InvalidNodeId) {
    return result;
  }

  const double nx = fabs(normal.x);
  const double ny = fabs(normal.y);
  const double nz = fabs(normal.z);
  double3 axis = make_double3(1.0, 0.0, 0.0);
  if (ny <= nx && ny <= nz) {
    axis = make_double3(0.0, 1.0, 0.0);
  } else if (nz <= nx && nz <= ny) {
    axis = make_double3(0.0, 0.0, 1.0);
  }
  const double3 tangentA = cudaMath::cross3(normal, axis);
  const double3 tangentB = cudaMath::cross3(normal, tangentA);
  constexpr double step = 1.0 / 1048576.0;
  const double3 candidates[5] = {
      normal,
      make_double3(normal.x + tangentA.x * step, normal.y + tangentA.y * step,
                   normal.z + tangentA.z * step),
      make_double3(normal.x - tangentA.x * step, normal.y - tangentA.y * step,
                   normal.z - tangentA.z * step),
      make_double3(normal.x + tangentB.x * step, normal.y + tangentB.y * step,
                   normal.z + tangentB.z * step),
      make_double3(normal.x - tangentB.x * step, normal.y - tangentB.y * step,
                   normal.z - tangentB.z * step)};

  for (const double3 &direction : candidates) {
    if (rvdDevice::dot3FilteredExact(normal, direction).sign <= 0) continue;
    const PointRay ray = castPointRay(
        triangles, nodes, root, origin, direction, excludedTriangle, false);
    if (ray.ambiguous) continue;
    result.value = ray.crossings;
    return result;
  }
  result.ambiguous = true;
  return result;
}

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

DeviceVoxelGrid makeRVDVoxelGrid(
    const glm::vec3& meshMin,
    const glm::vec3& meshMax,
    uint32_t voxelResolution);

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

} // namespace voronoi::cudaGeometry
