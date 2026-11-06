#include "RVDModelGeometry.cuh"

#include "../../cuda/CudaCheck.cuh"
#include "../../cuda/CudaMath.cuh"
#include "../../cuda/VoxelGrid.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace voronoi::cudaGeometry {

using cudaMath::min3;
using cudaMath::max3;

namespace {

constexpr uint32_t BlockSize = 256;
using BlockReduce = cub::BlockReduce<float, BlockSize>;

__device__ inline uint32_t expandBits(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

__device__ inline uint32_t morton3D(float3 p, float3 bMin, float3 bInvExt) {
  const float x = fminf(fmaxf((p.x - bMin.x) * bInvExt.x, 0.0f), 1.0f);
  const float y = fminf(fmaxf((p.y - bMin.y) * bInvExt.y, 0.0f), 1.0f);
  const float z = fminf(fmaxf((p.z - bMin.z) * bInvExt.z, 0.0f), 1.0f);

  const uint32_t ix = min(max(int(x * 1023.0f), 0), 1023);
  const uint32_t iy = min(max(int(y * 1023.0f), 0), 1023);
  const uint32_t iz = min(max(int(z * 1023.0f), 0), 1023);

  return (expandBits(ix) << 2) | (expandBits(iy) << 1) | expandBits(iz);
}

__device__ float atomicMinFloat(float* address, float value) {
  int* bits = reinterpret_cast<int*>(address);
  int old = *bits;
  while (value < __int_as_float(old)) {
    const int assumed = old;
    old = atomicCAS(bits, assumed, __float_as_int(value));
    if (old == assumed) break;
  }
  return __int_as_float(old);
}

__device__ float atomicMaxFloat(float* address, float value) {
  int* bits = reinterpret_cast<int*>(address);
  int old = *bits;
  while (value > __int_as_float(old)) {
    const int assumed = old;
    old = atomicCAS(bits, assumed, __float_as_int(value));
    if (old == assumed) break;
  }
  return __int_as_float(old);
}

__global__ void computeCentroidBoundsKernel(
    const DeviceTriangle* triangles,
    uint32_t count,
    float3* outMin,
    float3* outMax) {
  float3 bMin = make_float3(1e30f, 1e30f, 1e30f);
  float3 bMax = make_float3(-1e30f, -1e30f, -1e30f);

  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
    const DeviceTriangle tri = triangles[i];
    const float3 c = make_float3(
        (tri.v0.x + tri.v1.x + tri.v2.x) * (1.0f / 3.0f),
        (tri.v0.y + tri.v1.y + tri.v2.y) * (1.0f / 3.0f),
        (tri.v0.z + tri.v1.z + tri.v2.z) * (1.0f / 3.0f));
    bMin = min3(bMin, c);
    bMax = max3(bMax, c);
  }

  __shared__ typename BlockReduce::TempStorage temp;
  const float minX = BlockReduce(temp).Reduce(bMin.x, cub::Min()); __syncthreads();
  const float minY = BlockReduce(temp).Reduce(bMin.y, cub::Min()); __syncthreads();
  const float minZ = BlockReduce(temp).Reduce(bMin.z, cub::Min()); __syncthreads();
  const float maxX = BlockReduce(temp).Reduce(bMax.x, cub::Max()); __syncthreads();
  const float maxY = BlockReduce(temp).Reduce(bMax.y, cub::Max()); __syncthreads();
  const float maxZ = BlockReduce(temp).Reduce(bMax.z, cub::Max()); __syncthreads();

  if (threadIdx.x == 0) {
    atomicMinFloat(&outMin->x, minX);
    atomicMinFloat(&outMin->y, minY);
    atomicMinFloat(&outMin->z, minZ);

    atomicMaxFloat(&outMax->x, maxX);
    atomicMaxFloat(&outMax->y, maxY);
    atomicMaxFloat(&outMax->z, maxZ);
  }
}

__global__ void computeMortonKeysKernel(
    const DeviceTriangle* triangles,
    uint32_t count,
    float3 bMin,
    float3 bInvExt,
    uint64_t* outKeys) {
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  const DeviceTriangle tri = triangles[idx];
  const float3 c = make_float3(
      (tri.v0.x + tri.v1.x + tri.v2.x) * (1.0f / 3.0f),
      (tri.v0.y + tri.v1.y + tri.v2.y) * (1.0f / 3.0f),
      (tri.v0.z + tri.v1.z + tri.v2.z) * (1.0f / 3.0f));

  const uint32_t m = morton3D(c, bMin, bInvExt);
  outKeys[idx] = (uint64_t(m) << 32) | uint64_t(idx);
}

__device__ inline int commonPrefix(const uint64_t* keys, int count, int a, int b) {
  if (b < 0 || b >= count) return -1;
  return __clzll(keys[a] ^ keys[b]);
}

__device__ inline int findSplit(const uint64_t* keys, int first, int last) {
  const uint64_t firstKey = keys[first];
  const uint64_t lastKey = keys[last];
  const int common = __clzll(firstKey ^ lastKey);

  int split = first;
  int step = last - first;
  do {
    step = (step + 1) >> 1;
    const int candidate = split + step;
    if (candidate < last) {
      const int prefix = __clzll(firstKey ^ keys[candidate]);
      if (prefix > common) {
        split = candidate;
      }
    }
  } while (step > 1);

  return split;
}

__global__ void generateHierarchyKernel(
    const uint64_t* sortedKeys,
    uint32_t count,
    TriangleBvhNode* nodes,
    uint32_t* nodeParents) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= int(count) - 1) return;

  const int deltaNext = commonPrefix(sortedKeys, int(count), i, i + 1);
  const int deltaPrev = commonPrefix(sortedKeys, int(count), i, i - 1);
  const int direction = deltaNext >= deltaPrev ? 1 : -1;
  const int deltaMin = commonPrefix(sortedKeys, int(count), i, i - direction);

  int lMax = 2;
  while (commonPrefix(sortedKeys, int(count), i, i + lMax * direction) > deltaMin) {
    lMax <<= 1;
  }

  int length = 0;
  for (int step = lMax >> 1; step > 0; step >>= 1) {
    if (commonPrefix(sortedKeys, int(count), i, i + (length + step) * direction) > deltaMin) {
      length += step;
    }
  }

  const int j = i + length * direction;
  const int first = min(i, j);
  const int last = max(i, j);
  const int split = findSplit(sortedKeys, first, last);

  const uint32_t leafBase = count - 1;
  const uint32_t leftChild = (split == first) ? (leafBase + uint32_t(split)) : uint32_t(split);
  const uint32_t rightChild = (split + 1 == last) ? (leafBase + uint32_t(split + 1)) : uint32_t(split + 1);

  nodes[i].left = leftChild;
  nodes[i].right = rightChild;
  nodeParents[leftChild] = uint32_t(i);
  nodeParents[rightChild] = uint32_t(i);
}

__global__ void initLeavesAndPropagateAabbKernel(
    const DeviceTriangle* triangles,
    const uint64_t* sortedKeys,
    uint32_t count,
    TriangleBvhNode* nodes,
    const uint32_t* nodeParents,
    int* nodeFlags) {
  const uint32_t leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (leafIdx >= count) return;

  const uint32_t triId = uint32_t(sortedKeys[leafIdx]);
  const DeviceTriangle tri = triangles[triId];
  const float3 a = make_float3(tri.v0.x, tri.v0.y, tri.v0.z);
  const float3 b = make_float3(tri.v1.x, tri.v1.y, tri.v1.z);
  const float3 c = make_float3(tri.v2.x, tri.v2.y, tri.v2.z);

  const uint32_t nodeIdx = (count - 1) + leafIdx;
  nodes[nodeIdx].minimum = min3(a, min3(b, c));
  nodes[nodeIdx].maximum = max3(a, max3(b, c));
  nodes[nodeIdx].left = triId | LeafBit;
  nodes[nodeIdx].right = InvalidNodeId;

  uint32_t current = nodeParents[nodeIdx];
  while (current != InvalidNodeId) {
    __threadfence();
    const int previous = atomicAdd(&nodeFlags[current], 1);
    if (previous == 0) break;

    const uint32_t left = nodes[current].leftChild();
    const uint32_t right = nodes[current].rightChild();
    nodes[current].minimum = min3(nodes[left].minimum, nodes[right].minimum);
    nodes[current].maximum = max3(nodes[left].maximum, nodes[right].maximum);
    current = nodeParents[current];
  }
}

__global__ void initSingleLeafKernel(
    const DeviceTriangle* triangles,
    TriangleBvhNode* nodes,
    uint32_t* parents) {
  const DeviceTriangle tri = triangles[0];
  const float3 a = make_float3(tri.v0.x, tri.v0.y, tri.v0.z);
  const float3 b = make_float3(tri.v1.x, tri.v1.y, tri.v1.z);
  const float3 c = make_float3(tri.v2.x, tri.v2.y, tri.v2.z);

  nodes[0].minimum = min3(a, min3(b, c));
  nodes[0].maximum = max3(a, max3(b, c));
  nodes[0].left = 0u | LeafBit;
  nodes[0].right = InvalidNodeId;
  parents[0] = InvalidNodeId;
}

__global__ void expandTrianglesKernel(
    const float3* positions,
    uint32_t vertexCount,
    const uint32_t* indices,
    uint32_t triangleCount,
    DeviceTriangle* triangles,
    uint3* triangleVertexIds) {
  const uint32_t triangle = blockIdx.x * blockDim.x + threadIdx.x;
  if (triangle >= triangleCount) return;

  const uint32_t i0 = indices[triangle * 3 + 0];
  const uint32_t i1 = indices[triangle * 3 + 1];
  const uint32_t i2 = indices[triangle * 3 + 2];
  if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) {
    triangles[triangle] = DeviceTriangle{};
    triangleVertexIds[triangle] = make_uint3(InvalidId, InvalidId, InvalidId);
    return;
  }

  const float3 p0 = positions[i0];
  const float3 p1 = positions[i1];
  const float3 p2 = positions[i2];
  triangles[triangle] = DeviceTriangle{
      make_float4(p0.x, p0.y, p0.z, 0.0f),
      make_float4(p1.x, p1.y, p1.z, 0.0f),
      make_float4(p2.x, p2.y, p2.z, 0.0f)};
  triangleVertexIds[triangle] = make_uint3(i0, i1, i2);
}

bool computePositionBounds(
    const std::vector<glm::vec3>& positions,
    glm::vec3& boundsMin,
    glm::vec3& boundsMax) {
  boundsMin = glm::vec3(std::numeric_limits<float>::max());
  boundsMax = glm::vec3(-std::numeric_limits<float>::max());
  for (const glm::vec3& position : positions) {
    boundsMin = glm::min(boundsMin, position);
    boundsMax = glm::max(boundsMax, position);
  }
  return boundsMin.x <= boundsMax.x &&
         boundsMin.y <= boundsMax.y &&
         boundsMin.z <= boundsMax.z;
}

} // namespace

bool buildTriangleBvh(
    const cudaUtils::CudaBuffer<DeviceTriangle>& triangles,
    TriangleBvh& bvh,
    cudaStream_t stream) {
  bvh.reset();
  const size_t triangleCount = triangles.size();
  if (triangleCount == 0) return true;
  if (triangleCount >= LeafBit) return false;

  const uint32_t count = uint32_t(triangleCount);

  if (count == 1) {
    if (!bvh.nodes.allocate(1) || !bvh.parents.allocate(1)) return false;
    initSingleLeafKernel<<<1, 1, 0, stream>>>(triangles.get(), bvh.nodes.get(), bvh.parents.get());
    bvh.root = 0;
    return cudaGetLastError() == cudaSuccess;
  }

  const uint32_t totalNodes = 2 * count - 1;
  if (!bvh.nodes.allocate(totalNodes)) return false;

  cudaUtils::CudaBuffer<float3> devBounds;
  if (!devBounds.allocate(2)) return false;
  const float3 initMin = make_float3(1e30f, 1e30f, 1e30f);
  const float3 initMax = make_float3(-1e30f, -1e30f, -1e30f);
  if (cudaMemcpyAsync(devBounds.get(), &initMin, sizeof(float3), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      cudaMemcpyAsync(devBounds.get() + 1, &initMax, sizeof(float3), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
    return false;
  }

  const uint32_t gridBounds = min((count + BlockSize - 1) / BlockSize, 64u);
  computeCentroidBoundsKernel<<<gridBounds, BlockSize, 0, stream>>>(
      triangles.get(), count, devBounds.get(), devBounds.get() + 1);

  float3 hBounds[2];
  if (cudaMemcpyAsync(hBounds, devBounds.get(), 2 * sizeof(float3), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return false;
  }

  const float3 bMin = hBounds[0];
  const float3 extent = make_float3(
      hBounds[1].x - hBounds[0].x,
      hBounds[1].y - hBounds[0].y,
      hBounds[1].z - hBounds[0].z);

  const float3 bInvExt = make_float3(
      extent.x > 0.0f ? 1.0f / extent.x : 0.0f,
      extent.y > 0.0f ? 1.0f / extent.y : 0.0f,
      extent.z > 0.0f ? 1.0f / extent.z : 0.0f);

  cudaUtils::CudaBuffer<uint64_t> devKeysIn;
  cudaUtils::CudaBuffer<uint64_t> devKeysOut;
  if (!devKeysIn.allocate(count) || !devKeysOut.allocate(count)) return false;

  const uint32_t grid = (count + BlockSize - 1) / BlockSize;
  computeMortonKeysKernel<<<grid, BlockSize, 0, stream>>>(
      triangles.get(), count, bMin, bInvExt, devKeysIn.get());

  size_t tempStorageBytes = 0;
  if (cub::DeviceRadixSort::SortKeys(
          nullptr, tempStorageBytes,
          devKeysIn.get(), devKeysOut.get(),
          int(count), 0, 64, stream) != cudaSuccess) return false;

  cudaUtils::CudaBuffer<uint8_t> devTempStorage;
  if (!devTempStorage.allocate(tempStorageBytes)) return false;

  if (cub::DeviceRadixSort::SortKeys(
          devTempStorage.get(), tempStorageBytes,
          devKeysIn.get(), devKeysOut.get(),
          int(count), 0, 64, stream) != cudaSuccess) return false;

  cudaUtils::CudaBuffer<uint32_t> devNodeParents;
  cudaUtils::CudaBuffer<int> devNodeFlags;
  if (!devNodeParents.allocate(totalNodes) || !devNodeFlags.allocate(count - 1)) return false;
  if (cudaMemsetAsync(devNodeParents.get(), 0xFF, totalNodes * sizeof(uint32_t), stream) != cudaSuccess ||
      cudaMemsetAsync(devNodeFlags.get(), 0, (count - 1) * sizeof(int), stream) != cudaSuccess) return false;

  const uint32_t gridInternal = (count - 1 + BlockSize - 1) / BlockSize;
  generateHierarchyKernel<<<gridInternal, BlockSize, 0, stream>>>(
      devKeysOut.get(), count, bvh.nodes.get(), devNodeParents.get());

  initLeavesAndPropagateAabbKernel<<<grid, BlockSize, 0, stream>>>(
      triangles.get(), devKeysOut.get(), count, bvh.nodes.get(), devNodeParents.get(), devNodeFlags.get());

  bvh.parents.swap(devNodeParents);
  bvh.root = 0;
  return cudaGetLastError() == cudaSuccess;
}

__global__ void accumulateBvhMomentsKernel(
    const DeviceTriangle* triangles,
    const TriangleBvhNode* nodes,
    uint32_t nodeCount,
    const uint32_t* parents,
    float4* moments) {
  const uint32_t nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (nodeIdx >= nodeCount) return;

  const TriangleBvhNode node = nodes[nodeIdx];
  if (!node.isLeaf()) return;

  const DeviceTriangle tri = triangles[node.triangleId()];
  const float3 a = make_float3(tri.v0.x, tri.v0.y, tri.v0.z);
  const float3 b = make_float3(tri.v1.x, tri.v1.y, tri.v1.z);
  const float3 c = make_float3(tri.v2.x, tri.v2.y, tri.v2.z);
  const float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
  const float3 ac = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
  const float3 areaVector = make_float3(
      0.5f * (ab.y * ac.z - ab.z * ac.y),
      0.5f * (ab.z * ac.x - ab.x * ac.z),
      0.5f * (ab.x * ac.y - ab.y * ac.x));

  uint32_t target = nodeIdx;
  while (target != InvalidNodeId) {
    atomicAdd(&moments[target].x, areaVector.x);
    atomicAdd(&moments[target].y, areaVector.y);
    atomicAdd(&moments[target].z, areaVector.z);
    target = parents[target];
  }
}

bool buildBvhMoments(DeviceModelGeometry& geometry, cudaStream_t stream) {
  if (!geometry.bvh.valid() || geometry.triangles.size() == 0) return false;

  const uint32_t nodeCount = static_cast<uint32_t>(geometry.bvh.nodes.size());
  if (!geometry.bvhMoments.allocate(nodeCount) ||
      !cudaUtils::cudaCheck(
          cudaMemsetAsync(geometry.bvhMoments.get(), 0, nodeCount * sizeof(float4), stream),
          "clear bvh moments")) {
    return false;
  }

  const uint32_t momentBlocks = (nodeCount + BlockSize - 1) / BlockSize;
  accumulateBvhMomentsKernel<<<momentBlocks, BlockSize, 0, stream>>>(
      geometry.triangles.get(),
      geometry.bvh.nodes.get(),
      nodeCount,
      geometry.bvh.parents.get(),
      geometry.bvhMoments.get());
  return cudaUtils::cudaCheck(cudaGetLastError(), "launch accumulateBvhMomentsKernel");
}

__global__ void buildWindingSolidMaskKernel(
    const DeviceModelGeometryView* views,
    uint32_t viewCount,
    float3 gridMin,
    int3 gridDim,
    float cellSize,
    uint8_t* outMask,
    uint32_t totalVoxels) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= totalVoxels) return;

  const int x = int(index % uint32_t(gridDim.x));
  const int y = int((index / uint32_t(gridDim.x)) % uint32_t(gridDim.y));
  const int z = int(index / uint32_t(gridDim.x * gridDim.y));
  const double3 point = make_double3(
      double(gridMin.x) + double(x) * double(cellSize),
      double(gridMin.y) + double(y) * double(cellSize),
      double(gridMin.z) + double(z) * double(cellSize));
  const float3 pointF = make_float3(
      float(point.x), float(point.y), float(point.z));

  uint8_t solid = 0;
  for (uint32_t viewIndex = 0; viewIndex < viewCount; ++viewIndex) {
    const DeviceModelGeometryView& view = views[viewIndex];
    if (view.moments == nullptr) continue;
    if (pointF.x < view.boundsMin.x || pointF.x > view.boundsMax.x ||
        pointF.y < view.boundsMin.y || pointF.y > view.boundsMax.y ||
        pointF.z < view.boundsMin.z || pointF.z > view.boundsMax.z) {
      continue;
    }
    const double winding = computeWinding(
        view.triangles,
        view.nodes,
        view.parents,
        view.root,
        view.moments,
        point,
        pointF);
    if (isfinite(winding) && fabs(winding) > 0.5) {
      solid = 1;
      break;
    }
  }
  outMask[index] = solid;
}

bool buildWindingSolidMask(
    const DeviceModelGeometryView* views,
    uint32_t viewCount,
    float3 gridMin,
    int3 gridDim,
    float cellSize,
    uint8_t* outMask,
    cudaStream_t stream) {
  if (views == nullptr || viewCount == 0 || outMask == nullptr ||
      gridDim.x <= 0 || gridDim.y <= 0 || gridDim.z <= 0 ||
      !(cellSize > 0.0f)) {
    return false;
  }

  const size_t totalVoxels = size_t(gridDim.x) * size_t(gridDim.y) * size_t(gridDim.z);
  if (totalVoxels == 0 || totalVoxels > UINT32_MAX) return false;

  cudaUtils::CudaBuffer<DeviceModelGeometryView> devViews;
  if (!devViews.allocate(viewCount) ||
      !devViews.uploadAsync(views, viewCount, stream)) {
    return false;
  }

  const uint32_t blocks = (uint32_t(totalVoxels) + BlockSize - 1) / BlockSize;
  buildWindingSolidMaskKernel<<<blocks, BlockSize, 0, stream>>>(
      devViews.get(),
      viewCount,
      gridMin,
      gridDim,
      cellSize,
      outMask,
      uint32_t(totalVoxels));
  return cudaUtils::cudaCheck(cudaGetLastError(), "launch buildWindingSolidMaskKernel");
}

bool uploadModelGeometry(
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>& indices,
    uint32_t runtimeModelId,
    uint32_t voxelResolution,
    DeviceModelGeometry& geometry,
    cudaStream_t stream) {
  geometry.reset();
  geometry.runtimeModelId = runtimeModelId;

  if (positions.empty() || indices.empty() || indices.size() % 3 != 0 || voxelResolution == 0) {
    return false;
  }

  const size_t triangleCount64 = indices.size() / 3;
  if (triangleCount64 > size_t(UINT32_MAX)) return false;
  const uint32_t triangleCount = static_cast<uint32_t>(triangleCount64);
  for (uint32_t index : indices) {
    if (size_t(index) >= positions.size()) return false;
  }

  glm::vec3 boundsMin;
  glm::vec3 boundsMax;
  if (!computePositionBounds(positions, boundsMin, boundsMax)) return false;

  static_assert(sizeof(glm::vec3) == sizeof(float3));
  cudaUtils::CudaBuffer<float3> devPositions;
  cudaUtils::CudaBuffer<uint32_t> devIndices;
  if (!devPositions.allocate(positions.size()) ||
      !devIndices.allocate(indices.size()) ||
      !devPositions.uploadAsync(
          reinterpret_cast<const float3*>(positions.data()),
          positions.size(), stream) ||
      !devIndices.uploadAsync(indices.data(), indices.size(), stream)) {
    return false;
  }

  if (!geometry.triangles.allocate(triangleCount) ||
      !geometry.triangleVertexIds.allocate(triangleCount)) return false;
  const uint32_t blocks = (triangleCount + BlockSize - 1) / BlockSize;
  expandTrianglesKernel<<<blocks, BlockSize, 0, stream>>>(
      devPositions.get(),
      static_cast<uint32_t>(positions.size()),
      devIndices.get(),
      triangleCount,
      geometry.triangles.get(),
      geometry.triangleVertexIds.get());
  if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch expandTrianglesKernel")) {
    geometry.reset();
    return false;
  }

  geometry.voxelGrid = makeRVDVoxelGrid(boundsMin, boundsMax, voxelResolution);
  if (!(geometry.voxelGrid.scale > 0.0f) || !(geometry.voxelGrid.voxelSize > 0.0f)) {
    geometry.reset();
    return false;
  }

  if (!::cudaGeometry::buildTriangleVoxelGrid(
          geometry.triangles.get(),
          triangleCount,
          geometry.voxelGrid,
          geometry.voxelOffsets,
          geometry.voxelTriangleIds,
          stream)) {
    geometry.reset();
    return false;
  }

  geometry.boundsMin = make_float3(boundsMin.x, boundsMin.y, boundsMin.z);
  geometry.boundsMax = make_float3(boundsMax.x, boundsMax.y, boundsMax.z);

  if (!buildTriangleBvh(geometry.triangles, geometry.bvh, stream) ||
      !buildBvhMoments(geometry, stream)) {
    geometry.reset();
    return false;
  }
  return true;
}

bool uploadGeometries(
    const RVDInput& input,
    std::vector<DeviceModelGeometry>& geometries,
    cudaStream_t stream) {
  geometries.clear();
  if (input.geometries == nullptr || input.geometries->empty()) return true;

  geometries.resize(input.geometries->size());
  for (size_t i = 0; i < input.geometries->size(); ++i) {
    const ModelGeometry &source = (*input.geometries)[i];
    if (source.positions == nullptr || source.indices == nullptr ||
        !uploadModelGeometry(
            *source.positions,
            *source.indices,
            source.runtimeModelId,
            source.voxelResolution,
            geometries[i],
            stream)) {
      std::cerr << "[RVD] failed to upload geometry model="
                << source.runtimeModelId << std::endl;
      return false;
    }
  }

  if (input.outDevSolidMask == nullptr) return true;
  std::vector<DeviceModelGeometryView> views;
  views.reserve(geometries.size());
  for (const DeviceModelGeometry &geometry : geometries) {
    views.push_back(geometry.view());
  }
  return buildWindingSolidMask(
      views.data(),
      uint32_t(views.size()),
      make_float3(
          input.solidMaskGridMin.x,
          input.solidMaskGridMin.y,
          input.solidMaskGridMin.z),
      make_int3(
          input.solidMaskGridDim.x,
          input.solidMaskGridDim.y,
          input.solidMaskGridDim.z),
      input.solidMaskCellSize,
      input.outDevSolidMask,
      stream);
}

} // namespace voronoi::cudaGeometry
