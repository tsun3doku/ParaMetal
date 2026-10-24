#include "TriangleBvh.cuh"
#include "../../cuda/CudaMath.cuh"
#include <cub/cub.cuh>
#include <algorithm>

namespace voronoi::cudaGeometry {

using cudaMath::min3;
using cudaMath::max3;

static constexpr uint32_t BlockSize = 256;
using BlockReduce = cub::BlockReduce<float, BlockSize>;

static __device__ inline uint32_t expandBits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

static __device__ inline uint32_t morton3D(float3 p, float3 bMin, float3 bInvExt) {
    const float x = fminf(fmaxf((p.x - bMin.x) * bInvExt.x, 0.0f), 1.0f);
    const float y = fminf(fmaxf((p.y - bMin.y) * bInvExt.y, 0.0f), 1.0f);
    const float z = fminf(fmaxf((p.z - bMin.z) * bInvExt.z, 0.0f), 1.0f);

    const uint32_t ix = min(max(int(x * 1023.0f), 0), 1023);
    const uint32_t iy = min(max(int(y * 1023.0f), 0), 1023);
    const uint32_t iz = min(max(int(z * 1023.0f), 0), 1023);

    return (expandBits(ix) << 2) | (expandBits(iy) << 1) | expandBits(iz);
}

static __device__ float atomicMinFloat(float* address, float value)
{
    int* bits = reinterpret_cast<int*>(address);
    int old = *bits;

    while (value < __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(bits, assumed, __float_as_int(value));
        if (old == assumed)
            break;
    }

    return __int_as_float(old);
}

static __device__ float atomicMaxFloat(float* address, float value)
{
    int* bits = reinterpret_cast<int*>(address);
    int old = *bits;

    while (value > __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(bits, assumed, __float_as_int(value));
        if (old == assumed)
            break;
    }

    return __int_as_float(old);
}

static __global__ void computeCentroidBoundsKernel(
    const DeviceTriangle* triangles,
    uint32_t count,
    float3* outMin,
    float3* outMax)
{
    float3 bMin = make_float3( 1e30f,  1e30f,  1e30f);
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

static __global__ void computeMortonKeysKernel(
    const DeviceTriangle* triangles,
    uint32_t count,
    float3 bMin,
    float3 bInvExt,
    uint64_t* outKeys)
{
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

static __device__ inline int commonPrefix(
    const uint64_t* keys,
    int count,
    int a,
    int b)
{
    if (b < 0 || b >= count) return -1;
    return __clzll(keys[a] ^ keys[b]);
}

static __device__ inline int findSplit(
    const uint64_t* keys,
    int first,
    int last)
{
    const uint64_t firstKey = keys[first];
    const uint64_t lastKey  = keys[last];

    const int common = __clzll(firstKey ^ lastKey);

    int split = first;
    int step = last - first;

    do {
        step = (step + 1) >> 1;
        const int candidate = split + step;

        if (candidate < last) {
            const int prefix =
                __clzll(firstKey ^ keys[candidate]);

            if (prefix > common) {
                split = candidate;
            }
        }
    } while (step > 1);

    return split;
}

static __global__ void generateHierarchyKernel(
    const uint64_t* sortedKeys,
    uint32_t count,
    TriangleBvhNode* nodes,
    uint32_t* nodeParents)
{
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= int(count) - 1) return;

    const int deltaNext =
        commonPrefix(sortedKeys, int(count), i, i + 1);
    const int deltaPrev =
        commonPrefix(sortedKeys, int(count), i, i - 1);

    const int direction = deltaNext >= deltaPrev ? 1 : -1;
    const int deltaMin =
        commonPrefix(sortedKeys, int(count), i, i - direction);

    int lMax = 2;
    while (commonPrefix(
               sortedKeys,
               int(count),
               i,
               i + lMax * direction) > deltaMin)
    {
        lMax <<= 1;
    }

    int length = 0;
    for (int step = lMax >> 1; step > 0; step >>= 1) {
        if (commonPrefix(
                sortedKeys,
                int(count),
                i,
                i + (length + step) * direction) > deltaMin)
        {
            length += step;
        }
    }

    const int j = i + length * direction;

    const int first = min(i, j);
    const int last  = max(i, j);
    const int split = findSplit(sortedKeys, first, last);

    const uint32_t leafBase = count - 1;

    const uint32_t leftChild =
        split == first
            ? leafBase + uint32_t(split)
            : uint32_t(split);

    const uint32_t rightChild =
        split + 1 == last
            ? leafBase + uint32_t(split + 1)
            : uint32_t(split + 1);

    nodes[i].left  = leftChild;
    nodes[i].right = rightChild;

    nodeParents[leftChild]  = uint32_t(i);
    nodeParents[rightChild] = uint32_t(i);
}

static __global__ void initLeavesAndPropagateAabbKernel(
    const DeviceTriangle* triangles,
    const uint64_t* sortedKeys,
    uint32_t count,
    TriangleBvhNode* nodes,
    const uint32_t* nodeParents,
    int* nodeFlags)
{
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
        // Publish this child's AABB before announcing completion
        __threadfence();

        const int previous = atomicAdd(&nodeFlags[current], 1);
        if (previous == 0) {
            break;
        }

        const uint32_t left = nodes[current].leftChild();
        const uint32_t right = nodes[current].rightChild();

        nodes[current].minimum = min3(nodes[left].minimum, nodes[right].minimum);
        nodes[current].maximum = max3(nodes[left].maximum, nodes[right].maximum);

        current = nodeParents[current];
    }
}

static __global__ void initSingleLeafKernel(
    const DeviceTriangle* triangles,
    TriangleBvhNode* nodes,
    uint32_t* parents)
{
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

bool buildTriangleBvh(
    const cudaUtils::CudaBuffer<DeviceTriangle>& triangles,
    TriangleBvh& bvh,
    cudaStream_t stream)
{
    bvh.reset();
    const size_t triangleCount = triangles.size();
    if (triangleCount == 0)
        return true;

    if (triangleCount >= LeafBit)
        return false;

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
    const float3 initMin = make_float3( 1e30f,  1e30f,  1e30f);
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

} // namespace voronoi::cudaGeometry
