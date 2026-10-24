#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "../../cuda/CudaBuffer.cuh"

namespace voronoi::cudaGeometry {

struct DeviceTriangle {
    float4 v0;
    float4 v1;
    float4 v2;
};

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
    float d2 = 0.0f;
    if (p.x < bMin.x) { float d = bMin.x - p.x; d2 += d * d; }
    else if (p.x > bMax.x) { float d = p.x - bMax.x; d2 += d * d; }

    if (p.y < bMin.y) { float d = bMin.y - p.y; d2 += d * d; }
    else if (p.y > bMax.y) { float d = p.y - bMax.y; d2 += d * d; }

    if (p.z < bMin.z) { float d = bMin.z - p.z; d2 += d * d; }
    else if (p.z > bMax.z) { float d = p.z - bMax.z; d2 += d * d; }

    return d2;
}

__host__ __device__ inline float aabbDistanceSquared(float3 minA, float3 maxA, float3 minB, float3 maxB) {
    float d2 = 0.0f;
    if (maxA.x < minB.x) { float d = minB.x - maxA.x; d2 += d * d; }
    else if (maxB.x < minA.x) { float d = minA.x - maxB.x; d2 += d * d; }

    if (maxA.y < minB.y) { float d = minB.y - maxA.y; d2 += d * d; }
    else if (maxB.y < minA.y) { float d = minA.y - maxB.y; d2 += d * d; }

    if (maxA.z < minB.z) { float d = minB.z - maxA.z; d2 += d * d; }
    else if (maxB.z < minA.z) { float d = minA.z - maxB.z; d2 += d * d; }

    return d2;
}

bool buildTriangleBvh(
    const cudaUtils::CudaBuffer<DeviceTriangle>& triangles,
    TriangleBvh& bvh,
    cudaStream_t stream);

} // namespace voronoi::cudaGeometry
