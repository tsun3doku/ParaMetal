#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "../../cuda/CudaBuffer.cuh"
#include "TriangleQueries.cuh"

namespace voronoi::cudaGeometry {

struct DeviceTriangle {
    float4 v0;
    float4 v1;
    float4 v2;
};

__device__ inline TriangleClosestPoint closestPointOnTriangle(float3 point, const DeviceTriangle& tri) {
    const float3 a = make_float3(tri.v0.x, tri.v0.y, tri.v0.z);
    const float3 b = make_float3(tri.v1.x, tri.v1.y, tri.v1.z);
    const float3 c = make_float3(tri.v2.x, tri.v2.y, tri.v2.z);
    return closestPointOnTriangle(point, a, b, c);
}

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

struct ClosestSurfaceHit {
    TriangleClosestPoint closest;
    uint32_t triangleId;
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

__device__ inline ClosestSurfaceHit closestPoint(
    float3 point,
    const DeviceTriangle* triangles,
    const TriangleBvhNode* nodes,
    const uint32_t* parents,
    uint32_t root)
{
    ClosestSurfaceHit bestHit{};
    bestHit.closest.distanceSquared = 1e30f;
    bestHit.triangleId = InvalidNodeId;

    if (root == InvalidNodeId || !nodes || !parents || !triangles) return bestHit;

    uint32_t curr = root;
    uint32_t prev = InvalidNodeId;

    while (curr != InvalidNodeId) {
        const TriangleBvhNode node = nodes[curr];
        const uint32_t parent = parents[curr];

        if (prev == parent) {
            // Descending from parent
            const float dBox = pointAabbDistanceSquared(point, node.minimum, node.maximum);
            if (dBox > bestHit.closest.distanceSquared) {
                // Prune subtree and ascend 
                prev = curr;
                curr = parent;
                continue;
            }

            if (node.isLeaf()) {
                const uint32_t triId = node.triangleId();
                const DeviceTriangle tri = triangles[triId];
                const TriangleClosestPoint cp = closestPointOnTriangle(point, tri);
                if (cp.distanceSquared < bestHit.closest.distanceSquared ||
                    (cp.distanceSquared == bestHit.closest.distanceSquared && triId < bestHit.triangleId)) {
                    bestHit.closest = cp;
                    bestHit.triangleId = triId;
                }
                prev = curr;
                curr = parent;
            } else {
                const uint32_t left = node.leftChild();
                const uint32_t right = node.rightChild();
                const float dL = pointAabbDistanceSquared(point, nodes[left].minimum, nodes[left].maximum);
                const float dR = pointAabbDistanceSquared(point, nodes[right].minimum, nodes[right].maximum);
                prev = curr;
                curr = (dL < dR) ? left : right; // Nearer child first
            }
        } else {
            // Ascending from a child
            const uint32_t left = node.leftChild();
            const uint32_t right = node.rightChild();
            const float dL = pointAabbDistanceSquared(point, nodes[left].minimum, nodes[left].maximum);
            const float dR = pointAabbDistanceSquared(point, nodes[right].minimum, nodes[right].maximum);
            const uint32_t nearer = (dL < dR) ? left : right;
            const uint32_t farther = (dL < dR) ? right : left;

            if (prev == nearer) {
                const float dFar = (dL < dR) ? dR : dL;
                if (dFar <= bestHit.closest.distanceSquared) {
                    prev = curr;
                    curr = farther;
                } else {
                    prev = curr;
                    curr = parent;
                }
            } else {
                // Ascending from farther child
                prev = curr;
                curr = parent;
            }
        }
    }

    return bestHit;
}

bool buildTriangleBvh(
    const cudaUtils::CudaBuffer<DeviceTriangle>& triangles,
    TriangleBvh& bvh,
    cudaStream_t stream);

} // namespace voronoi::cudaGeometry
