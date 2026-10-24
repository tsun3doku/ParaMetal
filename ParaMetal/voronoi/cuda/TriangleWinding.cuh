#pragma once

#include "TriangleBvh.cuh"
#include "../../cuda/CudaMath.cuh"

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

namespace voronoi::cudaGeometry {

using cudaMath::toDouble3;
using cudaMath::sub3;
using cudaMath::cross3;
using cudaMath::dot3;

__device__ inline double triangleSolidAngle(double3 a, double3 b, double3 c) {
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
    float3 cornerF)
{
    double winding = 0.0;
    uint32_t curr = root;
    uint32_t prev = InvalidNodeId;

    while (curr != InvalidNodeId) {
        const TriangleBvhNode node = nodes[curr];

        if (prev == parents[curr]) {
            if (node.isLeaf()) {
                const DeviceTriangle tri = triangles[node.triangleId()];
                winding += triangleSolidAngle(
                    sub3(toDouble3(tri.v0), corner),
                    sub3(toDouble3(tri.v1), corner),
                    sub3(toDouble3(tri.v2), corner));
                prev = curr;
                curr = parents[curr];
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
                prev = curr;
                curr = parents[curr];
            } else {
                prev = curr;
                curr = node.leftChild();
            }
        } else if (prev == node.leftChild()) {
            prev = curr;
            curr = node.rightChild();
        } else {
            prev = curr;
            curr = parents[curr];
        }
    }

    return winding / (4.0 * 3.14159265358979323846);
}

} // namespace voronoi::cudaGeometry
