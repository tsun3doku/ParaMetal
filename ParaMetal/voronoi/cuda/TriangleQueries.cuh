#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace voronoi::cudaGeometry {

struct TriangleClosestPoint {
    float3 point;
    float distanceSquared;
};

__device__ __host__ inline float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline float3 mul3(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __host__ inline float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ inline float3 cross3(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ __host__ inline float len3(float3 a) {
    return sqrtf(dot3(a, a));
}

namespace detail {

__device__ inline TriangleClosestPoint hit(float3 p, float3 q) {
    const float3 d = sub3(p, q);
    return {q, dot3(d, d)};
}

} // namespace detail

__device__ inline TriangleClosestPoint closestPointOnTriangle(
    float3 point,
    float3 a,
    float3 b,
    float3 c)
{
    using namespace detail;

    const float3 ab = sub3(b, a);
    const float3 ac = sub3(c, a);
    const float3 ap = sub3(point, a);

    const float d1 = dot3(ab, ap);
    const float d2 = dot3(ac, ap);

    // Region A (vertex A)
    if (d1 <= 0.0f && d2 <= 0.0f) {
        return hit(point, a);
    }

    const float3 bp = sub3(point, b);
    const float d3 = dot3(ab, bp);
    const float d4 = dot3(ac, bp);

    // Region B (vertex B)
    if (d3 >= 0.0f && d4 <= d3) {
        return hit(point, b);
    }

    // Region AB (edge AB)
    const float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        return hit(point, add3(a, mul3(ab, v)));
    }

    const float3 cp = sub3(point, c);
    const float d5 = dot3(ab, cp);
    const float d6 = dot3(ac, cp);

    // Region C (vertex C)
    if (d6 >= 0.0f && d5 <= d6) {
        return hit(point, c);
    }

    // Region AC (edge AC)
    const float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        return hit(point, add3(a, mul3(ac, w)));
    }

    // Region BC (edge BC)
    const float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return hit(point, add3(b, mul3(sub3(c, b), w)));
    }

    // Region ABC (face interior)
    const float denom = 1.0f / (va + vb + vc);
    const float v = vb * denom;
    const float w = vc * denom;
    return hit(point, add3(a, add3(mul3(ab, v), mul3(ac, w))));
}

} // namespace voronoi::cudaGeometry
