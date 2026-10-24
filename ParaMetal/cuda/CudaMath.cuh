#pragma once

#include <cuda_runtime.h>

namespace cudaMath {

__host__ __device__ inline double3 add3(double3 a, double3 b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline double3 sub3(double3 a, double3 b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline double3 mul3(double3 a, double s) {
    return make_double3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline double dot3(double3 a, double3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline double3 cross3(double3 a, double3 b) {
    return make_double3(a.y * b.z - a.z * b.y,
                        a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
}

__host__ __device__ inline double lengthSquared3(double3 a) {
    return dot3(a, a);
}

__host__ __device__ inline double length3(double3 a) {
    return sqrt(lengthSquared3(a));
}

__host__ __device__ inline double3 normalize3(double3 a) {
    const double length = length3(a);
    return length > 0.0 ? mul3(a, 1.0 / length) : make_double3(0.0, 0.0, 0.0);
}

__host__ __device__ inline float3 min3(float3 a, float3 b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__host__ __device__ inline float3 max3(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__host__ __device__ inline double3 toDouble3(float4 v) {
    return make_double3(double(v.x), double(v.y), double(v.z));
}

} // namespace cudaMath
