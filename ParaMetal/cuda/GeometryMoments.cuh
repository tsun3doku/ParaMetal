#pragma once

#include <cuda_runtime.h>

#include "CudaMath.cuh"

namespace cudaMoments {

constexpr uint32_t degreeLimit = 6;
constexpr uint32_t monomialLimit = 84;

__host__ __device__ constexpr uint32_t monomialStart(uint32_t degree) {
  return degree == 0u ? 0u : (degree * (degree + 1u) * (degree + 2u)) / 6u;
}

__host__ __device__ constexpr uint32_t monomialCount(uint32_t degree) {
  return monomialStart(degree + 1u);
}

struct Monomial {
  uint32_t x;
  uint32_t y;
  uint32_t z;
};

__host__ __device__ constexpr Monomial monomialAt(uint32_t index) {
  uint32_t degree = 0u;
  while (monomialStart(degree + 1u) <= index) ++degree;
  uint32_t offset = index - monomialStart(degree);
  for (uint32_t x = degree + 1u; x-- > 0u;) {
    const uint32_t row = degree - x + 1u;
    if (offset < row) return {x, degree - x - offset, offset};
    offset -= row;
  }
  return {0u, 0u, 0u};
}

__host__ __device__ constexpr uint32_t monomialDivisor(const Monomial &m) {
  return m.x + m.y + m.z + 3u;
}

__host__ __device__ constexpr uint32_t monomialIndex(uint32_t x, uint32_t y, uint32_t z) {
  const uint32_t degree = x + y + z;
  const uint32_t remaining = degree - x;
  return monomialStart(degree) + (remaining * (remaining + 1u)) / 2u + (remaining - y);
}

__host__ __device__ inline double binomialCoefficient(uint32_t n, uint32_t k) {
  if (k > n) return 0.0;
  double result = 1.0;
  for (uint32_t i = 1u; i <= k; ++i) {
    result = result * double(n - k + i) / double(i);
  }
  return result;
}

__host__ __device__ inline void translateMoments(
    const double *source, const double3 &offset, uint32_t degree, double *out) {
  const uint32_t total = monomialCount(degree);

  double powers[3][degreeLimit + 1u];
  powers[0][0] = powers[1][0] = powers[2][0] = 1.0;
  for (uint32_t p = 1u; p <= degree; ++p) {
    powers[0][p] = powers[0][p - 1u] * offset.x;
    powers[1][p] = powers[1][p - 1u] * offset.y;
    powers[2][p] = powers[2][p - 1u] * offset.z;
  }

  for (uint32_t target = 0; target < total; ++target) {
    const Monomial m = monomialAt(target);
    double sum = 0.0;
    for (uint32_t kx = 0; kx <= m.x; ++kx) {
      const double wx = binomialCoefficient(m.x, kx) * powers[0][m.x - kx];
      for (uint32_t ky = 0; ky <= m.y; ++ky) {
        const double wy = wx * binomialCoefficient(m.y, ky) * powers[1][m.y - ky];
        for (uint32_t kz = 0; kz <= m.z; ++kz) {
          sum += wy * binomialCoefficient(m.z, kz) * powers[2][m.z - kz] *
                 source[monomialIndex(kx, ky, kz)];
        }
      }
    }
    out[target] = sum;
  }
}

__host__ __device__ inline double faceMonomial(
    const double3 *points, uint32_t count, const double3 &origin,
    const double3 &normal, Monomial m) {
  if (count < 3u) return 0.0;

  constexpr double nodes[4] = {
      0.0694318442029737, 0.3300094782075720,
      0.6699905217924280, 0.9305681557970263};
  constexpr double weights[4] = {
      0.1739274225687269, 0.3260725774312731,
      0.3260725774312731, 0.1739274225687269};

  const double3 base = points[0];
  double sum = 0.0;
  for (uint32_t triangle = 1u; triangle + 1u < count; ++triangle) {
    const double3 edgeB = cudaMath::sub3(points[triangle], base);
    const double3 edgeC = cudaMath::sub3(points[triangle + 1u], base);
    const double twoArea =
        cudaMath::dot3(cudaMath::cross3(edgeB, edgeC), normal);
    if (twoArea == 0.0) continue;
    for (uint32_t sIndex = 0; sIndex < 4u; ++sIndex) {
      const double s = nodes[sIndex];
      for (uint32_t tIndex = 0; tIndex < 4u; ++tIndex) {
        const double t = nodes[tIndex];
        const double weight = weights[sIndex] * weights[tIndex] * s * twoArea;
        const double3 offset = cudaMath::add3(
            cudaMath::mul3(edgeB, s * (1.0 - t)), cudaMath::mul3(edgeC, s * t));
        const double3 d = cudaMath::sub3(cudaMath::add3(base, offset), origin);
        double px = 1.0;
        double py = 1.0;
        double pz = 1.0;
        for (uint32_t p = 0; p < m.x; ++p) px *= d.x;
        for (uint32_t p = 0; p < m.y; ++p) py *= d.y;
        for (uint32_t p = 0; p < m.z; ++p) pz *= d.z;
        sum += weight * px * py * pz;
      }
    }
  }
  return sum;
}

} // namespace cudaMoments
