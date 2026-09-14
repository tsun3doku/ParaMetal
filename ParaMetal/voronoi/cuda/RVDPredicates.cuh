#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace voronoi::rvdDevice {

enum class PredicateMode : uint8_t {
    FilteredFloat,
    FilteredDouble
};

struct PredicateResult {
    double value = 0.0;
    double errorBound = 0.0;
};

__device__ inline PredicateResult determinant4x4FilteredFloat(float4 p0, float4 p1, float4 p2, float4 equation) {
    const float m12a = p1.x * p0.y, m12b = -p0.x * p1.y;
    const float m13a = p2.x * p0.y, m13b = -p0.x * p2.y;
    const float m14a = equation.x * p0.y, m14b = -p0.x * equation.y;
    const float m23a = p2.x * p1.y, m23b = -p1.x * p2.y;
    const float m24a = equation.x * p1.y, m24b = -p1.x * equation.y;
    const float m34a = equation.x * p2.y, m34b = -p2.x * equation.y;

    const float M12 = fabsf(m12a) + fabsf(m12b);
    const float M13 = fabsf(m13a) + fabsf(m13b);
    const float M14 = fabsf(m14a) + fabsf(m14b);
    const float M23 = fabsf(m23a) + fabsf(m23b);
    const float M24 = fabsf(m24a) + fabsf(m24b);
    const float M34 = fabsf(m34a) + fabsf(m34b);

    const float m12 = m12a + m12b, m13 = m13a + m13b, m14 = m14a + m14b;
    const float m23 = m23a + m23b, m24 = m24a + m24b, m34 = m34a + m34b;

    const float m123 = m23 * p0.z - m13 * p1.z + m12 * p2.z;
    const float m124 = m24 * p0.z - m14 * p1.z + m12 * equation.z;
    const float m134 = m34 * p0.z - m14 * p2.z + m13 * equation.z;
    const float m234 = m34 * p1.z - m24 * p2.z + m23 * equation.z;

    const float M123 = M23 * fabsf(p0.z) + M13 * fabsf(p1.z) + M12 * fabsf(p2.z);
    const float M124 = M24 * fabsf(p0.z) + M14 * fabsf(p1.z) + M12 * fabsf(equation.z);
    const float M134 = M34 * fabsf(p0.z) + M14 * fabsf(p2.z) + M13 * fabsf(equation.z);
    const float M234 = M34 * fabsf(p1.z) + M24 * fabsf(p2.z) + M23 * fabsf(equation.z);

    const float permanent = M234 * fabsf(p0.w) + M134 * fabsf(p1.w) + M124 * fabsf(p2.w) + M123 * fabsf(equation.w);
    const float det = m234 * p0.w - m134 * p1.w + m124 * p2.w - m123 * equation.w;

    // Unit roundoff eps_f = 2^-24 ~ 5.96e-8. Accumulated maximum relative error bound on permanent:
    constexpr double epsFloatBound = 3.5e-7;
    return { double(det), double(permanent) * epsFloatBound };
}

template <class T>
__device__ inline PredicateResult determinant4x4FilteredDouble(T p0, T p1, T p2, T equation) {
    const double p0x = double(p0.x), p0y = double(p0.y), p0z = double(p0.z), p0w = double(p0.w);
    const double p1x = double(p1.x), p1y = double(p1.y), p1z = double(p1.z), p1w = double(p1.w);
    const double p2x = double(p2.x), p2y = double(p2.y), p2z = double(p2.z), p2w = double(p2.w);
    const double eqx = double(equation.x), eqy = double(equation.y), eqz = double(equation.z), eqw = double(equation.w);

    const double m12a = p1x * p0y, m12b = -p0x * p1y;
    const double m13a = p2x * p0y, m13b = -p0x * p2y;
    const double m14a = eqx * p0y, m14b = -p0x * eqy;
    const double m23a = p2x * p1y, m23b = -p1x * p2y;
    const double m24a = eqx * p1y, m24b = -p1x * eqy;
    const double m34a = eqx * p2y, m34b = -p2x * eqy;

    const double M12 = fabs(m12a) + fabs(m12b);
    const double M13 = fabs(m13a) + fabs(m13b);
    const double M14 = fabs(m14a) + fabs(m14b);
    const double M23 = fabs(m23a) + fabs(m23b);
    const double M24 = fabs(m24a) + fabs(m24b);
    const double M34 = fabs(m34a) + fabs(m34b);

    const double m12 = m12a + m12b, m13 = m13a + m13b, m14 = m14a + m14b;
    const double m23 = m23a + m23b, m24 = m24a + m24b, m34 = m34a + m34b;

    const double m123 = m23 * p0z - m13 * p1z + m12 * p2z;
    const double m124 = m24 * p0z - m14 * p1z + m12 * eqz;
    const double m134 = m34 * p0z - m14 * p2z + m13 * eqz;
    const double m234 = m34 * p1z - m24 * p2z + m23 * eqz;

    const double M123 = M23 * fabs(p0z) + M13 * fabs(p1z) + M12 * fabs(p2z);
    const double M124 = M24 * fabs(p0z) + M14 * fabs(p1z) + M12 * fabs(eqz);
    const double M134 = M34 * fabs(p0z) + M14 * fabs(p2z) + M13 * fabs(eqz);
    const double M234 = M34 * fabs(p1z) + M24 * fabs(p2z) + M23 * fabs(eqz);

    const double permanent = M234 * fabs(p0w) + M134 * fabs(p1w) + M124 * fabs(p2w) + M123 * fabs(eqw);
    const double det = m234 * p0w - m134 * p1w + m124 * p2w - m123 * eqw;

    // IEEE 754 double unit roundoff eps_d = 2^-53 ~ 1.110223e-16.
    // Error bound: permanent * (23 * eps_d) ~ permanent * 2.55e-15, with safety factor: 3.5e-15
    constexpr double epsDoubleBound = 3.5e-15;
    return { det, permanent * epsDoubleBound };
}

} // namespace voronoi::rvdDevice
