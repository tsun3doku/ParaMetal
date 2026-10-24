#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace voronoi::rvdDevice {

struct PredicateResult {
    double value = 0.0;
    double errorBound = 0.0;
};

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

    constexpr double epsDoubleBound = 3.5e-15;
    return { det, permanent * epsDoubleBound };
}

__device__ inline void twoSum(double a, double b, double& x, double& y) {
    x = a + b;
    const double bVirtual = x - a;
    const double aVirtual = x - bVirtual;
    const double bRoundoff = b - bVirtual;
    const double aRoundoff = a - aVirtual;
    y = aRoundoff + bRoundoff;
}

__device__ inline void twoProduct(double a, double b, double& x, double& y) {
    x = a * b;
    y = fma(a, b, -x);
}

__device__ inline int twoTwoDiff(double a1, double a0, double b1, double b0, double* res) {
    double _i, _j;
    double _0, _1, _2, _3;
    twoSum(a0, -b0, _i, _0);
    twoSum(a1, _i, _j, _1);
    twoSum(_j, -b1, _3, _2);
    res[0] = _0;
    res[1] = _1;
    res[2] = _2;
    res[3] = _3;
    return 4;
}

__device__ inline int scaleExpansion(int elen, const double* e, double b, double* h) {
    if (elen <= 0) return 0;
    double bProduct, bRoundoff;
    twoProduct(e[0], b, bProduct, bRoundoff);
    h[0] = bRoundoff;
    double Q = bProduct;
    for (int i = 1; i < elen; ++i) {
        double pHigh, pLow;
        twoProduct(e[i], b, pHigh, pLow);
        double sumHigh, sumLow;
        twoSum(Q, pLow, sumHigh, sumLow);
        h[2 * i - 1] = sumLow;
        twoSum(pHigh, sumHigh, Q, h[2 * i]);
    }
    h[2 * elen - 1] = Q;
    return 2 * elen;
}

__device__ inline int expansionSum(int elen, const double* e, int flen, const double* f, double* h) {
    if (elen == 0) {
        for (int i = 0; i < flen; ++i) h[i] = f[i];
        return flen;
    }
    if (flen == 0) {
        for (int i = 0; i < elen; ++i) h[i] = e[i];
        return elen;
    }

    int eindex = 0, findex = 0, hindex = 0;
    double Q, Qnew, hh;
    double enow = e[0], fnow = f[0];

    if ((fnow > enow) == (fnow > -enow)) {
        Q = enow;
        eindex = 1;
    } else {
        Q = fnow;
        findex = 1;
    }

    if (eindex < elen && findex < flen) {
        enow = e[eindex];
        fnow = f[findex];
        if ((fnow > enow) == (fnow > -enow)) {
            twoSum(enow, Q, Qnew, hh);
            eindex++;
        } else {
            twoSum(fnow, Q, Qnew, hh);
            findex++;
        }
        Q = Qnew;
        h[hindex++] = hh;
        while (eindex < elen && findex < flen) {
            enow = e[eindex];
            fnow = f[findex];
            if ((fnow > enow) == (fnow > -enow)) {
                twoSum(Q, enow, Qnew, hh);
                eindex++;
            } else {
                twoSum(Q, fnow, Qnew, hh);
                findex++;
            }
            Q = Qnew;
            h[hindex++] = hh;
        }
    }

    while (eindex < elen) {
        enow = e[eindex++];
        twoSum(Q, enow, Qnew, hh);
        Q = Qnew;
        h[hindex++] = hh;
    }

    while (findex < flen) {
        fnow = f[findex++];
        twoSum(Q, fnow, Qnew, hh);
        Q = Qnew;
        h[hindex++] = hh;
    }

    h[hindex++] = Q;
    return hindex;
}

__device__ inline int expansionSign(int hlen, const double* h) {
    for (int i = hlen - 1; i >= 0; --i) {
        if (h[i] > 0.0) return 1;
        if (h[i] < 0.0) return -1;
    }
    return 0;
}

__device__ inline int det4x4ExactSign(double4 p0, double4 p1, double4 p2, double4 p3) {
    auto compute2x2 = [](double ax, double ay, double bx, double by, double* res) {
        double v0, v1, v2, v3;
        twoProduct(ax, by, v0, v1);
        twoProduct(bx, ay, v2, v3);
        twoTwoDiff(v0, v1, v2, v3, res);
    };

    double m01[4], m02[4], m03[4], m12[4], m13[4], m23[4];
    compute2x2(p0.x, p0.y, p1.x, p1.y, m01);
    compute2x2(p0.x, p0.y, p2.x, p2.y, m02);
    compute2x2(p0.x, p0.y, p3.x, p3.y, m03);
    compute2x2(p1.x, p1.y, p2.x, p2.y, m12);
    compute2x2(p1.x, p1.y, p3.x, p3.y, m13);
    compute2x2(p2.x, p2.y, p3.x, p3.y, m23);

    auto compute3x3 = [](const double* a, double za, const double* b, double zb, const double* c, double zc, double* out, int& outLen) {
        double t0[8], t1[8], t2[8];
        const int l0 = scaleExpansion(4, a, za, t0);
        const int l1 = scaleExpansion(4, b, -zb, t1);
        const int l2 = scaleExpansion(4, c, zc, t2);
        double s0[16];
        const int ls0 = expansionSum(l0, t0, l1, t1, s0);
        outLen = expansionSum(ls0, s0, l2, t2, out);
    };

    double m012[24], m013[24], m023[24], m123[24];
    int l012 = 0, l013 = 0, l023 = 0, l123 = 0;
    compute3x3(m12, p0.z, m02, p1.z, m01, p2.z, m012, l012);
    compute3x3(m13, p0.z, m03, p1.z, m01, p3.z, m013, l013);
    compute3x3(m23, p0.z, m03, p2.z, m02, p3.z, m023, l023);
    compute3x3(m23, p1.z, m13, p2.z, m12, p3.z, m123, l123);

    double t0[48], t1[48], t2[48], t3[48];
    const int lt0 = scaleExpansion(l123, m123, p0.w, t0);
    const int lt1 = scaleExpansion(l023, m023, -p1.w, t1);
    const int lt2 = scaleExpansion(l013, m013, p2.w, t2);
    const int lt3 = scaleExpansion(l012, m012, -p3.w, t3);

    double s0[96], s1[96], sFinal[192];
    const int ls0 = expansionSum(lt0, t0, lt1, t1, s0);
    const int ls1 = expansionSum(lt2, t2, lt3, t3, s1);
    const int lFinal = expansionSum(ls0, s0, ls1, s1, sFinal);

    return expansionSign(lFinal, sFinal);
}

} // namespace voronoi::rvdDevice
