#include "../CFDReconstruction.hpp"

#include "../../cuda/CudaCheck.cuh"
#include "../../cuda/CudaGraphBuilder.cuh"
#include "../../cuda/CudaMath.cuh"
#include "../../cuda/DomainMoments.cuh"
#include "../../cuda/GeometryMoments.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>

static_assert(sizeof(glm::dvec3) == sizeof(double3), "glm::dvec3 must pack as double3");

namespace cfd {

namespace {

constexpr uint32_t BlockSize = 128u;
constexpr uint32_t InvalidIndex = 0xffffffffu;
constexpr uint32_t CentralPoolLimit = 80u;
constexpr uint32_t CentralUnsortedCount = 12u;
constexpr uint32_t DirectionalPoolLimit = 256u;
constexpr uint32_t MaxFaceDirections = 32u;
constexpr uint32_t MaxSelectedFaceDirections = 12u;
constexpr uint32_t MaxFaceVertices = 96u;
constexpr uint32_t MaxHullVertices = MaxFaceVertices * 2u;
constexpr uint32_t CentralHashSize = 128u;
constexpr uint32_t DirectionalHashSize = 512u;
constexpr double CentralWeightPrime = 1e4;
constexpr double SmoothnessEpsilon = 1e-6;
constexpr double EnoThreshold = 1e-6;
constexpr double DirectionalConditionLimit = 1.0e3;
constexpr uint32_t CoverageHistogramBins = ReconDirectionalCandidateLimit + 1u;

enum DiagnosticSlot : uint32_t {
    CentralValid = 0u,
    CentralInvalid = 1u,
    DirectionalCandidates = 2u,
    DirectionalCellsZero = 3u,
    DirectionalInvalid = 4u,
    FaceInvalid = 5u,
    LoopOverflow = 6u,
    CellInvalid = 7u,
    CoefficientInvalid = 8u,
    FaceAreaMismatch = 9u,
    CellGeometryMismatch = 10u,
    FaceOrientationMismatch = 11u,
    DirectionalRankDeficient = 12u,
    DirectionalIllConditioned = 13u,
    CoverageMinimum = 14u,
    CoverageMaximum = 15u,
    CoverageHistogram = 16u,
    DiagnosticCount = CoverageHistogram + CoverageHistogramBins,
};

__device__ inline bool validNode(uint32_t node) {
    return node != InvalidIndex;
}

__host__ __device__ inline double3 zero3() {
    return make_double3(0.0, 0.0, 0.0);
}

__device__ inline bool insertUnique(
    uint32_t* hash, uint32_t mask, uint32_t value) {
    uint32_t slot = value & mask;
    while (hash[slot] != InvalidIndex) {
        if (hash[slot] == value) return false;
        slot = (slot + 1u) & mask;
    }
    hash[slot] = value;
    return true;
}

__device__ inline void clearHash(uint32_t* hash, uint32_t count) {
    for (uint32_t i = 0u; i < count; ++i) hash[i] = InvalidIndex;
}

__global__ void countIncidenceKernel(
    uint32_t faceCount,
    const uint32_t* __restrict__ nodeA,
    const uint32_t* __restrict__ nodeB,
    const double3* __restrict__ normals,
    uint32_t* __restrict__ counts) {
    const uint32_t face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= faceCount) return;
    if (cudaMath::lengthSquared3(normals[face]) <= 0.0) return;
    const uint32_t a = nodeA[face];
    const uint32_t b = nodeB[face];
    if (validNode(a)) atomicAdd(&counts[a], 1u);
    if (validNode(b) && b != a) atomicAdd(&counts[b], 1u);
}

__global__ void fillIncidenceKernel(
    uint32_t faceCount,
    const uint32_t* __restrict__ nodeA,
    const uint32_t* __restrict__ nodeB,
    const double3* __restrict__ normals,
    uint32_t* __restrict__ cursor,
    uint32_t* __restrict__ incidenceFaces,
    double* __restrict__ incidenceSides) {
    const uint32_t face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= faceCount) return;
    if (cudaMath::lengthSquared3(normals[face]) <= 0.0) return;
    const uint32_t a = nodeA[face];
    const uint32_t b = nodeB[face];
    if (validNode(a)) {
        const uint32_t slot = atomicAdd(&cursor[a], 1u);
        incidenceFaces[slot] = face;
        incidenceSides[slot] = 1.0;
    }
    if (validNode(b) && b != a) {
        const uint32_t slot = atomicAdd(&cursor[b], 1u);
        incidenceFaces[slot] = face;
        incidenceSides[slot] = -1.0;
    }
}

__global__ void countRingKernel(
    uint32_t faceCount,
    const uint32_t* __restrict__ nodeA,
    const uint32_t* __restrict__ nodeB,
    const double3* __restrict__ normals,
    uint32_t* __restrict__ counts) {
    const uint32_t face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= faceCount) return;
    if (cudaMath::lengthSquared3(normals[face]) <= 0.0) return;
    const uint32_t a = nodeA[face];
    const uint32_t b = nodeB[face];
    if (validNode(a) && validNode(b) && a != b) {
        atomicAdd(&counts[a], 1u);
        atomicAdd(&counts[b], 1u);
    }
}

__global__ void fillRingKernel(
    uint32_t faceCount,
    const uint32_t* __restrict__ nodeA,
    const uint32_t* __restrict__ nodeB,
    const double3* __restrict__ normals,
    uint32_t* __restrict__ cursor,
    uint32_t* __restrict__ ring) {
    const uint32_t face = blockIdx.x * blockDim.x + threadIdx.x;
    if (face >= faceCount) return;
    if (cudaMath::lengthSquared3(normals[face]) <= 0.0) return;
    const uint32_t a = nodeA[face];
    const uint32_t b = nodeB[face];
    if (validNode(a) && validNode(b) && a != b) {
        ring[atomicAdd(&cursor[a], 1u)] = b;
        ring[atomicAdd(&cursor[b], 1u)] = a;
    }
}

__device__ inline double cross2(
    const double2& a, const double2& b, const double2& c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

__device__ inline void faceMomentsAll(
    const double3* points,
    uint32_t count,
    const double3& origin,
    const double3& normal,
    double* out) {
    for (uint32_t i = 0u; i < 10u; ++i) out[i] = 0.0;
    if (count < 3u) return;

    constexpr double nodes[4] = {
        0.0694318442029737, 0.3300094782075720,
        0.6699905217924280, 0.9305681557970263};
    constexpr double weights[4] = {
        0.1739274225687269, 0.3260725774312731,
        0.3260725774312731, 0.1739274225687269};

    const double3 base = points[0];
    for (uint32_t triangle = 1u; triangle + 1u < count; ++triangle) {
        const double3 edgeB = cudaMath::sub3(points[triangle], base);
        const double3 edgeC = cudaMath::sub3(points[triangle + 1u], base);
        const double twoArea =
            cudaMath::dot3(cudaMath::cross3(edgeB, edgeC), normal);
        if (twoArea == 0.0) continue;
        for (uint32_t sIndex = 0u; sIndex < 4u; ++sIndex) {
            const double s = nodes[sIndex];
            for (uint32_t tIndex = 0u; tIndex < 4u; ++tIndex) {
                const double t = nodes[tIndex];
                const double weight = weights[sIndex] * weights[tIndex] * s * twoArea;
                const double3 offset = cudaMath::add3(
                    cudaMath::mul3(edgeB, s * (1.0 - t)),
                    cudaMath::mul3(edgeC, s * t));
                const double3 d = cudaMath::sub3(
                    cudaMath::add3(base, offset), origin);
                out[0] += weight;
                out[1] += weight * d.x;
                out[2] += weight * d.y;
                out[3] += weight * d.z;
                out[4] += weight * d.x * d.x;
                out[5] += weight * d.x * d.y;
                out[6] += weight * d.x * d.z;
                out[7] += weight * d.y * d.y;
                out[8] += weight * d.y * d.z;
                out[9] += weight * d.z * d.z;
            }
        }
    }
}

__device__ inline double faceSignedArea(
    const double3* points,
    uint32_t count,
    const double3& normal) {
    if (count < 3u) return 0.0;
    const double3 base = points[0];
    double area = 0.0;
    for (uint32_t triangle = 1u; triangle + 1u < count; ++triangle) {
        const double3 edgeB = cudaMath::sub3(points[triangle], base);
        const double3 edgeC = cudaMath::sub3(points[triangle + 1u], base);
        area += 0.5 * cudaMath::dot3(cudaMath::cross3(edgeB, edgeC), normal);
    }
    return area;
}

__global__ void faceGeometryKernel(
    uint32_t faceCount,
    uint32_t domainFaceCount,
    const uint32_t* __restrict__ faceGlobalIndex,
    const uint32_t* __restrict__ faceLoopOffsets,
    const uint32_t* __restrict__ loopVertexOffsets,
    const double3* __restrict__ loopVertices,
    const double3* __restrict__ faceNormals,
    const double* __restrict__ expectedAreas,
    ReconFaceGeometry* __restrict__ faceGeometry,
    uint32_t* __restrict__ diagnostics) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= faceCount) return;

    ReconFaceGeometry result{};
    const uint32_t globalFace = faceGlobalIndex[idx];
    if (globalFace >= domainFaceCount) {
        faceGeometry[idx] = result;
        return;
    }

    const uint32_t loopBegin = faceLoopOffsets[globalFace];
    const uint32_t loopEnd = faceLoopOffsets[globalFace + 1u];
    if (loopBegin >= loopEnd) {
        faceGeometry[idx] = result;
        return;
    }

    const double3 normal = faceNormals[globalFace];
    if (cudaMath::lengthSquared3(normal) <= 0.0) {
        faceGeometry[idx] = result;
        return;
    }

    uint32_t referenceLoop = loopBegin;
    while (referenceLoop < loopEnd) {
        const uint32_t begin = loopVertexOffsets[referenceLoop];
        const uint32_t end = loopVertexOffsets[referenceLoop + 1u];
        if (end - begin >= 3u) break;
        ++referenceLoop;
    }
    if (referenceLoop >= loopEnd) {
        faceGeometry[idx] = result;
        return;
    }

    const double3 origin = loopVertices[loopVertexOffsets[referenceLoop]];
    double total[10] = {};

    for (uint32_t loop = loopBegin; loop < loopEnd; ++loop) {
        const uint32_t begin = loopVertexOffsets[loop];
        const uint32_t end = loopVertexOffsets[loop + 1u];
        if (end - begin < 3u) continue;
        double loopMoments[10];
        faceMomentsAll(loopVertices + begin, end - begin, origin, normal, loopMoments);
        for (uint32_t i = 0u; i < 10u; ++i) total[i] += loopMoments[i];
    }

    if (total[0] < 0.0) {
        atomicAdd(&diagnostics[FaceOrientationMismatch], 1u);
        for (uint32_t i = 0u; i < 10u; ++i) total[i] = -total[i];
    }

    if (!(isfinite(total[0])) || total[0] <= 0.0) {
        atomicAdd(&diagnostics[FaceInvalid], 1u);
        faceGeometry[idx] = result;
        return;
    }

    const double inverseArea = 1.0 / total[0];
    const double3 centroid = make_double3(
        origin.x + total[1] * inverseArea,
        origin.y + total[2] * inverseArea,
        origin.z + total[3] * inverseArea);

    const double mxx = total[4] - total[1] * total[1] * inverseArea;
    const double mxy = total[5] - total[1] * total[2] * inverseArea;
    const double mxz = total[6] - total[1] * total[3] * inverseArea;
    const double myy = total[7] - total[2] * total[2] * inverseArea;
    const double myz = total[8] - total[2] * total[3] * inverseArea;
    const double mzz = total[9] - total[3] * total[3] * inverseArea;

    result.centroid = centroid;
    result.area = total[0];
    result.m2[0] = mxx * inverseArea;
    result.m2[1] = mxy * inverseArea;
    result.m2[2] = mxz * inverseArea;
    result.m2[3] = myy * inverseArea;
    result.m2[4] = myz * inverseArea;
    result.m2[5] = mzz * inverseArea;
    const double expectedArea = expectedAreas[idx];
    if (!(expectedArea > 0.0) ||
        fabs(result.area - expectedArea) > 1e-6 * fmax(expectedArea, 1e-12)) {
        atomicAdd(&diagnostics[FaceAreaMismatch], 1u);
    }
    faceGeometry[idx] = result;
}

__global__ void cellMomentsKernel(
    uint32_t cellCount,
    const uint32_t* __restrict__ incidenceOffsets,
    const uint32_t* __restrict__ incidenceFaces,
    const double* __restrict__ incidenceSides,
    const double3* __restrict__ cellCentroids,
    const uint32_t* __restrict__ faceLoopOffsets,
    const uint32_t* __restrict__ loopVertexOffsets,
    const double3* __restrict__ loopVertices,
    const double3* __restrict__ faceNormals,
    double* __restrict__ moments) {
    const uint32_t cell = blockIdx.x;
    const uint32_t slot = threadIdx.x;
    if (cell >= cellCount || slot >= 10u) return;

    const cudaMoments::Monomial monomial = cudaMoments::monomialAt(slot);
    const double3 origin = cellCentroids[cell];
    double total = 0.0;

    for (uint32_t k = incidenceOffsets[cell]; k < incidenceOffsets[cell + 1u]; ++k) {
        const uint32_t face = incidenceFaces[k];
        const double3 normal = faceNormals[face];
        if (cudaMath::lengthSquared3(normal) <= 0.0) continue;
        const uint32_t loopBegin = faceLoopOffsets[face];
        const uint32_t loopEnd = faceLoopOffsets[face + 1u];
        if (loopBegin >= loopEnd) continue;
        const uint32_t anchorIndex = loopVertexOffsets[loopBegin];
        if (anchorIndex >= loopVertexOffsets[loopBegin + 1u]) continue;
        const double3 anchor = loopVertices[anchorIndex];
        const double side = incidenceSides[k];
        const double offset =
            side * cudaMath::dot3(cudaMath::sub3(anchor, origin), normal);
        double signedArea = 0.0;
        for (uint32_t loop = loopBegin; loop < loopEnd; ++loop) {
            const uint32_t vertexBegin = loopVertexOffsets[loop];
            const uint32_t vertexEnd = loopVertexOffsets[loop + 1u];
            if (vertexEnd - vertexBegin < 3u) continue;
            signedArea += faceSignedArea(
                loopVertices + vertexBegin, vertexEnd - vertexBegin, normal);
        }
        const double faceSign = signedArea < 0.0 ? -1.0 : 1.0;
        for (uint32_t loop = loopBegin; loop < loopEnd; ++loop) {
            const uint32_t vertexBegin = loopVertexOffsets[loop];
            const uint32_t vertexEnd = loopVertexOffsets[loop + 1u];
            if (vertexEnd - vertexBegin < 3u) continue;
            total += faceSign * offset * cudaMoments::faceMonomial(
                loopVertices + vertexBegin, vertexEnd - vertexBegin,
                origin, normal, monomial);
        }
    }

    moments[size_t(cell) * 10u + slot] =
        total / double(cudaMoments::monomialDivisor(monomial));
}

__global__ void finalizeCellGeometryKernel(
    uint32_t cellCount,
    const double3* __restrict__ cellCentroids,
    const double* __restrict__ expectedVolumes,
    const double* __restrict__ moments,
    ReconCellGeometry* __restrict__ geometry,
    uint32_t* __restrict__ diagnostics) {
    const uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cellCount) return;

    ReconCellGeometry result{};
    result.centroid = cellCentroids[cell];
    result.volume = moments[size_t(cell) * 10u];
    if (!(isfinite(result.volume)) || result.volume <= 0.0) {
        atomicAdd(&diagnostics[CellInvalid], 1u);
        geometry[cell] = result;
        return;
    }

    const size_t momentBase = size_t(cell) * 10u;
    const double expectedVolume = expectedVolumes[cell];
    const double toleranceVolume = 1e-6 * fmax(expectedVolume, 1e-12);
    const double3 computedCentroid = make_double3(
        result.centroid.x + moments[momentBase + 1u] / result.volume,
        result.centroid.y + moments[momentBase + 2u] / result.volume,
        result.centroid.z + moments[momentBase + 3u] / result.volume);
    const double toleranceCentroid = 1e-6 * cbrt(fmax(expectedVolume, 1e-12));
    const double3 centroidError = cudaMath::sub3(computedCentroid, result.centroid);
    if (!(expectedVolume > 0.0) ||
        fabs(result.volume - expectedVolume) > toleranceVolume ||
        cudaMath::lengthSquared3(centroidError) > toleranceCentroid * toleranceCentroid) {
        atomicAdd(&diagnostics[CellGeometryMismatch], 1u);
    }

    result.s[0] = moments[size_t(cell) * 10u + 4u];
    result.s[1] = moments[size_t(cell) * 10u + 5u];
    result.s[2] = moments[size_t(cell) * 10u + 6u];
    result.s[3] = moments[size_t(cell) * 10u + 7u];
    result.s[4] = moments[size_t(cell) * 10u + 8u];
    result.s[5] = moments[size_t(cell) * 10u + 9u];
    const double inverseVolume = 1.0 / result.volume;
    for (uint32_t i = 0u; i < 6u; ++i) {
        result.m2[i] = result.s[i] * inverseVolume;
    }
    result.h = cbrt(result.volume);
    if (!(result.h > 0.0) || !isfinite(result.h)) {
        atomicAdd(&diagnostics[CellInvalid], 1u);
        geometry[cell] = ReconCellGeometry{};
        return;
    }
    const double inverseHSquared = 1.0 / (result.h * result.h);
    for (uint32_t i = 0u; i < 6u; ++i) {
        result.q[i] = result.m2[i] * inverseHSquared;
    }
    geometry[cell] = result;
}

__device__ inline bool cholesky9(
    const double input[9][9], double lower[9][9], double& minDiag, double& maxDiag) {
    minDiag = std::numeric_limits<double>::max();
    maxDiag = 0.0;
    for (uint32_t i = 0u; i < 9u; ++i) {
        for (uint32_t j = 0u; j <= i; ++j) {
            double sum = input[i][j];
            for (uint32_t k = 0u; k < j; ++k) {
                sum -= lower[i][k] * lower[j][k];
            }
            if (i == j) {
                if (!(isfinite(sum)) || sum <= 0.0) return false;
                lower[i][i] = sqrt(sum);
                minDiag = fmin(minDiag, lower[i][i]);
                maxDiag = fmax(maxDiag, lower[i][i]);
            } else {
                if (!(isfinite(sum)) || lower[j][j] == 0.0) return false;
                lower[i][j] = sum / lower[j][j];
            }
        }
    }
    return isfinite(minDiag) && isfinite(maxDiag) && maxDiag > 0.0;
}

__device__ inline void solve9(
    const double lower[9][9], const double rhs[9], double out[9]) {
    double y[9];
    for (uint32_t i = 0u; i < 9u; ++i) {
        double sum = rhs[i];
        for (uint32_t k = 0u; k < i; ++k) sum -= lower[i][k] * y[k];
        y[i] = sum / lower[i][i];
    }
    for (uint32_t i = 9u; i-- > 0u;) {
        double sum = y[i];
        for (uint32_t k = i + 1u; k < 9u; ++k) sum -= lower[k][i] * out[k];
        out[i] = sum / lower[i][i];
    }
}

__device__ inline bool cholesky3(
    const double input[3][3], double lower[3][3], double& minDiag, double& maxDiag) {
    minDiag = std::numeric_limits<double>::max();
    maxDiag = 0.0;
    for (uint32_t i = 0u; i < 3u; ++i) {
        for (uint32_t j = 0u; j <= i; ++j) {
            double sum = input[i][j];
            for (uint32_t k = 0u; k < j; ++k) {
                sum -= lower[i][k] * lower[j][k];
            }
            if (i == j) {
                if (!(isfinite(sum)) || sum <= 0.0) return false;
                lower[i][i] = sqrt(sum);
                minDiag = fmin(minDiag, lower[i][i]);
                maxDiag = fmax(maxDiag, lower[i][i]);
            } else {
                if (!(isfinite(sum)) || lower[j][j] == 0.0) return false;
                lower[i][j] = sum / lower[j][j];
            }
        }
    }
    return isfinite(minDiag) && isfinite(maxDiag) && maxDiag > 0.0;
}

__device__ inline bool symmetricEigenExtremes3(
    const double input[3][3], double& minimum, double& maximum) {
    const double p1 = input[0][1] * input[0][1] + input[0][2] * input[0][2] +
                      input[1][2] * input[1][2];
    const double q = (input[0][0] + input[1][1] + input[2][2]) / 3.0;
    if (p1 == 0.0) {
        minimum = fmin(input[0][0], fmin(input[1][1], input[2][2]));
        maximum = fmax(input[0][0], fmax(input[1][1], input[2][2]));
        return isfinite(minimum) && isfinite(maximum) && maximum > 0.0;
    }
    const double p2 = (input[0][0] - q) * (input[0][0] - q) +
                      (input[1][1] - q) * (input[1][1] - q) +
                      (input[2][2] - q) * (input[2][2] - q) + 2.0 * p1;
    const double p = sqrt(p2 / 6.0);
    if (!(p > 0.0) || !isfinite(p)) return false;

    double normalized[3][3];
    for (uint32_t i = 0u; i < 3u; ++i) {
        for (uint32_t j = 0u; j < 3u; ++j) {
            normalized[i][j] = (input[i][j] - (i == j ? q : 0.0)) / p;
        }
    }
    const double determinant =
        normalized[0][0] * (normalized[1][1] * normalized[2][2] -
                            normalized[1][2] * normalized[2][1]) -
        normalized[0][1] * (normalized[1][0] * normalized[2][2] -
                            normalized[1][2] * normalized[2][0]) +
        normalized[0][2] * (normalized[1][0] * normalized[2][1] -
                            normalized[1][1] * normalized[2][0]);
    const double r = fmin(1.0, fmax(-1.0, 0.5 * determinant));
    const double phi = acos(r) / 3.0;
    maximum = q + 2.0 * p * cos(phi);
    minimum = q + 2.0 * p * cos(phi + 2.0943951023931953);
    return isfinite(minimum) && isfinite(maximum) && maximum > 0.0;
}

__device__ inline void solve3(
    const double lower[3][3], const double rhs[3], double out[3]) {
    double y[3];
    for (uint32_t i = 0u; i < 3u; ++i) {
        double sum = rhs[i];
        for (uint32_t k = 0u; k < i; ++k) sum -= lower[i][k] * y[k];
        y[i] = sum / lower[i][i];
    }
    for (uint32_t i = 3u; i-- > 0u;) {
        double sum = y[i];
        for (uint32_t k = i + 1u; k < 3u; ++k) sum -= lower[k][i] * out[k];
        out[i] = sum / lower[i][i];
    }
}

__global__ void buildCentralStencilKernel(
    uint32_t cellCount,
    const uint32_t* __restrict__ ringOffsets,
    const uint32_t* __restrict__ ring,
    const ReconCellGeometry* __restrict__ geometry,
    uint32_t* __restrict__ counts,
    uint32_t* __restrict__ neighbors,
    float* __restrict__ ops,
    uint32_t* __restrict__ diagnostics) {
    const uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cellCount) return;

    const ReconCellGeometry target = geometry[cell];
    if (!(target.volume > 0.0) || !(target.h > 0.0)) {
        counts[cell] = 0u;
        atomicAdd(&diagnostics[CentralInvalid], 1u);
        return;
    }

    uint32_t pool[CentralPoolLimit];
    uint32_t hash[CentralHashSize];
    clearHash(hash, CentralHashSize);
    uint32_t poolCount = 0u;

    for (uint32_t k = ringOffsets[cell]; k < ringOffsets[cell + 1u]; ++k) {
        const uint32_t neighbor = ring[k];
        if (neighbor == cell || neighbor >= cellCount) continue;
        if (!insertUnique(hash, CentralHashSize - 1u, neighbor)) continue;
        pool[poolCount++] = neighbor;
        if (poolCount == CentralPoolLimit) break;
    }
    for (uint32_t head = 0u; head < poolCount && poolCount < CentralPoolLimit; ++head) {
        const uint32_t current = pool[head];
        for (uint32_t k = ringOffsets[current]; k < ringOffsets[current + 1u]; ++k) {
            const uint32_t neighbor = ring[k];
            if (neighbor == cell || neighbor >= cellCount) continue;
            if (!insertUnique(hash, CentralHashSize - 1u, neighbor)) continue;
            pool[poolCount++] = neighbor;
            if (poolCount == CentralPoolLimit) break;
        }
    }

    const uint32_t count = min(ReconCentralNeighborLimit, poolCount);
    if (count < ReconQuadraticUnknownCount) {
        counts[cell] = 0u;
        atomicAdd(&diagnostics[CentralInvalid], 1u);
        return;
    }

    uint32_t selected[ReconCentralNeighborLimit];
    bool used[CentralPoolLimit] = {};
    const uint32_t unsorted = min(CentralUnsortedCount, count);
    for (uint32_t i = 0u; i < unsorted; ++i) {
        selected[i] = pool[i];
        used[i] = true;
    }
    for (uint32_t slot = unsorted; slot < count; ++slot) {
        uint32_t best = InvalidIndex;
        double bestDistance = std::numeric_limits<double>::max();
        for (uint32_t i = unsorted; i < poolCount; ++i) {
            if (used[i]) continue;
            const double3 delta = cudaMath::sub3(geometry[pool[i]].centroid, target.centroid);
            const double distance = cudaMath::lengthSquared3(delta);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = i;
            }
        }
        if (best == InvalidIndex) break;
        selected[slot] = pool[best];
        used[best] = true;
    }

    double rows[ReconCentralNeighborLimit][ReconQuadraticUnknownCount];
    const double inverseH = 1.0 / target.h;
    const double inverseHSquared = inverseH * inverseH;
    for (uint32_t slot = 0u; slot < count; ++slot) {
        const ReconCellGeometry neighbor = geometry[selected[slot]];
        const double3 delta = cudaMath::sub3(neighbor.centroid, target.centroid);
        const double inverseVolume = (neighbor.volume > 0.0)
            ? 1.0 / neighbor.volume : 0.0;
        rows[slot][0] = delta.x * inverseH;
        rows[slot][1] = delta.y * inverseH;
        rows[slot][2] = delta.z * inverseH;
        rows[slot][3] = ((neighbor.s[0] * inverseVolume + delta.x * delta.x) -
                         target.m2[0]) * inverseHSquared;
        rows[slot][4] = ((neighbor.s[1] * inverseVolume + delta.x * delta.y) -
                         target.m2[1]) * inverseHSquared;
        rows[slot][5] = ((neighbor.s[2] * inverseVolume + delta.x * delta.z) -
                         target.m2[2]) * inverseHSquared;
        rows[slot][6] = ((neighbor.s[3] * inverseVolume + delta.y * delta.y) -
                         target.m2[3]) * inverseHSquared;
        rows[slot][7] = ((neighbor.s[4] * inverseVolume + delta.y * delta.z) -
                         target.m2[4]) * inverseHSquared;
        rows[slot][8] = ((neighbor.s[5] * inverseVolume + delta.z * delta.z) -
                         target.m2[5]) * inverseHSquared;
    }

    double normal[9][9] = {};
    for (uint32_t slot = 0u; slot < count; ++slot) {
        for (uint32_t i = 0u; i < 9u; ++i) {
            for (uint32_t j = 0u; j < 9u; ++j) {
                normal[i][j] += rows[slot][i] * rows[slot][j];
            }
        }
    }

    double lower[9][9] = {};
    double minDiag = 0.0;
    double maxDiag = 0.0;
    if (!cholesky9(normal, lower, minDiag, maxDiag) ||
        minDiag <= maxDiag * 1e-14) {
        counts[cell] = 0u;
        atomicAdd(&diagnostics[CentralInvalid], 1u);
        return;
    }

    for (uint32_t slot = 0u; slot < count; ++slot) {
        neighbors[slot * cellCount + cell] = selected[slot];
        double rhs[9];
        for (uint32_t i = 0u; i < 9u; ++i) rhs[i] = rows[slot][i];
        double solution[9];
        solve9(lower, rhs, solution);
        for (uint32_t i = 0u; i < 9u; ++i) {
            const float stored = float(solution[i]);
            if (!isfinite(solution[i]) || !isfinite(stored)) {
                counts[cell] = 0u;
                atomicAdd(&diagnostics[CentralInvalid], 1u);
                return;
            }
            ops[(i * ReconCentralNeighborLimit + slot) * cellCount + cell] = stored;
        }
    }
    counts[cell] = count;
    atomicAdd(&diagnostics[CentralValid], 1u);
}

__device__ inline bool selectCoverageDirections(
    const double3* directions, uint32_t count, uint32_t* selected, uint32_t target,
    uint32_t& selectedCountOut) {
    selectedCountOut = 0u;
    if (count == 0u || target == 0u) return false;
    bool used[MaxFaceDirections] = {};
    selected[0] = 0u;
    used[0] = true;
    uint32_t selectedCount = 1u;
    while (selectedCount < target && selectedCount < count) {
        uint32_t best = InvalidIndex;
        double bestScore = std::numeric_limits<double>::max();
        for (uint32_t candidate = 0u; candidate < count; ++candidate) {
            if (used[candidate]) continue;
            double worstDot = -1.0;
            for (uint32_t i = 0u; i < selectedCount; ++i) {
                const double dot = cudaMath::dot3(
                    directions[candidate], directions[selected[i]]);
                worstDot = fmax(worstDot, dot);
            }
            if (worstDot < bestScore) {
                bestScore = worstDot;
                best = candidate;
            }
        }
        if (best == InvalidIndex) break;
        selected[selectedCount++] = best;
        used[best] = true;
    }
    selectedCountOut = selectedCount;
    return selectedCount > 0u;
}

__global__ void buildDirectionalStencilKernel(
    uint32_t cellCount,
    const uint32_t* __restrict__ ringOffsets,
    const uint32_t* __restrict__ ring,
    const uint32_t* __restrict__ incidenceOffsets,
    const uint32_t* __restrict__ incidenceFaces,
    const double* __restrict__ incidenceSides,
    const uint32_t* __restrict__ faceNodeA,
    const uint32_t* __restrict__ faceNodeB,
    const uint32_t* __restrict__ faceLoopOffsets,
    const uint32_t* __restrict__ loopVertexOffsets,
    const double3* __restrict__ loopVertices,
    const double3* __restrict__ faceNormals,
    const ReconCellGeometry* __restrict__ geometry,
    uint32_t* __restrict__ cellCounts,
    uint32_t* __restrict__ neighborCounts,
    uint32_t* __restrict__ neighbors,
    float* __restrict__ ops,
    uint32_t* __restrict__ diagnostics) {
    const uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cellCount) return;

    uint32_t outputCount = 0u;
    const ReconCellGeometry target = geometry[cell];
    if (!(target.volume > 0.0) || !(target.h > 0.0)) {
        cellCounts[cell] = 0u;
        atomicAdd(&diagnostics[DirectionalCellsZero], 1u);
        return;
    }

    uint32_t pool[DirectionalPoolLimit];
    uint32_t hash[DirectionalHashSize];
    clearHash(hash, DirectionalHashSize);
    uint32_t poolCount = 0u;
    bool poolOverflow = false;
    for (uint32_t k = ringOffsets[cell]; k < ringOffsets[cell + 1u]; ++k) {
        const uint32_t neighbor = ring[k];
        if (neighbor == cell || neighbor >= cellCount) continue;
        if (poolCount == DirectionalPoolLimit) {
            poolOverflow = true;
            break;
        }
        if (!insertUnique(hash, DirectionalHashSize - 1u, neighbor)) continue;
        pool[poolCount++] = neighbor;
    }
    const uint32_t levelOneEnd = poolCount;
    for (uint32_t head = 0u; head < levelOneEnd; ++head) {
        const uint32_t current = pool[head];
        for (uint32_t k = ringOffsets[current]; k < ringOffsets[current + 1u]; ++k) {
            const uint32_t neighbor = ring[k];
            if (neighbor == cell || neighbor >= cellCount) continue;
            if (poolCount == DirectionalPoolLimit) {
                poolOverflow = true;
                break;
            }
            if (!insertUnique(hash, DirectionalHashSize - 1u, neighbor)) continue;
            pool[poolCount++] = neighbor;
        }
        if (poolOverflow) break;
    }
    if (poolOverflow) atomicAdd(&diagnostics[LoopOverflow], 1u);

    uint32_t faceList[MaxFaceDirections];
    double3 faceDirections[MaxFaceDirections];
    uint32_t faceDirectionCount = 0u;
    bool faceDirectionOverflow = false;
    for (uint32_t k = incidenceOffsets[cell]; k < incidenceOffsets[cell + 1u]; ++k) {
        const uint32_t face = incidenceFaces[k];
        const uint32_t a = faceNodeA[face];
        const uint32_t b = faceNodeB[face];
        if (!validNode(a) || !validNode(b) || a == b) continue;
        const double3 normal = faceNormals[face];
        if (cudaMath::lengthSquared3(normal) <= 0.0) continue;
        if (faceDirectionCount == MaxFaceDirections) {
            faceDirectionOverflow = true;
            break;
        }
        const double3 outward = cudaMath::mul3(normal, incidenceSides[k]);
        faceList[faceDirectionCount] = face;
        faceDirections[faceDirectionCount] = outward;
        ++faceDirectionCount;
    }
    if (faceDirectionOverflow) {
        atomicAdd(&diagnostics[LoopOverflow], 1u);
    }

    uint32_t selectedFaces[MaxSelectedFaceDirections];
    uint32_t selectedFaceCount = 0u;
    if (faceDirectionCount <= MaxSelectedFaceDirections) {
        for (uint32_t i = 0u; i < faceDirectionCount; ++i) selectedFaces[i] = i;
        selectedFaceCount = faceDirectionCount;
    } else {
        uint32_t selected[MaxFaceDirections];
        uint32_t selectedCount = 0u;
        if (selectCoverageDirections(faceDirections, faceDirectionCount,
                                     selected, MaxSelectedFaceDirections,
                                     selectedCount)) {
            for (uint32_t i = 0u; i < selectedCount; ++i) {
                selectedFaces[i] = selected[i];
            }
            selectedFaceCount = selectedCount;
        }
    }

    for (uint32_t faceSlot = 0u;
         faceSlot < selectedFaceCount && outputCount < ReconDirectionalCandidateLimit;
         ++faceSlot) {
        const uint32_t selectedIndex = selectedFaces[faceSlot];
        const uint32_t face = faceList[selectedIndex];
        const double3 normal = faceDirections[selectedIndex];
        const uint32_t loopBegin = faceLoopOffsets[face];
        const uint32_t loopEnd = faceLoopOffsets[face + 1u];
        if (loopBegin >= loopEnd) continue;

        double3 reference = zero3();
        bool hasReference = false;
        for (uint32_t loop = loopBegin; loop < loopEnd && !hasReference; ++loop) {
            const uint32_t begin = loopVertexOffsets[loop];
            const uint32_t end = loopVertexOffsets[loop + 1u];
            if (end - begin >= 3u) {
                reference = loopVertices[begin];
                hasReference = true;
            }
        }
        if (!hasReference) continue;

        uint32_t faceVertexCount = 0u;
        bool faceVertexOverflow = false;
        for (uint32_t loop = loopBegin; loop < loopEnd; ++loop) {
            const uint32_t begin = loopVertexOffsets[loop];
            const uint32_t end = loopVertexOffsets[loop + 1u];
            if (end - begin < 3u) continue;
            const uint32_t loopCount = end - begin;
            if (faceVertexCount > MaxFaceVertices - loopCount) {
                faceVertexOverflow = true;
                break;
            }
            faceVertexCount += loopCount;
        }
        if (faceVertexOverflow) {
            atomicAdd(&diagnostics[LoopOverflow], 1u);
            continue;
        }

        const double3 axis = (fabs(normal.x) < 0.9)
            ? make_double3(1.0, 0.0, 0.0)
            : make_double3(0.0, 1.0, 0.0);
        const double3 tangentU = cudaMath::normalize3(cudaMath::cross3(axis, normal));
        const double3 tangentV = cudaMath::cross3(normal, tangentU);

        double2 points[MaxFaceVertices];
        uint32_t pointCount = 0u;
        for (uint32_t loop = loopBegin; loop < loopEnd && pointCount < MaxFaceVertices; ++loop) {
            const uint32_t begin = loopVertexOffsets[loop];
            const uint32_t end = loopVertexOffsets[loop + 1u];
            for (uint32_t vertex = begin;
                 vertex < end && pointCount < MaxFaceVertices; ++vertex) {
                const double3 delta = cudaMath::sub3(loopVertices[vertex], reference);
                points[pointCount++] = make_double2(
                    cudaMath::dot3(delta, tangentU),
                    cudaMath::dot3(delta, tangentV));
            }
        }
        if (pointCount < 3u) continue;

        for (uint32_t i = 1u; i < pointCount; ++i) {
            const double2 value = points[i];
            uint32_t j = i;
            while (j > 0u && (points[j - 1u].x > value.x ||
                   (points[j - 1u].x == value.x && points[j - 1u].y > value.y))) {
                points[j] = points[j - 1u];
                --j;
            }
            points[j] = value;
        }

        double2 hull[MaxHullVertices];
        uint32_t hullCount = 0u;
        for (uint32_t i = 0u; i < pointCount; ++i) {
            while (hullCount >= 2u &&
                   cross2(hull[hullCount - 2u], hull[hullCount - 1u], points[i]) <= 0.0) {
                --hullCount;
            }
            hull[hullCount++] = points[i];
        }
        const uint32_t lowerSize = hullCount + 1u;
        for (uint32_t i = pointCount - 1u; i-- > 0u;) {
            while (hullCount >= lowerSize &&
                   cross2(hull[hullCount - 2u], hull[hullCount - 1u], points[i]) <= 0.0) {
                --hullCount;
            }
            hull[hullCount++] = points[i];
        }
        if (hullCount > 0u) --hullCount;
        if (hullCount < 3u) continue;

        const double planeDistance =
            cudaMath::dot3(cudaMath::sub3(reference, target.centroid), normal);
        uint32_t hits[DirectionalPoolLimit];
        uint32_t hitCount = 0u;
        for (uint32_t p = 0u; p < poolCount; ++p) {
            const uint32_t neighbor = pool[p];
            const double3 direction =
                cudaMath::sub3(geometry[neighbor].centroid, target.centroid);
            const double normalComponent = cudaMath::dot3(direction, normal);
            if (normalComponent <= 0.0) continue;
            const double t = planeDistance / normalComponent;
            if (!(t > 0.0) || !isfinite(t)) continue;
            const double3 intersection = cudaMath::add3(
                target.centroid, cudaMath::mul3(direction, t));
            const double3 projectedDelta = cudaMath::sub3(intersection, reference);
            const double2 projected = make_double2(
                cudaMath::dot3(projectedDelta, tangentU),
                cudaMath::dot3(projectedDelta, tangentV));
            bool inside = true;
            const double lengthScale = fmax(1.0, cudaMath::length3(projectedDelta));
            for (uint32_t edge = 0u; edge < hullCount; ++edge) {
                const double2 next = hull[edge + 1u < hullCount ? edge + 1u : 0u];
                const double edgeLength = sqrt(
                    (next.x - hull[edge].x) * (next.x - hull[edge].x) +
                    (next.y - hull[edge].y) * (next.y - hull[edge].y));
                const double tolerance = 1e-10 * lengthScale * fmax(1.0, edgeLength);
                if (cross2(hull[edge], next, projected) < -tolerance) {
                    inside = false;
                    break;
                }
            }
            if (!inside) continue;
            hits[hitCount++] = neighbor;
            if (hitCount == DirectionalPoolLimit) break;
        }
        if (hitCount < ReconLinearUnknownCount) continue;

        const uint32_t count = min(ReconDirectionalNeighborLimit, hitCount);
        uint32_t selected[ReconDirectionalNeighborLimit];
        bool used[DirectionalPoolLimit] = {};
        for (uint32_t slot = 0u; slot < count; ++slot) {
            uint32_t best = InvalidIndex;
            double bestDistance = std::numeric_limits<double>::max();
            for (uint32_t i = 0u; i < hitCount; ++i) {
                if (used[i]) continue;
                const double3 delta = cudaMath::sub3(
                    geometry[hits[i]].centroid, target.centroid);
                const double distance = cudaMath::lengthSquared3(delta);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    best = i;
                }
            }
            if (best == InvalidIndex) break;
            selected[slot] = hits[best];
            used[best] = true;
        }

        double rows[ReconDirectionalNeighborLimit][ReconLinearUnknownCount];
        const double inverseH = 1.0 / target.h;
        for (uint32_t slot = 0u; slot < count; ++slot) {
            const double3 delta = cudaMath::sub3(
                geometry[selected[slot]].centroid, target.centroid);
            rows[slot][0] = delta.x * inverseH;
            rows[slot][1] = delta.y * inverseH;
            rows[slot][2] = delta.z * inverseH;
        }

        double normalMatrix[3][3] = {};
        for (uint32_t slot = 0u; slot < count; ++slot) {
            for (uint32_t i = 0u; i < 3u; ++i) {
                for (uint32_t j = 0u; j < 3u; ++j) {
                    normalMatrix[i][j] += rows[slot][i] * rows[slot][j];
                }
            }
        }

        double lower[3][3] = {};
        double minDiag = 0.0;
        double maxDiag = 0.0;
        if (!cholesky3(normalMatrix, lower, minDiag, maxDiag)) {
            atomicAdd(&diagnostics[DirectionalRankDeficient], 1u);
            continue;
        }
        double minimum = 0.0;
        double maximum = 0.0;
        if (!symmetricEigenExtremes3(normalMatrix, minimum, maximum) ||
            minimum * DirectionalConditionLimit * DirectionalConditionLimit < maximum) {
            atomicAdd(&diagnostics[DirectionalIllConditioned], 1u);
            continue;
        }

        bool valid = true;
        float candidateOps[ReconDirectionalNeighborLimit][ReconLinearUnknownCount];
        for (uint32_t slot = 0u; slot < count && valid; ++slot) {
            double rhs[3] = {rows[slot][0], rows[slot][1], rows[slot][2]};
            double solution[3];
            solve3(lower, rhs, solution);
            for (uint32_t i = 0u; i < 3u; ++i) {
                const float stored = float(solution[i]);
                if (!isfinite(solution[i]) || !isfinite(stored)) {
                    valid = false;
                    break;
                }
                candidateOps[slot][i] = stored;
            }
        }
        if (!valid) {
            atomicAdd(&diagnostics[DirectionalInvalid], 1u);
            continue;
        }

        const uint32_t candidateIndex = outputCount;
        neighborCounts[candidateIndex * cellCount + cell] = count;
        for (uint32_t slot = 0u; slot < count; ++slot) {
            neighbors[(candidateIndex * ReconDirectionalNeighborLimit + slot) * cellCount + cell] =
                selected[slot];
            for (uint32_t unknown = 0u; unknown < ReconLinearUnknownCount; ++unknown) {
                ops[((candidateIndex * ReconLinearUnknownCount + unknown) *
                         ReconDirectionalNeighborLimit + slot) * cellCount + cell] =
                    candidateOps[slot][unknown];
            }
        }
        ++outputCount;
    }

    if (outputCount == 0u) {
        atomicAdd(&diagnostics[DirectionalCellsZero], 1u);
    }
    atomicAdd(&diagnostics[DirectionalCandidates], outputCount);
    atomicAdd(&diagnostics[CoverageHistogram +
        min(outputCount, CoverageHistogramBins - 1u)], 1u);
    atomicMin(&diagnostics[CoverageMinimum], outputCount);
    atomicMax(&diagnostics[CoverageMaximum], outputCount);
    cellCounts[cell] = outputCount;
}

__device__ inline float cellComponent(const glm::vec4& value, uint32_t component) {
    return component == 0u ? value.x
         : component == 1u ? value.y
                           : value.z;
}

__device__ inline double smoothnessIndicator(
    const float coefficients[ReconQuadraticUnknownCount],
    const double q[6]) {
    const double hxx = 2.0 * coefficients[3];
    const double hxy = coefficients[4];
    const double hxz = coefficients[5];
    const double hyy = 2.0 * coefficients[6];
    const double hyz = coefficients[7];
    const double hzz = 2.0 * coefficients[8];

    const double qxx = q[0];
    const double qxy = q[1];
    const double qxz = q[2];
    const double qyy = q[3];
    const double qyz = q[4];
    const double qzz = q[5];

    const double m00 = hxx * hxx + hxy * hxy + hxz * hxz;
    const double m01 = hxx * hxy + hxy * hyy + hxz * hyz;
    const double m02 = hxx * hxz + hxy * hyz + hxz * hzz;
    const double m11 = hxy * hxy + hyy * hyy + hyz * hyz;
    const double m12 = hxy * hxz + hyy * hyz + hyz * hzz;
    const double m22 = hxz * hxz + hyz * hyz + hzz * hzz;

    double indicator =
        coefficients[0] * coefficients[0] +
        coefficients[1] * coefficients[1] +
        coefficients[2] * coefficients[2];
    indicator += m00 * qxx + 2.0 * m01 * qxy + 2.0 * m02 * qxz +
                 m11 * qyy + 2.0 * m12 * qyz + m22 * qzz;
    indicator += hxx * hxx + hyy * hyy + hzz * hzz +
                 hxy * hxy + hxz * hxz + hyz * hyz;
    return indicator;
}

__device__ inline double linearSmoothnessIndicator(
    const float coefficients[ReconLinearUnknownCount]) {
    const double cx = coefficients[0];
    const double cy = coefficients[1];
    const double cz = coefficients[2];
    return cx * cx + cy * cy + cz * cz;
}

__global__ void computeCoefficientsKernel(
    uint32_t cellCount,
    const glm::vec4* __restrict__ cellAverages,
    const double* __restrict__ referenceVelocity,
    const uint32_t* __restrict__ centralCount,
    const uint32_t* __restrict__ centralNeighbor,
    const float* __restrict__ centralOp,
    const uint32_t* __restrict__ directionalCount,
    const uint32_t* __restrict__ directionalNeighborCount,
    const uint32_t* __restrict__ directionalNeighbor,
    const float* __restrict__ directionalOp,
    const ReconCellGeometry* __restrict__ geometry,
    ReconFinalCoefficients* __restrict__ coefficients,
    uint32_t* __restrict__ diagnostics) {
    const uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cellCount) return;

    ReconFinalCoefficients result{};
    const ReconCellGeometry& target = geometry[cell];
    const uint32_t centralSlots = centralCount[cell];
    if (!(target.volume > 0.0) || !(target.h > 0.0) ||
        centralSlots < ReconQuadraticUnknownCount) {
        coefficients[cell] = result;
        atomicAdd(&diagnostics[CoefficientInvalid], 1u);
        return;
    }

    const uint32_t candidateCount =
        min(directionalCount[cell], ReconDirectionalCandidateLimit);
    const float referenceFloat = static_cast<float>(referenceVelocity[0]);
    const float inverseReference = 1.0f / referenceFloat;
    const double q[6] = {
        target.q[0], target.q[1], target.q[2],
        target.q[3], target.q[4], target.q[5]};

    const glm::vec4 cellAverage = cellAverages[cell];

    float optimal[3][ReconQuadraticUnknownCount] = {};
    {
        const uint32_t* centralNeighborRow = centralNeighbor + cell;
        const float* centralOpRow = centralOp + cell;
        for (uint32_t slot = 0u; slot < centralSlots; ++slot) {
            const glm::vec4 neighbor = cellAverages[centralNeighborRow[slot * cellCount]];
            const float difference[3] = {
                (neighbor.x - cellAverage.x) * inverseReference,
                (neighbor.y - cellAverage.y) * inverseReference,
                (neighbor.z - cellAverage.z) * inverseReference};
            const float* op = centralOpRow + slot * cellCount;
            for (uint32_t unknown = 0u; unknown < ReconQuadraticUnknownCount; ++unknown) {
                const float coefficient = op[unknown * (ReconCentralNeighborLimit * cellCount)];
                optimal[0][unknown] += coefficient * difference[0];
                optimal[1][unknown] += coefficient * difference[1];
                optimal[2][unknown] += coefficient * difference[2];
            }
        }
    }

    float candidate[ReconDirectionalCandidateLimit][3][ReconLinearUnknownCount] = {};
#pragma unroll
    for (uint32_t candidateIndex = 0u; candidateIndex < ReconDirectionalCandidateLimit;
         ++candidateIndex) {
        if (candidateIndex >= candidateCount) break;
        const uint32_t neighborCount =
            directionalNeighborCount[candidateIndex * cellCount + cell];
        const uint32_t* neighborRow = directionalNeighbor +
            (candidateIndex * ReconDirectionalNeighborLimit) * cellCount + cell;
        const float* opRow = directionalOp +
            (candidateIndex * ReconLinearUnknownCount * ReconDirectionalNeighborLimit) *
                cellCount + cell;
        for (uint32_t slot = 0u; slot < neighborCount; ++slot) {
            const glm::vec4 neighbor = cellAverages[neighborRow[slot * cellCount]];
            const float difference[3] = {
                (neighbor.x - cellAverage.x) * inverseReference,
                (neighbor.y - cellAverage.y) * inverseReference,
                (neighbor.z - cellAverage.z) * inverseReference};
            const float* op = opRow + slot * cellCount;
            for (uint32_t unknown = 0u; unknown < ReconLinearUnknownCount; ++unknown) {
                const float coefficient =
                    op[unknown * (ReconDirectionalNeighborLimit * cellCount)];
                candidate[candidateIndex][0][unknown] += coefficient * difference[0];
                candidate[candidateIndex][1][unknown] += coefficient * difference[1];
                candidate[candidateIndex][2][unknown] += coefficient * difference[2];
            }
        }
    }

    bool valid = true;
#pragma unroll
    for (uint32_t component = 0u; component < 3u; ++component) {
        float final[ReconQuadraticUnknownCount] = {};
        if (candidateCount == 0u) {
            for (uint32_t unknown = 0u; unknown < ReconQuadraticUnknownCount; ++unknown) {
                final[unknown] = optimal[component][unknown];
            }
        } else {
            const double lambda1 = 1.0 - 1.0 / CentralWeightPrime;
            const double lambdaS =
                (1.0 - lambda1) / double(candidateCount);

            float centralPolynomial[ReconQuadraticUnknownCount] = {};
            for (uint32_t unknown = 0u; unknown < ReconQuadraticUnknownCount; ++unknown) {
                double directionalSum = 0.0;
                if (unknown < ReconLinearUnknownCount) {
                    for (uint32_t candidateIndex = 0u;
                         candidateIndex < candidateCount; ++candidateIndex) {
                        directionalSum += candidate[candidateIndex][component][unknown];
                    }
                }
                centralPolynomial[unknown] =
                    (optimal[component][unknown] - lambdaS * directionalSum) / lambda1;
            }

            double indicators[1u + ReconDirectionalCandidateLimit] = {};
            indicators[0] = smoothnessIndicator(centralPolynomial, q);
            for (uint32_t candidateIndex = 0u; candidateIndex < candidateCount; ++candidateIndex) {
                indicators[1u + candidateIndex] =
                    linearSmoothnessIndicator(candidate[candidateIndex][component]);
            }

            double tauSum = 0.0;
            for (uint32_t candidateIndex = 0u; candidateIndex < candidateCount; ++candidateIndex) {
                tauSum += fabs(indicators[1u + candidateIndex] - indicators[0]);
            }
            const double scaled = tauSum / double(candidateCount);
            const double tau =
                scaled * scaled * scaled * scaled * scaled * scaled;

            double gamma[1u + ReconDirectionalCandidateLimit] = {};
            double gammaSum = 0.0;
            for (uint32_t stencil = 0u; stencil <= candidateCount; ++stencil) {
                gamma[stencil] = 1.0 + tau / (indicators[stencil] + SmoothnessEpsilon);
                gammaSum += gamma[stencil];
            }

            const double selectionBarrier = EnoThreshold * gammaSum;
            uint32_t accepted = 0u;
            uint32_t activeDirectional = 0u;
            for (uint32_t stencil = 0u; stencil <= candidateCount; ++stencil) {
                if (gamma[stencil] < selectionBarrier) continue;
                accepted |= 1u << stencil;
                if (stencil != 0u) ++activeDirectional;
            }
            const bool centralSelected = (accepted & 1u) != 0u;

            if (centralSelected || activeDirectional == 0u) {
                for (uint32_t unknown = 0u; unknown < ReconQuadraticUnknownCount; ++unknown) {
                    final[unknown] = centralPolynomial[unknown];
                }
            } else {
                const double weight = 1.0 / double(activeDirectional);
                for (uint32_t candidateIndex = 0u;
                     candidateIndex < candidateCount; ++candidateIndex) {
                    if ((accepted & (2u << candidateIndex)) == 0u) continue;
                    for (uint32_t unknown = 0u; unknown < ReconLinearUnknownCount; ++unknown) {
                        final[unknown] += static_cast<float>(
                            weight * static_cast<double>(candidate[candidateIndex][component][unknown]));
                    }
                }
            }
        }

        for (uint32_t unknown = 0u; unknown < ReconQuadraticUnknownCount; ++unknown) {
            const float stored = final[unknown] * referenceFloat;
            if (!isfinite(stored)) {
                valid = false;
                break;
            }
            result.c[component][unknown] = stored;
        }
        if (!valid) break;
    }

    result.valid = valid ? 1u : 0u;
    if (!valid) {
        atomicAdd(&diagnostics[CoefficientInvalid], 1u);
        result = ReconFinalCoefficients{};
    }
    coefficients[cell] = result;
}

bool inclusiveSumU32(
    cudaUtils::CudaBuffer<uint32_t>& values,
    cudaUtils::CudaBuffer<uint32_t>& offsets,
    uint32_t count,
    cudaStream_t stream) {
    if (count == 0u) {
        return cudaUtils::cudaCheck(
            cudaMemsetAsync(offsets.get(), 0, sizeof(uint32_t), stream),
            "initialize empty scan offset");
    }
    size_t tempBytes = 0u;
    if (cub::DeviceScan::InclusiveSum(
            nullptr, tempBytes, values.get(), offsets.get() + 1u,
            int(count), stream) != cudaSuccess) {
        return false;
    }
    cudaUtils::CudaBuffer<uint8_t> temp;
    if (!temp.allocate(tempBytes)) return false;
    if (cub::DeviceScan::InclusiveSum(
            temp.get(), tempBytes, values.get(), offsets.get() + 1u,
            int(count), stream) != cudaSuccess) {
        return false;
    }
    return cudaUtils::cudaCheck(
        cudaMemsetAsync(offsets.get(), 0, sizeof(uint32_t), stream),
        "initialize scan offset");
}

bool verifyCsr(
    cudaUtils::CudaBuffer<uint32_t>& offsets,
    cudaUtils::CudaBuffer<uint32_t>& cursor,
    uint32_t count,
    uint32_t expectedTotal,
    const char* label,
    cudaStream_t stream) {
    std::vector<uint32_t> hostOffsets(size_t(count) + 1u, 0u);
    std::vector<uint32_t> hostCursor(count, 0u);
    if (!offsets.downloadAsync(hostOffsets.data(), hostOffsets.size(), stream) ||
        !cursor.downloadAsync(hostCursor.data(), hostCursor.size(), stream) ||
        !cudaUtils::cudaCheck(cudaStreamSynchronize(stream), label)) {
        return false;
    }
    if (hostOffsets[0] != 0u || hostOffsets[count] != expectedTotal) {
        std::cerr << "[CFDRecon] " << label << " offsets invalid: first="
                  << hostOffsets[0] << " total=" << hostOffsets[count]
                  << " expected=" << expectedTotal << "\n";
        return false;
    }
    for (uint32_t cell = 0u; cell < count; ++cell) {
        if (hostCursor[cell] != hostOffsets[cell + 1u]) {
            std::cerr << "[CFDRecon] " << label << " cursor mismatch cell=" << cell
                      << " cursor=" << hostCursor[cell]
                      << " expected=" << hostOffsets[cell + 1u] << "\n";
            return false;
        }
    }
    return true;
}

template <class T>
bool uploadVector(
    cudaUtils::CudaBuffer<T>& buffer,
    const std::vector<T>& values,
    cudaStream_t stream) {
    if (!buffer.allocate(values.size())) return false;
    if (values.empty()) return true;
    return buffer.uploadAsync(values.data(), values.size(), stream);
}

} // namespace

static constexpr uint32_t CheckFieldMonomials = 20u;
static constexpr uint32_t CheckFaceSamples = 32768u;
static constexpr double CheckReferenceVelocity = 1.5;
static constexpr double CheckQuadraticFaceLimit = 1.0e-4;

using CheckField = std::array<double, CheckFieldMonomials>;

struct CheckContext {
    const uint32_t cellCount;
    const uint32_t fineFaceCount;
    const double extent;
    const std::vector<ReconCellGeometry>& cellGeometry;
    const std::vector<ReconFaceGeometry>& faceGeometry;
    const std::vector<ReconFinalCoefficients>& coefficients;
    const std::vector<uint32_t>& centralCounts;
    const std::vector<uint32_t>& centralNeighbors;
    const std::vector<float>& centralOps;
    const std::vector<uint32_t>& faceNodeA;
    const std::vector<uint32_t>& faceNodeB;
    const std::vector<uint32_t>& fineFaceGlobal;
    const std::vector<double3>& faceNormals;
    const std::vector<double3>& loopVertices;
    const std::vector<uint32_t>& faceLoopOffsets;
    const std::vector<uint32_t>& loopVertexOffsets;
};

static const std::array<cudaMoments::Monomial, CheckFieldMonomials>& checkMonomials() {
    static const std::array<cudaMoments::Monomial, CheckFieldMonomials> table = [] {
        std::array<cudaMoments::Monomial, CheckFieldMonomials> values{};
        for (uint32_t slot = 0u; slot < CheckFieldMonomials; ++slot) {
            values[slot] = cudaMoments::monomialAt(slot);
        }
        return values;
    }();
    return table;
}

static void fieldPowers(const double3& point, double powers[3][4]) {
    powers[0][0] = powers[1][0] = powers[2][0] = 1.0;
    powers[0][1] = point.x;
    powers[1][1] = point.y;
    powers[2][1] = point.z;
    powers[0][2] = point.x * point.x;
    powers[1][2] = point.y * point.y;
    powers[2][2] = point.z * point.z;
    powers[0][3] = powers[0][2] * point.x;
    powers[1][3] = powers[1][2] * point.y;
    powers[2][3] = powers[2][2] * point.z;
}

static void fieldValues(const CheckField fields[3], const double3& point, double out[3]) {
    const auto& monomials = checkMonomials();
    double powers[3][4];
    fieldPowers(point, powers);
    double values[CheckFieldMonomials];
    for (uint32_t slot = 0u; slot < CheckFieldMonomials; ++slot) {
        const cudaMoments::Monomial& monomial = monomials[slot];
        values[slot] = powers[0][monomial.x] * powers[1][monomial.y] * powers[2][monomial.z];
    }
    for (uint32_t component = 0u; component < 3u; ++component) {
        double sum = 0.0;
        for (uint32_t slot = 0u; slot < CheckFieldMonomials; ++slot) {
            sum += fields[component][slot] * values[slot];
        }
        out[component] = sum;
    }
}

static void fieldGradients(const CheckField fields[3], const double3& point, double out[3][3]) {
    const auto& monomials = checkMonomials();
    double powers[3][4];
    fieldPowers(point, powers);
    for (uint32_t component = 0u; component < 3u; ++component) {
        out[component][0] = out[component][1] = out[component][2] = 0.0;
        for (uint32_t slot = 0u; slot < CheckFieldMonomials; ++slot) {
            const cudaMoments::Monomial& monomial = monomials[slot];
            const uint32_t exponent[3] = {monomial.x, monomial.y, monomial.z};
            for (uint32_t axis = 0u; axis < 3u; ++axis) {
                if (exponent[axis] == 0u) continue;
                out[component][axis] += fields[component][slot] * double(exponent[axis]) *
                    powers[0][exponent[0] - (axis == 0u ? 1u : 0u)] *
                    powers[1][exponent[1] - (axis == 1u ? 1u : 0u)] *
                    powers[2][exponent[2] - (axis == 2u ? 1u : 0u)];
            }
        }
    }
}

static void fieldHalfHessians(const CheckField fields[3], const double3& point, double out[3][6]) {
    const auto& monomials = checkMonomials();
    double powers[3][4];
    fieldPowers(point, powers);
    const uint32_t pairs[6][2] = {{0u, 0u}, {0u, 1u}, {0u, 2u}, {1u, 1u}, {1u, 2u}, {2u, 2u}};
    for (uint32_t component = 0u; component < 3u; ++component) {
        for (uint32_t pair = 0u; pair < 6u; ++pair) {
            const uint32_t i = pairs[pair][0];
            const uint32_t j = pairs[pair][1];
            double sum = 0.0;
            for (uint32_t slot = 0u; slot < CheckFieldMonomials; ++slot) {
                const cudaMoments::Monomial& monomial = monomials[slot];
                const uint32_t exponent[3] = {monomial.x, monomial.y, monomial.z};
                if (exponent[i] == 0u || exponent[j] == 0u) continue;
                const double factor = (i == j)
                    ? double(exponent[i]) * double(exponent[i] - 1u)
                    : double(exponent[i]) * double(exponent[j]);
                if (factor == 0.0) continue;
                sum += fields[component][slot] * factor *
                    powers[0][exponent[0] - (i == 0u ? 1u : 0u) - (j == 0u ? 1u : 0u)] *
                    powers[1][exponent[1] - (i == 1u ? 1u : 0u) - (j == 1u ? 1u : 0u)] *
                    powers[2][exponent[2] - (i == 2u ? 1u : 0u) - (j == 2u ? 1u : 0u)];
            }
            out[component][pair] = 0.5 * sum;
        }
    }
}

static double packedHessian(double halfHessian, uint32_t pair) {
    const bool offDiagonal = pair == 1u || pair == 2u || pair == 4u;
    return offDiagonal ? 2.0 * halfHessian : halfHessian;
}

static CheckField fieldTranslate(const CheckField& field, const double3& origin) {
    const auto& monomials = checkMonomials();
    double powers[3][4];
    fieldPowers(origin, powers);
    CheckField result{};
    for (uint32_t target = 0u; target < CheckFieldMonomials; ++target) {
        const cudaMoments::Monomial& t = monomials[target];
        double sum = 0.0;
        for (uint32_t source = 0u; source < CheckFieldMonomials; ++source) {
            const cudaMoments::Monomial& s = monomials[source];
            if (s.x < t.x || s.y < t.y || s.z < t.z) continue;
            sum += field[source] *
                cudaMoments::binomialCoefficient(s.x, t.x) * powers[0][s.x - t.x] *
                cudaMoments::binomialCoefficient(s.y, t.y) * powers[1][s.y - t.y] *
                cudaMoments::binomialCoefficient(s.z, t.z) * powers[2][s.z - t.z];
        }
        result[target] = sum;
    }
    return result;
}

static void quadraticCellAverages(
    const CheckField fields[3], const double3& centroid, const double* m2, double out[3]) {
    fieldValues(fields, centroid, out);
    for (uint32_t component = 0u; component < 3u; ++component) {
        for (uint32_t k = 0u; k < 6u; ++k) {
            out[component] += fields[component][4u + k] * m2[k];
        }
    }
}

static void cubicCellAverages(
    const CheckField fields[3], const double3& centroid, const double* moments, double out[3]) {
    const double volume = moments[0];
    for (uint32_t component = 0u; component < 3u; ++component) {
        out[component] = 0.0;
        if (!(volume > 0.0)) continue;
        const CheckField central = fieldTranslate(fields[component], centroid);
        double sum = 0.0;
        for (uint32_t slot = 0u; slot < CheckFieldMonomials; ++slot) {
            sum += central[slot] * moments[slot];
        }
        out[component] = sum / volume;
    }
}

static void addFaceQuadrature(
    const CheckField fields[3],
    const double3* vertices,
    uint32_t vertexBegin,
    uint32_t vertexEnd,
    const double3& normal,
    double integral[3],
    double& area) {
    if (vertexEnd - vertexBegin < 3u) return;
    static constexpr double sNodes[3] = {0.11270166537925831, 0.5, 0.88729833462074170};
    static constexpr double sWeights[3] = {5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0};
    static constexpr double tNodes[2] = {0.21132486540518713, 0.78867513459481287};
    static constexpr double tWeights[2] = {0.5, 0.5};
    const double3* points = vertices + vertexBegin;
    const uint32_t count = vertexEnd - vertexBegin;
    const double3 base = points[0];
    for (uint32_t triangle = 1u; triangle + 1u < count; ++triangle) {
        const double3 edgeB = cudaMath::sub3(points[triangle], base);
        const double3 edgeC = cudaMath::sub3(points[triangle + 1u], base);
        const double twoArea = cudaMath::dot3(cudaMath::cross3(edgeB, edgeC), normal);
        if (twoArea == 0.0) continue;
        for (uint32_t sIndex = 0u; sIndex < 3u; ++sIndex) {
            for (uint32_t tIndex = 0u; tIndex < 2u; ++tIndex) {
                const double s = sNodes[sIndex];
                const double t = tNodes[tIndex];
                const double weight = sWeights[sIndex] * tWeights[tIndex] * s * twoArea;
                const double3 offset = cudaMath::add3(
                    cudaMath::mul3(edgeB, s * (1.0 - t)), cudaMath::mul3(edgeC, s * t));
                double values[3];
                fieldValues(fields, cudaMath::add3(base, offset), values);
                area += weight;
                for (uint32_t component = 0u; component < 3u; ++component) {
                    integral[component] += weight * values[component];
                }
            }
        }
    }
}

static void reconstructedFaceValues(
    const ReconFinalCoefficients& coefficients,
    const ReconCellGeometry& cell,
    const ReconFaceGeometry& face,
    const glm::vec4& cellAverage,
    double out[3]) {
    const double inverseH = 1.0 / cell.h;
    const double inverseH2 = inverseH * inverseH;
    const double dx = face.centroid.x - cell.centroid.x;
    const double dy = face.centroid.y - cell.centroid.y;
    const double dz = face.centroid.z - cell.centroid.z;
    const double offsets[6] = {dx * dx, dx * dy, dx * dz, dy * dy, dy * dz, dz * dz};
    const double average[3] = {double(cellAverage.x), double(cellAverage.y), double(cellAverage.z)};
    for (uint32_t component = 0u; component < 3u; ++component) {
        double value = average[component];
        value += double(coefficients.c[component][0]) * inverseH * dx;
        value += double(coefficients.c[component][1]) * inverseH * dy;
        value += double(coefficients.c[component][2]) * inverseH * dz;
        for (uint32_t k = 0u; k < 6u; ++k) {
            value += double(coefficients.c[component][3u + k]) * inverseH2 *
                (offsets[k] + face.m2[k] - cell.m2[k]);
        }
        out[component] = value;
    }
}

static void lowOrderFaceValues(
    const glm::vec4& averageA,
    const glm::vec4& averageB,
    bool upwindIsA,
    double out[3]) {
    const double blend = 0.20;
    const double average[3] = {
        0.5 * (double(averageA.x) + double(averageB.x)),
        0.5 * (double(averageA.y) + double(averageB.y)),
        0.5 * (double(averageA.z) + double(averageB.z))};
    const glm::vec4& upwind = upwindIsA ? averageA : averageB;
    const double upwindValues[3] = {double(upwind.x), double(upwind.y), double(upwind.z)};
    for (uint32_t component = 0u; component < 3u; ++component) {
        out[component] = blend * upwindValues[component] + (1.0 - blend) * average[component];
    }
}

static void quadraticCheckFields(double extent, CheckField fields[3]) {
    static constexpr double constant[3] = {0.30, -0.20, 0.10};
    static constexpr double gradient[3][3] = {
        {0.90, 0.40, -0.60}, {-0.50, 0.80, 0.30}, {0.20, -0.70, 0.90}};
    static constexpr double quadratic[3][6] = {
        {1.70, 0.80, -0.60, 1.30, 0.50, 0.90},
        {-0.90, 0.50, 0.70, 1.10, -0.80, 0.60},
        {0.60, -1.20, 0.40, -0.70, 0.90, 1.40}};
    const double gradientScale = 2.4 / extent;
    const double quadraticScale = 1.0 / (extent * extent);
    for (uint32_t component = 0u; component < 3u; ++component) {
        fields[component] = CheckField{};
        fields[component][0] = constant[component];
        for (uint32_t axis = 0u; axis < 3u; ++axis) {
            fields[component][1u + axis] = gradient[component][axis] * gradientScale;
        }
        for (uint32_t k = 0u; k < 6u; ++k) {
            fields[component][4u + k] = quadratic[component][k] * quadraticScale;
        }
    }
}

static void cubicCheckFields(double extent, CheckField fields[3]) {
    static constexpr double constant[3] = {0.25, -0.15, 0.05};
    static constexpr double gradient[3][3] = {
        {0.80, -0.30, 0.50}, {0.40, 0.90, -0.70}, {-0.60, 0.20, 0.85}};
    static constexpr double quadratic[3][6] = {
        {1.10, -0.70, 0.40, 0.90, 0.60, -1.20},
        {-0.80, 0.50, 1.30, 0.70, -0.90, 0.40},
        {0.60, 1.20, -0.50, -1.10, 0.30, 0.80}};
    static constexpr double cubic[3][10] = {
        {0.90, -0.60, 0.50, 1.20, -0.80, 0.70, -1.10, 0.40, 0.60, -0.90},
        {-0.70, 1.10, 0.30, -0.50, 0.90, -1.20, 0.60, 0.80, -0.40, 1.00},
        {1.30, 0.40, -0.90, 0.70, 1.10, 0.50, -0.60, -1.00, 0.80, 0.30}};
    const double linearScale = 2.4 / extent;
    const double quadraticScale = 1.0 / (extent * extent);
    const double cubicScale = 1.0 / (extent * extent * extent);
    for (uint32_t component = 0u; component < 3u; ++component) {
        fields[component] = CheckField{};
        fields[component][0] = constant[component];
        for (uint32_t axis = 0u; axis < 3u; ++axis) {
            fields[component][1u + axis] = gradient[component][axis] * linearScale;
        }
        for (uint32_t k = 0u; k < 6u; ++k) {
            fields[component][4u + k] = quadratic[component][k] * quadraticScale;
        }
        for (uint32_t k = 0u; k < 10u; ++k) {
            fields[component][10u + k] = cubic[component][k] * cubicScale;
        }
    }
}

static double relativeRatio(const double error[2], const double scale[2]) {
    return scale[0] > 0.0 ? std::sqrt(error[0] / scale[0]) : 0.0;
}

static double relativeMaximum(const double error[2], const double scale[2]) {
    return scale[1] > 0.0 ? error[1] / scale[1] : 0.0;
}

static void reportCoefficientError(
    const CheckContext& ctx,
    const CheckField fields[3],
    const std::vector<glm::vec4>& averages,
    const char* label) {
    double gradientError[2] = {};
    double gradientScale[2] = {};
    double hessianError[2] = {};
    double hessianScale[2] = {};
    double stencilGradient[2] = {};
    double stencilHessian[2] = {};
    uint32_t invalid = 0u;
    for (uint32_t cell = 0u; cell < ctx.cellCount; ++cell) {
        const ReconCellGeometry& geometry = ctx.cellGeometry[cell];
        if (ctx.coefficients[cell].valid == 0u || !(geometry.volume > 0.0) ||
            !(geometry.h > 0.0)) {
            ++invalid;
            continue;
        }
        const uint32_t stencilCount = ctx.centralCounts[cell];
        double gradients[3][3];
        double hessians[3][6];
        fieldGradients(fields, geometry.centroid, gradients);
        fieldHalfHessians(fields, geometry.centroid, hessians);
        for (uint32_t component = 0u; component < 3u; ++component) {
            double expected[ReconQuadraticUnknownCount];
            for (uint32_t axis = 0u; axis < 3u; ++axis) {
                expected[axis] = geometry.h * gradients[component][axis];
            }
            for (uint32_t k = 0u; k < 6u; ++k) {
                expected[3u + k] = geometry.h * geometry.h *
                    packedHessian(hessians[component][k], k);
            }
            double optimal[ReconQuadraticUnknownCount] = {};
            if (stencilCount >= ReconQuadraticUnknownCount) {
                for (uint32_t slot = 0u; slot < stencilCount; ++slot) {
                    const uint32_t neighbor =
                        ctx.centralNeighbors[slot * ctx.cellCount + cell];
                    if (neighbor >= ctx.cellCount) continue;
                    const double difference =
                        (double(averages[neighbor][component]) -
                         double(averages[cell][component])) / CheckReferenceVelocity;
                    for (uint32_t unknown = 0u; unknown < ReconQuadraticUnknownCount; ++unknown) {
                        optimal[unknown] += double(ctx.centralOps[
                            (unknown * ReconCentralNeighborLimit + slot) * ctx.cellCount + cell]) *
                            difference;
                    }
                }
            }
            for (uint32_t unknown = 0u; unknown < ReconQuadraticUnknownCount; ++unknown) {
                const double expectedValue = expected[unknown];
                const double error =
                    double(ctx.coefficients[cell].c[component][unknown]) - expectedValue;
                const double stencilError =
                    optimal[unknown] * CheckReferenceVelocity - expectedValue;
                double* errorAccumulator = unknown < 3u ? gradientError : hessianError;
                double* stencilAccumulator = unknown < 3u ? stencilGradient : stencilHessian;
                double* scaleAccumulator = unknown < 3u ? gradientScale : hessianScale;
                errorAccumulator[0] += error * error;
                errorAccumulator[1] = std::max(errorAccumulator[1], std::fabs(error));
                stencilAccumulator[0] += stencilError * stencilError;
                stencilAccumulator[1] = std::max(stencilAccumulator[1], std::fabs(stencilError));
                scaleAccumulator[0] += expectedValue * expectedValue;
                scaleAccumulator[1] = std::max(scaleAccumulator[1], std::fabs(expectedValue));
            }
        }
    }
}

static bool checkQuadraticFaces(
    const CheckContext& ctx,
    const CheckField fields[3],
    const std::vector<glm::vec4>& averages) {
    double error[2] = {};
    double reference[2] = {};
    double lowOrderError[2] = {};
    uint64_t pairs = 0u;
    uint64_t unavailable = 0u;
    for (uint32_t face = 0u; face < ctx.fineFaceCount; ++face) {
        const uint32_t domainFace = ctx.fineFaceGlobal[face];
        const ReconFaceGeometry& geometry = ctx.faceGeometry[face];
        if (!(geometry.area > 0.0)) continue;
        double exact[3];
        quadraticCellAverages(fields, geometry.centroid, geometry.m2, exact);
        const uint32_t adjacent[2] = {
            ctx.faceNodeA[domainFace], ctx.faceNodeB[domainFace]};
        const bool bothValid = adjacent[0] < ctx.cellCount && adjacent[1] < ctx.cellCount;
        const double3 normal = ctx.faceNormals[domainFace];
        const double flux = geometry.area *
            (exact[0] * normal.x + exact[1] * normal.y + exact[2] * normal.z);
        for (uint32_t side = 0u; side < 2u; ++side) {
            const uint32_t cell = adjacent[side];
            if (cell >= ctx.cellCount) continue;
            ++pairs;
            if (ctx.coefficients[cell].valid == 0u || !(ctx.cellGeometry[cell].volume > 0.0) ||
                !(ctx.cellGeometry[cell].h > 0.0)) {
                ++unavailable;
                continue;
            }
            double values[3];
            reconstructedFaceValues(ctx.coefficients[cell], ctx.cellGeometry[cell],
                                    geometry, averages[cell], values);
            for (uint32_t component = 0u; component < 3u; ++component) {
                const double delta = values[component] - exact[component];
                const double span = exact[component] - double(averages[cell][component]);
                error[0] += delta * delta;
                error[1] = std::max(error[1], std::fabs(delta));
                reference[0] += span * span;
                reference[1] = std::max(reference[1], std::fabs(span));
            }
            if (bothValid) {
                double low[3];
                lowOrderFaceValues(averages[adjacent[0]], averages[adjacent[1]],
                                   flux >= 0.0, low);
                for (uint32_t component = 0u; component < 3u; ++component) {
                    const double delta = low[component] - exact[component];
                    lowOrderError[0] += delta * delta;
                    lowOrderError[1] = std::max(lowOrderError[1], std::fabs(delta));
                }
            }
        }
    }
    const double ratio = relativeRatio(error, reference);
    if (!(ratio <= CheckQuadraticFaceLimit)) {
        std::cerr << "[CFDRecon] quadratic reproduction rejected rel=" << ratio
                  << " limit=" << CheckQuadraticFaceLimit << "\n";
        return false;
    }
    return true;
}

static void checkCubicFaces(
    const CheckContext& ctx,
    const CheckField fields[3],
    const std::vector<glm::vec4>& averages) {
    double error[2] = {};
    double reference[2] = {};
    double lowOrderError[2] = {};
    uint64_t pairs = 0u;
    uint64_t sampledFaces = 0u;
    uint64_t unavailable = 0u;
    double heightSum = 0.0;
    uint32_t heightCount = 0u;
    for (uint32_t cell = 0u; cell < ctx.cellCount; ++cell) {
        if (!(ctx.cellGeometry[cell].h > 0.0)) continue;
        heightSum += ctx.cellGeometry[cell].h;
        ++heightCount;
    }
    const uint32_t stride = std::max<uint32_t>(
        1u, (ctx.fineFaceCount + CheckFaceSamples - 1u) / CheckFaceSamples);
    for (uint32_t face = 0u; face < ctx.fineFaceCount; face += stride) {
        const uint32_t domainFace = ctx.fineFaceGlobal[face];
        const uint32_t loopBegin = ctx.faceLoopOffsets[domainFace];
        const uint32_t loopEnd = ctx.faceLoopOffsets[domainFace + 1u];
        const double3 normal = ctx.faceNormals[domainFace];
        double integral[3] = {};
        double area = 0.0;
        for (uint32_t loop = loopBegin; loop < loopEnd; ++loop) {
            addFaceQuadrature(fields, ctx.loopVertices.data(),
                              ctx.loopVertexOffsets[loop], ctx.loopVertexOffsets[loop + 1u],
                              normal, integral, area);
        }
        if (area < 0.0) {
            area = -area;
            for (uint32_t component = 0u; component < 3u; ++component) {
                integral[component] = -integral[component];
            }
        }
        if (!(area > 0.0)) continue;
        ++sampledFaces;
        const double exact[3] = {
            integral[0] / area, integral[1] / area, integral[2] / area};
        const uint32_t adjacent[2] = {
            ctx.faceNodeA[domainFace], ctx.faceNodeB[domainFace]};
        const bool bothValid = adjacent[0] < ctx.cellCount && adjacent[1] < ctx.cellCount;
        const double flux = area * (exact[0] * normal.x + exact[1] * normal.y +
                                    exact[2] * normal.z);
        for (uint32_t side = 0u; side < 2u; ++side) {
            const uint32_t cell = adjacent[side];
            if (cell >= ctx.cellCount) continue;
            ++pairs;
            if (ctx.coefficients[cell].valid == 0u || !(ctx.cellGeometry[cell].volume > 0.0) ||
                !(ctx.cellGeometry[cell].h > 0.0) || !(ctx.faceGeometry[face].area > 0.0)) {
                ++unavailable;
                continue;
            }
            double values[3];
            reconstructedFaceValues(ctx.coefficients[cell], ctx.cellGeometry[cell],
                                    ctx.faceGeometry[face], averages[cell], values);
            for (uint32_t component = 0u; component < 3u; ++component) {
                const double delta = values[component] - exact[component];
                const double span = exact[component] - double(averages[cell][component]);
                error[0] += delta * delta;
                error[1] = std::max(error[1], std::fabs(delta));
                reference[0] += span * span;
                reference[1] = std::max(reference[1], std::fabs(span));
            }
            if (bothValid) {
                double low[3];
                lowOrderFaceValues(averages[adjacent[0]], averages[adjacent[1]],
                                   flux >= 0.0, low);
                for (uint32_t component = 0u; component < 3u; ++component) {
                    const double delta = low[component] - exact[component];
                    lowOrderError[0] += delta * delta;
                    lowOrderError[1] = std::max(lowOrderError[1], std::fabs(delta));
                }
            }
        }
    }
}

static double divergenceAverage(
    const CheckField fields[3], const double3& centroid, const double* m2) {
    double gradients[3][3];
    fieldGradients(fields, centroid, gradients);
    double value = gradients[0][0] + gradients[1][1] + gradients[2][2];
    double halfHessians[3][6];
    fieldHalfHessians(fields, centroid, halfHessians);
    for (uint32_t k = 0u; k < 6u; ++k) {
        const double summed =
            halfHessians[0][k] + halfHessians[1][k] + halfHessians[2][k];
        value += packedHessian(summed, k) * m2[k];
    }
    return value;
}

static void checkDivergence(
    const CheckContext& ctx,
    const CheckField fields[3],
    const std::vector<glm::vec4>& averages,
    const char* label) {
    std::vector<double> residual(ctx.cellCount, 0.0);
    std::vector<uint8_t> complete(ctx.cellCount, 1u);
    std::vector<uint8_t> open(ctx.cellCount, 0u);
    for (uint32_t face = 0u; face < uint32_t(ctx.faceNodeA.size()); ++face) {
        const uint32_t a = ctx.faceNodeA[face];
        const uint32_t b = ctx.faceNodeB[face];
        if (a < ctx.cellCount && b >= ctx.cellCount) open[a] = 1u;
        if (b < ctx.cellCount && a >= ctx.cellCount) open[b] = 1u;
    }
    uint64_t faces = 0u;
    for (uint32_t face = 0u; face < ctx.fineFaceCount; ++face) {
        const uint32_t domainFace = ctx.fineFaceGlobal[face];
        const uint32_t adjacent[2] = {
            ctx.faceNodeA[domainFace], ctx.faceNodeB[domainFace]};
        const double area = ctx.faceGeometry[face].area;
        if (!(area > 0.0)) {
            for (uint32_t side = 0u; side < 2u; ++side) {
                if (adjacent[side] < ctx.cellCount) complete[adjacent[side]] = 0u;
            }
            continue;
        }
        ++faces;
        const double3 normal = ctx.faceNormals[domainFace];
        for (uint32_t side = 0u; side < 2u; ++side) {
            const uint32_t cell = adjacent[side];
            if (cell >= ctx.cellCount) continue;
            if (ctx.coefficients[cell].valid == 0u || !(ctx.cellGeometry[cell].volume > 0.0) ||
                !(ctx.cellGeometry[cell].h > 0.0)) {
                complete[cell] = 0u;
                continue;
            }
            double values[3];
            reconstructedFaceValues(ctx.coefficients[cell], ctx.cellGeometry[cell],
                                    ctx.faceGeometry[face], averages[cell], values);
            const double flux = area *
                (values[0] * normal.x + values[1] * normal.y + values[2] * normal.z);
            residual[cell] += side == 0u ? flux : -flux;
        }
    }
    double error[2] = {};
    double scale[2] = {};
    uint64_t counted = 0u;
    uint64_t skipped = 0u;
    for (uint32_t cell = 0u; cell < ctx.cellCount; ++cell) {
        const ReconCellGeometry& geometry = ctx.cellGeometry[cell];
        if (open[cell] != 0u || complete[cell] == 0u ||
            !(geometry.volume > 0.0) || !(geometry.h > 0.0)) {
            ++skipped;
            continue;
        }
        ++counted;
        const double discrete = residual[cell] / geometry.volume;
        const double exact = divergenceAverage(fields, geometry.centroid, geometry.m2);
        const double delta = discrete - exact;
        error[0] += delta * delta;
        error[1] = std::max(error[1], std::fabs(delta));
        scale[0] += exact * exact;
        scale[1] = std::max(scale[1], std::fabs(exact));
    }
}

static void reportMomentConsistency(
    const CheckContext& ctx, const std::vector<double>& moments) {
    double volumeError = 0.0;
    double momentError = 0.0;
    uint32_t counted = 0u;
    for (uint32_t cell = 0u; cell < ctx.cellCount; ++cell) {
        const ReconCellGeometry& geometry = ctx.cellGeometry[cell];
        if (!(geometry.volume > 0.0) || !(geometry.h > 0.0)) continue;
        const double* cellMoments = moments.data() + size_t(cell) * cudaMoments::monomialLimit;
        if (!(cellMoments[0] > 0.0)) continue;
        ++counted;
        volumeError = std::max(volumeError,
                               std::fabs(cellMoments[0] - geometry.volume) / geometry.volume);
        const double inverseH2 = 1.0 / (geometry.h * geometry.h);
        for (uint32_t k = 0u; k < 6u; ++k) {
            const double average = cellMoments[4u + k] / cellMoments[0];
            momentError = std::max(momentError, std::fabs(average - geometry.m2[k]) * inverseH2);
        }
    }
}

bool CFDReconstruction::verifyReconstruction(
    const std::vector<double>& faceAreasA,
    const std::vector<double>& faceAreasB,
    const std::vector<glm::dvec3>& faceAreaVectorsA,
    const std::vector<glm::dvec3>& faceAreaVectorsB,
    const std::vector<uint32_t>& faceLoopOffsets,
    const std::vector<uint32_t>& loopVertexOffsets,
    const std::vector<double3>& cellCentroids,
    const std::vector<uint32_t>& faceNodeA,
    const std::vector<uint32_t>& faceNodeB,
    const std::vector<double3>& faceNormals,
    const std::vector<uint32_t>& fineFaceGlobal,
    const std::vector<double3>& loopVertices,
    cudaStream_t stream) {
    const uint32_t cellCount = uint32_t(cellCentroids.size());
    const uint32_t fineFaceCount = uint32_t(fineFaceGlobal.size());
    if (cellCount_ != cellCount || cellCount == 0u || fineFaceCount == 0u) return true;

    std::vector<ReconCellGeometry> cellGeometry(cellCount);
    std::vector<ReconFaceGeometry> faceGeometry(fineFaceCount);
    std::vector<ReconFinalCoefficients> coefficients(cellCount);
    std::vector<uint32_t> centralCounts(cellCount);
    std::vector<uint32_t> centralNeighbors(size_t(cellCount) * ReconCentralNeighborLimit);
    std::vector<float> centralOps(size_t(cellCount) * ReconCentralNeighborLimit *
                                  ReconQuadraticUnknownCount);
    if (!d_cellGeometry_.download(cellGeometry.data(), cellCount) ||
        !d_faceGeometry_.download(faceGeometry.data(), fineFaceCount) ||
        !d_coefficients_.download(coefficients.data(), cellCount) ||
        !d_centralCount_.download(centralCounts.data(), cellCount) ||
        !d_centralNeighbor_.download(centralNeighbors.data(), centralNeighbors.size()) ||
        !d_centralOp_.download(centralOps.data(), centralOps.size())) {
        std::cerr << "[CFDRecon] check download failed\n";
        return false;
    }

    double3 minimum = cellCentroids[0];
    double3 maximum = cellCentroids[0];
    for (const double3& centroid : cellCentroids) {
        minimum = make_double3(std::min(minimum.x, centroid.x),
                               std::min(minimum.y, centroid.y),
                               std::min(minimum.z, centroid.z));
        maximum = make_double3(std::max(maximum.x, centroid.x),
                               std::max(maximum.y, centroid.y),
                               std::max(maximum.z, centroid.z));
    }
    const double extent = std::max(maximum.x - minimum.x,
                          std::max(maximum.y - minimum.y, maximum.z - minimum.z));
    if (!(extent > 0.0) || !isfinite(extent)) return true;

    const CheckContext ctx{
        cellCount, fineFaceCount, extent, cellGeometry, faceGeometry,
        coefficients, centralCounts, centralNeighbors, centralOps,
        faceNodeA, faceNodeB, fineFaceGlobal, faceNormals,
        loopVertices, faceLoopOffsets, loopVertexOffsets};

    cudaUtils::CudaBuffer<double> d_reference;
    cudaUtils::CudaBuffer<glm::vec4> d_averages;
    if (!d_reference.allocate(1u) || !d_averages.allocate(cellCount) ||
        !d_reference.upload(&CheckReferenceVelocity, 1u)) {
        std::cerr << "[CFDRecon] check allocation failed\n";
        return false;
    }
    const auto evaluate = [&](const std::vector<glm::vec4>& averages) {
        return d_averages.upload(averages.data(), averages.size()) &&
               computeCoefficients(d_averages.get(), d_reference.get(), stream) &&
               cudaUtils::cudaCheck(cudaStreamSynchronize(stream),
                                    "synchronize reconstruction check") &&
               d_coefficients_.download(coefficients.data(), cellCount);
    };

    std::vector<glm::vec4> averages(cellCount);
    CheckField quadraticFields[3];
    quadraticCheckFields(extent, quadraticFields);
    for (uint32_t cell = 0u; cell < cellCount; ++cell) {
        double values[3];
        quadraticCellAverages(quadraticFields, cellCentroids[cell],
                              cellGeometry[cell].m2, values);
        averages[cell] = glm::vec4(float(values[0]), float(values[1]), float(values[2]), 0.0f);
    }
    if (!evaluate(averages)) {
        std::cerr << "[CFDRecon] check coefficient evaluation failed\n";
        return false;
    }
    reportCoefficientError(ctx, quadraticFields, averages, "quadratic");
    if (!checkQuadraticFaces(ctx, quadraticFields, averages)) return false;
    checkDivergence(ctx, quadraticFields, averages, "quadratic");

    std::vector<uint32_t> cellFaceOffsets(size_t(cellCount) + 1u, 0u);
    for (uint32_t face = 0u; face < uint32_t(faceNodeA.size()); ++face) {
        if (cudaMath::lengthSquared3(faceNormals[face]) <= 0.0) continue;
        const uint32_t a = faceNodeA[face];
        const uint32_t b = faceNodeB[face];
        if (a < cellCount) ++cellFaceOffsets[a + 1u];
        if (b < cellCount && b != a) ++cellFaceOffsets[b + 1u];
    }
    for (uint32_t cell = 0u; cell < cellCount; ++cell) {
        cellFaceOffsets[cell + 1u] += cellFaceOffsets[cell];
    }
    std::vector<uint32_t> cellFaceCursor(cellFaceOffsets.begin(), cellFaceOffsets.end() - 1);
    std::vector<uint32_t> cellFaces(cellFaceOffsets.back());
    std::vector<double> cellFaceSides(cellFaceOffsets.back());
    for (uint32_t face = 0u; face < uint32_t(faceNodeA.size()); ++face) {
        if (cudaMath::lengthSquared3(faceNormals[face]) <= 0.0) continue;
        const uint32_t a = faceNodeA[face];
        const uint32_t b = faceNodeB[face];
        if (a < cellCount) {
            const uint32_t slot = cellFaceCursor[a]++;
            cellFaces[slot] = face;
            cellFaceSides[slot] = 1.0;
        }
        if (b < cellCount && b != a) {
            const uint32_t slot = cellFaceCursor[b]++;
            cellFaces[slot] = face;
            cellFaceSides[slot] = -1.0;
        }
    }
    std::vector<glm::dvec3> cellOrigins(cellCount);
    for (uint32_t cell = 0u; cell < cellCount; ++cell) {
        cellOrigins[cell] = glm::dvec3(cellCentroids[cell].x, cellCentroids[cell].y,
                                       cellCentroids[cell].z);
    }
    std::vector<glm::dvec3> loopVerticesGlm(loopVertices.size());
    for (size_t i = 0; i < loopVertices.size(); ++i) {
        loopVerticesGlm[i] = glm::dvec3(
            loopVertices[i].x, loopVertices[i].y, loopVertices[i].z);
    }
    std::vector<double> moments;
    if (!domainMoments::buildCellMoments(
            static_cast<uint32_t>(faceAreasA.size()),
            faceAreasA,
            faceAreasB,
            faceAreaVectorsA,
            faceAreaVectorsB,
            faceLoopOffsets,
            loopVertexOffsets,
            loopVerticesGlm,
            cellCount,
            cellFaceOffsets,
            cellFaces,
            cellFaceSides,
            cellOrigins,
            moments)) {
        std::cerr << "[CFDRecon] check moment build failed\n";
        return false;
    }
    reportMomentConsistency(ctx, moments);

    CheckField cubicFields[3];
    cubicCheckFields(extent, cubicFields);
    for (uint32_t cell = 0u; cell < cellCount; ++cell) {
        double values[3];
        cubicCellAverages(cubicFields, cellCentroids[cell],
                          moments.data() + size_t(cell) * cudaMoments::monomialLimit, values);
        averages[cell] = glm::vec4(float(values[0]), float(values[1]), float(values[2]), 0.0f);
    }
    if (!evaluate(averages)) {
        std::cerr << "[CFDRecon] check coefficient evaluation failed\n";
        return false;
    }
    reportCoefficientError(ctx, cubicFields, averages, "cubic");
    checkCubicFaces(ctx, cubicFields, averages);
    checkDivergence(ctx, cubicFields, averages, "cubic");
    return true;
}

bool CFDReconstruction::build(
    const CFDRuntime& runtime,
    const std::vector<glm::dvec3>& fragmentCentroids,
    const std::vector<double>& fragmentVolumes,
    const std::vector<uint32_t>& faceFragmentA,
    const std::vector<uint32_t>& faceFragmentB,
    const std::vector<double>& faceAreasA,
    const std::vector<double>& faceAreasB,
    const std::vector<glm::dvec3>& faceAreaVectorsA,
    const std::vector<glm::dvec3>& faceAreaVectorsB,
    const std::vector<uint32_t>& faceLoopOffsets,
    const std::vector<uint32_t>& loopVertexOffsets,
    const std::vector<glm::dvec3>& loopVertices,
    const std::vector<uint32_t>& globalFragmentToFluidNode,
    const std::vector<uint32_t>& fluidNodeToGlobalFragment,
    cudaStream_t stream) {

    reset();

    const uint32_t cellCount = uint32_t(fluidNodeToGlobalFragment.size());
    const uint32_t domainFaceCount = uint32_t(faceFragmentA.size());
    const uint32_t fragmentCount = uint32_t(fragmentCentroids.size());
    const uint32_t fineFaceCount = uint32_t(runtime.faceTopology().size());
    if (cellCount == 0u) return true;
    if (domainFaceCount == 0u || runtime.faceGlobalIndex().size() != fineFaceCount) {
        std::cerr << "[CFDRecon] invalid topology sizes cells=" << cellCount
                  << " domainFaces=" << domainFaceCount
                  << " fineFaces=" << fineFaceCount << "\n";
        return false;
    }
    if (faceLoopOffsets.size() != size_t(domainFaceCount) + 1u ||
        loopVertexOffsets.empty() ||
        loopVertexOffsets.size() != size_t(faceLoopOffsets.back()) + 1u ||
        loopVertices.size() != size_t(loopVertexOffsets.back())) {
        std::cerr << "[CFDRecon] face loop arrays invalid\n";
        return false;
    }

    std::vector<double3> cellCentroids(cellCount);
    std::vector<double> cellVolumes(cellCount);
    for (uint32_t cell = 0u; cell < cellCount; ++cell) {
        const uint32_t fragment = fluidNodeToGlobalFragment[cell];
        if (fragment >= fragmentCount ||
            globalFragmentToFluidNode[fragment] != cell) {
            std::cerr << "[CFDRecon] invalid fragment mapping cell=" << cell << "\n";
            return false;
        }
        const double volume = fragmentVolumes[fragment];
        const glm::dvec3& centroid = fragmentCentroids[fragment];
        if (!(volume > 0.0) || !isfinite(volume) ||
            !isfinite(centroid.x) || !isfinite(centroid.y) ||
            !isfinite(centroid.z)) {
            std::cerr << "[CFDRecon] invalid fragment geometry cell=" << cell << "\n";
            return false;
        }
        cellCentroids[cell] = make_double3(centroid.x, centroid.y, centroid.z);
        cellVolumes[cell] = volume;
    }

    std::vector<uint32_t> faceNodeA(domainFaceCount, InvalidIndex);
    std::vector<uint32_t> faceNodeB(domainFaceCount, InvalidIndex);
    std::vector<double3> faceNormals(domainFaceCount, zero3());
    uint32_t expectedIncidenceCount = 0u;
    uint32_t expectedRingCount = 0u;
    for (uint32_t face = 0u; face < domainFaceCount; ++face) {
        const uint32_t fragmentA = faceFragmentA[face];
        const uint32_t fragmentB = faceFragmentB[face];
        if (fragmentA < globalFragmentToFluidNode.size()) {
            faceNodeA[face] = globalFragmentToFluidNode[fragmentA];
        }
        if (fragmentB < globalFragmentToFluidNode.size()) {
            faceNodeB[face] = globalFragmentToFluidNode[fragmentB];
        }
        glm::dvec3 normal = faceAreaVectorsA[face];
        double length = glm::length(normal);
        if (!(length > 0.0)) {
            normal = -faceAreaVectorsB[face];
            length = glm::length(normal);
        }
        if (length > 0.0 && isfinite(length)) {
            normal /= length;
            faceNormals[face] = make_double3(normal.x, normal.y, normal.z);
            const uint32_t a = faceNodeA[face];
            const uint32_t b = faceNodeB[face];
            if (a != InvalidIndex) ++expectedIncidenceCount;
            if (b != InvalidIndex && b != a) ++expectedIncidenceCount;
            if (a != InvalidIndex && b != InvalidIndex && a != b) {
                expectedRingCount += 2u;
            }
        }
    }

    std::vector<uint32_t> fineFaceGlobal(fineFaceCount);
    std::vector<double> fineFaceAreas(fineFaceCount);
    for (uint32_t face = 0u; face < fineFaceCount; ++face) {
        const uint32_t globalFace = runtime.faceGlobalIndex()[face];
        if (globalFace >= domainFaceCount) {
            std::cerr << "[CFDRecon] fine face mapping invalid face=" << face << "\n";
            return false;
        }
        fineFaceGlobal[face] = globalFace;
        fineFaceAreas[face] = faceAreasA[globalFace] > 0.0
            ? faceAreasA[globalFace]
            : faceAreasB[globalFace];
    }

    std::vector<double3> loopVerticesD3(loopVertices.size());
    for (size_t i = 0u; i < loopVertices.size(); ++i) {
        const glm::dvec3& source = loopVertices[i];
        loopVerticesD3[i] = make_double3(source.x, source.y, source.z);
    }

    cudaUtils::CudaBuffer<uint32_t> d_faceNodeA;
    cudaUtils::CudaBuffer<uint32_t> d_faceNodeB;
    cudaUtils::CudaBuffer<double3> d_faceNormals;
    cudaUtils::CudaBuffer<uint32_t> d_faceLoopOffsets;
    cudaUtils::CudaBuffer<uint32_t> d_loopVertexOffsets;
    cudaUtils::CudaBuffer<double3> d_loopVertices;
    cudaUtils::CudaBuffer<double3> d_cellCentroids;
    cudaUtils::CudaBuffer<double> d_cellVolumes;
    cudaUtils::CudaBuffer<uint32_t> d_fineFaceGlobal;
    cudaUtils::CudaBuffer<double> d_fineFaceAreas;
    cudaUtils::CudaBuffer<uint32_t> d_incidenceCounts;
    cudaUtils::CudaBuffer<uint32_t> d_incidenceOffsets;
    cudaUtils::CudaBuffer<uint32_t> d_incidenceCursor;
    cudaUtils::CudaBuffer<uint32_t> d_incidenceFaces;
    cudaUtils::CudaBuffer<double> d_incidenceSides;
    cudaUtils::CudaBuffer<uint32_t> d_ringCounts;
    cudaUtils::CudaBuffer<uint32_t> d_ringOffsets;
    cudaUtils::CudaBuffer<uint32_t> d_ringCursor;
    cudaUtils::CudaBuffer<uint32_t> d_ring;
    cudaUtils::CudaBuffer<double> d_cellMoments;

    if (!uploadVector(d_faceNodeA, faceNodeA, stream) ||
        !uploadVector(d_faceNodeB, faceNodeB, stream) ||
        !uploadVector(d_faceNormals, faceNormals, stream) ||
        !uploadVector(d_faceLoopOffsets, faceLoopOffsets, stream) ||
        !uploadVector(d_loopVertexOffsets, loopVertexOffsets, stream) ||
        !uploadVector(d_loopVertices, loopVerticesD3, stream) ||
        !uploadVector(d_cellCentroids, cellCentroids, stream) ||
        !uploadVector(d_cellVolumes, cellVolumes, stream) ||
        !uploadVector(d_fineFaceGlobal, fineFaceGlobal, stream) ||
        !uploadVector(d_fineFaceAreas, fineFaceAreas, stream) ||
        !d_cellGeometry_.allocate(cellCount) ||
        !d_centralCount_.allocate(cellCount) ||
        !d_centralNeighbor_.allocate(size_t(cellCount) * ReconCentralNeighborLimit) ||
        !d_centralOp_.allocate(size_t(cellCount) * ReconCentralNeighborLimit *
                               ReconQuadraticUnknownCount) ||
        !d_directionalCount_.allocate(cellCount) ||
        !d_directionalNeighborCount_.allocate(size_t(cellCount) *
                                              ReconDirectionalCandidateLimit) ||
        !d_directionalNeighbor_.allocate(size_t(cellCount) * ReconDirectionalCandidateLimit *
                                         ReconDirectionalNeighborLimit) ||
        !d_directionalOp_.allocate(size_t(cellCount) * ReconDirectionalCandidateLimit *
                                   ReconDirectionalNeighborLimit * ReconLinearUnknownCount) ||
        !d_faceGeometry_.allocate(fineFaceCount) ||
        !d_coefficients_.allocate(cellCount) ||
        !d_diagnostics_.allocate(DiagnosticCount)) {
        std::cerr << "[CFDRecon] device allocation or upload failed\n";
        reset();
        return false;
    }

    if (!cudaUtils::cudaCheck(
            cudaMemsetAsync(d_diagnostics_.get(), 0, sizeof(uint32_t) * DiagnosticCount,
                            stream),
            "clear reconstruction diagnostics") ||
        !cudaUtils::cudaCheck(
            cudaMemsetAsync(d_diagnostics_.get() + CoverageMinimum, 0xFF,
                            sizeof(uint32_t), stream),
            "initialize coverage minimum") ||
        !cudaUtils::cudaCheck(
            cudaMemsetAsync(d_cellGeometry_.get(), 0,
                            sizeof(ReconCellGeometry) * cellCount, stream),
            "clear reconstruction geometry") ||
        !cudaUtils::cudaCheck(
            cudaMemsetAsync(d_coefficients_.get(), 0,
                            sizeof(ReconFinalCoefficients) * cellCount, stream),
            "clear reconstruction coefficients")) {
        reset();
        return false;
    }

    const uint32_t faceBlocks = (domainFaceCount + BlockSize - 1u) / BlockSize;
    const uint32_t cellBlocks = (cellCount + BlockSize - 1u) / BlockSize;

    if (!d_incidenceCounts.allocate(cellCount) ||
        !d_incidenceOffsets.allocate(size_t(cellCount) + 1u) ||
        !d_incidenceCursor.allocate(cellCount) ||
        !d_incidenceFaces.allocate(size_t(domainFaceCount) * 2u) ||
        !d_incidenceSides.allocate(size_t(domainFaceCount) * 2u) ||
        !d_ringCounts.allocate(cellCount) ||
        !d_ringOffsets.allocate(size_t(cellCount) + 1u) ||
        !d_ringCursor.allocate(cellCount) ||
        !d_ring.allocate(size_t(domainFaceCount) * 2u)) {
        std::cerr << "[CFDRecon] incidence allocation failed\n";
        reset();
        return false;
    }

    if (!cudaUtils::cudaCheck(
            cudaMemsetAsync(d_incidenceCounts.get(), 0,
                            sizeof(uint32_t) * cellCount, stream),
            "clear incidence counts") ||
        !cudaUtils::cudaCheck(
            cudaMemsetAsync(d_ringCounts.get(), 0,
                            sizeof(uint32_t) * cellCount, stream),
            "clear ring counts")) {
        reset();
        return false;
    }

    countIncidenceKernel<<<faceBlocks, BlockSize, 0, stream>>>(
        domainFaceCount, d_faceNodeA.get(), d_faceNodeB.get(),
        d_faceNormals.get(), d_incidenceCounts.get());
    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch count incidence")) {
        reset();
        return false;
    }
    if (!inclusiveSumU32(d_incidenceCounts, d_incidenceOffsets, cellCount, stream) ||
        !cudaUtils::cudaCheck(
            cudaMemcpyAsync(d_incidenceCursor.get(), d_incidenceOffsets.get(),
                            sizeof(uint32_t) * cellCount,
                            cudaMemcpyDeviceToDevice, stream),
            "initialize incidence cursor")) {
        reset();
        return false;
    }
    fillIncidenceKernel<<<faceBlocks, BlockSize, 0, stream>>>(
        domainFaceCount, d_faceNodeA.get(), d_faceNodeB.get(),
        d_faceNormals.get(), d_incidenceCursor.get(),
        d_incidenceFaces.get(), d_incidenceSides.get());
    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch fill incidence") ||
        !verifyCsr(d_incidenceOffsets, d_incidenceCursor, cellCount,
                   expectedIncidenceCount, "incidence CSR", stream)) {
        reset();
        return false;
    }

    countRingKernel<<<faceBlocks, BlockSize, 0, stream>>>(
        domainFaceCount, d_faceNodeA.get(), d_faceNodeB.get(),
        d_faceNormals.get(), d_ringCounts.get());
    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch count ring")) {
        reset();
        return false;
    }
    if (!inclusiveSumU32(d_ringCounts, d_ringOffsets, cellCount, stream) ||
        !cudaUtils::cudaCheck(
            cudaMemcpyAsync(d_ringCursor.get(), d_ringOffsets.get(),
                            sizeof(uint32_t) * cellCount,
                            cudaMemcpyDeviceToDevice, stream),
            "initialize ring cursor")) {
        reset();
        return false;
    }
    fillRingKernel<<<faceBlocks, BlockSize, 0, stream>>>(
        domainFaceCount, d_faceNodeA.get(), d_faceNodeB.get(),
        d_faceNormals.get(), d_ringCursor.get(), d_ring.get());
    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch fill ring") ||
        !verifyCsr(d_ringOffsets, d_ringCursor, cellCount,
                   expectedRingCount, "ring CSR", stream)) {
        reset();
        return false;
    }

    if (fineFaceCount > 0u) {
        const uint32_t fineBlocks = (fineFaceCount + BlockSize - 1u) / BlockSize;
        faceGeometryKernel<<<fineBlocks, BlockSize, 0, stream>>>(
            fineFaceCount, domainFaceCount, d_fineFaceGlobal.get(), d_faceLoopOffsets.get(),
            d_loopVertexOffsets.get(), d_loopVertices.get(), d_faceNormals.get(),
            d_fineFaceAreas.get(), d_faceGeometry_.get(), d_diagnostics_.get());
        if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch face geometry")) {
            reset();
            return false;
        }
    }

    if (!d_cellMoments.allocate(size_t(cellCount) * 10u)) {
        reset();
        return false;
    }
    cellMomentsKernel<<<cellCount, 10u, 0, stream>>>(
        cellCount, d_incidenceOffsets.get(), d_incidenceFaces.get(),
        d_incidenceSides.get(), d_cellCentroids.get(), d_faceLoopOffsets.get(),
        d_loopVertexOffsets.get(), d_loopVertices.get(), d_faceNormals.get(),
        d_cellMoments.get());
    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch cell moments")) {
        reset();
        return false;
    }
    finalizeCellGeometryKernel<<<cellBlocks, BlockSize, 0, stream>>>(
        cellCount, d_cellCentroids.get(), d_cellVolumes.get(), d_cellMoments.get(),
        d_cellGeometry_.get(), d_diagnostics_.get());
    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch finalize cell geometry")) {
        reset();
        return false;
    }

    buildCentralStencilKernel<<<cellBlocks, BlockSize, 0, stream>>>(
        cellCount, d_ringOffsets.get(), d_ring.get(), d_cellGeometry_.get(),
        d_centralCount_.get(), d_centralNeighbor_.get(), d_centralOp_.get(),
        d_diagnostics_.get());
    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch central stencils")) {
        reset();
        return false;
    }

    buildDirectionalStencilKernel<<<cellBlocks, BlockSize, 0, stream>>>(
        cellCount, d_ringOffsets.get(), d_ring.get(), d_incidenceOffsets.get(),
        d_incidenceFaces.get(), d_incidenceSides.get(), d_faceNodeA.get(),
        d_faceNodeB.get(), d_faceLoopOffsets.get(), d_loopVertexOffsets.get(),
        d_loopVertices.get(), d_faceNormals.get(), d_cellGeometry_.get(),
        d_directionalCount_.get(), d_directionalNeighborCount_.get(),
        d_directionalNeighbor_.get(), d_directionalOp_.get(), d_diagnostics_.get());
    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch directional stencils")) {
        reset();
        return false;
    }

    if (!cudaUtils::cudaCheck(cudaStreamSynchronize(stream),
                              "synchronize reconstruction build")) {
        reset();
        return false;
    }

    uint32_t diagnostics[DiagnosticCount] = {};
    if (!d_diagnostics_.download(diagnostics, DiagnosticCount)) {
        reset();
        return false;
    }

    const bool centralComplete =
        diagnostics[CentralValid] == cellCount &&
        diagnostics[CentralInvalid] == 0u;
    const bool geometryValid =
        diagnostics[CellInvalid] == 0u &&
        diagnostics[CellGeometryMismatch] == 0u &&
        diagnostics[FaceInvalid] == 0u &&
        diagnostics[FaceAreaMismatch] == 0u;
    const bool withinLimits = diagnostics[LoopOverflow] == 0u;
    if (!centralComplete || !geometryValid || !withinLimits) {
        std::cerr << "[CFDRecon] reconstruction build rejected:"
                  << " centralComplete=" << (centralComplete ? 1 : 0)
                  << " geometryValid=" << (geometryValid ? 1 : 0)
                  << " withinLimits=" << (withinLimits ? 1 : 0)
                  << "\n";
        reset();
        return false;
    }

    cellCount_ = cellCount;
    faceCount_ = fineFaceCount;
    domainFaceCount_ = domainFaceCount;

    if (!verifyReconstruction(faceAreasA, faceAreasB, faceAreaVectorsA, faceAreaVectorsB,
                               faceLoopOffsets, loopVertexOffsets, cellCentroids, faceNodeA,
                               faceNodeB, faceNormals, fineFaceGlobal, loopVerticesD3, stream)) {
        reset();
        return false;
    }
    return true;
}

bool CFDReconstruction::computeCoefficients(
    const glm::vec4* cellAverages,
    const double* referenceVelocity,
    cudaStream_t stream) {
    if (cellCount_ == 0u || cellAverages == nullptr || referenceVelocity == nullptr ||
        d_centralCount_.get() == nullptr || d_directionalCount_.get() == nullptr ||
        d_cellGeometry_.get() == nullptr || d_coefficients_.get() == nullptr) {
        return false;
    }

    const uint32_t blocks = (cellCount_ + BlockSize - 1u) / BlockSize;
    computeCoefficientsKernel<<<blocks, BlockSize, 0, stream>>>(
        cellCount_, cellAverages, referenceVelocity, d_centralCount_.get(),
        d_centralNeighbor_.get(), d_centralOp_.get(), d_directionalCount_.get(),
        d_directionalNeighborCount_.get(), d_directionalNeighbor_.get(),
        d_directionalOp_.get(), d_cellGeometry_.get(), d_coefficients_.get(),
        d_diagnostics_.get());
    return cudaUtils::cudaCheck(cudaGetLastError(), "launch CTENOZ coefficients");
}

cudaError_t CFDReconstruction::appendCoefficientsNode(
    cudaGraph_t graph,
    cudaGraphNode_t& prev,
    const glm::vec4* cellAverages,
    const double* referenceVelocity) {
    if (cellCount_ == 0u || cellAverages == nullptr || referenceVelocity == nullptr ||
        d_centralCount_.get() == nullptr || d_directionalCount_.get() == nullptr ||
        d_cellGeometry_.get() == nullptr || d_coefficients_.get() == nullptr) {
        return cudaErrorInvalidValue;
    }

    const uint32_t blocks = (cellCount_ + BlockSize - 1u) / BlockSize;
    return cudaGraphBuilder::appendKernel(
        graph, prev, (void*)computeCoefficientsKernel, dim3(blocks), dim3(BlockSize),
        cellCount_, cellAverages, referenceVelocity, d_centralCount_.get(),
        d_centralNeighbor_.get(), d_centralOp_.get(), d_directionalCount_.get(),
        d_directionalNeighborCount_.get(), d_directionalNeighbor_.get(),
        d_directionalOp_.get(), d_cellGeometry_.get(), d_coefficients_.get(),
        d_diagnostics_.get());
}

bool CFDReconstruction::downloadInvalidCoefficientCount(
    uint32_t& count,
    cudaStream_t stream) const {
    if (d_diagnostics_.get() == nullptr) {
        return false;
    }
    return cudaUtils::cudaCheck(
        cudaMemcpyAsync(&count, d_diagnostics_.get() + CoefficientInvalid, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, stream),
        "download invalid coefficient count");
}

void CFDReconstruction::reset() {
    cellCount_ = 0u;
    faceCount_ = 0u;
    domainFaceCount_ = 0u;
    d_centralCount_.reset();
    d_centralNeighbor_.reset();
    d_centralOp_.reset();
    d_directionalCount_.reset();
    d_directionalNeighborCount_.reset();
    d_directionalNeighbor_.reset();
    d_directionalOp_.reset();
    d_cellGeometry_.reset();
    d_faceGeometry_.reset();
    d_coefficients_.reset();
    d_diagnostics_.reset();
}

} // namespace cfd
