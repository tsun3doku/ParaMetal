#pragma once

#include "RVD.hpp"
#include "RVDStatus.cuh"
#include "RVDPredicates.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace voronoi::rvdDevice {

constexpr uint32_t EndOfList = 0xffffffffu;

static constexpr int NoPlaneNeighbor = -1;
static constexpr int DomainFaceBase = -2;

__host__ __device__ inline int domainFaceTag(uint32_t face) {
    return DomainFaceBase - static_cast<int>(face); // -2 .. -7
}

__host__ __device__ inline int domainFaceFromTag(int tag) {
    if (tag > DomainFaceBase || tag < DomainFaceBase - 5) {
        return -1;
    }
    return DomainFaceBase - tag;
}

struct NormalLimits {
    static constexpr int P = 50;
    static constexpr int V = 50;
};

struct DiscoveryLimits {
    static constexpr int P = int(RVDRetryPlaneCount);
    static constexpr int V = int(RVDRetryVertexCount);
    static constexpr int T = int(RVDRetryTriangleCount);
};

__device__ inline float3 add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 mul(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
__device__ inline float dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 cross3(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ inline float lengthSquared(float3 a) { return dot3(a, a); }
__device__ inline float3 min3(float3 a, float3 b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
__device__ inline float3 max3(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
__device__ inline bool finite3(float3 a) { return isfinite(a.x) && isfinite(a.y) && isfinite(a.z); }

template <class Limits>
struct ConvexCellStorage {
    alignas(16) uint3 vertices[Limits::V];
    alignas(16) float4 planes[Limits::P];
    int planeNeighborIds[Limits::P];
    uint32_t boundaryNext[Limits::P];
    double planeAreas[Limits::P];
    uint32_t vertexCount = 0;
    uint32_t planeCount = 0;
};

template <PredicateMode Mode>
struct ConvexCell {
    float4* planes = nullptr;
    int* planeNeighborIds = nullptr;
    uint32_t* boundaryNext = nullptr;
    uint3* vertices = nullptr;
    double* planeAreas = nullptr;
    uint32_t cellId = 0;
    uint32_t planeCapacity = 0;
    uint32_t vertexCapacity = 0;
    int vertexCount = 0;
    int planeCount = 0;
    RVDStatus status = RVDStatus::Success;

    template <class Limits>
    __device__ void attachStorage(ConvexCellStorage<Limits>& storage,
                                  uint32_t planeLimit = uint32_t(Limits::P),
                                  uint32_t vertexLimit = uint32_t(Limits::V)) {
        planes = storage.planes;
        planeNeighborIds = storage.planeNeighborIds;
        boundaryNext = storage.boundaryNext;
        vertices = storage.vertices;
        planeAreas = storage.planeAreas;
        planeCapacity = planeLimit;
        vertexCapacity = vertexLimit;
    }

    template <class Limits>
    __device__ bool loadStorage(const ConvexCellStorage<Limits>& source) {
        if (source.planeCount > planeCapacity || source.vertexCount > vertexCapacity) {
            fail(RVDStatus::InvalidIndex);
            return false;
        }

        planeCount = int(source.planeCount);
        vertexCount = int(source.vertexCount);

        for (int p = 0; p < planeCount; ++p) {
            planes[p] = source.planes[p];
            planeNeighborIds[p] = source.planeNeighborIds[p];
        }

        for (int v = 0; v < vertexCount; ++v) {
            vertices[v] = source.vertices[v];
        }

        return true;
    }

    template <class Limits>
    __device__ bool saveStorage(ConvexCellStorage<Limits>& target) const {
        if (planeCount > int(Limits::P) || vertexCount > int(Limits::V)) {
            return false;
        }
        target.planeCount = uint32_t(planeCount);
        target.vertexCount = uint32_t(vertexCount);
        for (int p = 0; p < planeCount; ++p) {
            target.planes[p] = planes[p];
            target.planeNeighborIds[p] = planeNeighborIds[p];
            target.planeAreas[p] = planeAreas[p];
        }
        for (int v = 0; v < vertexCount; ++v) {
            target.vertices[v] = vertices[v];
        }
        return true;
    }

    __device__ bool initializeDomainBox(const float4* domainPlanes) {
        if (planeCapacity < 6 || vertexCapacity < 8) {
            fail(RVDStatus::InvalidIndex);
            return false;
        }

        status = RVDStatus::Success;
        planeCount = 6;
        vertexCount = 8;

        for (uint32_t i = 0; i < 6; ++i) {
            planes[i] = domainPlanes[i];
            planeNeighborIds[i] = domainFaceTag(i);
            planeAreas[i] = 0.0;
        }

        vertices[0] = make_uint3(2, 5, 0);
        vertices[1] = make_uint3(5, 3, 0);
        vertices[2] = make_uint3(1, 5, 2);
        vertices[3] = make_uint3(5, 1, 3);
        vertices[4] = make_uint3(4, 2, 0);
        vertices[5] = make_uint3(4, 0, 3);
        vertices[6] = make_uint3(2, 4, 1);
        vertices[7] = make_uint3(4, 3, 1);

        return true;
    }

    __device__ void fail(RVDStatus newStatus) {
        if (status == RVDStatus::Success) {
            status = newStatus;
        }
    }

    __device__ float3 vertexPosition(uint3 triplet) const {
        if (triplet.x >= uint32_t(planeCount) || triplet.y >= uint32_t(planeCount) ||
            triplet.z >= uint32_t(planeCount)) {
            return make_float3(0.0f, 0.0f, 0.0f);
        }
        const float4 p0 = planes[triplet.x], p1 = planes[triplet.y], p2 = planes[triplet.z];
        const double m12 = double(p1.x) * double(p2.y) - double(p1.y) * double(p2.x);
        const double m13 = double(p1.x) * double(p2.z) - double(p1.z) * double(p2.x);
        const double m14 = double(p1.x) * double(p2.w) - double(p1.w) * double(p2.x);
        const double m23 = double(p1.y) * double(p2.z) - double(p1.z) * double(p2.y);
        const double m24 = double(p1.y) * double(p2.w) - double(p1.w) * double(p2.y);
        const double m34 = double(p1.z) * double(p2.w) - double(p1.w) * double(p2.z);
        const double x = -double(p0.w) * m23 - double(p0.y) * m34 + double(p0.z) * m24;
        const double y = double(p0.x) * m34 + double(p0.w) * m13 - double(p0.z) * m14;
        const double z = -double(p0.x) * m24 + double(p0.y) * m14 - double(p0.w) * m12;
        const double w = double(p0.x) * m23 - double(p0.y) * m13 + double(p0.z) * m12;
        return make_float3(float(x / w), float(y / w), float(z / w));
    }

    __device__ bool vertexOutside(uint3 triplet, uint32_t newPlane) {
        if (triplet.x >= uint32_t(planeCount) || triplet.y >= uint32_t(planeCount) ||
            triplet.z >= uint32_t(planeCount) || newPlane >= uint32_t(planeCount)) {
            fail(RVDStatus::InvalidIndex);
            return false;
        }
        const float4 p0 = planes[triplet.x], p1 = planes[triplet.y], p2 = planes[triplet.z];
        const float4 cut = planes[newPlane];

        if constexpr (Mode == PredicateMode::FilteredFloat) {
            const PredicateResult res = determinant4x4FilteredFloat(p0, p1, p2, cut);
            if (fabs(res.value) > res.errorBound) {
                return res.value > 0.0;
            }
            fail(RVDStatus::NeedsRobustPredicate);
            return false;
        } else {
            const PredicateResult res = determinant4x4FilteredDouble(p0, p1, p2, cut);
            return res.value > res.errorBound;
        }
    }

    __device__ bool newVertex(uint32_t a, uint32_t b, uint32_t c) {
        if (a >= uint32_t(planeCount) || b >= uint32_t(planeCount) || c >= uint32_t(planeCount)) {
            fail(RVDStatus::InvalidIndex);
            return false;
        }
        if (vertexCount >= int(vertexCapacity)) {
            fail(RVDStatus::VertexOverflow);
            return false;
        }
        vertices[vertexCount++] = make_uint3(a, b, c);
        return true;
    }

    __device__ static inline uint32_t planeAt(uint3 triplet, int index) {
        return index == 0 ? triplet.x : (index == 1 ? triplet.y : triplet.z);
    }

    __device__ uint32_t computeBoundary(int removedCount, uint32_t newPlane) {
        for (uint32_t i = 0; i < planeCapacity; ++i) boundaryNext[i] = EndOfList;
        uint32_t firstBoundary = EndOfList;
        int rejected = 0;
        int triangle = vertexCount;
        while (removedCount > 0) {
            bool isInBorder[3];
            bool nextIsOpposite[3];
            for (int edge = 0; edge < 3; edge++)
                isInBorder[edge] = boundaryNext[planeAt(vertices[triangle], edge)] != EndOfList;
            for (int edge = 0; edge < 3; edge++)
                nextIsOpposite[edge] = boundaryNext[planeAt(vertices[triangle], (edge + 1) % 3)] ==
                                       planeAt(vertices[triangle], edge);
            bool simple = true;
            for (int edge = 0; edge < 3; edge++) {
                if (!nextIsOpposite[edge] && !nextIsOpposite[(edge + 1) % 3] &&
                    isInBorder[(edge + 1) % 3]) simple = false;
            }
            if (!nextIsOpposite[0] && !nextIsOpposite[1] && !nextIsOpposite[2]) {
                if (firstBoundary == EndOfList) {
                    for (int edge = 0; edge < 3; edge++)
                        boundaryNext[planeAt(vertices[triangle], edge)] =
                            planeAt(vertices[triangle], (edge + 1) % 3);
                    firstBoundary = vertices[triangle].x;
                } else simple = false;
            }
            if (!simple) {
                ++triangle;
                if (triangle == vertexCount + removedCount) triangle = vertexCount;
                if (++rejected >= removedCount) {
                    fail(RVDStatus::InconsistentBoundary);
                    return firstBoundary;
                }
                continue;
            }
            rejected = 0;
            for (int edge = 0; edge < 3; edge++) {
                if (!nextIsOpposite[edge])
                    boundaryNext[planeAt(vertices[triangle], edge)] =
                        planeAt(vertices[triangle], (edge + 1) % 3);
            }
            for (int edge = 0; edge < 3; edge++) {
                if (nextIsOpposite[edge] && nextIsOpposite[(edge + 1) % 3]) {
                    const uint32_t plane = planeAt(vertices[triangle], (edge + 1) % 3);
                    if (firstBoundary == plane) firstBoundary = boundaryNext[plane];
                    boundaryNext[plane] = EndOfList;
                }
            }
            const uint3 temporary = vertices[triangle];
            vertices[triangle] = vertices[vertexCount + removedCount - 1];
            vertices[vertexCount + removedCount - 1] = temporary;
            triangle = vertexCount;
            --removedCount;
        }
        return firstBoundary;
    }

    __device__ bool clipLastPlane(bool* changed = nullptr) {
        if (changed) *changed = false;
        if (planeCount <= 0 || uint32_t(planeCount) > planeCapacity) {
            fail(RVDStatus::InvalidIndex);
            return false;
        }
        const uint32_t newPlane = uint32_t(planeCount - 1);
        int removedCount = 0;
        int vertex = 0;
        while (vertex < vertexCount) {
            if (vertexOutside(vertices[vertex], newPlane)) {
                --vertexCount;
                const uint3 temp = vertices[vertex];
                vertices[vertex] = vertices[vertexCount];
                vertices[vertexCount] = temp;
                ++removedCount;
            } else {
                ++vertex;
            }
        }
        if (status != RVDStatus::Success) return false;
        if (vertexCount == 0) {
            fail(RVDStatus::EmptyCell);
            return false;
        }
        if (removedCount == 0) {
            --planeCount;
            return true;
        }
        if (changed) *changed = true;
        const uint32_t firstBoundary = computeBoundary(removedCount, newPlane);
        if (status != RVDStatus::Success) return false;
        if (firstBoundary == EndOfList) return true;
        uint32_t current = firstBoundary;
        do {
            const uint32_t next = boundaryNext[current];
            if (!newVertex(newPlane, current, next)) return false;
            current = next;
        } while (current != firstBoundary);
        return true;
    }

    __device__ bool compactPlanes() {
        if (planeCount <= 0) return true;
        for (uint32_t i = 0; i < planeCapacity; ++i) boundaryNext[i] = 0;
        for (int v = 0; v < vertexCount; ++v) {
            const uint3 t = vertices[v];
            boundaryNext[t.x] = 1;
            boundaryNext[t.y] = 1;
            boundaryNext[t.z] = 1;
        }
        int newPlaneCount = 0;
        for (int p = 0; p < planeCount; ++p) {
            if (boundaryNext[p]) {
                if (newPlaneCount != p) {
                    planes[newPlaneCount] = planes[p];
                    planeNeighborIds[newPlaneCount] = planeNeighborIds[p];
                    boundaryNext[p] = uint32_t(newPlaneCount);
                } else {
                    boundaryNext[p] = uint32_t(p);
                }
                ++newPlaneCount;
            } else {
                boundaryNext[p] = EndOfList;
            }
        }
        planeCount = newPlaneCount;
        for (int v = 0; v < vertexCount; ++v) {
            uint3& t = vertices[v];
            t.x = boundaryNext[t.x];
            t.y = boundaryNext[t.y];
            t.z = boundaryNext[t.z];
            if (t.x == EndOfList || t.y == EndOfList || t.z == EndOfList) {
                fail(RVDStatus::InvalidIndex);
                return false;
            }
        }
        return true;
    }

    __device__ bool addPlane(float4 plane, int neighborId = NoPlaneNeighbor, bool* changed = nullptr) {
        if (uint32_t(planeCount) >= planeCapacity) {
            fail(RVDStatus::PlaneOverflow);
            return false;
        }
        if (!isfinite(plane.x) || !isfinite(plane.y) || !isfinite(plane.z) || !isfinite(plane.w)) {
            fail(RVDStatus::NonFiniteGeometry);
            return false;
        }
        planes[planeCount] = plane;
        planeNeighborIds[planeCount] = neighborId;
        planeAreas[planeCount] = 0.0;
        ++planeCount;
        return clipLastPlane(changed);
    }

    __device__ float maximumVertexDistanceSquared(float3 seed) {
        float maximum = 0.0f;
        for (int i = 0; i < vertexCount && status == RVDStatus::Success; ++i) {
            const float3 vertex = vertexPosition(vertices[i]);
            const float distanceSquared = lengthSquared(sub(vertex, seed));
            if (distanceSquared > maximum) maximum = distanceSquared;
        }
        return maximum;
    }
};

__device__ inline double determinant3Double(float3 a, float3 b, float3 c) {
    return double(a.x) * (double(b.y) * c.z - double(b.z) * c.y) -
           double(b.x) * (double(a.y) * c.z - double(a.z) * c.y) +
           double(c.x) * (double(a.y) * b.z - double(a.z) * b.y);
}

template <PredicateMode Mode>
__device__ double4 integrateCell(
    ConvexCell<Mode>& cell,
    float3 seed,
    int basePlaneCount = -1,
    double sign = 1.0)
{
    double4 bary = make_double4(0.0, 0.0, 0.0, 0.0);

    for (int vertex = 0;
         vertex < cell.vertexCount && cell.status == RVDStatus::Success;
         ++vertex) {

        const float3 a = cell.vertexPosition(cell.vertices[vertex]);
        const float3 ac = sub(seed, a);
        const uint3 planeIds = cell.vertices[vertex];
        const uint32_t ids[3] = {planeIds.x, planeIds.y, planeIds.z};

        float3 projected[3];
        float3 lines[3];

        for (int i = 0; i < 3; ++i) {
            const float4 p = cell.planes[ids[i]];
            const float3 normal = make_float3(p.x, p.y, p.z);
            const float n2 = lengthSquared(normal);

            projected[i] = n2 > 1e-30f
                ? sub(ac, mul(normal, dot3(ac, normal) / n2))
                : make_float3(0, 0, 0);
        }

        for (int i = 0; i < 3; ++i) {
            const float4 p0 = cell.planes[ids[i]];
            const float4 p1 = cell.planes[ids[(i + 1) % 3]];

            const float3 line = cross3(
                make_float3(p0.x, p0.y, p0.z),
                make_float3(p1.x, p1.y, p1.z));

            const float line2 = lengthSquared(line);
            const float n0Sq = p0.x*p0.x + p0.y*p0.y + p0.z*p0.z;
            const float n1Sq = p1.x*p1.x + p1.y*p1.y + p1.z*p1.z;
            constexpr float ParallelEpsilon2 = 1e-12f;
            lines[i] = line2 > ParallelEpsilon2 * n0Sq * n1Sq
                ? mul(line, dot3(ac, line) / line2)
                : make_float3(0, 0, 0);
        }

        for (int i = 0; i < 3; ++i) {
            for (int step = 0; step < 2; ++step) {
                const int j = (i + 3 - step) % 3;
                const double orientation = step ? 1.0 : -1.0;

                const double weight =
                    sign * orientation * determinant3Double(ac, projected[i], lines[j]);

                bary.w += weight;

                const double rx =
                    0.25 * (
                        double(ac.x) +
                        double(projected[i].x) +
                        double(lines[j].x)) -
                    double(ac.x);

                const double ry =
                    0.25 * (
                        double(ac.y) +
                        double(projected[i].y) +
                        double(lines[j].y)) -
                    double(ac.y);

                const double rz =
                    0.25 * (
                        double(ac.z) +
                        double(projected[i].z) +
                        double(lines[j].z)) -
                    double(ac.z);

                bary.x += weight * rx;
                bary.y += weight * ry;
                bary.z += weight * rz;

                const int tag = cell.planeNeighborIds[ids[i]];

                if (ids[i] < uint32_t(basePlaneCount) &&
                    (tag >= 0 || domainFaceFromTag(tag) >= 0)) {

                    const float4 plane = cell.planes[ids[i]];
                    const float3 normal =
                        make_float3(plane.x, plane.y, plane.z);

                    const double nx = double(plane.x);
                    const double ny = double(plane.y);
                    const double nz = double(plane.z);

                    const double normalLength =
                        sqrt(nx * nx + ny * ny + nz * nz);

                    if (!(normalLength > 0.0) || !isfinite(normalLength)) {
                        cell.fail(RVDStatus::NonFiniteGeometry);
                        return bary;
                    }

                    const double dArea =
                        0.5 *
                        sign *
                        orientation *
                        determinant3Double(
                            projected[i],
                            lines[j],
                            normal) /
                        normalLength;

                    cell.planeAreas[ids[i]] += dArea;
                }
            }
        }
    }

    if (!isfinite(bary.x) ||
        !isfinite(bary.y) ||
        !isfinite(bary.z) ||
        !isfinite(bary.w)) {
        cell.fail(RVDStatus::NonFiniteGeometry);
    }

    return bary;
}

static constexpr int MaxClippedPolygonVertices = DiscoveryLimits::P + 3;

template <int MaxVertices = MaxClippedPolygonVertices>
__device__ inline int clipPolygonByPlanes(
    const float3* inPolygon,
    int inCount,
    const float4* planes,
    int planeCount,
    float3* outPolygon,
    int maxOutVertices) {
    if (inCount < 3 || planeCount <= 0) return 0;
    float3 polygon[MaxClippedPolygonVertices];
    float3 clipped[MaxClippedPolygonVertices];
    const int initialCount = (inCount < MaxClippedPolygonVertices) ? inCount : MaxClippedPolygonVertices;
    for (int i = 0; i < initialCount; ++i) {
        polygon[i] = inPolygon[i];
    }
    int polygonCount = initialCount;

    for (int planeIndex = 0; planeIndex < planeCount && polygonCount >= 3; ++planeIndex) {
        const float4 plane = planes[planeIndex];
        const float normalLength = sqrtf(
            plane.x * plane.x +
            plane.y * plane.y +
            plane.z * plane.z);
        const float planeEpsilon = 1e-6f * normalLength;

        int clippedCount = 0;
        float3 previous = polygon[polygonCount - 1];
        float previousValue = plane.x * previous.x + plane.y * previous.y + plane.z * previous.z + plane.w;
        bool previousInside = previousValue >= -planeEpsilon;
        for (int vertex = 0; vertex < polygonCount; ++vertex) {
            const float3 current = polygon[vertex];
            const float currentValue = plane.x * current.x + plane.y * current.y + plane.z * current.z + plane.w;
            const bool currentInside = currentValue >= -planeEpsilon;
            if (currentInside != previousInside) {
                if (clippedCount >= MaxClippedPolygonVertices) return -1; // Hard geometry capacity error
                const float denominator = previousValue - currentValue;
                const float t = fabsf(denominator) > 1e-12f * normalLength
                    ? fminf(fmaxf(previousValue / denominator, 0.0f), 1.0f)
                    : 0.0f;
                clipped[clippedCount++] = add(previous, mul(sub(current, previous), t));
            }
            if (currentInside) {
                if (clippedCount >= MaxClippedPolygonVertices) return -1; // Hard geometry capacity error
                clipped[clippedCount++] = current;
            }
            previous = current;
            previousValue = currentValue;
            previousInside = currentInside;
        }
        polygonCount = clippedCount;
        for (int i = 0; i < polygonCount; ++i) polygon[i] = clipped[i];
    }
    if (polygonCount < 3) return 0;
    const int writeCount = (maxOutVertices > 0 && polygonCount < maxOutVertices) ? polygonCount : (maxOutVertices > 0 ? maxOutVertices : polygonCount);
    if (outPolygon) {
        for (int i = 0; i < writeCount; ++i) {
            outPolygon[i] = polygon[i];
        }
    }
    return polygonCount;
}

template <PredicateMode Mode>
__device__ inline int clipTriangleToCell(
    float3 v0, float3 v1, float3 v2,
    const ConvexCell<Mode>& cell,
    int staticPlaneCount,
    float3* outPolygon,
    int maxOutVertices) {
    float3 tri[3] = { v0, v1, v2 };
    return clipPolygonByPlanes(tri, 3, cell.planes, staticPlaneCount, outPolygon, maxOutVertices);
}

__device__ inline float polygonArea(const float3* polygon, int polygonCount) {
    if (polygonCount < 3) return 0.0f;
    float doubleArea = 0.0f;
    for (int i = 1; i + 1 < polygonCount; ++i) {
        doubleArea += sqrtf(lengthSquared(cross3(sub(polygon[i], polygon[0]), sub(polygon[i + 1], polygon[0]))));
    }
    return 0.5f * doubleArea;
}

__device__ inline float3 polygonCentroid(const float3* polygon, int polygonCount, float totalArea) {
    if (polygonCount < 3 || totalArea <= 1e-12f) return make_float3(0.0f, 0.0f, 0.0f);
    double3 sum = make_double3(0.0, 0.0, 0.0);
    const float3 v0 = polygon[0];
    for (int i = 1; i + 1 < polygonCount; ++i) {
        const float3 v1 = polygon[i];
        const float3 v2 = polygon[i + 1];
        const float3 c = cross3(sub(v1, v0), sub(v2, v0));
        const float a = 0.5f * sqrtf(lengthSquared(c));
        if (a > 1e-12f) {
            sum.x += (double(v0.x) + double(v1.x) + double(v2.x)) * (double(a) / 3.0);
            sum.y += (double(v0.y) + double(v1.y) + double(v2.y)) * (double(a) / 3.0);
            sum.z += (double(v0.z) + double(v1.z) + double(v2.z)) * (double(a) / 3.0);
        }
    }
    return make_float3(float(sum.x / totalArea), float(sum.y / totalArea), float(sum.z / totalArea));
}

} // namespace voronoi::rvdDevice
