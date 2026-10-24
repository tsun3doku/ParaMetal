#pragma once

#include "RVD.hpp"
#include "RVDStatus.cuh"
#include "RVDPredicates.cuh"
#include "../../cuda/CudaMath.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace voronoi::rvdDevice {

using cudaMath::add3;
using cudaMath::sub3;
using cudaMath::mul3;
using cudaMath::dot3;
using cudaMath::cross3;
using cudaMath::lengthSquared3;

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

struct DiscoveryLimits {
    static constexpr int P = int(RVDRetryPlaneCount);
    static constexpr int V = int(RVDRetryVertexCount);
    static constexpr int T = int(RVDRetryTriangleCount);
};

struct OverflowLimits {
    static constexpr int P = int(RVDOverflowPlaneCount);
    static constexpr int V = int(RVDOverflowVertexCount);
};

template <class Limits>
struct ConvexCellStorageT {
    alignas(16) uint3 vertices[Limits::V];
    alignas(16) double4 planes[Limits::P];
    int planeNeighborIds[Limits::P];
    uint32_t boundaryNext[Limits::P];
    double planeAreas[Limits::P];
    double planeAreaAbsSums[Limits::P];
    double3 planeMoments[Limits::P];
    uint32_t vertexCount = 0;
    uint32_t planeCount = 0;
};

using ConvexCellStorage = ConvexCellStorageT<DiscoveryLimits>;
using ConvexCellOverflowStorage = ConvexCellStorageT<OverflowLimits>;

struct ConvexCell {
    double4* planes = nullptr;
    int* planeNeighborIds = nullptr;
    uint32_t* boundaryNext = nullptr;
    uint3* vertices = nullptr;
    double* planeAreas = nullptr;
    double* planeAreaAbsSums = nullptr;
    double3* planeMoments = nullptr;
    uint32_t planeCapacity = 0;
    uint32_t vertexCapacity = 0;
    int vertexCount = 0;
    int planeCount = 0;
    RVDStatus status = RVDStatus::Success;

    template <typename StorageT>
    __device__ void attachStorage(StorageT& storage,
                                  uint32_t planeLimit = DiscoveryLimits::P,
                                  uint32_t vertexLimit = DiscoveryLimits::V) {
        planes = storage.planes;
        planeNeighborIds = storage.planeNeighborIds;
        boundaryNext = storage.boundaryNext;
        vertices = storage.vertices;
        planeAreas = storage.planeAreas;
        planeAreaAbsSums = storage.planeAreaAbsSums;
        planeMoments = storage.planeMoments;
        planeCapacity = planeLimit;
        vertexCapacity = vertexLimit;
    }

    template <typename SourceT>
    __device__ bool loadStorage(const SourceT& source) {
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

    template <typename TargetT>
    __device__ bool saveStorage(TargetT& target) const {
        if (planeCount > int(planeCapacity) || vertexCount > int(vertexCapacity)) {
            return false;
        }
        target.planeCount = uint32_t(planeCount);
        target.vertexCount = uint32_t(vertexCount);
        for (int p = 0; p < planeCount; ++p) {
            target.planes[p] = planes[p];
            target.planeNeighborIds[p] = planeNeighborIds[p];
            target.planeAreas[p] = planeAreas[p];
            target.planeAreaAbsSums[p] = planeAreaAbsSums[p];
            if (planeMoments != nullptr) target.planeMoments[p] = planeMoments[p];
        }
        for (int v = 0; v < vertexCount; ++v) {
            target.vertices[v] = vertices[v];
        }
        return true;
    }

    __device__ void clearPlaneMeasures(int count) {
        for (int p = 0; p < count; ++p) {
            planeAreas[p] = 0.0;
            planeAreaAbsSums[p] = 0.0;
            if (planeMoments != nullptr) {
                planeMoments[p] = make_double3(0.0, 0.0, 0.0);
            }
        }
    }

    __device__ bool initializeDomainBox(const double4* domainPlanes) {
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
            planeAreaAbsSums[i] = 0.0;
            if (planeMoments != nullptr) {
                planeMoments[i] = make_double3(0.0, 0.0, 0.0);
            }
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

    __device__ double3 vertexPosition(uint3 triplet) const {
        if (triplet.x >= uint32_t(planeCount) || triplet.y >= uint32_t(planeCount) ||
            triplet.z >= uint32_t(planeCount)) {
            return make_double3(0.0, 0.0, 0.0);
        }
        const double4 p0 = planes[triplet.x], p1 = planes[triplet.y], p2 = planes[triplet.z];
        const double m12 = p1.x * p2.y - p1.y * p2.x;
        const double m13 = p1.x * p2.z - p1.z * p2.x;
        const double m14 = p1.x * p2.w - p1.w * p2.x;
        const double m23 = p1.y * p2.z - p1.z * p2.y;
        const double m24 = p1.y * p2.w - p1.w * p2.y;
        const double m34 = p1.z * p2.w - p1.w * p2.z;
        const double x = -p0.w * m23 - p0.y * m34 + p0.z * m24;
        const double y = p0.x * m34 + p0.w * m13 - p0.z * m14;
        const double z = -p0.x * m24 + p0.y * m14 - p0.w * m12;
        const double w = p0.x * m23 - p0.y * m13 + p0.z * m12;
        return make_double3(x / w, y / w, z / w);
    }

    __device__ bool vertexOutside(uint3 triplet, uint32_t newPlane) {
        if (triplet.x >= uint32_t(planeCount) || triplet.y >= uint32_t(planeCount) ||
            triplet.z >= uint32_t(planeCount) || newPlane >= uint32_t(planeCount)) {
            fail(RVDStatus::InvalidIndex);
            return false;
        }
        const double4 p0 = planes[triplet.x], p1 = planes[triplet.y], p2 = planes[triplet.z];
        const double4 cut = planes[newPlane];

        const PredicateResult res = determinant4x4FilteredDouble(p0, p1, p2, cut);
        if (fabs(res.value) > res.errorBound) {
            return res.value > 0.0;
        }

        const int exactSign = det4x4ExactSign(p0, p1, p2, cut);
        return exactSign > 0;
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
                    planeAreas[newPlaneCount] = planeAreas[p];
                    planeAreaAbsSums[newPlaneCount] = planeAreaAbsSums[p];
                    if (planeMoments != nullptr) {
                        planeMoments[newPlaneCount] = planeMoments[p];
                    }
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

    __device__ bool addPlane(double4 plane, int neighborId = NoPlaneNeighbor, bool* changed = nullptr) {
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
        planeAreaAbsSums[planeCount] = 0.0;
        if (planeMoments != nullptr) {
            planeMoments[planeCount] = make_double3(0.0, 0.0, 0.0);
        }
        ++planeCount;
        return clipLastPlane(changed);
    }

    template <typename SeedT>
    __device__ double maximumVertexDistanceSquared(SeedT seed) {
        double maximum = 0.0;
        for (int i = 0; i < vertexCount && status == RVDStatus::Success; ++i) {
            const auto vertex = vertexPosition(vertices[i]);
            const double dx = double(vertex.x) - double(seed.x);
            const double dy = double(vertex.y) - double(seed.y);
            const double dz = double(vertex.z) - double(seed.z);
            const double distanceSquared = dx * dx + dy * dy + dz * dz;
            if (distanceSquared > maximum) maximum = distanceSquared;
        }
        return maximum;
    }
};

__device__ inline double determinant3Double(double3 a, double3 b, double3 c) {
    return a.x * (b.y * c.z - b.z * c.y) -
           b.x * (a.y * c.z - a.z * c.y) +
           c.x * (a.y * b.z - a.z * b.y);
}

__device__ inline double4 integrateCell(
    ConvexCell& cell,
    float3 seedF3,
    int basePlaneCount = -1,
    double sign = 1.0)
{
    const double3 seed = make_double3(seedF3.x, seedF3.y, seedF3.z);
    double4 bary = make_double4(0.0, 0.0, 0.0, 0.0);

    for (int vertex = 0;
         vertex < cell.vertexCount && cell.status == RVDStatus::Success;
         ++vertex) {

        const auto aPos = cell.vertexPosition(cell.vertices[vertex]);
        const double3 a = make_double3(aPos.x, aPos.y, aPos.z);
        const double3 ac = sub3(seed, a);
        const uint3 planeIds = cell.vertices[vertex];
        const uint32_t ids[3] = {planeIds.x, planeIds.y, planeIds.z};

        double3 projected[3];
        double3 lines[3];

        for (int i = 0; i < 3; ++i) {
            const double4 p = cell.planes[ids[i]];
            const double3 normal = make_double3(p.x, p.y, p.z);
            const double n2 = lengthSquared3(normal);

            projected[i] = n2 > 1e-30
                ? sub3(ac, mul3(normal, dot3(ac, normal) / n2))
                : make_double3(0, 0, 0);
        }

        for (int i = 0; i < 3; ++i) {
            const double4 p0 = cell.planes[ids[i]];
            const double4 p1 = cell.planes[ids[(i + 1) % 3]];

            const double3 line = cross3(
                make_double3(p0.x, p0.y, p0.z),
                make_double3(p1.x, p1.y, p1.z));

            const double line2 = lengthSquared3(line);
            const double n0Sq = double(p0.x)*double(p0.x) + double(p0.y)*double(p0.y) + double(p0.z)*double(p0.z);
            const double n1Sq = double(p1.x)*double(p1.x) + double(p1.y)*double(p1.y) + double(p1.z)*double(p1.z);
            constexpr double ParallelEpsilon2 = 1e-12;
            lines[i] = line2 > ParallelEpsilon2 * n0Sq * n1Sq
                ? mul3(line, dot3(ac, line) / line2)
                : make_double3(0, 0, 0);
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
                        ac.x +
                        projected[i].x +
                        lines[j].x) -
                    ac.x;

                const double ry =
                    0.25 * (
                        ac.y +
                        projected[i].y +
                        lines[j].y) -
                    ac.y;

                const double rz =
                    0.25 * (
                        ac.z +
                        projected[i].z +
                        lines[j].z) -
                    ac.z;

                bary.x += weight * rx;
                bary.y += weight * ry;
                bary.z += weight * rz;

                const int tag = cell.planeNeighborIds[ids[i]];

                if (ids[i] < uint32_t(basePlaneCount) &&
                    (tag >= 0 || domainFaceFromTag(tag) >= 0)) {

                    const double4 plane = cell.planes[ids[i]];
                    const double3 normal = make_double3(plane.x, plane.y, plane.z);
                    const double normalLength = sqrt(lengthSquared3(normal));

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
                    cell.planeAreaAbsSums[ids[i]] += fabs(dArea);

                    if (cell.planeMoments != nullptr) {
                        const double triX = a.x + (projected[i].x + lines[j].x) / 3.0;
                        const double triY = a.y + (projected[i].y + lines[j].y) / 3.0;
                        const double triZ = a.z + (projected[i].z + lines[j].z) / 3.0;
                        cell.planeMoments[ids[i]].x += dArea * triX;
                        cell.planeMoments[ids[i]].y += dArea * triY;
                        cell.planeMoments[ids[i]].z += dArea * triZ;
                    }
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

static constexpr int MaxClippedPolygonVertices = OverflowLimits::P + 3;

struct PlaneClippedSegment {
    double3 a;
    double3 b;
    int planeIndex;
};

__device__ inline int clipPolygonByPlanes(
    const double3* input,
    int inputCount,
    const double4* planes,
    int planeCount,
    double3* output,
    int capacity,
    PlaneClippedSegment* segments = nullptr,
    int* segmentCount = nullptr,
    int segmentCapacity = 0) {
    if (inputCount < 3 || planeCount <= 0 || capacity < 3) return 0;
    double3 polygon[MaxClippedPolygonVertices];
    double3 clipped[MaxClippedPolygonVertices];
    bool clippedOnPlane[MaxClippedPolygonVertices];
    const int initialCount = (inputCount < MaxClippedPolygonVertices) ? inputCount : MaxClippedPolygonVertices;
    for (int i = 0; i < initialCount; ++i) {
        polygon[i] = input[i];
    }
    int polygonCount = initialCount;
    if (segmentCount != nullptr) *segmentCount = 0;

    for (int planeIndex = 0; planeIndex < planeCount && polygonCount >= 3; ++planeIndex) {
        const double4 plane = planes[planeIndex];
        const double normalLength = sqrt(
            plane.x * plane.x +
            plane.y * plane.y +
            plane.z * plane.z);
        if (!(normalLength > 0.0) || !isfinite(normalLength)) return -1;
        const double planeEpsilon = 1e-10 * normalLength;

        int clippedCount = 0;
        double3 previous = polygon[polygonCount - 1];
        double prevDist = plane.x * previous.x + plane.y * previous.y + plane.z * previous.z + plane.w;
        if (fabs(prevDist) <= planeEpsilon) prevDist = 0.0;
        bool previousInside = prevDist >= 0.0;

        for (int vertex = 0; vertex < polygonCount; ++vertex) {
            const double3 current = polygon[vertex];
            double currDist = plane.x * current.x + plane.y * current.y + plane.z * current.z + plane.w;
            if (fabs(currDist) <= planeEpsilon) currDist = 0.0;
            const bool currentInside = currDist >= 0.0;

            if (currentInside != previousInside) {
                if ((prevDist > 0.0 && currDist < 0.0) || (prevDist < 0.0 && currDist > 0.0)) {
                    if (clippedCount >= MaxClippedPolygonVertices) return -1; // Hard geometry capacity error
                    const double denominator = prevDist - currDist;
                    if (fabs(denominator) <= 1e-15 * normalLength) {
                        return -1;
                    }
                    const double t = prevDist / denominator;
                    if (!isfinite(t) || t < 0.0 || t > 1.0) {
                        return -1;
                    }
                    clipped[clippedCount] = add3(previous, mul3(sub3(current, previous), t));
                    clippedOnPlane[clippedCount] = true;
                    ++clippedCount;
                }
            }
            if (currentInside) {
                if (clippedCount >= MaxClippedPolygonVertices) return -1; // Hard geometry capacity error
                clipped[clippedCount] = current;
                clippedOnPlane[clippedCount] = (currDist == 0.0);
                ++clippedCount;
            }
            previous = current;
            prevDist = currDist;
            previousInside = currentInside;
        }

        if (segments != nullptr && segmentCount != nullptr) {
            for (int i = 0; i < clippedCount; ++i) {
                const int j = (i + 1) % clippedCount;
                if (!clippedOnPlane[i] || !clippedOnPlane[j]) continue;
                if (lengthSquared3(sub3(clipped[i], clipped[j])) <= 0.0) continue;
                if (*segmentCount >= segmentCapacity) return -1;
                segments[*segmentCount] = {clipped[i], clipped[j], planeIndex};
                ++(*segmentCount);
            }
        }

        polygonCount = clippedCount;
        for (int i = 0; i < polygonCount; ++i) polygon[i] = clipped[i];
    }
    if (polygonCount < 3) return 0;
    if (polygonCount > capacity) return -1;
    if (output) {
        for (int i = 0; i < polygonCount; ++i) {
            output[i] = polygon[i];
        }
    }
    return polygonCount;
}

__device__ inline int clipTriangleToCellSegments(
    double3 a,
    double3 b,
    double3 c,
    const ConvexCell& cell,
    int planeCount,
    double3* output,
    int capacity,
    PlaneClippedSegment* segments,
    int* segmentCount,
    int segmentCapacity) {
    const double3 tri[3] = { a, b, c };
    return clipPolygonByPlanes(tri, 3, cell.planes, planeCount, output, capacity,
                               segments, segmentCount, segmentCapacity);
}

__device__ inline bool integratePolygon(
    const double3* vertices,
    int count,
    double& area,
    double3& moment) {
    area = 0.0;
    moment = make_double3(0.0, 0.0, 0.0);

    if (count < 3)
        return true;

    const double3 a = vertices[0];

    for (int i = 1; i + 1 < count; ++i) {
        const double3 b = vertices[i];
        const double3 c = vertices[i + 1];

        const double3 cross = cross3(sub3(b, a), sub3(c, a));
        const double triangleArea =
            0.5 * sqrt(dot3(cross, cross));

        area += triangleArea;

        const double weight = triangleArea / 3.0;

        moment = add3(
            moment,
            mul3(add3(add3(a, b), c), weight));
    }

    return isfinite(area) &&
           isfinite(moment.x) &&
           isfinite(moment.y) &&
           isfinite(moment.z);
}

} // namespace voronoi::rvdDevice
