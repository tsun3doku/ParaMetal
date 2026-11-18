#include "CFDSliceGeometry.hpp"

#include "mapbox/earcut.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>

namespace cfd {

bool CFDSliceGeometrySource::isValid() const {
    if (fragmentCount == 0 || fluidNodeCount == 0 ||
        fluidNodeCount > fragmentCount ||
        globalFragmentToFluidNode.size() != fragmentCount ||
        faceFragmentA.empty() ||
        faceLoopOffsets.empty() || loopVertexOffsets.empty()) {
        return false;
    }
    if (faceFragmentA.size() != faceFragmentB.size() ||
        faceFragmentA.size() != faceAreaVectorA.size() ||
        faceFragmentA.size() != faceAreaVectorB.size() ||
        faceLoopOffsets.size() != faceFragmentA.size() + 1 ||
        faceLoopOffsets.front() != 0 || faceLoopOffsets.back() + 1 != loopVertexOffsets.size() ||
        loopVertexOffsets.empty() || loopVertexOffsets.front() != 0 ||
        loopVertices.size() != loopVertexOffsets.back()) {
        return false;
    }
    for (size_t i = 1; i < faceLoopOffsets.size(); ++i) {
        if (faceLoopOffsets[i] < faceLoopOffsets[i - 1]) return false;
    }
    for (size_t i = 1; i < loopVertexOffsets.size(); ++i) {
        if (loopVertexOffsets[i] < loopVertexOffsets[i - 1]) return false;
    }
    for (const auto& p : loopVertices) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) return false;
    }

    for (size_t face = 0; face < faceFragmentA.size(); ++face) {
        const uint32_t fragments[2] = {faceFragmentA[face], faceFragmentB[face]};
        for (uint32_t fragment : fragments) {
            if (fragment != InvalidFragment && fragment >= fragmentCount) return false;
        }
    }

    std::vector<uint8_t> fluidNodes(fluidNodeCount, 0);
    uint32_t mappedNodes = 0;
    for (uint32_t node : globalFragmentToFluidNode) {
        if (node == NonFluidNode) continue;
        if (node >= fluidNodeCount || fluidNodes[node] != 0) return false;
        fluidNodes[node] = 1;
        ++mappedNodes;
    }
    return mappedNodes == fluidNodeCount;
}

bool buildCFDSliceMesh(
    const CFDSliceGeometrySource& source,
    uint32_t axis,
    double coordinate,
    double domainScale,
    CFDSliceMesh& output,
    std::string* error) {
    output = {};
    if (axis > 2 || !std::isfinite(coordinate) || !std::isfinite(domainScale) ||
        !(domainScale > 0.0) || !source.isValid()) {
        if (error) *error = "invalid geometry source or plane";
        return false;
    }

    const double tolerance = std::max(domainScale * 1e-10, 1e-13);
    const double areaTolerance = tolerance * tolerance;
    const glm::dvec3 n = axisNormal(axis);
    std::vector<std::vector<LineSegment3D>> segmentsByFragment(source.globalFragmentToFluidNode.size());
    std::map<QuantizedKey3D, glm::dvec3> canonicalPoints;

    for (size_t face = 0; face < source.faceFragmentA.size(); ++face) {
        const uint32_t firstLoop = source.faceLoopOffsets[face];
        const uint32_t endLoop = source.faceLoopOffsets[face + 1];
        const glm::dvec3 normals[2] = {source.faceAreaVectorA[face], source.faceAreaVectorB[face]};
        const uint32_t fragments[2] = {source.faceFragmentA[face], source.faceFragmentB[face]};

        std::vector<glm::dvec3> hits;
        for (uint32_t loopIndex = firstLoop; loopIndex < endLoop; ++loopIndex) {
            const uint32_t begin = source.loopVertexOffsets[loopIndex];
            const uint32_t end = source.loopVertexOffsets[loopIndex + 1];
            if (end - begin < 3) continue;
            for (uint32_t edge = begin; edge < end; ++edge) {
                const glm::dvec3& p0 = source.loopVertices[edge];
                const glm::dvec3& p1 = source.loopVertices[edge + 1 == end ? begin : edge + 1];
                const double h0 = p0[axis] - coordinate;
                const double h1 = p1[axis] - coordinate;
                const bool on0 = std::abs(h0) <= tolerance;
                const bool on1 = std::abs(h1) <= tolerance;
                if (on0 && on1) continue; // coplanar edge handled by adjacent faces
                if (on0) {
                    hits.push_back(p0);
                } else if (on1) {
                    hits.push_back(p1);
                } else if ((h0 < 0.0) != (h1 < 0.0)) {
                    const double t = h0 / (h0 - h1);
                    hits.push_back(p0 + t * (p1 - p0));
                }
            }
        }
        std::sort(hits.begin(), hits.end(), [&](const glm::dvec3& a, const glm::dvec3& b) {
            const Point2D pa = projectPointTo2D(a, axis);
            const Point2D pb = projectPointTo2D(b, axis);
            return pa < pb;
        });
        hits.erase(std::unique(hits.begin(), hits.end(), [&](const glm::dvec3& a, const glm::dvec3& b) {
            return glm::length(a - b) <= tolerance;
        }), hits.end());
        if (hits.size() < 2) continue;

        // Even-odd polygon fill across loop intersections
        for (size_t hit = 0; hit + 1 < hits.size(); hit += 2) {
            const glm::dvec3 hitA = hits[hit];
            const glm::dvec3 hitB = hits[hit + 1];
            const glm::dvec3 edgeDirection = hitB - hitA;
            if (glm::length(glm::cross(edgeDirection, n)) <= tolerance) continue;
            for (int side = 0; side < 2; ++side) {
                const uint32_t fragment = fragments[side];
                if (fragment >= segmentsByFragment.size() ||
                    source.globalFragmentToFluidNode[fragment] >= source.fluidNodeCount) continue;
                glm::dvec3 outward = normals[side];
                const double normalLength = glm::length(outward);
                if (!(normalLength > 0.0)) continue;
                outward /= normalLength;
                const glm::dvec3 desired = glm::cross(n, outward);
                glm::dvec3 a = hitA;
                glm::dvec3 b = hitB;
                if (glm::dot(b - a, desired) < 0.0) std::swap(a, b);
                const QuantizedKey3D ka = quantizePoint3D(a, tolerance);
                const QuantizedKey3D kb = quantizePoint3D(b, tolerance);
                canonicalPoints.try_emplace(ka, a);
                canonicalPoints.try_emplace(kb, b);
                segmentsByFragment[fragment].push_back({canonicalPoints[ka], canonicalPoints[kb]});
            }
        }
    }

    std::vector<std::vector<CFDSliceLoop>> loopsByFragment(segmentsByFragment.size());
    for (uint32_t fragment = 0; fragment < segmentsByFragment.size(); ++fragment) {
        const auto& segments = segmentsByFragment[fragment];
        std::map<QuantizedKey3D, std::vector<size_t>> outgoing;
        std::vector<QuantizedKey3D> starts(segments.size());
        std::vector<QuantizedKey3D> ends(segments.size());
        for (size_t i = 0; i < segments.size(); ++i) {
            starts[i] = quantizePoint3D(segments[i].a, tolerance);
            ends[i] = quantizePoint3D(segments[i].b, tolerance);
            outgoing[starts[i]].push_back(i);
        }
        std::vector<uint8_t> used(segments.size(), 0);
        for (size_t seed = 0; seed < segments.size(); ++seed) {
            if (used[seed]) continue;
            CFDSliceLoop loop;
            size_t current = seed;
            const QuantizedKey3D startKey = starts[seed];
            bool closed = false;
            for (size_t guard = 0; guard <= segments.size(); ++guard) {
                if (used[current]) break;
                used[current] = 1;
                loop.points.push_back(segments[current].a);
                const QuantizedKey3D endKey = ends[current];
                if (endKey == startKey) {
                    closed = true;
                    break;
                }
                const auto it = outgoing.find(endKey);
                if (it == outgoing.end() || it->second.size() != 1) break;
                current = it->second.front();
            }
            if (!closed || loop.points.size() < 3) {
                if (error) *error = "slice segment graph is open or branches at a cut vertex";
                output = {};
                return false;
            }
            loop.projected.reserve(loop.points.size());
            for (const auto& p : loop.points) loop.projected.push_back(projectPointTo2D(p, axis));
            loop.signedArea = polygonDoubleArea2D(loop.projected);
            if (std::abs(loop.signedArea) > areaTolerance) loopsByFragment[fragment].push_back(std::move(loop));
        }
    }

    for (uint32_t fragment = 0; fragment < loopsByFragment.size(); ++fragment) {
        auto& loops = loopsByFragment[fragment];
        for (size_t i = 0; i < loops.size(); ++i) {
            const Point2D probe = loops[i].projected.front();
            double parentArea = std::numeric_limits<double>::infinity();
            for (size_t j = 0; j < loops.size(); ++j) {
                if (i == j || std::abs(loops[j].signedArea) <= std::abs(loops[i].signedArea)) continue;
                if (pointInPolygon2D(probe, loops[j].projected) &&
                    std::abs(loops[j].signedArea) < parentArea) {
                    parentArea = std::abs(loops[j].signedArea);
                    loops[i].parent = static_cast<int>(j);
                }
            }
        }
        for (size_t i = 0; i < loops.size(); ++i) {
            int parent = loops[i].parent;
            int depth = 0;
            size_t guard = 0;
            while (parent >= 0 && guard++ < loops.size()) {
                ++depth;
                parent = loops[size_t(parent)].parent;
            }
            loops[i].depth = depth;
        }

        for (size_t outer = 0; outer < loops.size(); ++outer) {
            if ((loops[outer].depth & 1) != 0) continue;
            std::vector<std::vector<Point2D>> polygon;
            std::vector<size_t> ringLoops;
            polygon.push_back(loops[outer].projected);
            ringLoops.push_back(outer);
            for (size_t hole = 0; hole < loops.size(); ++hole) {
                if (loops[hole].parent == static_cast<int>(outer) && loops[hole].depth == loops[outer].depth + 1) {
                    polygon.push_back(loops[hole].projected);
                    ringLoops.push_back(hole);
                }
            }
            std::vector<uint32_t> indices;
            try {
                indices = mapbox::earcut<uint32_t>(polygon);
            } catch (...) {
                if (error) *error = "polygon triangulation failed";
                output = {};
                return false;
            }
            std::vector<glm::dvec3> flatPoints;
            flatPoints.reserve([&]() { size_t n = 0; for (size_t r : ringLoops) n += loops[r].points.size(); return n; }());
            for (size_t r : ringLoops) {
                flatPoints.insert(flatPoints.end(), loops[r].points.begin(), loops[r].points.end());
                for (size_t i = 0; i < loops[r].points.size(); ++i) {
                    const auto& a = loops[r].points[i];
                    const auto& b = loops[r].points[(i + 1) % loops[r].points.size()];
                    const uint32_t node = source.globalFragmentToFluidNode[fragment];
                    output.perimeterLines.push_back({glm::vec3(a), node});
                    output.perimeterLines.push_back({glm::vec3(b), node});
                }
            }
            const uint32_t node = source.globalFragmentToFluidNode[fragment];
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                const uint32_t ia = indices[i], ib = indices[i + 1], ic = indices[i + 2];
                if (ia >= flatPoints.size() || ib >= flatPoints.size() || ic >= flatPoints.size()) {
                    if (error) *error = "triangulator returned an invalid vertex index";
                    output = {};
                    return false;
                }
                const glm::dvec3& a = flatPoints[ia];
                const glm::dvec3& b = flatPoints[ib];
                const glm::dvec3& c = flatPoints[ic];
                const double triArea = glm::length(glm::cross(b - a, c - a));
                if (!(triArea > areaTolerance)) continue;
                output.triangles.push_back({glm::vec3(a), node});
                output.triangles.push_back({glm::vec3(b), node});
                output.triangles.push_back({glm::vec3(c), node});
            }
        }
    }

    return true;
}

} // namespace cfd
