#include "IsocontourBuilder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <utility>

namespace {

struct EdgeKey {
    uint32_t a = 0;
    uint32_t b = 0;
    bool operator==(const EdgeKey& other) const noexcept {
        return a == other.a && b == other.b;
    }
};

struct EdgeKeyHash {
    size_t operator()(const EdgeKey& key) const noexcept {
        const size_t first = std::hash<uint32_t>{}(key.a);
        const size_t second = std::hash<uint32_t>{}(key.b);
        return first ^ (second + 0x9e3779b9u + (first << 6) + (first >> 2));
    }
};

EdgeKey edgeKey(uint32_t a, uint32_t b) {
    return a < b ? EdgeKey{a, b} : EdgeKey{b, a};
}

struct MeshEdge {
    uint32_t a = 0;
    uint32_t b = 0;
};

struct Segment {
    uint32_t a = 0;
    uint32_t b = 0;
};

} // namespace

std::vector<mesh::Polyline> IsocontourBuilder::build(
    const std::vector<glm::vec3>& positions,
    const std::vector<uint32_t>& triangleIndices,
    const std::vector<float>& scalarValues,
    const std::vector<float>& isoValues) {
    std::vector<mesh::Polyline> result;
    if (positions.empty() || positions.size() != scalarValues.size() ||
        triangleIndices.size() < 3 || triangleIndices.size() % 3 != 0) {
        return result;
    }

    const size_t triangleCount = triangleIndices.size() / 3;
    std::unordered_map<EdgeKey, uint32_t, EdgeKeyHash> edgeIndices;
    std::vector<MeshEdge> edges;
    std::vector<std::array<int32_t, 3>> triangleEdges(
        triangleCount, std::array<int32_t, 3>{-1, -1, -1});
    edgeIndices.reserve(triangleCount * 3 / 2);
    edges.reserve(triangleCount * 3 / 2);

    for (size_t triangle = 0; triangle < triangleCount; ++triangle) {
        const uint32_t vertex[3] = {
            triangleIndices[triangle * 3],
            triangleIndices[triangle * 3 + 1],
            triangleIndices[triangle * 3 + 2]};
        if (vertex[0] >= positions.size() || vertex[1] >= positions.size() ||
            vertex[2] >= positions.size()) {
            continue;
        }

        for (uint32_t localEdge = 0; localEdge < 3; ++localEdge) {
            const uint32_t a = vertex[localEdge];
            const uint32_t b = vertex[(localEdge + 1) % 3];
            if (a == b) continue;

            const EdgeKey key = edgeKey(a, b);
            auto [it, inserted] = edgeIndices.emplace(
                key, static_cast<uint32_t>(edges.size()));
            if (inserted) edges.push_back({key.a, key.b});
            triangleEdges[triangle][localEdge] = static_cast<int32_t>(it->second);
        }
    }

    for (size_t isoIndex = 0; isoIndex < isoValues.size(); ++isoIndex) {
        const float level = isoValues[isoIndex];
        if (!std::isfinite(level)) continue;

        constexpr float epsilon = 1e-6f;
        std::vector<float> effective(scalarValues.size());
        bool validScalars = true;
        for (size_t i = 0; i < scalarValues.size(); ++i) {
            const float value = scalarValues[i];
            if (!std::isfinite(value)) {
                validScalars = false;
                break;
            }
            effective[i] = std::abs(value - level) < epsilon
                ? level + epsilon
                : value;
        }
        if (!validScalars) continue;

        std::vector<glm::vec3> crossingPositions(edges.size());
        std::vector<uint8_t> crossingValid(edges.size(), 0);
        for (size_t edgeIndex = 0; edgeIndex < edges.size(); ++edgeIndex) {
            const MeshEdge& edge = edges[edgeIndex];
            const float s0 = effective[edge.a];
            const float s1 = effective[edge.b];
            if ((s0 > level) == (s1 > level)) continue;

            const float denominator = s1 - s0;
            if (denominator == 0.0f) continue;
            const float t = std::clamp((level - s0) / denominator, 0.0f, 1.0f);
            const glm::vec3 point = positions[edge.a] + t * (positions[edge.b] - positions[edge.a]);
            if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
                !std::isfinite(point.z)) {
                continue;
            }
            crossingPositions[edgeIndex] = point;
            crossingValid[edgeIndex] = 1;
        }

        std::vector<Segment> segments;
        std::vector<std::vector<uint32_t>> incident(edges.size());
        for (size_t triangle = 0; triangle < triangleCount; ++triangle) {
            uint32_t crossed[3]{};
            uint32_t count = 0;
            for (int32_t edgeIndex : triangleEdges[triangle]) {
                if (edgeIndex < 0 || !crossingValid[static_cast<size_t>(edgeIndex)]) continue;
                const uint32_t crossing = static_cast<uint32_t>(edgeIndex);
                if (count == 0 || crossed[count - 1] != crossing) crossed[count++] = crossing;
            }
            if (count != 2 || crossed[0] == crossed[1]) continue;

            const uint32_t segmentIndex = static_cast<uint32_t>(segments.size());
            segments.push_back({crossed[0], crossed[1]});
            incident[crossed[0]].push_back(segmentIndex);
            incident[crossed[1]].push_back(segmentIndex);
        }

        std::vector<uint8_t> segmentVisited(segments.size(), 0);
        auto trace = [&](uint32_t startNode, uint32_t firstSegment) {
            mesh::Polyline line;
            line.points.push_back(crossingPositions[startNode]);

            uint32_t node = startNode;
            uint32_t segmentIndex = firstSegment;
            while (segmentIndex < segments.size() && !segmentVisited[segmentIndex]) {
                segmentVisited[segmentIndex] = 1;
                const Segment& segment = segments[segmentIndex];
                const uint32_t nextNode = segment.a == node ? segment.b : segment.a;
                if (nextNode == startNode) {
                    line.closed = true;
                    break;
                }
                line.points.push_back(crossingPositions[nextNode]);

                node = nextNode;
                uint32_t nextSegment = UINT32_MAX;
                if (incident[node].size() == 2) {
                    for (uint32_t candidate : incident[node]) {
                        if (!segmentVisited[candidate]) {
                            nextSegment = candidate;
                            break;
                        }
                    }
                }
                if (nextSegment == UINT32_MAX) break;
                segmentIndex = nextSegment;
            }
            return line;
        };

        for (uint32_t node = 0; node < incident.size(); ++node) {
            if (incident[node].size() == 2) continue;
            for (uint32_t segment : incident[node]) {
                if (segmentVisited[segment]) continue;
                mesh::Polyline line = trace(node, segment);
                if (line.points.size() >= 2) result.push_back(std::move(line));
            }
        }

        for (uint32_t segment = 0; segment < segments.size(); ++segment) {
            if (segmentVisited[segment]) continue;
            mesh::Polyline line = trace(segments[segment].a, segment);
            if (line.points.size() >= 3 ||
                (line.closed && line.points.size() >= 2)) {
                result.push_back(std::move(line));
            }
        }
    }

    return result;
}
