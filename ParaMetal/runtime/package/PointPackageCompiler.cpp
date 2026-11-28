#include "PointPackageCompiler.hpp"

#include "PackageCompileContext.hpp"
#include "domain/GeometryData.hpp"
#include "domain/PointData.hpp"
#include "domain/RemeshData.hpp"
#include "hash/HashBuilder.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/RuntimeProductManager.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/package/RuntimePackages.hpp"
#include "runtime/package/passes/CoordinatePass.hpp"
#include "util/GeometryUtils.hpp"
#include "util/Units.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace runtimepackage {

enum class PointSourceType {
    Points,
    Geometry,
    Remesh,
    Invalid,
};


static bool isFinite(const glm::vec4& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) &&
        std::isfinite(value.z) && std::isfinite(value.w);
}

static bool isFinite(const glm::mat4& value) {
    for (int column = 0; column < 4; ++column) {
        if (!isFinite(value[column])) {
            return false;
        }
    }
    return true;
}

static bool isFinite(const std::array<float, 16>& values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

static bool isFiniteDomain(const glm::vec3& minimum, const glm::vec3& maximum) {
    return GeometryUtils::isFinite(minimum) && GeometryUtils::isFinite(maximum) &&
        minimum.x <= maximum.x &&
        minimum.y <= maximum.y &&
        minimum.z <= maximum.z;
}

static bool isMergeTransformUsable(const std::array<float, 16>& values) {
    if (!isFinite(values)) {
        return false;
    }

    const glm::dmat3 linear(
        values[0], values[1], values[2],
        values[4], values[5], values[6],
        values[8], values[9], values[10]);
    const glm::dvec3 column0(values[0], values[1], values[2]);
    const glm::dvec3 column1(values[4], values[5], values[6]);
    const glm::dvec3 column2(values[8], values[9], values[10]);

    const double maxLength = std::max({
        glm::length(column0),
        glm::length(column1),
        glm::length(column2)});
    const double minLength = std::min({
        glm::length(column0),
        glm::length(column1),
        glm::length(column2)});
    if (!(maxLength > 0.0) || !(minLength > 0.0) ||
        maxLength / minLength > 1.0e12) {
        return false;
    }

    const double determinant = glm::determinant(linear);
    if (!std::isfinite(determinant)) {
        return false;
    }

    const double scale = maxLength * maxLength * maxLength;
    if (!(scale > 0.0) || std::abs(determinant) / scale <= 1.0e-12) {
        return false;
    }

    const glm::dmat3 inverse = glm::inverse(linear);
    for (int column = 0; column < 3; ++column) {
        for (int row = 0; row < 3; ++row) {
            if (!std::isfinite(inverse[column][row])) {
                return false;
            }
        }
    }
    return true;
}


static void computeBounds(
    const std::vector<glm::vec4>& positions,
    glm::vec3& outMinimum,
    glm::vec3& outMaximum) {
    outMinimum = glm::vec3(std::numeric_limits<float>::max());
    outMaximum = glm::vec3(std::numeric_limits<float>::lowest());
    for (const glm::vec4& position : positions) {
        const glm::vec3 point(position);
        outMinimum = glm::min(outMinimum, point);
        outMaximum = glm::max(outMaximum, point);
    }
}

static bool resolvePlacement(
    const PackageCompileContext& context,
    std::array<float, 16>& outPlacement,
    std::string& error) {
    if (!context.payloads.resolveLocalToWorld(context.output.payloadHandle, outPlacement)) {
        error = "Point package has no resolvable placement.";
        return false;
    }
    if (!isFinite(outPlacement)) {
        error = "Point package placement is not finite.";
        return false;
    }
    return true;
}

static PointSourceType classifySource(
    const PackageCompileContext& context,
    const NodeDataHandle& source) {
    if (context.payloads.resolvePoints(source) != nullptr) {
        return PointSourceType::Points;
    }

    NodeDataHandle remeshHandle{};
    if (context.payloads.resolveRemesh(source, &remeshHandle) &&
        context.payloads.get<RemeshData>(remeshHandle) != nullptr) {
        return PointSourceType::Remesh;
    }

    if (context.payloads.resolveGeometry(source) != nullptr) {
        return PointSourceType::Geometry;
    }

    return PointSourceType::Invalid;
}

static void sealPointPackage(PointPackage& package, const HashValues& sourceHashes) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, sourceHashes.geometry);
    HashBuilder::combinePodVector(geometryHash, package.positions);
    HashBuilder::combine(geometryHash, static_cast<uint64_t>(package.pointCount));
    HashBuilder::combinePod(geometryHash, package.domainMinimum);
    HashBuilder::combinePod(geometryHash, package.domainMaximum);
    for (float value : package.localToWorld) {
        HashBuilder::combineFloat(geometryHash, value);
    }

    package.hashes.geometry = geometryHash;
    package.hashes.simulation = geometryHash;
    package.hashes.full = geometryHash;
    package.hashes.display = geometryHash;
    package.hashes.properties = 0;
}

static bool validatePointPackage(const PointPackage& package, std::string& error) {
    if (package.positions.empty()) {
        error = "Point package contains no positions.";
        return false;
    }
    if (package.positions.size() >
        static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
        error = "Point package point count exceeds 32-bit capacity.";
        return false;
    }
    for (const glm::vec4& position : package.positions) {
        if (!isFinite(position)) {
            error = "Point package contains a non-finite position.";
            return false;
        }
        if (std::abs(position.w - 1.0f) > 1e-6f) {
            error = "Point package position is not homogeneous.";
            return false;
        }
    }
    if (!isFinite(package.localToWorld)) {
        error = "Point package placement is not finite.";
        return false;
    }
    if (!isFiniteDomain(package.domainMinimum, package.domainMaximum)) {
        error = "Point package domain bounds are invalid.";
        return false;
    }
    return true;
}

static bool resolveAuthored(
    const PackageCompileContext& context,
    const PointData& points,
    PointPackage& package,
    HashValues& sourceHashes,
    std::string& error) {
    for (const glm::vec4& position : points.positions) {
        if (!isFinite(position)) {
            error = "Point payload contains a non-finite position.";
            return false;
        }
    }

    package.positions = points.positions;
    package.domainMinimum = points.domainMinimum;
    package.domainMaximum = points.domainMaximum;
    if (!isFiniteDomain(package.domainMinimum, package.domainMaximum)) {
        error = "Point payload has invalid domain bounds.";
        return false;
    }
    if (!resolvePlacement(context, package.localToWorld, error)) {
        return false;
    }
    CoordinatePass{}.run(package, context.worldUnit);
    sourceHashes = context.output.hashes;
    return true;
}

static bool resolveGeometrySource(
    const PackageCompileContext& context,
    const PointData& points,
    PointPackage& package,
    HashValues& sourceHashes,
    std::string& error) {
    if (points.sources.size() != 1) {
        error = "Mesh Points requires exactly one source.";
        return false;
    }

    const NodeDataHandle source = points.sources.front();
    NodeDataHandle geometryHandle{};
    const GeometryData* geometry = context.payloads.resolveGeometry(source, &geometryHandle);
    if (!geometry) {
        error = "Mesh Points geometry source cannot be resolved.";
        return false;
    }
    if (geometry->vertexPositions.empty() ||
        geometry->vertexPositions.size() % 3 != 0) {
        error = "Mesh Points geometry source has no complete vertex positions.";
        return false;
    }

    const std::size_t vertexCount = geometry->vertexPositions.size() / 3;
    package.positions.reserve(vertexCount);
    for (std::size_t index = 0; index < vertexCount; ++index) {
        const glm::vec4 position(
            geometry->vertexPositions[3 * index + 0],
            geometry->vertexPositions[3 * index + 1],
            geometry->vertexPositions[3 * index + 2],
            1.0f);
        if (!isFinite(position)) {
            error = "Mesh Points geometry source contains a non-finite vertex.";
            return false;
        }
        package.positions.push_back(position);
    }
    computeBounds(package.positions, package.domainMinimum, package.domainMaximum);

    if (!resolvePlacement(context, package.localToWorld, error)) {
        return false;
    }
    CoordinatePass{}.run(package, context.worldUnit);

    uint64_t sourceHash = context.output.hashes.geometry;
    HashBuilder::combine(sourceHash, source.key);
    HashBuilder::combine(sourceHash, context.payloads.resolveHash(geometryHandle, HashDomain::Geometry));
    sourceHashes.geometry = sourceHash;
    return true;
}

static bool resolveRemeshSource(
    const PackageCompileContext& context,
    const PointData& points,
    PointPackage& package,
    HashValues& sourceHashes,
    std::string& error) {
    if (points.sources.size() != 1) {
        error = "Mesh Points requires exactly one source.";
        return false;
    }

    const NodeDataHandle source = points.sources.front();
    NodeDataHandle remeshHandle{};
    const RemeshData* remesh = context.payloads.resolveRemesh(source, &remeshHandle);
    if (!remesh || remesh->surfacePositions.empty() ||
        remesh->surfaceTriangleIndices.empty()) {
        error = "Mesh Points requires completed Remesh geometry.";
        return false;
    }

    package.positions.reserve(remesh->surfacePositions.size());
    for (const glm::vec3& position : remesh->surfacePositions) {
        if (!GeometryUtils::isFinite(position)) {
            error = "Mesh Points remesh output contains a non-finite surface position.";
            return false;
        }
        package.positions.emplace_back(position, 1.0f);
    }
    computeBounds(package.positions, package.domainMinimum, package.domainMaximum);

    if (!resolvePlacement(context, package.localToWorld, error)) {
        return false;
    }
    CoordinatePass::scaleTranslation(package.localToWorld, units::canonicalToWorldScale(context.worldUnit));

    uint64_t sourceHash = context.output.hashes.geometry;
    HashBuilder::combine(sourceHash, source.key);
    HashBuilder::combine(sourceHash, context.payloads.resolveHash(remeshHandle, HashDomain::Geometry));
    sourceHashes.geometry = sourceHash;
    return true;
}

static bool mergePointPackages(
    const std::vector<const PointPackage*>& sourcePackages,
    const std::vector<std::array<float, 16>>& sourcePlacements,
    float mergeDistance,
    PointPackage& package,
    std::string& error) {
    if (sourcePackages.empty() || sourcePackages.size() != sourcePlacements.size()) {
        error = "Point Merge has no inputs.";
        return false;
    }
    if (!std::isfinite(mergeDistance) || mergeDistance < 0.0f) {
        error = "Point Merge has an invalid distance.";
        return false;
    }
    const std::array<float, 16>& referenceLocalToWorld = sourcePlacements.front();
    if (!isMergeTransformUsable(referenceLocalToWorld)) {
        error = "Point Merge reference placement is not invertible.";
        return false;
    }

    const glm::mat4 referenceInverse = glm::inverse(toMat4(referenceLocalToWorld));
    if (!isFinite(referenceInverse)) {
        error = "Point Merge reference placement is not invertible.";
        return false;
    }

    const bool filterPoints = mergeDistance > 0.0f;
    const double inverseCellSize =
        filterPoints ? 1.0 / static_cast<double>(mergeDistance) : 0.0;
    if (filterPoints && !std::isfinite(inverseCellSize)) {
        error = "Point Merge cell size is not finite.";
        return false;
    }
    const double mergeDistanceSquared =
        static_cast<double>(mergeDistance) * static_cast<double>(mergeDistance);
    constexpr double cellLimit = 1.0e15;

    using CellKey = std::array<int64_t, 3>;
    struct CellKeyHash {
        std::size_t operator()(const CellKey& key) const noexcept {
            std::size_t hash = std::hash<int64_t>{}(key[0]);
            const auto combine = [&hash](int64_t value) {
                hash ^= std::hash<int64_t>{}(value) +
                    0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
            };
            combine(key[1]);
            combine(key[2]);
            return hash;
        }
    };

    std::unordered_map<CellKey, std::vector<uint32_t>, CellKeyHash> grid;
    std::vector<glm::vec3> acceptedWorldPositions;
    const std::size_t maxPointCount =
        static_cast<std::size_t>(std::numeric_limits<uint32_t>::max());

    package.positions.clear();
    package.domainMinimum = glm::vec3(std::numeric_limits<float>::max());
    package.domainMaximum = glm::vec3(std::numeric_limits<float>::lowest());

    for (std::size_t sourceIndex = 0; sourceIndex < sourcePackages.size(); ++sourceIndex) {
        const PointPackage* sourcePackage = sourcePackages[sourceIndex];
        const std::array<float, 16>& sourcePlacement = sourcePlacements[sourceIndex];
        if (!sourcePackage || sourcePackage->positions.empty()) {
            error = "Point Merge input package is unavailable or empty.";
            return false;
        }
        if (!isMergeTransformUsable(sourcePlacement)) {
            error = "Point Merge input placement is not invertible.";
            return false;
        }
        if (!isFiniteDomain(sourcePackage->domainMinimum, sourcePackage->domainMaximum)) {
            error = "Point Merge input domain is invalid.";
            return false;
        }

        const glm::mat4 inputToWorld = toMat4(sourcePlacement);
        const glm::mat4 inputToReference = referenceInverse * inputToWorld;

        for (uint32_t corner = 0; corner < 8; ++corner) {
            const glm::vec4 point(
                (corner & 1u) ? sourcePackage->domainMaximum.x : sourcePackage->domainMinimum.x,
                (corner & 2u) ? sourcePackage->domainMaximum.y : sourcePackage->domainMinimum.y,
                (corner & 4u) ? sourcePackage->domainMaximum.z : sourcePackage->domainMinimum.z,
                1.0f);
            const glm::vec4 transformed = inputToReference * point;
            if (!isFinite(transformed)) {
                error = "Point Merge domain transform produced a non-finite bound.";
                return false;
            }
            const glm::vec3 transformedPoint(transformed);
            package.domainMinimum = glm::min(package.domainMinimum, transformedPoint);
            package.domainMaximum = glm::max(package.domainMaximum, transformedPoint);
        }

        for (const glm::vec4& position : sourcePackage->positions) {
            if (!isFinite(position)) {
                error = "Point Merge input contains a non-finite position.";
                return false;
            }

            if (!filterPoints) {
                if (package.positions.size() >= maxPointCount) {
                    error = "Point Merge exceeds 32-bit point capacity.";
                    return false;
                }
                const glm::vec4 transformed = inputToReference * position;
                if (!isFinite(transformed)) {
                    error = "Point Merge produced a non-finite position.";
                    return false;
                }
                package.positions.push_back(transformed);
                continue;
            }

            const glm::vec4 world = inputToWorld * position;
            if (!isFinite(world)) {
                error = "Point Merge encountered a non-finite world position.";
                return false;
            }

            const double gridX = std::floor(static_cast<double>(world.x) * inverseCellSize);
            const double gridY = std::floor(static_cast<double>(world.y) * inverseCellSize);
            const double gridZ = std::floor(static_cast<double>(world.z) * inverseCellSize);
            if (!std::isfinite(gridX) || !std::isfinite(gridY) || !std::isfinite(gridZ) ||
                gridX < -cellLimit || gridX > cellLimit ||
                gridY < -cellLimit || gridY > cellLimit ||
                gridZ < -cellLimit || gridZ > cellLimit) {
                error = "Point Merge grid index is out of range.";
                return false;
            }

            const CellKey cell{
                static_cast<int64_t>(gridX),
                static_cast<int64_t>(gridY),
                static_cast<int64_t>(gridZ)};
            const glm::vec3 worldPosition(world);

            bool tooClose = false;
            for (int64_t dz = -1; dz <= 1 && !tooClose; ++dz) {
                for (int64_t dy = -1; dy <= 1 && !tooClose; ++dy) {
                    for (int64_t dx = -1; dx <= 1 && !tooClose; ++dx) {
                        const CellKey neighbor{
                            cell[0] + dx,
                            cell[1] + dy,
                            cell[2] + dz};
                        const auto it = grid.find(neighbor);
                        if (it == grid.end()) {
                            continue;
                        }
                        for (uint32_t acceptedIndex : it->second) {
                            const glm::vec3& existing = acceptedWorldPositions[acceptedIndex];
                            const double ddx = static_cast<double>(worldPosition.x) - existing.x;
                            const double ddy = static_cast<double>(worldPosition.y) - existing.y;
                            const double ddz = static_cast<double>(worldPosition.z) - existing.z;
                            if (ddx * ddx + ddy * ddy + ddz * ddz < mergeDistanceSquared) {
                                tooClose = true;
                                break;
                            }
                        }
                    }
                }
            }

            if (tooClose) {
                continue;
            }

            if (package.positions.size() >= maxPointCount) {
                error = "Point Merge exceeds 32-bit point capacity.";
                return false;
            }
            const glm::vec4 transformed = referenceInverse * world;
            if (!isFinite(transformed)) {
                error = "Point Merge produced a non-finite position.";
                return false;
            }
            const uint32_t newIndex = static_cast<uint32_t>(package.positions.size());
            package.positions.push_back(transformed);
            acceptedWorldPositions.push_back(worldPosition);
            grid[cell].push_back(newIndex);
        }
    }

    if (package.positions.empty()) {
        error = "Point Merge produced no points.";
        return false;
    }
    return true;
}

static bool resolveMergedSources(
    const PackageCompileContext& context,
    const RuntimeProductManager& products,
    const PointData& points,
    PointPackage& package,
    HashValues& sourceHashes,
    std::string& error) {
    if (points.sources.empty()) {
        error = "Point Merge has no source references.";
        return false;
    }
    if (!std::isfinite(points.mergeDistance) || points.mergeDistance < 0.0f) {
        error = "Point Merge has an invalid merge distance.";
        return false;
    }

    std::vector<const PointPackage*> sourcePackages;
    std::vector<std::array<float, 16>> sourcePlacements;
    sourcePackages.reserve(points.sources.size());
    sourcePlacements.reserve(points.sources.size());
    uint64_t sourceHash = context.output.hashes.geometry;

    for (const NodeDataHandle& source : points.sources) {
        NodeDataHandle resolvedPointHandle{};
        const PointData* sourcePoints = context.payloads.resolvePoints(
            source, &resolvedPointHandle);
        if (!sourcePoints || resolvedPointHandle.key == 0) {
            error = "Point Merge requires only Points inputs.";
            return false;
        }

        const PointPackage* sourcePackage =
            context.packages.findAny<PointPackage>(resolvedPointHandle.key);
        if (!sourcePackage || sourcePackage->positions.empty()) {
            error = "Point Merge input package is unavailable or empty.";
            return false;
        }
        if (!products.isCurrent<PointProduct>(sourcePackage->productHandle)) {
            error = "Point Merge input product is not current.";
            return false;
        }
        if (!products.resolve<PointProduct>(sourcePackage->productHandle)) {
            error = "Point Merge input product is unavailable.";
            return false;
        }

        std::array<float, 16> placement{};
        if (!context.payloads.resolveLocalToWorld(source, placement)) {
            error = "Point Merge input placement is unavailable.";
            return false;
        }
        CoordinatePass::scaleTranslation(
            placement,
            units::canonicalToWorldScale(context.worldUnit));

        sourcePackages.push_back(sourcePackage);
        sourcePlacements.push_back(placement);
        HashBuilder::combine(sourceHash, source.key);
        HashBuilder::combine(sourceHash, sourcePackage->hashes.geometry);
    }

    const float mergeDistance =
        points.mergeDistance * units::canonicalToWorldScale(context.worldUnit);
    if (!std::isfinite(mergeDistance)) {
        error = "Point Merge distance is not finite after unit conversion.";
        return false;
    }

    if (!mergePointPackages(
            sourcePackages,
            sourcePlacements,
            mergeDistance,
            package,
            error)) {
        return false;
    }

    if (!resolvePlacement(context, package.localToWorld, error)) {
        return false;
    }
    CoordinatePass::scaleTranslation(package.localToWorld, units::canonicalToWorldScale(context.worldUnit));

    HashBuilder::combineFloat(sourceHash, points.mergeDistance);
    sourceHashes.geometry = sourceHash;
    return true;
}

bool compilePointPackage(
    const PackageCompileContext& context,
    const RuntimeProductManager& products) {
    PointPackage package{};
    const PointPackage* previous =
        context.packages.findStored<PointPackage>(context.outputSocketKey);
    package.productHandle = previous ? previous->productHandle : ProductHandle{};

    HashValues sourceHashes{};
    std::string error;
    bool resolved = false;

    const PointData* points =
        context.payloads.resolvePoints(context.output.payloadHandle);
    if (!points) {
        error = "Point package is missing its authored payload.";
    } else if (!points->active) {
        return true;
    } else if (!points->positions.empty() && !points->sources.empty()) {
        error = "Point payload mixes authored positions with sources.";
    } else if (!points->positions.empty()) {
        resolved = resolveAuthored(context, *points, package, sourceHashes, error);
    } else if (!points->sources.empty()) {
        const PointSourceType firstType =
            classifySource(context, points->sources.front());
        if (firstType == PointSourceType::Invalid) {
            error = "Point source cannot be resolved.";
        } else if (firstType == PointSourceType::Geometry) {
            resolved = resolveGeometrySource(
                context, *points, package, sourceHashes, error);
        } else if (firstType == PointSourceType::Remesh) {
            resolved = resolveRemeshSource(
                context, *points, package, sourceHashes, error);
        } else {
            resolved = resolveMergedSources(
                context, products, *points, package, sourceHashes, error);
        }
    } else {
        error = "Point payload contains no positions or sources.";
    }

    if (!resolved) {
        if (error.empty()) {
            context.errors.push_back("Point package could not be resolved.");
        } else {
            context.errors.push_back(std::move(error));
        }
        return false;
    }
    if (!validatePointPackage(package, error)) {
        context.errors.push_back(std::move(error));
        return false;
    }

    package.pointCount = static_cast<uint32_t>(package.positions.size());
    sealPointPackage(package, sourceHashes);
    context.packages.apply<PointPackage>(context.outputSocketKey, std::move(package));
    return true;
}

} // namespace runtimepackage
