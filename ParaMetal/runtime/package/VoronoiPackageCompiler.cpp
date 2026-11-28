#include "VoronoiPackageCompiler.hpp"

#include "PackageCompileContext.hpp"
#include "domain/RemeshData.hpp"
#include "domain/VoronoiData.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashProductHandle.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "nodegraph/NodeVoronoiParams.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/package/RuntimePackages.hpp"
#include "runtime/package/passes/CoordinatePass.hpp"
#include "util/GeometryUtils.hpp"

#include <cmath>
#include <limits>
#include <unordered_set>
#include <utility>

namespace runtimepackage {

static void sealVoronoiPackage(VoronoiPackage& package, const HashValues& authoredHashes) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, authoredHashes.full);
    HashBuilder::combine(geometryHash, static_cast<uint64_t>(package.voxelResolution));
    HashBuilder::combineFloat(geometryHash, package.volumePadding);
    HashBuilder::combineFloat(geometryHash, package.contactRange);
    HashBuilder::combine(geometryHash, static_cast<uint64_t>(package.lloydIterations));
    HashBuilder::combineFloat(geometryHash, package.lloydVolumeThreshold);
    HashBuilder::combineFloat(geometryHash, package.lloydDamping);
    combineProductHandleHash(geometryHash, package.pointsProduct, HashDomain::Geometry);
    for (const glm::vec3& corner : package.pointDomainCorners) {
        HashBuilder::combinePod(geometryHash, corner);
    }
    HashBuilder::combine(geometryHash, static_cast<uint64_t>(package.globalRemeshProducts.size()));
    for (const ProductHandle& handle : package.globalRemeshProducts) {
        combineProductHandleHash(geometryHash, handle, HashDomain::Geometry);
    }
    for (const ProductHandle& handle : package.globalModelProducts) {
        combineProductHandleHash(geometryHash, handle, HashDomain::Geometry);
    }
    for (const std::array<float, 16>& transform : package.globalRemeshLocalToWorld) {
        HashBuilder::combinePod(geometryHash, transform);
    }
    HashBuilder::combinePodVector(geometryHash, package.pointPositions);
    HashBuilder::combineFloat(geometryHash, package.canonicalToWorldScale);

    uint64_t displayHash = HashBuilder::start();
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showVoronoi ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showPoints ? 1u : 0u));

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, geometryHash);
    HashBuilder::combine(fullHash, displayHash);

    package.hashes.geometry = geometryHash;
    package.hashes.simulation = geometryHash;
    package.hashes.full = fullHash;
    package.hashes.display = displayHash;
    package.hashes.properties = 0;
}

bool compileVoronoiPackage(const PackageCompileContext& context) {
    const VoronoiData* voronoi = context.payloads.get<VoronoiData>(context.output.payloadHandle);
    if (!voronoi) {
        context.errors.push_back("Voronoi package is missing its authored payload.");
        return false;
    }
    if (!voronoi->active) {
        return true;
    }

    const PointPackage* points = context.packages.findAny<PointPackage>(
        voronoi->pointsPayloadHandle.key);
    if (!points || points->positions.empty()) {
        context.errors.push_back("Voronoi package is missing its upstream point package.");
        return false;
    }
    if (!points->productHandle.isValid()) {
        context.errors.push_back("Voronoi package is missing its upstream point product.");
        return false;
    }

    VoronoiPackage package{};
    const VoronoiPackage* previous = context.packages.findStored<VoronoiPackage>(context.outputSocketKey);
    package.productHandle = previous ? previous->productHandle : ProductHandle{};
    package.active = voronoi->active;
    package.voxelResolution = voronoi->voxelResolution;
    package.volumePadding = voronoi->volumePadding;
    package.lloydIterations = voronoi->lloydIterations;
    package.lloydVolumeThreshold = voronoi->lloydVolumeThreshold;
    package.lloydDamping = voronoi->lloydDamping;
    package.contactRange = voronoi->contactRange;
    package.pointsProduct = points->productHandle;
    const VoronoiNodeParams params = readVoronoiNodeParams(context.node);
    package.display.showVoronoi = params.preview.showVoronoi;
    package.display.showPoints = params.preview.showPoints;

    if (voronoi->modelMeshHandles.empty()) {
        context.errors.push_back("Voronoi domain requires at least one unique Remeshes input.");
        return false;
    }

    std::unordered_set<uint64_t> seenRemeshKeys;
    std::vector<NodeDataHandle> remeshHandles;
    remeshHandles.reserve(voronoi->modelMeshHandles.size());
    for (const NodeDataHandle& remeshHandle : voronoi->modelMeshHandles) {
        NodeDataHandle resolvedRemeshHandle{};
        const RemeshData* remeshData = context.payloads.resolveRemesh(
            remeshHandle, &resolvedRemeshHandle);
        const uint64_t remeshKey = resolvedRemeshHandle.key;
        if (!remeshData || remeshKey == 0 ||
            !seenRemeshKeys.insert(remeshKey).second) {
            context.errors.push_back("Voronoi domain contains an empty or duplicate remesh payload.");
            return false;
        }
        const RemeshPackage* remesh = context.packages.findAny<RemeshPackage>(remeshKey);
        const ModelPackage* model = remesh
            ? context.packages.findAny<ModelPackage>(remesh->sourceModelProduct.outputSocketKey)
            : nullptr;
        if (!remesh || !model ||
            !remesh->productHandle.isValid() || !model->productHandle.isValid()) {
            context.errors.push_back("Voronoi domain is missing an upstream remesh product.");
            return false;
        }
        remeshHandles.push_back(resolvedRemeshHandle);
        package.globalRemeshProducts.push_back(remesh->productHandle);
        package.globalModelProducts.push_back(model->productHandle);
        package.displayDependencyKeys.push_back(remesh->productHandle.outputSocketKey);
    }

    CoordinatePass{}.run(package, context.worldUnit);
    const glm::mat4 pointTransform = toMat4(points->localToWorld);
    package.pointPositions = points->positions;
    for (glm::vec4& position : package.pointPositions) {
        position = pointTransform * position;
    }

    glm::vec3 worldDomainMin(std::numeric_limits<float>::max());
    glm::vec3 worldDomainMax(-std::numeric_limits<float>::max());
    for (uint32_t corner = 0; corner < 8; ++corner) {
        const glm::vec3 localCorner(
            (corner & 1u) ? points->domainMaximum.x : points->domainMinimum.x,
            (corner & 2u) ? points->domainMaximum.y : points->domainMinimum.y,
            (corner & 4u) ? points->domainMaximum.z : points->domainMinimum.z);
        const glm::vec3 worldCorner = glm::vec3(pointTransform * glm::vec4(localCorner, 1.0f));
        package.pointDomainCorners[corner] = worldCorner;
        worldDomainMin = glm::min(worldDomainMin, worldCorner);
        worldDomainMax = glm::max(worldDomainMax, worldCorner);
    }

    if (package.voxelResolution < 2) {
        context.errors.push_back("Voronoi domain requires voxelResolution >= 2.");
        return false;
    }
    const glm::vec3 physicalExtent = worldDomainMax - worldDomainMin;
    const float longestExtent = std::max(std::max(physicalExtent.x, physicalExtent.y), physicalExtent.z);
    const float spacing = longestExtent /
        static_cast<float>(package.voxelResolution - 1);
    const float domainVolume = physicalExtent.x * physicalExtent.y * physicalExtent.z;
    if (package.pointPositions.empty() || !(domainVolume > 0.0f)) {
        context.errors.push_back("Voronoi domain has no point set to size the voxel resolution against.");
        return false;
    }
    const float pointSpacing = std::cbrt(
        domainVolume / static_cast<float>(package.pointPositions.size()));
    if (spacing > 0.5f * pointSpacing) {
        context.errors.push_back(
            "Voronoi domain violates the resolution rule: spacing (" +
            std::to_string(spacing) +
            ") must be <= 0.5 * point spacing (" +
            std::to_string(0.5f * pointSpacing) + ").");
        return false;
    }

    const glm::mat4 worldToPoint = glm::inverse(pointTransform);
    constexpr float epsilon = 1e-4f;
    const glm::vec3 localDomainMin = points->domainMinimum - glm::vec3(epsilon);
    const glm::vec3 localDomainMax = points->domainMaximum + glm::vec3(epsilon);

    for (const NodeDataHandle& remeshHandle : remeshHandles) {
        const RemeshPackage* remesh =
            context.packages.findAny<RemeshPackage>(remeshHandle.key);
        const RemeshData* completed =
            context.payloads.get<RemeshData>(remeshHandle);
        if (!remesh || !completed || completed->surfacePositions.empty()) {
            context.errors.push_back("Voronoi domain cannot resolve completed Remesh surface.");
            return false;
        }
        package.globalRemeshLocalToWorld.push_back(remesh->localToWorld);

        const glm::mat4 remeshToPointLocal =
            worldToPoint * toMat4(remesh->localToWorld);
        for (const glm::vec3& position : completed->surfacePositions) {
            const glm::vec3 pointLocalPos = glm::vec3(
                remeshToPointLocal * glm::vec4(position, 1.0f));
            if (glm::any(glm::lessThan(pointLocalPos, localDomainMin)) ||
                glm::any(glm::greaterThan(pointLocalPos, localDomainMax))) {
                context.errors.push_back(
                    "Voronoi domain remesh " + std::to_string(remeshHandle.key) +
                    " lies outside the PointData domain bounds.");
                return false;
            }
        }
    }

    sealVoronoiPackage(package, context.output.hashes);
    context.packages.apply<VoronoiPackage>(context.outputSocketKey, std::move(package));
    return true;
}

}
