#include "ModelPackageCompiler.hpp"

#include "PackageCompileContext.hpp"
#include "domain/GeometryData.hpp"
#include "hash/HashBuilder.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/package/RuntimePackages.hpp"
#include "runtime/package/passes/CoordinatePass.hpp"
#include "util/GeometryUtils.hpp"

#include <limits>
#include <utility>

namespace runtimepackage {

static void computeLocalBounds(
    const std::vector<float>& positions,
    glm::vec3& outMin,
    glm::vec3& outMax) {
    outMin = glm::vec3(std::numeric_limits<float>::max());
    outMax = glm::vec3(std::numeric_limits<float>::lowest());
    for (std::size_t i = 0; i + 2 < positions.size(); i += 3) {
        const glm::vec3 position(positions[i], positions[i + 1], positions[i + 2]);
        outMin = glm::min(outMin, position);
        outMax = glm::max(outMax, position);
    }
}

static void computeWorldBounds(ModelPackage& package) {
    const glm::mat4 localToWorld = toMat4(package.localToWorld);
    package.worldBoundsMin = glm::vec3(std::numeric_limits<float>::max());
    package.worldBoundsMax = glm::vec3(std::numeric_limits<float>::lowest());
    for (int corner = 0; corner < 8; ++corner) {
        const glm::vec3 local(
            (corner & 1) != 0 ? package.localBoundsMax.x : package.localBoundsMin.x,
            (corner & 2) != 0 ? package.localBoundsMax.y : package.localBoundsMin.y,
            (corner & 4) != 0 ? package.localBoundsMax.z : package.localBoundsMin.z);
        const glm::vec3 world = glm::vec3(localToWorld * glm::vec4(local, 1.0f));
        package.worldBoundsMin = glm::min(package.worldBoundsMin, world);
        package.worldBoundsMax = glm::max(package.worldBoundsMax, world);
    }
}

static void sealModelPackage(
    ModelPackage& package,
    const HashValues& sourceGeometryHashes,
    const HashValues& outputHashes) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, sourceGeometryHashes.geometry);
    HashBuilder::combineFloat(geometryHash, package.canonicalToWorldScale);

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, geometryHash);
    HashBuilder::combine(fullHash, outputHashes.full);
    HashBuilder::combinePod(fullHash, package.localToWorld);

    package.hashes.geometry = geometryHash;
    package.hashes.simulation = fullHash;
    package.hashes.full = fullHash;
    package.hashes.display = fullHash;
    package.hashes.properties = 0;
}

bool compileModelPackage(const PackageCompileContext& context) {
    NodeDataHandle sourceGeometryHandle{};
    const GeometryData* geometry = context.payloads.resolveGeometry(
        context.output.payloadHandle, &sourceGeometryHandle);
    if (!geometry || sourceGeometryHandle.key == 0) {
        context.errors.push_back("Model package is missing its geometry payload.");
        return false;
    }
    if (geometry->vertexPositions.empty() || geometry->triangleIndices.empty()) {
        context.errors.push_back("Model package geometry has no triangles.");
        return false;
    }
    const HashValues* sourceHashes = context.payloads.findHashes(sourceGeometryHandle);
    if (!sourceHashes || sourceHashes->geometry == 0) {
        context.errors.push_back("Model package geometry has no source hash.");
        return false;
    }

    ModelPackage package{};
    const ModelPackage* previous = context.packages.findStored<ModelPackage>(context.outputSocketKey);
    package.productHandle = previous ? previous->productHandle : ProductHandle{};
    if (!context.payloads.resolveLocalToWorld(
            context.output.payloadHandle, package.localToWorld)) {
        context.errors.push_back("Model package cannot resolve its placement.");
        return false;
    }

    CoordinatePass{}.run(package, context.worldUnit);
    sealModelPackage(package, *sourceHashes, context.output.hashes);

    const bool computeChanged =
        !previous || !previous->productHandle.isValid() ||
        previous->computeHash() != package.computeHash();
    if (computeChanged) {
        package.vertexPositions = geometry->vertexPositions;
        package.triangleIndices = geometry->triangleIndices;
        package.renderPositions = geometry->renderPositions;
        package.renderNormals = geometry->renderNormals;
        package.renderTexcoords = geometry->renderTexcoords;
        package.renderIndices = geometry->renderIndices;
        CoordinatePass::scalePositions(
            package.vertexPositions, package.canonicalToWorldScale);
        CoordinatePass::scalePositions(
            package.renderPositions, package.canonicalToWorldScale);
        computeLocalBounds(
            package.vertexPositions,
            package.localBoundsMin,
            package.localBoundsMax);
    } else {
        package.localBoundsMin = previous->localBoundsMin;
        package.localBoundsMax = previous->localBoundsMax;
    }
    computeWorldBounds(package);

    context.packages.apply<ModelPackage>(context.outputSocketKey, std::move(package));
    return true;
}

}
