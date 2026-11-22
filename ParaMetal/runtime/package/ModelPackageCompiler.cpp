#include "ModelPackageCompiler.hpp"

#include "PackageCompileContext.hpp"
#include "domain/GeometryData.hpp"
#include "hash/HashBuilder.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/package/RuntimePackages.hpp"
#include "runtime/package/passes/CoordinatePass.hpp"

#include <utility>

namespace runtimepackage {

static void sealModelPackage(
    ModelPackage& package,
    const HashValues& sourceGeometryHashes,
    const HashValues& outputHashes) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, sourceGeometryHashes.geometry);
    HashBuilder::combine(geometryHash, package.geometryHandle.key);
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
    package.geometryHandle = sourceGeometryHandle;
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
    }

    context.packages.apply<ModelPackage>(context.outputSocketKey, std::move(package));
    return true;
}

}
