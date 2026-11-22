#include "RemeshPackageCompiler.hpp"

#include "PackageCompileContext.hpp"
#include "domain/GeometryData.hpp"
#include "domain/RemeshData.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashProductHandle.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "nodegraph/NodeRemeshParams.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/package/RuntimePackages.hpp"
#include "runtime/package/passes/CoordinatePass.hpp"

#include <utility>

namespace runtimepackage {

static void sealRemeshPackage(
    RemeshPackage& package,
    const HashValues& sourceGeometryHashes) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, sourceGeometryHashes.full);
    HashBuilder::combine(geometryHash, package.sourceMeshHandle.key);
    combineProductHandleHash(geometryHash, package.sourceModelProduct, HashDomain::Geometry);
    HashBuilder::combine(geometryHash, static_cast<uint64_t>(package.iterations));
    HashBuilder::combineFloat(geometryHash, package.minAngleDegrees);
    HashBuilder::combineFloat(geometryHash, package.maxEdgeLength);
    HashBuilder::combineFloat(geometryHash, package.stepSize);

    uint64_t displayHash = HashBuilder::start();
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showRemeshOverlay ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showFaceNormals ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showVertexNormals ? 1u : 0u));
    HashBuilder::combineFloat(displayHash, package.display.normalLength);
    HashBuilder::combinePod(displayHash, package.localToWorld);

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, geometryHash);
    HashBuilder::combine(fullHash, displayHash);

    package.hashes.geometry = geometryHash;
    package.hashes.simulation = geometryHash;
    package.hashes.full = fullHash;
    package.hashes.display = displayHash;
    package.hashes.properties = 0;
}

bool compileRemeshPackage(const PackageCompileContext& context) {
    const RemeshData* remesh = context.payloads.get<RemeshData>(context.output.payloadHandle);
    if (!remesh) {
        context.errors.push_back("Remesh package is missing its authored payload.");
        return false;
    }
    if (!remesh->active) {
        return true;
    }
    if (remesh->sourceMeshHandle.key == 0) {
        context.errors.push_back("Remesh package is missing its source mesh.");
        return false;
    }

    const GeometryData* geometry = context.payloads.resolveGeometry(remesh->sourceMeshHandle);
    const ModelPackage* model = context.packages.findAny<ModelPackage>(remesh->sourceMeshHandle.key);
    if (!geometry || geometry->vertexPositions.empty() || geometry->triangleIndices.empty() ||
        !model || !model->productHandle.isValid()) {
        context.errors.push_back("Remesh package is missing its upstream model product.");
        return false;
    }

    RemeshPackage package{};
    const RemeshPackage* previous = context.packages.findStored<RemeshPackage>(context.outputSocketKey);
    package.productHandle = previous ? previous->productHandle : ProductHandle{};
    package.vertexPositions = geometry->vertexPositions;
    package.triangleIndices = geometry->triangleIndices;
    if (!context.payloads.resolveLocalToWorld(remesh->sourceMeshHandle, package.localToWorld)) {
        context.errors.push_back("Remesh package cannot resolve source placement.");
        return false;
    }

    const RemeshNodeParams params = readRemeshNodeParams(context.node);
    package.iterations = remesh->iterations;
    package.minAngleDegrees = remesh->minAngleDegrees;
    package.maxEdgeLength = remesh->maxEdgeLength;
    package.stepSize = remesh->stepSize;
    package.display.showRemeshOverlay = params.preview.showRemeshOverlay;
    package.display.showFaceNormals = params.preview.showFaceNormals;
    package.display.showVertexNormals = params.preview.showVertexNormals;
    package.display.normalLength = static_cast<float>(params.normalLength);
    package.remeshHandle = context.output.payloadHandle;
    package.sourceMeshHandle = remesh->sourceMeshHandle;
    package.sourceModelProduct = model->productHandle;
    package.displayDependencyKeys.push_back(model->productHandle.outputSocketKey);

    CoordinatePass{}.run(package, context.worldUnit);
    HashValues sourceHashes{};
    sourceHashes.full = context.payloads.resolveHash(remesh->sourceMeshHandle, HashDomain::Full);
    sealRemeshPackage(package, sourceHashes);
    context.packages.apply<RemeshPackage>(context.outputSocketKey, std::move(package));
    return true;
}

}
