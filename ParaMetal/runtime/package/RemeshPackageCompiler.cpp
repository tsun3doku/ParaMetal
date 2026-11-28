#include "RemeshPackageCompiler.hpp"

#include "PackageCompileContext.hpp"
#include "domain/GeometryData.hpp"
#include "domain/RemeshData.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashProductHandle.hpp"
#include "nodegraph/NodeGraphUtils.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "nodegraph/NodeRemeshParams.hpp"
#include "nodegraph/NodeGraphRuntime.hpp"
#include "runtime/RuntimeProductManager.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/package/RuntimePackages.hpp"
#include "runtime/package/passes/CoordinatePass.hpp"

#include <utility>

namespace runtimepackage {

static void sealRemeshPackage(
    RemeshPackage& package,
    const HashValues& sourceGeometryHashes) {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combine(geometryHash, sourceGeometryHashes.geometry);
    combineProductHandleHash(geometryHash, package.sourceModelProduct, HashDomain::Geometry);
    HashBuilder::combine(geometryHash, static_cast<uint64_t>(package.iterations));
    HashBuilder::combineFloat(geometryHash, package.minAngleDegrees);
    HashBuilder::combineFloat(geometryHash, package.maxEdgeLength);
    HashBuilder::combineFloat(geometryHash, package.stepSize);

    uint64_t displayHash = HashBuilder::start();
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
    const NodeDataBlock* source = context.input(NodeGraphValueType::Mesh);
    if (!source || source->payloadHandle.key == 0) {
        context.errors.push_back("Remesh package requires an evaluated Mesh input.");
        return false;
    }

    NodeDataHandle sourceGeometryHandle{};
    const GeometryData* geometry = context.payloads.resolveGeometry(
        source->payloadHandle, &sourceGeometryHandle);
    const ModelPackage* model = context.packages.findAny<ModelPackage>(
        source->payloadHandle.key);
    if (!model && sourceGeometryHandle.key != source->payloadHandle.key) {
        model = context.packages.findAny<ModelPackage>(sourceGeometryHandle.key);
    }
    if (!geometry || geometry->vertexPositions.empty() || geometry->triangleIndices.empty() ||
        sourceGeometryHandle.key == 0 ||
        !model || !model->productHandle.isValid()) {
        context.errors.push_back("Remesh package is missing its upstream model product.");
        return false;
    }

    RemeshPackage package{};
    const RemeshPackage* previous = context.packages.findStored<RemeshPackage>(context.outputSocketKey);
    const ProductHandle current = context.products
        ? context.products->getCurrentHandle<RemeshProduct>(context.outputSocketKey)
        : ProductHandle{};
    package.productHandle = current.isValid() && previous
        ? previous->productHandle : ProductHandle{};
    if (!context.payloads.resolveLocalToWorld(source->payloadHandle, package.localToWorld)) {
        context.errors.push_back("Remesh package cannot resolve source placement.");
        return false;
    }

    const RemeshNodeParams params = readRemeshNodeParams(context.node);
    package.iterations = params.iterations;
    package.minAngleDegrees = static_cast<float>(params.minAngleDegrees);
    package.maxEdgeLength = static_cast<float>(params.maxEdgeLength);
    package.stepSize = static_cast<float>(params.stepSize);
    package.display.showFaceNormals = params.preview.showFaceNormals;
    package.display.showVertexNormals = params.preview.showVertexNormals;
    package.display.normalLength = static_cast<float>(params.normalLength);
    package.sourceModelProduct = model->productHandle;

    CoordinatePass{}.run(package, context.worldUnit);
    const HashValues* sourceHashes = context.payloads.findHashes(sourceGeometryHandle);
    if (!sourceHashes || sourceHashes->geometry == 0) {
        context.errors.push_back("Remesh package source geometry has no geometry hash.");
        return false;
    }
    sealRemeshPackage(package, *sourceHashes);

    const bool computeChanged =
        !previous || !previous->productHandle.isValid() ||
        previous->computeHash() != package.computeHash();
    if (computeChanged) {
        package.vertexPositions = geometry->vertexPositions;
        package.triangleIndices = geometry->triangleIndices;
        CoordinatePass::scalePositions(
            package.vertexPositions,
            units::canonicalToWorldScale(context.worldUnit));
    }

    context.packages.apply<RemeshPackage>(context.outputSocketKey, std::move(package));
    return true;
}

static HashValues completedHashes(
    uint64_t surfaceIdentity,
    const RemeshPackage& package,
    const HashValues& provisional) {
    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combine(simulationHash, surfaceIdentity);
    HashBuilder::combinePod(simulationHash, package.localToWorld);

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, simulationHash);
    HashBuilder::combine(fullHash, package.hashes.display);

    HashValues hashes = provisional;
    hashes.full = fullHash;
    hashes.geometry = surfaceIdentity;
    hashes.simulation = simulationHash;
    hashes.display = package.hashes.display;
    return hashes;
}

static bool needsRemeshPayload(
    const PackageCompileContext& context,
    const NodeGraphRuntime& graphRuntime,
    const ProductHandle& productHandle) {
    const RemeshPackage* incoming = context.packages.findAny<RemeshPackage>(context.outputSocketKey);
    const RemeshPackage* committed = context.packages.findStored<RemeshPackage>(context.outputSocketKey);
    if (!incoming || !committed || !productHandle.isValid() ||
        incoming->computeHash() != committed->computeHash()) {
        return true;
    }
    const NodeDataBlock* previous = graphRuntime.cachedOutput(
        context.node.id, NodeSocketKey(context.outputSocketKey).socket());
    const RemeshData* completed = previous
        ? context.payloads.get<RemeshData>(previous->payloadHandle) : nullptr;
    const NodeDataBlock* source = context.input(NodeGraphValueType::Mesh);
    return !completed || !source ||
        completed->sourceMeshHandle != source->payloadHandle ||
        completed->surfaceGeometryHash != surfaceGeometryIdentity(productHandle);
}

bool preflightRemeshPackage(
    const PackageCompileContext& context,
    NodeGraphRuntime& graphRuntime,
    NodePayloadRegistry& payloads,
    bool& backpressured) {
    if (!context.products) return false;
    const ProductHandle handle =
        context.products->getCurrentHandle<RemeshProduct>(context.outputSocketKey);
    if (!needsRemeshPayload(context, graphRuntime, handle) ||
        payloads.canPublish(context.outputSocketKey)) {
        return true;
    }
    context.packages.retainCommitted(context.outputSocketKey);
    backpressured = true;
    return false;
}

bool publishRemeshPackage(
    const PackageCompileContext& context,
    NodeGraphRuntime& graphRuntime,
    NodePayloadRegistry& payloads) {
    if (!context.products) return false;
    const ProductHandle handle =
        context.products->getCurrentHandle<RemeshProduct>(context.outputSocketKey);
    const RemeshProduct* product = context.products->isCurrent<RemeshProduct>(handle)
        ? context.products->resolve<RemeshProduct>(handle) : nullptr;
    const RemeshPackage* package = context.packages.findAny<RemeshPackage>(context.outputSocketKey);
    const NodeDataBlock* source = context.input(NodeGraphValueType::Mesh);
    if (!product || !package || !source || source->payloadHandle.key == 0 ||
        product->surfacePositions.empty() || product->surfaceTriangleIndices.empty()) {
        context.errors.push_back("Remesh did not produce a completed surface mesh.");
        return false;
    }

    const NodeGraphSocketId socketId =
        NodeSocketKey(context.outputSocketKey).socket();
    if (!socketId.isValid()) {
        context.errors.push_back("Remesh node has no Remesh output socket.");
        return false;
    }
    const NodeDataBlock* previous = graphRuntime.cachedOutput(context.node.id, socketId);
    const RemeshData* completed = previous
        ? payloads.get<RemeshData>(previous->payloadHandle) : nullptr;
    const uint64_t surfaceIdentity = surfaceGeometryIdentity(handle);
    const HashValues hashes = completedHashes(
        surfaceIdentity, *package, context.output.hashes);
    if (completed && completed->sourceMeshHandle == source->payloadHandle &&
        completed->surfaceGeometryHash == surfaceIdentity) {
        NodeDataBlock output = *previous;
        output.hashes = hashes;
        return graphRuntime.replaceEvaluatedOutput(context.node.id, socketId, output);
    }

    RemeshData mesh{};
    mesh.sourceMeshHandle = source->payloadHandle;
    mesh.iterations = package->iterations;
    mesh.minAngleDegrees = package->minAngleDegrees;
    mesh.maxEdgeLength = package->maxEdgeLength;
    mesh.stepSize = package->stepSize;
    mesh.active = true;
    mesh.surfacePositions = product->surfacePositions;
    mesh.surfaceTriangleIndices = product->surfaceTriangleIndices;
    mesh.surfaceFaceIds = product->surfaceFaceIds;
    mesh.surfaceVertexIds = product->surfaceVertexIds;
    mesh.surfaceEdgeIds = product->surfaceEdgeIds;
    mesh.surfaceGeometryHash = surfaceIdentity;

    NodeDataBlock output = context.output;
    output.hashes = hashes;
    output.payloadHandle = payloads.store(context.outputSocketKey, std::move(mesh), hashes);
    if (!output.payloadHandle.key) {
        context.errors.push_back("Remesh payload publication blocked after backend commit.");
        return false;
    }
    return graphRuntime.replaceEvaluatedOutput(context.node.id, socketId, output);
}

}
