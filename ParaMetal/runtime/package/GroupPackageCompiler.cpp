#include "GroupPackageCompiler.hpp"

#include "PackageCompileContext.hpp"
#include "domain/GeometryData.hpp"
#include "domain/RemeshData.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/RuntimeProductManager.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/package/RuntimePackages.hpp"

namespace runtimepackage {

bool compileGroupPackage(const PackageCompileContext& context) {
    if (!context.products) {
        context.errors.push_back("Group package compilation requires runtime products.");
        return false;
    }

    GroupPackage package{};
    package.hashes = context.output.hashes;

    const std::size_t inputIndex = inputIndexOf(context.node, "Mesh");
    if (inputIndex >= context.node.inputs.size()) {
        context.errors.push_back("Group has no Mesh input socket.");
        return false;
    }
    const NodeGraphEdge* incoming = context.graph.edges.incomingEdge(
        context.node.id, context.node.inputs[inputIndex].id);
    if (!incoming || !incoming->fromNode.isValid()) {
        context.errors.push_back("Group input is not connected.");
        return false;
    }
    const uint64_t sourceSocketKey =
        NodeSocketKey(incoming->fromNode, incoming->fromSocket).value;

    if (const GroupPackage* sourcePackage =
            context.packages.findAny<GroupPackage>(sourceSocketKey)) {
        package.surfaceProduct = sourcePackage->surfaceProduct;
    } else if (const ModelPackage* sourcePackage =
                   context.packages.findAny<ModelPackage>(sourceSocketKey)) {
        package.surfaceProduct = sourcePackage->productHandle;
    } else if (const RemeshPackage* sourcePackage =
                   context.packages.findAny<RemeshPackage>(sourceSocketKey)) {
        package.surfaceProduct = sourcePackage->productHandle;
    } else {
        context.errors.push_back(
            "Group source has no runtime surface package.");
        return false;
    }

    const bool sourceCurrent =
        package.surfaceProduct.type == ProductType::Model
            ? context.products->isCurrent<ModelProduct>(package.surfaceProduct)
            : package.surfaceProduct.type == ProductType::Remesh
                ? context.products->isCurrent<RemeshProduct>(package.surfaceProduct)
                : false;
    if (!sourceCurrent) {
        context.errors.push_back("Group source publication is not current.");
        return false;
    }

    package.displayDependencyKeys = {package.surfaceProduct.outputSocketKey};
    context.packages.apply<GroupPackage>(
        context.outputSocketKey, std::move(package));
    return true;
}

}
