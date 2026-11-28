#include "RuntimePackageCompiler.hpp"

#include "CFDPackageCompiler.hpp"
#include "GroupPackageCompiler.hpp"
#include "HeatPackageCompiler.hpp"
#include "ModelPackageCompiler.hpp"
#include "PackageCompileContext.hpp"
#include "PointPackageCompiler.hpp"
#include "RemeshPackageCompiler.hpp"
#include "VoronoiPackageCompiler.hpp"
#include "nodegraph/NodeGraphRuntime.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/package/RuntimePackageController.hpp"
#include "runtime/RuntimeProductManager.hpp"

using Compile = bool (*)(const runtimepackage::PackageCompileContext&);
using Preflight = bool (*)(
    const runtimepackage::PackageCompileContext&, NodeGraphRuntime&,
    NodePayloadRegistry&, bool&);
using Publish = bool (*)(
    const runtimepackage::PackageCompileContext&, NodeGraphRuntime&,
    NodePayloadRegistry&);

struct PackageOperation {
    uint8_t payloadType;
    Compile compile;
    Preflight preflight = nullptr;
    Publish publish = nullptr;
};

static bool compilePoints(const runtimepackage::PackageCompileContext& context) {
    if (!context.products) {
        context.errors.push_back("Point package compilation requires runtime products.");
        return false;
    }
    return runtimepackage::compilePointPackage(context, *context.products);
}

static const PackageOperation* operationFor(uint8_t payloadType) {
    // The only per-type dispatch is here; graph execution itself is type-agnostic.
    static const PackageOperation operations[] = {
        {payloadtypes::Geometry, &runtimepackage::compileModelPackage},
        {payloadtypes::Group, &runtimepackage::compileGroupPackage},
        {payloadtypes::Points, &compilePoints},
        {payloadtypes::Remesh, &runtimepackage::compileRemeshPackage,
            &runtimepackage::preflightRemeshPackage, &runtimepackage::publishRemeshPackage},
        {payloadtypes::Voronoi, &runtimepackage::compileVoronoiPackage},
        {payloadtypes::Heat, &runtimepackage::compileHeatPackage},
        {payloadtypes::CFD, &runtimepackage::compileCFDPackage},
    };
    for (const PackageOperation& operation : operations) {
        if (operation.payloadType == payloadType) return &operation;
    }
    return nullptr;
}

bool RuntimePackageCompiler::executeNode(
    const NodeGraphState& graph,
    const NodeGraphNode& node,
    NodeGraphRuntime& graphRuntime,
    NodePayloadRegistry& payloads,
    units::LengthUnit worldUnit,
    FrozenPackages frozenPackages,
    RuntimePackageController& backend,
    std::vector<std::string>& errors,
    bool& backpressured) const {
    const NodeGraphEvaluation& evaluation = graphRuntime.evaluation();
    RuntimePackageManager& packages = backend.packages();
    const RuntimeProductManager* products = backend.products();

    // The incoming edges have already been evaluated in topological order.
    for (const NodeGraphSocket& socket : node.outputs) {
        const uint64_t key = NodeSocketKey(node.id, socket.id).value;
        const EvaluatedSocketValue* evaluated = evaluation.outputFor(key);
        const bool required = socket.contract.requiresRuntimePackage;
        if (!required) continue;
        if (!evaluated || evaluated->status != EvaluatedSocketStatus::Value) {
            errors.push_back(
                evaluated && !evaluated->error.empty()
                    ? evaluated->error
                    : "Runtime output '" + socket.name +
                      "' on node '" + node.title + "' is unavailable.");
            return false;
        }

        const uint8_t payloadType = evaluated->data.dataType != 0
            ? evaluated->data.dataType
            : socket.contract.producedPayloadType;
        const PackageOperation* operation = operationFor(payloadType);
        const bool publishesAfterApply = operation && operation->publish;
        if (!publishesAfterApply && evaluated->data.payloadHandle.key == 0) {
            errors.push_back("Runtime output '" + socket.name +
                "' on node '" + node.title + "' is unavailable.");
            return false;
        }
        if (evaluated->data.isFrozen && frozenPackages == FrozenPackages::Preserve)
            continue;

        if (!operation || operation->payloadType != payloadType) {
            errors.push_back("Unregistered runtime payload type " +
                std::to_string(payloadType) + ".");
            return false;
        }
        const runtimepackage::PackageCompileContext context{
            graph, node, evaluated->data, key, payloads, evaluation,
            worldUnit, packages, errors, products};
        if (!operation->compile(context)) return false;
        if (operation->preflight &&
            !operation->preflight(context, graphRuntime, payloads, backpressured)) {
            return false;
        }
    }

    if (!backend.applyNode(node)) {
        errors.push_back("Runtime application failed for '" + node.title + "'.");
        return false;
    }

    for (const NodeGraphSocket& socket : node.outputs) {
        if (!socket.contract.requiresRuntimePackage) continue;
        const uint64_t key = NodeSocketKey(node.id, socket.id).value;
        const EvaluatedSocketValue* evaluated = evaluation.outputFor(key);
        if (!evaluated || evaluated->status != EvaluatedSocketStatus::Value ||
            (evaluated->data.isFrozen && frozenPackages == FrozenPackages::Preserve))
            continue;
        const uint8_t payloadType = evaluated->data.dataType != 0
            ? evaluated->data.dataType
            : socket.contract.producedPayloadType;
        const PackageOperation* operation = operationFor(payloadType);
        if (!operation || !operation->publish) continue;
        const runtimepackage::PackageCompileContext context{
            graph, node, evaluated->data, key, payloads, evaluation,
            worldUnit, packages, errors, products};
        if (!operation->publish(context, graphRuntime, payloads)) return false;
    }
    if (!graphRuntime.commitNode(node.id)) {
        errors.push_back("Completed output could not be committed for '" + node.title + "'.");
        return false;
    }
    return true;
}
