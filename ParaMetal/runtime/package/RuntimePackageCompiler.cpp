#include "RuntimePackageCompiler.hpp"

#include "CFDPackageCompiler.hpp"
#include "HeatPackageCompiler.hpp"
#include "ModelPackageCompiler.hpp"
#include "PackageCompileContext.hpp"
#include "PointPackageCompiler.hpp"
#include "RemeshPackageCompiler.hpp"
#include "VoronoiPackageCompiler.hpp"
#include "nodegraph/NodeGraphEvaluation.hpp"
#include "nodegraph/NodeGraphPayloadTypes.hpp"
#include "nodegraph/NodeGraphState.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/package/passes/ValidationPass.hpp"

bool RuntimePackageCompiler::validate(
    const NodeGraphState& graph,
    const NodeGraphCompiled& graphPlan,
    const NodeGraphEvaluation& evaluation,
    const NodePayloadRegistry& payloads,
    std::vector<std::string>& errors) const {
    errors.clear();
    const runtimepackage::ValidationPass validationPass{};
    return validationPass.run(graph, graphPlan, evaluation, payloads, errors);
}

bool RuntimePackageCompiler::compileNode(
    const NodeGraphState& graph,
    const NodeGraphNode& node,
    const NodeGraphEvaluation& evaluation,
    const NodePayloadRegistry& payloads,
    units::LengthUnit worldUnit,
    FrozenPackages frozenPackages,
    RuntimePackageManager& packages,
    const RuntimeProductManager* products,
    std::vector<std::string>& errors) const {
    for (const NodeGraphSocket& outputSocket : node.outputs) {
        const uint64_t outputSocketKey = NodeSocketKey(node.id, outputSocket.id).value;
        const bool requiresRuntimePackage = outputSocket.contract.requiresRuntimePackage;
        const EvaluatedSocketValue* evaluated = evaluation.outputFor(outputSocketKey);
        if (!evaluated || evaluated->status != EvaluatedSocketStatus::Value ||
            evaluated->data.payloadHandle.key == 0) {
            if (requiresRuntimePackage) {
                errors.push_back(
                    "Runtime output '" + outputSocket.name + "' on node '" +
                    node.title + "' was not evaluated.");
                return false;
            }
            continue;
        }

        const NodeDataBlock& output = evaluated->data;
        if (output.isFrozen && frozenPackages == FrozenPackages::Preserve) {
            continue;
        }

        const runtimepackage::PackageCompileContext context{
            graph,
            node,
            output,
            outputSocketKey,
            payloads,
            worldUnit,
            packages,
            errors
        };

        bool compiled = true;
        if (requiresRuntimePackage && output.dataType == payloadtypes::Remesh)
            compiled = runtimepackage::compileRemeshPackage(context);
        else if (requiresRuntimePackage && output.dataType == payloadtypes::Voronoi)
            compiled = runtimepackage::compileVoronoiPackage(context);
        else if (requiresRuntimePackage && output.dataType == payloadtypes::Heat)
            compiled = runtimepackage::compileHeatPackage(context);
        else if (requiresRuntimePackage && output.dataType == payloadtypes::CFD)
            compiled = runtimepackage::compileCFDPackage(context);
        else if (output.dataType == payloadtypes::Points) {
            if (!products) {
                errors.push_back("Point package compilation requires runtime product access.");
                return false;
            }
            compiled = runtimepackage::compilePointPackage(context, *products);
        }
        else if (output.dataType == payloadtypes::Geometry)
            compiled = runtimepackage::compileModelPackage(context);
        else if (requiresRuntimePackage) {
            errors.push_back("No package compiler is registered for runtime output type " +
                std::to_string(output.dataType) + ".");
            return false;
        }
        if (!compiled) return false;
    }

    return true;
}
