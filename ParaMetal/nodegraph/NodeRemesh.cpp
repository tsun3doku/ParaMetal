#include "NodeRemesh.hpp"
#include "NodeGraphDefinitionUtils.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "NodeGraph.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeRemeshParams.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"

static NodeTypeDefinition buildRemeshNode() {
    return {
        nodegraphtypes::Remesh,
        "Remesh",
        NodeGraphNodeCategory::Meshing,
        {
            makeInputSocket("Mesh", NodeGraphValueType::Mesh),
            makeOutputSocket("Remesh", NodeGraphValueType::Remesh, payloadtypes::Remesh, true),
        },
        {
            {nodegraphparams::remesh::Iterations, "Iterations", NodeGraphParamType::Int, 0.0, 1, false, "", false},
            {nodegraphparams::remesh::MinAngleDegrees, "Min Angle", NodeGraphParamType::Float, 20.0, 0, false, "", false},
            {nodegraphparams::remesh::MaxEdgeLength, "Max Edge Length", NodeGraphParamType::Float, 0.1, 0, false, "", false},
            {nodegraphparams::remesh::StepSize, "Step Size", NodeGraphParamType::Float, 0.25, 0, false, "", false},
            {nodegraphparams::remesh::RunRequested, "Run Requested", NodeGraphParamType::Bool, 0.0, 0, false, "", true},
            {nodegraphparams::remesh::ShowFaceNormals, "Show Face Normals", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::remesh::ShowVertexNormals, "Show Vertex Normals", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::remesh::NormalLength, "Normal Length", NodeGraphParamType::Float, 0.05, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildRemeshDefinition() {
    return buildRemeshNode();
}

std::unique_ptr<NodeKernel> createRemeshKernel() {
    return std::make_unique<NodeRemesh>();
}

const char* NodeRemesh::typeId() const {
    return nodegraphtypes::Remesh;
}

void NodeRemesh::execute(NodeKernelEval& eval) const {
    for (std::size_t outputIndex = 0; outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;
    }
}

HashValues NodeRemesh::computeOutputHashes(const NodeKernelHash& hash) const {
    const RemeshNodeParams params = readRemeshNodeParams(hash.node);
    uint64_t hashValue = HashBuilder::start();
    HashBuilder::combineString(hashValue, nodegraphtypes::Remesh);
    HashNodeCache::combineSocket(hashValue, hash, NodeGraphValueType::Mesh, HashDomain::Geometry);
    HashBuilder::combine(hashValue, static_cast<uint64_t>(params.iterations));
    HashBuilder::combineFloat(hashValue, static_cast<float>(params.minAngleDegrees));
    HashBuilder::combineFloat(hashValue, static_cast<float>(params.maxEdgeLength));
    HashBuilder::combineFloat(hashValue, static_cast<float>(params.stepSize));

    HashValues values{};
    values.full = hashValue;
    values.geometry = hashValue;
    values.simulation = hashValue;
    return values;
}
