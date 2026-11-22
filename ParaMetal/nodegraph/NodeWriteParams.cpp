#include "NodeWriteParams.hpp"

#include "NodeGraphDefinitionUtils.hpp"
#include "NodeGraphKernels.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "nodegraph/NodeParamUtils.hpp"

#include <memory>
#include <vector>

static NodeTypeDefinition buildWriteNode() {
    NodeGraphParamDefinition producerNode{};
    producerNode.id = 1;
    producerNode.name = "producerNodeId";
    producerNode.type = NodeGraphParamType::Int;
    NodeGraphParamDefinition outputSocket{};
    outputSocket.id = 2;
    outputSocket.name = "outputSocketId";
    outputSocket.type = NodeGraphParamType::Int;
    NodeGraphParamDefinition productId{};
    productId.id = 3;
    productId.name = "productId";
    productId.type = NodeGraphParamType::String;
    NodeGraphParamDefinition productFormat = makeEnumParamDefinition(
        4, "format", "OBJ", {"OBJ", "PLY", "STL", "XYZ", "CSV"});
    NodeGraphParamDefinition productStruct = makeStructParamDefinition(
        nodegraphparams::write::Product, "Product", {
            {"producerNodeId", std::make_shared<NodeGraphParamDefinition>(producerNode)},
            {"outputSocketId", std::make_shared<NodeGraphParamDefinition>(outputSocket)},
            {"productId", std::make_shared<NodeGraphParamDefinition>(productId)},
            {"format", std::make_shared<NodeGraphParamDefinition>(productFormat)},
        });
    return {
        nodegraphtypes::Write,
        "Write",
        NodeGraphNodeCategory::Misc,
        {
            makePolymorphicInputSocket("Data", {NodeGraphValueType::Mesh, NodeGraphValueType::Remesh, NodeGraphValueType::Points, NodeGraphValueType::Simulation}, false),
        },
        {
            {nodegraphparams::write::FilePath, "File", NodeGraphParamType::String, 0.0, 0, false, "", false},
            std::move(productStruct),
        },
    };
}

NodeTypeDefinition buildWriteDefinition() {
    return buildWriteNode();
}

WriteNodeParams readWriteNodeParams(const NodeGraphNode& node) {
    WriteNodeParams params{};
    params.filePath = NodeParamUtils::readStringParam(node, nodegraphparams::write::FilePath);

    if (const NodeGraphParamValue* product = findNodeParamValue(node, nodegraphparams::write::Product)) {
        if (product->type == NodeGraphParamType::Struct) {
            for (const NodeGraphParamFieldValue& field : product->fieldValues) {
                if (!field.value) continue;
                if (field.name == "producerNodeId") params.product.producerNodeId = static_cast<uint32_t>(field.value->intValue);
                else if (field.name == "outputSocketId") params.product.outputSocketId = static_cast<uint32_t>(field.value->intValue);
                else if (field.name == "productId") params.product.productId = field.value->stringValue;
                else if (field.name == "format" && !field.value->enumValue.empty()) params.product.format = field.value->enumValue;
            }
        }
    }

    return params;
}
