#include "NodeWriteParams.hpp"

#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "nodegraph/NodeParamUtils.hpp"

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
