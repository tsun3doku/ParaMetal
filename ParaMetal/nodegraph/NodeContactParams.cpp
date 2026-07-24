#include "NodeContactParams.hpp"

#include "NodeGraphEditor.hpp"
#include "NodeGraphRegistry.hpp"
#include "nodegraph/NodeParamUtils.hpp"

ContactNodeParams readContactNodeParams(const NodeGraphNode& node) {
    ContactNodeParams params{};
    params.minNormalDot = NodeParamUtils::readFloatParam(node, nodegraphparams::contact::MinNormalDot, -0.65);
    params.contactRadius = NodeParamUtils::readFloatParam(node, nodegraphparams::contact::ContactRadius, 0.01);
    params.preview.showContactLines = NodeParamUtils::readBoolParam(node, nodegraphparams::contact::ShowContactLines, false);
    return params;
}

bool writeContactNodeParams(NodeGraphEditor& editor, NodeGraphNodeId nodeId, const ContactNodeParams& params) {
    return editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::contact::MinNormalDot, NodeGraphParamType::Float, params.minNormalDot}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::contact::ContactRadius, NodeGraphParamType::Float, params.contactRadius}) &&
        editor.setNodeParameter(nodeId, NodeGraphParamValue{nodegraphparams::contact::ShowContactLines, NodeGraphParamType::Bool, 0.0, 0, params.preview.showContactLines});
}
