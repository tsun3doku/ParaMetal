#include "NodeGroupParams.hpp"

#include "NodeParamUtils.hpp"

NodeGroupParams readNodeGroupParams(const NodeGraphNode& node) {
    NodeGroupParams params{};
    params.elementType = static_cast<MeshElementType>(
        NodeParamUtils::readEnumParam(
            node,
            nodegraphparams::group::ElementType,
            static_cast<int>(MeshElementType::Face)));
    params.groupName = NodeParamUtils::readStringParam(
        node, nodegraphparams::group::GroupName);
    params.operation = static_cast<GroupOperation>(
        NodeParamUtils::readEnumParam(
            node,
            nodegraphparams::group::Operation,
            static_cast<int>(GroupOperation::Replace)));
    params.useBaseGroup = NodeParamUtils::readBoolParam(
        node, nodegraphparams::group::UseBaseGroup);
    params.selectedElements = NodeParamUtils::readStringParam(
        node, nodegraphparams::group::SelectedElements);
    params.selectionSignature = NodeParamUtils::readStringParam(
        node, nodegraphparams::group::SelectionSignature);
    return params;
}
