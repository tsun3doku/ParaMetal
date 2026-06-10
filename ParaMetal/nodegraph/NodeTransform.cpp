#include "NodeTransform.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "NodeGraphHash.hpp"
#include "NodePayloadRegistry.hpp"
#include "NodeTransformParams.hpp"
#include "domain/PointData.hpp"
#include "../util/GeometryUtils.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <utility>

const char* NodeTransform::typeId() const {
    return nodegraphtypes::Transform;
}

void NodeTransform::execute(NodeGraphKernelContext& context) const {
    const NodeGraphSocket* inSocket = context.node.input("Geometry");
    const EvaluatedSocketValue* inputEval =
        inSocket ? readEvaluatedInput(context.node, inSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* inputData = readInputValue(inputEval);

    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;
    const std::array<float, 16> localTransform = buildLocalTransformArray(context.node);
    const glm::mat4 transformMat = toMat4(localTransform);

    NodeDataBlock& outputValue = context.outputs[0];
    outputValue = {};

    if (payloadRegistry && inputData && inputData->payloadHandle.key != 0) {
        if (const GeometryData* inputGeometry = payloadRegistry->resolveGeometry(inputData->payloadHandle)) {
            GeometryData forwardedGeometry = *inputGeometry;
            forwardedGeometry.localToWorld = toMatrixArray(
                toMat4(forwardedGeometry.localToWorld) * transformMat);
            const uint64_t payloadKey = NodeSocketKey(context.node.id, context.node.outputs[0].id);
            outputValue.dataType = payloadtypes::Geometry;
            outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(forwardedGeometry));
        } else if (const PointData* inputPointData = payloadRegistry->resolvePoints(inputData->payloadHandle)) {
            if (!inputPointData->positions.empty()) {
                PointData forwardedPoints = *inputPointData;
                forwardedPoints.localToWorld = toMatrixArray(
                    toMat4(forwardedPoints.localToWorld) * transformMat);
                forwardedPoints.sealPayload();
                const uint64_t payloadKey = NodeSocketKey(context.node.id, context.node.outputs[0].id);
                outputValue.dataType = payloadtypes::Points;
                outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(forwardedPoints));
            }
        }
    }

    populateMetadata(outputValue, nullptr, payloadRegistry);
}

bool NodeTransform::computeInputHash(const NodeGraphKernelHashContext& context, uint64_t& outHash) const {
    const NodeGraphSocket* inSocket = context.node.input("Geometry");
    const EvaluatedSocketValue* inputEval =
        inSocket ? readEvaluatedInput(context.node, inSocket->id, context.executionState) : nullptr;
    const NodeDataBlock* inputData = readInputValue(inputEval);

    outHash = NodeGraphHash::start();
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(context.node.id.value));
    NodeGraphHash::combineInputHash(outHash, inputData);
    combineTransformParams(context.node, outHash);
    return true;
}

glm::mat4 NodeTransform::buildLocalTransform(const NodeGraphNode& node) {
    const TransformNodeParams params = readTransformNodeParams(node);
    const float translateX = static_cast<float>(params.translateX);
    const float translateY = static_cast<float>(params.translateY);
    const float translateZ = static_cast<float>(params.translateZ);
    const float rotateXDegrees = static_cast<float>(params.rotateXDegrees);
    const float rotateYDegrees = static_cast<float>(params.rotateYDegrees);
    const float rotateZDegrees = static_cast<float>(params.rotateZDegrees);
    const float scaleX = static_cast<float>(params.scaleX);
    const float scaleY = static_cast<float>(params.scaleY);
    const float scaleZ = static_cast<float>(params.scaleZ);

    glm::mat4 transform(1.0f);
    transform = glm::translate(transform, glm::vec3(translateX, translateY, translateZ));
    transform = glm::rotate(transform, glm::radians(rotateXDegrees), glm::vec3(1.0f, 0.0f, 0.0f));
    transform = glm::rotate(transform, glm::radians(rotateYDegrees), glm::vec3(0.0f, 1.0f, 0.0f));
    transform = glm::rotate(transform, glm::radians(rotateZDegrees), glm::vec3(0.0f, 0.0f, 1.0f));
    transform = glm::scale(transform, glm::vec3(scaleX, scaleY, scaleZ));
    return transform;
}

void NodeTransform::combineTransformParams(const NodeGraphNode& node, uint64_t& outHash) {
    const TransformNodeParams params = readTransformNodeParams(node);
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.translateX));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.translateY));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.translateZ));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.rotateXDegrees));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.rotateYDegrees));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.rotateZDegrees));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.scaleX));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.scaleY));
    NodeGraphHash::combineFloat(outHash, static_cast<float>(params.scaleZ));
}

std::array<float, 16> NodeTransform::buildLocalTransformArray(const NodeGraphNode& node) {
    return toMatrixArray(buildLocalTransform(node));
}
