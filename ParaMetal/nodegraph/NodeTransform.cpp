#include "NodeTransform.hpp"
#include "NodeGraphDefinitionUtils.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodePayloadRegistry.hpp"
#include "NodeTransformParams.hpp"
#include "domain/PointData.hpp"
#include "domain/TransformData.hpp"
#include "../util/GeometryUtils.hpp"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <utility>

static NodeTypeDefinition buildTransformNode() {
    return {
        nodegraphtypes::Transform,
        "Transform",
        NodeGraphNodeCategory::Meshing,
        {
            makePolymorphicInputSocket("Geometry", {NodeGraphValueType::Mesh, NodeGraphValueType::Points}, false, false),
            makePolymorphicOutputSocket("Geometry", {NodeGraphValueType::Mesh, NodeGraphValueType::Points}, payloadtypes::None, true),
        },
        {
            {nodegraphparams::transform::TranslateX, "Translate X", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::transform::TranslateY, "Translate Y", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::transform::TranslateZ, "Translate Z", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::transform::RotateXDegrees, "Rotate X", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::transform::RotateYDegrees, "Rotate Y", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::transform::RotateZDegrees, "Rotate Z", NodeGraphParamType::Float, 0.0, 0, false, "", false},
            {nodegraphparams::transform::ScaleX, "Scale X", NodeGraphParamType::Float, 1.0, 0, false, "", false},
            {nodegraphparams::transform::ScaleY, "Scale Y", NodeGraphParamType::Float, 1.0, 0, false, "", false},
            {nodegraphparams::transform::ScaleZ, "Scale Z", NodeGraphParamType::Float, 1.0, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildTransformDefinition() {
    return buildTransformNode();
}

std::unique_ptr<NodeKernel> createTransformKernel() {
    return std::make_unique<NodeTransform>();
}

const char* NodeTransform::typeId() const {
    return nodegraphtypes::Transform;
}

void NodeTransform::execute(NodeKernelEval& eval) const {
    const std::size_t geometrySocketIndex = inputIndexOf(eval.node, "Geometry");
    const NodeDataBlock* inputData =
        (geometrySocketIndex < eval.inputs.size() && !eval.inputs[geometrySocketIndex].empty())
            ? eval.inputs[geometrySocketIndex].front() : nullptr;

    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;
    const std::array<float, 16> localTransform = buildLocalTransformArray(eval.node);

    NodeDataBlock& outputValue = eval.outputs[0];
    outputValue = {};

    if (payloadRegistry && inputData && inputData->payloadHandle.key != 0) {
        std::array<float, 16> sourceLocalToWorld{};
        const bool hasGeometry = payloadRegistry->resolveGeometry(inputData->payloadHandle) != nullptr;
        const bool hasPoints = payloadRegistry->resolvePoints(inputData->payloadHandle) != nullptr;
        if ((hasGeometry || hasPoints) &&
            payloadRegistry->resolveLocalToWorld(inputData->payloadHandle, sourceLocalToWorld)) {
            TransformData transform{};
            transform.sourceHandle = inputData->payloadHandle;
            transform.localToWorld = toMatrixArray(toMat4(sourceLocalToWorld) * toMat4(localTransform));
            const uint64_t payloadKey = NodeSocketKey(eval.node.id, eval.node.outputs[0].id);
            outputValue.dataType = inputData->dataType;
            outputValue.payloadHandle = payloadRegistry->store(payloadKey, transform, eval.outputHashes);
            if (outputValue.payloadHandle.key == 0) {
                eval.error = "Payload publication backpressured.";
                return;
            }
        }
    }
}

HashValues NodeTransform::computeOutputHashes(const NodeKernelHash& hash) const {
    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combineString(geometryHash, nodegraphtypes::Transform);
    HashNodeCache::combineOptionalSocket(geometryHash, hash, "Geometry", HashDomain::Geometry);
    combineTransformParams(hash.node, geometryHash);

    uint64_t propertiesHash = HashBuilder::start();
    HashBuilder::combineString(propertiesHash, "transform.properties");
    HashNodeCache::combineOptionalSocket(propertiesHash, hash, "Geometry", HashDomain::Properties);

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combineString(fullHash, nodegraphtypes::Transform);
    HashNodeCache::combineOptionalSocket(fullHash, hash, "Geometry", HashDomain::Full);
    combineTransformParams(hash.node, fullHash);

    HashValues values{};
    values.full = fullHash;
    values.geometry = geometryHash;
    values.properties = propertiesHash;
    values.simulation = fullHash;
    return values;
}

glm::mat4 NodeTransform::buildLocalTransform(const NodeGraphNode& node) {
    return buildLocalTransform(readTransformNodeParams(node));
}

glm::mat4 NodeTransform::buildLocalTransform(const TransformNodeParams& params) {
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
    HashBuilder::combineFloat(outHash, static_cast<float>(params.translateX));
    HashBuilder::combineFloat(outHash, static_cast<float>(params.translateY));
    HashBuilder::combineFloat(outHash, static_cast<float>(params.translateZ));
    HashBuilder::combineFloat(outHash, static_cast<float>(params.rotateXDegrees));
    HashBuilder::combineFloat(outHash, static_cast<float>(params.rotateYDegrees));
    HashBuilder::combineFloat(outHash, static_cast<float>(params.rotateZDegrees));
    HashBuilder::combineFloat(outHash, static_cast<float>(params.scaleX));
    HashBuilder::combineFloat(outHash, static_cast<float>(params.scaleY));
    HashBuilder::combineFloat(outHash, static_cast<float>(params.scaleZ));
}

std::array<float, 16> NodeTransform::buildLocalTransformArray(const NodeGraphNode& node) {
    return toMatrixArray(buildLocalTransform(node));
}

bool NodeTransform::beginEdit(
    const NodeGraphNode& node,
    const glm::mat4& currentWorld,
    units::LengthUnit worldUnit,
    int mode,
    int axis,
    NodeTransformEditState& outState) {
    outState = {};
    outState.start = readTransformNodeParams(node);
    outState.preview = outState.start;
    outState.canonicalToWorldScale = units::canonicalToWorldScale(worldUnit);
    outState.mode = mode;
    outState.axis = axis;

    glm::mat4 initialLocal = buildLocalTransform(outState.start);
    initialLocal[3][0] *= outState.canonicalToWorldScale;
    initialLocal[3][1] *= outState.canonicalToWorldScale;
    initialLocal[3][2] *= outState.canonicalToWorldScale;

    const glm::mat4 inverseLocal = glm::inverse(initialLocal);
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            if (!std::isfinite(inverseLocal[column][row])) {
                return false;
            }
        }
    }

    outState.parentWorld = currentWorld * inverseLocal;
    const glm::mat3 parentInverse = glm::inverse(glm::mat3(outState.parentWorld));
    for (int column = 0; column < 3; ++column) {
        for (int row = 0; row < 3; ++row) {
            if (!std::isfinite(parentInverse[column][row])) {
                return false;
            }
        }
    }
    return true;
}

bool NodeTransform::previewEdit(
    NodeTransformEditState& state,
    const glm::vec3& translationDeltaWorld,
    float rotationDeltaDegrees,
    glm::mat4& outWorld) {
    state.preview = state.start;
    if (state.mode == 0) {
        const glm::vec3 localDelta =
            glm::inverse(glm::mat3(state.parentWorld)) *
            translationDeltaWorld /
            state.canonicalToWorldScale;
        state.preview.translateX += localDelta.x;
        state.preview.translateY += localDelta.y;
        state.preview.translateZ += localDelta.z;
    } else if (state.mode == 1) {
        if (state.axis == 1) {
            state.preview.rotateXDegrees += rotationDeltaDegrees;
        } else if (state.axis == 2) {
            state.preview.rotateYDegrees += rotationDeltaDegrees;
        } else if (state.axis == 3) {
            state.preview.rotateZDegrees += rotationDeltaDegrees;
        } else {
            return false;
        }
    } else {
        return false;
    }

    glm::mat4 previewLocal = buildLocalTransform(state.preview);
    previewLocal[3][0] *= state.canonicalToWorldScale;
    previewLocal[3][1] *= state.canonicalToWorldScale;
    previewLocal[3][2] *= state.canonicalToWorldScale;
    outWorld = state.parentWorld * previewLocal;
    return true;
}

bool NodeTransform::buildEditParameters(
    const NodeTransformEditState& state,
    std::vector<NodeGraphParamValue>& outParameters) {
    outParameters.clear();
    if (state.mode == 0) {
        outParameters = {
            {nodegraphparams::transform::TranslateX, NodeGraphParamType::Float, state.preview.translateX},
            {nodegraphparams::transform::TranslateY, NodeGraphParamType::Float, state.preview.translateY},
            {nodegraphparams::transform::TranslateZ, NodeGraphParamType::Float, state.preview.translateZ}
        };
        return true;
    }
    if (state.mode == 1) {
        outParameters = {
            {nodegraphparams::transform::RotateXDegrees, NodeGraphParamType::Float, state.preview.rotateXDegrees},
            {nodegraphparams::transform::RotateYDegrees, NodeGraphParamType::Float, state.preview.rotateYDegrees},
            {nodegraphparams::transform::RotateZDegrees, NodeGraphParamType::Float, state.preview.rotateZDegrees}
        };
        return true;
    }
    return false;
}
