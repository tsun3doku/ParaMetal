#include "NodeMerge.hpp"
#include "NodeGraphDefinitionUtils.hpp"
#include "NodeMergeParams.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "NodePayloadRegistry.hpp"
#include "domain/GeometryData.hpp"
#include "domain/PointData.hpp"

#include <cmath>
#include <limits>
#include "../util/GeometryUtils.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <unordered_map>
#include <vector>
#include <algorithm>

static NodeTypeDefinition buildMergeNode() {
    return {
        nodegraphtypes::Merge,
        "Merge",
        NodeGraphNodeCategory::Meshing,
        {
            makePolymorphicInputSocket("Geometry", {NodeGraphValueType::Mesh, NodeGraphValueType::Points}, true),
            makePolymorphicOutputSocket("Geometry", {NodeGraphValueType::Mesh, NodeGraphValueType::Points}, payloadtypes::None, true),
        },
        {
            {nodegraphparams::merge::MergeDistance, "Merge Distance", NodeGraphParamType::Float, 0.0005, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildMergeDefinition() {
    return buildMergeNode();
}

std::unique_ptr<NodeKernel> createMergeKernel() {
    return std::make_unique<NodeMerge>();
}

const char* NodeMerge::typeId() const {
    return nodegraphtypes::Merge;
}

void NodeMerge::execute(NodeKernelEval& eval) const {
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;
    if (!payloadRegistry) {
        return;
    }

    const std::size_t geometrySocketIndex = inputIndexOf(eval.node, "Geometry");
    if (geometrySocketIndex >= eval.inputs.size()) {
        return;
    }

    const std::vector<const NodeDataBlock*>& inputs = eval.inputs[geometrySocketIndex];

    bool hasPoints = false;
    bool hasGeometry = false;
    for (const NodeDataBlock* block : inputs) {
        if (!block || block->payloadHandle.key == 0) {
            continue;
        }
        if (block->dataType == payloadtypes::Points) {
            hasPoints = true;
        } else if (block->dataType == payloadtypes::Geometry ||
                   block->dataType == payloadtypes::Remesh) {
            hasGeometry = true;
        }
    }

    NodeDataBlock& outputValue = eval.outputs[0];
    outputValue = {};

    if (!hasPoints && !hasGeometry) {
        return;
    }
    if (hasPoints && hasGeometry) {
        eval.error = "Merge cannot mix Points with Geometry or Remesh inputs.";
        return;
    }

    if (hasPoints) {
        PointData payload{};
        bool haveReference = false;
        for (const NodeDataBlock* block : inputs) {
            if (!block || block->dataType != payloadtypes::Points || block->payloadHandle.key == 0) {
                eval.error = "Point Merge requires valid Points inputs.";
                return;
            }
            const PointData* pointData = payloadRegistry->resolvePoints(block->payloadHandle);
            if (!pointData || !pointData->active) {
                eval.error = "Point Merge received an inactive Point input.";
                return;
            }
            payload.sources.push_back(block->payloadHandle);
            if (!haveReference) {
                if (!payloadRegistry->resolveLocalToWorld(
                        block->payloadHandle,
                        payload.localToWorld)) {
                    eval.error = "Point Merge cannot resolve its reference placement.";
                    return;
                }
                haveReference = true;
            }
        }
        if (payload.sources.empty()) {
            eval.error = "Point Merge has no valid point inputs.";
            return;
        }
        payload.mergeDistance = readMergeNodeParams(eval.node).mergeDistance;
        payload.active = true;
        const uint64_t payloadKey = NodeSocketKey(eval.node.id, eval.node.outputs[0].id);
        outputValue.dataType = payloadtypes::Points;
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, eval.outputHashes);
    } else if (hasGeometry) {
        std::vector<const GeometryData*> geometryInputs;
        std::vector<std::array<float, 16>> geometryMatrices;
        for (const NodeDataBlock* block : inputs) {
            if (block &&
                (block->dataType == payloadtypes::Geometry || block->dataType == payloadtypes::Remesh) &&
                block->payloadHandle.key != 0) {
                const GeometryData* geometryData = payloadRegistry->resolveGeometry(block->payloadHandle);
                std::array<float, 16> localToWorld{};
                if (geometryData && !geometryData->vertexPositions.empty() &&
                    payloadRegistry->resolveLocalToWorld(block->payloadHandle, localToWorld)) {
                    geometryInputs.push_back(geometryData);
                    geometryMatrices.push_back(localToWorld);
                }
            }
        }

        if (geometryInputs.empty()) {
            return;
        }

        bool carryRenderMesh = true;
        std::size_t renderVertexCount = 0;
        std::size_t renderIndexCount = 0;
        for (const GeometryData* input : geometryInputs) {
            const std::size_t cornerCount = input->renderPositions.size() / 3;
            if (input->renderPositions.empty() ||
                input->renderPositions.size() % 3 != 0 ||
                input->renderNormals.size() != input->renderPositions.size() ||
                input->renderTexcoords.size() != cornerCount * 2 ||
                input->renderIndices.size() != input->triangleIndices.size()) {
                carryRenderMesh = false;
                break;
            }
            renderVertexCount += cornerCount;
            renderIndexCount += input->renderIndices.size();
        }

        GeometryData payload{};
        payload.localToWorld = geometryMatrices[0];
        payload.baseModelPath = geometryInputs[0]->baseModelPath;

        if (carryRenderMesh) {
            payload.renderPositions.reserve(renderVertexCount * 3);
            payload.renderNormals.reserve(renderVertexCount * 3);
            payload.renderTexcoords.reserve(renderVertexCount * 2);
            payload.renderIndices.reserve(renderIndexCount);
        }

        glm::mat4 referenceL2W = toMat4(payload.localToWorld);
        glm::mat4 invReferenceL2W = glm::inverse(referenceL2W);

        std::vector<GeometryGroup> mergedGroups;
        std::vector<uint32_t> mergedTriangleGroupIds;
        uint32_t nextGroupId = 1;

        for (size_t i = 0; i < geometryInputs.size(); ++i) {
            const GeometryData* input = geometryInputs[i];
            glm::mat4 inputL2W = toMat4(geometryMatrices[i]);
            glm::mat4 inputToReference = invReferenceL2W * inputL2W;

            const uint32_t vertexOffset = static_cast<uint32_t>(payload.vertexPositions.size() / 3);

            // Merge vertex positions
            for (size_t vIdx = 0; vIdx + 2 < input->vertexPositions.size(); vIdx += 3) {
                glm::vec4 localPos(input->vertexPositions[vIdx], input->vertexPositions[vIdx+1], input->vertexPositions[vIdx+2], 1.0f);
                glm::vec4 refPos = inputToReference * localPos;
                payload.vertexPositions.push_back(refPos.x);
                payload.vertexPositions.push_back(refPos.y);
                payload.vertexPositions.push_back(refPos.z);
            }

            // Merge triangle indices
            for (uint32_t idx : input->triangleIndices) {
                payload.triangleIndices.push_back(idx + vertexOffset);
            }

            if (carryRenderMesh) {
                const uint32_t renderVertexOffset = static_cast<uint32_t>(payload.renderPositions.size() / 3);
                const glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(inputToReference)));

                for (std::size_t corner = 0; corner + 2 < input->renderPositions.size(); corner += 3) {
                    const glm::vec4 localPos(
                        input->renderPositions[corner],
                        input->renderPositions[corner + 1],
                        input->renderPositions[corner + 2],
                        1.0f);
                    const glm::vec4 refPos = inputToReference * localPos;
                    payload.renderPositions.push_back(refPos.x);
                    payload.renderPositions.push_back(refPos.y);
                    payload.renderPositions.push_back(refPos.z);

                    glm::vec3 normal = normalMatrix * glm::vec3(
                        input->renderNormals[corner],
                        input->renderNormals[corner + 1],
                        input->renderNormals[corner + 2]);
                    const float lengthSquared = glm::dot(normal, normal);
                    if (std::isfinite(lengthSquared) && lengthSquared > 1e-12f) {
                        normal *= 1.0f / std::sqrt(lengthSquared);
                    } else {
                        normal = glm::vec3(0.0f);
                    }
                    payload.renderNormals.push_back(normal.x);
                    payload.renderNormals.push_back(normal.y);
                    payload.renderNormals.push_back(normal.z);
                }

                payload.renderTexcoords.insert(
                    payload.renderTexcoords.end(),
                    input->renderTexcoords.begin(),
                    input->renderTexcoords.end());

                for (uint32_t renderIndex : input->renderIndices) {
                    payload.renderIndices.push_back(renderIndex + renderVertexOffset);
                }
            }

            // Merge groups
            std::unordered_map<uint32_t, uint32_t> groupIdMap;
            groupIdMap[0] = 0; // 0 stays 0
            for (const auto& inputGroup : input->groups) {
                uint32_t targetId = 0;
                for (const auto& existing : mergedGroups) {
                    if (existing.name == inputGroup.name) {
                        targetId = existing.id;
                        break;
                    }
                }

                if (targetId == 0) {
                    targetId = nextGroupId++;
                    GeometryGroup newGroup = inputGroup;
                    newGroup.id = targetId;
                    mergedGroups.push_back(newGroup);
                }

                groupIdMap[inputGroup.id] = targetId;
            }

            // Merge triangle group IDs
            for (uint32_t gId : input->triangleGroupIds) {
                auto it = groupIdMap.find(gId);
                mergedTriangleGroupIds.push_back(it != groupIdMap.end() ? it->second : 0);
            }
        }

        payload.groups = mergedGroups;
        payload.triangleGroupIds = mergedTriangleGroupIds;

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, eval.node.outputs[0].id);
        outputValue.dataType = payloadtypes::Geometry;
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, eval.outputHashes);
    }
}

HashValues NodeMerge::computeOutputHashes(const NodeKernelHash& hash) const {
    const MergeNodeParams params = readMergeNodeParams(hash.node);

    uint64_t hashValue = HashBuilder::start();
    HashBuilder::combineString(hashValue, nodegraphtypes::Merge);
    HashBuilder::combineFloat(hashValue, params.mergeDistance);
    HashNodeCache::combineSocketList(hashValue, hash, "Geometry", HashDomain::Geometry);

    HashValues values{};
    values.full = hashValue;
    values.geometry = hashValue;
    values.simulation = hashValue;
    return values;
}
