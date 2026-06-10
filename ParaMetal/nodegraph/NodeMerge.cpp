#include "NodeMerge.hpp"
#include "NodeGraphHash.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "NodePayloadRegistry.hpp"
#include "domain/GeometryData.hpp"
#include "domain/PointData.hpp"
#include "../util/GeometryUtils.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <unordered_map>
#include <vector>
#include <algorithm>

const char* NodeMerge::typeId() const {
    return nodegraphtypes::Merge;
}

void NodeMerge::execute(NodeGraphKernelContext& context) const {
    NodePayloadRegistry* const payloadRegistry = context.executionState.services.payloadRegistry;
    if (!payloadRegistry) {
        return;
    }

    const NodeGraphSocket* inSocket = context.node.input("Geometry");
    if (!inSocket) {
        return;
    }

    const auto evals = readEvaluatedInputs(context.node, inSocket->id, context.executionState);
    
    // Determine the effective dataType based on the first valid input
    uint8_t dataType = payloadtypes::None;
    for (const auto* eval : evals) {
        if (eval && eval->status == EvaluatedSocketStatus::Value && eval->data.payloadHandle.key != 0) {
            if (eval->data.dataType == payloadtypes::Points ||
                eval->data.dataType == payloadtypes::Geometry ||
                eval->data.dataType == payloadtypes::Remesh) {
                dataType = eval->data.dataType;
                break;
            }
        }
    }

    NodeDataBlock& outputValue = context.outputs[0];
    outputValue = {};

    if (dataType == payloadtypes::Points) {
        std::vector<const PointData*> pointInputs;
        for (const auto* eval : evals) {
            if (eval && eval->status == EvaluatedSocketStatus::Value && eval->data.dataType == payloadtypes::Points && eval->data.payloadHandle.key != 0) {
                const PointData* pointData = payloadRegistry->get<PointData>(eval->data.payloadHandle);
                if (pointData && pointData->active && !pointData->positions.empty()) {
                    pointInputs.push_back(pointData);
                }
            }
        }

        if (pointInputs.empty()) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            return;
        }

        PointData payload{};
        payload.localToWorld = pointInputs[0]->localToWorld;
        payload.active = true;

        glm::mat4 referenceL2W = toMat4(payload.localToWorld);
        glm::mat4 invReferenceL2W = glm::inverse(referenceL2W);

        for (size_t i = 0; i < pointInputs.size(); ++i) {
            const PointData* input = pointInputs[i];
            glm::mat4 inputL2W = toMat4(input->localToWorld);
            glm::mat4 inputToReference = invReferenceL2W * inputL2W;

            for (const auto& pos : input->positions) {
                payload.positions.push_back(inputToReference * pos);
            }
        }

        payload.sealPayload();
        const uint64_t payloadKey = NodeSocketKey(context.node.id, context.node.outputs[0].id);
        outputValue.dataType = payloadtypes::Points;
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(payload));
        populateMetadata(outputValue, nullptr, payloadRegistry);

    } else if (dataType == payloadtypes::Geometry || dataType == payloadtypes::Remesh) {
        std::vector<const GeometryData*> geometryInputs;
        for (const auto* eval : evals) {
            if (eval && eval->status == EvaluatedSocketStatus::Value && 
                (eval->data.dataType == payloadtypes::Geometry || eval->data.dataType == payloadtypes::Remesh) && 
                eval->data.payloadHandle.key != 0) {
                const GeometryData* geometryData = payloadRegistry->resolveGeometry(eval->data.payloadHandle);
                if (geometryData && !geometryData->pointPositions.empty()) {
                    geometryInputs.push_back(geometryData);
                }
            }
        }

        if (geometryInputs.empty()) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            return;
        }

        GeometryData payload{};
        payload.localToWorld = geometryInputs[0]->localToWorld;
        payload.baseModelPath = geometryInputs[0]->baseModelPath;

        glm::mat4 referenceL2W = toMat4(payload.localToWorld);
        glm::mat4 invReferenceL2W = glm::inverse(referenceL2W);

        std::vector<GeometryGroup> mergedGroups;
        std::vector<uint32_t> mergedTriangleGroupIds;
        uint32_t nextGroupId = 1;

        for (size_t i = 0; i < geometryInputs.size(); ++i) {
            const GeometryData* input = geometryInputs[i];
            glm::mat4 inputL2W = toMat4(input->localToWorld);
            glm::mat4 inputToReference = invReferenceL2W * inputL2W;

            const uint32_t vertexOffset = static_cast<uint32_t>(payload.pointPositions.size() / 3);

            // Merge vertex positions
            for (size_t vIdx = 0; vIdx + 2 < input->pointPositions.size(); vIdx += 3) {
                glm::vec4 localPos(input->pointPositions[vIdx], input->pointPositions[vIdx+1], input->pointPositions[vIdx+2], 1.0f);
                glm::vec4 refPos = inputToReference * localPos;
                payload.pointPositions.push_back(refPos.x);
                payload.pointPositions.push_back(refPos.y);
                payload.pointPositions.push_back(refPos.z);
            }

            // Merge triangle indices
            for (uint32_t idx : input->triangleIndices) {
                payload.triangleIndices.push_back(idx + vertexOffset);
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

        payload.groups = std::move(mergedGroups);
        payload.triangleGroupIds = std::move(mergedTriangleGroupIds);

        payload.sealPayload();
        const uint64_t payloadKey = NodeSocketKey(context.node.id, context.node.outputs[0].id);
        outputValue.dataType = payloadtypes::Geometry;
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(payload));
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

bool NodeMerge::computeInputHash(const NodeGraphKernelHashContext& context, uint64_t& outHash) const {
    outHash = NodeGraphHash::start();
    NodeGraphHash::combine(outHash, static_cast<uint64_t>(context.node.id.value));

    const NodeGraphSocket* inSocket = context.node.input("Geometry");
    if (inSocket) {
        const auto evals = readEvaluatedInputs(context.node, inSocket->id, context.executionState);
        for (const EvaluatedSocketValue* input : evals) {
            const NodeDataBlock* inputData = (input && input->status == EvaluatedSocketStatus::Value) ? &input->data : nullptr;
            NodeGraphHash::combineInputHash(outHash, inputData);
        }
    }
    return true;
}
