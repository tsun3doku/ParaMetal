#include "NodeMerge.hpp"
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

    // Determine the dataType 
    uint8_t dataType = payloadtypes::None;
    for (const NodeDataBlock* block : inputs) {
        if (block && block->payloadHandle.key != 0) {
            if (block->dataType == payloadtypes::Points ||
                block->dataType == payloadtypes::Geometry ||
                block->dataType == payloadtypes::Remesh) {
                dataType = block->dataType;
                break;
            }
        }
    }

    NodeDataBlock& outputValue = eval.outputs[0];
    outputValue = {};

    if (dataType == payloadtypes::Points) {
        std::vector<const PointData*> pointInputs;
        std::vector<std::array<float, 16>> pointMatrices;
        for (const NodeDataBlock* block : inputs) {
            if (block && block->dataType == payloadtypes::Points && block->payloadHandle.key != 0) {
                const PointData* pointData = payloadRegistry->resolvePoints(block->payloadHandle);
                std::array<float, 16> localToWorld{};
                if (pointData && pointData->active && !pointData->positions.empty() &&
                    payloadRegistry->resolveLocalToWorld(block->payloadHandle, localToWorld)) {
                    pointInputs.push_back(pointData);
                    pointMatrices.push_back(localToWorld);
                }
            }
        }

        if (pointInputs.empty()) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            return;
        }

        const MergeNodeParams mergeParams = readMergeNodeParams(eval.node);
        const float mergeDistance = mergeParams.mergeDistance;
        const bool filterPoints = (mergeDistance > 0.0f && std::isfinite(mergeDistance));

        PointData payload{};
        payload.localToWorld = pointMatrices[0];
        payload.active = true;
        payload.domainMinimum = glm::vec3(std::numeric_limits<float>::max());
        payload.domainMaximum = glm::vec3(std::numeric_limits<float>::lowest());

        glm::mat4 referenceL2W = toMat4(payload.localToWorld);
        glm::mat4 invReferenceL2W = glm::inverse(referenceL2W);

        if (!filterPoints) {
            for (size_t i = 0; i < pointInputs.size(); ++i) {
                const PointData* input = pointInputs[i];
                glm::mat4 inputL2W = toMat4(pointMatrices[i]);
                glm::mat4 inputToReference = invReferenceL2W * inputL2W;

                for (const auto& pos : input->positions) {
                    payload.positions.push_back(inputToReference * pos);
                }

                // Preserve the existing transformed input union 
                const glm::vec3& minimum = input->domainMinimum;
                const glm::vec3& maximum = input->domainMaximum;
                for (uint32_t corner = 0; corner < 8; ++corner) {
                    const glm::vec4 point(
                        (corner & 1u) ? maximum.x : minimum.x,
                        (corner & 2u) ? maximum.y : minimum.y,
                        (corner & 4u) ? maximum.z : minimum.z,
                        1.0f);
                    const glm::vec3 transformed = glm::vec3(inputToReference * point);
                    payload.domainMinimum = glm::min(payload.domainMinimum, transformed);
                    payload.domainMaximum = glm::max(payload.domainMaximum, transformed);
                }
            }
        } else {
            const float invCellSize = 1.0f / mergeDistance;
            const float mergeDistanceSq = mergeDistance * mergeDistance;

            struct CellKey {
                int64_t x, y, z;
                bool operator==(const CellKey& other) const noexcept {
                    return x == other.x && y == other.y && z == other.z;
                }
            };

            struct CellKeyHash {
                std::size_t operator()(const CellKey& k) const noexcept {
                    std::size_t h1 = std::hash<int64_t>{}(k.x);
                    std::size_t h2 = std::hash<int64_t>{}(k.y);
                    std::size_t h3 = std::hash<int64_t>{}(k.z);
                    return h1 ^ (h2 * 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2))
                              ^ (h3 * 0x517cc1b727220a95ULL + (h2 << 6) + (h2 >> 2));
                }
            };

            std::unordered_map<CellKey, std::vector<uint32_t>, CellKeyHash> grid;
            std::vector<glm::vec3> acceptedWorldPositions;

            for (size_t i = 0; i < pointInputs.size(); ++i) {
                const PointData* input = pointInputs[i];
                glm::mat4 inputL2W = toMat4(pointMatrices[i]);
                glm::mat4 inputToReference = invReferenceL2W * inputL2W;

                for (const auto& pos : input->positions) {
                    const glm::vec4 world = inputL2W * pos;
                    if (!std::isfinite(world.x) || !std::isfinite(world.y) || !std::isfinite(world.z)) {
                        continue;
                    }

                    const glm::vec3 wPos(world.x, world.y, world.z);
                    const int64_t cx = static_cast<int64_t>(std::floor(wPos.x * invCellSize));
                    const int64_t cy = static_cast<int64_t>(std::floor(wPos.y * invCellSize));
                    const int64_t cz = static_cast<int64_t>(std::floor(wPos.z * invCellSize));

                    bool tooClose = false;
                    for (int64_t dz = -1; dz <= 1 && !tooClose; ++dz) {
                        for (int64_t dy = -1; dy <= 1 && !tooClose; ++dy) {
                            for (int64_t dx = -1; dx <= 1 && !tooClose; ++dx) {
                                auto it = grid.find({cx + dx, cy + dy, cz + dz});
                                if (it != grid.end()) {
                                    for (uint32_t acceptedIdx : it->second) {
                                        const glm::vec3& existingWorld = acceptedWorldPositions[acceptedIdx];
                                        const float ddx = wPos.x - existingWorld.x;
                                        const float ddy = wPos.y - existingWorld.y;
                                        const float ddz = wPos.z - existingWorld.z;
                                        if (ddx * ddx + ddy * ddy + ddz * ddz < mergeDistanceSq) {
                                            tooClose = true;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (!tooClose) {
                        const glm::vec4 transformed = invReferenceL2W * world;
                        const uint32_t newIdx = static_cast<uint32_t>(payload.positions.size());
                        payload.positions.push_back(transformed);
                        acceptedWorldPositions.push_back(wPos);
                        grid[{cx, cy, cz}].push_back(newIdx);
                    }
                }

                const glm::vec3& minimum = input->domainMinimum;
                const glm::vec3& maximum = input->domainMaximum;
                for (uint32_t corner = 0; corner < 8; ++corner) {
                    const glm::vec4 point(
                        (corner & 1u) ? maximum.x : minimum.x,
                        (corner & 2u) ? maximum.y : minimum.y,
                        (corner & 4u) ? maximum.z : minimum.z,
                        1.0f);
                    const glm::vec3 transformed = glm::vec3(inputToReference * point);
                    payload.domainMinimum = glm::min(payload.domainMinimum, transformed);
                    payload.domainMaximum = glm::max(payload.domainMaximum, transformed);
                }
            }
        }

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, eval.node.outputs[0].id);
        outputValue.dataType = payloadtypes::Points;
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, eval.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);

    } else if (dataType == payloadtypes::Geometry || dataType == payloadtypes::Remesh) {
        std::vector<const GeometryData*> geometryInputs;
        std::vector<std::array<float, 16>> geometryMatrices;
        for (const NodeDataBlock* block : inputs) {
            if (block &&
                (block->dataType == payloadtypes::Geometry || block->dataType == payloadtypes::Remesh) &&
                block->payloadHandle.key != 0) {
                const GeometryData* geometryData = payloadRegistry->resolveGeometry(block->payloadHandle);
                std::array<float, 16> localToWorld{};
                if (geometryData && !geometryData->pointPositions.empty() &&
                    payloadRegistry->resolveLocalToWorld(block->payloadHandle, localToWorld)) {
                    geometryInputs.push_back(geometryData);
                    geometryMatrices.push_back(localToWorld);
                }
            }
        }

        if (geometryInputs.empty()) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            return;
        }

        GeometryData payload{};
        payload.localToWorld = geometryMatrices[0];
        payload.baseModelPath = geometryInputs[0]->baseModelPath;

        glm::mat4 referenceL2W = toMat4(payload.localToWorld);
        glm::mat4 invReferenceL2W = glm::inverse(referenceL2W);

        std::vector<GeometryGroup> mergedGroups;
        std::vector<uint32_t> mergedTriangleGroupIds;
        uint32_t nextGroupId = 1;

        for (size_t i = 0; i < geometryInputs.size(); ++i) {
            const GeometryData* input = geometryInputs[i];
            glm::mat4 inputL2W = toMat4(geometryMatrices[i]);
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

        payload.groups = mergedGroups;
        payload.triangleGroupIds = mergedTriangleGroupIds;

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, eval.node.outputs[0].id);
        outputValue.dataType = payloadtypes::Geometry;
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, eval.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);
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
