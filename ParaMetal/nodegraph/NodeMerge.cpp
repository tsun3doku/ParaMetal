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
#include "mesh/MeshElementIds.hpp"

#include <cassert>
#include <cmath>
#include "../util/GeometryUtils.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iterator>
#include <utility>

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
            eval.error = "Merge received an invalid connected input.";
            return;
        }
        if (block->dataType == payloadtypes::Points) {
            hasPoints = true;
        } else if (block->dataType == payloadtypes::Geometry ||
                   block->dataType == payloadtypes::Group) {
            hasGeometry = true;
        } else if (block->dataType == payloadtypes::Remesh) {
            eval.error = "Merge requires concrete mesh geometry, not remeshed topology.";
            return;
        } else {
            eval.error = "Merge received an unsupported input payload.";
            return;
        }
    }

    NodeDataBlock& outputValue = eval.outputs[0];
    outputValue = {};

    if (!hasPoints && !hasGeometry) {
        return;
    }
    if (hasPoints && hasGeometry) {
        eval.error = "Merge cannot mix Points with Geometry inputs.";
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
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(payload), eval.outputHashes);
        if (outputValue.payloadHandle.key == 0) {
            eval.error = "Payload publication backpressured.";
            return;
        }
    } else if (hasGeometry) {
        std::vector<const GeometryData*> geometryInputs;
        std::vector<std::array<float, 16>> geometryMatrices;
        std::vector<std::vector<MeshGroup>> geometryGroups;
        for (const NodeDataBlock* block : inputs) {
            if (!block || block->payloadHandle.key == 0 ||
                (block->dataType != payloadtypes::Geometry &&
                 block->dataType != payloadtypes::Group)) {
                eval.error = "Merge received an invalid geometry input.";
                return;
            }
            const GeometryData* geometryData = payloadRegistry->resolveGeometry(block->payloadHandle);
            std::array<float, 16> localToWorld{};
            if (!geometryData || geometryData->vertexPositions.empty()) {
                eval.error = "Merge cannot resolve a connected geometry input.";
                return;
            }
            if (!payloadRegistry->resolveLocalToWorld(block->payloadHandle, localToWorld)) {
                eval.error = "Merge cannot resolve a connected input placement.";
                return;
            }
            std::vector<MeshGroup> groups;
            if (!payloadRegistry->resolveGroups(block->payloadHandle, groups)) {
                eval.error = "Merge cannot resolve connected input groups.";
                return;
            }
            geometryInputs.push_back(geometryData);
            geometryMatrices.push_back(localToWorld);
            geometryGroups.push_back(std::move(groups));
        }

        if (geometryInputs.empty()) {
            return;
        }

        bool hasAnyRenderData = false;
        bool hasMissingRenderData = false;
        std::size_t renderVertexCount = 0;
        std::size_t renderIndexCount = 0;
        for (const GeometryData* input : geometryInputs) {
            const bool hasRenderData =
                !input->renderPositions.empty() ||
                !input->renderNormals.empty() ||
                !input->renderTexcoords.empty() ||
                !input->renderIndices.empty();
            if (!hasRenderData) {
                hasMissingRenderData = true;
                continue;
            }
            hasAnyRenderData = true;
            const std::size_t cornerCount = input->renderPositions.size() / 3;
            if (input->renderPositions.empty() ||
                input->renderPositions.size() % 3 != 0 ||
                input->renderNormals.size() != input->renderPositions.size() ||
                input->renderTexcoords.size() != cornerCount * 2 ||
                input->renderIndices.size() != input->triangleIndices.size()) {
                eval.error = "Merge received inconsistent render mesh data.";
                return;
            }
            renderVertexCount += cornerCount;
            renderIndexCount += input->renderIndices.size();
        }
        if (hasAnyRenderData && hasMissingRenderData) {
            eval.error = "Merge cannot mix inputs with and without render meshes.";
            return;
        }
        const bool carryRenderMesh = hasAnyRenderData && !hasMissingRenderData;

        uint64_t totalVertexCount = 0;
        uint64_t totalTriangleCount = 0;
        for (const GeometryData* input : geometryInputs) {
            totalVertexCount += input->vertexPositions.size() / 3;
            totalTriangleCount += input->triangleIndices.size() / 3;
        }
        if (totalVertexCount > UINT32_MAX ||
            totalTriangleCount > UINT32_MAX ||
            renderVertexCount > UINT32_MAX ||
            renderIndexCount > UINT32_MAX) {
            eval.error = "Merge result exceeds the 32-bit geometry element range.";
            return;
        }

        GeometryData payload{};
        payload.localToWorld = geometryMatrices[0];
        payload.baseModelPath = geometryInputs[0]->baseModelPath;
        payload.vertexPositions.reserve(static_cast<std::size_t>(totalVertexCount) * 3);
        payload.triangleIndices.reserve(static_cast<std::size_t>(totalTriangleCount) * 3);

        if (carryRenderMesh) {
            payload.renderPositions.reserve(renderVertexCount * 3);
            payload.renderNormals.reserve(renderVertexCount * 3);
            payload.renderTexcoords.reserve(renderVertexCount * 2);
            payload.renderIndices.reserve(renderIndexCount);
        }

        const glm::mat4 referenceL2W = toMat4(payload.localToWorld);
        const float referenceDeterminant = glm::determinant(glm::mat3(referenceL2W));
        if (!std::isfinite(referenceDeterminant) ||
            std::abs(referenceDeterminant) <= 1e-12f) {
            eval.error = "Merge reference placement is not invertible.";
            return;
        }
        const glm::mat4 invReferenceL2W = glm::inverse(referenceL2W);

        std::vector<MeshGroup> mergedGroups;
        std::unordered_map<std::string, std::size_t> mergedGroupByKey;

        for (size_t i = 0; i < geometryInputs.size(); ++i) {
            const GeometryData* input = geometryInputs[i];
            glm::mat4 inputL2W = toMat4(geometryMatrices[i]);
            glm::mat4 inputToReference = invReferenceL2W * inputL2W;

            const uint32_t vertexOffset = static_cast<uint32_t>(payload.vertexPositions.size() / 3);
            const uint32_t triangleOffset = static_cast<uint32_t>(payload.triangleIndices.size() / 3);

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

            // Merge groups by stable key
            for (const MeshGroup& inputGroup : geometryGroups[i]) {
                MeshGroup remapped = inputGroup;
                switch (remapped.elementType) {
                case MeshElementType::Vertex:
                    for (uint64_t& element : remapped.elements) {
                        element += vertexOffset;
                    }
                    break;
                case MeshElementType::Edge:
                    for (uint64_t& element : remapped.elements) {
                        uint32_t a = 0;
                        uint32_t b = 0;
                        unpackMeshEdge(element, a, b);
                        element = packMeshEdge(a + vertexOffset, b + vertexOffset);
                    }
                    break;
                case MeshElementType::Face:
                    for (uint64_t& element : remapped.elements) {
                        element += triangleOffset;
                    }
                    break;
                }

                const auto existingIt = mergedGroupByKey.find(remapped.key);
                if (existingIt == mergedGroupByKey.end()) {
                    mergedGroupByKey.emplace(remapped.key, mergedGroups.size());
                    mergedGroups.push_back(std::move(remapped));
                    continue;
                }

                MeshGroup& existing = mergedGroups[existingIt->second];
                if (existing.elementType != remapped.elementType ||
                    existing.name != remapped.name ||
                    existing.colorRGBA != remapped.colorRGBA ||
                    existing.source != remapped.source) {
                    eval.error = "Merge group identity conflict for key '" + remapped.key + "'.";
                    return;
                }

                std::vector<uint64_t> united;
                united.reserve(existing.elements.size() + remapped.elements.size());
                std::set_union(
                    existing.elements.begin(), existing.elements.end(),
                    remapped.elements.begin(), remapped.elements.end(),
                    std::back_inserter(united));
                existing.elements = std::move(united);
            }
        }

        payload.groups = std::move(mergedGroups);
        assert(!carryRenderMesh ||
            payload.renderIndices.size() == payload.triangleIndices.size());
        assert(!carryRenderMesh ||
            payload.renderPositions.size() / 3 == payload.renderIndices.size());

        const uint64_t payloadKey = NodeSocketKey(eval.node.id, eval.node.outputs[0].id);
        outputValue.dataType = payloadtypes::Geometry;
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(payload), eval.outputHashes);
        if (outputValue.payloadHandle.key == 0) {
            eval.error = "Payload publication backpressured.";
            return;
        }
    }
}

HashValues NodeMerge::computeOutputHashes(const NodeKernelHash& hash) const {
    const MergeNodeParams params = readMergeNodeParams(hash.node);
    bool pointsInput = false;
    const std::size_t geometrySocketIndex = inputIndexOf(hash.node, "Geometry");
    if (geometrySocketIndex < hash.inputs.size()) {
        for (const NodeDataBlock* input : hash.inputs[geometrySocketIndex]) {
            if (input && input->dataType == payloadtypes::Points) {
                pointsInput = true;
                break;
            }
        }
    }

    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combineString(geometryHash, nodegraphtypes::Merge);
    if (pointsInput) {
        HashBuilder::combineFloat(geometryHash, params.mergeDistance);
    }
    HashNodeCache::combineSocketList(geometryHash, hash, "Geometry", HashDomain::Geometry);

    uint64_t propertiesHash = HashBuilder::start();
    HashBuilder::combineString(propertiesHash, "merge.properties");
    HashNodeCache::combineSocketList(propertiesHash, hash, "Geometry", HashDomain::Properties);

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combineString(fullHash, nodegraphtypes::Merge);
    if (pointsInput) {
        HashBuilder::combineFloat(fullHash, params.mergeDistance);
    }
    HashNodeCache::combineSocketList(fullHash, hash, "Geometry", HashDomain::Full);

    HashValues values{};
    values.full = fullHash;
    values.geometry = geometryHash;
    values.properties = propertiesHash;
    values.simulation = fullHash;
    return values;
}
