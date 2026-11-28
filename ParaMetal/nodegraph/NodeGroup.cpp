#include "NodeGroup.hpp"

#include "NodeGraphDefinitionUtils.hpp"
#include "NodeGraphParamUtils.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "NodeGroupParams.hpp"
#include "NodePayloadRegistry.hpp"
#include "domain/GeometryData.hpp"
#include "domain/GroupData.hpp"
#include "domain/RemeshData.hpp"
#include "hash/HashBuilder.hpp"
#include "mesh/MeshElementIds.hpp"
#include "mesh/MeshSelection.hpp"
#include "hash/HashNodeCache.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

static std::vector<uint64_t> combineSelection(
    const std::vector<uint64_t>& base,
    const std::vector<uint64_t>& selection,
    GroupOperation operation) {
    std::vector<uint64_t> result;
    switch (operation) {
    case GroupOperation::Replace:
        return selection;
    case GroupOperation::Union:
        std::set_union(
            base.begin(), base.end(),
            selection.begin(), selection.end(),
            std::back_inserter(result));
        return result;
    case GroupOperation::Subtract:
        std::set_difference(
            base.begin(), base.end(),
            selection.begin(), selection.end(),
            std::back_inserter(result));
        return result;
    case GroupOperation::Intersect:
        std::set_intersection(
            base.begin(), base.end(),
            selection.begin(), selection.end(),
            std::back_inserter(result));
        return result;
    }
    return result;
}

static std::string groupOutputKey(const NodeGraphNode& node) {
    return "group:" + std::to_string(node.id.value);
}

static bool buildBaseElements(
    const GeometryData& geometry,
    MeshElementType elementType,
    std::vector<uint64_t>& elements,
    std::string& error) {
    elements.clear();
    error.clear();

    if (elementType == MeshElementType::Vertex) {
        const uint64_t count = geometry.vertexPositions.size() / 3;
        elements.resize(count);
        std::iota(elements.begin(), elements.end(), uint64_t{0});
        return true;
    }

    if (elementType == MeshElementType::Face) {
        const uint64_t count = geometry.triangleIndices.size() / 3;
        elements.resize(count);
        std::iota(elements.begin(), elements.end(), uint64_t{0});
        return true;
    }

    if (elementType == MeshElementType::Edge) {
        elements.reserve(geometry.triangleIndices.size());

        for (size_t i = 0; i + 2 < geometry.triangleIndices.size(); i += 3) {
            const uint32_t a = geometry.triangleIndices[i];
            const uint32_t b = geometry.triangleIndices[i + 1];
            const uint32_t c = geometry.triangleIndices[i + 2];

            elements.push_back(packMeshEdge(a, b));
            elements.push_back(packMeshEdge(b, c));
            elements.push_back(packMeshEdge(c, a));
        }

        std::sort(elements.begin(), elements.end());
        elements.erase(
            std::unique(elements.begin(), elements.end()),
            elements.end());

        return true;
    }

    error = "Unsupported base element type.";
    return false;
}

static bool buildBaseElements(
    const RemeshData& remesh,
    MeshElementType elementType,
    std::vector<uint64_t>& elements,
    std::string& error) {
    elements.clear();
    error.clear();

    switch (elementType) {
    case MeshElementType::Vertex:
        elements.assign(
            remesh.surfaceVertexIds.begin(),
            remesh.surfaceVertexIds.end());
        break;
    case MeshElementType::Edge:
        elements.assign(
            remesh.surfaceEdgeIds.begin(),
            remesh.surfaceEdgeIds.end());
        break;
    case MeshElementType::Face:
        elements.assign(
            remesh.surfaceFaceIds.begin(),
            remesh.surfaceFaceIds.end());
        break;
    default:
        error = "Unsupported intrinsic selection type.";
        return false;
    }

    std::sort(elements.begin(), elements.end());
    return true;
}

static bool groupSourceSocketKey(
    const NodeGraphState& graph,
    const NodeGraphNode& group,
    uint64_t& outSocketKey);

static std::string groupSelectionSignature(uint64_t geometryIdentity) {
    return "geometry:" + std::to_string(geometryIdentity);
}

static std::string groupSelectionSignature(const RemeshData& remesh) {
    return "remesh:" + std::to_string(remesh.surfaceGeometryHash);
}

static uint64_t groupGeometryIdentity(
    const NodePayloadRegistry& payloads,
    const NodeDataHandle& handle) {
    NodeDataHandle geometryHandle{};
    if (!payloads.resolveGeometry(handle, &geometryHandle)) {
        return 0;
    }
    return payloads.resolveHash(geometryHandle, HashDomain::Geometry);
}

static bool validateGroupSelection(
    const GeometryData& geometry,
    const MeshSelection& selection,
    std::string& error) {
    error.clear();
    const uint64_t vertexCount = geometry.vertexPositions.size() / 3;
    const uint64_t triangleCount = geometry.triangleIndices.size() / 3;

    if (!std::is_sorted(selection.elements.begin(), selection.elements.end())) {
        error = "Selection elements must be sorted.";
        return false;
    }
    if (std::adjacent_find(selection.elements.begin(), selection.elements.end()) !=
        selection.elements.end()) {
        error = "Selection elements must be unique.";
        return false;
    }

    switch (selection.elementType) {
    case MeshElementType::Vertex:
        for (uint64_t element : selection.elements) {
            if (element >= vertexCount) {
                error = "Selection references a missing vertex.";
                return false;
            }
        }
        return true;
    case MeshElementType::Face:
        for (uint64_t element : selection.elements) {
            if (element >= triangleCount) {
                error = "Selection references a missing face.";
                return false;
            }
        }
        return true;
    case MeshElementType::Edge: {
        std::unordered_set<uint64_t> edges;
        edges.reserve(geometry.triangleIndices.size());
        for (std::size_t index = 0;
             index + 2 < geometry.triangleIndices.size();
             index += 3) {
            const uint32_t a = geometry.triangleIndices[index + 0];
            const uint32_t b = geometry.triangleIndices[index + 1];
            const uint32_t c = geometry.triangleIndices[index + 2];
            edges.insert(packMeshEdge(a, b));
            edges.insert(packMeshEdge(b, c));
            edges.insert(packMeshEdge(c, a));
        }
        for (uint64_t element : selection.elements) {
            uint32_t a = 0;
            uint32_t b = 0;
            unpackMeshEdge(element, a, b);
            if (a == b || a >= vertexCount || b >= vertexCount ||
                edges.find(element) == edges.end()) {
                error = "Selection references a missing edge.";
                return false;
            }
        }
        return true;
    }
    }

    error = "Selection has an unsupported element type.";
    return false;
}

static bool validateGroupSelection(
    const RemeshData& remesh,
    const MeshSelection& selection,
    std::string& error) {
    error.clear();

    if (!std::is_sorted(
            selection.elements.begin(), selection.elements.end()) ||
        std::adjacent_find(
            selection.elements.begin(), selection.elements.end()) !=
            selection.elements.end()) {
        error = "Selection elements must be sorted and unique.";
        return false;
    }

    const std::vector<uint32_t>* validIds = nullptr;
    switch (selection.elementType) {
    case MeshElementType::Face:
        if (remesh.surfaceTriangleIndices.size() % 3 != 0 ||
            remesh.surfaceFaceIds.size() !=
                remesh.surfaceTriangleIndices.size() / 3) {
            error = "Intrinsic face mapping does not match surface topology.";
            return false;
        }
        validIds = &remesh.surfaceFaceIds;
        break;
    case MeshElementType::Vertex:
        if (remesh.surfaceVertexIds.size() != remesh.surfacePositions.size()) {
            error = "Intrinsic vertex mapping does not match surface positions.";
            return false;
        }
        validIds = &remesh.surfaceVertexIds;
        break;
    case MeshElementType::Edge:
        if (remesh.surfaceEdgeIds.empty() ||
            !std::is_sorted(
                remesh.surfaceEdgeIds.begin(), remesh.surfaceEdgeIds.end())) {
            error = "Intrinsic edge mapping is invalid.";
            return false;
        }
        validIds = &remesh.surfaceEdgeIds;
        break;
    default:
        error = "Selection has an unsupported element type.";
        return false;
    }

    std::unordered_set<uint32_t> validElements;
    validElements.reserve(validIds->size());
    for (uint32_t id : *validIds) {
        if (!validElements.insert(id).second) {
            error = "Intrinsic element mapping contains duplicate IDs.";
            return false;
        }
    }

    for (uint64_t element : selection.elements) {
        if (element > UINT32_MAX ||
            validElements.find(static_cast<uint32_t>(element)) ==
                validElements.end()) {
            error = "Selection references a missing intrinsic element.";
            return false;
        }
    }

    return true;
}

static bool decodeAuthoredSelection(
    const NodeGroupParams& params,
    MeshSelection& outSelection,
    std::string& error) {
    outSelection = {};
    outSelection.elementType = params.elementType;
    if (params.selectedElements.empty()) {
        return true;
    }
    if (!decodeMeshSelection(params.selectedElements, outSelection, error)) {
        return false;
    }
    if (outSelection.elementType != params.elementType) {
        error = "Group element type mismatch.";
        return false;
    }
    if (!outSelection.elements.empty() &&
        params.selectionSignature.empty()) {
        error = "Group selection is missing its source signature.";
        return false;
    }
    return true;
}

static bool loadAuthoredMeshSelection(
    const NodeGroupParams& params,
    const GeometryData& geometry,
    uint64_t geometryIdentity,
    MeshSelection& outSelection,
    std::string& error) {
    if (!decodeAuthoredSelection(params, outSelection, error)) {
        return false;
    }
    if (!outSelection.elements.empty() && geometryIdentity == 0) {
        error = "Group input geometry identity is unavailable.";
        return false;
    }
    if (!outSelection.elements.empty() &&
        params.selectionSignature !=
            groupSelectionSignature(geometryIdentity)) {
        error = "Group selection is stale for the current input geometry.";
        return false;
    }
    return validateGroupSelection(geometry, outSelection, error);
}

static bool resolveGroupPayloadSurface(
    const NodeDataBlock& block,
    const NodePayloadRegistry& payloads,
    const GeometryData*& outGeometry,
    const RemeshData*& outRemesh,
    SurfaceBinding& outBinding,
    std::string& error) {
    outGeometry = payloads.resolveGeometry(block.payloadHandle);
    if (outGeometry) {
        outRemesh = nullptr;
        outBinding = SurfaceBinding::Source;
        return true;
    }
    outRemesh = payloads.resolveRemesh(block.payloadHandle);
    if (outRemesh) {
        outBinding = SurfaceBinding::Intrinsic;
        return true;
    }
    error = "Group input has no surface topology.";
    return false;
}

static bool resolveGroupInputSurface(
    const NodeGraphState& graph,
    const NodeGraphEvaluation& evaluation,
    const NodePayloadRegistry& payloads,
    const NodeGraphNode& node,
    uint64_t& outSocketKey,
    const EvaluatedSocketValue*& outEvaluated,
    const GeometryData*& outGeometry,
    const RemeshData*& outRemesh,
    SurfaceBinding& outBinding,
    std::string& error) {
    outSocketKey = 0;
    outEvaluated = nullptr;
    outGeometry = nullptr;
    outRemesh = nullptr;
    outBinding = SurfaceBinding::Source;
    if (!groupSourceSocketKey(graph, node, outSocketKey)) {
        error = "Group must be connected to a mesh.";
        return false;
    }
    outEvaluated = evaluation.outputFor(outSocketKey);
    if (!outEvaluated ||
        outEvaluated->status != EvaluatedSocketStatus::Value) {
        error = "Group input is not evaluated.";
        return false;
    }
    return resolveGroupPayloadSurface(
        outEvaluated->data,
        payloads,
        outGeometry,
        outRemesh,
        outBinding,
        error);
}

static bool loadAuthoredIntrinsicSelection(
    const NodeGroupParams& params,
    const RemeshData& remesh,
    MeshSelection& outSelection,
    std::string& error) {
    if (!decodeAuthoredSelection(params, outSelection, error)) {
        return false;
    }
    if (!outSelection.elements.empty() &&
        params.selectionSignature != groupSelectionSignature(remesh)) {
        error = "Group selection is stale for the current Remesh topology.";
        return false;
    }
    return validateGroupSelection(remesh, outSelection, error);
}

static bool groupSourceSocketKey(
    const NodeGraphState& graph,
    const NodeGraphNode& group,
    uint64_t& outSocketKey) {
    const std::size_t meshInputIndex =
        inputIndexOf(group, "Mesh");
    if (meshInputIndex >= group.inputs.size()) {
        return false;
    }
    const NodeGraphEdge* edge = graph.edges.incomingEdge(
        group.id, group.inputs[meshInputIndex].id);
    if (!edge) {
        return false;
    }
    outSocketKey = NodeSocketKey(edge->fromNode, edge->fromSocket).value;
    return outSocketKey != 0;
}

static const char* groupElementTypeName(MeshElementType elementType) {
    switch (elementType) {
    case MeshElementType::Vertex:
        return "Vertex";
    case MeshElementType::Edge:
        return "Edge";
    case MeshElementType::Face:
        return "Face";
    }
    return "Face";
}

static NodeTypeDefinition buildGroupNode() {
    return {
        nodegraphtypes::Group,
        "Group",
        NodeGraphNodeCategory::Meshing,
        {
            makePolymorphicInputSocket(
                "Mesh",
                {NodeGraphValueType::Mesh, NodeGraphValueType::Remesh}),
            makePolymorphicOutputSocket(
                "Mesh",
                {NodeGraphValueType::Mesh, NodeGraphValueType::Remesh},
                payloadtypes::Group,
                true),
        },
        {
            makeEnumParamDefinition(
                nodegraphparams::group::ElementType,
                "Element Type",
                "Face",
                {"Vertex", "Edge", "Face"}),
            {nodegraphparams::group::GroupName, "Group Name", NodeGraphParamType::String, 0.0, 0, false, "Group", false},
            makeEnumParamDefinition(
                nodegraphparams::group::Operation,
                "Operation",
                "Replace",
                {"Replace", "Union", "Subtract", "Intersect"}),
            {nodegraphparams::group::UseBaseGroup, "Base Group", NodeGraphParamType::Bool, 0.0, 0, false, "", false},
            {nodegraphparams::group::SelectedElements, "Selected Elements", NodeGraphParamType::String, 0.0, 0, false, "", false},
            {nodegraphparams::group::SelectionSignature, "Selection Signature", NodeGraphParamType::String, 0.0, 0, false, "", false},
        },
    };
}

NodeTypeDefinition buildGroupDefinition() {
    NodeTypeDefinition definition = buildGroupNode();
    definition.actions.push_back({
        nodegraphparams::group::ActionSelectElements,
        "Select Elements",
        NodeInteractionType::MeshSubselection,
    });
    for (NodeGraphParamDefinition& parameter : definition.parameters) {
        switch (parameter.id) {
        case nodegraphparams::group::ElementType:
        case nodegraphparams::group::SelectedElements:
        case nodegraphparams::group::SelectionSignature:
            parameter.visible = false;
            break;
        default:
            break;
        }
    }
    return definition;
}

std::unique_ptr<NodeKernel> createGroupKernel() {
    return std::make_unique<NodeGroup>();
}

const char* NodeGroup::typeId() const {
    return nodegraphtypes::Group;
}

void NodeGroup::execute(NodeKernelEval& eval) const {
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;
    const std::size_t inputSocketIndex = inputIndexOf(eval.node, "Mesh");
    const NodeDataBlock* inputBlock = (inputSocketIndex < eval.inputs.size() && !eval.inputs[inputSocketIndex].empty())
            ? eval.inputs[inputSocketIndex].front()
            : nullptr;

    NodeDataBlock& outputValue = eval.outputs[0];
    outputValue = {};
    outputValue.dataType = payloadtypes::Group;

    if (!payloadRegistry) {
        eval.error = "Group payload registry is unavailable.";
        return;
    }

    if (!inputBlock || inputBlock->payloadHandle.key == 0) {
        eval.error = "Group input has no completed Mesh or Remesh payload.";
        return;
    }

    const GeometryData* inputGeometry = nullptr;
    const RemeshData* inputRemesh = nullptr;
    SurfaceBinding binding = SurfaceBinding::Source;
    std::string surfaceError;
    if (!resolveGroupPayloadSurface(
            *inputBlock,
            *payloadRegistry,
            inputGeometry,
            inputRemesh,
            binding,
            surfaceError)) {
        eval.error = std::move(surfaceError);
        return;
    }

    const NodeGroupParams params = readNodeGroupParams(eval.node);

    MeshSelection selection{};
    std::string selectionError;
    const uint64_t geometryIdentity = inputRemesh
        ? 0
        : groupGeometryIdentity(
              *payloadRegistry, inputBlock->payloadHandle);
    const bool selectionLoaded = inputRemesh
        ? loadAuthoredIntrinsicSelection(
            params, *inputRemesh, selection, selectionError)
        : loadAuthoredMeshSelection(
            params, *inputGeometry, geometryIdentity, selection, selectionError);
    if (!selectionLoaded) {
        eval.error = std::move(selectionError);
        return;
    }

    std::vector<uint64_t> baseElements;
    if (params.useBaseGroup &&
        params.operation != GroupOperation::Replace) {
        std::string baseError;
        const bool baseBuilt = inputRemesh
            ? buildBaseElements(
                *inputRemesh, params.elementType, baseElements, baseError)
            : buildBaseElements(
                *inputGeometry, params.elementType, baseElements, baseError);
        if (!baseBuilt) {
            eval.error = std::move(baseError);
            return;
        }
    }

    const std::vector<uint64_t> resultElements =
        combineSelection(baseElements, selection.elements, params.operation);

    const std::string outputKey = groupOutputKey(eval.node);

    GroupData payload{};
    payload.sourceHandle = inputBlock->payloadHandle;
    payload.group.key = outputKey;
    payload.group.name = params.groupName.empty() ? "Group" : params.groupName;
    payload.group.binding = binding;
    payload.group.elementType = params.elementType;
    payload.group.elements = resultElements;
    payload.group.source = "node.group";

    const uint64_t payloadKey = NodeSocketKey(eval.node.id, eval.node.outputs[0].id);
    outputValue.payloadHandle = payloadRegistry->store(payloadKey, std::move(payload), eval.outputHashes);
    if (outputValue.payloadHandle.key == 0) {
        eval.error = "Payload publication backpressured.";
        return;
    }
}

void NodeGroup::inspect(
    const NodeAuthoringContext& context,
    NodeInspection& out) const {
    out = {};

    if (!context.node.outputs.empty()) {
        const uint64_t outputKey =
            NodeSocketKey(context.node.id, context.node.outputs.front().id).value;
        const EvaluatedSocketValue* evaluated =
            context.evaluation.outputFor(outputKey);
        if (evaluated) {
            if (evaluated->status == EvaluatedSocketStatus::Error) {
                out.fields.push_back({
                    "Error",
                    evaluated->error.empty()
                        ? "Group evaluation failed."
                        : evaluated->error,
                });
                return;
            }
            if (evaluated->status == EvaluatedSocketStatus::Value) {
                const GroupData* payload =
                    context.payloads.get<GroupData>(evaluated->data.payloadHandle);
                if (!payload) {
                    out.fields.push_back({
                        "Error",
                        "Group output is not a named group.",
                    });
                    return;
                }
                out.fields.push_back({"Name", payload->group.name});
                out.fields.push_back({
                    "Element Type",
                    groupElementTypeName(payload->group.elementType),
                });
                out.fields.push_back({
                    "Elements",
                    std::to_string(payload->group.elements.size()),
                });
            }
        }
    }
}

bool NodeGroup::beginInteraction(
    const NodeAuthoringContext& context,
    uint32_t actionId,
    NodeInteractionRequest& out,
    std::string& error) const {
    out = {};
    error.clear();
    if (actionId != nodegraphparams::group::ActionSelectElements) {
        error = "Unknown Group action.";
        return false;
    }

    uint64_t sourceSocketKey = 0;
    const EvaluatedSocketValue* evaluated = nullptr;
    const GeometryData* geometry = nullptr;
    const RemeshData* remesh = nullptr;
    SurfaceBinding binding = SurfaceBinding::Source;
    if (!resolveGroupInputSurface(
            context.graph,
            context.evaluation,
            context.payloads,
            context.node,
            sourceSocketKey,
            evaluated,
            geometry,
            remesh,
            binding,
            error)) {
        return false;
    }

    const NodeGroupParams params = readNodeGroupParams(context.node);
    const uint64_t geometryIdentity = remesh
        ? 0
        : groupGeometryIdentity(
              context.payloads, evaluated->data.payloadHandle);
    const bool hasAuthoredSelection = !params.selectedElements.empty();
    const bool selectionStale =
        hasAuthoredSelection &&
        !params.selectionSignature.empty() &&
        (remesh
             ? params.selectionSignature != groupSelectionSignature(*remesh)
             : geometryIdentity != 0 &&
               params.selectionSignature !=
                   groupSelectionSignature(geometryIdentity));

    MeshSelection selection{};
    if (selectionStale) {
        selection.elementType = params.elementType;
    } else {
        const bool loaded = remesh
            ? loadAuthoredIntrinsicSelection(
                params, *remesh, selection, error)
            : loadAuthoredMeshSelection(
                params, *geometry, geometryIdentity, selection, error);
        if (!loaded) {
            return false;
        }
    }

    const NodeGraphSocket* output =
        context.node.outputOfType(payloadtypes::Group);
    if (!output) {
        error = "Group output is missing.";
        return false;
    }

    uint64_t displaySocketKey =
        NodeSocketKey(context.node.id, output->id).value;
    if (remesh) {
        NodeDataHandle remeshHandle{};
        if (!context.payloads.resolveRemesh(
                evaluated->data.payloadHandle, &remeshHandle) ||
            remeshHandle.key == 0) {
            error = "Group intrinsic display surface is unavailable.";
            return false;
        }
        displaySocketKey = remeshHandle.key;
    }

    out.type = NodeInteractionType::MeshSubselection;
    out.binding = binding;
    out.sourceSocketKey = sourceSocketKey;
    out.displaySocketKey = displaySocketKey;
    out.initialValue = std::move(selection);
    return true;
}

bool NodeGroup::previewInteraction(
    const NodeAuthoringContext& context,
    const NodeInteractionSession& session,
    const NodeInteractionValue& authored,
    NodeInteractionValue& display,
    std::string& error) const {
    display = authored;
    error.clear();
    if (session.type != NodeInteractionType::MeshSubselection) {
        error = "Group received an unexpected interaction preview.";
        return false;
    }

    const MeshSelection* selection =
        std::get_if<MeshSelection>(&authored);
    if (!selection) {
        error = "Expected a mesh selection.";
        return false;
    }

    uint64_t sourceSocketKey = 0;
    const EvaluatedSocketValue* evaluated = nullptr;
    const GeometryData* geometry = nullptr;
    const RemeshData* remesh = nullptr;
    SurfaceBinding binding = SurfaceBinding::Source;
    if (!resolveGroupInputSurface(
            context.graph,
            context.evaluation,
            context.payloads,
            context.node,
            sourceSocketKey,
            evaluated,
            geometry,
            remesh,
            binding,
            error)) {
        return false;
    }
    if (sourceSocketKey != session.sourceSocketKey ||
        binding != session.binding) {
        error = "Group input changed during selection editing.";
        return false;
    }

    NodeGroupParams params = readNodeGroupParams(context.node);
    params.elementType = selection->elementType;
    const bool validSelection = remesh
        ? validateGroupSelection(*remesh, *selection, error)
        : validateGroupSelection(*geometry, *selection, error);
    if (!validSelection) {
        return false;
    }
    std::vector<uint64_t> baseElements;
    if (params.useBaseGroup &&
        params.operation != GroupOperation::Replace) {
        const bool baseBuilt = remesh
            ? buildBaseElements(
                *remesh, params.elementType, baseElements, error)
            : buildBaseElements(
                *geometry, params.elementType, baseElements, error);
        if (!baseBuilt) {
            return false;
        }
    }

    MeshSelection preview = *selection;
    preview.elements = combineSelection(
        baseElements,
        selection->elements,
        params.operation);
    display = std::move(preview);
    return true;
}

bool NodeGroup::completeInteraction(
    const NodeAuthoringContext& context,
    const NodeInteractionSession& session,
    const NodeInteractionResult& result,
    std::vector<NodeGraphParamValue>& patch,
    std::string& error) const {
    patch.clear();
    error.clear();
    if (session.type != NodeInteractionType::MeshSubselection) {
        error = "Group received an unexpected interaction result.";
        return false;
    }

    const MeshSelection* selection =
        std::get_if<MeshSelection>(&result.value);
    if (!selection) {
        error = "Expected a mesh selection result.";
        return false;
    }

    uint64_t sourceSocketKey = 0;
    const EvaluatedSocketValue* evaluated = nullptr;
    const GeometryData* geometry = nullptr;
    const RemeshData* remesh = nullptr;
    SurfaceBinding binding = SurfaceBinding::Source;
    if (!resolveGroupInputSurface(
            context.graph,
            context.evaluation,
            context.payloads,
            context.node,
            sourceSocketKey,
            evaluated,
            geometry,
            remesh,
            binding,
            error)) {
        return false;
    }
    if (sourceSocketKey != session.sourceSocketKey ||
        binding != session.binding) {
        error = "Group input changed during selection editing.";
        return false;
    }

    const bool validSelection = remesh
        ? validateGroupSelection(*remesh, *selection, error)
        : validateGroupSelection(*geometry, *selection, error);
    if (!validSelection) {
        return false;
    }

    const uint64_t geometryIdentity = remesh
        ? 0
        : groupGeometryIdentity(
              context.payloads, evaluated->data.payloadHandle);
    if (!remesh && !selection->elements.empty() && geometryIdentity == 0) {
        error = "Group input geometry identity is unavailable.";
        return false;
    }

    NodeGraphParamValue elementTypeParam{};
    elementTypeParam.id = nodegraphparams::group::ElementType;
    elementTypeParam.type = NodeGraphParamType::Enum;
    elementTypeParam.enumValue =
        groupElementTypeName(selection->elementType);
    elementTypeParam.intValue = static_cast<int64_t>(selection->elementType);

    NodeGraphParamValue selectionParam{};
    selectionParam.id = nodegraphparams::group::SelectedElements;
    selectionParam.type = NodeGraphParamType::String;
    selectionParam.stringValue = encodeMeshSelection(*selection);

    NodeGraphParamValue signatureParam{};
    signatureParam.id = nodegraphparams::group::SelectionSignature;
    signatureParam.type = NodeGraphParamType::String;
    signatureParam.stringValue = selection->elements.empty()
        ? std::string{}
        : remesh
            ? groupSelectionSignature(*remesh)
            : groupSelectionSignature(geometryIdentity);

    patch = {
        std::move(elementTypeParam),
        std::move(selectionParam),
        std::move(signatureParam),
    };
    return true;
}

HashValues NodeGroup::computeOutputHashes(const NodeKernelHash& hash) const {
    const NodeGroupParams params = readNodeGroupParams(hash.node);
    const std::string outputKey = groupOutputKey(hash.node);

    uint64_t geometryHash = HashBuilder::start();
    HashBuilder::combineString(geometryHash, nodegraphtypes::Group);
    HashNodeCache::combineSocket(geometryHash, hash, "Mesh", HashDomain::Geometry);

    uint64_t propertiesHash = HashBuilder::start();
    HashBuilder::combineString(propertiesHash, "group.properties");
    HashBuilder::combine(propertiesHash, static_cast<uint64_t>(hash.node.id.value));
    HashBuilder::combineString(propertiesHash, outputKey);
    HashBuilder::combineString(propertiesHash, params.groupName);
    HashBuilder::combine(propertiesHash, static_cast<uint64_t>(params.elementType));
    HashBuilder::combine(propertiesHash, static_cast<uint64_t>(params.operation));
    HashBuilder::combine(propertiesHash, static_cast<uint64_t>(params.useBaseGroup));
    HashBuilder::combineString(propertiesHash, params.selectedElements);
    HashBuilder::combineString(propertiesHash, params.selectionSignature);
    HashNodeCache::combineSocket(propertiesHash, hash, "Mesh", HashDomain::Properties);

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combineString(fullHash, nodegraphtypes::Group);
    HashBuilder::combine(fullHash, static_cast<uint64_t>(hash.node.id.value));
    HashBuilder::combineString(fullHash, params.groupName);
    HashBuilder::combine(fullHash, static_cast<uint64_t>(params.elementType));
    HashBuilder::combine(fullHash, static_cast<uint64_t>(params.operation));
    HashBuilder::combine(fullHash, static_cast<uint64_t>(params.useBaseGroup));
    HashBuilder::combineString(fullHash, params.selectedElements);
    HashBuilder::combineString(fullHash, params.selectionSignature);
    HashNodeCache::combineSocket(fullHash, hash, "Mesh", HashDomain::Full);

    HashValues values{};
    values.full = fullHash;
    values.geometry = geometryHash;
    values.properties = propertiesHash;
    values.simulation = fullHash;
    return values;
}
