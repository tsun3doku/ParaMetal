#include "SubselectionDisplayController.hpp"

#include "mesh/MeshElementIds.hpp"
#include "mesh/SubselectionEditState.hpp"
#include "renderers/SubselectionRenderer.hpp"

#include <unordered_set>

void SubselectionDisplayController::setRenderer(
    SubselectionRenderer* updatedRenderer) {
    renderer = updatedRenderer;
}

void SubselectionDisplayController::applySelection(
    uint64_t socketKey,
    const SubselectionEditState& state,
    const MeshSelection& selection,
    uint64_t revision) {
    syncedThisFrame = true;
    if (hasConfig &&
        activeSocketKey == socketKey &&
        activeSurfaceIdentity == state.surfaceIdentity &&
        config.runtimeModelId == state.runtimeModelId &&
        config.revision == revision &&
        config.modelMatrix == state.modelMatrix) {
        return;
    }

    Config projected{};
    projected.runtimeModelId = state.runtimeModelId;
    projected.modelMatrix = state.modelMatrix;
    projected.revision = revision;

    if (selection.elementType == MeshElementType::Face) {
        projected.selectedModelIndexOffsets.reserve(
            selection.elements.size());
        for (uint64_t element : selection.elements) {
            if (element <= 0xffffffffull / 3u) {
                projected.selectedModelIndexOffsets.push_back(
                    static_cast<uint32_t>(element * 3u));
            }
        }
    } else if (selection.elementType == MeshElementType::Vertex) {
        std::unordered_set<uint32_t> selectedVertices;
        selectedVertices.reserve(selection.elements.size());
        for (uint64_t element : selection.elements) {
            if (element < state.vertexPositions.size()) {
                selectedVertices.insert(static_cast<uint32_t>(element));
                projected.pointPositions.push_back(
                    state.vertexPositions[static_cast<size_t>(element)]);
            }
        }

        std::unordered_set<uint64_t> emittedEdges;
        for (size_t index = 0;
             index + 2 < state.triangleIndices.size();
             index += 3) {
            const uint32_t a = state.triangleIndices[index + 0];
            const uint32_t b = state.triangleIndices[index + 1];
            const uint32_t c = state.triangleIndices[index + 2];
            const uint32_t corners[3] = {a, b, c};
            for (int corner = 0; corner < 3; ++corner) {
                const uint32_t first = corners[corner];
                const uint32_t second = corners[(corner + 1) % 3];
                if (selectedVertices.find(first) == selectedVertices.end() &&
                    selectedVertices.find(second) == selectedVertices.end()) {
                    continue;
                }
                const uint64_t edgeKey = packMeshEdge(first, second);
                if (!emittedEdges.insert(edgeKey).second ||
                    first >= state.vertexPositions.size() ||
                    second >= state.vertexPositions.size()) {
                    continue;
                }
                projected.linePositions.push_back(
                    state.vertexPositions[first]);
                projected.lineFade.push_back(
                    selectedVertices.find(first) != selectedVertices.end()
                        ? 1.0f : 0.0f);
                projected.linePositions.push_back(
                    state.vertexPositions[second]);
                projected.lineFade.push_back(
                    selectedVertices.find(second) != selectedVertices.end()
                        ? 1.0f : 0.0f);
            }
        }
    } else {
        std::unordered_set<uint64_t> emittedEdges;
        for (uint64_t element : selection.elements) {
            uint32_t a = 0;
            uint32_t b = 0;
            unpackMeshEdge(element, a, b);
            if (a == b ||
                a >= state.vertexPositions.size() ||
                b >= state.vertexPositions.size()) {
                continue;
            }
            const uint64_t edgeKey = packMeshEdge(a, b);
            if (!emittedEdges.insert(edgeKey).second) {
                continue;
            }
            projected.linePositions.push_back(state.vertexPositions[a]);
            projected.lineFade.push_back(1.0f);
            projected.linePositions.push_back(state.vertexPositions[b]);
            projected.lineFade.push_back(1.0f);
        }
    }

    apply(socketKey, projected);
    if (hasConfig) {
        activeSurfaceIdentity = state.surfaceIdentity;
    }
}

void SubselectionDisplayController::apply(
    uint64_t socketKey,
    const Config& updatedConfig) {
    if (socketKey == 0 || updatedConfig.runtimeModelId == 0 ||
        (updatedConfig.linePositions.empty() &&
         updatedConfig.pointPositions.empty() &&
         updatedConfig.selectedModelIndexOffsets.empty())) {
        remove(activeSocketKey);
        return;
    }

    syncedThisFrame = true;
    if (hasConfig &&
        activeSocketKey == socketKey &&
        config.runtimeModelId == updatedConfig.runtimeModelId &&
        config.revision == updatedConfig.revision &&
        config.modelMatrix == updatedConfig.modelMatrix) {
        return;
    }

    if (renderer) {
        renderer->apply(socketKey, updatedConfig);
    }
    activeSocketKey = socketKey;
    config = updatedConfig;
    hasConfig = true;
}

void SubselectionDisplayController::remove(uint64_t socketKey) {
    if (socketKey == 0 || socketKey != activeSocketKey) {
        return;
    }
    if (renderer) {
        renderer->remove(socketKey);
    }
    activeSocketKey = 0;
    activeSurfaceIdentity = 0;
    config = {};
    hasConfig = false;
    syncedThisFrame = false;
}

void SubselectionDisplayController::finalizeSync() {
    if (!syncedThisFrame) {
        remove(activeSocketKey);
    }
    syncedThisFrame = false;
}
