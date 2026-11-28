#include "RemeshDisplayController.hpp"

#include "mesh/SubselectionEditState.hpp"
#include "renderers/IntrinsicRenderer.hpp"

#include <algorithm>
#include <unordered_set>
#include <vector>

void RemeshDisplayController::setIntrinsicRenderer(IntrinsicRenderer* updatedIntrinsicRenderer) {
    intrinsicRenderer = updatedIntrinsicRenderer;
}

bool RemeshDisplayController::apply(uint64_t socketKey, const Config& config) {
    if (socketKey == 0) {
        return false;
    }

    if (!config.isValid()) {
        remove(socketKey);
        return false;
    }

    if (!intrinsicRenderer || !intrinsicRenderer->apply(socketKey, config)) {
        return false;
    }

    configsBySocket[socketKey] = config;
    return true;
}

void RemeshDisplayController::remove(uint64_t socketKey) {
    if (socketKey == 0) {
        return;
    }

    removeSelection(socketKey);

    if (intrinsicRenderer) {
        intrinsicRenderer->remove(socketKey);
    }

    configsBySocket.erase(socketKey);
}

void RemeshDisplayController::applySelection(
    uint64_t socketKey,
    const SubselectionEditState& state,
    const MeshSelection& selection,
    uint64_t revision) {
    if (socketKey == 0 ||
        state.surfaceIdentity == 0 ||
        selection.elements.empty()) {
        removeSelection(socketKey);
        return;
    }
    if (selectionSocketKey == socketKey &&
        selectionSurfaceIdentity == state.surfaceIdentity &&
        selectionRevision == revision &&
        selectionElementType == selection.elementType) {
        return;
    }

    SelectionConfig config{};
    config.elementType = selection.elementType;
    config.surfaceIdentity = state.surfaceIdentity;
    config.revision = revision;

    std::unordered_set<uint64_t> selected(
        selection.elements.begin(), selection.elements.end());

    switch (selection.elementType) {
    case MeshElementType::Face: {
        if (state.surfaceFaceIds.empty()) {
            removeSelection(socketKey);
            return;
        }
        config.elementMask.assign(state.surfaceFaceIds.size(), 0u);
        for (size_t i = 0; i < state.surfaceFaceIds.size(); ++i) {
            const bool isSelected =
                selected.find(state.surfaceFaceIds[i]) != selected.end();
            config.elementMask[i] = isSelected ? 1u : 0u;
            config.selectedCount += isSelected ? 1u : 0u;
        }
        break;
    }
    case MeshElementType::Vertex: {
        if (state.surfaceVertexIds.empty()) {
            removeSelection(socketKey);
            return;
        }
        config.elementMask.assign(state.surfaceVertexIds.size(), 0u);
        for (size_t i = 0; i < state.surfaceVertexIds.size(); ++i) {
            const bool isSelected =
                selected.find(state.surfaceVertexIds[i]) != selected.end();
            config.elementMask[i] = isSelected ? 1u : 0u;
            config.selectedCount += isSelected ? 1u : 0u;
        }
        break;
    }
    case MeshElementType::Edge: {
        if (state.surfaceEdgeIds.empty()) {
            removeSelection(socketKey);
            return;
        }
        const size_t maskSize =
            static_cast<size_t>(state.surfaceEdgeIds.back()) + 1u;
        config.elementMask.assign(maskSize, 0u);
        for (uint32_t edgeId : state.surfaceEdgeIds) {
            const bool isSelected =
                selected.find(edgeId) != selected.end();
            config.elementMask[edgeId] = isSelected ? 1u : 0u;
            config.selectedCount += isSelected ? 1u : 0u;
        }
        break;
    }
    default:
        removeSelection(socketKey);
        return;
    }

    if (config.selectedCount == 0) {
        removeSelection(socketKey);
        return;
    }

    applyElementSelection(socketKey, config);
    selectionSocketKey = socketKey;
    selectionSurfaceIdentity = state.surfaceIdentity;
    selectionRevision = revision;
    selectionElementType = selection.elementType;
}

void RemeshDisplayController::applyElementSelection(
    uint64_t socketKey,
    const SelectionConfig& config) {
    if (intrinsicRenderer && socketKey != 0 &&
        config.surfaceIdentity != 0 &&
        !config.elementMask.empty()) {
        intrinsicRenderer->applySelection(socketKey, config);
    }
}

void RemeshDisplayController::removeSelection(uint64_t socketKey) {
    if (intrinsicRenderer && socketKey != 0) {
        intrinsicRenderer->removeSelection(socketKey);
    }
    if (selectionSocketKey == socketKey) {
        selectionSocketKey = 0;
        selectionSurfaceIdentity = 0;
        selectionRevision = UINT64_MAX;
        selectionElementType = MeshElementType::Face;
    }
}

bool RemeshDisplayController::surfaceMatches(
    uint64_t socketKey,
    uint64_t surfaceIdentity) const {
    const auto it = configsBySocket.find(socketKey);
    return it != configsBySocket.end() &&
        it->second.isValid() &&
        it->second.surfaceIdentity == surfaceIdentity;
}

bool RemeshDisplayController::findByRuntimeModelId(
    uint32_t runtimeModelId,
    uint64_t& outSocketKey,
    uint64_t& outSurfaceIdentity) const {
    outSocketKey = 0;
    outSurfaceIdentity = 0;
    if (runtimeModelId == 0) {
        return false;
    }
    for (const auto& [socketKey, config] : configsBySocket) {
        if (!config.isValid() || config.runtimeModelId != runtimeModelId) {
            continue;
        }
        if (outSocketKey != 0) {
            outSocketKey = 0;
            outSurfaceIdentity = 0;
            return false;
        }
        outSocketKey = socketKey;
        outSurfaceIdentity = config.surfaceIdentity;
    }
    return outSocketKey != 0;
}

std::vector<uint32_t> RemeshDisplayController::registeredRuntimeModelIds() const {
    std::vector<uint32_t> ids;
    ids.reserve(configsBySocket.size());
    for (const auto& [socketKey, config] : configsBySocket) {
        (void)socketKey;
        if (!config.isValid() || config.runtimeModelId == 0) {
            continue;
        }
        if (std::find(ids.begin(), ids.end(), config.runtimeModelId) == ids.end()) {
            ids.push_back(config.runtimeModelId);
        }
    }
    return ids;
}
