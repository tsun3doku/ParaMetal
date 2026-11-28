#include "SubselectionInteractionController.hpp"

#include "SubselectionPickService.hpp"
#include "mesh/MeshElementIds.hpp"

#include <algorithm>
#include <utility>

SubselectionInteractionController::SubselectionInteractionController(
    SubselectionPickService* service)
    : pickService(service) {
}

void SubselectionInteractionController::setPickService(SubselectionPickService* service) {
    pickService = service;
}

void SubselectionInteractionController::begin(SubselectionEditState editState) {
    if (pickService) {
        pickService->cancelThrough(sessionId, UINT64_MAX);
    }
    state = std::move(editState);
    authored = {};
    displayed = {};
    for (auto& elements : inactiveElements) {
        elements.clear();
    }
    ++revision;
    ++displayRevisionValue;
    ++sessionId;
    nextRequestId = 1;
    nextApplyRequestId = 1;
    cancellationBoundary = 0;
    completedResults.clear();
    active = true;
    pointerCaptured = false;
}

void SubselectionInteractionController::end() {
    cancelOutstandingPicks();
    active = false;
    pointerCaptured = false;
    state = {};
    authored = {};
    displayed = {};
    for (auto& elements : inactiveElements) {
        elements.clear();
    }
    ++revision;
    ++displayRevisionValue;
    ++sessionId;
}

SubselectionPickState SubselectionInteractionController::pickState() const {
    SubselectionPickState pick{};
    pick.active = active;
    pick.binding = state.binding;
    pick.elementType = authored.elementType;
    pick.displaySocketKey = state.displaySocketKey;
    pick.pickGeometryIdentity = state.pickGeometryIdentity;
    pick.runtimeModelId = state.runtimeModelId;
    pick.modelMatrix = state.modelMatrix;
    return pick;
}

void SubselectionInteractionController::tick() {
    if (!active || !pickService) {
        return;
    }

    while (const std::optional<SubselectionPickResult> result = pickService->takeResult()) {
        if (result->sessionId != sessionId ||
            result->requestId <= cancellationBoundary) {
            continue;
        }
        completedResults.emplace(result->requestId, *result);
    }

    while (!completedResults.empty() && completedResults.begin()->first == nextApplyRequestId) {
        const SubselectionPickResult result = completedResults.begin()->second;
        completedResults.erase(completedResults.begin());
        ++nextApplyRequestId;

        if (result.cancelled) {
            continue;
        }
        if (!result.hit) {
            if (!result.additive) {
                clearSelectionFromPickMiss();
            }
            continue;
        }
        uint64_t element = result.elementId;
        if (result.elementType == MeshElementType::Edge &&
            state.binding == SurfaceBinding::Source) {
            element = packMeshEdge(result.endpoints.x, result.endpoints.y);
        }
        applyPickedElement(element, result.additive);
    }
}

void SubselectionInteractionController::cancelEdit() {
    if (!active) {
        return;
    }
    cancelOutstandingPicks();
    authored.elements.clear();
    displayed.elements.clear();
    ++revision;
    ++displayRevisionValue;
}

void SubselectionInteractionController::changeElementType(
    MeshElementType type) {
    if (!active || type == authored.elementType ||
        type > MeshElementType::Face) {
        return;
    }

    cancelOutstandingPicks();

    const auto oldIndex = static_cast<size_t>(authored.elementType);
    const auto newIndex = static_cast<size_t>(type);

    inactiveElements[oldIndex] = std::move(authored.elements);

    authored = {type, std::move(inactiveElements[newIndex])};

    displayed = authored;

    ++revision;
    ++displayRevisionValue;
}

bool SubselectionInteractionController::setSelection(const MeshSelection& value) {
    if (!active) {
        return false;
    }

    cancelOutstandingPicks();
    authored = value;
    std::sort(authored.elements.begin(), authored.elements.end());
    authored.elements.erase(std::unique(authored.elements.begin(), authored.elements.end()), authored.elements.end());
    ++revision;
    return true;
}

bool SubselectionInteractionController::setDisplaySelection(
    const MeshSelection& value) {
    if (!active || value.elementType != authored.elementType) {
        return false;
    }

    MeshSelection sorted = value;
    std::sort(sorted.elements.begin(), sorted.elements.end());
    sorted.elements.erase(std::unique(sorted.elements.begin(), sorted.elements.end()), sorted.elements.end());
    if (displayed.elements == sorted.elements && displayed.elementType == sorted.elementType) {
        return true;
    }
    displayed = std::move(sorted);
    ++displayRevisionValue;
    return true;
}

void SubselectionInteractionController::clearSelection() {
    if (!active) {
        return;
    }
    cancelOutstandingPicks();
    if (authored.elements.empty() && displayed.elements.empty()) {
        return;
    }
    authored.elements.clear();
    displayed.elements.clear();
    ++revision;
    ++displayRevisionValue;
}

void SubselectionInteractionController::clearSelectionFromPickMiss() {
    if (!active || authored.elements.empty()) {
        return;
    }
    authored.elements.clear();
    displayed.elements.clear();
    ++revision;
    ++displayRevisionValue;
}

void SubselectionInteractionController::cancelOutstandingPicks() {
    cancellationBoundary = nextRequestId - 1;
    nextApplyRequestId = nextRequestId;
    completedResults.clear();
    if (pickService) {
        pickService->cancelThrough(sessionId, cancellationBoundary);
    }
}

bool SubselectionInteractionController::handlePointerPress(
    float mouseX,
    float mouseY,
    int button,
    bool shiftPressed) {
    if (!active) {
        return false;
    }
    if (button == 2) {
        clearSelection();
        return true;
    }
    if (button != 1) {
        return false;
    }
    pointerCaptured = true;
    if (mouseX < 0.0f || mouseY < 0.0f) {
        if (!shiftPressed) {
            clearSelectionFromPickMiss();
        }
        return true;
    }
    if (pickService) {
        SubselectionPickRequest request{};
        request.x = static_cast<int>(mouseX);
        request.y = static_cast<int>(mouseY);
        request.sessionId = sessionId;
        request.requestId = nextRequestId++;
        request.binding = state.binding;
        request.displaySocketKey = state.displaySocketKey;
        request.pickGeometryIdentity = state.pickGeometryIdentity;
        request.runtimeModelId = state.runtimeModelId;
        request.elementType = authored.elementType;
        request.viewport = glm::uvec2(state.viewportSize);
        request.viewProjection = state.viewProjection;
        request.modelMatrix = state.modelMatrix;
        request.additive = shiftPressed;
        pickService->queuePickRequest(request);
    }
    return true;
}

bool SubselectionInteractionController::handlePointerMove(float mouseX, float mouseY) {
    (void)mouseX;
    (void)mouseY;
    return pointerCaptured;
}

bool SubselectionInteractionController::handlePointerRelease(
    float mouseX,
    float mouseY,
    int button) {
    (void)mouseX;
    (void)mouseY;
    (void)button;
    if (!active) {
        return false;
    }
    pointerCaptured = false;
    return true;
}

void SubselectionInteractionController::setViewProjection(
    const glm::mat4& viewProjection,
    const glm::vec2& viewportSize) {
    state.viewProjection = viewProjection;
    state.viewportSize = viewportSize;
}

void SubselectionInteractionController::setModelMatrix(const glm::mat4& modelMatrix) {
    state.modelMatrix = modelMatrix;
}

void SubselectionInteractionController::applyPickedElement(
    uint64_t element,
    bool additive) {
    if (state.binding == SurfaceBinding::Intrinsic) {
        if (element > UINT32_MAX) {
            return;
        }
        const uint32_t rawElement = static_cast<uint32_t>(element);
        const std::vector<uint32_t>* ids = nullptr;
        switch (authored.elementType) {
        case MeshElementType::Vertex:
            ids = &state.surfaceVertexIds;
            break;
        case MeshElementType::Edge:
            ids = &state.surfaceEdgeIds;
            break;
        case MeshElementType::Face:
            ids = &state.surfaceFaceIds;
            break;
        default:
            return;
        }
        if (!std::binary_search(
                ids->begin(), ids->end(), rawElement)) {
            return;
        }
    } else if (authored.elementType == MeshElementType::Face) {
        if (element >= state.faceCount) {
            return;
        }
    } else if (authored.elementType == MeshElementType::Vertex) {
        if (element >= state.vertexPositions.size()) {
            return;
        }
    }

    const auto it = std::lower_bound(authored.elements.begin(), authored.elements.end(), element);
    const bool exists = it != authored.elements.end() && *it == element;
    if (additive) {
        if (exists) {
            authored.elements.erase(it);
        } else {
            authored.elements.insert(it, element);
        }
        ++revision;
        return;
    }

    if (authored.elements.size() == 1 && authored.elements.front() == element) {
        return;
    }
    authored.elements.assign(1, element);
    ++revision;
}
