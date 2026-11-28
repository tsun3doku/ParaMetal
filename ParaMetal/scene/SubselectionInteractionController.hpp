#pragma once

#include "SubselectionPickService.hpp"
#include "SubselectionPickState.hpp"
#include "interaction/IViewportInteraction.hpp"
#include "mesh/SubselectionEditState.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

#include <array>
#include <map>
#include <optional>
#include <vector>

class SubselectionPickService;

class SubselectionInteractionController final : public IViewportInteraction {
public:
    explicit SubselectionInteractionController(SubselectionPickService* pickService = nullptr);

    void setPickService(SubselectionPickService* service);
    void begin(SubselectionEditState state);
    void end();
    bool isActive() const { return active; }

    const SubselectionEditState& editState() const { return state; }
    const MeshSelection& authoredSelection() const { return authored; }
    const MeshSelection& displaySelection() const { return displayed; }
    const std::vector<uint64_t>& selectedElements() const { return authored.elements; }
    const std::vector<uint64_t>& displayElements() const { return displayed.elements; }
    uint64_t selectionRevision() const { return revision; }
    uint64_t displayRevision() const { return displayRevisionValue; }
    SubselectionPickState pickState() const;

    void tick();
    void setViewProjection(const glm::mat4& viewProjection, const glm::vec2& viewportSize);
    void setModelMatrix(const glm::mat4& modelMatrix);
    void cancelEdit();
    void clearSelection();
    void changeElementType(MeshElementType type);
    bool setSelection(const MeshSelection& value);
    bool setDisplaySelection(const MeshSelection& value);

    bool handlePointerPress(
        float mouseX,
        float mouseY,
        int button = 1,
        bool shiftPressed = false) override;
    bool handlePointerMove(float mouseX, float mouseY) override;
    bool handlePointerRelease(float mouseX, float mouseY, int button = 1) override;
    bool hasPointerCapture() const override { return pointerCaptured; }

private:
    void applyPickedElement(uint64_t element, bool additive);
    void clearSelectionFromPickMiss();
    void cancelOutstandingPicks();

    SubselectionPickService* pickService = nullptr;
    SubselectionEditState state{};
    MeshSelection authored{};
    MeshSelection displayed{};
    std::array<std::vector<uint64_t>, 3> inactiveElements;
    std::map<uint64_t, SubselectionPickResult> completedResults;
    uint64_t revision = 0;
    uint64_t displayRevisionValue = 0;
    uint64_t sessionId = 0;
    uint64_t nextRequestId = 1;
    uint64_t nextApplyRequestId = 1;
    uint64_t cancellationBoundary = 0;
    bool active = false;
    bool pointerCaptured = false;
};
