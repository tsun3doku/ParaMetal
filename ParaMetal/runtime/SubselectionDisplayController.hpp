#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <cstdint>
#include <vector>

class SubselectionRenderer;
struct MeshSelection;
struct SubselectionEditState;

class SubselectionDisplayController {
public:
    struct Config {
        uint32_t runtimeModelId = 0;
        glm::mat4 modelMatrix{1.0f};
        std::vector<glm::vec3> linePositions;
        std::vector<float> lineFade;
        std::vector<glm::vec3> pointPositions;
        std::vector<uint32_t> selectedModelIndexOffsets;
        uint64_t revision = 0;
    };

    void setRenderer(SubselectionRenderer* updatedRenderer);
    void applySelection(
        uint64_t socketKey,
        const SubselectionEditState& state,
        const MeshSelection& selection,
        uint64_t revision);
    void apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);
    void finalizeSync();

private:
    SubselectionRenderer* renderer = nullptr;
    uint64_t activeSocketKey = 0;
    uint64_t activeSurfaceIdentity = 0;
    Config config{};
    bool hasConfig = false;
    bool syncedThisFrame = false;
};
