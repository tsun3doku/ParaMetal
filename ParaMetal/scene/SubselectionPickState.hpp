#pragma once

#include "mesh/MeshElementType.hpp"

#include <glm/mat4x4.hpp>

#include <cstdint>

struct SubselectionPickState {
    bool active = false;
    SurfaceBinding binding = SurfaceBinding::Source;
    MeshElementType elementType = MeshElementType::Face;
    uint64_t displaySocketKey = 0;
    uint64_t pickGeometryIdentity = 0;
    uint32_t runtimeModelId = 0;
    uint32_t primitiveBase = 0;
    glm::mat4 modelMatrix{1.0f};

    bool isValid() const {
        return active && runtimeModelId != 0;
    }
};
