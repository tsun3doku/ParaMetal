#pragma once

#include <cstdint>

enum class MeshElementType : uint8_t {
    Vertex,
    Edge,
    Face
};

enum class SurfaceBinding : uint8_t {
    Source,
    Intrinsic
};
