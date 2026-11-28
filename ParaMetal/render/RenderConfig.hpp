#pragma once

#include <cstdint>

#include <glm/vec3.hpp>

namespace renderconfig {

constexpr uint32_t MaxFramesInFlight = 2;
constexpr uint32_t MaxIntrinsicSurfaces = 255;
constexpr int ModelLoadPauseMs = 50;

inline const glm::vec3 WireframeColor{0.05f};
inline const glm::vec3 SelectedWireframeColor{0.0f, 0.19751609f, 0.6120656f};

} 
