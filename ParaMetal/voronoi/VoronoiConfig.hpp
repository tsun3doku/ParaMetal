#pragma once

#include <cstdint>

namespace voronoi {

inline constexpr uint32_t DefaultLloydIterations = 1;
inline constexpr float DefaultLloydVolumeThresholdFraction = 1e-4f;
inline constexpr float DefaultLloydDamping = 0.75f;
inline constexpr float DefaultContactRange = 0.002f;

} // namespace voronoi
