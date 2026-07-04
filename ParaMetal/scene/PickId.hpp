#pragma once

#include <cstdint>

enum class PickedGizmoMode {
    None = 0,
    Translate = 1,
    Rotate = 2
};

namespace pickid {

inline constexpr uint32_t None = 0u;
inline constexpr uint32_t TypeShift = 24u;
inline constexpr uint32_t PayloadMask = 0x00ffffffu;
inline constexpr uint32_t ModelType = 1u;
inline constexpr uint32_t GizmoType = 2u;

inline uint32_t encodeModel(uint32_t runtimeModelId) {
    return (ModelType << TypeShift) | (runtimeModelId & PayloadMask);
}

inline uint32_t encodeGizmo(PickedGizmoMode mode, uint32_t axis) {
    const uint32_t modeValue = static_cast<uint32_t>(mode) & 0xffu;
    const uint32_t payload = ((modeValue & 0xffu) << 8u) | (axis & 0xffu);
    return (GizmoType << TypeShift) | payload;
}

inline uint32_t typeOf(uint32_t pickId) {
    return pickId >> TypeShift;
}

inline uint32_t payloadOf(uint32_t pickId) {
    return pickId & PayloadMask;
}

inline PickedGizmoMode gizmoModeOf(uint32_t pickId) {
    return static_cast<PickedGizmoMode>((payloadOf(pickId) >> 8u) & 0xffu);
}

inline uint32_t gizmoAxisOf(uint32_t pickId) {
    return payloadOf(pickId) & 0xffu;
}

}
