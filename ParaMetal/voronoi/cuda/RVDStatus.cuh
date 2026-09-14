#pragma once

#include <cstdint>

namespace voronoi {

enum class RVDStatus : uint32_t {
    Success,
    EmptyCell,
    VertexOverflow,
    PlaneOverflow,
    TriangleOverflow,
    NeedsRobustPredicate,
    InconsistentBoundary,
    NonFiniteGeometry,
    InvalidIndex,
    NegativeVolume,
    Count
};

inline const char* rvdStatusName(RVDStatus status) {
    switch (status) {
    case RVDStatus::Success: return "Success";
    case RVDStatus::EmptyCell: return "EmptyCell";
    case RVDStatus::VertexOverflow: return "VertexOverflow";
    case RVDStatus::PlaneOverflow: return "PlaneOverflow";
    case RVDStatus::TriangleOverflow: return "TriangleOverflow";
    case RVDStatus::NeedsRobustPredicate: return "NeedsRobustPredicate";
    case RVDStatus::InconsistentBoundary: return "InconsistentBoundary";
    case RVDStatus::NonFiniteGeometry: return "NonFiniteGeometry";
    case RVDStatus::InvalidIndex: return "InvalidIndex";
    case RVDStatus::NegativeVolume: return "NegativeVolume";
    default: return "Unknown";
    }
}

} // namespace voronoi
