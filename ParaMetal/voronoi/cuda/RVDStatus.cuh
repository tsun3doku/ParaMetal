#pragma once

#include <cstdint>

namespace voronoi {

enum class RVDStatus : uint32_t {
    Success,
    EmptyCell,
    VertexOverflow,
    PlaneOverflow,
    TriangleOverflow,
    InconsistentBoundary,
    NonFiniteGeometry,
    InvalidIndex,
    NegativeVolume,
    SegmentOverflow,
    Count
};

inline const char* rvdStatusName(RVDStatus status) {
    switch (status) {
    case RVDStatus::Success: return "Success";
    case RVDStatus::EmptyCell: return "EmptyCell";
    case RVDStatus::VertexOverflow: return "VertexOverflow";
    case RVDStatus::PlaneOverflow: return "PlaneOverflow";
    case RVDStatus::TriangleOverflow: return "TriangleOverflow";
    case RVDStatus::InconsistentBoundary: return "InconsistentBoundary";
    case RVDStatus::NonFiniteGeometry: return "NonFiniteGeometry";
    case RVDStatus::InvalidIndex: return "InvalidIndex";
    case RVDStatus::NegativeVolume: return "NegativeVolume";
    case RVDStatus::SegmentOverflow: return "SegmentOverflow";
    default: return "Unknown";
    }
}

} // namespace voronoi
