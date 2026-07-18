#pragma once

#include <cmath>

namespace nodegraphlayout {

inline constexpr double gridCellSize = 20.0;

inline double snapCoordinate(double coordinate) {
    return std::round(coordinate / gridCellSize) * gridCellSize;
}

}
