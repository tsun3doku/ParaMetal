#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace heat {

// Engineering contour levels are multiples of authored spacing, anchored at 0 C.
inline bool makeIsocontourLevels(
    const std::vector<float>& temperaturesC,
    float spacingC,
    std::vector<float>& outLevels,
    std::string& outError) {
    outLevels.clear();
    outError.clear();
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (float temperature : temperaturesC) {
        if (!std::isfinite(temperature) || temperature < -1.0e20f) {
            outError = "Heat surface contains invalid temperatures";
            return false;
        }
        minimum = std::min(minimum, temperature);
        maximum = std::max(maximum, temperature);
    }
    if (temperaturesC.empty() || !std::isfinite(spacingC) || spacingC <= 0.0f) {
        outError = "Cannot determine isocontour temperature range";
        return false;
    }
    const double spacing = spacingC;
    const double first = std::ceil(static_cast<double>(minimum) / spacing);
    const double last = std::floor(static_cast<double>(maximum) / spacing);
    if (!std::isfinite(first) || !std::isfinite(last) || first > last) return true;
    if (last - first > 100000.0) {
        outError = "Isocontour spacing produces too many levels";
        return false;
    }
    outLevels.reserve(static_cast<std::size_t>(last - first + 1.0));
    for (double multiple = first; multiple <= last; multiple += 1.0)
        outLevels.push_back(static_cast<float>(multiple * spacing));
    return true;
}

} // namespace heat
