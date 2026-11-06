#pragma once

#include "ExportTypes.hpp"

#include <array>
#include <cstdint>
#include <glm/vec4.hpp>
#include <string>
#include <vector>

class PointWriter {
public:
    static ExportResult writeXYZ(
        const std::string& filePath,
        const std::vector<glm::vec4>& points,
        const std::array<float, 16>* localToWorld = nullptr);

    static ExportResult writeCSV(
        const std::string& filePath,
        const std::vector<glm::vec4>& points,
        const std::array<float, 16>* localToWorld = nullptr);

    static ExportResult writePLY(
        const std::string& filePath,
        const std::vector<glm::vec4>& points,
        const std::array<float, 16>* localToWorld = nullptr);

    // Flat positions overload (3 floats per point)
    static ExportResult writePoints(
        const std::string& filePath,
        ExportFormat format,
        const std::vector<float>& flatPositions,
        const std::array<float, 16>* localToWorld = nullptr);
};
