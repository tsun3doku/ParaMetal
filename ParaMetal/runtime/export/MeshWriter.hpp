#pragma once

#include "ExportTypes.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

class MeshWriter {
public:
    static ExportResult writeOBJ(
        const std::string& filePath,
        const std::vector<float>& positions,
        const std::vector<uint32_t>& triangleIndices,
        const std::vector<float>& normals = {},
        const std::array<float, 16>* localToWorld = nullptr);

    static ExportResult writePLY(
        const std::string& filePath,
        const std::vector<float>& positions,
        const std::vector<uint32_t>& triangleIndices,
        const std::array<float, 16>* localToWorld = nullptr);

    static ExportResult writeSTL(
        const std::string& filePath,
        const std::vector<float>& positions,
        const std::vector<uint32_t>& triangleIndices,
        const std::array<float, 16>* localToWorld = nullptr);
};
