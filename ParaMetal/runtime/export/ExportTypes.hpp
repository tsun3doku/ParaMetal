#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "nodegraph/NodeGraphCoreTypes.hpp"

enum class ExportFormat : uint8_t {
    Unknown = 0,
    Auto,
    OBJ,
    PLY,
    STL,
    XYZ,
    CSV
};

struct ExportResult {
    bool success = false;
    std::string message;
    std::size_t writtenCount = 0;
};

struct ExportProductKey {
    NodeGraphNodeId producerNodeId{};
    NodeGraphSocketId outputSocketId{};
    std::string productId;
};

struct ExportProduct {
    ExportProductKey key;
    NodeGraphNodeId labelNodeId{};
    std::string typeName;
    std::vector<ExportFormat> supportedFormats;
    ExportFormat defaultFormat = ExportFormat::Unknown;
    bool available = true;
    std::string unavailableReason;
};

namespace exportutils {

inline std::string exportFormatToString(ExportFormat format) {
    switch (format) {
    case ExportFormat::OBJ: return "OBJ";
    case ExportFormat::PLY: return "PLY";
    case ExportFormat::STL: return "STL";
    case ExportFormat::CSV: return "CSV";
    case ExportFormat::XYZ: return "XYZ";
    case ExportFormat::Auto: return "Auto";
    case ExportFormat::Unknown:
    default: return "Unknown";
    }
}

inline ExportFormat stringToExportFormat(const std::string& str) {
    if (str == "OBJ" || str == ".obj") return ExportFormat::OBJ;
    if (str == "PLY" || str == ".ply") return ExportFormat::PLY;
    if (str == "STL" || str == ".stl") return ExportFormat::STL;
    if (str == "CSV" || str == ".csv") return ExportFormat::CSV;
    if (str == "XYZ" || str == ".xyz") return ExportFormat::XYZ;
    if (str == "Auto" || str.empty()) return ExportFormat::Auto;
    return ExportFormat::Unknown;
}

inline std::string extensionOf(const std::string& filePath) {
    std::string ext = std::filesystem::path(filePath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

inline ExportFormat formatFromExtension(const std::string& filePath) {
    const std::string ext = extensionOf(filePath);
    if (ext == ".obj") return ExportFormat::OBJ;
    if (ext == ".ply") return ExportFormat::PLY;
    if (ext == ".stl") return ExportFormat::STL;
    if (ext == ".xyz") return ExportFormat::XYZ;
    if (ext == ".csv") return ExportFormat::CSV;
    return ExportFormat::Unknown;
}

} // namespace exportutils
