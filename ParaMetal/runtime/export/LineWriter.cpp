#define NOMINMAX
#include "LineWriter.hpp"

#include <Windows.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <vector>

ExportResult LineWriter::writeOBJ(
    const std::string& filePath,
    const std::vector<mesh::Polyline>& polylines) {
    const std::filesystem::path destination(filePath);
    const std::filesystem::path directory = destination.has_parent_path()
        ? destination.parent_path()
        : std::filesystem::current_path();

    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) {
        return {false, "Failed to create destination directory: " + ec.message(), 0};
    }

    wchar_t temporaryName[MAX_PATH]{};
    const std::wstring directoryWide = directory.wstring();
    if (!GetTempFileNameW(directoryWide.c_str(), L"pml", 0, temporaryName)) {
        return {false, "Failed to create temporary OBJ file", 0};
    }
    const std::filesystem::path temporaryPath(temporaryName);

    size_t writtenPolylineCount = 0;
    bool writeSucceeded = false;
    {
        std::ofstream out(temporaryPath, std::ios::out | std::ios::trunc);
        if (out.is_open()) {
            out.imbue(std::locale::classic());
            out << "# Wavefront OBJ lines exported from ParaMetal\n";
            out << std::fixed << std::setprecision(9);
            uint64_t nextVertexIndex = 1;
            for (const mesh::Polyline& polyline : polylines) {
                if (polyline.points.size() < 2) continue;

                std::vector<uint64_t> indices;
                indices.reserve(polyline.points.size());
                for (const glm::vec3& point : polyline.points) {
                    out << "v " << point.x << " " << point.y << " " << point.z << '\n';
                    indices.push_back(nextVertexIndex++);
                }

                out << "l";
                for (uint64_t index : indices) out << " " << index;
                if (polyline.closed) out << " " << indices.front();
                out << '\n';
                ++writtenPolylineCount;
            }
            out.flush();
            writeSucceeded = static_cast<bool>(out);
            out.close();
            writeSucceeded = writeSucceeded && !out.fail();
        }
    }

    if (!writeSucceeded) {
        std::filesystem::remove(temporaryPath, ec);
        return {false, "Failed while writing temporary OBJ file", 0};
    }

    if (!MoveFileExW(
            temporaryPath.c_str(), destination.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const DWORD error = GetLastError();
        std::filesystem::remove(temporaryPath, ec);
        return {false, "Failed to replace destination OBJ file (Windows error " +
            std::to_string(error) + ")", 0};
    }

    return {true, "Successfully exported " + std::to_string(writtenPolylineCount) +
        " lines to OBJ", writtenPolylineCount};
}
