#pragma once

#include "ExportTypes.hpp"
#include "mesh/Polyline.hpp"

#include <string>
#include <vector>

class LineWriter {
public:
    static ExportResult writeOBJ(
        const std::string& filePath,
        const std::vector<mesh::Polyline>& polylines);
};
