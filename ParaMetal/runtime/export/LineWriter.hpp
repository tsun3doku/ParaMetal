#pragma once

#include "ExportTypes.hpp"
#include "mesh/IsocontourBuilder.hpp"

#include <string>
#include <vector>

class LineWriter {
public:
    static ExportResult writeOBJ(
        const std::string& filePath,
        const std::vector<IsocontourPolyline>& polylines);
};
