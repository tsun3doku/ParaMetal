#pragma once

#include "mesh/MeshElementType.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct MeshSelection {
    MeshElementType elementType = MeshElementType::Face;
    std::vector<uint64_t> elements;
};

std::string encodeMeshSelection(const MeshSelection& selection);
bool decodeMeshSelection(const std::string& text, MeshSelection& outSelection, std::string& outError);
