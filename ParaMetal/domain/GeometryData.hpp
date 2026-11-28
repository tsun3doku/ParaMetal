#pragma once

#include "mesh/MeshElementType.hpp"
#include "nodegraph/NodeGraphCoreTypes.hpp"
#include "nodegraph/NodeGraphTypes.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

//                                                      [ Invariant:
//                                                        - Payloads are node graph authored data
//                                                        - They may contain authored values and NodeDataHandle values
//                                                        - They must not contain runtime objects/ids, scene objects,
//                                                          backend/controller objects or GPU resources
//                                                        - This header must not be included in any backend ]

struct MeshGroup {
    std::string key;
    std::string name;
    SurfaceBinding binding = SurfaceBinding::Source;
    MeshElementType elementType = MeshElementType::Face;
    std::vector<uint64_t> elements;
    uint32_t colorRGBA = 0xffffffffu;
    std::string source;
};

struct GeometryData {
    std::string baseModelPath;
    std::array<float, 16> localToWorld{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    std::vector<float> vertexPositions;
    std::vector<uint32_t> triangleIndices;
    std::vector<float> renderPositions;
    std::vector<float> renderNormals;
    std::vector<float> renderTexcoords;
    std::vector<uint32_t> renderIndices;
    std::vector<MeshGroup> groups;

};
