#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <string>
#include <vector>

#include "util/GeometryUtils.hpp"

namespace cfd {

struct CFDSliceGeometrySource {
    static constexpr uint32_t InvalidFragment = 0xffffffffu;
    static constexpr uint32_t NonFluidNode = 0xffffffffu;

    std::vector<uint32_t> faceFragmentA;
    std::vector<uint32_t> faceFragmentB;
    std::vector<glm::dvec3> faceAreaVectorA;
    std::vector<glm::dvec3> faceAreaVectorB;
    std::vector<uint32_t> faceLoopOffsets;
    std::vector<uint32_t> loopVertexOffsets;
    std::vector<glm::dvec3> loopVertices;
    std::vector<uint32_t> globalFragmentToFluidNode;
    uint32_t fragmentCount = 0;
    uint32_t fluidNodeCount = 0;
    uint64_t geometryHash = 0;

    bool isValid() const;
};

struct CFDSliceVertex {
    glm::vec3 position{0.0f};
    uint32_t fluidNode = 0;
};
static_assert(sizeof(CFDSliceVertex) == 16, "CFDSliceVertex vertex stride must be 16 bytes");

struct CFDSliceMesh {
    std::vector<CFDSliceVertex> triangles;
    std::vector<CFDSliceVertex> perimeterLines;
};

struct CFDSliceLoop {
    std::vector<glm::dvec3> points;
    std::vector<Point2D> projected;
    double signedArea = 0.0;
    int parent = -1;
    int depth = 0;
};

bool buildCFDSliceMesh(
    const CFDSliceGeometrySource& source,
    uint32_t axis,
    double coordinate,
    double domainScale,
    CFDSliceMesh& output,
    std::string* error = nullptr);

} // namespace cfd
