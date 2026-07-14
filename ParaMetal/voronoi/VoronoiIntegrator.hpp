#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "voronoi/VoronoiGpuStructs.hpp"
#include <glm/glm.hpp>

class VoronoiIntegrator {
  public:
    VoronoiIntegrator() = default;

    void extractNeighborIndices(const std::vector<std::vector<uint32_t>> &neighborIndices,
                                const std::vector<glm::dvec3> &seedPositions, int K,
                                std::vector<glm::vec4> &outSeedPositions, std::vector<uint32_t> &outNeighborIndices);
    void extractMeshTriangles(const std::vector<glm::vec3> &positions, const std::vector<uint32_t> &indices,
                              std::vector<std::array<glm::vec4, 3>> &outMeshTriangles);
    void computeNeighbors(const std::vector<glm::dvec3> &seedPositions, int K, std::vector<glm::vec4> &outSeedPositions,
                          std::vector<uint32_t> &outNeighborIndices);
    bool buildPointCouplings(const std::vector<glm::vec4> &positions, const std::vector<uint32_t> &nodeFlags,
                              const std::vector<uint32_t> &neighborIndices, uint32_t maxNeighbors,
                              std::vector<voronoi::Node> &nodes, std::vector<voronoi::NodeCoupling> &couplings) const;
};