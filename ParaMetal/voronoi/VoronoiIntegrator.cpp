#include "VoronoiIntegrator.hpp"

#include <algorithm>
#include <iostream>
#include <libs/nanoflann/include/nanoflann.hpp>

namespace {
struct VoronoiDoubleAdapter {
    const std::vector<glm::dvec3> &points;
    size_t kdtree_get_point_count() const { return points.size(); }
    double kdtree_get_pt(size_t index, size_t dimension) const { return points[index][static_cast<int>(dimension)]; }
    template <class BoundingBox> bool kdtree_get_bbox(BoundingBox &) const { return false; }
};
} // namespace

void VoronoiIntegrator::extractNeighborIndices(const std::vector<std::vector<uint32_t>> &inputNeighborIndices,
                                               const std::vector<glm::dvec3> &inputSeedPositions, int K,
                                               std::vector<glm::vec4> &outSeedPositions,
                                               std::vector<uint32_t> &outNeighborIndices) {
    outNeighborIndices.clear();
    outSeedPositions.clear();

    // Flatten 2D neighbor array to 1D GPU buffer
    outNeighborIndices.reserve(inputNeighborIndices.size() * K);
    for (const auto &cellNeighbors : inputNeighborIndices) {
        outNeighborIndices.insert(outNeighborIndices.end(), cellNeighbors.begin(), cellNeighbors.end());
    }

    // Store seed positions for GPU
    for (const auto &seed : inputSeedPositions) {
        outSeedPositions.push_back(glm::vec4(seed, 1.0f)); // w=1.0 for regular cells
    }
}

void VoronoiIntegrator::computeNeighbors(const std::vector<glm::dvec3> &seedPositions, int K,
                                         std::vector<glm::vec4> &outSeedPositions,
                                         std::vector<uint32_t> &outNeighborIndices) {
    VoronoiDoubleAdapter cloud{seedPositions};

    // Build KDtree
    nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, VoronoiDoubleAdapter>,
                                        VoronoiDoubleAdapter, 3>
        index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    std::vector<std::vector<uint32_t>> neighborIndices;
    neighborIndices.reserve(seedPositions.size());

    // For each seed, find K+1 nearest using KDtree
    for (size_t i = 0; i < seedPositions.size(); i++) {
        const double query_pt[3] = {seedPositions[i].x, seedPositions[i].y, seedPositions[i].z};

        // Query K+1 neighbors
        std::vector<size_t> ret_indices(K + 1);
        std::vector<double> out_dists_sqr(K + 1);

        nanoflann::KNNResultSet<double> resultSet(K + 1);
        resultSet.init(ret_indices.data(), out_dists_sqr.data());
        index.findNeighbors(resultSet, query_pt);

        // Filter out self and take first K neighbors
        std::vector<uint32_t> kNeighbors;
        for (size_t j = 0; j < ret_indices.size() && kNeighbors.size() < K; j++) {
            if (ret_indices[j] != i) { // Skip self
                kNeighbors.push_back(static_cast<uint32_t>(ret_indices[j]));
            }
        }

        while (kNeighbors.size() < K) {
            kNeighbors.push_back(UINT32_MAX);
        }

        neighborIndices.push_back(kNeighbors);
    }

    extractNeighborIndices(neighborIndices, seedPositions, K, outSeedPositions, outNeighborIndices);
}

void VoronoiIntegrator::extractMeshTriangles(const std::vector<glm::vec3> &positions,
                                             const std::vector<uint32_t> &indices,
                                             std::vector<std::array<glm::vec4, 3>> &outMeshTriangles) {
    outMeshTriangles.clear();

    for (size_t i = 0; i < indices.size(); i += 3) {
        if (i + 2 >= indices.size() || indices[i] >= positions.size() || indices[i + 1] >= positions.size() ||
            indices[i + 2] >= positions.size()) {
            continue;
        }

        outMeshTriangles.push_back({glm::vec4(positions[indices[i]], 0.0f), glm::vec4(positions[indices[i + 1]], 0.0f),
                                    glm::vec4(positions[indices[i + 2]], 0.0f)});
    }
}

bool VoronoiIntegrator::buildPointCouplings(const std::vector<glm::vec4> &positions,
                                             const std::vector<uint32_t> &nodeFlags,
                                             const std::vector<uint32_t> &neighborIndices, uint32_t maxNeighbors,
                                             std::vector<voronoi::Node> &nodes,
                                             std::vector<voronoi::NodeCoupling> &couplings) const {
    if (positions.empty() || nodeFlags.size() != positions.size() || nodes.size() != positions.size() ||
        neighborIndices.size() < positions.size() * static_cast<size_t>(maxNeighbors)) {
        return false;
    }

    couplings.clear();
    couplings.reserve(positions.size() * static_cast<size_t>(maxNeighbors));
    std::vector<voronoi::NodeCoupling> cellCouplings;
    cellCouplings.reserve(maxNeighbors);

    for (uint32_t nodeId = 0; nodeId < static_cast<uint32_t>(positions.size()); ++nodeId) {
        voronoi::Node &node = nodes[nodeId];
        node.neighborOffset = static_cast<uint32_t>(couplings.size());
        node.neighborCount = 0;
        node.interfaceNeighborCount = 0;
        cellCouplings.clear();

        if ((nodeFlags[nodeId] & voronoi::NodeFlags::Ghost) != 0u) {
            continue;
        }

        const glm::vec3 nodePosition(positions[nodeId]);
        const size_t neighborOffset = static_cast<size_t>(nodeId) * maxNeighbors;
        for (uint32_t neighborIndex = 0; neighborIndex < maxNeighbors; ++neighborIndex) {
            const uint32_t neighborId = neighborIndices[neighborOffset + neighborIndex];
            if (neighborId == UINT32_MAX || neighborId >= positions.size()) {
                continue;
            }

            const float distance = glm::length(glm::vec3(positions[neighborId]) - nodePosition);
            if (distance <= 1e-12f) {
                continue;
            }
            cellCouplings.push_back({neighborId, 1.0f / distance});
        }

        std::sort(cellCouplings.begin(), cellCouplings.end(),
                  [](const voronoi::NodeCoupling &left, const voronoi::NodeCoupling &right) {
                      return left.neighborNodeId < right.neighborNodeId;
                  });

        couplings.insert(couplings.end(), cellCouplings.begin(), cellCouplings.end());
        node.neighborCount = static_cast<uint32_t>(cellCouplings.size());
        node.interfaceNeighborCount = node.neighborCount;
    }

    return !couplings.empty();
}
