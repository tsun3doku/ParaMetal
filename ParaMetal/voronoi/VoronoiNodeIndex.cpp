#include "VoronoiNodeIndex.hpp"

VoronoiNodeIndex::VoronoiNodeIndex()
    : adapter{&nodePositions},
      nodeTree(3, adapter, nanoflann::KDTreeSingleIndexAdaptorParams(10)) {
}

VoronoiNodeIndex::VoronoiNodeIndex(const VoronoiNodeIndex& other)
    : VoronoiNodeIndex() {
    rebuild(other.nodePositions);
}

VoronoiNodeIndex& VoronoiNodeIndex::operator=(const VoronoiNodeIndex& other) {
    if (this != &other) {
        rebuild(other.nodePositions);
    }
    return *this;
}

void VoronoiNodeIndex::rebuild(const std::vector<glm::vec3>& updatedNodePositions) {
    nodePositions = updatedNodePositions;
    if (!nodePositions.empty()) {
        nodeTree.buildIndex();
    }
}

void VoronoiNodeIndex::findKNearest(
    const glm::vec3& point,
    uint32_t requestedCount,
    std::vector<uint32_t>& nodeIds,
    std::vector<float>& distanceSquared) const {
    nodeIds.clear();
    distanceSquared.clear();

    const uint32_t count = std::min<uint32_t>(requestedCount, static_cast<uint32_t>(nodePositions.size()));
    if (count == 0) {
        return;
    }

    std::vector<size_t> nearestIndices(count);
    distanceSquared.resize(count);
    const float queryPoint[3] = {point.x, point.y, point.z};
    nanoflann::KNNResultSet<float> resultSet(count);
    resultSet.init(nearestIndices.data(), distanceSquared.data());
    nodeTree.findNeighbors(resultSet, queryPoint);

    nodeIds.resize(count);
    for (uint32_t index = 0; index < count; ++index) {
        nodeIds[index] = static_cast<uint32_t>(nearestIndices[index]);
    }
}
