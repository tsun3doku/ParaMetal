#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <libs/nanoflann/include/nanoflann.hpp>

class VoronoiNodeIndex {
public:
    VoronoiNodeIndex();
    VoronoiNodeIndex(const VoronoiNodeIndex& other);
    VoronoiNodeIndex& operator=(const VoronoiNodeIndex& other);
    VoronoiNodeIndex(VoronoiNodeIndex&&) = delete;
    VoronoiNodeIndex& operator=(VoronoiNodeIndex&&) = delete;

    void rebuild(const std::vector<glm::vec3>& updatedNodePositions);

    bool isValid() const { return nodePositions.size() >= 4; }

    void findKNearest(const glm::vec3& point, uint32_t requestedCount, std::vector<uint32_t>& nodeIds, std::vector<float>& distanceSquared) const;

    const std::vector<glm::vec3>& getNodePositions() const { return nodePositions; }

private:
    struct Adapter {
        const std::vector<glm::vec3>* nodePositions = nullptr;

        size_t kdtree_get_point_count() const { return nodePositions ? nodePositions->size() : 0; }
        float kdtree_get_pt(size_t index, size_t dimension) const {
            return (*nodePositions)[index][static_cast<int>(dimension)];
        }
        template <class BoundingBox>
        bool kdtree_get_bbox(BoundingBox&) const { return false; }
    };

    std::vector<glm::vec3> nodePositions;
    Adapter adapter{&nodePositions};
    nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, Adapter>, Adapter, 3> nodeTree;
};
