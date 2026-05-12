#pragma once

#include <vector>
#include <cstddef>

#include <glm/glm.hpp>
#include <libs/nanoflann/include/nanoflann.hpp>

struct VoronoiFloatAdapter {
    const std::vector<glm::vec3>& pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx][static_cast<int>(dim)];
    }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

struct VoronoiDoubleAdapter {
    const std::vector<glm::dvec3>& pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx][static_cast<int>(dim)];
    }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

struct StencilKDTree {
    std::vector<glm::vec3> regularSeedPositions;
    std::vector<uint32_t> regularLocalIndices;
    VoronoiFloatAdapter cloud;
    nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, VoronoiFloatAdapter>,
        VoronoiFloatAdapter, 3> index;
    size_t supportCount;

    template <typename VecType>
    StencilKDTree(const std::vector<uint32_t>& seedFlags, const std::vector<VecType>& seedPositions)
        : cloud{ regularSeedPositions },
          index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10)),
          supportCount(0) {
        regularSeedPositions.reserve(seedPositions.size());
        regularLocalIndices.reserve(seedPositions.size());
        for (uint32_t localIndex = 0; localIndex < seedPositions.size(); ++localIndex) {
            if (localIndex >= seedFlags.size() || (seedFlags[localIndex] & 1u) != 0u) {
                continue;
            }
            regularSeedPositions.push_back(glm::vec3(seedPositions[localIndex]));
            regularLocalIndices.push_back(localIndex);
        }
        if (regularSeedPositions.size() >= 4) {
            index.buildIndex();
            supportCount = std::min<size_t>(50, regularSeedPositions.size());
        }
    }

    bool isValid() const { return regularSeedPositions.size() >= 4; }
};
