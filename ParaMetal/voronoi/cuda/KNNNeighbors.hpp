#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "KNNNeighborsKernels.cuh"

class VulkanDevice;

namespace voronoi {

class KNNNeighbors {
public:
    KNNNeighbors();
    ~KNNNeighbors();

    KNNNeighbors(const KNNNeighbors&) = delete;
    KNNNeighbors& operator=(const KNNNeighbors&) = delete;

    bool initialize(VulkanDevice& vulkanDevice);

    bool buildNeighbors(uint32_t k, const std::vector<uint32_t>& queryIds);
    bool buildSpatialIndex(const std::vector<glm::vec4>& seeds);

    void cleanup();

    DeviceKnn deviceKnn() const;

    uint32_t seedCount() const;

    bool downloadPrefix(uint32_t maxNeighbors, std::vector<uint32_t>& out) const;

private:
    class Implementation;
    std::unique_ptr<Implementation> implementation;
};

} // namespace voronoi
