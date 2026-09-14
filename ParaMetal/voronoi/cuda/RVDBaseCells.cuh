#pragma once

#include "RVD.hpp"
#include "RVDStatus.cuh"
#include "RVDGeometry.cuh"
#include "ConvexCell.cuh"
#include "KNNNeighbors.hpp"
#include "../../cuda/CudaBuffer.cuh"

#include <vector>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>

namespace voronoi {

struct BaseCellData {
    double volume = 0.0;
    glm::dvec3 localMoment{0.0};
};

struct DeviceBaseCells {
    const rvdDevice::ConvexCellStorage<rvdDevice::NormalLimits>* cells = nullptr;
    const rvdDevice::ConvexCellStorage<rvdDevice::DiscoveryLimits>* retryCells = nullptr;
    const int32_t* retrySlots = nullptr;
    uint32_t cellCount = 0;
    uint32_t retryCount = 0;
};

inline constexpr float BasePlaneTolerance = 1e-6f;

__device__ inline uint32_t basePlaneCount(DeviceBaseCells cells, uint32_t cell) {
    if (cell >= cells.cellCount) return 0;
    const int32_t rSlot = cells.retrySlots[cell];
    if (rSlot >= 0 && uint32_t(rSlot) < cells.retryCount) {
        return cells.retryCells[uint32_t(rSlot)].planeCount;
    }
    return cells.cells[cell].planeCount;
}

__device__ inline float4 basePlane(DeviceBaseCells cells, uint32_t cell, uint32_t plane) {
    const int32_t rSlot = cells.retrySlots[cell];
    if (rSlot >= 0 && uint32_t(rSlot) < cells.retryCount) {
        return cells.retryCells[uint32_t(rSlot)].planes[plane];
    }
    return cells.cells[cell].planes[plane];
}

__device__ inline int basePlaneNeighbor(DeviceBaseCells cells, uint32_t cell, uint32_t plane) {
    if (cell >= cells.cellCount) return -1;
    const int32_t rSlot = cells.retrySlots[cell];
    if (rSlot >= 0 && uint32_t(rSlot) < cells.retryCount) {
        return cells.retryCells[uint32_t(rSlot)].planeNeighborIds[plane];
    }
    return cells.cells[cell].planeNeighborIds[plane];
}

struct BaseCellsContext {
    uint32_t count = 0;
    float baseScale = 1.0f;
    double invBaseScale = 1.0;
    double invBaseVolumeScale = 1.0;
    double invBaseMomentScale = 1.0;
    double invBaseAreaScale = 1.0;
    double areaTolerance = 0.0;

    glm::vec3 domainMin{0.0f};
    glm::vec3 domainMax{0.0f};
    std::array<glm::dvec4, 6> outwardDomainPlanes{};

    cudaGeometry::DeviceVoxelGrid baseGrid;
    float4 domainPlanes[6];

    std::vector<BaseCellData> baseCells;
    std::vector<rvdDevice::ConvexCellStorage<rvdDevice::NormalLimits>> hostBaseStates;
    std::vector<rvdDevice::ConvexCellStorage<rvdDevice::DiscoveryLimits>> hostRetryBaseStates;
    std::vector<int32_t> retrySlots;
    std::vector<uint32_t> cellPlaneOffsets;

    cudaUtils::CudaBuffer<int32_t> devRetrySlots;
    cudaUtils::CudaBuffer<uint32_t> devCellPlaneOffsets;
    cudaUtils::CudaBuffer<rvdDevice::ConvexCellStorage<rvdDevice::NormalLimits>> devBaseStates;
    cudaUtils::CudaBuffer<rvdDevice::ConvexCellStorage<rvdDevice::DiscoveryLimits>> devRetryBaseStates;

    DeviceBaseCells deviceView() const {
        DeviceBaseCells view{};
        view.cells = devBaseStates.get();
        view.retryCells = devRetryBaseStates.get();
        view.retrySlots = devRetrySlots.get();
        view.cellCount = count;
        view.retryCount = uint32_t(devRetryBaseStates.size());
        return view;
    }

    uint32_t planeCount(uint32_t cell) const {
        const int32_t rSlot = retrySlots[cell];
        return rSlot >= 0 ? hostRetryBaseStates[size_t(rSlot)].planeCount
                          : hostBaseStates[cell].planeCount;
    }

    int planeNeighbor(uint32_t cell, uint32_t localPlane) const {
        const int32_t rSlot = retrySlots[cell];
        return rSlot >= 0 ? hostRetryBaseStates[size_t(rSlot)].planeNeighborIds[localPlane]
                          : hostBaseStates[cell].planeNeighborIds[localPlane];
    }

    double baseFaceArea(uint32_t cell, uint32_t localPlane) const {
        const int32_t rSlot = retrySlots[cell];
        const double canonicalArea = rSlot >= 0
            ? double(hostRetryBaseStates[size_t(rSlot)].planeAreas[localPlane])
            : double(hostBaseStates[cell].planeAreas[localPlane]);
        return canonicalArea * invBaseAreaScale;
    }
};

bool prepareBaseDomain(const RVDInput& input, BaseCellsContext& ctx);

bool buildBaseCells(
    const RVDInput& input,
    const RVDConfig& config,
    BaseCellsContext& ctx,
    cudaStream_t stream);

} // namespace voronoi
