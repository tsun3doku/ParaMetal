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
#include <algorithm>
#include <cmath>
#include <limits>
#include <glm/glm.hpp>

namespace voronoi {

struct DeviceBaseCells {
    const rvdDevice::ConvexCellStorage* cells = nullptr;
    const rvdDevice::ConvexCellOverflowStorage* overflowCells = nullptr;
    const int32_t* overflowSlots = nullptr;
    uint32_t overflowCount = 0;
    uint32_t cellCount = 0;
};

struct BaseCellArrays {
    const double4* planes = nullptr;
    const uint64_t* planeTags = nullptr;
    const int* planeNeighborIds = nullptr;
    const uint3* vertices = nullptr;
    const double* planeAreas = nullptr;
    const double* planeAreaAbsSums = nullptr;
    const double3* planeMoments = nullptr;
    uint32_t planeCount = 0;
    uint32_t vertexCount = 0;
};

template <typename StorageT>
__device__ inline BaseCellArrays toBaseCellArrays(const StorageT& storage) {
    BaseCellArrays arrays{};
    arrays.planes = storage.planes;
    arrays.planeTags = storage.planeTags;
    arrays.planeNeighborIds = storage.planeNeighborIds;
    arrays.vertices = storage.vertices;
    arrays.planeAreas = storage.planeAreas;
    arrays.planeAreaAbsSums = storage.planeAreaAbsSums;
    arrays.planeMoments = storage.planeMoments;
    arrays.planeCount = storage.planeCount;
    arrays.vertexCount = storage.vertexCount;
    return arrays;
}

__device__ inline BaseCellArrays resolveBaseCell(const DeviceBaseCells& baseCells, uint32_t cellId) {
    if (baseCells.overflowSlots != nullptr && baseCells.overflowCells != nullptr) {
        const int32_t slot = baseCells.overflowSlots[cellId];
        if (slot >= 0 && uint32_t(slot) < baseCells.overflowCount) {
            return toBaseCellArrays(baseCells.overflowCells[slot]);
        }
    }
    return toBaseCellArrays(baseCells.cells[cellId]);
}

struct BaseCellDomain {
    uint32_t count = 0;
    float baseScale = 1.0f;
    double invBaseScale = 1.0;
    double invBaseVolumeScale = 1.0;
    double invBaseMomentScale = 1.0;
    double invBaseAreaScale = 1.0;

    glm::vec3 domainMin{0.0f};
    glm::vec3 domainMax{0.0f};
    std::array<glm::dvec4, 6> outwardDomainPlanes{};

    cudaGeometry::DeviceVoxelGrid baseGrid;
    double4 domainPlanes[6];

    uint32_t totalPlanes = 0;

    cudaUtils::CudaBuffer<uint32_t> devCellPlaneOffsets;
    cudaUtils::CudaBuffer<rvdDevice::ConvexCellStorage> devBaseStates;
    cudaUtils::CudaBuffer<double4> devBaseMoments;
    cudaUtils::CudaBuffer<rvdDevice::ConvexCellOverflowStorage> devOverflowStates;
    cudaUtils::CudaBuffer<int32_t> devOverflowSlots;
    uint32_t overflowCount = 0;

    DeviceBaseCells deviceView() const {
        DeviceBaseCells view{};
        view.cells = devBaseStates.get();
        view.overflowCells = devOverflowStates.get();
        view.overflowSlots = devOverflowSlots.get();
        view.overflowCount = overflowCount;
        view.cellCount = count;
        return view;
    }

    float nominalCellSize = 0.0f;

    double facetFloor() const {
        return 64.0 * std::numeric_limits<double>::epsilon()
               * double(nominalCellSize * nominalCellSize);
    }
};

bool prepareBaseDomain(const RVDInput& input, BaseCellDomain& baseDomain);

bool buildBaseCells(
    const RVDInput& input,
    const RVDConfig& config,
    BaseCellDomain& baseDomain,
    cudaStream_t stream);

} // namespace voronoi
