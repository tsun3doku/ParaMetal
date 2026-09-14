#include "RVDGeometry.cuh"

#include "../../spatial/VoxelGrid.hpp"

#include <climits>

namespace voronoi::cudaGeometry {

bool uploadGeometry(
    const std::vector<std::array<glm::vec4, 3>>& triangles,
    const VoxelGrid& voxelGrid,
    DeviceModelGeometry& geometry,
    cudaStream_t stream) {
    geometry.reset();
    if (triangles.empty() || triangles.size() > size_t(INT32_MAX)) return false;

    const auto& params = voxelGrid.getParams();
    const auto& occupancy = voxelGrid.getOccupancyData();
    const auto& triangleIds = voxelGrid.getTrianglesList();
    const auto& offsets = voxelGrid.getOffsets();
    const size_t expectedOccupancy = size_t(params.gridDim.x + 1) *
                                     size_t(params.gridDim.y + 1) *
                                     size_t(params.gridDim.z + 1);
    const size_t cellCount = size_t(params.gridDim.x) * size_t(params.gridDim.y) * size_t(params.gridDim.z);
    if (!(params.cellSize > 0.0f) || params.gridDim.x <= 0 || params.gridDim.y <= 0 ||
        params.gridDim.z <= 0 || params.totalCells != cellCount ||
        occupancy.size() != expectedOccupancy ||
        offsets.size() != cellCount + 1) return false;

    int32_t previousOffset = 0;
    for (int32_t offset : offsets) {
        if (offset < previousOffset || offset < 0 ||
            size_t(offset) > triangleIds.size()) return false;
        previousOffset = offset;
    }
    if (size_t(offsets.back()) != triangleIds.size()) return false;
    for (int32_t triangleId : triangleIds) {
        if (triangleId < 0 || size_t(triangleId) >= triangles.size()) return false;
    }

    static_assert(sizeof(std::array<glm::vec4, 3>) == sizeof(DeviceTriangle));
    if (!geometry.triangles.allocate(triangles.size()) ||
        !geometry.occupancy.allocate(occupancy.size()) ||
        !geometry.voxelTriangleIds.allocate(triangleIds.size()) ||
        !geometry.voxelOffsets.allocate(offsets.size()) ||
        !geometry.triangles.uploadAsync(
            reinterpret_cast<const DeviceTriangle*>(triangles.data()), triangles.size(), stream) ||
        !geometry.occupancy.uploadAsync(occupancy.data(), occupancy.size(), stream) ||
        !geometry.voxelTriangleIds.uploadAsync(triangleIds.data(), triangleIds.size(), stream) ||
        !geometry.voxelOffsets.uploadAsync(offsets.data(), offsets.size(), stream)) {
        geometry.reset();
        return false;
    }

    geometry.voxelGrid.gridMin = make_float3(
        params.gridMin.x, params.gridMin.y, params.gridMin.z);
    geometry.voxelGrid.scale = params.cellSize;
    geometry.voxelGrid.gridDim = make_int3(
        params.gridDim.x, params.gridDim.y, params.gridDim.z);

    if (!buildTriangleBvh(geometry.triangles, geometry.bvh, stream)) {
        geometry.reset();
        return false;
    }

    return true;
}

} // namespace voronoi::cudaGeometry
