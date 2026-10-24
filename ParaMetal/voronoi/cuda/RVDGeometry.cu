#include "RVDGeometry.cuh"
#include "VoxelOccupancy.cuh"

#include "../../spatial/VoxelGrid.hpp"

#include <climits>
#include <limits>

namespace voronoi::cudaGeometry {

bool uploadGeometry(
    const std::vector<std::array<glm::vec4, 3>>& triangles,
    const VoxelGrid& voxelGrid,
    DeviceModelGeometry& geometry,
    cudaStream_t stream,
    uint32_t runtimeModelId) {
    geometry.reset();
    geometry.runtimeModelId = runtimeModelId;
    if (triangles.empty() || triangles.size() > size_t(INT32_MAX)) return false;

    const auto& params = voxelGrid.getParams();
    const auto& triangleIds = voxelGrid.getTrianglesList();
    const auto& offsets = voxelGrid.getOffsets();
    const size_t cellCount = size_t(params.gridDim.x) * size_t(params.gridDim.y) * size_t(params.gridDim.z);
    if (!(params.cellSize > 0.0f) || params.gridDim.x <= 0 || params.gridDim.y <= 0 ||
        params.gridDim.z <= 0 || params.totalCells != cellCount ||
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

    glm::vec3 boundsMin(std::numeric_limits<float>::max());
    glm::vec3 boundsMax(-std::numeric_limits<float>::max());
    for (const auto& triangle : triangles) {
        for (const glm::vec4& vertex : triangle) {
            const glm::vec3 position(vertex);
            boundsMin = glm::min(boundsMin, position);
            boundsMax = glm::max(boundsMax, position);
        }
    }
    if (!(boundsMin.x <= boundsMax.x) || !(boundsMin.y <= boundsMax.y) || !(boundsMin.z <= boundsMax.z)) {
        return false;
    }

    static_assert(sizeof(std::array<glm::vec4, 3>) == sizeof(DeviceTriangle));
    if (!geometry.triangles.allocate(triangles.size()) ||
        !geometry.voxelTriangleIds.allocate(triangleIds.size()) ||
        !geometry.voxelOffsets.allocate(offsets.size()) ||
        !geometry.triangles.uploadAsync(
            reinterpret_cast<const DeviceTriangle*>(triangles.data()), triangles.size(), stream) ||
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
    geometry.boundsMin = make_float3(boundsMin.x, boundsMin.y, boundsMin.z);
    geometry.boundsMax = make_float3(boundsMax.x, boundsMax.y, boundsMax.z);

    if (!buildTriangleBvh(geometry.triangles, geometry.bvh, stream)) {
        geometry.reset();
        return false;
    }

    if (!buildBvhMoments(geometry, stream)) {
        geometry.reset();
        return false;
    }

    return true;
}

} // namespace voronoi::cudaGeometry
