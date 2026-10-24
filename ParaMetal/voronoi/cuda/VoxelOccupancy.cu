#include "VoxelOccupancy.cuh"
#include "TriangleWinding.cuh"

#include "../../cuda/CudaCheck.cuh"

#include <cmath>
#include <cstdint>

namespace voronoi::cudaGeometry {

constexpr uint32_t BlockSize = 256;

static __global__ void accumulateBvhMomentsKernel(
    const DeviceTriangle* triangles,
    const TriangleBvhNode* nodes,
    uint32_t nodeCount,
    const uint32_t* parents,
    float4* moments)
{
    const uint32_t nodeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (nodeIdx >= nodeCount) {
        return;
    }

    const TriangleBvhNode node = nodes[nodeIdx];
    if (!node.isLeaf()) {
        return;
    }

    const DeviceTriangle tri = triangles[node.triangleId()];
    const float3 a = make_float3(tri.v0.x, tri.v0.y, tri.v0.z);
    const float3 b = make_float3(tri.v1.x, tri.v1.y, tri.v1.z);
    const float3 c = make_float3(tri.v2.x, tri.v2.y, tri.v2.z);
    const float3 ab = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
    const float3 ac = make_float3(c.x - a.x, c.y - a.y, c.z - a.z);
    const float3 areaVector = make_float3(
        0.5f * (ab.y * ac.z - ab.z * ac.y),
        0.5f * (ab.z * ac.x - ab.x * ac.z),
        0.5f * (ab.x * ac.y - ab.y * ac.x));

    uint32_t target = nodeIdx;
    while (target != InvalidNodeId) {
        atomicAdd(&moments[target].x, areaVector.x);
        atomicAdd(&moments[target].y, areaVector.y);
        atomicAdd(&moments[target].z, areaVector.z);
        target = parents[target];
    }
}

bool buildBvhMoments(DeviceModelGeometry& geometry, cudaStream_t stream) {
    if (!geometry.bvh.valid() || geometry.triangles.size() == 0) {
        return false;
    }

    const uint32_t nodeCount = static_cast<uint32_t>(geometry.bvh.nodes.size());
    if (!geometry.bvhMoments.allocate(nodeCount) ||
        !cudaUtils::cudaCheck(
            cudaMemsetAsync(geometry.bvhMoments.get(), 0, nodeCount * sizeof(float4), stream),
            "clear bvh moments")) {
        return false;
    }

    const uint32_t momentBlocks = (nodeCount + BlockSize - 1) / BlockSize;
    accumulateBvhMomentsKernel<<<momentBlocks, BlockSize, 0, stream>>>(
        geometry.triangles.get(),
        geometry.bvh.nodes.get(),
        nodeCount,
        geometry.bvh.parents.get(),
        geometry.bvhMoments.get());
    return cudaUtils::cudaCheck(cudaGetLastError(), "launch accumulateBvhMomentsKernel");
}

static __global__ void buildWindingSolidMaskKernel(
    const DeviceModelGeometryView* views,
    uint32_t viewCount,
    float3 gridMin,
    int3 gridDim,
    float cellSize,
    uint8_t* outMask,
    uint32_t totalVoxels)
{
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= totalVoxels) return;

    const int x = int(index % uint32_t(gridDim.x));
    const int y = int((index / uint32_t(gridDim.x)) % uint32_t(gridDim.y));
    const int z = int(index / uint32_t(gridDim.x * gridDim.y));
    const double3 point = make_double3(
        double(gridMin.x) + double(x) * double(cellSize),
        double(gridMin.y) + double(y) * double(cellSize),
        double(gridMin.z) + double(z) * double(cellSize));
    const float3 pointF = make_float3(
        float(point.x), float(point.y), float(point.z));

    uint8_t solid = 0;
    for (uint32_t viewIndex = 0; viewIndex < viewCount; ++viewIndex) {
        const DeviceModelGeometryView& view = views[viewIndex];
        if (view.moments == nullptr) continue;
        if (pointF.x < view.boundsMin.x || pointF.x > view.boundsMax.x ||
            pointF.y < view.boundsMin.y || pointF.y > view.boundsMax.y ||
            pointF.z < view.boundsMin.z || pointF.z > view.boundsMax.z) {
            continue;
        }
        const double winding = computeWinding(
            view.triangles,
            view.nodes,
            view.parents,
            view.root,
            view.moments,
            point,
            pointF);
        if (isfinite(winding) && fabs(winding) > 0.5) {
            solid = 1;
            break;
        }
    }
    outMask[index] = solid;
}

bool buildWindingSolidMask(
    const DeviceModelGeometryView* views,
    uint32_t viewCount,
    float3 gridMin,
    int3 gridDim,
    float cellSize,
    uint8_t* outMask,
    cudaStream_t stream)
{
    if (views == nullptr || viewCount == 0 || outMask == nullptr ||
        gridDim.x <= 0 || gridDim.y <= 0 || gridDim.z <= 0 ||
        !(cellSize > 0.0f)) {
        return false;
    }

    const size_t totalVoxels = size_t(gridDim.x) * size_t(gridDim.y) * size_t(gridDim.z);
    if (totalVoxels == 0 || totalVoxels > UINT32_MAX) return false;

    cudaUtils::CudaBuffer<DeviceModelGeometryView> devViews;
    if (!devViews.allocate(viewCount) ||
        !devViews.uploadAsync(views, viewCount, stream)) {
        return false;
    }

    const uint32_t blocks = (uint32_t(totalVoxels) + BlockSize - 1) / BlockSize;
    buildWindingSolidMaskKernel<<<blocks, BlockSize, 0, stream>>>(
        devViews.get(),
        viewCount,
        gridMin,
        gridDim,
        cellSize,
        outMask,
        uint32_t(totalVoxels));
    return cudaUtils::cudaCheck(cudaGetLastError(), "launch buildWindingSolidMaskKernel");
}

} // namespace voronoi::cudaGeometry
