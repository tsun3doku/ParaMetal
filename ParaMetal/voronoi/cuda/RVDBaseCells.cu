#include "RVDBaseCells.cuh"
#include "RVDGeometry.cuh"

#include <iostream>
#include <algorithm>

namespace voronoi {

using rvdDevice::ConvexCell;
using rvdDevice::ConvexCellStorage;
using rvdDevice::DiscoveryLimits;
using rvdDevice::NormalLimits;
using rvdDevice::PredicateMode;
using rvdDevice::integrateCell;

struct BaseDeviceInput {
    DeviceKnn knn;
    cudaGeometry::DeviceVoxelGrid grid;
    float4 domainPlanes[6];
};

struct BaseDeviceOutput {
    double4* moments = nullptr;
    RVDStatus* statuses = nullptr;
};

template <PredicateMode Mode>
static __device__ bool buildUnrestrictedCell(
    const BaseDeviceInput& input,
    uint32_t cellId,
    ConvexCell<Mode>& cell) {

    if (!cell.initializeDomainBox(input.domainPlanes)) {
        return false;
    }

    const float4 seedPosition = input.knn.seeds[cellId];
    const float3 seed = canonicalFromWorld(
        input.grid,
        make_float3(seedPosition.x, seedPosition.y, seedPosition.z));
    const int3 queryCell = cellIndexOf(
        make_float3(seedPosition.x, seedPosition.y, seedPosition.z),
        input.knn.grid);
    float maxVertexDistSq = cell.maximumVertexDistanceSquared(seed);
    const float canonicalScaleSq = input.grid.scale * input.grid.scale;

    for (uint32_t offsetIndex = 0; offsetIndex < input.knn.cellOffsetCount; ++offsetIndex) {
        const double distSq = input.knn.cellOffsetDistances[offsetIndex];
        if (distSq > 0.0) {
            const float minDist = 0.5f * sqrtf(float(distSq) * canonicalScaleSq);
            if (minDist * minDist > maxVertexDistSq) {
                break;
            }
        }

        const int3 offset = input.knn.cellOffsets[offsetIndex];
        const int x = queryCell.x + offset.x;
        const int y = queryCell.y + offset.y;
        const int z = queryCell.z + offset.z;
        if (x < 0 || x >= input.knn.grid.gridDim.x ||
            y < 0 || y >= input.knn.grid.gridDim.y ||
            z < 0 || z >= input.knn.grid.gridDim.z) continue;

        const uint32_t gridCell = linearCell(x, y, z, input.knn.grid);
        const uint32_t begin = min(input.knn.cellStart[gridCell], input.knn.seedCount);
        const uint32_t end = min(max(input.knn.cellStart[gridCell + 1], begin), input.knn.seedCount);

        bool bucketChanged = false;

        for (uint32_t index = begin; index < end; ++index) {
            const BucketEntry entry = input.knn.bucketEntries[index];
            const uint32_t neighbor = entry.id;
            if (neighbor >= input.knn.seedCount || neighbor == cellId) continue;

            const double scale = double(input.grid.scale);

            const double dx =
                (double(seedPosition.x) - double(entry.position.x)) * scale;
            const double dy =
                (double(seedPosition.y) - double(entry.position.y)) * scale;
            const double dz =
                (double(seedPosition.z) - double(entry.position.z)) * scale;

            const double directionSquared =
                dx * dx + dy * dy + dz * dz;

            if (!isfinite(directionSquared)) {
                cell.fail(RVDStatus::NonFiniteGeometry);
                return false;
            }

            if (!(directionSquared > 0.0)) {
                cell.fail(RVDStatus::NonFiniteGeometry);
                return false;
            }

            const double directionLength = sqrt(directionSquared);

            const double nx = dx / directionLength;
            const double ny = dy / directionLength;
            const double nz = dz / directionLength;

            const double mx =
                ((double(seedPosition.x) + double(entry.position.x)) * 0.5 -
                 double(input.grid.gridMin.x)) * scale;

            const double my =
                ((double(seedPosition.y) + double(entry.position.y)) * 0.5 -
                 double(input.grid.gridMin.y)) * scale;

            const double mz =
                ((double(seedPosition.z) + double(entry.position.z)) * 0.5 -
                 double(input.grid.gridMin.z)) * scale;

            const double weightOffset =
                0.5 *
                (double(entry.position.w) - double(seedPosition.w)) *
                scale * scale / directionLength;

            const double offset =
                -(mx * nx + my * ny + mz * nz) +
                weightOffset;

            const float4 plane = make_float4(
                float(nx),
                float(ny),
                float(nz),
                float(offset));

            if (uint32_t(cell.planeCount) >= cell.planeCapacity && !cell.compactPlanes()) return false;
            bool changed = false;
            if (!cell.addPlane(plane, int(neighbor), &changed)) return false;
            if (cell.status != RVDStatus::Success) return false;
            bucketChanged |= changed;
        }

        if (bucketChanged) {
            maxVertexDistSq = cell.maximumVertexDistanceSquared(seed);
        }
    }

    return cell.status == RVDStatus::Success;
}

template <class StateLimits, PredicateMode Mode>
static __device__ void processBaseCell(
    uint32_t cellId,
    const BaseDeviceInput& input,
    BaseDeviceOutput output,
    ConvexCell<Mode>& cell,
    ConvexCellStorage<StateLimits>& storage) {

    cell.cellId = cellId;
    output.statuses[cellId] = RVDStatus::Success;

    const bool builtUnrestricted = buildUnrestrictedCell(input, cellId, cell);
    if (!builtUnrestricted) {
        output.statuses[cellId] = cell.status == RVDStatus::Success ? RVDStatus::NonFiniteGeometry : cell.status;
        return;
    }
    if (!cell.compactPlanes()) {
        output.statuses[cellId] = cell.status == RVDStatus::Success ? RVDStatus::NonFiniteGeometry : cell.status;
        return;
    }

    const double4 rawMoment = integrateCell(
        cell,
        canonicalFromWorld(
            input.grid,
            make_float3(input.knn.seeds[cellId].x, input.knn.seeds[cellId].y, input.knn.seeds[cellId].z)));

    if (!isfinite(rawMoment.w) || rawMoment.w <= 0.0) {
        output.statuses[cellId] = RVDStatus::NonFiniteGeometry;
        return;
    }

    if (!cell.template saveStorage<StateLimits>(storage)) {
        output.statuses[cellId] = cell.status == RVDStatus::Success ? RVDStatus::VertexOverflow : cell.status;
        return;
    }

    output.moments[cellId] = rawMoment;
}

template <PredicateMode Mode>
static __global__ void baseKernel(
    BaseDeviceInput input,
    BaseDeviceOutput output,
    ConvexCellStorage<NormalLimits>* baseStates,
    uint32_t begin,
    uint32_t count) {

    const uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;

    const uint32_t cellId = begin + local;
    if (cellId >= input.knn.seedCount) return;

    ConvexCellStorage<NormalLimits>& storage = baseStates[cellId];

    ConvexCell<Mode> cell;
    cell.attachStorage(storage);
    processBaseCell<NormalLimits, Mode>(cellId, input, output, cell, storage);
}

template <PredicateMode Mode>
static __global__ void retryBaseKernel(
    BaseDeviceInput input,
    BaseDeviceOutput output,
    const uint32_t* cellIds,
    const int32_t* retrySlots,
    uint32_t retryCount,
    ConvexCellStorage<DiscoveryLimits>* retryBaseStates,
    uint32_t begin,
    uint32_t count) {

    const uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;

    const uint32_t index = begin + local;
    if (index >= retryCount) return;

    const uint32_t cellId = cellIds[index];
    if (cellId >= input.knn.seedCount) return;

    const int32_t slot = retrySlots[cellId];
    if (slot < 0) return;

    ConvexCellStorage<DiscoveryLimits>& storage = retryBaseStates[slot];

    ConvexCell<Mode> cell;
    cell.attachStorage(storage);
    processBaseCell<DiscoveryLimits, Mode>(cellId, input, output, cell, storage);
}

template <class Launch>
static bool launchChunks(uint32_t count, uint32_t initialChunkSize, cudaStream_t stream, Launch launch) {
    if (count == 0) return true;
    const uint32_t chunkSize = std::max(1u, initialChunkSize);
    for (uint32_t begin = 0; begin < count;) {
        const uint32_t chunkCount = std::min(chunkSize, count - begin);
        launch(begin, chunkCount);
        if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch base kernel")) return false;
        begin += chunkCount;
    }
    return cudaUtils::cudaCheck(cudaStreamSynchronize(stream), "synchronize base pass");
}

static void buildDomainPlanes(
    const glm::vec3& domainMin,
    const glm::vec3& domainMax,
    const cudaGeometry::DeviceVoxelGrid& grid,
    float4 canonicalPlanes[6],
    std::array<glm::dvec4, 6>& outwardPlanes) {

    const float3 minCorner = canonicalFromWorld(grid, make_float3(domainMin.x, domainMin.y, domainMin.z));
    const float3 maxCorner = canonicalFromWorld(grid, make_float3(domainMax.x, domainMax.y, domainMax.z));

    canonicalPlanes[0] = make_float4( 1.0f,  0.0f,  0.0f, -minCorner.x);
    canonicalPlanes[1] = make_float4(-1.0f,  0.0f,  0.0f,  maxCorner.x);
    canonicalPlanes[2] = make_float4( 0.0f,  1.0f,  0.0f, -minCorner.y);
    canonicalPlanes[3] = make_float4( 0.0f, -1.0f,  0.0f,  maxCorner.y);
    canonicalPlanes[4] = make_float4( 0.0f,  0.0f,  1.0f, -minCorner.z);
    canonicalPlanes[5] = make_float4( 0.0f,  0.0f, -1.0f,  maxCorner.z);

    const glm::dvec3 worldMin(domainMin);
    const glm::dvec3 worldMax(domainMax);

    outwardPlanes[0] = glm::dvec4(-1.0,  0.0,  0.0,  worldMin.x);
    outwardPlanes[1] = glm::dvec4( 1.0,  0.0,  0.0, -worldMax.x);
    outwardPlanes[2] = glm::dvec4( 0.0, -1.0,  0.0,  worldMin.y);
    outwardPlanes[3] = glm::dvec4( 0.0,  1.0,  0.0, -worldMax.y);
    outwardPlanes[4] = glm::dvec4( 0.0,  0.0, -1.0,  worldMin.z);
    outwardPlanes[5] = glm::dvec4( 0.0,  0.0,  1.0, -worldMax.z);
}

bool prepareBaseDomain(const RVDInput& input, BaseCellsContext& ctx) {
    if (!input.seedSpatialIndex || input.seedSpatialIndex->seedCount() == 0 || !input.seeds || input.seeds->empty() || !input.domainCorners) {
        return false;
    }

    ctx.count = uint32_t(input.seeds->size());
    ctx.domainMin = (*input.domainCorners)[0];
    ctx.domainMax = (*input.domainCorners)[0];
    for (const auto& corner : *input.domainCorners) {
        ctx.domainMin = glm::min(ctx.domainMin, corner);
        ctx.domainMax = glm::max(ctx.domainMax, corner);
    }
    const glm::vec3 domainSize = ctx.domainMax - ctx.domainMin;
    const float maxDim = std::max(domainSize.x, std::max(domainSize.y, domainSize.z));
    if (!(maxDim > 1e-6f) || !std::isfinite(maxDim)) {
        std::cerr << "[RVD] invalid domain extent" << std::endl;
        return false;
    }

    ctx.baseScale = cudaGeometry::CanonicalDomainSize / maxDim;
    ctx.invBaseScale = 1.0 / double(ctx.baseScale);
    ctx.invBaseVolumeScale = ctx.invBaseScale * ctx.invBaseScale * ctx.invBaseScale;
    ctx.invBaseMomentScale = ctx.invBaseVolumeScale * ctx.invBaseScale;
    ctx.invBaseAreaScale = ctx.invBaseScale * ctx.invBaseScale;
    ctx.areaTolerance = std::max(1e-12, double(input.nominalCellSize * input.nominalCellSize) * 1e-7);

    ctx.baseGrid.gridMin = make_float3(ctx.domainMin.x, ctx.domainMin.y, ctx.domainMin.z);
    ctx.baseGrid.scale = ctx.baseScale;

    buildDomainPlanes(
        ctx.domainMin,
        ctx.domainMax,
        ctx.baseGrid,
        ctx.domainPlanes,
        ctx.outwardDomainPlanes);

    return true;
}

bool buildBaseCells(
    const RVDInput& input,
    const RVDConfig& config,
    BaseCellsContext& ctx,
    cudaStream_t stream) {

    cudaUtils::CudaBuffer<RVDStatus> baseStatuses;
    cudaUtils::CudaBuffer<double4> devBaseMoments;
    if (!devBaseMoments.allocate(ctx.count) ||
        !baseStatuses.allocate(ctx.count) ||
        !ctx.devBaseStates.allocate(ctx.count)) return false;

    BaseDeviceInput baseInput{};
    baseInput.knn = input.seedSpatialIndex->deviceKnn();
    baseInput.grid = ctx.baseGrid;
    for (int i = 0; i < 6; ++i) baseInput.domainPlanes[i] = ctx.domainPlanes[i];

    BaseDeviceOutput baseOutput{};
    baseOutput.moments = devBaseMoments.get();
    baseOutput.statuses = baseStatuses.get();

    bool ok = launchChunks(ctx.count, config.baseChunkSize, stream,
        [&](uint32_t begin, uint32_t chunkCount) {
            constexpr uint32_t BlockSize = 32;
            const uint32_t blocks = (chunkCount + BlockSize - 1) / BlockSize;
            baseKernel<PredicateMode::FilteredFloat><<<blocks, BlockSize, 0, stream>>>(
                baseInput,
                baseOutput,
                ctx.devBaseStates.get(),
                begin,
                chunkCount);
        });
    if (!ok) {
        std::cerr << "[RVD] base unrestricted kernel phase failed" << std::endl;
        return false;
    }

    std::vector<RVDStatus> downloadedBaseStatuses(ctx.count);
    if (!baseStatuses.downloadAsync(downloadedBaseStatuses.data(), ctx.count, stream) ||
        !cudaUtils::cudaCheck(cudaStreamSynchronize(stream), "synchronize initial base status download")) return false;

    std::vector<uint32_t> baseRetryQueue;
    for (uint32_t cell = 0; cell < ctx.count; ++cell) {
        switch (downloadedBaseStatuses[cell]) {
        case RVDStatus::Success:
            break;

        case RVDStatus::VertexOverflow:
        case RVDStatus::PlaneOverflow:
        case RVDStatus::NeedsRobustPredicate:
            baseRetryQueue.push_back(cell);
            break;

        default:
            std::cerr << "[RVD] base pass failed cell=" << cell
                      << " status="
                      << rvdStatusName(downloadedBaseStatuses[cell])
                      << std::endl;
            return false;
        }
    }

    ctx.retrySlots.assign(ctx.count, -1);
    for (uint32_t slot = 0; slot < uint32_t(baseRetryQueue.size()); ++slot) {
        ctx.retrySlots[baseRetryQueue[slot]] = int32_t(slot);
    }

    if (!ctx.devRetrySlots.allocate(ctx.count) ||
        !ctx.devRetrySlots.uploadAsync(ctx.retrySlots.data(), ctx.count, stream)) {
        return false;
    }

    if (!baseRetryQueue.empty()) {
        const uint32_t retryCount = uint32_t(baseRetryQueue.size());
        cudaUtils::CudaBuffer<uint32_t> devBaseRetryQueue;
        if (!devBaseRetryQueue.allocate(retryCount) ||
            !devBaseRetryQueue.uploadAsync(baseRetryQueue.data(), retryCount, stream) ||
            !ctx.devRetryBaseStates.allocate(retryCount)) return false;

        ok = launchChunks(retryCount, config.restrictedChunkSize, stream,
            [&](uint32_t begin, uint32_t chunkCount) {
                constexpr uint32_t BlockSize = 32;
                const uint32_t blocks = (chunkCount + BlockSize - 1) / BlockSize;
                retryBaseKernel<PredicateMode::FilteredDouble><<<blocks, BlockSize, 0, stream>>>(
                    baseInput, baseOutput, devBaseRetryQueue.get(),
                    ctx.devRetrySlots.get(), retryCount,
                    ctx.devRetryBaseStates.get(), begin, chunkCount);
            });
        if (!ok) {
            std::cerr << "[RVD] robust base retry pass failed" << std::endl;
            return false;
        }

        if (!baseStatuses.downloadAsync(downloadedBaseStatuses.data(), ctx.count, stream) ||
            !cudaUtils::cudaCheck(cudaStreamSynchronize(stream), "synchronize base retry status download")) return false;

        uint32_t retryFailures = 0;
        for (uint32_t cell : baseRetryQueue) {
            const RVDStatus status = downloadedBaseStatuses[cell];
            if (status != RVDStatus::Success) {
                std::cerr << "[RVD] post-double failure cell=" << cell
                          << " status=" << rvdStatusName(status) << std::endl;
                ++retryFailures;
            }
        }
        if (retryFailures > 0) {
            return false;
        }
    }

    std::vector<double4> downloadedBaseMoments(ctx.count);
    if (!devBaseMoments.downloadAsync(downloadedBaseMoments.data(), ctx.count, stream)) {
        return false;
    }

    ctx.hostBaseStates.resize(ctx.count);
    if (!ctx.devBaseStates.downloadAsync(ctx.hostBaseStates.data(), ctx.count, stream)) {
        return false;
    }

    ctx.hostRetryBaseStates.resize(baseRetryQueue.size());
    if (!baseRetryQueue.empty() &&
        !ctx.devRetryBaseStates.downloadAsync(
            ctx.hostRetryBaseStates.data(),
            uint32_t(baseRetryQueue.size()),
            stream)) {
        return false;
    }

    if (!cudaUtils::cudaCheck(
            cudaStreamSynchronize(stream),
            "download base cell results")) {
        return false;
    }

    ctx.cellPlaneOffsets.assign(ctx.count + 1, 0);
    for (uint32_t cell = 0; cell < ctx.count; ++cell)
        ctx.cellPlaneOffsets[cell + 1] = ctx.cellPlaneOffsets[cell] + ctx.planeCount(cell);

    if (!ctx.devCellPlaneOffsets.allocate(ctx.count + 1) ||
        !ctx.devCellPlaneOffsets.uploadAsync(ctx.cellPlaneOffsets.data(), ctx.count + 1, stream)) {
        return false;
    }

    ctx.baseCells.resize(ctx.count);
    for (uint32_t cell = 0; cell < ctx.count; ++cell) {
        const double4 rawMoment = downloadedBaseMoments[cell];
        if (!std::isfinite(rawMoment.w) || rawMoment.w <= 0.0) {
            std::cerr << "[RVD-Invariant] invalid base volume cell=" << cell
                      << " rawVolume=" << rawMoment.w << std::endl;
            return false;
        }

        ctx.baseCells[cell].volume = (rawMoment.w / 6.0) * ctx.invBaseVolumeScale;
        ctx.baseCells[cell].localMoment = (glm::dvec3(rawMoment.x, rawMoment.y, rawMoment.z) / 6.0) * ctx.invBaseMomentScale;
    }

    return true;
}

} // namespace voronoi
