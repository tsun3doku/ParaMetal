#include "RVDBaseCells.cuh"
#include "RVDGeometry.cuh"
#include "RVDDiagnostics.hpp"
#include "RVDHostUtils.cuh"

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

namespace voronoi {

namespace {

using rvdDevice::ConvexCell;
using rvdDevice::ConvexCellOverflowStorage;
using rvdDevice::ConvexCellStorage;
using rvdDevice::integrateCell;

constexpr uint32_t BaseCellBlockSize = 32;

struct BaseDeviceInput {
  DeviceKnn knn;
  cudaGeometry::DeviceVoxelGrid grid;
  double4 domainPlanes[6];
};

struct BaseDeviceOutput {
  double4 *moments = nullptr;
  RVDStatus *statuses = nullptr;
};

__global__ void countBaseCellPlanesKernel(
    DeviceBaseCells baseCells,
    uint32_t count,
    uint32_t *cellPlaneCounts) {
  const uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell < count) {
    cellPlaneCounts[cell] = resolveBaseCell(baseCells, cell).planeCount;
  } else if (cell == count) {
    cellPlaneCounts[cell] = 0;
  }
}

__device__ inline double4 canonicalBisectorPlane(
    uint32_t cellA,
    uint32_t cellB,
    float4 seedA,
    float4 seedB,
    const cudaGeometry::DeviceVoxelGrid &grid) {
  const float4 seedU = (cellA < cellB) ? seedA : seedB;
  const float4 seedV = (cellA < cellB) ? seedB : seedA;

  const double scale = double(grid.scale);
  const double dx = (double(seedU.x) - double(seedV.x)) * scale;
  const double dy = (double(seedU.y) - double(seedV.y)) * scale;
  const double dz = (double(seedU.z) - double(seedV.z)) * scale;

  const double directionSquared = dx * dx + dy * dy + dz * dz;
  if (!isfinite(directionSquared) || !(directionSquared > 0.0)) {
    return make_double4(0.0, 0.0, 0.0, 0.0);
  }

  const double directionLength = sqrt(directionSquared);
  const double nx = dx / directionLength;
  const double ny = dy / directionLength;
  const double nz = dz / directionLength;

  const double mx = ((double(seedU.x) + double(seedV.x)) * 0.5 - double(grid.gridMin.x)) * scale;
  const double my = ((double(seedU.y) + double(seedV.y)) * 0.5 - double(grid.gridMin.y)) * scale;
  const double mz = ((double(seedU.z) + double(seedV.z)) * 0.5 - double(grid.gridMin.z)) * scale;

  const double weightOffset = 0.5 * (double(seedV.w) - double(seedU.w)) * scale * scale / directionLength;
  const double offset = -(mx * nx + my * ny + mz * nz) + weightOffset;

  if (cellA < cellB) {
    return make_double4(nx, ny, nz, offset);
  }
  return make_double4(-nx, -ny, -nz, -offset);
}

__device__ bool addNeighborPlanes(
    const BaseDeviceInput &input,
    uint32_t cellId,
    float4 seedPosition,
    float3 seedCanonical,
    ConvexCell &cell) {
  const int3 queryCell = cellIndexOf(
      make_float3(seedPosition.x, seedPosition.y, seedPosition.z),
      input.knn.grid);
  double maxVertexDistSq = cell.maximumVertexDistanceSquared(seedCanonical);
  double maxVertexDistance = sqrt(maxVertexDistSq);
  const double canonicalScale = double(input.grid.scale);
  const double canonicalScaleSq = canonicalScale * canonicalScale;

  for (uint32_t offsetIndex = 0; offsetIndex < input.knn.cellOffsetCount; ++offsetIndex) {
    const double distSq = input.knn.cellOffsetDistances[offsetIndex];
    if (distSq > 0.0) {
      const double minDist = 0.5 * sqrt(distSq * canonicalScaleSq);
      if (minDist * minDist > maxVertexDistSq) break;
    }

    const int3 offset = input.knn.cellOffsets[offsetIndex];
    const int x = queryCell.x + offset.x;
    const int y = queryCell.y + offset.y;
    const int z = queryCell.z + offset.z;
    if (x < 0 || x >= input.knn.grid.gridDim.x ||
        y < 0 || y >= input.knn.grid.gridDim.y ||
        z < 0 || z >= input.knn.grid.gridDim.z) {
      continue;
    }

    const uint32_t gridCell = linearCell(x, y, z, input.knn.grid);
    const uint32_t begin = min(input.knn.cellStart[gridCell], input.knn.seedCount);
    const uint32_t end = min(max(input.knn.cellStart[gridCell + 1], begin), input.knn.seedCount);

    bool bucketChanged = false;
    for (uint32_t index = begin; index < end; ++index) {
      const BucketEntry entry = input.knn.bucketEntries[index];
      const uint32_t neighbor = entry.id;
      if (neighbor >= input.knn.seedCount || neighbor == cellId) continue;

      const double4 plane = canonicalBisectorPlane(
          cellId, neighbor, seedPosition, entry.position, input.grid);
      if (!isfinite(plane.x) || !isfinite(plane.y) ||
          !isfinite(plane.z) || !isfinite(plane.w)) {
        cell.fail(RVDStatus::NonFiniteGeometry);
        return false;
      }

      const double planeDistance =
          double(plane.x) * double(seedCanonical.x) +
          double(plane.y) * double(seedCanonical.y) +
          double(plane.z) * double(seedCanonical.z) +
          double(plane.w);
      if (planeDistance >= maxVertexDistance) continue;

      if (uint32_t(cell.planeCount) >= cell.planeCapacity && !cell.compactPlanes()) {
        return false;
      }

      bool changed = false;
      if (!cell.addPlane(plane, int(neighbor), &changed)) return false;
      if (cell.status != RVDStatus::Success) return false;
      bucketChanged |= changed;
    }

    if (bucketChanged) {
      maxVertexDistSq = cell.maximumVertexDistanceSquared(seedCanonical);
      maxVertexDistance = sqrt(maxVertexDistSq);
    }
  }
  return cell.status == RVDStatus::Success;
}

__device__ bool buildUnrestrictedCell(
    const BaseDeviceInput &input,
    uint32_t cellId,
    float3 seedCanonical,
    ConvexCell &cell) {
  if (!cell.initializeDomainBox(input.domainPlanes)) return false;

  const float4 seedPosition = input.knn.seeds[cellId];
  return addNeighborPlanes(input, cellId, seedPosition, seedCanonical, cell);
}

template <typename StorageT>
__device__ void processBaseCell(
    uint32_t cellId,
    const BaseDeviceInput &input,
    BaseDeviceOutput output,
    ConvexCell &cell,
    StorageT &storage) {
  const auto failWith = [&](RVDStatus fallback) {
    output.statuses[cellId] =
        cell.status == RVDStatus::Success ? fallback : cell.status;
  };

  output.statuses[cellId] = RVDStatus::Success;

  const float4 seed = input.knn.seeds[cellId];
  const float3 seedCanonical = canonicalFromWorld(
      input.grid, make_float3(seed.x, seed.y, seed.z));

  if (!buildUnrestrictedCell(input, cellId, seedCanonical, cell)) {
    failWith(RVDStatus::NonFiniteGeometry);
    return;
  }
  if (!cell.compactPlanes()) {
    failWith(RVDStatus::NonFiniteGeometry);
    return;
  }

  const double4 rawMoment = integrateCell(cell, seedCanonical);
  if (!isfinite(rawMoment.w) || rawMoment.w <= 0.0) {
    output.statuses[cellId] = RVDStatus::NonFiniteGeometry;
    return;
  }

  if (!cell.saveStorage(storage)) {
    failWith(RVDStatus::VertexOverflow);
    return;
  }

  output.moments[cellId] = rawMoment;
}

__global__ void baseKernel(
    BaseDeviceInput input,
    BaseDeviceOutput output,
    ConvexCellStorage *baseStates,
    uint32_t begin,
    uint32_t count) {
  const uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local >= count) return;

  const uint32_t cellId = begin + local;
  if (cellId >= input.knn.seedCount) return;

  ConvexCellStorage &storage = baseStates[cellId];
  ConvexCell cell;
  cell.attachStorage(storage);
  processBaseCell(cellId, input, output, cell, storage);
}

__global__ void overflowBaseKernel(
    BaseDeviceInput input,
    BaseDeviceOutput output,
    const uint32_t *cellIds,
    uint32_t begin,
    uint32_t count,
    const int32_t *overflowSlots,
    ConvexCellOverflowStorage *overflowStates) {
  const uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local >= count) return;

  const uint32_t cellId = cellIds[begin + local];
  if (cellId >= input.knn.seedCount) return;

  const int32_t slot = overflowSlots[cellId];
  if (slot < 0) return;

  ConvexCellOverflowStorage &storage = overflowStates[slot];
  ConvexCell cell;
  cell.attachStorage(storage, RVDOverflowPlaneCount, RVDOverflowVertexCount);
  processBaseCell(cellId, input, output, cell, storage);
}

__global__ void markOverflowCellsKernel(
    const RVDStatus *statuses,
    uint32_t count,
    uint32_t *flags,
    uint32_t *cellIndices) {
  const uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= count) return;

  const RVDStatus status = statuses[cell];
  flags[cell] = status == RVDStatus::VertexOverflow ||
                status == RVDStatus::PlaneOverflow ||
                status == RVDStatus::InconsistentBoundary
      ? 1u
      : 0u;
  cellIndices[cell] = cell;
}

__global__ void scatterOverflowSlotsKernel(
    const uint32_t *overflowCells,
    uint32_t count,
    int32_t *overflowSlots) {
  const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= count) return;

  overflowSlots[overflowCells[slot]] = int32_t(slot);
}

__global__ void detectBaseFailuresKernel(
    const RVDStatus *statuses,
    uint32_t count,
    uint32_t *anyFailure) {
  const uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= count) return;

  if (statuses[cell] != RVDStatus::Success) {
    atomicExch(anyFailure, 1u);
  }
}

template <typename Launch>
bool launchChunks(
    uint32_t count,
    uint32_t requestedChunkSize,
    cudaStream_t stream,
    const char *launchLabel,
    const char *syncLabel,
    Launch &&launch) {
  if (count == 0) return true;

  const uint32_t chunkSize = std::max(1u, requestedChunkSize);
  for (uint32_t begin = 0; begin < count;) {
    const uint32_t chunkCount = std::min(chunkSize, count - begin);
    launch(begin, chunkCount);
    if (!rvdHost::checkLaunch(launchLabel)) return false;
    begin += chunkCount;
  }
  return rvdHost::synchronize(stream, syncLabel);
}

void buildDomainPlanes(
    const glm::vec3 &domainMin,
    const glm::vec3 &domainMax,
    const cudaGeometry::DeviceVoxelGrid &grid,
    double4 canonicalPlanes[6],
    std::array<glm::dvec4, 6> &outwardPlanes) {
  const float3 minCorner = canonicalFromWorld(grid, make_float3(domainMin.x, domainMin.y, domainMin.z));
  const float3 maxCorner = canonicalFromWorld(grid, make_float3(domainMax.x, domainMax.y, domainMax.z));

  canonicalPlanes[0] = make_double4(1.0, 0.0, 0.0, -double(minCorner.x));
  canonicalPlanes[1] = make_double4(-1.0, 0.0, 0.0, double(maxCorner.x));
  canonicalPlanes[2] = make_double4(0.0, 1.0, 0.0, -double(minCorner.y));
  canonicalPlanes[3] = make_double4(0.0, -1.0, 0.0, double(maxCorner.y));
  canonicalPlanes[4] = make_double4(0.0, 0.0, 1.0, -double(minCorner.z));
  canonicalPlanes[5] = make_double4(0.0, 0.0, -1.0, double(maxCorner.z));

  const glm::dvec3 worldMin(domainMin);
  const glm::dvec3 worldMax(domainMax);

  outwardPlanes[0] = glm::dvec4(-1.0, 0.0, 0.0, worldMin.x);
  outwardPlanes[1] = glm::dvec4(1.0, 0.0, 0.0, -worldMax.x);
  outwardPlanes[2] = glm::dvec4(0.0, -1.0, 0.0, worldMin.y);
  outwardPlanes[3] = glm::dvec4(0.0, 1.0, 0.0, -worldMax.y);
  outwardPlanes[4] = glm::dvec4(0.0, 0.0, -1.0, worldMin.z);
  outwardPlanes[5] = glm::dvec4(0.0, 0.0, 1.0, -worldMax.z);
}

BaseDeviceInput makeBaseDeviceInput(const RVDInput &input, const BaseCellDomain &baseDomain) {
  BaseDeviceInput deviceInput{};
  deviceInput.knn = input.seedSpatialIndex->deviceKnn();
  deviceInput.grid = baseDomain.baseGrid;
  for (int i = 0; i < 6; ++i) {
    deviceInput.domainPlanes[i] = baseDomain.domainPlanes[i];
  }
  return deviceInput;
}

bool runUnrestrictedPass(
    const RVDConfig &config,
    BaseCellDomain &baseDomain,
    BaseDeviceInput deviceInput,
    BaseDeviceOutput deviceOutput,
    cudaStream_t stream) {
  const bool launched = launchChunks(
      baseDomain.count, config.baseChunkSize, stream,
      "launch base kernel", "synchronize base pass",
      [&](uint32_t begin, uint32_t count) {
        baseKernel<<<rvdHost::blockCount(count, BaseCellBlockSize), BaseCellBlockSize, 0, stream>>>(
            deviceInput, deviceOutput, baseDomain.devBaseStates.get(), begin, count);
      });
  if (!launched) {
    std::cerr << "[RVD] base unrestricted kernel phase failed" << std::endl;
  }
  return launched;
}

bool downloadStatuses(
    const cudaUtils::CudaBuffer<RVDStatus> &deviceStatuses,
    std::vector<RVDStatus> &hostStatuses,
    cudaStream_t stream,
    const char *label) {
  return rvdHost::copyToHost(
      hostStatuses.data(), deviceStatuses.get(), hostStatuses.size(), stream, label);
}

bool buildOverflowList(
    const cudaUtils::CudaBuffer<RVDStatus> &deviceStatuses,
    uint32_t cellCount,
    cudaUtils::CudaBuffer<uint32_t> &deviceOverflowCells,
    uint32_t &overflowCount,
    cudaStream_t stream) {
  overflowCount = 0;
  if (cellCount == 0) return true;

  cudaUtils::CudaBuffer<uint32_t> overflowFlags;
  cudaUtils::CudaBuffer<uint32_t> cellIndices;
  cudaUtils::CudaBuffer<int> selectedCount;
  if (!overflowFlags.allocate(cellCount) ||
      !cellIndices.allocate(cellCount) ||
      !deviceOverflowCells.allocate(cellCount) ||
      !selectedCount.allocate(1)) {
    return false;
  }

  markOverflowCellsKernel<<<
      rvdHost::blockCount(cellCount), rvdHost::BlockSize, 0, stream>>>(
      deviceStatuses.get(), cellCount, overflowFlags.get(), cellIndices.get());
  if (!rvdHost::checkLaunch("launch markOverflowCellsKernel")) return false;

  cudaUtils::CudaBuffer<uint8_t> selectScratch;
  if (!rvdHost::selectFlagged(
          cellIndices.get(),
          overflowFlags.get(),
          deviceOverflowCells.get(),
          selectedCount.get(),
          int(cellCount),
          selectScratch,
          stream)) {
    return false;
  }

  int hostCount = 0;
  if (!rvdHost::copyToHost(
          &hostCount, selectedCount.get(), 1, stream,
          "read overflow cell count")) {
    return false;
  }
  overflowCount = uint32_t(hostCount);
  return true;
}

bool retryOverflowCells(
    const RVDConfig &config,
    BaseCellDomain &baseDomain,
    BaseDeviceInput deviceInput,
    BaseDeviceOutput deviceOutput,
    const cudaUtils::CudaBuffer<uint32_t> &deviceOverflowCells,
    uint32_t overflowCount,
    cudaStream_t stream) {
  if (overflowCount == 0) return true;

  if (!baseDomain.devOverflowStates.allocate(overflowCount) ||
      !baseDomain.devOverflowSlots.allocate(baseDomain.count) ||
      !rvdHost::fill(
          baseDomain.devOverflowSlots.get(), 0xFF,
          size_t(baseDomain.count) * sizeof(int32_t), stream,
          "initialize overflow slots")) {
    return false;
  }
  baseDomain.overflowCount = overflowCount;

  scatterOverflowSlotsKernel<<<
      rvdHost::blockCount(overflowCount), rvdHost::BlockSize, 0, stream>>>(
      deviceOverflowCells.get(), overflowCount,
      baseDomain.devOverflowSlots.get());
  if (!rvdHost::checkLaunch("launch scatterOverflowSlotsKernel")) {
    return false;
  }

  const bool launched = launchChunks(
      overflowCount, config.restrictedChunkSize, stream,
      "launch overflow base kernel", "synchronize overflow base pass",
      [&](uint32_t begin, uint32_t count) {
        overflowBaseKernel<<<rvdHost::blockCount(count, BaseCellBlockSize), BaseCellBlockSize, 0, stream>>>(
            deviceInput,
            deviceOutput,
            deviceOverflowCells.get(),
            begin,
            count,
            baseDomain.devOverflowSlots.get(),
            baseDomain.devOverflowStates.get());
      });
  if (!launched) {
    std::cerr << "[RVD] overflow base retry pass failed" << std::endl;
  }
  return launched;
}

bool detectBaseFailures(
    const cudaUtils::CudaBuffer<RVDStatus> &deviceStatuses,
    uint32_t cellCount,
    cudaStream_t stream,
    bool &hasFailures) {
  hasFailures = false;
  if (cellCount == 0) return true;

  cudaUtils::CudaBuffer<uint32_t> anyFailure;
  if (!anyFailure.allocate(1) ||
      !rvdHost::zero(
          anyFailure.get(), sizeof(uint32_t), stream,
          "clear base failure flag")) {
    return false;
  }

  detectBaseFailuresKernel<<<
      rvdHost::blockCount(cellCount), rvdHost::BlockSize, 0, stream>>>(
      deviceStatuses.get(), cellCount, anyFailure.get());
  if (!rvdHost::checkLaunch("launch detectBaseFailuresKernel")) return false;

  uint32_t hostFailure = 0;
  if (!rvdHost::copyToHost(
          &hostFailure, anyFailure.get(), 1, stream,
          "read base failure flag")) {
    return false;
  }
  hasFailures = hostFailure != 0;
  return true;
}
bool reportBaseFailures(
    const RVDInput &input,
    const std::vector<RVDStatus> &statuses) {
  bool success = true;
  for (uint32_t cell = 0; cell < statuses.size(); ++cell) {
    const RVDStatus status = statuses[cell];
    if (status == RVDStatus::Success) continue;

    std::cerr << "[RVD] base pass failed cell=" << cell
              << " status=" << rvdStatusName(status) << std::endl;
    success = false;
  }

  if (!success) {
    reportBaseCellFailures(input, statuses);
  }
  return success;
}

bool buildCellPlaneOffsets(BaseCellDomain &baseDomain, cudaStream_t stream) {
  cudaUtils::CudaBuffer<uint32_t> cellPlaneCounts;
  if (!cellPlaneCounts.allocate(size_t(baseDomain.count) + 1) ||
      !baseDomain.devCellPlaneOffsets.allocate(size_t(baseDomain.count) + 1)) {
    return false;
  }

  const uint32_t blocks = rvdHost::blockCount(baseDomain.count + 1);
  countBaseCellPlanesKernel<<<blocks, rvdHost::BlockSize, 0, stream>>>(
      baseDomain.deviceView(), baseDomain.count, cellPlaneCounts.get());
  if (!rvdHost::checkLaunch("launch countBaseCellPlanesKernel")) return false;

  cudaUtils::CudaBuffer<uint8_t> scanScratch;
  if (!rvdHost::exclusiveSum(
          cellPlaneCounts.get(),
          baseDomain.devCellPlaneOffsets.get(),
          int(baseDomain.count + 1),
          scanScratch,
          stream)) {
    return false;
  }

  return rvdHost::readAt(
      baseDomain.totalPlanes,
      baseDomain.devCellPlaneOffsets.get(),
      baseDomain.count,
      stream,
      "read total base cell planes");
}

} // namespace

bool prepareBaseDomain(const RVDInput &input, BaseCellDomain &baseDomain) {
  if (!input.seedSpatialIndex || input.seedSpatialIndex->seedCount() == 0 ||
      !input.seeds || input.seeds->empty() || !input.domainCorners) {
    return false;
  }

  baseDomain.count = uint32_t(input.seeds->size());
  baseDomain.domainMin = (*input.domainCorners)[0];
  baseDomain.domainMax = (*input.domainCorners)[0];
  for (const glm::vec3 &corner : *input.domainCorners) {
    baseDomain.domainMin = glm::min(baseDomain.domainMin, corner);
    baseDomain.domainMax = glm::max(baseDomain.domainMax, corner);
  }

  const glm::vec3 domainSize = baseDomain.domainMax - baseDomain.domainMin;
  const float maxDimension = std::max(domainSize.x, std::max(domainSize.y, domainSize.z));
  if (!(maxDimension > 1e-6f) || !std::isfinite(maxDimension)) {
    std::cerr << "[RVD] invalid domain extent" << std::endl;
    return false;
  }

  baseDomain.baseScale = cudaGeometry::CanonicalDomainSize / maxDimension;
  baseDomain.invBaseScale = 1.0 / double(baseDomain.baseScale);
  baseDomain.invBaseVolumeScale = baseDomain.invBaseScale * baseDomain.invBaseScale * baseDomain.invBaseScale;
  baseDomain.invBaseMomentScale = baseDomain.invBaseVolumeScale * baseDomain.invBaseScale;
  baseDomain.invBaseAreaScale = baseDomain.invBaseScale * baseDomain.invBaseScale;
  baseDomain.nominalCellSize = input.nominalCellSize;

  baseDomain.baseGrid.gridMin = make_float3(baseDomain.domainMin.x, baseDomain.domainMin.y, baseDomain.domainMin.z);
  baseDomain.baseGrid.scale = baseDomain.baseScale;

  buildDomainPlanes(
      baseDomain.domainMin, baseDomain.domainMax, baseDomain.baseGrid,
      baseDomain.domainPlanes, baseDomain.outwardDomainPlanes);
  return true;
}

bool buildBaseCells(
    const RVDInput &input,
    const RVDConfig &config,
    BaseCellDomain &baseDomain,
    cudaStream_t stream) {
  cudaUtils::CudaBuffer<RVDStatus> deviceStatuses;
  cudaUtils::CudaBuffer<double4> deviceMoments;
  if (!deviceMoments.allocate(baseDomain.count) ||
      !deviceStatuses.allocate(baseDomain.count) ||
      !baseDomain.devBaseStates.allocate(baseDomain.count)) {
    return false;
  }

  const BaseDeviceInput deviceInput = makeBaseDeviceInput(input, baseDomain);
  const BaseDeviceOutput deviceOutput{deviceMoments.get(), deviceStatuses.get()};

  if (!runUnrestrictedPass(config, baseDomain, deviceInput, deviceOutput, stream)) {
    return false;
  }

  cudaUtils::CudaBuffer<uint32_t> deviceOverflowCells;
  uint32_t overflowCount = 0;
  if (!buildOverflowList(
          deviceStatuses, baseDomain.count, deviceOverflowCells, overflowCount,
          stream)) {
    return false;
  }

  if (!retryOverflowCells(
          config, baseDomain, deviceInput, deviceOutput, deviceOverflowCells,
          overflowCount, stream)) {
    return false;
  }

  bool hasFailures = false;
  if (!detectBaseFailures(deviceStatuses, baseDomain.count, stream, hasFailures)) {
    return false;
  }
  if (hasFailures) {
    std::vector<RVDStatus> hostStatuses(baseDomain.count);
    if (!downloadStatuses(
            deviceStatuses, hostStatuses, stream, "download base statuses") ||
        !reportBaseFailures(input, hostStatuses)) {
      return false;
    }
  }

  if (!buildCellPlaneOffsets(baseDomain, stream)) return false;

  baseDomain.devBaseMoments = std::move(deviceMoments);
  return true;
}

} // namespace voronoi
