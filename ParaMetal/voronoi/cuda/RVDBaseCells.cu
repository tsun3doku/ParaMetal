#include "RVDBaseCells.cuh"
#include "RVDGeometry.cuh"

#include <algorithm>
#include <cub/cub.cuh>
#include <iostream>

namespace voronoi {

static __global__ void countBaseCellPlanesKernel(DeviceBaseCells baseCells,
                                                 uint32_t count,
                                                 uint32_t *cellPlaneCounts) {
  const uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell < count) {
    cellPlaneCounts[cell] = resolveBaseCell(baseCells, cell).planeCount;
  } else if (cell == count) {
    cellPlaneCounts[cell] = 0;
  }
}

using rvdDevice::ConvexCell;
using rvdDevice::ConvexCellOverflowStorage;
using rvdDevice::ConvexCellStorage;
using rvdDevice::integrateCell;

struct BaseDeviceInput {
  DeviceKnn knn;
  cudaGeometry::DeviceVoxelGrid grid;
  double4 domainPlanes[6];
};

struct BaseDeviceOutput {
  double4 *moments = nullptr;
  RVDStatus *statuses = nullptr;
};

__device__ inline double4
canonicalBisectorPlane(uint32_t cellA, uint32_t cellB, float4 seedA,
                       float4 seedB, const cudaGeometry::DeviceVoxelGrid &grid,
                       uint64_t &outTag) {
  const uint32_t u = min(cellA, cellB);
  const uint32_t v = max(cellA, cellB);
  outTag = (uint64_t(u) << 32) | uint64_t(v);

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
  } else {
    return make_double4(-nx, -ny, -nz, -offset);
  }
}

static __device__ bool
buildUnrestrictedCell(const BaseDeviceInput &input, BaseDeviceOutput output,
                      uint32_t cellId, float3 seedCanonical, ConvexCell &cell) {

  if (!cell.initializeDomainBox(input.domainPlanes)) {
    return false;
  }

  const float4 seedPosition = input.knn.seeds[cellId];
  const int3 queryCell = cellIndexOf(make_float3(seedPosition.x, seedPosition.y, seedPosition.z), input.knn.grid);
  double maxVertexDistSq = cell.maximumVertexDistanceSquared(seedCanonical);
  double maxVertexDistance = sqrt(maxVertexDistSq);
  const double canonicalScale = double(input.grid.scale);
  const double canonicalScaleSq = canonicalScale * canonicalScale;

  for (uint32_t offsetIndex = 0; offsetIndex < input.knn.cellOffsetCount;
       ++offsetIndex) {
    const double distSq = input.knn.cellOffsetDistances[offsetIndex];
    if (distSq > 0.0) {
      const double minDist = 0.5 * sqrt(distSq * canonicalScaleSq);
      if (minDist * minDist > maxVertexDistSq) {
        break;
      }
    }

    const int3 offset = input.knn.cellOffsets[offsetIndex];
    const int x = queryCell.x + offset.x;
    const int y = queryCell.y + offset.y;
    const int z = queryCell.z + offset.z;
    if (x < 0 || x >= input.knn.grid.gridDim.x || y < 0 ||
        y >= input.knn.grid.gridDim.y || z < 0 || z >= input.knn.grid.gridDim.z)
      continue;

    const uint32_t gridCell = linearCell(x, y, z, input.knn.grid);
    const uint32_t begin = min(input.knn.cellStart[gridCell], input.knn.seedCount);
    const uint32_t end = min(max(input.knn.cellStart[gridCell + 1], begin), input.knn.seedCount);

    bool bucketChanged = false;

    for (uint32_t index = begin; index < end; ++index) {
      const BucketEntry entry = input.knn.bucketEntries[index];
      const uint32_t neighbor = entry.id;
      if (neighbor >= input.knn.seedCount || neighbor == cellId)
        continue;

      uint64_t planeTag = 0;
      const double4 plane = canonicalBisectorPlane(
          cellId, neighbor, seedPosition, entry.position, input.grid, planeTag);

      if (!isfinite(plane.x) || !isfinite(plane.y) || !isfinite(plane.z) ||
          !isfinite(plane.w)) {
        cell.fail(RVDStatus::NonFiniteGeometry);
        return false;
      }

      const double planeDistance = double(plane.x) * double(seedCanonical.x) +
                                   double(plane.y) * double(seedCanonical.y) +
                                   double(plane.z) * double(seedCanonical.z) +
                                   double(plane.w);
      if (planeDistance >= maxVertexDistance) {
        continue;
      }

      if (uint32_t(cell.planeCount) >= cell.planeCapacity &&
          !cell.compactPlanes())
        return false;
      bool changed = false;
      if (!cell.addPlane(plane, int(neighbor), planeTag, &changed)) {
        return false;
      }
      if (cell.status != RVDStatus::Success) {
        return false;
      }
      bucketChanged |= changed;
    }

    if (bucketChanged) {
      maxVertexDistSq = cell.maximumVertexDistanceSquared(seedCanonical);
      maxVertexDistance = sqrt(maxVertexDistSq);
    }
  }

  return cell.status == RVDStatus::Success;
}

template <typename StorageT>
static __device__ void
processBaseCell(uint32_t cellId, const BaseDeviceInput &input,
                BaseDeviceOutput output, ConvexCell &cell, StorageT &storage) {

  const auto failWith = [&](RVDStatus fallback) {
    output.statuses[cellId] =
        cell.status == RVDStatus::Success ? fallback : cell.status;
  };

  output.statuses[cellId] = RVDStatus::Success;

  const float3 seedCanonical =
      canonicalFromWorld(input.grid, make_float3(input.knn.seeds[cellId].x,
                                                 input.knn.seeds[cellId].y,
                                                 input.knn.seeds[cellId].z));

  if (!buildUnrestrictedCell(input, output, cellId, seedCanonical, cell)) {
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

static __global__ void baseKernel(BaseDeviceInput input,
                                  BaseDeviceOutput output,
                                  ConvexCellStorage *baseStates, uint32_t begin,
                                  uint32_t count) {

  const uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local >= count)
    return;

  const uint32_t cellId = begin + local;
  if (cellId >= input.knn.seedCount)
    return;

  ConvexCellStorage &storage = baseStates[cellId];

  ConvexCell cell;
  cell.attachStorage(storage);
  processBaseCell(cellId, input, output, cell, storage);
}

static __global__ void
overflowBaseKernel(BaseDeviceInput input, BaseDeviceOutput output,
                   const uint32_t *cellIds, uint32_t begin, uint32_t count,
                   const int32_t *overflowSlots,
                   ConvexCellOverflowStorage *overflowStates) {

  const uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
  if (local >= count)
    return;

  const uint32_t cellId = cellIds[begin + local];
  if (cellId >= input.knn.seedCount)
    return;

  const int32_t slot = overflowSlots[cellId];
  if (slot < 0)
    return;

  ConvexCellOverflowStorage &storage = overflowStates[slot];

  ConvexCell cell;
  cell.attachStorage(storage, RVDOverflowPlaneCount, RVDOverflowVertexCount);
  processBaseCell(cellId, input, output, cell, storage);
}

template <class Launch>
static bool launchChunks(uint32_t count, uint32_t initialChunkSize,
                         cudaStream_t stream, Launch launch) {
  if (count == 0)
    return true;
  const uint32_t chunkSize = std::max(1u, initialChunkSize);
  for (uint32_t begin = 0; begin < count;) {
    const uint32_t chunkCount = std::min(chunkSize, count - begin);
    launch(begin, chunkCount);
    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch base kernel"))
      return false;
    begin += chunkCount;
  }
  return cudaUtils::cudaCheck(cudaStreamSynchronize(stream),
                              "synchronize base pass");
}

static void
reportBaseFailureDiagnostics(const RVDInput &input,
                             const std::vector<RVDStatus> &statuses) {
  if (!input.seeds || input.seeds->empty() || !input.domainCorners)
    return;
  const auto &seeds = *input.seeds;

  glm::vec3 domainMin = (*input.domainCorners)[0];
  glm::vec3 domainMax = (*input.domainCorners)[0];
  for (const glm::vec3 &corner : *input.domainCorners) {
    domainMin = glm::min(domainMin, corner);
    domainMax = glm::max(domainMax, corner);
  }

  constexpr size_t nearestCount = 64;
  for (size_t cell = 0; cell < statuses.size(); ++cell) {
    if (statuses[cell] == RVDStatus::Success)
      continue;
    const glm::vec4 seed4 = seeds[cell];
    const glm::dvec3 seed(seed4.x, seed4.y, seed4.z);

    std::vector<std::pair<double, size_t>> nearest;
    nearest.reserve(nearestCount);
    for (size_t other = 0; other < seeds.size(); ++other) {
      if (other == cell)
        continue;
      const glm::vec4 p = seeds[other];
      const glm::dvec3 delta(p.x - seed4.x, p.y - seed4.y, p.z - seed4.z);
      const double d2 = glm::dot(delta, delta);
      if (nearest.size() < nearestCount) {
        nearest.emplace_back(d2, other);
        if (nearest.size() == nearestCount)
          std::sort(nearest.begin(), nearest.end());
      } else if (d2 < nearest.back().first) {
        nearest.back() = {d2, other};
        std::sort(nearest.begin(), nearest.end());
      }
    }

    double minPairDist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < nearest.size(); ++i) {
      for (size_t j = i + 1; j < nearest.size(); ++j) {
        const glm::vec4 a = seeds[nearest[i].second];
        const glm::vec4 b = seeds[nearest[j].second];
        const glm::dvec3 ab(a.x - b.x, a.y - b.y, a.z - b.z);
        minPairDist = std::min(minPairDist, glm::length(ab));
      }
    }

    std::vector<std::pair<glm::dvec3, double>> planes;
    planes.reserve(nearest.size());
    for (const auto &entry : nearest) {
      const glm::vec4 p = seeds[entry.second];
      const glm::dvec3 dir(p.x - seed.x, p.y - seed.y, p.z - seed.z);
      const double len = glm::length(dir);
      if (!(len > 0.0))
        continue;
      const glm::dvec3 normal = dir / len;
      const glm::dvec3 midpoint = (seed + glm::dvec3(p.x, p.y, p.z)) * 0.5;
      planes.emplace_back(normal, glm::dot(normal, midpoint));
    }

    double maxPlaneDot = 0.0;
    double offsetAtMaxDot = 0.0;
    for (size_t i = 0; i < planes.size(); ++i) {
      for (size_t j = i + 1; j < planes.size(); ++j) {
        const double dot =
            std::fabs(glm::dot(planes[i].first, planes[j].first));
        if (dot > maxPlaneDot) {
          maxPlaneDot = dot;
          offsetAtMaxDot = std::fabs(planes[i].second - planes[j].second);
        }
      }
    }

    const glm::dvec3 boundaryGap(
        std::min(seed.x - double(domainMin.x), double(domainMax.x) - seed.x),
        std::min(seed.y - double(domainMin.y), double(domainMax.y) - seed.y),
        std::min(seed.z - double(domainMin.z), double(domainMax.z) - seed.z));

    std::cerr << "[RVD] base failure cell=" << cell
              << " status=" << rvdStatusName(statuses[cell]) << " nearestDist="
              << std::sqrt(nearest.empty() ? 0.0 : nearest.front().first)
              << " minNeighborPairDist=" << minPairDist
              << " maxPlaneDot=" << maxPlaneDot
              << " offsetAtMaxDot=" << offsetAtMaxDot << " boundaryGap=("
              << boundaryGap.x << "," << boundaryGap.y << "," << boundaryGap.z
              << ")"
              << " nominalCellSize=" << input.nominalCellSize << std::endl;
  }
}

static void buildDomainPlanes(const glm::vec3 &domainMin,
                              const glm::vec3 &domainMax,
                              const cudaGeometry::DeviceVoxelGrid &grid,
                              double4 canonicalPlanes[6],
                              std::array<glm::dvec4, 6> &outwardPlanes) {

  const float3 minCorner = canonicalFromWorld(
      grid, make_float3(domainMin.x, domainMin.y, domainMin.z));
  const float3 maxCorner = canonicalFromWorld(
      grid, make_float3(domainMax.x, domainMax.y, domainMax.z));

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

bool prepareBaseDomain(const RVDInput &input, BaseCellDomain &baseDomain) {
  if (!input.seedSpatialIndex || input.seedSpatialIndex->seedCount() == 0 ||
      !input.seeds || input.seeds->empty() || !input.domainCorners) {
    return false;
  }

  baseDomain.count = uint32_t(input.seeds->size());
  baseDomain.domainMin = (*input.domainCorners)[0];
  baseDomain.domainMax = (*input.domainCorners)[0];
  for (const auto &corner : *input.domainCorners) {
    baseDomain.domainMin = glm::min(baseDomain.domainMin, corner);
    baseDomain.domainMax = glm::max(baseDomain.domainMax, corner);
  }
  const glm::vec3 domainSize = baseDomain.domainMax - baseDomain.domainMin;
  const float maxDim =
      std::max(domainSize.x, std::max(domainSize.y, domainSize.z));
  if (!(maxDim > 1e-6f) || !std::isfinite(maxDim)) {
    std::cerr << "[RVD] invalid domain extent" << std::endl;
    return false;
  }

  baseDomain.baseScale = cudaGeometry::CanonicalDomainSize / maxDim;
  baseDomain.invBaseScale = 1.0 / double(baseDomain.baseScale);
  baseDomain.invBaseVolumeScale = baseDomain.invBaseScale *
                                  baseDomain.invBaseScale *
                                  baseDomain.invBaseScale;
  baseDomain.invBaseMomentScale =
      baseDomain.invBaseVolumeScale * baseDomain.invBaseScale;
  baseDomain.invBaseAreaScale =
      baseDomain.invBaseScale * baseDomain.invBaseScale;
  baseDomain.nominalCellSize = input.nominalCellSize;

  baseDomain.baseGrid.gridMin = make_float3(
      baseDomain.domainMin.x, baseDomain.domainMin.y, baseDomain.domainMin.z);
  baseDomain.baseGrid.scale = baseDomain.baseScale;

  buildDomainPlanes(baseDomain.domainMin, baseDomain.domainMax,
                    baseDomain.baseGrid, baseDomain.domainPlanes,
                    baseDomain.outwardDomainPlanes);

  return true;
}

bool buildBaseCells(const RVDInput &input, const RVDConfig &config,
                    BaseCellDomain &baseDomain, cudaStream_t stream) {

  cudaUtils::CudaBuffer<RVDStatus> baseStatuses;
  cudaUtils::CudaBuffer<double4> devBaseMoments;
  if (!devBaseMoments.allocate(baseDomain.count) ||
      !baseStatuses.allocate(baseDomain.count) ||
      !baseDomain.devBaseStates.allocate(baseDomain.count))
    return false;

  BaseDeviceInput baseInput{};
  baseInput.knn = input.seedSpatialIndex->deviceKnn();
  baseInput.grid = baseDomain.baseGrid;
  for (int i = 0; i < 6; ++i)
    baseInput.domainPlanes[i] = baseDomain.domainPlanes[i];

  BaseDeviceOutput baseOutput{};
  baseOutput.moments = devBaseMoments.get();
  baseOutput.statuses = baseStatuses.get();

  bool ok =
      launchChunks(baseDomain.count, config.baseChunkSize, stream,
                   [&](uint32_t begin, uint32_t chunkCount) {
                     constexpr uint32_t BlockSize = 32;
                     const uint32_t blocks =
                         (chunkCount + BlockSize - 1) / BlockSize;
                     baseKernel<<<blocks, BlockSize, 0, stream>>>(
                         baseInput, baseOutput, baseDomain.devBaseStates.get(),
                         begin, chunkCount);
                   });
  if (!ok) {
    std::cerr << "[RVD] base unrestricted kernel phase failed" << std::endl;
    return false;
  }

  std::vector<RVDStatus> downloadedBaseStatuses(baseDomain.count);
  if (!baseStatuses.downloadAsync(downloadedBaseStatuses.data(),
                                  baseDomain.count, stream) ||
      !cudaUtils::cudaCheck(cudaStreamSynchronize(stream),
                            "synchronize initial base status download"))
    return false;

  std::vector<uint32_t> overflowQueue;
  for (uint32_t cell = 0; cell < baseDomain.count; ++cell) {
    const RVDStatus status = downloadedBaseStatuses[cell];
    if (status == RVDStatus::VertexOverflow ||
        status == RVDStatus::PlaneOverflow ||
        status == RVDStatus::InconsistentBoundary) {
      overflowQueue.push_back(cell);
    }
  }

  if (!overflowQueue.empty()) {
    const uint32_t overflowCount = static_cast<uint32_t>(overflowQueue.size());
    std::vector<int32_t> hostOverflowSlots(baseDomain.count, -1);
    for (uint32_t slot = 0; slot < overflowCount; ++slot) {
      hostOverflowSlots[overflowQueue[slot]] = static_cast<int32_t>(slot);
    }
    if (!baseDomain.devOverflowStates.allocate(overflowCount) ||
        !baseDomain.devOverflowSlots.allocate(baseDomain.count) ||
        !baseDomain.devOverflowSlots.uploadAsync(hostOverflowSlots.data(),
                                                 baseDomain.count, stream)) {
      return false;
    }
    baseDomain.overflowCount = overflowCount;

    cudaUtils::CudaBuffer<uint32_t> devOverflowCells;
    if (!devOverflowCells.allocate(overflowCount) ||
        !devOverflowCells.uploadAsync(overflowQueue.data(), overflowCount,
                                      stream)) {
      return false;
    }

    const bool retried =
        launchChunks(overflowCount, config.restrictedChunkSize, stream,
                     [&](uint32_t begin, uint32_t chunkCount) {
                       constexpr uint32_t BlockSize = 32;
                       const uint32_t blocks =
                           (chunkCount + BlockSize - 1) / BlockSize;
                       overflowBaseKernel<<<blocks, BlockSize, 0, stream>>>(
                           baseInput, baseOutput, devOverflowCells.get(), begin,
                           chunkCount, baseDomain.devOverflowSlots.get(),
                           baseDomain.devOverflowStates.get());
                     });
    if (!retried) {
      std::cerr << "[RVD] overflow base retry pass failed" << std::endl;
      return false;
    }

    if (!baseStatuses.downloadAsync(downloadedBaseStatuses.data(),
                                    baseDomain.count, stream) ||
        !cudaUtils::cudaCheck(cudaStreamSynchronize(stream),
                              "synchronize overflow retry status")) {
      return false;
    }
  }

  uint32_t baseFailures = 0;
  for (uint32_t cell = 0; cell < baseDomain.count; ++cell) {
    const RVDStatus status = downloadedBaseStatuses[cell];
    if (status != RVDStatus::Success) {
      std::cerr << "[RVD] base pass failed cell=" << cell
                << " status=" << rvdStatusName(status) << std::endl;
      ++baseFailures;
    }
  }
  if (baseFailures > 0) {
    reportBaseFailureDiagnostics(input, downloadedBaseStatuses);
    return false;
  }

  cudaUtils::CudaBuffer<uint32_t> devCellPlaneCounts;
  if (!devCellPlaneCounts.allocate(baseDomain.count + 1) ||
      !baseDomain.devCellPlaneOffsets.allocate(baseDomain.count + 1)) {
    return false;
  }

  {
    constexpr uint32_t BlockSize = 256;
    const uint32_t blocks = (baseDomain.count + 1 + BlockSize - 1) / BlockSize;
    countBaseCellPlanesKernel<<<blocks, BlockSize, 0, stream>>>(
        baseDomain.deviceView(), baseDomain.count, devCellPlaneCounts.get());
    if (!cudaUtils::cudaCheck(cudaGetLastError(),
                              "launch countBaseCellPlanesKernel"))
      return false;
  }

  size_t scanBytes = 0;
  if (cub::DeviceScan::ExclusiveSum(
          nullptr, scanBytes, devCellPlaneCounts.get(),
          baseDomain.devCellPlaneOffsets.get(), int(baseDomain.count + 1),
          stream) != cudaSuccess)
    return false;

  cudaUtils::CudaBuffer<uint8_t> scanScratch;
  if (!scanScratch.allocate(scanBytes))
    return false;

  if (cub::DeviceScan::ExclusiveSum(
          scanScratch.get(), scanBytes, devCellPlaneCounts.get(),
          baseDomain.devCellPlaneOffsets.get(), int(baseDomain.count + 1),
          stream) != cudaSuccess)
    return false;

  if (cudaMemcpyAsync(&baseDomain.totalPlanes,
                      baseDomain.devCellPlaneOffsets.get() + baseDomain.count,
                      sizeof(uint32_t), cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      !cudaUtils::cudaCheck(cudaStreamSynchronize(stream),
                            "readback totalPlanes")) {
    return false;
  }

  baseDomain.devBaseMoments = std::move(devBaseMoments);

  return true;
}

} // namespace voronoi
