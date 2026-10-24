#include "RVDFaceCoverage.cuh"
#include "RVDFaceGeometry.cuh"
#include "RVDFaceOverlay.cuh"
#include "RVDFacePartition.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cub/cub.cuh>
#include <iostream>
#include <limits>
#include <vector>

namespace voronoi {

static const char *facePartitionErrorName(uint32_t code) {
  switch (code) {
  case InvalidInput:
    return "InvalidInput";
  case InvalidFacePolygon:
    return "InvalidFacePolygon";
  case InvalidModelBvh:
    return "InvalidModelBvh";
  case TooManyLines:
    return "TooManyLines";
  case TooManyVertices:
    return "TooManyVertices";
  case TooManyRegions:
    return "TooManyRegions";
  case InvalidIntersection:
    return "InvalidIntersection";
  case AmbiguousOwnership:
    return "AmbiguousOwnership";
  case OverlappingModels:
    return "OverlappingModels";
  case AreaConservation:
    return "AreaConservation";
  case MomentConservation:
    return "MomentConservation";
  case PassMismatch:
    return "PassMismatch";
  case ClipAreaConservation:
    return "ClipAreaConservation";
  case CoverageAreaConservation:
    return "CoverageAreaConservation";
  case BaseAreaConservation:
    return "BaseAreaConservation";
  case BaseMomentConservation:
    return "BaseMomentConservation";
  case UnresolvedRegion:
    return "UnresolvedRegion";
  case UnresolvedCoverageArea:
    return "UnresolvedCoverageArea";
  case UnresolvedCoverageMoment:
    return "UnresolvedCoverageMoment";
  case TooManySegments:
    return "TooManySegments";
  case EmptyFacePartition:
    return "EmptyFacePartition";
  default:
    return "Unknown";
  }
}

__device__ inline void facePartitionReport(unsigned long long *error,
                                           uint32_t code, uint32_t faceId) {
  atomicCAS(error, 0ULL,
            (static_cast<unsigned long long>(code) << 32) |
                static_cast<unsigned long long>(faceId));
}

__device__ inline uint32_t facePartitionOne(
    uint32_t faceId, const BaseFace *baseFaces, const uint32_t *polygonOffsets,
    const double3 *polygonVertices, const BaseFaceMeasure *faceMeasures,
    const cudaGeometry::DeviceModelGeometryView *models, uint32_t modelCount,
    const double *const *modelPlaneAreas,
    const double *const *modelPlaneAbsSums,
    const double3 *const *modelPlaneMoments,
    const FaceBoundarySegment *const *modelSegments,
    const uint32_t *const *modelSegmentOffsets, DeviceBaseCells baseCells,
    const uint32_t *cellPlaneOffsets, double invBaseAreaScale,
    FaceOverlayWorkspace *overlayWork, FaceRegion *fastGroups,
    uint32_t &fastGroupCount, bool allowOverlay, bool &handled) {
  FaceRegion *groups = fastGroups;
  uint32_t *groupCountPtr = &fastGroupCount;
  if (overlayWork != nullptr) {
    overlayWork->segmentCount = overlayWork->pointCount =
        overlayWork->edgeCount = overlayWork->cycleCount =
            overlayWork->groupCount = 0;
    groups = overlayWork->groups;
    groupCountPtr = &overlayWork->groupCount;
  }
  uint32_t &groupCount = *groupCountPtr;
  groupCount = 0;
  handled = false;
  const BaseFace &baseFace = baseFaces[faceId];
  const BaseFaceMeasure base = faceMeasures[faceId];
  FacePolygon boundary{};
  double3 origin, u, v, n, bboxMin, bboxMax;
  double scale;
  if (!faceSetupPolygon(faceId, polygonOffsets, polygonVertices, base, boundary,
                        origin, u, v, n, scale, bboxMin, bboxMax))
    return InvalidFacePolygon;

  FaceSideCoverage sideA{}, sideB{};
  uint32_t error = faceCollectSideCoverage(
      baseFace, base, models, modelCount, modelPlaneAreas, modelPlaneAbsSums,
      modelPlaneMoments, baseCells, cellPlaneOffsets, invBaseAreaScale, sideA,
      sideB);
  if (error)
    return error;

  error = faceBuildCoverageRegions(base, boundary, origin, u, v, n, scale,
                                   sideA, sideB, groups, groupCount, handled);
  if (error)
    return error;
  if (handled) {
    return faceValidateConservation(groups, groupCount, boundary, origin, u, v,
                                    scale, base);
  }
  if (!allowOverlay || overlayWork == nullptr)
    return FacePartitionSuccess;

  uint32_t boundaryCount = 0;
  uint32_t e = faceCollectRestrictedSegments(
      baseFace, boundary, origin, u, v, scale, modelSegments,
      modelSegmentOffsets, modelCount, boundaryCount, *overlayWork);
  if (e)
    return e;
  e = faceBuildOverlay(faceId, base, boundary, origin, u, v, n, scale, models,
                       modelCount, sideA, sideB, boundaryCount, *overlayWork);
  if (e)
    return e;
  return faceValidateConservation(overlayWork->groups,
                                  overlayWork->groupCount, boundary, origin, u,
                                  v, scale, base);
}

static __global__ void buildFastFaceRegionsKernel(
    uint32_t faceCount, const BaseFace *baseFaces,
    const uint32_t *faceVertexOffsets, const double3 *faceVertices,
    const BaseFaceMeasure *baseMeasures,
    const cudaGeometry::DeviceModelGeometryView *modelViews,
    uint32_t modelCount, const double *const *modelPlaneAreas,
    const double *const *modelPlaneAbsSums,
    const double3 *const *modelPlaneMoments,
    const FaceBoundarySegment *const *modelSegments,
    const uint32_t *const *modelSegmentOffsets, DeviceBaseCells baseCells,
    const uint32_t *cellPlaneOffsets, double invBaseAreaScale,
    uint32_t *regionCounts, const uint32_t *regionOffsets, FaceRegion *regions,
    uint32_t *overlayFaceIds, uint32_t *overlayCount, uint32_t *overlaySlot,
    bool appendOverlay, bool emit, unsigned long long *packedError) {
  const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
  if (faceId >= faceCount)
    return;
  if (emit && overlaySlot[faceId] != InvalidId)
    return;
  FaceRegion groups[2];
  uint32_t groupCount = 0;
  bool handled = false;
  const uint32_t result = facePartitionOne(
      faceId, baseFaces, faceVertexOffsets, faceVertices, baseMeasures,
      modelViews, modelCount, modelPlaneAreas, modelPlaneAbsSums,
      modelPlaneMoments, modelSegments, modelSegmentOffsets, baseCells,
      cellPlaneOffsets, invBaseAreaScale, nullptr, groups, groupCount, false,
      handled);
  if (result) {
    facePartitionReport(packedError, result, faceId);
    return;
  }
  if (!handled) {
    if (!appendOverlay) {
      facePartitionReport(packedError, PassMismatch, faceId);
      return;
    }
    const uint32_t slot = atomicAdd(overlayCount, 1u);
    overlayFaceIds[slot] = faceId;
    overlaySlot[faceId] = slot;
    regionCounts[faceId] = 0;
    return;
  }
  if (appendOverlay)
    overlaySlot[faceId] = InvalidId;
  if (!emit) {
    if (groupCount == 0) {
      facePartitionReport(packedError, EmptyFacePartition, faceId);
      return;
    }
    regionCounts[faceId] = groupCount;
    return;
  }
  const uint32_t dst = regionOffsets[faceId];
  if (dst + groupCount != regionOffsets[faceId + 1]) {
    facePartitionReport(packedError, PassMismatch, faceId);
    return;
  }
  for (uint32_t i = 0; i < groupCount; ++i)
    regions[dst + i] = groups[i];
}

static __global__ void buildOverlayFaceRegionsKernel(
    uint32_t overlayStart, uint32_t overlayEnd, const uint32_t *overlayFaceIds,
    const BaseFace *baseFaces, const uint32_t *faceVertexOffsets,
    const double3 *faceVertices, const BaseFaceMeasure *baseMeasures,
    const cudaGeometry::DeviceModelGeometryView *modelViews,
    uint32_t modelCount, const double *const *modelPlaneAreas,
    const double *const *modelPlaneAbsSums,
    const double3 *const *modelPlaneMoments,
    const FaceBoundarySegment *const *modelSegments,
    const uint32_t *const *modelSegmentOffsets, DeviceBaseCells baseCells,
    const uint32_t *cellPlaneOffsets, double invBaseAreaScale,
    FaceOverlayWorkspace *workspaces, uint32_t *regionCounts,
    const uint32_t *regionOffsets, FaceRegion *regions,
    bool emit, unsigned long long *packedError) {
  const uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t slot = overlayStart + local;
  if (slot >= overlayEnd)
    return;
  const uint32_t faceId = overlayFaceIds[slot];
  auto &work = workspaces[local];
  bool handled = false;
  const uint32_t result = facePartitionOne(
      faceId, baseFaces, faceVertexOffsets, faceVertices, baseMeasures,
      modelViews, modelCount, modelPlaneAreas, modelPlaneAbsSums,
      modelPlaneMoments, modelSegments, modelSegmentOffsets, baseCells,
      cellPlaneOffsets, invBaseAreaScale, &work, work.groups, work.groupCount,
      true, handled);
  if (result) {
    facePartitionReport(packedError, result, faceId);
    return;
  }
  if (!emit) {
    if (work.groupCount == 0) {
      facePartitionReport(packedError, EmptyFacePartition, faceId);
      return;
    }
    regionCounts[faceId] = work.groupCount;
    return;
  }
  const uint32_t dst = regionOffsets[faceId];
  if (dst + work.groupCount != regionOffsets[faceId + 1]) {
    facePartitionReport(packedError, PassMismatch, faceId);
    return;
  }
  for (uint32_t i = 0; i < work.groupCount; ++i)
    regions[dst + i] = work.groups[i];
}

static bool facePartitionFinishPass(
    cudaUtils::CudaBuffer<unsigned long long> &devError,
    cudaStream_t stream) {
  unsigned long long err = 0;
  if (!cudaUtils::cudaCheck(cudaMemcpyAsync(&err, devError.get(), sizeof(err),
                                            cudaMemcpyDeviceToHost, stream),
                            "read partition error") ||
      !cudaUtils::cudaCheck(cudaStreamSynchronize(stream),
                            "synchronize partition"))
    return false;
  if (err) {
    std::cerr << "[RVD-FacePartition] "
              << facePartitionErrorName(uint32_t(err >> 32))
              << " face=" << uint32_t(err) << std::endl;
    return false;
  }
  return true;
}

static bool facePartitionRunFast(
    uint32_t count, BaseFaceTopology &faceTopology,
    const cudaUtils::CudaBuffer<cudaGeometry::DeviceModelGeometryView> &devModels,
    uint32_t modelCount,
    const cudaUtils::CudaBuffer<const double *> &devPlaneAreas,
    const cudaUtils::CudaBuffer<const double *> &devPlaneAbsSums,
    const cudaUtils::CudaBuffer<const double3 *> &devPlaneMoments,
    const cudaUtils::CudaBuffer<const FaceBoundarySegment *> &devModelSegments,
    const cudaUtils::CudaBuffer<const uint32_t *> &devModelSegmentOffsets,
    const BaseCellDomain &baseDomain,
    cudaUtils::CudaBuffer<uint32_t> &devCounts,
    cudaUtils::CudaBuffer<uint32_t> &devOverlayIds,
    cudaUtils::CudaBuffer<uint32_t> &devOverlaySlot,
    cudaUtils::CudaBuffer<uint32_t> &devOverlayCount,
    cudaUtils::CudaBuffer<unsigned long long> &devError,
    cudaStream_t stream, uint32_t fastBlocks, uint32_t fastThreads,
    bool appendOverlay, bool emit) {
  buildFastFaceRegionsKernel<<<fastBlocks, fastThreads, 0, stream>>>(
      count, faceTopology.devBaseFaces.get(),
      faceTopology.devFaceVertexOffsets.get(),
      faceTopology.devFaceVertices.get(), faceTopology.devMeasures.get(),
      devModels.get(), modelCount, devPlaneAreas.get(),
      devPlaneAbsSums.get(), devPlaneMoments.get(), devModelSegments.get(),
      devModelSegmentOffsets.get(), baseDomain.deviceView(),
      baseDomain.devCellPlaneOffsets.get(), baseDomain.invBaseAreaScale,
      devCounts.get(),
      emit ? faceTopology.devRegionOffsets.get() : nullptr,
      emit ? faceTopology.devRegions.get() : nullptr, devOverlayIds.get(),
      devOverlayCount.get(), devOverlaySlot.get(), appendOverlay, emit,
      devError.get());
  if (!cudaUtils::cudaCheck(cudaGetLastError(), "buildFastFaceRegionsKernel"))
    return false;
  return facePartitionFinishPass(devError, stream);
}

static bool facePartitionRunOverlay(
    uint32_t overlayCount, BaseFaceTopology &faceTopology,
    const cudaUtils::CudaBuffer<cudaGeometry::DeviceModelGeometryView> &devModels,
    uint32_t modelCount,
    const cudaUtils::CudaBuffer<const double *> &devPlaneAreas,
    const cudaUtils::CudaBuffer<const double *> &devPlaneAbsSums,
    const cudaUtils::CudaBuffer<const double3 *> &devPlaneMoments,
    const cudaUtils::CudaBuffer<const FaceBoundarySegment *> &devModelSegments,
    const cudaUtils::CudaBuffer<const uint32_t *> &devModelSegmentOffsets,
    const BaseCellDomain &baseDomain,
    cudaUtils::CudaBuffer<FaceOverlayWorkspace> &overlayWorkspace,
    cudaUtils::CudaBuffer<uint32_t> &devCounts,
    cudaUtils::CudaBuffer<uint32_t> &devOverlayIds,
    cudaUtils::CudaBuffer<unsigned long long> &devError,
    cudaStream_t stream, uint32_t threads, uint32_t overlayBatch, bool emit) {
  if (overlayCount == 0)
    return true;
  const uint32_t chunk = std::min(overlayCount, overlayBatch);
  for (uint32_t start = 0; start < overlayCount; start += chunk) {
    const uint32_t n = std::min(chunk, overlayCount - start);
    const uint32_t blocks = (n + threads - 1) / threads;
    buildOverlayFaceRegionsKernel<<<blocks, threads, 0, stream>>>(
        start, start + n, devOverlayIds.get(),
        faceTopology.devBaseFaces.get(),
        faceTopology.devFaceVertexOffsets.get(),
        faceTopology.devFaceVertices.get(), faceTopology.devMeasures.get(),
        devModels.get(), modelCount, devPlaneAreas.get(),
        devPlaneAbsSums.get(), devPlaneMoments.get(), devModelSegments.get(),
        devModelSegmentOffsets.get(), baseDomain.deviceView(),
        baseDomain.devCellPlaneOffsets.get(), baseDomain.invBaseAreaScale,
        overlayWorkspace.get(), devCounts.get(),
        emit ? faceTopology.devRegionOffsets.get() : nullptr,
        emit ? faceTopology.devRegions.get() : nullptr,
        emit, devError.get());
    if (!cudaUtils::cudaCheck(cudaGetLastError(),
                              "buildOverlayFaceRegionsKernel"))
      return false;
  }
  return facePartitionFinishPass(devError, stream);
}

bool buildFacePartitionsGPU(
    const BaseCellDomain &baseDomain, BaseFaceTopology &faceTopology,
    const std::vector<cudaGeometry::DeviceModelGeometryView> &modelViews,
    const std::vector<ModelRestriction> &models, cudaStream_t stream) {
  const uint32_t count = faceTopology.totalBaseFaces;
  if (modelViews.size() != models.size())
    return false;
  if (!count) {
    faceTopology.devRegionOffsets.reset();
    faceTopology.devRegions.reset();
    return true;
  }
  if (faceTopology.devFaceVertexOffsets.size() < size_t(count) + 1 ||
      !faceTopology.devFaceVertices.get() ||
      faceTopology.devMeasures.size() < count)
    return false;

  std::vector<const double *> modelPlaneAreaPointers;
  std::vector<const double *> modelPlaneAbsSumPointers;
  std::vector<const double3 *> modelPlaneMomentPointers;
  std::vector<const FaceBoundarySegment *> modelSegmentPointers;
  std::vector<const uint32_t *> modelSegmentOffsetPointers;
  modelPlaneAreaPointers.reserve(models.size());
  modelPlaneAbsSumPointers.reserve(models.size());
  modelPlaneMomentPointers.reserve(models.size());
  modelSegmentPointers.reserve(models.size());
  modelSegmentOffsetPointers.reserve(models.size());
  for (const ModelRestriction &model : models) {
    modelPlaneAreaPointers.push_back(model.devRestrictedPlaneAreas.get());
    modelPlaneAbsSumPointers.push_back(model.devRestrictedPlaneAbsSums.get());
    modelPlaneMomentPointers.push_back(model.devRestrictedPlaneMoments.get());
    modelSegmentPointers.push_back(model.devFaceSegments.get());
    modelSegmentOffsetPointers.push_back(model.devFaceSegmentOffsets.get());
  }

  cudaUtils::CudaBuffer<cudaGeometry::DeviceModelGeometryView> devModels;
  cudaUtils::CudaBuffer<const double *> devPlaneAreas;
  cudaUtils::CudaBuffer<const double *> devPlaneAbsSums;
  cudaUtils::CudaBuffer<const double3 *> devPlaneMoments;
  cudaUtils::CudaBuffer<const FaceBoundarySegment *> devModelSegments;
  cudaUtils::CudaBuffer<const uint32_t *> devModelSegmentOffsets;
  cudaUtils::CudaBuffer<FaceOverlayWorkspace> overlayWorkspace;
  cudaUtils::CudaBuffer<uint32_t> devCounts;
  cudaUtils::CudaBuffer<uint32_t> devOverlayIds;
  cudaUtils::CudaBuffer<uint32_t> devOverlaySlot;
  cudaUtils::CudaBuffer<uint32_t> devOverlayCount;
  cudaUtils::CudaBuffer<unsigned long long> devError;
  if (!devModels.allocate(modelViews.size()) ||
      !devCounts.allocate(size_t(count) + 1) ||
      !faceTopology.devRegionOffsets.allocate(size_t(count) + 1) ||
      !devOverlayIds.allocate(count) || !devOverlaySlot.allocate(count) ||
      !devOverlayCount.allocate(1) || !devError.allocate(1))
    return false;
  if (!modelPlaneAreaPointers.empty() &&
      (!devPlaneAreas.allocate(modelPlaneAreaPointers.size()) ||
       !devPlaneAbsSums.allocate(modelPlaneAbsSumPointers.size()) ||
       !devPlaneMoments.allocate(modelPlaneMomentPointers.size()) ||
       !devPlaneAreas.uploadAsync(
           modelPlaneAreaPointers.data(),
           static_cast<uint32_t>(modelPlaneAreaPointers.size()), stream) ||
       !devPlaneAbsSums.uploadAsync(
           modelPlaneAbsSumPointers.data(),
           static_cast<uint32_t>(modelPlaneAbsSumPointers.size()), stream) ||
       !devPlaneMoments.uploadAsync(
           modelPlaneMomentPointers.data(),
           static_cast<uint32_t>(modelPlaneMomentPointers.size()), stream) ||
       !devModelSegments.allocate(modelSegmentPointers.size()) ||
       !devModelSegmentOffsets.allocate(modelSegmentOffsetPointers.size()) ||
       !devModelSegments.uploadAsync(
           modelSegmentPointers.data(),
           static_cast<uint32_t>(modelSegmentPointers.size()), stream) ||
       !devModelSegmentOffsets.uploadAsync(
           modelSegmentOffsetPointers.data(),
           static_cast<uint32_t>(modelSegmentOffsetPointers.size()), stream)))
    return false;
  if (!devModels.uploadAsync(modelViews.data(), modelViews.size(), stream) ||
      !cudaUtils::cudaCheck(cudaMemsetAsync(devError.get(), 0,
                                            sizeof(unsigned long long), stream),
                            "clear face partition error") ||
      !cudaUtils::cudaCheck(
          cudaMemsetAsync(devOverlayCount.get(), 0, sizeof(uint32_t), stream),
          "clear face overlay count") ||
      !cudaUtils::cudaCheck(
          cudaMemsetAsync(devCounts.get() + count, 0, sizeof(uint32_t), stream),
          "clear face count tail"))
    return false;

  constexpr uint32_t threads = 64;
  constexpr uint32_t fastThreads = 256;
  const uint32_t fastBlocks = (count + fastThreads - 1) / fastThreads;
  constexpr uint32_t OverlayBatch = 4096;

  if (!facePartitionRunFast(
          count, faceTopology, devModels, uint32_t(modelViews.size()),
          devPlaneAreas, devPlaneAbsSums, devPlaneMoments, devModelSegments,
          devModelSegmentOffsets, baseDomain, devCounts, devOverlayIds,
          devOverlaySlot, devOverlayCount, devError, stream, fastBlocks,
          fastThreads, true, false))
    return false;

  uint32_t overlayCount = 0;
  if (!cudaUtils::cudaCheck(
          cudaMemcpyAsync(&overlayCount, devOverlayCount.get(),
                          sizeof(overlayCount), cudaMemcpyDeviceToHost, stream),
          "read overlay face count") ||
      !cudaUtils::cudaCheck(cudaStreamSynchronize(stream),
                            "sync overlay face count"))
    return false;

  if (overlayCount != 0) {
    const uint32_t overlayBatch = std::min(overlayCount, OverlayBatch);
    if (!overlayWorkspace.allocate(overlayBatch))
      return false;
  }

  if (!facePartitionRunOverlay(
          overlayCount, faceTopology, devModels, uint32_t(modelViews.size()),
          devPlaneAreas, devPlaneAbsSums, devPlaneMoments, devModelSegments,
          devModelSegmentOffsets, baseDomain, overlayWorkspace, devCounts,
          devOverlayIds, devError, stream, threads, OverlayBatch, false))
      return false;

  size_t scratchBytes = 0;
  if (!cudaUtils::cudaCheck(
          cub::DeviceScan::ExclusiveSum(nullptr, scratchBytes, devCounts.get(),
              faceTopology.devRegionOffsets.get(),
              int(count + 1), stream),
      "face CSR scan size"))
      return false;

  cudaUtils::CudaBuffer<uint8_t> scanScratch;
  if (!scanScratch.allocate(scratchBytes)) return false;
  if (!cudaUtils::cudaCheck(
          cub::DeviceScan::ExclusiveSum(
              scanScratch.get(), scratchBytes, devCounts.get(),
              faceTopology.devRegionOffsets.get(), int(count + 1), stream),
          "face CSR scan"))
    return false;
  uint32_t total = 0;
  if (!cudaUtils::cudaCheck(
          cudaMemcpyAsync(&total, faceTopology.devRegionOffsets.get() + count,
              sizeof(total), cudaMemcpyDeviceToHost, stream),
      "read face region count") ||
      !cudaUtils::cudaCheck(cudaStreamSynchronize(stream),
                            "sync face region count"))
    return false;
  if (total == 0 || !faceTopology.devRegions.allocate(total)) return false;
  if (!facePartitionRunFast(
          count, faceTopology, devModels, uint32_t(modelViews.size()),
          devPlaneAreas, devPlaneAbsSums, devPlaneMoments, devModelSegments,
          devModelSegmentOffsets, baseDomain, devCounts, devOverlayIds,
          devOverlaySlot, devOverlayCount, devError, stream, fastBlocks,
          fastThreads, false, true))
    return false;
  if (!facePartitionRunOverlay(
          overlayCount, faceTopology, devModels, uint32_t(modelViews.size()),
          devPlaneAreas, devPlaneAbsSums, devPlaneMoments, devModelSegments,
          devModelSegmentOffsets, baseDomain, overlayWorkspace, devCounts,
          devOverlayIds, devError, stream, threads, OverlayBatch, true))
    return false;
  return true;
}

} // namespace voronoi

