#include "RVDFaceCoverage.cuh"
#include "RVDFaceGeometry.cuh"
#include "RVDFaceOverlay.cuh"
#include "RVDFacePartition.hpp"
#include "RVDHostUtils.cuh"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace voronoi {

namespace {

constexpr uint32_t OverlayBlockSize = 64;
constexpr uint32_t FastBlockSize = 256;
constexpr uint32_t OverlayBatchSize = 4096;

const char *facePartitionErrorName(uint32_t code) {
  switch (code) {
  case InvalidInput: return "InvalidInput";
  case InvalidFacePolygon: return "InvalidFacePolygon";
  case TooManyRegions: return "TooManyRegions";
  case AmbiguousOwnership: return "AmbiguousOwnership";
  case OverlappingModels: return "OverlappingModels";
  case AreaConservation: return "AreaConservation";
  case MomentConservation: return "MomentConservation";
  case PassMismatch: return "PassMismatch";
  case CoverageAreaConservation: return "CoverageAreaConservation";
  case BaseAreaConservation: return "BaseAreaConservation";
  case BaseMomentConservation: return "BaseMomentConservation";
  case UnresolvedRegion: return "UnresolvedRegion";
  case UnresolvedCoverageArea: return "UnresolvedCoverageArea";
  case UnresolvedCoverageMoment: return "UnresolvedCoverageMoment";
  case TooManySegments: return "TooManySegments";
  case EmptyFacePartition: return "EmptyFacePartition";
  default: return "Unknown";
  }
}

__device__ inline void reportFacePartitionError(
    RvdPackedError *packedError,
    uint32_t code,
    uint32_t faceId) {
  rvdReportError(packedError, code, faceId);
}

struct FacePartitionInput {
  const BaseFace *baseFaces = nullptr;
  const uint32_t *faceVertexOffsets = nullptr;
  const double3 *faceVertices = nullptr;
  const BaseFaceMeasure *baseMeasures = nullptr;
  const cudaGeometry::DeviceModelGeometryView *models = nullptr;
  uint32_t modelCount = 0;
  const double *const *modelPlaneAreas = nullptr;
  const double *const *modelPlaneAbsSums = nullptr;
  const double3 *const *modelPlaneMoments = nullptr;
  const FaceBoundarySegment *const *modelSegments = nullptr;
  const uint32_t *const *modelSegmentOffsets = nullptr;
  DeviceBaseCells baseCells{};
  const uint32_t *cellPlaneOffsets = nullptr;
  double invBaseAreaScale = 1.0;
};

struct FacePartitionOutput {
  uint32_t *regionCounts = nullptr;
  const uint32_t *regionOffsets = nullptr;
  FaceRegion *regions = nullptr;
  uint32_t *overlayFaceIds = nullptr;
  uint32_t *overlayCount = nullptr;
  uint32_t *overlaySlot = nullptr;
};

__device__ inline uint32_t partitionFace(
    uint32_t faceId,
    const FacePartitionInput &input,
    FaceOverlayWorkspace *overlayWork,
    FaceRegion *fastGroups,
    uint32_t &fastGroupCount,
    bool allowOverlay,
    bool &handled) {
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

  const BaseFace &baseFace = input.baseFaces[faceId];
  const BaseFaceMeasure base = input.baseMeasures[faceId];

  FacePolygon boundary{};
  double3 origin;
  double3 u;
  double3 v;
  double3 normal;
  double3 bboxMin;
  double3 bboxMax;
  double scale;
  if (!faceSetupPolygon(
          faceId, input.faceVertexOffsets, input.faceVertices, base, boundary,
          origin, u, v, normal, scale, bboxMin, bboxMax)) {
    return InvalidFacePolygon;
  }

  FaceSideCoverage sideA{};
  FaceSideCoverage sideB{};
  const uint32_t coverageError = faceCollectSideCoverage(
      baseFace, base, input.models, input.modelCount, input.modelPlaneAreas,
      input.modelPlaneAbsSums, input.modelPlaneMoments, input.baseCells,
      input.cellPlaneOffsets, input.invBaseAreaScale, sideA, sideB);
  if (coverageError) return coverageError;

  const uint32_t regionError = faceBuildCoverageRegions(
      base, boundary, origin, u, v, normal, scale, sideA, sideB, groups,
      groupCount, handled);
  if (regionError) return regionError;

  if (handled) {
    return faceValidateConservation(
        groups, groupCount, boundary, origin, u, v, scale, base);
  }
  if (!allowOverlay || overlayWork == nullptr) return FacePartitionSuccess;

  uint32_t boundaryCount = 0;
  const uint32_t segmentError = faceCollectRestrictedSegments(
      baseFace, boundary, origin, u, v, scale, input.modelSegments,
      input.modelSegmentOffsets, input.modelCount, boundaryCount, *overlayWork);
  if (segmentError) return segmentError;

  const uint32_t overlayError = faceBuildOverlay(
      faceId, base, boundary, origin, u, v, normal, scale, input.models,
      input.modelCount, sideA, sideB, boundaryCount, *overlayWork);
  if (overlayError) return overlayError;

  return faceValidateConservation(
      overlayWork->groups, overlayWork->groupCount, boundary, origin, u, v,
      scale, base);
}

__global__ void buildFastFaceRegionsKernel(
    uint32_t faceCount,
    FacePartitionInput input,
    FacePartitionOutput output,
    bool appendOverlay,
    bool emit,
    RvdPackedError *packedError) {
  const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
  if (faceId >= faceCount) return;
  if (emit && output.overlaySlot[faceId] != InvalidId) return;

  FaceRegion groups[2];
  uint32_t groupCount = 0;
  bool handled = false;
  const uint32_t result = partitionFace(faceId, input, nullptr, groups, groupCount, false, handled);
  if (result) {
    reportFacePartitionError(packedError, result, faceId);
    return;
  }

  if (!handled) {
    if (!appendOverlay) {
      reportFacePartitionError(packedError, PassMismatch, faceId);
      return;
    }
    const uint32_t slot = atomicAdd(output.overlayCount, 1u);
    output.overlayFaceIds[slot] = faceId;
    output.overlaySlot[faceId] = slot;
    output.regionCounts[faceId] = 0;
    return;
  }

  if (appendOverlay) output.overlaySlot[faceId] = InvalidId;

  if (!emit) {
    if (groupCount == 0) {
      reportFacePartitionError(packedError, EmptyFacePartition, faceId);
      return;
    }
    output.regionCounts[faceId] = groupCount;
    return;
  }

  const uint32_t destination = output.regionOffsets[faceId];
  if (destination + groupCount != output.regionOffsets[faceId + 1]) {
    reportFacePartitionError(packedError, PassMismatch, faceId);
    return;
  }
  for (uint32_t i = 0; i < groupCount; ++i) {
    output.regions[destination + i] = groups[i];
  }
}

__global__ void buildOverlayFaceRegionsKernel(
    uint32_t overlayStart,
    uint32_t overlayEnd,
    FacePartitionInput input,
    FacePartitionOutput output,
    FaceOverlayWorkspace *workspaces,
    bool emit,
    RvdPackedError *packedError) {
  const uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t slot = overlayStart + local;
  if (slot >= overlayEnd) return;

  const uint32_t faceId = output.overlayFaceIds[slot];
  auto &workspace = workspaces[local];
  bool handled = false;
  const uint32_t result = partitionFace(
      faceId, input, &workspace, workspace.groups, workspace.groupCount,
      true, handled);
  if (result) {
    reportFacePartitionError(packedError, result, faceId);
    return;
  }

  if (!emit) {
    if (workspace.groupCount == 0) {
      reportFacePartitionError(packedError, EmptyFacePartition, faceId);
      return;
    }
    output.regionCounts[faceId] = workspace.groupCount;
    return;
  }

  const uint32_t destination = output.regionOffsets[faceId];
  if (destination + workspace.groupCount != output.regionOffsets[faceId + 1]) {
    reportFacePartitionError(packedError, PassMismatch, faceId);
    return;
  }
  for (uint32_t i = 0; i < workspace.groupCount; ++i) {
    output.regions[destination + i] = workspace.groups[i];
  }
}

struct FacePartitionBuffers {
  cudaUtils::CudaBuffer<uint32_t> regionCounts;
  cudaUtils::CudaBuffer<uint32_t> overlayFaceIds;
  cudaUtils::CudaBuffer<uint32_t> overlaySlot;
  cudaUtils::CudaBuffer<uint32_t> overlayCount;
  cudaUtils::CudaBuffer<RvdPackedError> packedError;
  cudaUtils::CudaBuffer<FaceOverlayWorkspace> overlayWorkspaces;
};

bool finishPartitionPass(
    const cudaUtils::CudaBuffer<RvdPackedError> &packedError,
    cudaStream_t stream) {
  RvdPackedError hostError = 0;
  if (!rvdHost::readPackedError(packedError, stream, hostError)) return false;
  if (hostError == 0) return true;

  std::cerr << "[RVD-FacePartition] "
            << facePartitionErrorName(rvdErrorCode(hostError))
            << " face=" << rvdErrorId(hostError) << std::endl;
  return false;
}

size_t surfaceWorkspaceCount(uint32_t overlayCount) {
  return std::min(overlayCount, OverlayBatchSize);
}

bool runFastPass(
    uint32_t faceCount,
    BaseFaceTopology &faceTopology,
    const FacePartitionInput &input,
    FacePartitionBuffers &buffers,
    cudaStream_t stream,
    bool appendOverlay,
    bool emit) {
  buildFastFaceRegionsKernel<<<
      rvdHost::blockCount(faceCount, FastBlockSize),
      FastBlockSize, 0, stream>>>(
      faceCount,
      input,
      FacePartitionOutput{
          buffers.regionCounts.get(),
          emit ? faceTopology.devRegionOffsets.get() : nullptr,
          emit ? faceTopology.devRegions.get() : nullptr,
          buffers.overlayFaceIds.get(),
          buffers.overlayCount.get(),
          buffers.overlaySlot.get()},
      appendOverlay,
      emit,
      buffers.packedError.get());

  if (!rvdHost::checkLaunch("launch buildFastFaceRegionsKernel")) return false;
  return finishPartitionPass(buffers.packedError, stream);
}

bool runOverlayPass(
    uint32_t overlayCount,
    BaseFaceTopology &faceTopology,
    const FacePartitionInput &input,
    FacePartitionBuffers &buffers,
    cudaStream_t stream,
    bool emit) {
  if (overlayCount == 0) return true;

  const uint32_t batchSize = std::min(overlayCount, OverlayBatchSize);
  for (uint32_t start = 0; start < overlayCount; start += batchSize) {
    const uint32_t count = std::min(batchSize, overlayCount - start);
    buildOverlayFaceRegionsKernel<<<
        rvdHost::blockCount(count, OverlayBlockSize),
        OverlayBlockSize, 0, stream>>>(
        start,
        start + count,
        input,
        FacePartitionOutput{
            buffers.regionCounts.get(),
            emit ? faceTopology.devRegionOffsets.get() : nullptr,
            emit ? faceTopology.devRegions.get() : nullptr,
            buffers.overlayFaceIds.get(),
            buffers.overlayCount.get(),
            buffers.overlaySlot.get()},
        buffers.overlayWorkspaces.get(),
        emit,
        buffers.packedError.get());

    if (!rvdHost::checkLaunch("launch buildOverlayFaceRegionsKernel")) {
      return false;
    }
  }
  return finishPartitionPass(buffers.packedError, stream);
}

struct ModelFacePointers {
  cudaUtils::CudaBuffer<cudaGeometry::DeviceModelGeometryView> views;
  cudaUtils::CudaBuffer<const double *> planeAreas;
  cudaUtils::CudaBuffer<const double *> planeAbsSums;
  cudaUtils::CudaBuffer<const double3 *> planeMoments;
  cudaUtils::CudaBuffer<const FaceBoundarySegment *> segments;
  cudaUtils::CudaBuffer<const uint32_t *> segmentOffsets;
};

bool uploadModelFacePointers(
    const std::vector<cudaGeometry::DeviceModelGeometryView> &modelViews,
    const std::vector<ModelRestriction> &models,
    cudaStream_t stream,
    ModelFacePointers &pointers) {
  std::vector<const double *> hostPlaneAreas;
  std::vector<const double *> hostPlaneAbsSums;
  std::vector<const double3 *> hostPlaneMoments;
  std::vector<const FaceBoundarySegment *> hostSegments;
  std::vector<const uint32_t *> hostSegmentOffsets;

  hostPlaneAreas.reserve(models.size());
  hostPlaneAbsSums.reserve(models.size());
  hostPlaneMoments.reserve(models.size());
  hostSegments.reserve(models.size());
  hostSegmentOffsets.reserve(models.size());
  for (const ModelRestriction &model : models) {
    hostPlaneAreas.push_back(model.devRestrictedPlaneAreas.get());
    hostPlaneAbsSums.push_back(model.devRestrictedPlaneAbsSums.get());
    hostPlaneMoments.push_back(model.devRestrictedPlaneMoments.get());
    hostSegments.push_back(model.devFaceSegments.get());
    hostSegmentOffsets.push_back(model.devFaceSegmentOffsets.get());
  }

  if (!pointers.views.allocate(modelViews.size()) ||
      !pointers.views.uploadAsync(
          modelViews.data(), modelViews.size(), stream)) {
    return false;
  }
  if (models.empty()) return true;

  const uint32_t modelCount = uint32_t(models.size());
  return pointers.planeAreas.allocate(modelCount) &&
         pointers.planeAbsSums.allocate(modelCount) &&
         pointers.planeMoments.allocate(modelCount) &&
         pointers.segments.allocate(modelCount) &&
         pointers.segmentOffsets.allocate(modelCount) &&
         pointers.planeAreas.uploadAsync(hostPlaneAreas.data(), modelCount, stream) &&
         pointers.planeAbsSums.uploadAsync(hostPlaneAbsSums.data(), modelCount, stream) &&
         pointers.planeMoments.uploadAsync(hostPlaneMoments.data(), modelCount, stream) &&
         pointers.segments.uploadAsync(hostSegments.data(), modelCount, stream) &&
         pointers.segmentOffsets.uploadAsync(
             hostSegmentOffsets.data(), modelCount, stream);
}

FacePartitionInput makePartitionInput(
    const BaseCellDomain &baseDomain,
    BaseFaceTopology &faceTopology,
    const ModelFacePointers &pointers,
    uint32_t modelCount) {
  FacePartitionInput input{};
  input.baseFaces = faceTopology.devBaseFaces.get();
  input.faceVertexOffsets = faceTopology.devFaceVertexOffsets.get();
  input.faceVertices = faceTopology.devFaceVertices.get();
  input.baseMeasures = faceTopology.devMeasures.get();
  input.models = pointers.views.get();
  input.modelCount = modelCount;
  input.modelPlaneAreas = pointers.planeAreas.get();
  input.modelPlaneAbsSums = pointers.planeAbsSums.get();
  input.modelPlaneMoments = pointers.planeMoments.get();
  input.modelSegments = pointers.segments.get();
  input.modelSegmentOffsets = pointers.segmentOffsets.get();
  input.baseCells = baseDomain.deviceView();
  input.cellPlaneOffsets = baseDomain.devCellPlaneOffsets.get();
  input.invBaseAreaScale = baseDomain.invBaseAreaScale;
  return input;
}

bool allocatePartitionBuffers(
    uint32_t faceCount,
    BaseFaceTopology &faceTopology,
    cudaStream_t stream,
    FacePartitionBuffers &buffers) {
  if (!buffers.regionCounts.allocate(size_t(faceCount) + 1) ||
      !faceTopology.devRegionOffsets.allocate(size_t(faceCount) + 1) ||
      !buffers.overlayFaceIds.allocate(faceCount) ||
      !buffers.overlaySlot.allocate(faceCount) ||
      !buffers.overlayCount.allocate(1) ||
      !buffers.packedError.allocate(1)) {
    return false;
  }

  return rvdHost::zero(
             buffers.packedError.get(), sizeof(RvdPackedError), stream,
             "clear face partition error") &&
         rvdHost::zero(
             buffers.overlayCount.get(), sizeof(uint32_t), stream,
             "clear face overlay count") &&
         rvdHost::zero(
             buffers.regionCounts.get() + faceCount, sizeof(uint32_t), stream,
             "clear face count tail");
}

} // namespace

bool buildFacePartitionsGPU(
    const BaseCellDomain &baseDomain,
    BaseFaceTopology &faceTopology,
    const std::vector<cudaGeometry::DeviceModelGeometryView> &modelViews,
    const std::vector<ModelRestriction> &models,
    cudaStream_t stream) {
  const uint32_t faceCount = faceTopology.totalBaseFaces;
  if (modelViews.size() != models.size()) return false;

  if (faceCount == 0) {
    faceTopology.devRegionOffsets.reset();
    faceTopology.devRegions.reset();
    return true;
  }

  if (faceTopology.devFaceVertexOffsets.size() < size_t(faceCount) + 1 ||
      faceTopology.devFaceVertices.get() == nullptr ||
      faceTopology.devMeasures.size() < faceCount) {
    return false;
  }

  ModelFacePointers pointers;
  if (!uploadModelFacePointers(modelViews, models, stream, pointers)) return false;

  FacePartitionBuffers buffers;
  if (!allocatePartitionBuffers(faceCount, faceTopology, stream, buffers)) {
    return false;
  }

  const FacePartitionInput input = makePartitionInput(
      baseDomain, faceTopology, pointers, uint32_t(modelViews.size()));

  if (!runFastPass(
          faceCount, faceTopology, input, buffers, stream, true, false)) {
    return false;
  }

  uint32_t overlayCount = 0;
  if (!rvdHost::readAt(
          overlayCount,
          buffers.overlayCount.get(),
          0,
          stream,
          "read overlay face count")) {
    return false;
  }

  if (overlayCount > 0 &&
      !buffers.overlayWorkspaces.allocate(surfaceWorkspaceCount(overlayCount))) {
    return false;
  }

  if (!runOverlayPass(
          overlayCount, faceTopology, input, buffers, stream, false)) {
    return false;
  }

  cudaUtils::CudaBuffer<uint8_t> scanScratch;
  if (!rvdHost::exclusiveSum(
          buffers.regionCounts.get(),
          faceTopology.devRegionOffsets.get(),
          int(faceCount + 1),
          scanScratch,
          stream)) {
    return false;
  }

  uint32_t totalRegions = 0;
  if (!rvdHost::readAt(
          totalRegions,
          faceTopology.devRegionOffsets.get(),
          faceCount,
          stream,
          "read face region count")) {
    return false;
  }

  if (totalRegions == 0 || !faceTopology.devRegions.allocate(totalRegions)) {
    return false;
  }

  if (!runFastPass(
          faceCount, faceTopology, input, buffers, stream, false, true)) {
    return false;
  }
  return runOverlayPass(
      overlayCount, faceTopology, input, buffers, stream, true);
}

} // namespace voronoi
