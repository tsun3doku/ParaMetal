#include "RVDFragments.hpp"
#include "RVDHostUtils.cuh"
#include "ConvexCell.cuh"

#include <cfloat>
#include <cmath>
#include <iostream>

namespace voronoi {

namespace {

constexpr uint32_t FragmentBlockSize = 256;

__device__ inline uint32_t candidateIndex(
    uint32_t cell,
    uint32_t modelCount,
    uint32_t model) {
  return cell * (modelCount + 1) + model;
}

__device__ inline void reportFragmentError(
    RvdPackedError *packedError,
    RVDFragmentError error,
    uint32_t id) {
  rvdReportError(packedError, uint32_t(error), id);
}

__global__ void buildFragmentsKernel(
    const double4 *baseMoments,
    const double4 *const *modelMoments,
    const uint32_t *runtimeModelIds,
    const float4 *seeds,
    uint32_t cellCount,
    uint32_t modelCount,
    float nominalCellSize,
    double invBaseVolumeScale,
    double invBaseMomentScale,
    Fragment *candidateFragments,
    RvdPackedError *packedError) {
  const uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= cellCount) return;

  const double4 rawBaseMoment = baseMoments[cell];
  const double baseVolume = (rawBaseMoment.w / 6.0) * invBaseVolumeScale;
  const glm::dvec3 baseMoment = (glm::dvec3(rawBaseMoment.x, rawBaseMoment.y, rawBaseMoment.z) / 6.0) * invBaseMomentScale;
  const float4 seed = seeds[cell];
  const glm::dvec3 seedPosition(seed.x, seed.y, seed.z);
  const double h = double(nominalCellSize);
  const double nominalVolume = h * h * h;
  const double volumeClosureTolerance = fmax(nominalVolume * 1e-6, baseVolume * 1e-4);

  double totalModelVolume = 0.0;
  glm::dvec3 totalModelMoment(0.0);

  for (uint32_t g = 0; g < modelCount; ++g) {
    const uint32_t candidate = candidateIndex(cell, modelCount, g);
    const double4 rawModelMoment = modelMoments[g][cell];
    const double modelVolume = rawModelMoment.w * invBaseVolumeScale;
    const glm::dvec3 modelMoment = glm::dvec3(rawModelMoment.x, rawModelMoment.y, rawModelMoment.z) * invBaseMomentScale;

    Fragment fragment{};
    fragment.cellId = cell;
    fragment.ownerModelId = runtimeModelIds[g];
    fragment.volume = modelVolume;

    if (modelVolume > 0.0 && isfinite(modelVolume)) {
      const glm::dvec3 centroid = seedPosition + modelMoment / modelVolume;
      if (isfinite(centroid.x) && isfinite(centroid.y) && isfinite(centroid.z)) {
        fragment.centroid = centroid;
      } else {
        fragment.volume = -1.0;
        fragment.centroid = seedPosition;
      }
    } else {
      fragment.centroid = seedPosition;
    }

    candidateFragments[candidate] = fragment;

    if (modelVolume > 0.0 && isfinite(modelVolume)) {
      totalModelVolume += modelVolume;
      totalModelMoment += modelMoment;
    }
  }

  const uint32_t fluidCandidate = candidateIndex(cell, modelCount, modelCount);
  const double rawUnownedVolume = baseVolume - totalModelVolume;
  if (rawUnownedVolume < -volumeClosureTolerance) {
    reportFragmentError(packedError, RVDFragmentError::BaseVolumeOvercoverage, cell);
    return;
  }

  const double unownedVolume = rawUnownedVolume <= volumeClosureTolerance ? 0.0 : rawUnownedVolume;

  Fragment fluid{};
  fluid.cellId = cell;
  fluid.ownerModelId = InvalidId;
  fluid.volume = unownedVolume;

  if (unownedVolume > 0.0 && isfinite(unownedVolume)) {
    const glm::dvec3 remainderMoment = baseMoment - totalModelMoment;
    const glm::dvec3 centroid = seedPosition + remainderMoment / unownedVolume;
    if (isfinite(centroid.x) && isfinite(centroid.y) && isfinite(centroid.z)) {
      fluid.centroid = centroid;
    } else {
      fluid.volume = -1.0;
      fluid.centroid = seedPosition;
    }
  } else {
    fluid.centroid = seedPosition;
  }
  candidateFragments[fluidCandidate] = fluid;
}

__device__ inline glm::dvec3 computeOutwardNormal(
    int tag,
    uint32_t cell,
    const float4 *seeds,
    const glm::dvec4 *outwardDomainPlanes) {
  if (tag < 0) {
    const int domainFace = rvdDevice::domainFaceFromTag(tag);
    if (domainFace >= 0 && domainFace < 6) {
      const glm::dvec4 plane = outwardDomainPlanes[domainFace];
      return glm::dvec3(plane.x, plane.y, plane.z);
    }
    return glm::dvec3(0.0);
  }

  const float4 self = seeds[cell];
  const float4 other = seeds[uint32_t(tag)];
  const glm::dvec3 direction(
      double(other.x) - double(self.x),
      double(other.y) - double(self.y),
      double(other.z) - double(self.z));
  const double length = glm::length(direction);
  return length > 1e-12 ? direction / length : glm::dvec3(0.0);
}

__device__ inline uint32_t fragmentForMaterial(
    uint32_t cell,
    uint32_t material,
    uint32_t modelCount,
    const uint32_t *runtimeModelIds) {
  if (material == InvalidId) return candidateIndex(cell, modelCount, modelCount);
  if (runtimeModelIds == nullptr) return InvalidId;

  for (uint32_t g = 0; g < modelCount; ++g) {
    if (runtimeModelIds[g] == material) {
      return candidateIndex(cell, modelCount, g);
    }
  }
  return InvalidId;
}

__global__ void emitBaseFragmentFacesKernel(
    const BaseFace *baseFaces,
    uint32_t baseFaceCount,
    const BaseFaceMeasure *baseFaceMeasures,
    const uint32_t *regionOffsets,
    const FaceRegion *regions,
    const uint32_t *runtimeModelIds,
    DeviceBaseCells baseCells,
    const uint32_t *cellPlaneOffsets,
    const float4 *seeds,
    const glm::dvec4 *outwardDomainPlanes,
    uint32_t modelCount,
    FragmentFace *fragmentFaces,
    RvdPackedError *packedError) {
  const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
  if (faceId >= baseFaceCount) return;

  const BaseFace &face = baseFaces[faceId];
  const BaseFaceMeasure &measure = baseFaceMeasures[faceId];
  if (measure.valid == 0u) {
    reportFragmentError(packedError, RVDFragmentError::InvalidBaseFaceMeasure, faceId);
    return;
  }

  const uint32_t cellA = face.cellA;
  const uint32_t cellB = face.cellB;

  const BaseCellArrays cellAState = resolveBaseCell(baseCells, cellA);
  const uint32_t localPlaneA = face.slotA - cellPlaneOffsets[cellA];
  const int tagA = cellAState.planeNeighborIds[localPlaneA];
  const glm::dvec3 normalA = computeOutwardNormal(tagA, cellA, seeds, outwardDomainPlanes);

  glm::dvec3 normalB(0.0);
  if (cellB != InvalidId) {
    const BaseCellArrays cellBState = resolveBaseCell(baseCells, cellB);
    const uint32_t localPlaneB = face.slotB - cellPlaneOffsets[cellB];
    const int tagB = cellBState.planeNeighborIds[localPlaneB];
    normalB = computeOutwardNormal(tagB, cellB, seeds, outwardDomainPlanes);
  }

  const uint32_t regionBegin = regionOffsets[faceId];
  const uint32_t regionEnd = regionOffsets[faceId + 1];
  if (regionBegin == regionEnd) {
    reportFragmentError(packedError, RVDFragmentError::MissingFaceRegions, faceId);
    return;
  }

  for (uint32_t r = regionBegin; r < regionEnd; ++r) {
    const FaceRegion &region = regions[r];
    const uint32_t fragmentA =
        fragmentForMaterial(cellA, region.materialA, modelCount, runtimeModelIds);
    const uint32_t fragmentB = cellB != InvalidId
        ? fragmentForMaterial(cellB, region.materialB, modelCount, runtimeModelIds)
        : InvalidId;

    if (fragmentA == InvalidId ||
        (cellB != InvalidId && fragmentB == InvalidId)) {
      reportFragmentError(packedError, RVDFragmentError::UnmappedModelMaterial, faceId);
      return;
    }

    FragmentFace fragmentFace{};
    fragmentFace.fragmentA = fragmentA;
    fragmentFace.fragmentB = fragmentB;
    fragmentFace.areaA = region.measure.area;
    fragmentFace.areaVectorA = normalA * fragmentFace.areaA;
    fragmentFace.momentA = glm::dvec3(
        region.measure.moment.x,
        region.measure.moment.y,
        region.measure.moment.z);

    if (cellB != InvalidId) {
      fragmentFace.areaB = region.measure.area;
      fragmentFace.areaVectorB = normalB * fragmentFace.areaB;
      fragmentFace.momentB = fragmentFace.momentA;
      fragmentFace.type = FaceType::Shared;
      fragmentFace.domainFace = InvalidId;
    } else {
      fragmentFace.type = FaceType::DomainBoundary;
      fragmentFace.domainFace = face.domainFace;
    }
    fragmentFaces[r] = fragmentFace;
  }
}

__global__ void countAndEmitSurfaceFragmentFacesKernel(
    const RestrictedSurface *surfaces,
    const RestrictedSurfaceGeometry *surfaceGeometry,
    const double4 *surfaceExposure,
    const Fragment *candidateFragments,
    uint32_t surfaceCount,
    uint32_t modelIndex,
    uint32_t modelCount,
    uint32_t cellCount,
    double invBaseScale,
    glm::dvec3 domainMin,
    uint32_t *surfaceFaceCounts,
    const uint32_t *surfaceFaceOffsets,
    FragmentFace *fragmentFaces,
    uint32_t baseFaceCount,
    uint32_t *contactFaceBySurface,
    uint32_t surfaceBase,
    bool emitPass,
    RvdPackedError *packedError) {
  const uint32_t surfaceIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (surfaceIndex >= surfaceCount) return;

  const auto skipSurface = [&]() {
    if (!emitPass) surfaceFaceCounts[surfaceIndex] = 0;
  };

  const RestrictedSurface &surface = surfaces[surfaceIndex];
  if (surface.vertexCount < 3) {
    skipSurface();
    return;
  }

  const uint32_t cell = surface.cellId;
  if (cell >= cellCount) {
    skipSurface();
    return;
  }

  const RestrictedSurfaceGeometry &geometry = surfaceGeometry[surfaceIndex];
  const double inverseScaleSquared = invBaseScale * invBaseScale;
  const double totalArea = double(geometry.centroidArea.w) * inverseScaleSquared;
  if (!(totalArea > 0.0) || !isfinite(totalArea)) {
    skipSurface();
    return;
  }

  glm::dvec3 normal(geometry.normal.x, geometry.normal.y, geometry.normal.z);
  const double normalLength = glm::length(normal);
  if (normalLength <= 1e-14 || !isfinite(normalLength)) {
    skipSurface();
    return;
  }
  normal /= normalLength;

  const glm::dvec3 canonicalCentroid(
      geometry.centroidArea.x,
      geometry.centroidArea.y,
      geometry.centroidArea.z);
  const glm::dvec3 worldCentroid = domainMin + canonicalCentroid * invBaseScale;
  const glm::dvec3 totalMoment = worldCentroid * totalArea;

  const double4 rawExposure = surfaceExposure[surfaceIndex];
  double exposedArea = rawExposure.w;
  glm::dvec3 exposedMoment(rawExposure.x, rawExposure.y, rawExposure.z);

  const double exposureEpsilon = 64.0 * DBL_EPSILON * totalArea;
  if (exposedArea > totalArea && exposedArea <= totalArea + exposureEpsilon) {
    exposedArea = totalArea;
    exposedMoment = totalMoment;
  } else if (exposedArea < 0.0 && exposedArea >= -exposureEpsilon) {
    exposedArea = 0.0;
    exposedMoment = glm::dvec3(0.0);
  }

  if (exposedArea < 0.0 || exposedArea > totalArea) {
    skipSurface();
    reportFragmentError(packedError, RVDFragmentError::InvalidSurfaceExposure, cell);
    return;
  }

  if (exposedArea <= exposureEpsilon) {
    exposedArea = 0.0;
    exposedMoment = glm::dvec3(0.0);
  } else if (totalArea - exposedArea <= exposureEpsilon) {
    exposedArea = totalArea;
    exposedMoment = totalMoment;
  }

  const uint32_t solidFragment = candidateIndex(cell, modelCount, modelIndex);
  const uint32_t fluidFragment = candidateIndex(cell, modelCount, modelCount);

  const Fragment fluidCandidate = candidateFragments[fluidFragment];
  const bool hasFluid = fluidCandidate.volume > 0.0 && isfinite(fluidCandidate.volume);

  uint32_t localFaceCount = 0;
  const uint32_t destinationBase = emitPass ? surfaceFaceOffsets[surfaceIndex] : 0;

  if (exposedArea > 0.0) {
    if (emitPass) {
      FragmentFace face{};
      face.fragmentA = solidFragment;
      face.fragmentB = hasFluid ? fluidFragment : InvalidId;
      face.areaA = exposedArea;
      face.areaVectorA = normal * exposedArea;
      face.momentA = exposedMoment;
      if (hasFluid) {
        face.areaB = exposedArea;
        face.areaVectorB = -normal * exposedArea;
        face.momentB = exposedMoment;
      }
      face.type = FaceType::Shared;
      face.domainFace = InvalidId;
      fragmentFaces[destinationBase + localFaceCount] = face;
    }
    ++localFaceCount;
  }

  const double contactArea = totalArea - exposedArea;
  const glm::dvec3 contactMoment = totalMoment - exposedMoment;
  if (contactArea > 0.0) {
    if (emitPass) {
      FragmentFace solidContactFace{};
      solidContactFace.fragmentA = solidFragment;
      solidContactFace.fragmentB = InvalidId;
      solidContactFace.areaA = contactArea;
      solidContactFace.areaVectorA = normal * contactArea;
      solidContactFace.momentA = contactMoment;
      solidContactFace.type = FaceType::ContactSide;
      solidContactFace.domainFace = InvalidId;
      fragmentFaces[destinationBase + localFaceCount] = solidContactFace;

      if (contactFaceBySurface != nullptr) {
        contactFaceBySurface[surfaceBase + surfaceIndex] =
            baseFaceCount + destinationBase + localFaceCount;
      }
    }
    ++localFaceCount;

    if (hasFluid) {
      if (emitPass) {
        FragmentFace fluidClosureFace{};
        fluidClosureFace.fragmentA = fluidFragment;
        fluidClosureFace.fragmentB = InvalidId;
        fluidClosureFace.areaA = contactArea;
        fluidClosureFace.areaVectorA = -normal * contactArea;
        fluidClosureFace.momentA = contactMoment;
        fluidClosureFace.type = FaceType::ContactSide;
        fluidClosureFace.domainFace = InvalidId;
        fragmentFaces[destinationBase + localFaceCount] = fluidClosureFace;
      }
      ++localFaceCount;
    }
  }

  if (!emitPass) {
    surfaceFaceCounts[surfaceIndex] = localFaceCount;
  }
}

__global__ void countCandidateFacesKernel(
    const FragmentFace *faces,
    uint32_t faceCount,
    uint32_t *candidateFaceCounts) {
  const uint32_t faceIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (faceIndex >= faceCount) return;

  const FragmentFace &face = faces[faceIndex];
  if (face.fragmentA != InvalidId) {
    atomicAdd(&candidateFaceCounts[face.fragmentA], 1u);
  }
  if (face.fragmentB != InvalidId) {
    atomicAdd(&candidateFaceCounts[face.fragmentB], 1u);
  }
}

__global__ void filterCandidatesKernel(
    const Fragment *candidates,
    const uint32_t *candidateFaceCounts,
    uint32_t candidateCount,
    uint32_t *admitFlags) {
  const uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidateCount) return;

  const Fragment &fragment = candidates[candidate];
  const bool keep = isfinite(fragment.volume) && fragment.volume > 0.0 && candidateFaceCounts[candidate] > 0u;
  admitFlags[candidate] = keep ? 1u : 0u;
}

__global__ void compactFragmentsKernel(
    const uint32_t *admitFlags,
    const uint32_t *admitOffsets,
    const Fragment *candidateFragments,
    uint32_t candidateCount,
    Fragment *fragments,
    uint32_t *candidateToGlobal) {
  const uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidateCount) return;

  if (admitFlags[candidate] == 1) {
    const uint32_t globalId = admitOffsets[candidate];
    fragments[globalId] = candidateFragments[candidate];
    candidateToGlobal[candidate] = globalId;
  } else {
    candidateToGlobal[candidate] = InvalidId;
  }
}

__global__ void remapFragmentFacesKernel(
    FragmentFace *faces,
    uint32_t faceCount,
    const uint32_t *candidateToGlobal) {
  const uint32_t faceIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (faceIndex >= faceCount) return;

  FragmentFace face = faces[faceIndex];
  if (face.fragmentA != InvalidId) {
    face.fragmentA = candidateToGlobal[face.fragmentA];
  }
  if (face.fragmentB != InvalidId) {
    face.fragmentB = candidateToGlobal[face.fragmentB];
  }
  faces[faceIndex] = face;
}

struct ModelDeviceReferences {
  cudaUtils::CudaBuffer<const double4 *> moments;
  cudaUtils::CudaBuffer<uint32_t> runtimeIds;
};

bool uploadModelReferences(
    const std::vector<ModelRestriction> &models,
    cudaStream_t stream,
    ModelDeviceReferences &references) {
  const uint32_t modelCount = uint32_t(models.size());
  if (modelCount == 0) return true;

  std::vector<const double4 *> hostMoments(modelCount);
  std::vector<uint32_t> hostModelIds(modelCount);
  for (uint32_t g = 0; g < modelCount; ++g) {
    hostMoments[g] = models[g].devMoments.get();
    hostModelIds[g] = models[g].runtimeModelId;
  }

  return references.moments.allocate(modelCount) &&
         references.moments.uploadAsync(hostMoments.data(), modelCount, stream) &&
         references.runtimeIds.allocate(modelCount) &&
         references.runtimeIds.uploadAsync(hostModelIds.data(), modelCount, stream);
}

bool buildCandidateFragments(
    const RVDInput &input,
    const BaseCellDomain &baseDomain,
    uint32_t modelCount,
    const float4 *seeds,
    const ModelDeviceReferences &references,
    cudaStream_t stream,
    cudaUtils::CudaBuffer<Fragment> &candidates,
    cudaUtils::CudaBuffer<RvdPackedError> &packedError) {
  const uint32_t cellCount = baseDomain.count;
  const uint32_t candidateCount = cellCount * (modelCount + 1);

  if (!candidates.allocate(candidateCount)) return false;

  buildFragmentsKernel<<<
      rvdHost::blockCount(cellCount, FragmentBlockSize),
      FragmentBlockSize, 0, stream>>>(
      baseDomain.devBaseMoments.get(),
      references.moments.get(),
      references.runtimeIds.get(),
      seeds,
      cellCount,
      modelCount,
      input.nominalCellSize,
      baseDomain.invBaseVolumeScale,
      baseDomain.invBaseMomentScale,
      candidates.get(),
      packedError.get());

  if (!rvdHost::checkLaunch("launch buildFragmentsKernel")) return false;

  RvdPackedError hostError = 0;
  if (!rvdHost::readPackedError(packedError, stream, hostError)) return false;
  if (hostError == 0) return true;

  std::cerr << "[RVD-Fragments-GPU] "
            << rvdFragmentErrorName(rvdErrorCode(hostError))
            << " (code=" << rvdErrorCode(hostError) << ")"
            << " detected on cell " << rvdErrorId(hostError) << std::endl;
  return false;
}

struct SurfaceFaceCounts {
  cudaUtils::CudaBuffer<uint32_t> counts;
  cudaUtils::CudaBuffer<uint32_t> offsets;
  uint32_t totalPatches = 0;
  uint32_t totalFaces = 0;
};

bool allocateSurfaceFaceCounts(
    uint32_t totalPatches,
    cudaStream_t stream,
    SurfaceFaceCounts &surfaceFaces) {
  surfaceFaces.totalPatches = totalPatches;
  if (totalPatches == 0) return true;

  if (!surfaceFaces.counts.allocate(size_t(totalPatches) + 1) ||
      !surfaceFaces.offsets.allocate(size_t(totalPatches) + 1)) {
    return false;
  }
  return rvdHost::zero(
      surfaceFaces.counts.get() + totalPatches,
      sizeof(uint32_t),
      stream,
      "clear surface face count tail");
}

void countSurfaceFragmentFaces(
    const std::vector<ModelRestriction> &models,
    const cudaUtils::CudaBuffer<double4> &surfaceExposure,
    const cudaUtils::CudaBuffer<Fragment> &candidates,
    uint32_t modelCount,
    uint32_t cellCount,
    const BaseCellDomain &baseDomain,
    SurfaceFaceCounts &surfaceFaces,
    cudaUtils::CudaBuffer<RvdPackedError> &packedError,
    cudaStream_t stream) {
  for (uint32_t g = 0; g < modelCount; ++g) {
    const ModelRestriction &model = models[g];
    if (model.totalPatches == 0) continue;

    countAndEmitSurfaceFragmentFacesKernel<<<
        rvdHost::blockCount(model.totalPatches, FragmentBlockSize),
        FragmentBlockSize, 0, stream>>>(
        model.devSurfaces.get(),
        model.devSurfaceGeometry.get(),
        surfaceExposure.get() + model.surfaceBase,
        candidates.get(),
        model.totalPatches,
        g,
        modelCount,
        cellCount,
        baseDomain.invBaseScale,
        glm::dvec3(baseDomain.domainMin),
        surfaceFaces.counts.get() + model.surfaceBase,
        surfaceFaces.offsets.get() + model.surfaceBase,
        nullptr,
        0,
        nullptr,
        model.surfaceBase,
        false,
        packedError.get());
  }
}

bool scanSurfaceFaceCounts(
    SurfaceFaceCounts &surfaceFaces,
    cudaStream_t stream) {
  if (surfaceFaces.totalPatches == 0) return true;

  cudaUtils::CudaBuffer<uint8_t> scanScratch;
  if (!rvdHost::exclusiveSum(
          surfaceFaces.counts.get(),
          surfaceFaces.offsets.get(),
          int(surfaceFaces.totalPatches + 1),
          scanScratch,
          stream)) {
    return false;
  }

  return rvdHost::readAt(
      surfaceFaces.totalFaces,
      surfaceFaces.offsets.get(),
      surfaceFaces.totalPatches,
      stream,
      "read total surface fragment faces");
}

bool readBaseFragmentFaceCount(
    const BaseFaceTopology &faceTopology,
    cudaStream_t stream,
    uint32_t &totalBaseFaces) {
  totalBaseFaces = 0;
  if (faceTopology.totalBaseFaces == 0) return true;

  return rvdHost::readAt(
      totalBaseFaces,
      faceTopology.devRegionOffsets.get(),
      faceTopology.totalBaseFaces,
      stream,
      "read total base fragment faces");
}

void emitSurfaceFragmentFaces(
    const std::vector<ModelRestriction> &models,
    const cudaUtils::CudaBuffer<double4> &surfaceExposure,
    const cudaUtils::CudaBuffer<Fragment> &candidates,
    uint32_t modelCount,
    uint32_t cellCount,
    const BaseCellDomain &baseDomain,
    SurfaceFaceCounts &surfaceFaces,
    cudaUtils::CudaBuffer<FragmentFace> &fragmentFaces,
    uint32_t totalBaseFaces,
    cudaUtils::CudaBuffer<uint32_t> &contactFaceBySurface,
    cudaUtils::CudaBuffer<RvdPackedError> &packedError,
    cudaStream_t stream) {
  for (uint32_t g = 0; g < modelCount; ++g) {
    const ModelRestriction &model = models[g];
    if (model.totalPatches == 0) continue;

    countAndEmitSurfaceFragmentFacesKernel<<<
        rvdHost::blockCount(model.totalPatches, FragmentBlockSize),
        FragmentBlockSize, 0, stream>>>(
        model.devSurfaces.get(),
        model.devSurfaceGeometry.get(),
        surfaceExposure.get() + model.surfaceBase,
        candidates.get(),
        model.totalPatches,
        g,
        modelCount,
        cellCount,
        baseDomain.invBaseScale,
        glm::dvec3(baseDomain.domainMin),
        surfaceFaces.counts.get() + model.surfaceBase,
        surfaceFaces.offsets.get() + model.surfaceBase,
        fragmentFaces.get() + totalBaseFaces,
        totalBaseFaces,
        contactFaceBySurface.get(),
        model.surfaceBase,
        true,
        packedError.get());
  }
}

struct FragmentCompaction {
  cudaUtils::CudaBuffer<uint32_t> candidateFaceCounts;
  cudaUtils::CudaBuffer<uint32_t> admitFlags;
  cudaUtils::CudaBuffer<uint32_t> admitOffsets;
  cudaUtils::CudaBuffer<uint32_t> candidateToGlobal;
  cudaUtils::CudaBuffer<Fragment> fragments;
  uint32_t totalFragments = 0;
};

bool compactFragmentCandidates(
    const cudaUtils::CudaBuffer<Fragment> &candidates,
    uint32_t candidateCount,
    cudaUtils::CudaBuffer<FragmentFace> &fragmentFaces,
    uint32_t faceCount,
    cudaStream_t stream,
    FragmentCompaction &compaction) {
  if (!compaction.candidateFaceCounts.allocate(candidateCount) ||
      !compaction.admitFlags.allocate(size_t(candidateCount) + 1) ||
      !compaction.admitOffsets.allocate(size_t(candidateCount) + 1) ||
      !compaction.candidateToGlobal.allocate(candidateCount)) {
    return false;
  }

  if (!rvdHost::zero(
          compaction.candidateFaceCounts.get(),
          candidateCount * sizeof(uint32_t),
          stream,
          "clear candidate face counts") ||
      !rvdHost::zero(
          compaction.admitFlags.get() + candidateCount,
          sizeof(uint32_t),
          stream,
          "clear candidate flag tail")) {
    return false;
  }

  const uint32_t candidateBlocks =
      rvdHost::blockCount(candidateCount, FragmentBlockSize);
  if (faceCount > 0) {
    countCandidateFacesKernel<<<
        rvdHost::blockCount(faceCount, FragmentBlockSize),
        FragmentBlockSize, 0, stream>>>(
        fragmentFaces.get(), faceCount, compaction.candidateFaceCounts.get());
    if (!rvdHost::checkLaunch("launch countCandidateFacesKernel")) return false;
  }

  filterCandidatesKernel<<<candidateBlocks, FragmentBlockSize, 0, stream>>>(
      candidates.get(),
      compaction.candidateFaceCounts.get(),
      candidateCount,
      compaction.admitFlags.get());
  if (!rvdHost::checkLaunch("launch filterCandidatesKernel")) return false;

  cudaUtils::CudaBuffer<uint8_t> scanScratch;
  if (!rvdHost::exclusiveSum(
          compaction.admitFlags.get(),
          compaction.admitOffsets.get(),
          int(candidateCount + 1),
          scanScratch,
          stream)) {
    return false;
  }

  if (!rvdHost::readAt(
          compaction.totalFragments,
          compaction.admitOffsets.get(),
          candidateCount,
          stream,
          "read total fragments")) {
    return false;
  }

  if (compaction.totalFragments > 0 &&
      !compaction.fragments.allocate(compaction.totalFragments)) {
    return false;
  }

  compactFragmentsKernel<<<candidateBlocks, FragmentBlockSize, 0, stream>>>(
      compaction.admitFlags.get(),
      compaction.admitOffsets.get(),
      candidates.get(),
      candidateCount,
      compaction.fragments.get(),
      compaction.candidateToGlobal.get());
  if (!rvdHost::checkLaunch("launch compactFragmentsKernel")) return false;

  if (faceCount > 0) {
    remapFragmentFacesKernel<<<
        rvdHost::blockCount(faceCount, FragmentBlockSize),
        FragmentBlockSize, 0, stream>>>(
        fragmentFaces.get(), faceCount, compaction.candidateToGlobal.get());
    if (!rvdHost::checkLaunch("launch remapFragmentFacesKernel")) return false;
  }
  return true;
}

bool downloadFragmentResults(
    const FragmentCompaction &compaction,
    uint32_t faceCount,
    cudaUtils::CudaBuffer<FragmentFace> &fragmentFaces,
    RVDResult &result,
    cudaStream_t stream) {
  result.fragments.resize(compaction.totalFragments);
  if (compaction.totalFragments > 0 &&
      !compaction.fragments.downloadAsync(
          result.fragments.data(), compaction.totalFragments, stream)) {
    return false;
  }

  result.fragmentFaces.resize(faceCount);
  if (faceCount > 0 &&
      !fragmentFaces.downloadAsync(result.fragmentFaces.data(), faceCount, stream)) {
    return false;
  }

  return rvdHost::synchronize(stream, "download fragment results");
}

void assignSeedToFragment(
    const RVDInput &input,
    RVDResult &result) {
  const uint32_t seedCount =
      input.seeds ? static_cast<uint32_t>(input.seeds->size()) : 0u;
  result.seedToFragment.assign(seedCount, InvalidId);

  for (uint32_t i = 0; i < result.fragments.size(); ++i) {
    const Fragment &fragment = result.fragments[i];
    if (fragment.ownerModelId == InvalidModelId &&
        fragment.cellId < seedCount) {
      result.seedToFragment[fragment.cellId] = i;
    }
  }
}

} // namespace

bool buildFragmentsAndFaces(
    const RVDInput &input,
    const BaseCellDomain &baseDomain,
    const BaseFaceTopology &faceTopology,
    const std::vector<ModelRestriction> &models,
    const cudaUtils::CudaBuffer<double4> &devSurfaceExposure,
    cudaUtils::CudaBuffer<uint32_t> &devContactFaceBySurface,
    uint32_t &totalBaseFragmentFaces,
    RVDResult &result,
    cudaStream_t stream) {
  totalBaseFragmentFaces = 0;

  const uint32_t cellCount = baseDomain.count;
  const uint32_t modelCount = uint32_t(models.size());
  const uint32_t candidateCount = cellCount * (modelCount + 1);
  if (cellCount == 0) return true;

  const float4 *seeds = input.seedSpatialIndex
      ? input.seedSpatialIndex->deviceKnn().seeds
      : nullptr;
  if (seeds == nullptr) return false;

  ModelDeviceReferences modelReferences;
  if (!uploadModelReferences(models, stream, modelReferences)) return false;

  cudaUtils::CudaBuffer<RvdPackedError> packedError;
  if (!packedError.allocate(1) ||
      !rvdHost::zero(
          packedError.get(), sizeof(RvdPackedError), stream,
          "clear fragment error")) {
    return false;
  }

  cudaUtils::CudaBuffer<Fragment> candidates;
  if (!buildCandidateFragments(
          input, baseDomain, modelCount, seeds, modelReferences, stream,
          candidates, packedError)) {
    return false;
  }

  cudaUtils::CudaBuffer<glm::dvec4> outwardPlanes;
  if (!outwardPlanes.allocate(6) ||
      !outwardPlanes.uploadAsync(
          baseDomain.outwardDomainPlanes.data(), 6, stream)) {
    return false;
  }

  uint32_t totalSurfacePatches = 0;
  for (const ModelRestriction &model : models) {
    totalSurfacePatches += model.totalPatches;
  }

  SurfaceFaceCounts surfaceFaces;
  if (!allocateSurfaceFaceCounts(totalSurfacePatches, stream, surfaceFaces)) {
    return false;
  }
  countSurfaceFragmentFaces(
      models, devSurfaceExposure, candidates, modelCount, cellCount,
      baseDomain, surfaceFaces, packedError, stream);
  if (!rvdHost::checkLaunch("launch surface face count pass")) return false;
  if (!scanSurfaceFaceCounts(surfaceFaces, stream)) return false;

  uint32_t totalBaseFaces = 0;
  if (!readBaseFragmentFaceCount(faceTopology, stream, totalBaseFaces)) {
    return false;
  }

  const uint32_t totalFaces = totalBaseFaces + surfaceFaces.totalFaces;
  totalBaseFragmentFaces = totalBaseFaces;

  cudaUtils::CudaBuffer<FragmentFace> fragmentFaces;
  if (!fragmentFaces.allocate(totalFaces)) return false;
  if (totalSurfacePatches > 0) {
    if (!devContactFaceBySurface.allocate(totalSurfacePatches) ||
        !rvdHost::fill(
            devContactFaceBySurface.get(), 0xFF,
            totalSurfacePatches * sizeof(uint32_t), stream,
            "clear contact face map")) {
      return false;
    }
  }

  if (totalBaseFaces > 0) {
    emitBaseFragmentFacesKernel<<<
        rvdHost::blockCount(faceTopology.totalBaseFaces, FragmentBlockSize),
        FragmentBlockSize, 0, stream>>>(
        faceTopology.devBaseFaces.get(),
        faceTopology.totalBaseFaces,
        faceTopology.devMeasures.get(),
        faceTopology.devRegionOffsets.get(),
        faceTopology.devRegions.get(),
        modelReferences.runtimeIds.get(),
        baseDomain.deviceView(),
        baseDomain.devCellPlaneOffsets.get(),
        seeds,
        outwardPlanes.get(),
        modelCount,
        fragmentFaces.get(),
        packedError.get());
    if (!rvdHost::checkLaunch("launch emitBaseFragmentFacesKernel")) return false;
  }

  if (surfaceFaces.totalFaces > 0) {
    emitSurfaceFragmentFaces(
        models, devSurfaceExposure, candidates, modelCount, cellCount,
        baseDomain, surfaceFaces, fragmentFaces, totalBaseFaces,
        devContactFaceBySurface, packedError, stream);
    if (!rvdHost::checkLaunch("launch surface face emit pass")) return false;
  }

  RvdPackedError hostError = 0;
  if (!rvdHost::readPackedError(packedError, stream, hostError)) return false;
  if (hostError != 0) {
    std::cerr << "[RVD-Faces-GPU] "
              << rvdFragmentErrorName(rvdErrorCode(hostError))
              << " (code=" << rvdErrorCode(hostError) << ")"
              << " on cell " << rvdErrorId(hostError) << std::endl;
    return false;
  }

  FragmentCompaction compaction;
  if (!compactFragmentCandidates(
          candidates, candidateCount, fragmentFaces, totalFaces, stream,
          compaction)) {
    return false;
  }

  if (!downloadFragmentResults(
          compaction, totalFaces, fragmentFaces, result, stream)) {
    return false;
  }
  assignSeedToFragment(input, result);
  return true;
}

} // namespace voronoi
