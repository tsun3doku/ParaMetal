#include "RVDFragments.cuh"
#include "RVDBaseFacePartition.cuh"
#include "RVDHostUtils.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

namespace voronoi {

namespace {

glm::dvec3 seedPosition(const RVDInput &input, uint32_t cell) {
  const glm::vec4 &seed = (*input.seeds)[cell];
  return glm::dvec3(seed.x, seed.y, seed.z);
}

glm::dvec3 worldFirstMoment(const glm::dvec3 &seed, double volume, const glm::dvec3 &relativeMoment) {
  return seed * volume + relativeMoment;
}

uint32_t materialSlot(
    uint32_t materialId,
    const std::vector<ModelRestriction> &models) {
  if (materialId == InvalidId) return uint32_t(models.size());
  for (uint32_t model = 0; model < models.size(); ++model) {
    if (models[model].runtimeModelId == materialId) return model;
  }
  return InvalidId;
}

bool finiteMoment(const glm::dvec3 &moment) {
  return std::isfinite(moment.x) && std::isfinite(moment.y) && std::isfinite(moment.z);
}

void compensatedAdd(double value, double &sum, double &compensation) {
  const double adjusted = value - compensation;
  const double next = sum + adjusted;
  compensation = (next - sum) - adjusted;
  sum = next;
}

void compensatedAdd(const glm::dvec3 &value, glm::dvec3 &sum, glm::dvec3 &compensation) {
  const glm::dvec3 adjusted = value - compensation;
  const glm::dvec3 next = sum + adjusted;
  compensation = (next - sum) - adjusted;
  sum = next;
}

} // namespace

struct LoopMeasure {
  glm::dvec3 areaVector{0.0};
  glm::dvec3 moment{0.0};
  glm::dvec3 boundsMin{std::numeric_limits<double>::max()};
  glm::dvec3 boundsMax{std::numeric_limits<double>::lowest()};
};

static bool integrateFaceLoops(
    const RVDResult &result,
    uint32_t face,
    const glm::dvec3 &reference,
    const glm::dvec3 &normal,
    LoopMeasure &out) {
  out = LoopMeasure{};
  if (size_t(face) + 1 >= result.faceLoopOffsets.size()) return false;
  const uint32_t loopBegin = result.faceLoopOffsets[face];
  const uint32_t loopEnd = result.faceLoopOffsets[face + 1];
  if (loopBegin > loopEnd ||
      size_t(loopEnd) + 1 > result.loopVertexOffsets.size()) {
    return false;
  }
  bool any = false;
  for (uint32_t loop = loopBegin; loop < loopEnd; ++loop) {
    const uint32_t begin = result.loopVertexOffsets[loop];
    const uint32_t end = result.loopVertexOffsets[loop + 1];
    if (begin > end || size_t(end) > result.loopVertices.size()) return false;
    if (end - begin < 3) return false;
    for (uint32_t vertex = begin; vertex < end; ++vertex) {
      const glm::dvec3 current = result.loopVertices[vertex] - reference;
      const glm::dvec3 next =
          result.loopVertices[vertex + 1 < end ? vertex + 1 : begin] - reference;
      const glm::dvec3 spanned = glm::cross(current, next);
      out.areaVector += spanned * 0.5;
      out.moment += (current + next) * (glm::dot(spanned, normal) / 6.0);
      out.boundsMin = glm::min(out.boundsMin, current);
      out.boundsMax = glm::max(out.boundsMax, current);
    }
    any = true;
  }
  return any;
}

static bool appendFaceLoop(
    RVDResult &result,
    const std::vector<glm::dvec3> &polygon,
    const glm::dvec3 &normal) {
  if (polygon.size() < 3 || result.faceLoopOffsets.empty() ||
      result.loopVertexOffsets.empty()) {
    return false;
  }
  glm::dvec3 boundsMin(std::numeric_limits<double>::max());
  glm::dvec3 boundsMax(std::numeric_limits<double>::lowest());
  for (const glm::dvec3 &point : polygon) {
    boundsMin = glm::min(boundsMin, point);
    boundsMax = glm::max(boundsMax, point);
  }
  const double extent = glm::length(boundsMax - boundsMin);
  if (!(extent > 0.0) || !std::isfinite(extent)) return false;
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * extent;

  std::vector<glm::dvec3> loop;
  loop.reserve(polygon.size());
  for (const glm::dvec3 &point : polygon) {
    if (!loop.empty() && glm::length(point - loop.back()) <= tolerance) continue;
    loop.push_back(point);
  }
  while (loop.size() > 1 &&
         glm::length(loop.front() - loop.back()) <= tolerance) {
    loop.pop_back();
  }
  if (loop.size() < 3) return false;

  glm::dvec3 reference(0.0);
  for (const glm::dvec3 &point : loop) reference += point;
  reference /= double(loop.size());
  double signedArea = 0.0;
  for (size_t vertex = 0; vertex < loop.size(); ++vertex) {
    const glm::dvec3 current = loop[vertex] - reference;
    const glm::dvec3 next = loop[(vertex + 1) % loop.size()] - reference;
    signedArea += glm::dot(glm::cross(current, next), normal);
  }
  if (!std::isfinite(signedArea) || signedArea == 0.0) return false;
  if (signedArea < 0.0) std::reverse(loop.begin(), loop.end());

  result.faceLoopOffsets.push_back(result.faceLoopOffsets.back() + 1u);
  result.loopVertexOffsets.push_back(
      result.loopVertexOffsets.back() + uint32_t(loop.size()));
  result.loopVertices.insert(result.loopVertices.end(), loop.begin(), loop.end());
  return true;
}

static bool validateFaceLoops(
    const RVDResult &result,
    uint32_t baseFaceCount,
    const std::vector<SurfaceRegion> &surfaceRegions) {
  if (result.faceLoopOffsets.empty() ||
      result.faceLoopOffsets.size() != result.fragmentFaces.size() + 1 ||
      result.faceLoopOffsets[0] != 0u ||
      result.loopVertexOffsets.empty() ||
      result.loopVertexOffsets.size() != size_t(result.faceLoopOffsets.back()) + 1 ||
      result.loopVertexOffsets[0] != 0u ||
      result.loopVertices.size() != result.loopVertexOffsets.back()) {
    std::cerr << "[RVDTopology] face loop arrays are inconsistent" << std::endl;
    return false;
  }

  for (uint32_t faceId = 0; faceId < result.fragmentFaces.size(); ++faceId) {
    const FragmentFace &face = result.fragmentFaces[faceId];
    if (!(face.areaA > 0.0) || !std::isfinite(face.areaA)) {
      std::cerr << "[RVDTopology] face has no A-side measure face=" << faceId
                << std::endl;
      return false;
    }
    const double areaA = face.areaA;
    const glm::dvec3 publishedCentroid = face.momentA / areaA;
    const glm::dvec3 normal = face.areaVectorA / areaA;
    LoopMeasure loop;
    if (!integrateFaceLoops(result, faceId, publishedCentroid, normal, loop)) {
      std::cerr << "[RVDTopology] invalid face loop face=" << faceId
                << " loops=" << result.faceLoopOffsets[faceId + 1] -
                       result.faceLoopOffsets[faceId]
                << std::endl;
      return false;
    }
    const double extent = glm::length(loop.boundsMax - loop.boundsMin);
    const double lengthTolerance =
        1e-9 * extent + 1e-12 * glm::length(publishedCentroid);
    const double areaTolerance = 1e-9 * areaA + extent * lengthTolerance;
    const double centroidTolerance =
        1e-6 * extent + 1e-11 * glm::length(publishedCentroid);
    if (!(glm::length(loop.areaVector - face.areaVectorA) <= areaTolerance)) {
      std::cerr << "[RVDTopology] face loop area mismatch face=" << faceId
                << " area=" << areaA
                << " loopArea=" << glm::dot(loop.areaVector, normal)
                << " error=" << glm::length(loop.areaVector - face.areaVectorA)
                << std::endl;
      return false;
    }

    const double loopArea = glm::dot(loop.areaVector, normal);
    if (!(loopArea > 0.0)) {
      std::cerr << "[RVDTopology] face loop winding mismatch face=" << faceId
                << " loopArea=" << loopArea << std::endl;
      return false;
    }
    const glm::dvec3 loopCentroid = publishedCentroid + loop.moment / loopArea;
    if (!(glm::length(loopCentroid - publishedCentroid) <= centroidTolerance)) {
      std::cerr << "[RVDTopology] face loop centroid mismatch face=" << faceId
                << " source=" << (faceId < baseFaceCount ? "base" : "surface")
                << " baseFaces=" << baseFaceCount
                << " fragmentA=" << face.fragmentA
                << " fragmentB=" << face.fragmentB
                << " faceType=" << uint32_t(face.type)
                << " domainFace=" << face.domainFace
                << " error="
                << glm::length(loopCentroid - publishedCentroid)
                << " tolerance=" << centroidTolerance
                << " area=" << areaA
                << " extent=" << extent
                << " centroid=(" << publishedCentroid.x << ","
                << publishedCentroid.y << "," << publishedCentroid.z << ")"
                << " moment=(" << face.momentA.x << "," << face.momentA.y
                << "," << face.momentA.z << ")"
                << " loopMoment=(" << loop.moment.x << "," << loop.moment.y
                << "," << loop.moment.z << ")"
                << " loops=" << result.faceLoopOffsets[faceId + 1] -
                       result.faceLoopOffsets[faceId]
                << " vertices=" << result.loopVertexOffsets[
                       result.faceLoopOffsets[faceId + 1]] -
                       result.loopVertexOffsets[result.faceLoopOffsets[faceId]];
      if (faceId >= baseFaceCount) {
        const uint32_t surfaceIndex = faceId - baseFaceCount;
        if (surfaceIndex < surfaceRegions.size()) {
          const SurfaceRegion &region = surfaceRegions[surfaceIndex];
          std::cerr << " surfaceIndex=" << surfaceIndex
                    << " cell=" << region.cellId
                    << " model=" << region.runtimeModelId
                    << " triangle=" << region.sourceTriangleId
                    << " contact=" << region.contactFacing
                    << " regionArea=" << region.area
                    << " regionMoment=(" << region.moment.x << ","
                    << region.moment.y << "," << region.moment.z << ")";
        }
      }
      std::cerr << " loopVertices=";
      for (uint32_t loopIndex = result.faceLoopOffsets[faceId];
           loopIndex < result.faceLoopOffsets[faceId + 1]; ++loopIndex) {
        std::cerr << "[";
        for (uint32_t vertexIndex = result.loopVertexOffsets[loopIndex];
             vertexIndex < result.loopVertexOffsets[loopIndex + 1];
             ++vertexIndex) {
          const glm::dvec3 &point = result.loopVertices[vertexIndex];
          std::cerr << "(" << point.x << "," << point.y << ","
                    << point.z << ")";
        }
        std::cerr << "]";
      }
      std::cerr << std::endl;
      return false;
    }
  }
  return true;
}


struct VerifySolidModel {
  uint32_t runtimeModelId;
  const double *restrictedPlaneAreas;
};

__global__ void verifySolidAreas(
    uint32_t faceCount,
    uint32_t totalPlanes,
    const BaseFace *baseFaces,
    const BaseFaceMeasure *baseMeasures,
    const uint32_t *regionOffsets,
    const FaceRegion *faceRegions,
    uint32_t modelCount,
    const VerifySolidModel *models,
    RvdPackedError *packedError) {
  const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
  if (faceId >= faceCount) return;

  const BaseFace &face = baseFaces[faceId];
  const uint32_t slots[2] = {face.slotA, face.slotB};
  const uint32_t cells[2] = {face.cellA, face.cellB};
  const uint32_t regionBegin = regionOffsets[faceId];
  const uint32_t regionEnd = regionOffsets[faceId + 1];

  for (int side = 0; side < 2; ++side) {
    const uint32_t slot = slots[side];
    const uint32_t cell = cells[side];
    if (slot == InvalidId || cell == InvalidId) continue;
    if (slot >= totalPlanes) {
      rvdReportErrorOnce(packedError, uint32_t(RVDStatus::InvalidIndex), faceId);
      return;
    }

    for (uint32_t m = 0; m < modelCount; ++m) {
      const uint32_t materialId = models[m].runtimeModelId;
      double partitionSolid = 0.0;
      for (uint32_t regionId = regionBegin; regionId < regionEnd; ++regionId) {
        const FaceRegion &region = faceRegions[regionId];
        const uint32_t material = side == 0 ? region.materialA : region.materialB;
        if (material != materialId) continue;
        partitionSolid += region.measure.area;
      }

      const double rawRestricted = models[m].restrictedPlaneAreas[slot];
      const double delta = partitionSolid - rawRestricted;
      const BaseFaceMeasure &baseMeasure = baseMeasures[faceId];
      const double tolerance = 1.0e-4 * fmax(baseMeasure.area, 1.0e-30);
      if (fabs(delta) > tolerance) {
        rvdReportErrorOnce(packedError, uint32_t(FacePartitionError::AreaConservation), faceId);
        return;
      }
    }
  }
}

bool verifySolidAreas(
    const BaseFaceTopology &faceTopology,
    const BaseCellDomain &baseDomain,
    const std::vector<ModelRestriction> &models,
    cudaStream_t stream) {
  const uint32_t totalPlanes = baseDomain.totalPlanes;
  const uint32_t faceCount = faceTopology.totalBaseFaces;
  if (totalPlanes == 0 || faceCount == 0 || models.empty()) return true;

  std::vector<VerifySolidModel> hostModels(models.size());
  for (size_t m = 0; m < models.size(); ++m) {
    if (models[m].devRestrictedPlaneAreas.size() < totalPlanes) return false;
    hostModels[m].runtimeModelId = models[m].runtimeModelId;
    hostModels[m].restrictedPlaneAreas = models[m].devRestrictedPlaneAreas.get();
  }

  cudaUtils::CudaBuffer<VerifySolidModel> devModels;
  cudaUtils::CudaBuffer<RvdPackedError> devPackedError;
  if (!devModels.allocate(models.size()) ||
      !devModels.uploadAsync(hostModels.data(), models.size(), stream) ||
      !devPackedError.allocate(1) ||
      !rvdHost::zero(devPackedError.get(), sizeof(RvdPackedError), stream, "clear solid area error")) {
    return false;
  }

  constexpr uint32_t BlockSize = 256;
  verifySolidAreas<<<
      rvdHost::blockCount(faceCount, BlockSize), BlockSize, 0, stream>>>(
      faceCount,
      totalPlanes,
      faceTopology.devBaseFaces.get(),
      faceTopology.devMeasures.get(),
      faceTopology.devRegionOffsets.get(),
      faceTopology.devRegions.get(),
      uint32_t(models.size()),
      devModels.get(),
      devPackedError.get());

  if (!rvdHost::checkLaunch("launch verifySolidAreas")) return false;

  RvdPackedError hostError = 0;
  if (!rvdHost::readPackedError(devPackedError, stream, hostError)) return false;
  if (hostError != 0) {
    std::cerr << "[RVD] face area mismatch face=" << rvdErrorId(hostError) << std::endl;
    return false;
  }
  return true;
}

bool buildFragmentsAndFaces(
    const RVDInput &input,
    const BaseCellDomain &baseDomain,
    const BaseFaceTopology &faceTopology,
    const std::vector<ModelRestriction> &models,
    std::vector<SurfaceRegion> &surfaceRegions,
    uint32_t &totalBaseFragmentFaces,
    RVDResult &result,
    cudaStream_t stream) {
  if (input.seeds == nullptr || input.seeds->size() != baseDomain.count) {
    return false;
  }

  const uint32_t cellCount = baseDomain.count;
  const uint32_t modelCount = uint32_t(models.size());
  const size_t phaseCount = size_t(modelCount) + 1;
  if (cellCount != 0 && phaseCount > std::numeric_limits<size_t>::max() / cellCount) {
    return false;
  }

  result.fragments.clear();
  result.fragmentFaces.clear();
  result.contactCouplings.clear();
  totalBaseFragmentFaces = 0;

  std::vector<double4> baseMoments(cellCount);
  if (baseDomain.devBaseMoments.size() < cellCount ||
      !baseDomain.devBaseMoments.download(baseMoments.data(), cellCount)) {
    std::cerr << "[RVDTopology] failed to download base-cell moments"
              << std::endl;
    return false;
  }

  std::vector<uint8_t> hasSolid(cellCount, 0u);
  std::vector<uint32_t> fragmentByCellPhase(
      size_t(cellCount) * phaseCount, InvalidId);

  for (uint32_t cell = 0; cell < cellCount; ++cell) {
    const double4 raw = baseMoments[cell];
    const double volume = raw.w * baseDomain.invBaseVolumeScale / 6.0;
    const glm::dvec3 relativeMoment(
        raw.x * baseDomain.invBaseMomentScale / 6.0,
        raw.y * baseDomain.invBaseMomentScale / 6.0,
        raw.z * baseDomain.invBaseMomentScale / 6.0);
    const glm::dvec3 moment =
        seedPosition(input, cell) * volume + relativeMoment;
    if (!(volume > 0.0) || !std::isfinite(volume) || !finiteMoment(moment)) {
      std::cerr << "[RVDTopology] invalid base volume/moment cell=" << cell
                << " volume=" << volume << std::endl;
      return false;
    }
  }

  for (uint32_t model = 0; model < modelCount; ++model) {
    if (models[model].runtimeModelId == InvalidId ||
        models[model].devMoments.size() < cellCount) {
      std::cerr << "[RVDTopology] invalid restricted moments model=" << model
                << " modelId=" << models[model].runtimeModelId << std::endl;
      return false;
    }
    for (uint32_t previous = 0; previous < model; ++previous) {
      if (models[previous].runtimeModelId == models[model].runtimeModelId) {
        std::cerr << "[RVDTopology] duplicate runtime model ID="
                  << models[model].runtimeModelId << std::endl;
        return false;
      }
    }

    std::vector<double4> restrictedMoments(cellCount);
    if (!models[model].devMoments.download(restrictedMoments.data(), cellCount)) {
      std::cerr << "[RVDTopology] failed to download restricted moments modelId="
                << models[model].runtimeModelId << std::endl;
      return false;
    }

    for (uint32_t cell = 0; cell < cellCount; ++cell) {
      const double4 raw = restrictedMoments[cell];
      if (raw.w == 0.0) continue;
      if (!(raw.w > 0.0) || !std::isfinite(raw.w) ||
          !std::isfinite(raw.x) || !std::isfinite(raw.y) ||
          !std::isfinite(raw.z)) {
        std::cerr << "[RVDTopology] invalid restricted phase moment cell="
                  << cell << " modelId=" << models[model].runtimeModelId
                  << std::endl;
        return false;
      }

      const double volume = raw.w * baseDomain.invBaseVolumeScale;
      const glm::dvec3 relativeMoment(
          raw.x * baseDomain.invBaseMomentScale,
          raw.y * baseDomain.invBaseMomentScale,
          raw.z * baseDomain.invBaseMomentScale);
      const glm::dvec3 moment = worldFirstMoment(
          seedPosition(input, cell), volume, relativeMoment);
      if (!(volume > 0.0) || !std::isfinite(volume) ||
          !finiteMoment(moment)) {
        std::cerr << "[RVDTopology] invalid restricted phase volume cell="
                  << cell << " modelId=" << models[model].runtimeModelId
                  << " volume=" << volume << std::endl;
        return false;
      }

      const size_t phase = size_t(cell) * phaseCount + model;
      const uint32_t fragmentId = uint32_t(result.fragments.size());
      const glm::dvec3 centroid = moment / volume;
      if (!finiteMoment(centroid)) return false;
      Fragment fragment{};
      fragment.cellId = cell;
      fragment.ownerModelId = models[model].runtimeModelId;
      fragment.volume = volume;
      fragment.centroid = centroid;
      result.fragments.push_back(fragment);
      fragmentByCellPhase[phase] = fragmentId;
      hasSolid[cell] = 1u;
    }
  }

  const uint32_t faceCount = faceTopology.totalBaseFaces;
  std::vector<BaseFace> baseFaces(faceCount);
  std::vector<uint32_t> regionOffsets(size_t(faceCount) + 1, 0u);
  if (faceCount > 0) {
    if (faceTopology.devBaseFaces.size() < faceCount ||
        faceTopology.devRegionOffsets.size() < size_t(faceCount) + 1 ||
        !faceTopology.devBaseFaces.download(baseFaces.data(), faceCount) ||
        !faceTopology.devRegionOffsets.download(
            regionOffsets.data(), regionOffsets.size())) {
      return false;
    }
  }
  const uint32_t regionCount = regionOffsets.back();
  for (uint32_t face = 0; face < faceCount; ++face) {
    if (regionOffsets[face] > regionOffsets[face + 1] ||
        regionOffsets[face + 1] > regionCount) {
      std::cerr << "[RVDTopology] invalid base-face region offsets face="
                << face << std::endl;
      return false;
    }
  }
  std::vector<FaceRegion> faceRegions(regionCount);
  if (regionCount > 0 &&
      (faceTopology.devRegions.size() < regionCount ||
       !faceTopology.devRegions.download(faceRegions.data(), regionCount))) {
    return false;
  }
  if (faceTopology.hostFaceLoopOffsets.size() != size_t(regionCount) + 1 ||
      faceTopology.hostFaceLoopOffsets[0] != 0u ||
      faceTopology.hostLoopVertexOffsets.empty() ||
      faceTopology.hostLoopVertexOffsets.size() !=
          size_t(faceTopology.hostFaceLoopOffsets.back()) + 1 ||
      faceTopology.hostLoopVertices.size() !=
          faceTopology.hostLoopVertexOffsets.back()) {
    std::cerr << "[RVDTopology] base face loops are incomplete regions="
              << regionCount << " loops=" << faceTopology.hostFaceLoopOffsets.size()
              << std::endl;
    return false;
  }
  result.faceLoopOffsets = faceTopology.hostFaceLoopOffsets;
  result.loopVertexOffsets = faceTopology.hostLoopVertexOffsets;
  result.loopVertices = faceTopology.hostLoopVertices;
  if (!verifySolidAreas(faceTopology, baseDomain, models, stream)) {
    return false;
  }

  // fluid phase bounded by material partitions
  std::vector<uint32_t> fluidBoundaryCounts(cellCount, 0u);
  for (uint32_t faceId = 0; faceId < faceCount; ++faceId) {
    const BaseFace &face = baseFaces[faceId];
    for (uint32_t regionId = regionOffsets[faceId];
         regionId < regionOffsets[faceId + 1]; ++regionId) {
      const FaceRegion &region = faceRegions[regionId];
      if (face.cellA < cellCount && region.materialA == InvalidId) {
        ++fluidBoundaryCounts[face.cellA];
      }
      if (face.cellB < cellCount && region.materialB == InvalidId) {
        ++fluidBoundaryCounts[face.cellB];
      }
    }
  }
  for (const SurfaceRegion &region : surfaceRegions) {
    if (region.cellId < cellCount && !region.contactFacing && region.outsideMaterialId == InvalidId) {
      ++fluidBoundaryCounts[region.cellId];
    }
  }

  for (uint32_t cell = 0; cell < cellCount; ++cell) {
    if (fluidBoundaryCounts[cell] == 0 && hasSolid[cell]) {
      continue;
    }

    const size_t phase = size_t(cell) * phaseCount + modelCount;
    const uint32_t fragmentId = uint32_t(result.fragments.size());
    Fragment fragment{};
    fragment.cellId = cell;
    fragment.ownerModelId = InvalidModelId;
    fragment.volume = 0.0;
    fragment.centroid = seedPosition(input, cell);
    result.fragments.push_back(fragment);
    fragmentByCellPhase[phase] = fragmentId;
  }

  const auto fragmentFor = [&](uint32_t cell, uint32_t material) {
    if (cell >= cellCount) return InvalidId;
    const uint32_t slot = materialSlot(material, models);
    if (slot == InvalidId) return InvalidId;
    return fragmentByCellPhase[size_t(cell) * phaseCount + slot];
  };

  for (uint32_t faceId = 0; faceId < faceCount; ++faceId) {
    const BaseFace &baseFace = baseFaces[faceId];
    for (uint32_t regionId = regionOffsets[faceId];
         regionId < regionOffsets[faceId + 1]; ++regionId) {
      const FaceRegion &region = faceRegions[regionId];
      const double area = region.measure.area;
      const double3 n = region.measure.normal;
      if (region.measure.valid == 0 || !(area > 0.0) ||
          !std::isfinite(area) || !std::isfinite(n.x) ||
          !std::isfinite(n.y) || !std::isfinite(n.z) ||
          !std::isfinite(region.measure.moment.x) ||
          !std::isfinite(region.measure.moment.y) ||
          !std::isfinite(region.measure.moment.z)) {
        std::cerr << "[RVDTopology] invalid base face region face=" << faceId
                  << " region=" << regionId << " area=" << area << std::endl;
        return false;
      }

      const uint32_t fragmentA = fragmentFor(baseFace.cellA, region.materialA);
      const uint32_t fragmentB = baseFace.cellB != InvalidId
          ? fragmentFor(baseFace.cellB, region.materialB)
          : InvalidId;
      if (fragmentA == InvalidId ||
          (baseFace.cellB != InvalidId && fragmentB == InvalidId)) {
        std::cerr << "[RVDTopology] base face region has no phase fragment "
                  << "face=" << faceId << " region=" << regionId
                  << " cellA=" << baseFace.cellA
                  << " materialA=" << region.materialA
                  << " fragmentA=" << fragmentA
                  << " cellB=" << baseFace.cellB
                  << " materialB=" << region.materialB
                  << " fragmentB=" << fragmentB << std::endl;
        return false;
      }

      const glm::dvec3 moment(
          region.measure.moment.x,
          region.measure.moment.y,
          region.measure.moment.z);
      FragmentFace face{};
      face.fragmentA = fragmentA;
      face.fragmentB = fragmentB;
      face.areaA = area;
      face.areaVectorA = glm::dvec3(n.x, n.y, n.z) * area;
      face.momentA = moment;
      if (fragmentB != InvalidId) {
        face.areaB = area;
        face.areaVectorB = -face.areaVectorA;
        face.momentB = moment;
        face.type = FaceType::Shared;
      } else {
        face.type = FaceType::DomainBoundary;
        face.domainFace = baseFace.domainFace;
      }
      result.fragmentFaces.push_back(face);
    }
  }
  totalBaseFragmentFaces = uint32_t(result.fragmentFaces.size());

  for (uint32_t regionId = 0; regionId < surfaceRegions.size(); ++regionId) {
    SurfaceRegion &region = surfaceRegions[regionId];
    region.faceId = InvalidId;
    if (!(region.area > 0.0) || !std::isfinite(region.area) ||
        region.polygon.size() < 3 || !finiteMoment(region.moment) ||
        !finiteMoment(region.normal)) {
      std::cerr << "[RVDTopology] invalid CAD surface region=" << regionId
                << " cell=" << region.cellId << " area=" << region.area
                << std::endl;
      return false;
    }
    const uint32_t solid = fragmentFor(region.cellId, region.runtimeModelId);
    if (solid == InvalidId) {
      std::cerr << "[RVDTopology] CAD surface missing phase fragment "
                << "region=" << regionId << " cell=" << region.cellId
                << " inside=" << region.runtimeModelId << std::endl;
      return false;
    }

    FragmentFace face{};
    face.fragmentA = solid;
    face.areaA = region.area;
    face.areaVectorA = region.normal * region.area;
    face.momentA = region.moment;
    face.domainFace = InvalidId;

    if (region.contactFacing) {
      face.fragmentB = InvalidId;
      face.areaB = 0.0;
      face.areaVectorB = glm::dvec3(0.0);
      face.momentB = glm::dvec3(0.0);
      face.type = FaceType::MaterialBoundary;
    } else {
      const uint32_t outside =
          fragmentFor(region.cellId, region.outsideMaterialId);
      if (outside == InvalidId || solid == outside) {
        std::cerr << "[RVDTopology] CAD surface missing phase fragment "
                  << "region=" << regionId << " cell=" << region.cellId
                  << " inside=" << region.runtimeModelId
                  << " outside=" << region.outsideMaterialId
                  << " solidFragment=" << solid
                  << " outsideFragment=" << outside << std::endl;
        return false;
      }
      face.fragmentB = outside;
      face.areaB = region.area;
      face.areaVectorB = -face.areaVectorA;
      face.momentB = region.moment;
      face.type = FaceType::Shared;
    }

    region.faceId = uint32_t(result.fragmentFaces.size());
    result.fragmentFaces.push_back(face);
    if (!appendFaceLoop(result, region.polygon, region.normal)) {
      std::cerr << "[RVDTopology] invalid CAD surface loop region=" << regionId
                << " cell=" << region.cellId << std::endl;
      return false;
    }
  }

  std::vector<double> fluidBoundaryVolumes(result.fragments.size(), 0.0);
  std::vector<double> fluidVolumeCompensations(result.fragments.size(), 0.0);
  std::vector<glm::dvec3> fluidBoundaryMoments(result.fragments.size(), glm::dvec3(0.0));
  std::vector<glm::dvec3> fluidMomentCompensations(result.fragments.size(), glm::dvec3(0.0));
  std::vector<glm::dvec3> seedRefs(result.fragments.size());
  for (size_t i = 0; i < result.fragments.size(); ++i) {
    seedRefs[i] = seedPosition(input, result.fragments[i].cellId);
  }

  for (const FragmentFace &face : result.fragmentFaces) {
    if (face.fragmentA < result.fragments.size() && face.areaA > 0.0) {
      const uint32_t a = face.fragmentA;
      if (result.fragments[a].ownerModelId == InvalidModelId) {
        const glm::dvec3 x_f = face.momentA / face.areaA;
        const glm::dvec3 r_f = x_f - seedRefs[a];
        const double vPyramid = glm::dot(r_f, face.areaVectorA) / 3.0;
        compensatedAdd(vPyramid, fluidBoundaryVolumes[a], fluidVolumeCompensations[a]);
        compensatedAdd(vPyramid * (0.75 * r_f), fluidBoundaryMoments[a], fluidMomentCompensations[a]);
      }
    }
    if (face.fragmentB < result.fragments.size() && face.areaB > 0.0) {
      const uint32_t b = face.fragmentB;
      if (result.fragments[b].ownerModelId == InvalidModelId) {
        const glm::dvec3 x_f = face.momentB / face.areaB;
        const glm::dvec3 r_f = x_f - seedRefs[b];
        const double vPyramid = glm::dot(r_f, face.areaVectorB) / 3.0;
        compensatedAdd(vPyramid, fluidBoundaryVolumes[b], fluidVolumeCompensations[b]);
        compensatedAdd(vPyramid * (0.75 * r_f), fluidBoundaryMoments[b], fluidMomentCompensations[b]);
      }
    }
  }

  std::vector<uint32_t> incidentFaceCounts(result.fragments.size(), 0u);
  for (const FragmentFace &face : result.fragmentFaces) {
    if (face.fragmentA < incidentFaceCounts.size()) {
      ++incidentFaceCounts[face.fragmentA];
    } else if (face.fragmentA != InvalidId) {
      return false;
    }
    if (face.fragmentB < incidentFaceCounts.size()) {
      ++incidentFaceCounts[face.fragmentB];
    } else if (face.fragmentB != InvalidId) {
      return false;
    }
  }

  for (size_t i = 0; i < result.fragments.size(); ++i) {
    if (result.fragments[i].ownerModelId == InvalidModelId) {
      const double V = fluidBoundaryVolumes[i];
      if (!(V > 0.0) || !std::isfinite(V)) {
        std::cerr << "[RVDTopology] invalid boundary-derived fluid volume fragment="
                  << i << " cell=" << result.fragments[i].cellId
                  << " signedVolume=" << V
                  << " incidentFaces=" << incidentFaceCounts[i] << std::endl;
        return false;
      }
      const glm::dvec3 c = seedRefs[i] + fluidBoundaryMoments[i] / V;
      if (!std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z)) {
        std::cerr << "[RVDTopology] non-finite boundary-derived fluid centroid fragment="
                  << i << " cell=" << result.fragments[i].cellId << std::endl;
        return false;
      }
      result.fragments[i].volume = V;
      result.fragments[i].centroid = c;
    }
  }
  if (!validateFaceLoops(result, totalBaseFragmentFaces, surfaceRegions)) return false;
  for (uint32_t fragmentId = 0; fragmentId < result.fragments.size();
       ++fragmentId) {
    const Fragment &fragment = result.fragments[fragmentId];
    if (incidentFaceCounts[fragmentId] == 0) {
      std::cerr << "[RVDTopology] positive-volume fragment has no physical "
                << "faces fragment=" << fragmentId
                << " cell=" << fragment.cellId
                << " owner=" << fragment.ownerModelId
                << " volume=" << fragment.volume << std::endl;
      return false;
    }
  }

  result.cellFragmentOffsets.assign(size_t(cellCount) + 1, 0u);
  for (const Fragment &fragment : result.fragments) {
    if (fragment.cellId >= cellCount) return false;
    ++result.cellFragmentOffsets[fragment.cellId + 1];
  }
  for (uint32_t cell = 0; cell < cellCount; ++cell) {
    result.cellFragmentOffsets[cell + 1] += result.cellFragmentOffsets[cell];
  }
  result.cellFragments.resize(result.fragments.size());
  std::vector<uint32_t> cursor = result.cellFragmentOffsets;
  for (uint32_t fragmentId = 0; fragmentId < result.fragments.size();
       ++fragmentId) {
    const uint32_t cell = result.fragments[fragmentId].cellId;
    result.cellFragments[cursor[cell]++] = fragmentId;
  }
  return true;
}

} // namespace voronoi
