#include "RVDTopology.hpp"
#include "ConvexCell.cuh"
#include "RVDHostUtils.cuh"

#include <cfloat>
#include <cmath>
#include <iostream>

namespace voronoi {

namespace {

using rvdHost::blockCount;

__device__ inline void reportTopologyError(
    RvdPackedError *packedError,
    RVDTopologyError error,
    uint32_t cell) {
  rvdReportError(packedError, uint32_t(error), cell);
}

__global__ void classifyAndVerifyBasePlanesKernel(
    DeviceBaseCells baseCells,
    const uint32_t *cellPlaneOffsets,
    uint32_t cellCount,
    double invBaseScale,
    double invBaseAreaScale,
    double facetFloor,
    float nominalCellSize,
    uint32_t *faceFlags,
    uint32_t *reciprocalSlots,
    RvdPackedError *packedError) {
  const uint32_t cellA = blockIdx.x * blockDim.x + threadIdx.x;
  if (cellA >= cellCount) return;

  const BaseCellArrays cellStateA = resolveBaseCell(baseCells, cellA);
  const uint32_t countA = cellStateA.planeCount;
  const uint32_t offsetA = cellPlaneOffsets[cellA];
  const double h = double(nominalCellSize);
  const double tol = fmax(facetFloor, 1e-10 * h * h);

  for (uint32_t localPlaneA = 0; localPlaneA < countA; ++localPlaneA) {
    const uint32_t slotA = offsetA + localPlaneA;
    faceFlags[slotA] = 0;
    reciprocalSlots[slotA] = InvalidId;

    const int tagA = cellStateA.planeNeighborIds[localPlaneA];
    const double canonicalAreaA = double(cellStateA.planeAreas[localPlaneA]);
    const double areaA = canonicalAreaA * invBaseAreaScale;

    if (!isfinite(areaA) || areaA < -tol) {
      reportTopologyError(packedError, RVDTopologyError::NonFiniteArea, cellA);
      return;
    }

    if (tagA < 0) {
      const int domainFace = rvdDevice::domainFaceFromTag(tagA);
      if (domainFace < 0 || domainFace >= 6) {
        reportTopologyError(packedError, RVDTopologyError::InvalidDomainFace, cellA);
        return;
      }

      if (areaA > tol) {
        faceFlags[slotA] = 1;
        reciprocalSlots[slotA] = InvalidId;
      }
      continue;
    }

    const uint32_t cellB = uint32_t(tagA);
    if (cellB >= cellCount || cellB == cellA) {
      reportTopologyError(packedError, RVDTopologyError::InvalidNeighborCell, cellA);
      return;
    }

    const BaseCellArrays cellStateB = resolveBaseCell(baseCells, cellB);
    const uint32_t countB = cellStateB.planeCount;
    const uint32_t offsetB = cellPlaneOffsets[cellB];
    const double4 planeA = cellStateA.planes[localPlaneA];

    uint32_t localPlaneB = InvalidId;
    for (uint32_t p = 0; p < countB; ++p) {
      if (cellStateB.planeNeighborIds[p] == int(cellA)) {
        localPlaneB = p;
        break;
      }
    }

    if (localPlaneB == InvalidId) {
      if (areaA <= tol) continue;
      reportTopologyError(packedError, RVDTopologyError::MissingReciprocal, cellA);
      return;
    }

    const double canonicalAreaB = double(cellStateB.planeAreas[localPlaneB]);
    const double areaB = canonicalAreaB * invBaseAreaScale;

    if (!isfinite(areaB) || areaB < -tol) {
      reportTopologyError(packedError, RVDTopologyError::NonFiniteReciprocalArea, cellA);
      return;
    }

    const bool activeA = areaA > tol;
    const bool activeB = areaB > tol;

    if (!activeA && !activeB) continue;

    if (activeA != activeB) {
      reportTopologyError(packedError, RVDTopologyError::NonReciprocalActivity, cellA);
      return;
    }

    const double maxA = fmax(areaA, areaB);
    const double areaDiff = fabs(areaA - areaB);
    if (areaDiff > fmax(tol, 1e-8 * maxA)) {
      reportTopologyError(packedError, RVDTopologyError::AreaMismatch, cellA);
      return;
    }

    const double4 planeB = cellStateB.planes[localPlaneB];
    const double dotN = planeA.x * planeB.x + planeA.y * planeB.y + planeA.z * planeB.z;
    if (dotN > -0.9999) {
      reportTopologyError(packedError, RVDTopologyError::NormalNotAntiparallel, cellA);
      return;
    }

    const double offsetSum = planeA.w + planeB.w;
    const double offsetTol = fmax(1e-5, 1e-4 * double(nominalCellSize));
    if (fabs(offsetSum) > offsetTol) {
      reportTopologyError(packedError, RVDTopologyError::OffsetMismatch, cellA);
      return;
    }

    const uint32_t slotB = offsetB + localPlaneB;
    if (cellA < cellB) {
      faceFlags[slotA] = 1;
      reciprocalSlots[slotA] = slotB;
    }
  }
}

__global__ void emitBaseFacesKernel(
    DeviceBaseCells baseCells,
    const uint32_t *cellPlaneOffsets,
    const uint32_t *faceFlags,
    const uint32_t *faceOffsets,
    const uint32_t *reciprocalSlots,
    uint32_t cellCount,
    BaseFace *outBaseFaces) {
  const uint32_t cellA = blockIdx.x * blockDim.x + threadIdx.x;
  if (cellA >= cellCount) return;

  const BaseCellArrays cellStateA = resolveBaseCell(baseCells, cellA);
  const uint32_t countA = cellStateA.planeCount;
  const uint32_t offsetA = cellPlaneOffsets[cellA];

  for (uint32_t localPlaneA = 0; localPlaneA < countA; ++localPlaneA) {
    const uint32_t slotA = offsetA + localPlaneA;
    if (faceFlags[slotA] != 1) continue;

    const uint32_t dstIdx = faceOffsets[slotA];
    const uint32_t slotB = reciprocalSlots[slotA];
    const int tagA = cellStateA.planeNeighborIds[localPlaneA];
    const uint32_t cellB = (tagA >= 0) ? uint32_t(tagA) : InvalidId;
    const int domainIndex = rvdDevice::domainFaceFromTag(tagA);
    const uint32_t domainFace = (domainIndex >= 0) ? uint32_t(domainIndex) : InvalidId;

    BaseFace face{};
    face.cellA = cellA;
    face.cellB = cellB;
    face.slotA = slotA;
    face.slotB = slotB;
    face.domainFace = domainFace;
    outBaseFaces[dstIdx] = face;
  }
}

__global__ void buildBaseFaceMeasuresKernel(
    const BaseFace *baseFaces,
    uint32_t faceCount,
    DeviceBaseCells baseCells,
    const uint32_t *cellPlaneOffsets,
    double invBaseAreaScale,
    double invBaseScale,
    double3 domainMin,
    BaseFaceMeasure *outMeasures) {
  const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
  if (faceId >= faceCount) return;

  const BaseFace &face = baseFaces[faceId];
  const BaseCellArrays cellA = resolveBaseCell(baseCells, face.cellA);
  const uint32_t lpA = face.slotA - cellPlaneOffsets[face.cellA];
  const double canAreaA = double(cellA.planeAreas[lpA]);
  const double areaA = canAreaA * invBaseAreaScale;
  const double3 canMomA = cellA.planeMoments[lpA];
  const double momentScale = invBaseAreaScale * invBaseScale;
  const double3 momentA = make_double3(
      domainMin.x * areaA + canMomA.x * momentScale,
      domainMin.y * areaA + canMomA.y * momentScale,
      domainMin.z * areaA + canMomA.z * momentScale);

  const double4 facePlane = cellA.planes[lpA];
  const double normalLength = sqrt(
      double(facePlane.x) * double(facePlane.x) +
      double(facePlane.y) * double(facePlane.y) +
      double(facePlane.z) * double(facePlane.z));
  const double3 normal = normalLength > 0.0
      ? make_double3(
            double(facePlane.x) / normalLength,
            double(facePlane.y) / normalLength,
            double(facePlane.z) / normalLength)
      : make_double3(0.0, 0.0, 0.0);

  const double absAreaA = cellA.planeAreaAbsSums != nullptr
      ? double(cellA.planeAreaAbsSums[lpA]) * invBaseAreaScale
      : fabs(areaA);
  double areaError = 256.0 * DBL_EPSILON * absAreaA;
  const bool validA = areaA > 0.0 && isfinite(areaA) &&
      isfinite(momentA.x) && isfinite(momentA.y) && isfinite(momentA.z) &&
      normalLength > 0.0 && isfinite(normalLength);

  double area = areaA;
  double3 moment = momentA;
  bool valid = validA;

  if (face.cellB != InvalidId) {
    const BaseCellArrays cellB = resolveBaseCell(baseCells, face.cellB);
    const uint32_t lpB = face.slotB - cellPlaneOffsets[face.cellB];
    const double areaB = double(cellB.planeAreas[lpB]) * invBaseAreaScale;
    const double3 canMomB = cellB.planeMoments[lpB];
    const double3 momentB = make_double3(
        domainMin.x * areaB + canMomB.x * momentScale,
        domainMin.y * areaB + canMomB.y * momentScale,
        domainMin.z * areaB + canMomB.z * momentScale);
    const double absAreaB = cellB.planeAreaAbsSums != nullptr
        ? double(cellB.planeAreaAbsSums[lpB]) * invBaseAreaScale
        : fabs(areaB);
    areaError += 256.0 * DBL_EPSILON * absAreaB;

    const bool validB = areaB > 0.0 && isfinite(areaB) && isfinite(momentB.x) && isfinite(momentB.y) && isfinite(momentB.z);

    valid = validA || validB;
    if (!validA && validB) {
      area = areaB;
      moment = momentB;
    }
  }

  BaseFaceMeasure measure{};
  measure.area = area;
  measure.areaError = areaError;
  measure.moment = moment;
  measure.normal = make_double3(-normal.x, -normal.y, -normal.z);
  measure.valid = valid ? 1u : 0u;
  outMeasures[faceId] = measure;
}

__global__ void countBaseFacePolygonVerticesKernel(
    const BaseFace *baseFaces,
    uint32_t faceCount,
    DeviceBaseCells baseCells,
    const uint32_t *cellPlaneOffsets,
    uint32_t *vertexCounts) {
  const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
  if (faceId > faceCount) return;
  if (faceId == faceCount) {
    vertexCounts[faceId] = 0u;
    return;
  }

  const BaseFace &face = baseFaces[faceId];
  const BaseCellArrays cell = resolveBaseCell(baseCells, face.cellA);
  const uint32_t localPlane = face.slotA - cellPlaneOffsets[face.cellA];

  uint32_t count = 0;
  for (uint32_t v = 0; v < cell.vertexCount; ++v) {
    const uint3 triplet = cell.vertices[v];
    if (triplet.x == localPlane || triplet.y == localPlane || triplet.z == localPlane) {
      ++count;
    }
  }
  vertexCounts[faceId] = count;
}

__global__ void emitBaseFacePolygonVerticesKernel(
    const BaseFace *baseFaces,
    uint32_t faceCount,
    DeviceBaseCells baseCells,
    const uint32_t *cellPlaneOffsets,
    const uint32_t *vertexOffsets,
    double invBaseScale,
    double3 domainMin,
    double3 *outVertices) {
  const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
  if (faceId >= faceCount) return;

  const BaseFace &face = baseFaces[faceId];
  const BaseCellArrays cell = resolveBaseCell(baseCells, face.cellA);
  const uint32_t localPlane = face.slotA - cellPlaneOffsets[face.cellA];

  rvdDevice::ConvexCell view{};
  view.planes = const_cast<double4 *>(cell.planes);
  view.planeCount = int(cell.planeCount);

  uint32_t dst = vertexOffsets[faceId];
  for (uint32_t v = 0; v < cell.vertexCount; ++v) {
    const uint3 triplet = cell.vertices[v];
    if (triplet.x != localPlane && triplet.y != localPlane && triplet.z != localPlane) {
      continue;
    }
    const double3 world = cudaGeometry::canonicalToWorld(
        view.vertexPosition(triplet), domainMin, invBaseScale);
    outVertices[dst++] = world;
  }
}

struct ClassifiedFaces {
  cudaUtils::CudaBuffer<uint32_t> flags;
  cudaUtils::CudaBuffer<uint32_t> offsets;
  cudaUtils::CudaBuffer<uint32_t> reciprocalSlots;
  uint32_t totalFaces = 0;
};

bool classifyBasePlanes(
    const BaseCellDomain &baseDomain,
    ClassifiedFaces &classified,
    cudaStream_t stream) {
  const uint32_t cellCount = baseDomain.count;
  const uint32_t totalPlanes = baseDomain.totalPlanes;

  cudaUtils::CudaBuffer<RvdPackedError> packedError;
  if (!classified.flags.allocate(size_t(totalPlanes) + 1) ||
      !classified.offsets.allocate(size_t(totalPlanes) + 1) ||
      !classified.reciprocalSlots.allocate(totalPlanes) ||
      !packedError.allocate(1)) {
    return false;
  }

  if (!rvdHost::zero(
          classified.flags.get() + totalPlanes, sizeof(uint32_t), stream,
          "clear face flag tail") ||
      !rvdHost::zero(
          packedError.get(), sizeof(RvdPackedError), stream,
          "clear topology error")) {
    return false;
  }

  const uint32_t blocks = blockCount(cellCount);
  classifyAndVerifyBasePlanesKernel<<<blocks, rvdHost::BlockSize, 0, stream>>>(
      baseDomain.deviceView(),
      baseDomain.devCellPlaneOffsets.get(),
      cellCount,
      baseDomain.invBaseScale,
      baseDomain.invBaseAreaScale,
      baseDomain.facetFloor(),
      baseDomain.nominalCellSize,
      classified.flags.get(),
      classified.reciprocalSlots.get(),
      packedError.get());

  if (!rvdHost::checkLaunch("launch classifyAndVerifyBasePlanesKernel")) {
    return false;
  }

  RvdPackedError hostError = 0;
  if (!rvdHost::readPackedError(packedError, stream, hostError)) return false;
  if (hostError != 0) {
    const uint32_t errorCode = rvdErrorCode(hostError);
    const uint32_t errorCell = rvdErrorId(hostError);
    std::cerr << "[RVD-Topology-GPU] Topological defect detected: "
              << rvdTopologyErrorName(errorCode) << " (code=" << errorCode
              << ") on cell " << errorCell << std::endl;
    return false;
  }

  cudaUtils::CudaBuffer<uint8_t> scanScratch;
  if (!rvdHost::exclusiveSum(
          classified.flags.get(),
          classified.offsets.get(),
          int(totalPlanes + 1),
          scanScratch,
          stream)) {
    return false;
  }

  if (!rvdHost::readAt(
          classified.totalFaces,
          classified.offsets.get(),
          totalPlanes,
          stream,
          "read total base faces")) {
    return false;
  }
  return true;
}

bool emitBaseFaces(
    const BaseCellDomain &baseDomain,
    const ClassifiedFaces &classified,
    BaseFaceTopology &faceTopology,
    cudaStream_t stream) {
  if (classified.totalFaces == 0) return true;

  const uint32_t blocks = blockCount(baseDomain.count);
  emitBaseFacesKernel<<<blocks, rvdHost::BlockSize, 0, stream>>>(
      baseDomain.deviceView(),
      baseDomain.devCellPlaneOffsets.get(),
      classified.flags.get(),
      classified.offsets.get(),
      classified.reciprocalSlots.get(),
      baseDomain.count,
      faceTopology.devBaseFaces.get());
  return rvdHost::checkLaunch("launch emitBaseFacesKernel");
}

bool measureBaseFaces(
    const BaseCellDomain &baseDomain,
    BaseFaceTopology &faceTopology,
    cudaStream_t stream) {
  if (faceTopology.totalBaseFaces == 0) return true;

  const uint32_t blocks = blockCount(faceTopology.totalBaseFaces);
  buildBaseFaceMeasuresKernel<<<blocks, rvdHost::BlockSize, 0, stream>>>(
      faceTopology.devBaseFaces.get(),
      faceTopology.totalBaseFaces,
      baseDomain.deviceView(),
      baseDomain.devCellPlaneOffsets.get(),
      baseDomain.invBaseAreaScale,
      baseDomain.invBaseScale,
      make_double3(
          baseDomain.domainMin.x,
          baseDomain.domainMin.y,
          baseDomain.domainMin.z),
      faceTopology.devMeasures.get());
  return rvdHost::checkLaunch("launch buildBaseFaceMeasuresKernel");
}

bool collectFacePolygons(
    const BaseCellDomain &baseDomain,
    BaseFaceTopology &faceTopology,
    cudaStream_t stream) {
  const uint32_t faceCount = faceTopology.totalBaseFaces;
  if (faceCount == 0) return true;

  cudaUtils::CudaBuffer<uint32_t> vertexCounts;
  if (!vertexCounts.allocate(size_t(faceCount) + 1) ||
      !faceTopology.devFaceVertexOffsets.allocate(size_t(faceCount) + 1)) {
    return false;
  }

  const uint32_t blocks = blockCount(faceCount + 1);
  countBaseFacePolygonVerticesKernel<<<blocks, rvdHost::BlockSize, 0, stream>>>(
      faceTopology.devBaseFaces.get(),
      faceCount,
      baseDomain.deviceView(),
      baseDomain.devCellPlaneOffsets.get(),
      vertexCounts.get());
  if (!rvdHost::checkLaunch("launch countBaseFacePolygonVerticesKernel")) {
    return false;
  }

  cudaUtils::CudaBuffer<uint8_t> scanScratch;
  if (!rvdHost::exclusiveSum(
          vertexCounts.get(),
          faceTopology.devFaceVertexOffsets.get(),
          int(faceCount + 1),
          scanScratch,
          stream)) {
    return false;
  }

  uint32_t totalVertices = 0;
  if (!rvdHost::readAt(
          totalVertices,
          faceTopology.devFaceVertexOffsets.get(),
          faceCount,
          stream,
          "read total face vertices")) {
    return false;
  }
  if (totalVertices == 0) return true;

  if (!faceTopology.devFaceVertices.allocate(totalVertices)) return false;

  const uint32_t faceBlocks = blockCount(faceCount);
  emitBaseFacePolygonVerticesKernel<<<faceBlocks, rvdHost::BlockSize, 0, stream>>>(
      faceTopology.devBaseFaces.get(),
      faceCount,
      baseDomain.deviceView(),
      baseDomain.devCellPlaneOffsets.get(),
      faceTopology.devFaceVertexOffsets.get(),
      baseDomain.invBaseScale,
      make_double3(
          baseDomain.domainMin.x,
          baseDomain.domainMin.y,
          baseDomain.domainMin.z),
      faceTopology.devFaceVertices.get());
  return rvdHost::checkLaunch("launch emitBaseFacePolygonVerticesKernel");
}

} // namespace

bool buildBaseFaces(
    const BaseCellDomain &baseDomain,
    BaseFaceTopology &faceTopology,
    cudaStream_t stream) {
  const uint32_t cellCount = baseDomain.count;
  const uint32_t totalPlanes = baseDomain.totalPlanes;
  if (cellCount == 0 || totalPlanes == 0) {
    faceTopology.totalBaseFaces = 0;
    return true;
  }

  ClassifiedFaces classified;
  if (!classifyBasePlanes(baseDomain, classified, stream)) return false;

  faceTopology.totalBaseFaces = classified.totalFaces;
  if (!faceTopology.devBaseFaces.allocate(classified.totalFaces) ||
      !faceTopology.devMeasures.allocate(classified.totalFaces)) {
    return false;
  }

  return emitBaseFaces(baseDomain, classified, faceTopology, stream) &&
         measureBaseFaces(baseDomain, faceTopology, stream) &&
         collectFacePolygons(baseDomain, faceTopology, stream);
}

} // namespace voronoi
