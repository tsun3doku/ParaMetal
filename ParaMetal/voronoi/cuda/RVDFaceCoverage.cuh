#pragma once

#include "RVDFaceGeometry.cuh"
#include <cuda_runtime.h>

namespace voronoi {

struct FaceSideCoverage {
  uint32_t material = InvalidId;
  uint32_t modelIndex = InvalidId;
  uint32_t positiveCount = 0;
  uint32_t fullCount = 0;
  bool resolved = true;
  bool partial = false;
  double baseArea = 0.0;
  double area = 0.0;
  double areaError = 0.0;
  double3 moment = make_double3(0.0, 0.0, 0.0);
};
__device__ inline uint32_t
faceValidateCoverageMeasure(const FacePolygon &boundary, double3 origin,
                            double3 u, double3 v, double scale, double area,
                            double3 moment, double areaError) {
  if (!(area > 0.0) || !isfinite(area) || !isfinite(areaError)) {
    return InvalidFacePolygon;
  }
  if (area <= areaError) {
    return UnresolvedCoverageArea;
  }
  const double3 centroid = mul3(moment, 1.0 / area);
  const FacePoint2 local = faceProject(centroid, origin, u, v);
  const double coordinateMagnitude =
      fmax(fmax(fabs(origin.x), fabs(origin.y)), fabs(origin.z)) + scale;
  const double tolerance = 64.0 * DBL_EPSILON * (coordinateMagnitude + scale) +
                           4.0 * (areaError / area) * scale;
  if (!facePointInsideConvex(local, boundary, tolerance)) {
    return UnresolvedCoverageMoment;
  }
  return FacePartitionSuccess;
}
__device__ inline uint32_t faceResolveSideCoverage(FaceSideCoverage &side,
                                                   const FacePolygon &boundary,
                                                   double3 origin, double3 u,
                                                   double3 v, double scale) {
  if (!(side.positiveCount == 1 && side.partial))
    return FacePartitionSuccess;
  const uint32_t error = faceValidateCoverageMeasure(
      boundary, origin, u, v, scale, side.area, side.moment, side.areaError);
  if (error == UnresolvedCoverageArea) {
    side.resolved = true;
    side.partial = false;
    side.material = InvalidId;
    side.positiveCount = 0;
    side.area = 0.0;
    side.areaError = 0.0;
    side.moment = make_double3(0.0, 0.0, 0.0);
    return FacePartitionSuccess;
  }
  return error;
}
__device__ inline uint32_t
faceCollectModelCoverage(uint32_t slot, bool domainSide, double sideBaseArea,
                         const BaseFaceMeasure &measure,
                         const cudaGeometry::DeviceModelGeometryView *models,
                         uint32_t modelCount,
                         const double *const *modelPlaneAreas,
                         const double *const *modelPlaneAbsSums,
                         const double3 *const *modelPlaneMoments,
                         FaceSideCoverage &side) {
  side = FaceSideCoverage{};
  side.material = InvalidId;
  side.resolved = true;
  side.baseArea = sideBaseArea;
  if (domainSide || slot == InvalidId || modelPlaneAreas == nullptr ||
      modelPlaneAbsSums == nullptr || modelPlaneMoments == nullptr) {
    return FacePartitionSuccess;
  }
  if (!(sideBaseArea > 0.0) || !isfinite(sideBaseArea))
    return InvalidInput;
  const double tolFull =
      fmax(measure.areaError, 64.0 * DBL_EPSILON * sideBaseArea);
  for (uint32_t g = 0; g < modelCount; ++g) {
    if (modelPlaneAreas[g] == nullptr || modelPlaneAbsSums[g] == nullptr ||
        modelPlaneMoments[g] == nullptr)
      return InvalidInput;
    const double area = modelPlaneAreas[g][slot];
    if (!isfinite(area) || area > sideBaseArea + tolFull)
      return InvalidInput;
    if (!(area > 0.0))
      continue;
    const double absSum = modelPlaneAbsSums[g][slot];
    if (!isfinite(absSum) || absSum < 0.0)
      return InvalidInput;
    const bool fullCandidate = area >= sideBaseArea - tolFull;
    const double areaError = 256.0 * DBL_EPSILON * fmax(absSum, area);
    if (!fullCandidate && area <= areaError) {
      continue;
    }
    ++side.positiveCount;
    if (side.positiveCount == 1) {
      side.area = area;
      side.areaError = areaError;
      side.moment = modelPlaneMoments[g][slot];
      if (!isfinite(side.moment.x) || !isfinite(side.moment.y) ||
          !isfinite(side.moment.z)) {
        return InvalidInput;
      }
      side.modelIndex = g;
      side.material = models[g].runtimeModelId;
    }
    if (fullCandidate)
      ++side.fullCount;
  }
  if (side.positiveCount == 0) {
    side.resolved = true;
    side.material = InvalidId;
    side.area = 0.0;
    side.moment = make_double3(0.0, 0.0, 0.0);
  } else if (side.positiveCount == 1 && side.fullCount == 1) {
    side.resolved = true;
    side.partial = false;
  } else if (side.positiveCount == 1) {
    side.resolved = false;
    side.partial = true;
  } else {
    side.resolved = false;
    side.partial = false;
  }
  return FacePartitionSuccess;
}
__device__ inline double faceSideBaseArea(const BaseFace &face, bool sideA,
                                          DeviceBaseCells baseCells,
                                          const uint32_t *cellPlaneOffsets,
                                          double invBaseAreaScale) {
  const uint32_t cell = sideA ? face.cellA : face.cellB;
  const uint32_t slot = sideA ? face.slotA : face.slotB;
  if (cell == InvalidId || slot == InvalidId || cellPlaneOffsets == nullptr)
    return 0.0;
  const BaseCellArrays state = resolveBaseCell(baseCells, cell);
  const uint32_t offset = cellPlaneOffsets[cell];
  if (slot < offset)
    return 0.0;
  const uint32_t local = slot - offset;
  if (local >= state.planeCount)
    return 0.0;
  return double(state.planeAreas[local]) * invBaseAreaScale;
}
__device__ inline uint32_t faceCollectSideCoverage(
    const BaseFace &face, const BaseFaceMeasure &measure,
    const cudaGeometry::DeviceModelGeometryView *models, uint32_t modelCount,
    const double *const *modelPlaneAreas,
    const double *const *modelPlaneAbsSums,
    const double3 *const *modelPlaneMoments, DeviceBaseCells baseCells,
    const uint32_t *cellPlaneOffsets, double invBaseAreaScale,
    FaceSideCoverage &sideA, FaceSideCoverage &sideB) {
  uint32_t error = faceCollectModelCoverage(
      face.slotA, false,
      faceSideBaseArea(face, true, baseCells, cellPlaneOffsets,
                       invBaseAreaScale),
      measure, models, modelCount, modelPlaneAreas, modelPlaneAbsSums,
      modelPlaneMoments, sideA);
  if (error)
    return error;
  return faceCollectModelCoverage(
      face.slotB, face.cellB == InvalidId,
      faceSideBaseArea(face, false, baseCells, cellPlaneOffsets,
                       invBaseAreaScale),
      measure, models, modelCount, modelPlaneAreas, modelPlaneAbsSums,
      modelPlaneMoments, sideB);
}
__device__ inline uint32_t faceBuildCoverageRegions(
    const BaseFaceMeasure &base, const FacePolygon &boundary, double3 origin,
    double3 u, double3 v, double3 normal, double scale,
    const FaceSideCoverage &sideA, const FaceSideCoverage &sideB,
    FaceRegion *groups, uint32_t &groupCount, bool &handled) {
  handled = false;
  FaceSideCoverage a = sideA;
  FaceSideCoverage b = sideB;
  uint32_t error = faceResolveSideCoverage(a, boundary, origin, u, v, scale);
  if (error == UnresolvedCoverageMoment)
    return FacePartitionSuccess;
  if (error)
    return error;
  error = faceResolveSideCoverage(b, boundary, origin, u, v, scale);
  if (error == UnresolvedCoverageMoment)
    return FacePartitionSuccess;
  if (error)
    return error;
  if (a.positiveCount > 1 || b.positiveCount > 1)
    return FacePartitionSuccess;

  double polygonArea = 0.0;
  double3 polygonMoment = make_double3(0.0, 0.0, 0.0);
  double polygonRoundoff = 0.0;
  if (!facePolygonIntegral(boundary, origin, u, v, polygonArea, polygonMoment,
                           polygonRoundoff)) {
    return InvalidFacePolygon;
  }
  const double coordinateMagnitude =
      fmax(fmax(fabs(origin.x), fabs(origin.y)), fabs(origin.z)) + scale;

  double solidArea = 0.0;
  double3 solidMoment = make_double3(0.0, 0.0, 0.0);
  double solidError = 0.0;
  uint32_t solidMaterialA = a.resolved ? a.material : InvalidId;
  uint32_t solidMaterialB = b.resolved ? b.material : InvalidId;

  if (a.positiveCount == 1 && b.positiveCount == 1) {
    solidMaterialA = a.material;
    solidMaterialB = b.material;
    const bool sameModel = a.modelIndex == b.modelIndex;
    if (a.partial && b.partial) {
      if (!sameModel)
        return FacePartitionSuccess;
      const double sideScale = fmax(a.baseArea, b.baseArea);
      const double areaTol = fmax(base.areaError, 1e-8 * sideScale);
      const double3 momentDelta = sub3(a.moment, b.moment);
      const double momentTol =
          areaTol * coordinateMagnitude +
          1e-8 * fmax(a.area, b.area) * coordinateMagnitude;
      if (fabs(a.area - b.area) > areaTol || fabs(momentDelta.x) > momentTol ||
          fabs(momentDelta.y) > momentTol || fabs(momentDelta.z) > momentTol) {
        return FacePartitionSuccess;
      }
      solidArea = a.area;
      solidMoment = a.moment;
      solidError = fmax(a.areaError, b.areaError);
    } else if (a.partial) {
      solidArea = a.area;
      solidMoment = a.moment;
      solidError = a.areaError;
    } else if (b.partial) {
      solidArea = b.area;
      solidMoment = b.moment;
      solidError = b.areaError;
    } else {
      solidArea = polygonArea;
      solidMoment = polygonMoment;
    }
  } else if (a.positiveCount == 1) {
    solidMaterialA = a.material;
    solidMaterialB = b.resolved ? b.material : InvalidId;
    if (a.partial) {
      solidArea = a.area;
      solidMoment = a.moment;
      solidError = a.areaError;
    } else {
      solidArea = polygonArea;
      solidMoment = polygonMoment;
    }
  } else if (b.positiveCount == 1) {
    solidMaterialA = a.resolved ? a.material : InvalidId;
    solidMaterialB = b.material;
    if (b.partial) {
      solidArea = b.area;
      solidMoment = b.moment;
      solidError = b.areaError;
    } else {
      solidArea = polygonArea;
      solidMoment = polygonMoment;
    }
  }

  if (!(solidArea > 0.0)) {
    solidArea = 0.0;
    solidMoment = make_double3(0.0, 0.0, 0.0);
    solidError = 0.0;
  }
  const double complementError =
      fmax(base.areaError, polygonRoundoff) + solidError;
  if (solidArea >= polygonArea - complementError) {
    solidArea = polygonArea;
    solidMoment = polygonMoment;
    solidError = 0.0;
  }
  if (solidArea > polygonArea + complementError)
    return CoverageAreaConservation;

  handled = true;
  if (solidArea > 0.0) {
    const uint32_t e = faceEmitRegion(
        groups, groupCount, 2u, solidArea, solidMoment, normal, solidMaterialA,
        solidMaterialB, fmax(base.areaError, solidError));
    if (e)
      return e;
  }
  const double complementArea = polygonArea - solidArea;
  if (complementArea > complementError) {
    const double3 complementMoment =
        solidArea > 0.0 ? sub3(polygonMoment, solidMoment) : polygonMoment;
    const uint32_t complementA = a.resolved ? a.material : InvalidId;
    const uint32_t complementB = b.resolved ? b.material : InvalidId;
    const double complementMeasureError =
        fmax(base.areaError, polygonRoundoff) + solidError;
    const double3 complementCentroid =
        mul3(complementMoment, 1.0 / complementArea);
    const FacePoint2 local = faceProject(complementCentroid, origin, u, v);
    const double tolerance =
        64.0 * DBL_EPSILON * (coordinateMagnitude + scale) +
        4.0 * (complementMeasureError / complementArea) * scale;
    if (!facePointInsideConvex(local, boundary, tolerance)) {
      groupCount = 0;
      handled = false;
      return FacePartitionSuccess;
    }
    const uint32_t e = faceEmitRegion(groups, groupCount, 2u, complementArea,
                                      complementMoment, normal, complementA,
                                      complementB, complementMeasureError);
    if (e)
      return e;
  }
  return FacePartitionSuccess;
}

} // namespace voronoi
