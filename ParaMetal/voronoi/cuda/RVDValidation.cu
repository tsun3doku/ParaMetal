#include "RVDValidation.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace voronoi {

namespace {

constexpr double ReciprocalFatalTolerance = 1e-4;
constexpr double ReciprocalWarningTolerance = 1e-5;
constexpr uint32_t MaxReportedFragments = 5;

struct ReciprocalError {
  double area = 0.0;
  double vector = 0.0;
  double moment = 0.0;
};

struct FragmentAudit {
  double areaSum = 0.0;
  glm::dvec3 areaVectorSum{0.0};
  double gaussSum[9] = {};
  double absTermSum = 0.0;
  uint32_t incidentFaceCount = 0;
  double baseAreaSum = 0.0;
  glm::dvec3 baseVectorSum{0.0};
  double surfaceAreaSum = 0.0;
  glm::dvec3 surfaceVectorSum{0.0};
};

ReciprocalError measureSharedFaceReciprocity(const std::vector<FragmentFace> &faces) {
  ReciprocalError error;

  for (const FragmentFace &face : faces) {
    if (face.type != FaceType::Shared ||
        !(face.areaA > 0.0) || !(face.areaB > 0.0)) {
      continue;
    }

    const double scale = std::max(face.areaA, face.areaB);
    error.area = std::max(error.area, fabs(face.areaA - face.areaB) / scale);
    error.vector = std::max(
        error.vector,
        glm::length(face.areaVectorA + face.areaVectorB) / scale);

    const double momentScale =
        std::max(glm::length(face.momentA), glm::length(face.momentB));
    if (momentScale > 0.0) {
      const double momentDifference = glm::length(face.momentA - face.momentB);
      error.moment = std::max(error.moment, momentDifference / momentScale);
    }
  }
  return error;
}

bool validateSharedFaceReciprocity(const std::vector<FragmentFace> &faces) {
  const ReciprocalError error = measureSharedFaceReciprocity(faces);
  const double worst =
      std::max(error.area, std::max(error.vector, error.moment));

  if (worst > ReciprocalFatalTolerance) {
    std::cerr << "[RVD-FaceReciprocity] shared-face measures exceed tolerance "
              << worst << " > " << ReciprocalFatalTolerance << std::endl;
    return false;
  }
  if (worst > ReciprocalWarningTolerance) {
    std::cerr << "[RVD-FaceReciprocity] shared-face measures near tolerance "
              << worst << " > " << ReciprocalWarningTolerance << std::endl;
  }
  return true;
}

void accumulateFaceForFragment(
    FragmentAudit &audit,
    const glm::dvec3 &fragmentCentroid,
    const glm::dvec3 &faceMoment,
    const glm::dvec3 &faceAreaVector,
    double area,
    bool isBaseFace) {
  const glm::dvec3 centroid = faceMoment / area;
  const glm::dvec3 offset = centroid - fragmentCentroid;

  audit.areaSum += area;
  audit.areaVectorSum += faceAreaVector;
  if (isBaseFace) {
    audit.baseAreaSum += area;
    audit.baseVectorSum += faceAreaVector;
  } else {
    audit.surfaceAreaSum += area;
    audit.surfaceVectorSum += faceAreaVector;
  }

  audit.absTermSum += glm::length(offset) * glm::length(faceAreaVector);
  ++audit.incidentFaceCount;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      audit.gaussSum[col * 3 + row] += offset[row] * faceAreaVector[col];
    }
  }
}

std::vector<FragmentAudit> auditFragmentFaces(
    const std::vector<Fragment> &fragments,
    const std::vector<FragmentFace> &faces,
    uint32_t baseFaceCount) {
  std::vector<FragmentAudit> audits(fragments.size());

  for (size_t faceIndex = 0; faceIndex < faces.size(); ++faceIndex) {
    const FragmentFace &face = faces[faceIndex];
    const bool isBaseFace = faceIndex < baseFaceCount;

    if (face.fragmentA < audits.size() &&
        face.areaA > 0.0 && std::isfinite(face.areaA)) {
      accumulateFaceForFragment(
          audits[face.fragmentA], fragments[face.fragmentA].centroid,
          face.momentA, face.areaVectorA, face.areaA, isBaseFace);
    }
    if (face.fragmentB < audits.size() &&
        face.areaB > 0.0 && std::isfinite(face.areaB)) {
      accumulateFaceForFragment(
          audits[face.fragmentB], fragments[face.fragmentB].centroid,
          face.momentB, face.areaVectorB, face.areaB, isBaseFace);
    }
  }
  return audits;
}

bool validateFragmentClosure(
    const std::vector<Fragment> &fragments,
    const std::vector<FragmentAudit> &audits,
    double toleranceClosure,
    double toleranceGauss) {
  bool valid = true;
  uint32_t reportedCount = 0;

  for (uint32_t i = 0; i < fragments.size(); ++i) {
    const Fragment &fragment = fragments[i];
    const FragmentAudit &audit = audits[i];
    const double volume = fragment.volume;

    if (!(volume > 0.0) || !std::isfinite(volume)) {
      if (reportedCount++ < MaxReportedFragments) {
        std::cerr << "[RVDFragments] fragment " << i
                  << " has non-positive volume " << volume << std::endl;
      }
      valid = false;
      continue;
    }
    if (!(audit.areaSum > 0.0)) {
      if (reportedCount++ < MaxReportedFragments) {
        std::cerr << "[RVDFragments] fragment " << i
                  << " has no positive-area faces" << std::endl;
      }
      valid = false;
      continue;
    }
    if (audit.incidentFaceCount < 4) {
      if (reportedCount++ < MaxReportedFragments) {
        std::cerr << "[RVDFragments] fragment " << i
                  << " has fewer than four incident faces ("
                  << audit.incidentFaceCount << ")"
                  << " owner=" << fragment.ownerModelId
                  << " cell=" << fragment.cellId
                  << " volume=" << volume << std::endl;
      }
      valid = false;
      continue;
    }

    const double closureError = glm::length(audit.areaVectorSum) / audit.areaSum;
    double gaussErrorSquared = 0.0;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        const double target = (row == col) ? volume : 0.0;
        const double difference = audit.gaussSum[col * 3 + row] - target;
        gaussErrorSquared += difference * difference;
      }
    }
    const double gaussError = std::sqrt(gaussErrorSquared) / volume;
    const double errorBound =
        128.0 * std::numeric_limits<double>::epsilon() *
        audit.absTermSum / volume;

    if (closureError <= toleranceClosure &&
        (gaussError <= toleranceGauss || gaussError <= errorBound)) {
      continue;
    }

    const double baseClosure = audit.baseAreaSum > 0.0
        ? glm::length(audit.baseVectorSum) / audit.baseAreaSum
        : 0.0;
    const double surfaceClosure = audit.surfaceAreaSum > 0.0
        ? glm::length(audit.surfaceVectorSum) / audit.surfaceAreaSum
        : 0.0;

    if (reportedCount++ < MaxReportedFragments) {
      std::cerr << "[RVDFragments] fragment " << i
                << " owner=" << fragment.ownerModelId
                << " cell=" << fragment.cellId
                << " failed geometry validation:"
                << " volume=" << volume
                << " closureErr=" << closureError
                << " (tol " << toleranceClosure << ")"
                << " gaussErr=" << gaussError
                << " (tol " << toleranceGauss << ", bound " << errorBound << ")"
                << std::endl;
    }
    valid = false;
  }
  return valid;
}

} // namespace
bool validateContactCouplings(
    const RVDResult &result,
    double relativeTolerance) {
  const std::vector<FragmentFace> &faces = result.fragmentFaces;
  std::vector<double> coverage(faces.size() * 2, 0.0);
  std::vector<double> maxSingleCoupling(faces.size() * 2, 0.0);
  std::vector<uint32_t> couplingCounts(faces.size() * 2, 0);

  bool valid = true;
  for (const ContactCoupling &coupling : result.contactCouplings) {
    const uint32_t sideIndexA = coupling.faceA < faces.size()
        ? (faces[coupling.faceA].fragmentA == coupling.fragmentA
               ? coupling.faceA * 2
               : (faces[coupling.faceA].fragmentB == coupling.fragmentA
                      ? coupling.faceA * 2 + 1
                      : InvalidId))
        : InvalidId;
    const uint32_t sideIndexB = coupling.faceB < faces.size()
        ? (faces[coupling.faceB].fragmentA == coupling.fragmentB
               ? coupling.faceB * 2
               : (faces[coupling.faceB].fragmentB == coupling.fragmentB
                      ? coupling.faceB * 2 + 1
                      : InvalidId))
        : InvalidId;
    if (sideIndexA == InvalidId || sideIndexB == InvalidId ||
        coupling.faceA == coupling.faceB) {
      std::cerr << "[RVD-ContactValidation] invalid fragment-side pair "
                << coupling.faceA << ':' << coupling.fragmentA << " -> "
                << coupling.faceB << ':' << coupling.fragmentB << std::endl;
      valid = false;
      continue;
    }
    if (!(coupling.couplingArea > 0.0) ||
        !std::isfinite(coupling.couplingArea)) {
      std::cerr << "[RVD-ContactValidation] invalid coupling area "
                << coupling.couplingArea << " for "
                << coupling.faceA << " -> " << coupling.faceB << std::endl;
      valid = false;
      continue;
    }
    const FragmentFace &faceA = faces[coupling.faceA];
    const FragmentFace &faceB = faces[coupling.faceB];
    const uint32_t ownerA = result.fragments[coupling.fragmentA].ownerModelId;
    const uint32_t ownerB = result.fragments[coupling.fragmentB].ownerModelId;
    if (faceA.type == FaceType::DomainBoundary ||
        faceB.type == FaceType::DomainBoundary ||
        ownerA == InvalidModelId || ownerB == InvalidModelId ||
        ownerA == ownerB) {
      std::cerr << "[RVD-ContactValidation] coupling must join distinct "
                   "solid-side incidences "
                << coupling.faceA << ':' << coupling.fragmentA << " -> "
                << coupling.faceB << ':' << coupling.fragmentB << std::endl;
      valid = false;
      continue;
    }
    coverage[sideIndexA] += coupling.couplingArea;
    coverage[sideIndexB] += coupling.couplingArea;
    if (coupling.couplingArea > maxSingleCoupling[sideIndexA]) {
      maxSingleCoupling[sideIndexA] = coupling.couplingArea;
    }
    if (coupling.couplingArea > maxSingleCoupling[sideIndexB]) {
      maxSingleCoupling[sideIndexB] = coupling.couplingArea;
    }
    ++couplingCounts[sideIndexA];
    ++couplingCounts[sideIndexB];
  }

  uint32_t overcoveredCount = 0;
  uint32_t singleExceedsCount = 0;
  uint32_t multiExceedsCount = 0;
  double worstOvercoverageRatio = 1.0;
  uint32_t worstFaceIndex = InvalidId;
  uint32_t worstSide = 0;
  double worstArea = 0.0;
  double worstCoverage = 0.0;
  double worstMaxSingle = 0.0;
  uint32_t worstCouplingCount = 0;

  for (uint32_t faceIndex = 0; faceIndex < faces.size(); ++faceIndex) {
    for (uint32_t side = 0; side < 2; ++side) {
      const uint32_t incidence = faceIndex * 2 + side;
      if (coverage[incidence] <= 0.0) continue;
      const double area = side == 0 ? faces[faceIndex].areaA
                                    : faces[faceIndex].areaB;
      if (!(area > 0.0)) {
        std::cerr << "[RVD-ContactValidation] face-side " << faceIndex
                  << ':' << side << " has contact coverage but no area"
                  << std::endl;
        valid = false;
        continue;
      }
      const double allowed = area * (1.0 + relativeTolerance);
      if (coverage[incidence] > allowed) {
        valid = false;
        ++overcoveredCount;
        const double ratio = coverage[incidence] / area;
        const bool singleExceeds = maxSingleCoupling[incidence] > allowed;
        if (singleExceeds) {
          ++singleExceedsCount;
        } else {
          ++multiExceedsCount;
        }
        if (ratio > worstOvercoverageRatio) {
          worstOvercoverageRatio = ratio;
          worstFaceIndex = faceIndex;
          worstSide = side;
          worstArea = area;
          worstCoverage = coverage[incidence];
          worstMaxSingle = maxSingleCoupling[incidence];
          worstCouplingCount = couplingCounts[incidence];
        }
      }
    }
  }

  if (overcoveredCount > 0) {
    std::cerr << "[RVD-ContactValidation] Summary: " << overcoveredCount
              << " overcovered face-sides ("
              << singleExceedsCount << " where single coupling > face area, "
              << multiExceedsCount << " where multiple valid couplings sum > face area).\n"
              << "[RVD-ContactValidation] Worst case face-side " << worstFaceIndex
              << ':' << worstSide << " area=" << worstArea
              << " coverage=" << worstCoverage
              << " (" << worstOvercoverageRatio << "x)"
              << " couplings=" << worstCouplingCount
              << " maxSingle=" << worstMaxSingle
              << " (" << (worstMaxSingle / worstArea) << "x)"
              << std::endl;
  }
  return valid;
}

bool validateFragmentTopology(
    const RVDResult &result,
    uint32_t totalBaseFragmentFaces,
    double toleranceClosure,
    double toleranceGauss) {
  if (result.fragments.empty()) return true;

  for (uint32_t faceIndex = 0;
       faceIndex < result.fragmentFaces.size(); ++faceIndex) {
    const FragmentFace &face = result.fragmentFaces[faceIndex];
    const bool hasA = face.fragmentA < result.fragments.size();
    const bool hasB = face.fragmentB < result.fragments.size();
    if (face.type == FaceType::Shared && (!hasA || !hasB)) {
      std::cerr << "[RVDFaces] Shared face " << faceIndex
                << " base=" << (faceIndex < totalBaseFragmentFaces)
                << " has invalid incidence A=" << face.fragmentA
                << " B=" << face.fragmentB;
      if (hasA) {
        const Fragment &fragment = result.fragments[face.fragmentA];
        std::cerr << " Acell=" << fragment.cellId
                  << " Aowner=" << fragment.ownerModelId
                  << " Avol=" << fragment.volume;
      }
      if (hasB) {
        const Fragment &fragment = result.fragments[face.fragmentB];
        std::cerr << " Bcell=" << fragment.cellId
                  << " Bowner=" << fragment.ownerModelId
                  << " Bvol=" << fragment.volume;
      }
      std::cerr << std::endl;
      return false;
    }
  }

  if (!validateSharedFaceReciprocity(result.fragmentFaces)) return false;
  if (!validateFragmentClosure(
          result.fragments,
          auditFragmentFaces(
              result.fragments, result.fragmentFaces, totalBaseFragmentFaces),
          toleranceClosure,
          toleranceGauss)) {
    return false;
  }
  return true;
}

} // namespace voronoi
