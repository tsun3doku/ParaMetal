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
                  << audit.incidentFaceCount << ")" << std::endl;
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
                << " baseArea=" << audit.baseAreaSum
                << " baseClosure=" << baseClosure
                << " surfaceArea=" << audit.surfaceAreaSum
                << " surfaceClosure=" << surfaceClosure
                << " gaussErr=" << gaussError
                << " (tol " << toleranceGauss << ", bound " << errorBound << ")"
                << std::endl;
    }
    valid = false;
  }
  return valid;
}

} // namespace

bool validateFragmentTopology(
    const RVDResult &result,
    uint32_t totalBaseFragmentFaces,
    double toleranceClosure,
    double toleranceGauss) {
  if (result.fragments.empty()) return true;

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
