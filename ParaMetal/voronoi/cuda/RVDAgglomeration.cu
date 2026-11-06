#include "RVDAgglomeration.cuh"
#include "RVDHostUtils.cuh"
#include "../../cuda/CudaMath.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace voronoi {

namespace {

struct DeviceFragment {
  uint32_t materialId;
  uint32_t faceCount;
  double volume;
  double normalCovariance[6];
  double conditionNumber;
  uint8_t needsMerge;
};

struct DeviceFragmentFace {
  uint32_t fragmentA;
  uint32_t fragmentB;
  double areaA;
  double areaB;
  double3 areaVectorA;
  double3 areaVectorB;
  double3 momentA;
  double3 momentB;
  uint8_t type;
  uint32_t domainFace;
};

__device__ inline double evaluateConditionNumberDevice(const double *B) {
  const double trB = B[0] + B[1] + B[2];
  if (!(trB > 0.0) || !isfinite(trB)) return 1e12;
  const double q = trB / 3.0;
  const double p =
      ((B[0] - q) * (B[0] - q) + (B[1] - q) * (B[1] - q) +
       (B[2] - q) * (B[2] - q) +
       2.0 * (B[3] * B[3] + B[4] * B[4] + B[5] * B[5])) / 6.0;
  if (p < 1e-14 * q * q) return 1.0;
  const double sqrtP = sqrt(p);
  const double b00 = (B[0] - q) / sqrtP;
  const double b11 = (B[1] - q) / sqrtP;
  const double b22 = (B[2] - q) / sqrtP;
  const double b01 = B[3] / sqrtP;
  const double b02 = B[4] / sqrtP;
  const double b12 = B[5] / sqrtP;
  const double detBnorm = b00 * (b11 * b22 - b12 * b12) -
                          b01 * (b01 * b22 - b12 * b02) +
                          b02 * (b01 * b12 - b11 * b02);
  const double clamped = fmax(-1.0, fmin(1.0, detBnorm * 0.5));
  const double phi = acos(clamped) / 3.0;
  constexpr double pi23 = 2.0 * 3.14159265358979323846 / 3.0;
  const double lam0 = q + 2.0 * sqrtP * cos(phi);
  const double lam1 = q + 2.0 * sqrtP * cos(phi + pi23);
  const double lam2 = q + 2.0 * sqrtP * cos(phi + 2.0 * pi23);
  const double lamMax = fmax(lam0, fmax(lam1, lam2));
  const double lamMin = fmin(lam0, fmin(lam1, lam2));
  if (lamMin <= 1e-12 * lamMax || !isfinite(lamMin)) return 1e12;
  return lamMax / lamMin;
}

__global__ void accumulateFaceStatsKernel(
    const DeviceFragmentFace *faces,
    uint32_t numFaces,
    const uint32_t *parent,
    uint32_t numFragments,
    double *devCovariance,
    uint32_t *devFaceCount) {
  uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= numFaces) return;

  const DeviceFragmentFace &face = faces[idx];
  uint32_t rootA = (face.fragmentA < numFragments) ? parent[face.fragmentA] : InvalidId;
  uint32_t rootB = (face.fragmentB < numFragments) ? parent[face.fragmentB] : InvalidId;

  if (rootA == rootB && rootA != InvalidId) return;

  if (rootA < numFragments && face.areaA > 0.0) {
    atomicAdd(&devFaceCount[rootA], 1u);
    const double3 &n = face.areaVectorA;
    atomicAdd(&devCovariance[rootA * 6 + 0], n.x * n.x);
    atomicAdd(&devCovariance[rootA * 6 + 1], n.y * n.y);
    atomicAdd(&devCovariance[rootA * 6 + 2], n.z * n.z);
    atomicAdd(&devCovariance[rootA * 6 + 3], n.x * n.y);
    atomicAdd(&devCovariance[rootA * 6 + 4], n.x * n.z);
    atomicAdd(&devCovariance[rootA * 6 + 5], n.y * n.z);
  }
  if (rootB < numFragments && face.areaB > 0.0) {
    atomicAdd(&devFaceCount[rootB], 1u);
    const double3 &n = face.areaVectorB;
    atomicAdd(&devCovariance[rootB * 6 + 0], n.x * n.x);
    atomicAdd(&devCovariance[rootB * 6 + 1], n.y * n.y);
    atomicAdd(&devCovariance[rootB * 6 + 2], n.z * n.z);
    atomicAdd(&devCovariance[rootB * 6 + 3], n.x * n.y);
    atomicAdd(&devCovariance[rootB * 6 + 4], n.x * n.z);
    atomicAdd(&devCovariance[rootB * 6 + 5], n.y * n.z);
  }
}

__global__ void evaluateFragmentQualityKernel(
    DeviceFragment *fragments,
    uint32_t numFragments,
    const uint32_t *parent,
    const double *devCovariance,
    const uint32_t *devFaceCount,
    double minVolumeFloor,
    uint32_t *devSliverCount) {
  uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= numFragments) return;

  if (parent[idx] != idx) {
    fragments[idx].needsMerge = 0;
    return;
  }

  fragments[idx].faceCount = devFaceCount[idx];
  for (int k = 0; k < 6; ++k) {
    fragments[idx].normalCovariance[k] = devCovariance[idx * 6 + k];
  }

  const double cond = evaluateConditionNumberDevice(fragments[idx].normalCovariance);
  fragments[idx].conditionNumber = cond;

  bool bad = false;
  if (!(fragments[idx].volume >= minVolumeFloor) || !isfinite(fragments[idx].volume)) {
    bad = true;
  } else if (fragments[idx].faceCount < 4) {
    bad = true;
  } else if (cond > 1e5) {
    bad = true;
  }

  fragments[idx].needsMerge = bad ? 1 : 0;
  if (bad) {
    atomicAdd(devSliverCount, 1u);
  }
}

__global__ void emitBaseFacePairKeysKernel(
    const DeviceFragmentFace *faces,
    uint32_t totalBaseFaces,
    const uint32_t *parent,
    const DeviceFragment *fragments,
    uint32_t numFragments,
    uint64_t *devPairKeysIn,
    uint32_t *devPairFacesIn) {
  uint32_t f = blockDim.x * blockIdx.x + threadIdx.x;
  if (f >= totalBaseFaces) return;

  const DeviceFragmentFace &face = faces[f];
  if (face.fragmentA >= numFragments || face.fragmentB >= numFragments) {
    devPairKeysIn[f] = UINT64_MAX;
    devPairFacesIn[f] = f;
    return;
  }
  uint32_t rA = parent[face.fragmentA];
  uint32_t rB = parent[face.fragmentB];
  if (rA == rB || rA >= numFragments || rB >= numFragments ||
      fragments[rA].materialId != fragments[rB].materialId ||
      (fragments[rA].needsMerge == 0 && fragments[rB].needsMerge == 0)) {
    devPairKeysIn[f] = UINT64_MAX;
    devPairFacesIn[f] = f;
    return;
  }

  uint32_t u = (rA < rB) ? rA : rB;
  uint32_t v = (rA < rB) ? rB : rA;
  devPairKeysIn[f] = (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(v);
  devPairFacesIn[f] = f;
}

__global__ void markPairRunsKernel(
    const uint64_t *pairKeys,
    uint32_t totalBaseFaces,
    uint32_t *runFlags) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= totalBaseFaces) return;

  if (pairKeys[i] == UINT64_MAX) {
    runFlags[i] = 0u;
    return;
  }
  runFlags[i] = (i == 0 || pairKeys[i] != pairKeys[i - 1]) ? 1u : 0u;
}

__global__ void scatterPairOffsetsKernel(
    const uint64_t *pairKeys,
    const uint32_t *runFlags,
    const uint32_t *runOffsets,
    uint32_t totalBaseFaces,
    uint32_t numPairs,
    uint32_t *pairOffsets) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= totalBaseFaces) return;

  if (runFlags[i] == 1u) {
    pairOffsets[runOffsets[i]] = i;
  }
  if (pairKeys[i] == UINT64_MAX && (i == 0 || pairKeys[i - 1] != UINT64_MAX)) {
    pairOffsets[numPairs] = i;
  }
}

__global__ void evaluatePairCandidatesKernel(
    uint32_t numPairs,
    const uint32_t *pairOffsets,
    const uint64_t *pairKeys,
    const uint32_t *pairFaces,
    const DeviceFragmentFace *faces,
    const DeviceFragment *fragments,
    uint64_t *bestProposal) {
  uint32_t p = blockDim.x * blockIdx.x + threadIdx.x;
  if (p >= numPairs) return;

  uint32_t start = pairOffsets[p];
  uint32_t end = pairOffsets[p + 1];
  if (start >= end) return;

  uint64_t key = pairKeys[start];
  uint32_t u = static_cast<uint32_t>(key >> 32);
  uint32_t v = static_cast<uint32_t>(key & 0xFFFFFFFF);

  double sharedArea = 0.0;
  double sharedCov[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  for (uint32_t idx = start; idx < end; ++idx) {
    uint32_t f = pairFaces[idx];
    const DeviceFragmentFace &face = faces[f];
    sharedArea += face.areaA;
    const double3 &n = face.areaVectorA;
    sharedCov[0] += n.x * n.x;
    sharedCov[1] += n.y * n.y;
    sharedCov[2] += n.z * n.z;
    sharedCov[3] += n.x * n.y;
    sharedCov[4] += n.x * n.z;
    sharedCov[5] += n.y * n.z;
  }

  double unionCov[6];
  for (int m = 0; m < 6; ++m) {
    unionCov[m] = fragments[u].normalCovariance[m] +
                  fragments[v].normalCovariance[m] -
                  2.0 * sharedCov[m];
  }
  const double unionCond = evaluateConditionNumberDevice(unionCov);
  const double condU = fragments[u].conditionNumber;
  const double condV = fragments[v].conditionNumber;

  const bool acceptable = (unionCond < 1e5) || (unionCond < fmin(condU, condV));
  if (!acceptable || !(sharedArea > 0.0) || !isfinite(sharedArea)) return;

  uint32_t score = __float_as_uint(static_cast<float>(sharedArea));

  if (fragments[u].needsMerge != 0) {
    atomicMax(reinterpret_cast<unsigned long long *>(&bestProposal[u]),
              (static_cast<uint64_t>(score) << 32) | static_cast<uint64_t>(v));
  }
  if (fragments[v].needsMerge != 0) {
    atomicMax(reinterpret_cast<unsigned long long *>(&bestProposal[v]),
              (static_cast<uint64_t>(score) << 32) | static_cast<uint64_t>(u));
  }
}

__global__ void submitClaimsKernel(
    uint32_t numFragments,
    const DeviceFragment *fragments,
    const uint64_t *bestProposal,
    uint64_t *bestClaim) {
  uint32_t u = blockDim.x * blockIdx.x + threadIdx.x;
  if (u >= numFragments) return;
  if (fragments[u].needsMerge == 0) return;

  uint64_t prop = bestProposal[u];
  if (prop == 0) return;

  uint32_t v = static_cast<uint32_t>(prop);
  uint32_t score = static_cast<uint32_t>(prop >> 32);

  atomicMax(reinterpret_cast<unsigned long long *>(&bestClaim[v]),
            (static_cast<uint64_t>(score) << 32) | static_cast<uint64_t>(u));
}

__global__ void contractDisjointEdgesKernel(
    uint32_t numFragments,
    DeviceFragment *fragments,
    uint32_t *parent,
    const uint64_t *bestProposal,
    const uint64_t *bestClaim,
    uint32_t *devMergesCount) {
  uint32_t u = blockDim.x * blockIdx.x + threadIdx.x;
  if (u >= numFragments) return;

  if (parent[u] != u || fragments[u].needsMerge == 0) return;

  uint64_t prop = bestProposal[u];
  if (prop == 0) return;

  uint32_t v = static_cast<uint32_t>(prop);
  if (v >= numFragments || parent[v] != v) return;

  bool accepted = false;
  if (fragments[v].needsMerge != 0) {
    uint64_t propV = bestProposal[v];
    if (static_cast<uint32_t>(propV) == u) {
      if (u < v) {
        accepted = true;
      }
    }
  } else {
    uint64_t claimV = bestClaim[v];
    if (static_cast<uint32_t>(claimV) == u) {
      accepted = true;
    }
  }

  if (accepted) {
    parent[u] = v;
    atomicAdd(&fragments[v].volume, fragments[u].volume);
    atomicAdd(devMergesCount, 1u);
  }
}

__global__ void flattenPointersKernel(
    uint32_t numFragments,
    uint32_t *parent) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= numFragments) return;

  uint32_t p = parent[i];
  while (parent[p] != p) {
    p = parent[p];
  }
  parent[i] = p;
}

__global__ void markKeptFacesKernel(
    const DeviceFragmentFace *faces,
    uint32_t numFaces,
    const uint32_t *oldFragToNew,
    uint32_t numFragments,
    uint32_t *devFaceKeep,
    uint32_t *devFaceNewA,
    uint32_t *devFaceNewB) {
  uint32_t f = blockDim.x * blockIdx.x + threadIdx.x;
  if (f >= numFaces) return;

  const DeviceFragmentFace &face = faces[f];
  uint32_t newA = (face.fragmentA < numFragments) ? oldFragToNew[face.fragmentA] : InvalidId;
  uint32_t newB = (face.fragmentB < numFragments) ? oldFragToNew[face.fragmentB] : InvalidId;

  if (newA == newB && newA != InvalidId) {
    devFaceKeep[f] = 0u;
    devFaceNewA[f] = InvalidId;
    devFaceNewB[f] = InvalidId;
  } else {
    devFaceKeep[f] = 1u;
    devFaceNewA[f] = newA;
    devFaceNewB[f] = newB;
  }
}

__global__ void scatterCompactedFacesKernel(
    const DeviceFragmentFace *faces,
    uint32_t numFaces,
    const uint32_t *devFaceKeep,
    const uint32_t *devFaceOffsets,
    const uint32_t *devFaceNewA,
    const uint32_t *devFaceNewB,
    DeviceFragmentFace *compactedFaces,
    uint32_t *oldFaceToNew) {
  uint32_t f = blockDim.x * blockIdx.x + threadIdx.x;
  if (f >= numFaces) return;

  if (devFaceKeep[f] == 1u) {
    uint32_t outIdx = devFaceOffsets[f];
    DeviceFragmentFace face = faces[f];
    face.fragmentA = devFaceNewA[f];
    face.fragmentB = devFaceNewB[f];
    compactedFaces[outIdx] = face;
    oldFaceToNew[f] = outIdx;
  } else {
    oldFaceToNew[f] = InvalidId;
  }
}

__global__ void emitHalfFaceKeysKernel(
    const DeviceFragmentFace *faces,
    uint32_t numFaces,
    uint32_t numCompacted,
    uint32_t *halfFaceKeys,
    uint32_t *halfFaceValues) {
  uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= numFaces * 2) return;

  uint32_t f = idx >> 1;
  uint32_t side = idx & 1u;
  const DeviceFragmentFace &face = faces[f];

  uint32_t frag = (side == 0) ? face.fragmentA : face.fragmentB;
  double area = (side == 0) ? face.areaA : face.areaB;

  if (frag < numCompacted && area > 0.0) {
    halfFaceKeys[idx] = frag;
  } else {
    halfFaceKeys[idx] = InvalidId;
  }
  halfFaceValues[idx] = idx;
}

__global__ void markSegmentBoundsKernel(
    const uint32_t *keys,
    uint32_t totalHalfFaces,
    uint32_t numCompacted,
    uint32_t *segmentStart,
    uint32_t *segmentEnd) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= totalHalfFaces) return;

  uint32_t k = keys[i];
  if (k >= numCompacted) return;

  if (i == 0 || keys[i - 1] != k) {
    segmentStart[k] = i;
  }
  if (i == totalHalfFaces - 1 || keys[i + 1] != k) {
    segmentEnd[k] = i + 1;
  }
}

__global__ void integrateSegmentedBoundaryKernel(
    uint32_t numCompacted,
    const uint32_t *segmentStart,
    const uint32_t *segmentEnd,
    const uint32_t *halfFaceValues,
    const DeviceFragmentFace *faces,
    const double3 *seedPositions,
    double *devBoundaryVolumes,
    double3 *devBoundaryMoments) {
  uint32_t a = blockDim.x * blockIdx.x + threadIdx.x;
  if (a >= numCompacted) return;

  uint32_t start = segmentStart[a];
  uint32_t end = segmentEnd[a];
  if (start >= end) return;

  const double3 ref = seedPositions[a];
  double vol = 0.0;
  double volComp = 0.0;
  double3 mom = make_double3(0.0, 0.0, 0.0);
  double3 momComp = make_double3(0.0, 0.0, 0.0);

  for (uint32_t idx = start; idx < end; ++idx) {
    uint32_t val = halfFaceValues[idx];
    uint32_t f = val >> 1;
    uint32_t side = val & 1u;
    const DeviceFragmentFace &face = faces[f];

    double area = (side == 0) ? face.areaA : face.areaB;
    if (area <= 0.0) continue;
    const double3 &areaVec = (side == 0) ? face.areaVectorA : face.areaVectorB;
    const double3 &moment = (side == 0) ? face.momentA : face.momentB;

    double3 x_f = cudaMath::mul3(moment, 1.0 / area);
    double3 r_f = cudaMath::sub3(x_f, ref);
    double vPyr = cudaMath::dot3(r_f, areaVec) / 3.0;

    double yV = vPyr - volComp;
    double tV = vol + yV;
    volComp = (tV - vol) - yV;
    vol = tV;

    double3 dM = cudaMath::mul3(r_f, vPyr * 0.75);
    double yMx = dM.x - momComp.x;
    double tMx = mom.x + yMx;
    momComp.x = (tMx - mom.x) - yMx;
    mom.x = tMx;

    double yMy = dM.y - momComp.y;
    double tMy = mom.y + yMy;
    momComp.y = (tMy - mom.y) - yMy;
    mom.y = tMy;

    double yMz = dM.z - momComp.z;
    double tMz = mom.z + yMz;
    momComp.z = (tMz - mom.z) - yMz;
    mom.z = tMz;
  }

  devBoundaryVolumes[a] = vol;
  devBoundaryMoments[a] = mom;
}

} // namespace

bool agglomerateFragments(
    const BaseCellDomain &baseDomain,
    std::vector<SurfaceRegion> &surfaceRegions,
    uint32_t &totalBaseFragmentFaces,
    RVDResult &result,
    cudaStream_t stream) {
  if (result.fragments.empty()) return true;

  const uint32_t numFragments = static_cast<uint32_t>(result.fragments.size());
  const uint32_t numFaces = static_cast<uint32_t>(result.fragmentFaces.size());
  const double nominalSize = double(baseDomain.nominalCellSize);
  if (!(nominalSize > 0.0) || !std::isfinite(nominalSize)) {
    std::cerr << "[RVDAgglomeration] invalid nominalCellSize=" << baseDomain.nominalCellSize << std::endl;
    return false;
  }
  const double nominalVolume = nominalSize * nominalSize * nominalSize;
  const double minVolumeFloor = 1e-4 * nominalVolume;

  std::vector<DeviceFragment> hostFragments(numFragments);
  std::vector<uint32_t> hostCellIds(numFragments);
  for (uint32_t i = 0; i < numFragments; ++i) {
    hostCellIds[i] = result.fragments[i].cellId;
    hostFragments[i].materialId = result.fragments[i].ownerModelId;
    hostFragments[i].faceCount = 0;
    hostFragments[i].volume = result.fragments[i].volume;
    for (int k = 0; k < 6; ++k) hostFragments[i].normalCovariance[k] = 0.0;
    hostFragments[i].conditionNumber = 0.0;
    hostFragments[i].needsMerge = 0;
  }

  std::vector<DeviceFragmentFace> hostFaces(numFaces);
  for (uint32_t f = 0; f < numFaces; ++f) {
    const FragmentFace &ff = result.fragmentFaces[f];
    hostFaces[f].fragmentA = ff.fragmentA;
    hostFaces[f].fragmentB = ff.fragmentB;
    hostFaces[f].areaA = ff.areaA;
    hostFaces[f].areaB = ff.areaB;
    hostFaces[f].areaVectorA = make_double3(ff.areaVectorA.x, ff.areaVectorA.y, ff.areaVectorA.z);
    hostFaces[f].areaVectorB = make_double3(ff.areaVectorB.x, ff.areaVectorB.y, ff.areaVectorB.z);
    hostFaces[f].momentA = make_double3(ff.momentA.x, ff.momentA.y, ff.momentA.z);
    hostFaces[f].momentB = make_double3(ff.momentB.x, ff.momentB.y, ff.momentB.z);
    hostFaces[f].type = static_cast<uint8_t>(ff.type);
    hostFaces[f].domainFace = ff.domainFace;
  }

  cudaUtils::CudaBuffer<DeviceFragment> devFragments;
  cudaUtils::CudaBuffer<DeviceFragmentFace> devFaces;
  cudaUtils::CudaBuffer<uint32_t> devParent;
  cudaUtils::CudaBuffer<double> devCovariance;
  cudaUtils::CudaBuffer<uint32_t> devFaceCount;
  cudaUtils::CudaBuffer<uint64_t> devPairKeysIn;
  cudaUtils::CudaBuffer<uint64_t> devPairKeysOut;
  cudaUtils::CudaBuffer<uint32_t> devPairFacesIn;
  cudaUtils::CudaBuffer<uint32_t> devPairFacesOut;
  cudaUtils::CudaBuffer<uint32_t> devRunFlags;
  cudaUtils::CudaBuffer<uint32_t> devRunOffsets;
  cudaUtils::CudaBuffer<uint32_t> devPairOffsets;
  cudaUtils::CudaBuffer<uint64_t> devBestProposal;
  cudaUtils::CudaBuffer<uint64_t> devBestClaim;
  cudaUtils::CudaBuffer<uint32_t> devFlags;
  cudaUtils::CudaBuffer<uint8_t> scratch;

  const uint32_t pairCapacity = totalBaseFragmentFaces + 1;
  if (!devFragments.allocate(numFragments) ||
      !devFaces.allocate(numFaces) ||
      !devParent.allocate(numFragments) ||
      !devCovariance.allocate(numFragments * 6) ||
      !devFaceCount.allocate(numFragments) ||
      !devPairKeysIn.allocate(pairCapacity) ||
      !devPairKeysOut.allocate(pairCapacity) ||
      !devPairFacesIn.allocate(pairCapacity) ||
      !devPairFacesOut.allocate(pairCapacity) ||
      !devRunFlags.allocate(pairCapacity) ||
      !devRunOffsets.allocate(pairCapacity) ||
      !devPairOffsets.allocate(pairCapacity) ||
      !devBestProposal.allocate(numFragments) ||
      !devBestClaim.allocate(numFragments) ||
      !devFlags.allocate(4)) {
    return false;
  }

  if (!devFragments.uploadAsync(hostFragments.data(), numFragments, stream) ||
      !devFaces.uploadAsync(hostFaces.data(), numFaces, stream)) {
    return false;
  }

  std::vector<uint32_t> initialParent(numFragments);
  for (uint32_t i = 0; i < numFragments; ++i) initialParent[i] = i;
  if (!devParent.uploadAsync(initialParent.data(), numFragments, stream)) {
    return false;
  }

  uint32_t totalMerges = 0;
  uint32_t initialDeficient = 0;
  constexpr uint32_t MaxMergeIterations = 50;

  for (uint32_t iter = 0; iter < MaxMergeIterations; ++iter) {
    cudaMemsetAsync(devCovariance.get(), 0, numFragments * 6 * sizeof(double), stream);
    cudaMemsetAsync(devFaceCount.get(), 0, numFragments * sizeof(uint32_t), stream);
    cudaMemsetAsync(devFlags.get(), 0, 4 * sizeof(uint32_t), stream); // [0]: sliverCount, [1]: mergesCount

    accumulateFaceStatsKernel<<<rvdHost::blockCount(numFaces), rvdHost::BlockSize, 0, stream>>>(
        devFaces.get(), numFaces, devParent.get(), numFragments,
        devCovariance.get(), devFaceCount.get());

    evaluateFragmentQualityKernel<<<rvdHost::blockCount(numFragments), rvdHost::BlockSize, 0, stream>>>(
        devFragments.get(), numFragments, devParent.get(),
        devCovariance.get(), devFaceCount.get(),
        minVolumeFloor, devFlags.get());

    uint32_t hostSliverCount = 0;
    cudaMemcpyAsync(&hostSliverCount, devFlags.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (iter == 0) {
      initialDeficient = hostSliverCount;
    }

    if (hostSliverCount == 0 || totalBaseFragmentFaces == 0) {
      break;
    }

    emitBaseFacePairKeysKernel<<<rvdHost::blockCount(totalBaseFragmentFaces), rvdHost::BlockSize, 0, stream>>>(
        devFaces.get(), totalBaseFragmentFaces, devParent.get(),
        devFragments.get(), numFragments,
        devPairKeysIn.get(), devPairFacesIn.get());

    if (!rvdHost::sortPairs(devPairKeysIn.get(), devPairKeysOut.get(),
                            devPairFacesIn.get(), devPairFacesOut.get(),
                            totalBaseFragmentFaces, scratch, stream)) {
      return false;
    }

    markPairRunsKernel<<<rvdHost::blockCount(totalBaseFragmentFaces), rvdHost::BlockSize, 0, stream>>>(
        devPairKeysOut.get(), totalBaseFragmentFaces, devRunFlags.get());
    cudaMemsetAsync(devRunFlags.get() + totalBaseFragmentFaces, 0, sizeof(uint32_t), stream);

    if (!rvdHost::exclusiveSum(devRunFlags.get(), devRunOffsets.get(),
                               totalBaseFragmentFaces + 1, scratch, stream)) {
      return false;
    }

    uint32_t hostNumPairs = 0;
    cudaMemcpyAsync(&hostNumPairs, devRunOffsets.get() + totalBaseFragmentFaces, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (hostNumPairs == 0) {
      break;
    }

    cudaMemcpyAsync(devPairOffsets.get() + hostNumPairs, &totalBaseFragmentFaces, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    scatterPairOffsetsKernel<<<rvdHost::blockCount(totalBaseFragmentFaces), rvdHost::BlockSize, 0, stream>>>(
        devPairKeysOut.get(), devRunFlags.get(), devRunOffsets.get(),
        totalBaseFragmentFaces, hostNumPairs, devPairOffsets.get());

    cudaMemsetAsync(devBestProposal.get(), 0, numFragments * sizeof(uint64_t), stream);
    cudaMemsetAsync(devBestClaim.get(), 0, numFragments * sizeof(uint64_t), stream);

    evaluatePairCandidatesKernel<<<rvdHost::blockCount(hostNumPairs), rvdHost::BlockSize, 0, stream>>>(
        hostNumPairs, devPairOffsets.get(), devPairKeysOut.get(), devPairFacesOut.get(),
        devFaces.get(), devFragments.get(), devBestProposal.get());

    submitClaimsKernel<<<rvdHost::blockCount(numFragments), rvdHost::BlockSize, 0, stream>>>(
        numFragments, devFragments.get(), devBestProposal.get(), devBestClaim.get());

    contractDisjointEdgesKernel<<<rvdHost::blockCount(numFragments), rvdHost::BlockSize, 0, stream>>>(
        numFragments, devFragments.get(), devParent.get(),
        devBestProposal.get(), devBestClaim.get(), devFlags.get() + 1);

    uint32_t hostMergesCount = 0;
    cudaMemcpyAsync(&hostMergesCount, devFlags.get() + 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (hostMergesCount == 0) break;

    totalMerges += hostMergesCount;
    flattenPointersKernel<<<rvdHost::blockCount(numFragments), rvdHost::BlockSize, 0, stream>>>(
        numFragments, devParent.get());
  }

  if (totalMerges == 0) {
    if (initialDeficient > 0) {
      std::cerr << "[RVDAgglomeration] 0 merges performed ("
                << initialDeficient << " deficient fragments remain)" << std::endl;
    }
    return true;
  }

  std::vector<uint32_t> hostParent(numFragments);
  devParent.downloadAsync(hostParent.data(), numFragments, stream);
  cudaStreamSynchronize(stream);

  std::vector<uint32_t> oldFragToNew(numFragments, InvalidId);
  std::vector<Fragment> compactedFragments;
  compactedFragments.reserve(numFragments);
  std::vector<double3> hostSeedPositions;
  hostSeedPositions.reserve(numFragments);

  for (uint32_t i = 0; i < numFragments; ++i) {
    if (hostParent[i] == i) {
      oldFragToNew[i] = static_cast<uint32_t>(compactedFragments.size());
      Fragment f = result.fragments[i];
      hostSeedPositions.push_back(make_double3(f.centroid.x, f.centroid.y, f.centroid.z));
      f.volume = 0.0;
      f.centroid = glm::dvec3(0.0);
      compactedFragments.push_back(f);
    }
  }
  for (uint32_t i = 0; i < numFragments; ++i) {
    oldFragToNew[i] = oldFragToNew[hostParent[i]];
  }

  const uint32_t numCompacted = static_cast<uint32_t>(compactedFragments.size());

  cudaUtils::CudaBuffer<uint32_t> devOldFragToNew;
  cudaUtils::CudaBuffer<uint32_t> devFaceKeep;
  cudaUtils::CudaBuffer<uint32_t> devFaceOffsets;
  cudaUtils::CudaBuffer<uint32_t> devFaceNewA;
  cudaUtils::CudaBuffer<uint32_t> devFaceNewB;
  cudaUtils::CudaBuffer<uint32_t> devOldFaceToNew;
  cudaUtils::CudaBuffer<DeviceFragmentFace> devCompactedFaces;
  cudaUtils::CudaBuffer<double3> devSeedPositions;
  cudaUtils::CudaBuffer<double> devBoundaryVolumes;
  cudaUtils::CudaBuffer<double3> devBoundaryMoments;

  if (!devOldFragToNew.allocate(numFragments) ||
      !devFaceKeep.allocate(numFaces + 1) ||
      !devFaceOffsets.allocate(numFaces + 1) ||
      !devFaceNewA.allocate(numFaces) ||
      !devFaceNewB.allocate(numFaces) ||
      !devOldFaceToNew.allocate(numFaces) ||
      !devCompactedFaces.allocate(numFaces) ||
      !devSeedPositions.allocate(numCompacted) ||
      !devBoundaryVolumes.allocate(numCompacted) ||
      !devBoundaryMoments.allocate(numCompacted)) {
    return false;
  }

  devOldFragToNew.uploadAsync(oldFragToNew.data(), numFragments, stream);
  devSeedPositions.uploadAsync(hostSeedPositions.data(), numCompacted, stream);

  markKeptFacesKernel<<<rvdHost::blockCount(numFaces), rvdHost::BlockSize, 0, stream>>>(
      devFaces.get(), numFaces, devOldFragToNew.get(), numFragments,
      devFaceKeep.get(), devFaceNewA.get(), devFaceNewB.get());
  cudaMemsetAsync(devFaceKeep.get() + numFaces, 0, sizeof(uint32_t), stream);

  if (!rvdHost::exclusiveSum(devFaceKeep.get(), devFaceOffsets.get(), numFaces + 1, scratch, stream)) {
    return false;
  }

  uint32_t totalCompactedFaces = 0;
  uint32_t newTotalBaseFaces = 0;
  cudaMemcpyAsync(&totalCompactedFaces, devFaceOffsets.get() + numFaces, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&newTotalBaseFaces, devFaceOffsets.get() + totalBaseFragmentFaces, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  scatterCompactedFacesKernel<<<rvdHost::blockCount(numFaces), rvdHost::BlockSize, 0, stream>>>(
      devFaces.get(), numFaces, devFaceKeep.get(), devFaceOffsets.get(),
      devFaceNewA.get(), devFaceNewB.get(), devCompactedFaces.get(), devOldFaceToNew.get());

  const uint32_t totalHalfFaces = totalCompactedFaces * 2;
  cudaUtils::CudaBuffer<uint32_t> devHalfFaceKeysIn;
  cudaUtils::CudaBuffer<uint32_t> devHalfFaceKeysOut;
  cudaUtils::CudaBuffer<uint32_t> devHalfFaceValuesIn;
  cudaUtils::CudaBuffer<uint32_t> devHalfFaceValuesOut;
  cudaUtils::CudaBuffer<uint32_t> devSegmentStart;
  cudaUtils::CudaBuffer<uint32_t> devSegmentEnd;

  if (totalHalfFaces > 0) {
    if (!devHalfFaceKeysIn.allocate(totalHalfFaces) ||
        !devHalfFaceKeysOut.allocate(totalHalfFaces) ||
        !devHalfFaceValuesIn.allocate(totalHalfFaces) ||
        !devHalfFaceValuesOut.allocate(totalHalfFaces) ||
        !devSegmentStart.allocate(numCompacted) ||
        !devSegmentEnd.allocate(numCompacted)) {
      return false;
    }

    emitHalfFaceKeysKernel<<<rvdHost::blockCount(totalHalfFaces), rvdHost::BlockSize, 0, stream>>>(
        devCompactedFaces.get(), totalCompactedFaces, numCompacted,
        devHalfFaceKeysIn.get(), devHalfFaceValuesIn.get());

    if (!rvdHost::sortPairs(devHalfFaceKeysIn.get(), devHalfFaceKeysOut.get(),
                            devHalfFaceValuesIn.get(), devHalfFaceValuesOut.get(),
                            totalHalfFaces, scratch, stream)) {
      return false;
    }

    cudaMemsetAsync(devSegmentStart.get(), 0, numCompacted * sizeof(uint32_t), stream);
    cudaMemsetAsync(devSegmentEnd.get(), 0, numCompacted * sizeof(uint32_t), stream);

    markSegmentBoundsKernel<<<rvdHost::blockCount(totalHalfFaces), rvdHost::BlockSize, 0, stream>>>(
        devHalfFaceKeysOut.get(), totalHalfFaces, numCompacted,
        devSegmentStart.get(), devSegmentEnd.get());

    cudaMemsetAsync(devBoundaryVolumes.get(), 0, numCompacted * sizeof(double), stream);
    cudaMemsetAsync(devBoundaryMoments.get(), 0, numCompacted * sizeof(double3), stream);

    integrateSegmentedBoundaryKernel<<<rvdHost::blockCount(numCompacted), rvdHost::BlockSize, 0, stream>>>(
        numCompacted, devSegmentStart.get(), devSegmentEnd.get(),
        devHalfFaceValuesOut.get(), devCompactedFaces.get(), devSeedPositions.get(),
        devBoundaryVolumes.get(), devBoundaryMoments.get());
  }

  std::vector<double> boundaryVols(numCompacted);
  std::vector<double3> boundaryMoms(numCompacted);
  devBoundaryVolumes.downloadAsync(boundaryVols.data(), numCompacted, stream);
  devBoundaryMoments.downloadAsync(boundaryMoms.data(), numCompacted, stream);

  std::vector<DeviceFragmentFace> compactedHostFaces(totalCompactedFaces);
  devCompactedFaces.downloadAsync(compactedHostFaces.data(), totalCompactedFaces, stream);

  std::vector<uint32_t> oldFaceToNew(numFaces);
  devOldFaceToNew.downloadAsync(oldFaceToNew.data(), numFaces, stream);
  cudaStreamSynchronize(stream);

  for (uint32_t i = 0; i < numCompacted; ++i) {
    const double V = boundaryVols[i];
    if (!(V > 0.0) || !std::isfinite(V)) {
      std::cerr << "[RVDAgglomeration] invalid boundary volume for compacted fragment=" << i
                << " vol=" << V << std::endl;
      return false;
    }
    const double3 &m = boundaryMoms[i];
    const double3 c = cudaMath::add3(hostSeedPositions[i], cudaMath::mul3(m, 1.0 / V));
    if (!std::isfinite(c.x) || !std::isfinite(c.y) || !std::isfinite(c.z)) {
      std::cerr << "[RVDAgglomeration] non-finite boundary centroid for compacted fragment=" << i << std::endl;
      return false;
    }
    compactedFragments[i].volume = V;
    compactedFragments[i].centroid = glm::dvec3(c.x, c.y, c.z);
  }

  std::vector<FragmentFace> compactedFaces(totalCompactedFaces);
  for (uint32_t f = 0; f < totalCompactedFaces; ++f) {
    const DeviceFragmentFace &df = compactedHostFaces[f];
    compactedFaces[f].fragmentA = df.fragmentA;
    compactedFaces[f].fragmentB = df.fragmentB;
    compactedFaces[f].areaA = df.areaA;
    compactedFaces[f].areaB = df.areaB;
    compactedFaces[f].areaVectorA = glm::dvec3(df.areaVectorA.x, df.areaVectorA.y, df.areaVectorA.z);
    compactedFaces[f].areaVectorB = glm::dvec3(df.areaVectorB.x, df.areaVectorB.y, df.areaVectorB.z);
    compactedFaces[f].momentA = glm::dvec3(df.momentA.x, df.momentA.y, df.momentA.z);
    compactedFaces[f].momentB = glm::dvec3(df.momentB.x, df.momentB.y, df.momentB.z);
    compactedFaces[f].type = static_cast<FaceType>(df.type);
    compactedFaces[f].domainFace = df.domainFace;
  }

  for (SurfaceRegion &region : surfaceRegions) {
    if (region.faceId != InvalidId) {
      if (region.faceId >= oldFaceToNew.size() || oldFaceToNew[region.faceId] == InvalidId) {
        std::cerr << "[RVDAgglomeration] lost CAD surface face mapping for region="
                  << region.runtimeModelId << " cell=" << region.cellId
                  << " oldFaceId=" << region.faceId << std::endl;
        return false;
      }
      region.faceId = oldFaceToNew[region.faceId];
    }
  }

  result.fragments = std::move(compactedFragments);
  result.fragmentFaces = std::move(compactedFaces);
  totalBaseFragmentFaces = newTotalBaseFaces;

  const uint32_t cellCount = baseDomain.count;
  std::vector<std::vector<uint32_t>> cellToFragments(cellCount);
  for (uint32_t oldFragId = 0; oldFragId < numFragments; ++oldFragId) {
    const uint32_t newFragId = oldFragToNew[oldFragId];
    const uint32_t cell = hostCellIds[oldFragId];
    if (cell < cellCount) {
      auto &list = cellToFragments[cell];
      if (std::find(list.begin(), list.end(), newFragId) == list.end()) {
        list.push_back(newFragId);
      }
    }
  }

  result.cellFragmentOffsets.assign(size_t(cellCount) + 1, 0u);
  for (uint32_t cell = 0; cell < cellCount; ++cell) {
    result.cellFragmentOffsets[cell + 1] =
        result.cellFragmentOffsets[cell] + static_cast<uint32_t>(cellToFragments[cell].size());
  }
  result.cellFragments.resize(result.cellFragmentOffsets.back());
  for (uint32_t cell = 0; cell < cellCount; ++cell) {
    const uint32_t offset = result.cellFragmentOffsets[cell];
    for (size_t k = 0; k < cellToFragments[cell].size(); ++k) {
      result.cellFragments[offset + k] = cellToFragments[cell][k];
    }
  }

  return true;
}

} // namespace voronoi
