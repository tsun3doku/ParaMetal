#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDBaseFaceTopology.hpp"
#include "RVDRestriction.cuh"
#include "RVDFaceCut.cuh"

#include "RVDFragments.cuh"
#include "RVDBaseFacePartition.cuh"
#include "RVDTraceSurface.hpp"
#include "RVDModelGeometry.cuh"
#include "RVDSurfacePartition.hpp"
#include "RVDContact.cuh"
#include "RVDValidation.hpp"
#include "RVDLloyd.cuh"
#include "RVDAgglomeration.cuh"
#include "KNNNeighbors.hpp"
#include "../../cuda/CudaBuffer.cuh"

#include "../../cuda/CudaVulkanDevice.hpp"
#include "../../vulkan/VulkanDevice.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

#include <vector>

namespace voronoi {


class RVD::Implementation {
public:
  ~Implementation() { cleanup(); }

  bool initialize(VulkanDevice &vulkanDevice) {
    cleanup();

    cudaDevice = cudaVulkan::findDevice(vulkanDevice.getPhysicalDevice());
    if (cudaDevice < 0) {
      std::cerr << "[RVD] Vulkan physical device has no matching CUDA UUID" << std::endl;
      return false;
    }

    if (!cudaUtils::cudaCheck(cudaSetDevice(cudaDevice), "select CUDA device") ||
        !cudaUtils::cudaCheck(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "create stream")) {
      cleanup();
      return false;
    }

    initialized = true;
    return true;
  }

  void cleanup() {
    if (cudaDevice >= 0) cudaSetDevice(cudaDevice);
    if (stream) cudaStreamDestroy(stream);
    stream = nullptr;
    cudaDevice = -1;
    initialized = false;
  }

  bool build(const RVDInput &input, const RVDConfig &config, RVDResult &result) {
    result = {};

    if (!initialized ||
        !cudaUtils::cudaCheck(cudaSetDevice(cudaDevice), "select RVD CUDA device")) {
      return false;
    }

    std::string validationReason;
    if (!validateInput(input, validationReason)) {
      std::cerr << "[RVD] input validation failed: " << validationReason << std::endl;
      return false;
    }
    if (!(config.contactRange > 0.0f) ||
        !std::isfinite(config.contactRange)) {
      std::cerr << "[RVD] invalid contactRange=" << config.contactRange
                << std::endl;
      return false;
    }


    std::vector<cudaGeometry::DeviceModelGeometry> geometries;
    if (!cudaGeometry::uploadGeometries(input, geometries, stream)) {
      return false;
    }


    std::vector<glm::vec4> relaxedSeeds = *input.seeds;
    RVDInput effectiveInput = input;
    effectiveInput.seeds = &relaxedSeeds;

    const double lloydConvergenceTolerance =
        double(effectiveInput.nominalCellSize) * 1e-4;
    for (uint32_t iteration = 0; iteration < config.lloydIterations; ++iteration) {
      if (iteration > 0 &&
          !effectiveInput.seedSpatialIndex->buildSpatialIndex(relaxedSeeds)) {
        std::cerr << "[RVD] Lloyd spatial index rebuild failed" << std::endl;
        return false;
      }

      BaseCellDomain lloydDomain;
      if (!prepareBaseDomain(effectiveInput, lloydDomain) ||
          !buildBaseCells(effectiveInput, config, lloydDomain, stream)) {
        std::cerr << "[RVD] Lloyd base cell build failed" << std::endl;
        return false;
      }
      if (lloydDomain.count != relaxedSeeds.size()) {
        std::cerr << "[RVD] Lloyd base cell count mismatch" << std::endl;
        return false;
      }

      uint32_t movedCount = 0;
      double maxDisplacement = 0.0;
      if (!updateLloydSeeds(
              effectiveInput,
              config,
              lloydDomain,
              relaxedSeeds,
              movedCount,
              maxDisplacement,
              stream)) {
        std::cerr << "[RVD] Lloyd seed update failed" << std::endl;
        return false;
      }
      if (maxDisplacement <= lloydConvergenceTolerance) {
        break;
      }
    }

    if (config.lloydIterations > 0 &&
        !effectiveInput.seedSpatialIndex->buildSpatialIndex(relaxedSeeds)) {
      std::cerr << "[RVD] Lloyd final spatial index rebuild failed" << std::endl;
      return false;
    }

    BaseCellDomain baseDomain;
    if (!prepareBaseDomain(effectiveInput, baseDomain)) return false;
    if (!buildBaseCells(effectiveInput, config, baseDomain, stream)) return false;

    BaseFaceTopology faceTopology;
    if (!buildBaseFaces(baseDomain, faceTopology, stream)) return false;

    std::vector<ModelRestriction> models;
    if (!restrictModels(effectiveInput, config, geometries, baseDomain, models)) {
      return false;
    }

    if (!canonicalizeFaceCutVertices(models, stream)) {
      std::cerr << "[RVD] face cut canonicalization failed" << std::endl;
      return false;
    }
    if (!validateFaceCutCanonicalization(models, stream)) {
      std::cerr << "[RVD] face cut canonicalization validation failed"
                << std::endl;
      return false;
    }

    std::vector<cudaGeometry::DeviceModelGeometryView> geometryViews;
    geometryViews.reserve(geometries.size());
    for (const cudaGeometry::DeviceModelGeometry &geometry : geometries) {
      geometryViews.push_back(geometry.view());
    }

    if (!buildFacePartitionsGPU(
            baseDomain, faceTopology, geometryViews, models, stream)) {
      std::cerr << "[RVD] face partition failed" << std::endl;
      return false;
    }

    std::vector<SurfaceRegion> surfaceRegions;
    if (!buildSurfacePartition(
            models,
            geometryViews,
            baseDomain,
            surfaceRegions,
            stream)) {
      std::cerr << "[RVD] surface adjacency construction failed" << std::endl;
      return false;
    }

    uint32_t totalBaseFragmentFaces = 0;
    if (!buildFragmentsAndFaces(
            effectiveInput,
            baseDomain,
            faceTopology,
            models,
            surfaceRegions,
            totalBaseFragmentFaces,
            result)) {
      std::cerr << "[RVD] cell/material fragment assembly failed" << std::endl;
      return false;
    }

    if (!agglomerateFragments(
            baseDomain,
            surfaceRegions,
            totalBaseFragmentFaces,
            result,
            stream)) {
      std::cerr << "[RVD] fragment agglomeration failed" << std::endl;
      return false;
    }
    buildVemTraceSurfaceMeshes(
        effectiveInput.geometries,
        surfaceRegions,
        result.vemTraceSurfaceMeshes);

    if (!validateFragmentTopology(result, totalBaseFragmentFaces, 0.005, 0.01)) {
      std::cerr << "[RVD] geometric topology validation failed" << std::endl;
      return false;
    }

    if (!buildContactCouplings(
            surfaceRegions,
            result,
            baseDomain,
            double(config.contactRange),
            result.contactCouplings,
            stream)) {
      std::cerr << "[RVD] contact coupling construction failed" << std::endl;
      return false;
    }

    if (!validateContactCouplings(result)) {
      std::cerr << "[RVD] contact coverage validation failed" << std::endl;
      return false;
    }

    result.seedPositions = relaxedSeeds;
    return true;
  }

private:
  static bool validateInput(const RVDInput &input, std::string &reason) {
    if (!input.seeds || !input.seedSpatialIndex || !input.domainCorners ||
        !input.geometries || input.geometries->empty()) {
      reason = "missing input collection";
      return false;
    }

    const size_t seedCount = input.seeds->size();
    if (seedCount == 0 || seedCount > std::numeric_limits<uint32_t>::max()) {
      reason = "invalid seed count";
      return false;
    }

    if (input.seedSpatialIndex->seedCount() != seedCount) {
      reason = "seed spatial index has the wrong seed count";
      return false;
    }

    if (!(input.nominalCellSize > 0.0f) || !std::isfinite(input.nominalCellSize)) {
      reason = "invalid nominal cell size";
      return false;
    }

    for (size_t seed = 0; seed < seedCount; ++seed) {
      const glm::vec4 point = (*input.seeds)[seed];
      if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
          !std::isfinite(point.z) || !std::isfinite(point.w)) {
        reason = "non-finite seed";
        return false;
      }
    }
    return true;
  }

  bool restrictModels(
      const RVDInput &input,
      const RVDConfig &config,
      const std::vector<cudaGeometry::DeviceModelGeometry> &geometries,
      const BaseCellDomain &baseDomain,
      std::vector<ModelRestriction> &models) {
    models.resize(geometries.size());

    for (size_t g = 0; g < models.size(); ++g) {
      if (!restrictModelGeometry(
              input, config, geometries[g], baseDomain, models[g], stream)) {
        return false;
      }
    }
    return true;
  }

  int cudaDevice = -1;
  cudaStream_t stream = nullptr;
  bool initialized = false;
};

RVD::RVD() : implementation(std::make_unique<Implementation>()) {}
RVD::~RVD() = default;

bool RVD::initialize(VulkanDevice &vulkanDevice) {
  return implementation->initialize(vulkanDevice);
}

bool RVD::build(const RVDInput &input, const RVDConfig &config, RVDResult &result) {
  return implementation->build(input, config, result);
}

void RVD::cleanup() {
  implementation->cleanup();
}

} // namespace voronoi
