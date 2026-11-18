#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDBaseFaceTopology.hpp"
#include "RVDRestriction.cuh"
#include "RVDFaceCut.cuh"

#include "RVDFragments.cuh"
#include "RVDBaseFacePartition.cuh"
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
#include <utility>

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
    if (cudaDevice >= 0) {
      cudaSetDevice(cudaDevice);
    }
    if (stream) {
      cudaStreamDestroy(stream);
    }
    stream = nullptr;
    cudaDevice = -1;
    initialized = false;
  }

  bool build(const RVDInput &input, const RVDConfig &config, RVDResult &result) {
    result = {};
    RVDResult candidate;

    if (!initialized ||
        !cudaUtils::cudaCheck(cudaSetDevice(cudaDevice), "select RVD CUDA device")) {
      return false;
    }
    const cudaError_t pendingError = cudaGetLastError();
    if (pendingError != cudaSuccess) {
      std::cerr << "[RVD] unexpected pending CUDA error before build: "
                << cudaGetErrorString(pendingError) << std::endl;
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


    auto stage = [&](auto &&fn) -> bool {
      const bool ok = fn();
      const cudaError_t error = cudaStreamSynchronize(stream);
      return ok && error == cudaSuccess;
    };

    std::vector<cudaGeometry::DeviceModelGeometry> geometries;
    if (!stage([&] {
          return cudaGeometry::uploadGeometries(input, geometries, stream);
        })) {
      return false;
    }


    std::vector<glm::vec4> relaxedSeeds = *input.seeds;
    RVDInput effectiveInput = input;
    effectiveInput.seeds = &relaxedSeeds;

    BaseCellDomain baseDomain;
    bool baseDomainReady = false;

    if (!stage([&] {
          BaseCellDomain lloydDomain;
          uint32_t lastMovedCount = 1;
          for (uint32_t iteration = 0; iteration < config.lloydIterations; ++iteration) {
            if (iteration > 0 &&
                !effectiveInput.seedSpatialIndex->buildSpatialIndex(relaxedSeeds)) {
              std::cerr << "[RVD] Lloyd spatial index rebuild failed" << std::endl;
              return false;
            }

            lloydDomain = {};
            if (!prepareBaseDomain(effectiveInput, lloydDomain) ||
                !buildBaseCells(effectiveInput, config, lloydDomain, stream)) {
              std::cerr << "[RVD] Lloyd base cell build failed" << std::endl;
              return false;
            }
            if (lloydDomain.count != relaxedSeeds.size()) {
              std::cerr << "[RVD] Lloyd base cell count mismatch" << std::endl;
              return false;
            }
            const glm::vec3 domainSize = lloydDomain.domainMax - lloydDomain.domainMin;
            const double meanSpacing = std::cbrt(
                double(domainSize.x) * double(domainSize.y) * double(domainSize.z) /
                double(lloydDomain.count));
            const double lloydConvergenceTolerance = 1e-4 * meanSpacing;

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
            lastMovedCount = movedCount;
            if (maxDisplacement <= lloydConvergenceTolerance) {
              break;
            }
          }

          if (config.lloydIterations > 0) {
            if (lastMovedCount == 0 && lloydDomain.count == relaxedSeeds.size()) {
              baseDomain = std::move(lloydDomain);
              baseDomainReady = true;
            } else if (!effectiveInput.seedSpatialIndex->buildSpatialIndex(relaxedSeeds)) {
              std::cerr << "[RVD] Lloyd final spatial index rebuild failed" << std::endl;
              return false;
            }
          }
          return true;
        })) {
      return false;
    }

    if (!baseDomainReady) {
      if (!stage([&] {
            return prepareBaseDomain(effectiveInput, baseDomain) &&
                buildBaseCells(effectiveInput, config, baseDomain, stream);
          })) {
        return false;
      }
    }

    BaseFaceTopology faceTopology;
    if (!stage([&] {
          return buildBaseFaces(baseDomain, faceTopology, stream);
        })) {
      return false;
    }

    std::vector<ModelRestriction> models;
    if (!stage([&] {
          if (!restrictModels(effectiveInput, config, geometries, baseDomain, models)) {
            return false;
          }
          if (!canonicalizeFaceCutVertices(models, stream)) {
            std::cerr << "[RVD] face cut canonicalization failed" << std::endl;
            return false;
          }
          return true;
        })) {
      return false;
    }

    std::vector<cudaGeometry::DeviceModelGeometryView> geometryViews;
    geometryViews.reserve(geometries.size());
    for (const cudaGeometry::DeviceModelGeometry &geometry : geometries) {
      geometryViews.push_back(geometry.view());
    }

    if (!stage([&] {
          return buildFacePartitionsGPU(
              baseDomain, faceTopology, geometryViews, models, stream);
        })) {
      std::cerr << "[RVD] face partition failed" << std::endl;
      return false;
    }

    std::vector<SurfaceRegion> surfaceRegions;
    if (!stage([&] {
          return buildSurfacePartition(
              models,
              geometryViews,
              baseDomain,
              surfaceRegions,
              stream);
        })) {
      std::cerr << "[RVD] surface adjacency construction failed" << std::endl;
      return false;
    }

    uint32_t totalBaseFragmentFaces = 0;
    if (!stage([&] {
          return buildFragmentsAndFaces(
              effectiveInput,
              baseDomain,
              faceTopology,
              models,
              surfaceRegions,
              totalBaseFragmentFaces,
              candidate,
              stream);
        })) {
      std::cerr << "[RVD] cell/material fragment assembly failed" << std::endl;
      return false;
    }

    if (!stage([&] {
          if (!agglomerateFragments(
                  baseDomain,
                  surfaceRegions,
                  totalBaseFragmentFaces,
                  candidate,
                  stream)) {
            std::cerr << "[RVD] fragment agglomeration failed" << std::endl;
            return false;
          }
          return validateFragmentTopology(
              candidate, totalBaseFragmentFaces, 0.005, 0.01);
        })) {
      std::cerr << "[RVD] geometric topology validation failed" << std::endl;
      return false;
    }

    if (!stage([&] {
          return buildContactCouplings(
              surfaceRegions,
              candidate,
              baseDomain,
              double(config.contactRange),
              candidate.contactCouplings,
              stream);
        })) {
      std::cerr << "[RVD] contact coupling construction failed" << std::endl;
      return false;
    }

    if (!stage([&] {
          return validateContactCouplings(candidate);
        })) {
      std::cerr << "[RVD] contact coverage validation failed" << std::endl;
      return false;
    }

    candidate.seedPositions = relaxedSeeds;
    result = std::move(candidate);
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
