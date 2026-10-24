#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDTopology.hpp"
#include "RVDRestriction.cuh"
#include "VoxelOccupancy.cuh"
#include "RVDFragments.hpp"
#include "RVDFacePartition.hpp"
#include "RVDContact.cuh"
#include "RVDValidation.hpp"
#include "KNNNeighbors.hpp"
#include "../../cuda/CudaBuffer.cuh"

#include "../../cuda/CudaVulkanDevice.hpp"
#include "../../vulkan/VulkanDevice.hpp"

#include <cuda_runtime.h>

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

    BaseCellDomain baseDomain;
    if (!prepareBaseDomain(input, baseDomain)) return false;
    if (!buildBaseCells(input, config, baseDomain, stream)) return false;

    BaseFaceTopology faceTopology;
    if (!buildBaseFaces(baseDomain, faceTopology, stream)) return false;

    std::vector<cudaGeometry::DeviceModelGeometry> geometries;
    if (!uploadGeometries(input, geometries)) return false;

    std::vector<ModelRestriction> models;
    if (!restrictModels(input, config, geometries, baseDomain, models)) {
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

    if (!(config.contactRange > 0.0f) || !std::isfinite(config.contactRange)) {
      std::cerr << "[RVD] invalid contactRange=" << config.contactRange << std::endl;
      return false;
    }
    const double contactRange = double(config.contactRange);

    cudaUtils::CudaBuffer<MortarOverlap> mortarOverlaps;
    cudaUtils::CudaBuffer<double4> surfaceExposure;
    uint32_t overlapCount = 0;
    if (!buildMortarOverlaps(
            models,
            geometryViews,
            baseDomain,
            contactRange,
            mortarOverlaps,
            overlapCount,
            surfaceExposure,
            stream)) {
      std::cerr << "[RVD] mortar overlap construction failed" << std::endl;
      return false;
    }

    cudaUtils::CudaBuffer<uint32_t> contactFaceBySurface;
    uint32_t totalBaseFragmentFaces = 0;
    if (!buildFragmentsAndFaces(
            input,
            baseDomain,
            faceTopology,
            models,
            surfaceExposure,
            contactFaceBySurface,
            totalBaseFragmentFaces,
            result,
            stream)) {
      return false;
    }

    if (!validateFragmentTopology(result, totalBaseFragmentFaces, 0.005, 0.01)) {
      std::cerr << "[RVD] geometric topology validation failed" << std::endl;
      return false;
    }

    if (!finalizeContactCouplings(
            mortarOverlaps,
            overlapCount,
            contactFaceBySurface,
            result,
            stream)) {
      std::cerr << "[RVD] contact coupling finalization failed" << std::endl;
      return false;
    }
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

  bool uploadGeometries(
      const RVDInput &input,
      std::vector<cudaGeometry::DeviceModelGeometry> &geometries) {
    geometries.resize(input.geometries->size());

    for (size_t g = 0; g < input.geometries->size(); ++g) {
      const ModelGeometry &model = (*input.geometries)[g];
      if (!model.triangles || !model.voxelGrid) {
        std::cerr << "[RVD] missing triangles or voxelGrid for modelId="
                  << model.runtimeModelId << std::endl;
        return false;
      }
      if (!cudaGeometry::uploadGeometry(
              *model.triangles,
              *model.voxelGrid,
              geometries[g],
              stream,
              model.runtimeModelId)) {
        std::cerr << "[RVD] failed to upload geometry modelId="
                  << model.runtimeModelId << std::endl;
        return false;
      }
    }

    if (input.outDevSolidMask == nullptr) return true;

    std::vector<cudaGeometry::DeviceModelGeometryView> maskViews;
    maskViews.reserve(geometries.size());
    for (const cudaGeometry::DeviceModelGeometry &geometry : geometries) {
      maskViews.push_back(geometry.view());
    }

    if (!cudaGeometry::buildWindingSolidMask(
            maskViews.data(),
            uint32_t(maskViews.size()),
            make_float3(
                input.solidMaskGridMin.x,
                input.solidMaskGridMin.y,
                input.solidMaskGridMin.z),
            make_int3(
                input.solidMaskGridDim.x,
                input.solidMaskGridDim.y,
                input.solidMaskGridDim.z),
            input.solidMaskCellSize,
            input.outDevSolidMask,
            stream)) {
      std::cerr << "[RVD] winding solid mask failed" << std::endl;
      return false;
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

    uint32_t totalSurfacePatches = 0;
    for (size_t g = 0; g < models.size(); ++g) {
      if (!restrictModelGeometry(
              input, config, geometries[g], baseDomain, models[g], stream)) {
        return false;
      }
      models[g].surfaceBase = totalSurfacePatches;
      totalSurfacePatches += models[g].totalPatches;
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
