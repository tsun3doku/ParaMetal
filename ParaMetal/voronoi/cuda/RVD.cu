#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDTopology.hpp"
#include "RVDRestriction.cuh"
#include "VoxelOccupancy.cuh"
#include "RVDFragments.hpp"
#include "RVDFacePartition.hpp"
#include "RVDContact.cuh"
#include "KNNNeighbors.hpp"
#include "../../cuda/CudaBuffer.cuh"

#include "../../cuda/CudaVulkanDevice.hpp"
#include "../../vulkan/VulkanDevice.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>



namespace voronoi {

class RVD::Implementation {
public:
    ~Implementation() { cleanup(); }

    bool initialize(VulkanDevice& vulkanDevice) {
        cleanup();
        cudaDevice = cudaVulkan::findDevice(vulkanDevice.getPhysicalDevice());
        if (cudaDevice < 0) {
            std::cerr << "[RVD] Vulkan physical device has no matching CUDA UUID" << std::endl;
            return false;
        }
        if (!cudaUtils::cudaCheck(cudaSetDevice(cudaDevice), "select CUDA device") ||
            !cudaUtils::cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream")) {
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

    bool validate(const RVDInput& input, std::string& reason) const {
        if (!input.seeds || !input.seedSpatialIndex || !input.domainCorners || !input.geometries || input.geometries->empty()) {
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
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z) || !std::isfinite(point.w)) {
                reason = "non-finite seed";
                return false;
            }
        }
        return true;
    }

    bool build(const RVDInput& input, const RVDConfig& config, RVDResult& result) {
        result = {};
        if (!initialized || !cudaUtils::cudaCheck(cudaSetDevice(cudaDevice), "select RVD CUDA device")) return false;

        std::string validationReason;
        if (!validate(input, validationReason)) {
            std::cerr << "[RVD] input validation failed: " << validationReason << std::endl;
            return false;
        }

        BaseCellDomain baseDomain;
        if (!prepareBaseDomain(input, baseDomain)) return false;
        if (!buildBaseCells(input, config, baseDomain, stream)) return false;

        BaseFaceTopology faceTopology;
        if (!buildBaseFaces(baseDomain, faceTopology, stream)) return false;

        std::vector<cudaGeometry::DeviceModelGeometry> deviceGeometries(input.geometries->size());
        for (size_t g = 0; g < input.geometries->size(); ++g) {
            const ModelGeometry& geom = (*input.geometries)[g];
            if (!geom.triangles || !geom.voxelGrid) {
                std::cerr << "[RVD] missing triangles or voxelGrid for modelId=" << geom.runtimeModelId << std::endl;
                return false;
            }
            if (!cudaGeometry::uploadGeometry(
                    *geom.triangles,
                    *geom.voxelGrid,
                    deviceGeometries[g],
                    stream,
                    geom.runtimeModelId)) {
                std::cerr << "[RVD] failed to upload geometry modelId=" << geom.runtimeModelId << std::endl;
                return false;
            }
        }

        if (input.outDevSolidMask != nullptr) {
            std::vector<cudaGeometry::DeviceModelGeometryView> maskViews(deviceGeometries.size());
            for (size_t g = 0; g < deviceGeometries.size(); ++g) {
                maskViews[g] = deviceGeometries[g].view();
            }
            if (!cudaGeometry::buildWindingSolidMask(
                    maskViews.data(),
                    uint32_t(maskViews.size()),
                    make_float3(input.solidMaskGridMin.x, input.solidMaskGridMin.y, input.solidMaskGridMin.z),
                    make_int3(input.solidMaskGridDim.x, input.solidMaskGridDim.y, input.solidMaskGridDim.z),
                    input.solidMaskCellSize,
                    input.outDevSolidMask,
                    stream)) {
                std::cerr << "[RVD] winding solid mask failed" << std::endl;
                return false;
            }
        }

        uint32_t totalSurfacePatches = 0;
        std::vector<ModelRestriction> models(input.geometries->size());
        for (size_t g = 0; g < models.size(); ++g) {
            if (!restrictModelGeometry(input, config, deviceGeometries[g], baseDomain, models[g], stream)) {
                return false;
            }
            models[g].surfaceBase = totalSurfacePatches;
            totalSurfacePatches += models[g].totalPatches;
        }

        const uint32_t numGeometries = uint32_t(deviceGeometries.size());
        std::vector<cudaGeometry::DeviceModelGeometryView> hostGeomViews(numGeometries);
        for (uint32_t g = 0; g < numGeometries; ++g) {
            hostGeomViews[g] = deviceGeometries[g].view();
        }

        if (!buildFacePartitionsGPU(baseDomain, faceTopology, hostGeomViews, models, stream)) {
            std::cerr << "[RVD] face partition failed" << std::endl;
            return false;
        }

        if (!(config.contactRange > 0.0f) || !std::isfinite(config.contactRange)) {
            std::cerr << "[RVD] invalid contactRange=" << config.contactRange << std::endl;
            return false;
        }
        const double contactRange = double(config.contactRange);

        cudaUtils::CudaBuffer<MortarOverlap> devMortarOverlaps;
        cudaUtils::CudaBuffer<double4> devSurfaceExposure;
        uint32_t overlapCount = 0;
        if (!buildMortarOverlaps(
            models,
            hostGeomViews,
            baseDomain,
            contactRange,
            devMortarOverlaps,
            overlapCount,
            devSurfaceExposure,
            stream))
        {
            std::cerr << "[RVD] mortar overlap construction failed" << std::endl;
            return false;
        }

        cudaUtils::CudaBuffer<uint32_t> devContactFaceBySurface;
        uint32_t totalBaseFragmentFaces = 0;

        if (!buildFragmentsAndFaces(
            input,
            baseDomain,
            faceTopology,
            models,
            devSurfaceExposure,
            devContactFaceBySurface,
            totalBaseFragmentFaces,
            result,
            stream))
        {
            return false;
        }

        if (!validateFragmentTopology(result, totalBaseFragmentFaces, 0.005, 0.01)) {
            std::cerr << "[RVD] geometric topology validation failed" << std::endl;
            return false;
        }

        if (!finalizeContactCouplings(
            devMortarOverlaps,
            overlapCount,
            devContactFaceBySurface,
            result,
            stream))
        {
            std::cerr << "[RVD] contact coupling finalization failed" << std::endl;
            return false;
        }

        return true;
    }

    int cudaDevice = -1;
    cudaStream_t stream = nullptr;
    bool initialized = false;
};

RVD::RVD() : implementation(std::make_unique<Implementation>()) {}
RVD::~RVD() = default;
bool RVD::initialize(VulkanDevice& vulkanDevice) { return implementation->initialize(vulkanDevice); }
bool RVD::build(const RVDInput& input, const RVDConfig& config, RVDResult& result) {
    return implementation->build(input, config, result);
}
void RVD::cleanup() { implementation->cleanup(); }

} // namespace voronoi
