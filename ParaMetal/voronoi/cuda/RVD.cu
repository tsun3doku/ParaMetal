#include "RVD.hpp"
#include "RVDBaseCells.cuh"
#include "RVDTopology.hpp"
#include "RVDRestriction.cuh"
#include "RVDFragments.hpp"
#include "RVDContact.cuh"
#include "RVDSurfaceExposure.cuh"
#include "KNNNeighbors.hpp"
#include "../../cuda/CudaBuffer.cuh"

#include "../../cuda/CudaVulkanDevice.hpp"
#include "../../vulkan/VulkanDevice.hpp"

#include <iostream>
#include <vector>
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

        BaseCellsContext baseCtx;
        if (!prepareBaseDomain(input, baseCtx)) return false;
        if (!buildBaseCells(input, config, baseCtx, stream)) return false;

        TopologyContext topoCtx;
        if (!buildBaseFaces(input, baseCtx, topoCtx)) return false;

        std::vector<cudaGeometry::DeviceModelGeometry> deviceGeometries(input.geometries->size());
        for (size_t g = 0; g < input.geometries->size(); ++g) {
            const ModelGeometry& geom = (*input.geometries)[g];
            if (!geom.triangles || !geom.voxelGrid) {
                std::cerr << "[RVD] missing triangles or voxelGrid for modelId=" << geom.runtimeModelId << std::endl;
                return false;
            }
            if (!cudaGeometry::uploadGeometry(*geom.triangles, *geom.voxelGrid, deviceGeometries[g], stream)) {
                std::cerr << "[RVD] failed to upload geometry modelId=" << geom.runtimeModelId << std::endl;
                return false;
            }
            deviceGeometries[g].runtimeModelId = geom.runtimeModelId;
        }

        std::vector<ModelRestrictionData> models(input.geometries->size());
        for (size_t g = 0; g < models.size(); ++g) {
            if (!restrictModelGeometry(input, config, deviceGeometries[g], baseCtx, topoCtx, models[g], result, stream)) {
                return false;
            }
        }

        const uint32_t numGeometries = uint32_t(deviceGeometries.size());
        std::vector<cudaGeometry::DeviceModelGeometryView> hostGeomViews(numGeometries);
        for (uint32_t g = 0; g < numGeometries; ++g) {
            hostGeomViews[g] = deviceGeometries[g].view();
        }
        cudaUtils::CudaBuffer<cudaGeometry::DeviceModelGeometryView> devGeomViews;
        if (!devGeomViews.allocate(numGeometries) ||
            !devGeomViews.uploadAsync(hostGeomViews.data(), numGeometries, stream)) {
            std::cerr << "[RVD] failed to upload geometry views" << std::endl;
            return false;
        }

        if (!(config.contactRange > 0.0f) || !std::isfinite(config.contactRange)) {
            std::cerr << "[RVD] invalid contactRange=" << config.contactRange << std::endl;
            return false;
        }
        const float contactResolution = 0.1f * input.nominalCellSize;
        const float contactRange = config.contactRange;

        if (!computeSurfaceExposure(models, devGeomViews, baseCtx, contactRange, contactResolution, result.surfaceExposure, stream)) {
            std::cerr << "[RVD] global surface exposure computation failed" << std::endl;
            return false;
        }

        std::vector<std::vector<uint32_t>> fragmentIdByModelCell;
        std::vector<uint32_t> unownedFragmentIdByCell;

        if (!buildFragments(input, baseCtx, models, fragmentIdByModelCell, unownedFragmentIdByCell, result)) return false;

        if (!buildContactFaces(
            models,
            devGeomViews,
            hostGeomViews,
            input.seedSpatialIndex->deviceKnn(),
            baseCtx,
            contactRange,
            contactResolution,
            fragmentIdByModelCell,
            result.fragments,
            result,
            stream))
        {
            std::cerr << "[RVD] contact face construction failed" << std::endl;
            return false;
        }

        const bool fragmentFacesOk =
            buildFragmentFaces(
                baseCtx,
                topoCtx,
                models,
                fragmentIdByModelCell,
                unownedFragmentIdByCell,
                result);

        if (!fragmentFacesOk)
            return false;

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
