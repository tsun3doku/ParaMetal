#include "RVDFragments.hpp"
#include "ConvexCell.cuh"

#include <cub/cub.cuh>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

namespace voronoi {

__device__ inline bool reportFragmentError(unsigned long long* devPackedError, uint32_t code, uint32_t id) {
    const unsigned long long packed = (static_cast<unsigned long long>(code) << 32) | static_cast<unsigned long long>(id);
    return atomicCAS(devPackedError, 0ULL, packed) == 0ULL;
}

static __global__ void buildFragmentsKernel(
    const double4* devBaseMoments,
    const double4* const* devModelMoments,
    const uint32_t* devRuntimeModelIds,
    const float4* devSeeds,
    uint32_t cellCount,
    uint32_t modelCount,
    float nominalCellSize,
    double invBaseVolumeScale,
    double invBaseMomentScale,
    Fragment* devCandidateFragments,
    unsigned long long* devPackedError)
{
    const uint32_t cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cellCount) return;

    const double4 rawBaseMom = devBaseMoments[cell];
    const double baseVol = (rawBaseMom.w / 6.0) * invBaseVolumeScale;
    const glm::dvec3 baseMom = (glm::dvec3(rawBaseMom.x, rawBaseMom.y, rawBaseMom.z) / 6.0) * invBaseMomentScale;
    const float4 seed = devSeeds[cell];
    const glm::dvec3 seedPos(seed.x, seed.y, seed.z);
    const double h = double(nominalCellSize);
    const double nominalVolume = h * h * h;
    const double volumeClosureTol = fmax(nominalVolume * 1e-6, baseVol * 1e-4);

    double totalModelVol = 0.0;
    glm::dvec3 totalModelMom(0.0);

    for (uint32_t g = 0; g < modelCount; ++g) {
        const uint32_t candIdx = cell * (modelCount + 1) + g;

        const double4 rawModelMom = devModelMoments[g][cell];
        const double modelVol = rawModelMom.w * invBaseVolumeScale;
        const glm::dvec3 modelMom = glm::dvec3(rawModelMom.x, rawModelMom.y, rawModelMom.z) * invBaseMomentScale;

        Fragment frag{};
        frag.cellId = cell;
        frag.ownerModelId = devRuntimeModelIds[g];
        frag.volume = modelVol;
        if (modelVol > 0.0 && isfinite(modelVol)) {
            const glm::dvec3 c = seedPos + modelMom / modelVol;
            if (isfinite(c.x) && isfinite(c.y) && isfinite(c.z)) {
                frag.centroid = c;
            } else {
                frag.volume = -1.0;
                frag.centroid = seedPos;
            }
        } else {
            frag.centroid = seedPos;
        }
        devCandidateFragments[candIdx] = frag;

        if (modelVol > 0.0 && isfinite(modelVol)) {
            totalModelVol += modelVol;
            totalModelMom += modelMom;
        }
    }

    const uint32_t fluidCandIdx = cell * (modelCount + 1) + modelCount;

    const double rawUnownedVol = baseVol - totalModelVol;
    if (rawUnownedVol < -volumeClosureTol) {
        reportFragmentError(devPackedError, 1, cell);
        return;
    }

    const double unownedVol = rawUnownedVol <= volumeClosureTol
        ? 0.0
        : rawUnownedVol;
    Fragment frag{};
    frag.cellId = cell;
    frag.ownerModelId = InvalidId;
    frag.volume = unownedVol;
    if (unownedVol > 0.0 && isfinite(unownedVol)) {
        const glm::dvec3 remainderMom = baseMom - totalModelMom;
        const glm::dvec3 c = seedPos + remainderMom / unownedVol;
        if (isfinite(c.x) && isfinite(c.y) && isfinite(c.z)) {
            frag.centroid = c;
        } else {
            frag.volume = -1.0;
            frag.centroid = seedPos;
        }
    } else {
        frag.centroid = seedPos;
    }
    devCandidateFragments[fluidCandIdx] = frag;
}

static __device__ inline glm::dvec3 computeOutwardNormal(
    int tag,
    uint32_t cell,
    const float4* seeds,
    const glm::dvec4* outwardDomainPlanes)
{
    if (tag < 0) {
        const int df = rvdDevice::domainFaceFromTag(tag);
        if (df >= 0 && df < 6) {
            const glm::dvec4 plane = outwardDomainPlanes[df];
            return glm::dvec3(plane.x, plane.y, plane.z);
        }
        return glm::dvec3(0.0);
    }
    const float4 sSelf = seeds[cell];
    const float4 sOther = seeds[uint32_t(tag)];
    const glm::dvec3 dir(double(sOther.x) - double(sSelf.x),
                         double(sOther.y) - double(sSelf.y),
                         double(sOther.z) - double(sSelf.z));
    const double len = glm::length(dir);
    return (len > 1e-12) ? (dir / len) : glm::dvec3(0.0);
}

static __device__ uint32_t fragmentForMaterial(
    uint32_t cell,
    uint32_t material,
    uint32_t modelCount,
    const uint32_t* runtimeModelIds)
{
    if (material == InvalidId) {
        return cell * (modelCount + 1) + modelCount;
    }
    if (runtimeModelIds == nullptr) return InvalidId;
    for (uint32_t g = 0; g < modelCount; ++g) {
        if (runtimeModelIds[g] == material) {
            return cell * (modelCount + 1) + g;
        }
    }
    return InvalidId;
}

static __global__ void emitBaseFragmentFacesKernel(
    const BaseFace* baseFaces,
    uint32_t numBaseFaces,
    const BaseFaceMeasure* baseFaceMeasures,
    const uint32_t* regionOffsets,
    const FaceRegion* regions,
    const uint32_t* runtimeModelIds,
    DeviceBaseCells baseCells,
    const uint32_t* cellPlaneOffsets,
    const float4* seeds,
    const glm::dvec4* outwardDomainPlanes,
    uint32_t modelCount,
    FragmentFace* outFragmentFaces,
    unsigned long long* devPackedError)
{
    const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
    if (faceId >= numBaseFaces) return;

    const BaseFace& face = baseFaces[faceId];
    const BaseFaceMeasure& measure = baseFaceMeasures[faceId];
    if (measure.valid == 0u) {
        reportFragmentError(devPackedError, 4, faceId);
        return;
    }

    const uint32_t cellA = face.cellA;
    const uint32_t cellB = face.cellB;
    const uint32_t slotA = face.slotA;
    const uint32_t slotB = face.slotB;

    const BaseCellArrays cellAState = resolveBaseCell(baseCells, cellA);
    const uint32_t lpA = slotA - cellPlaneOffsets[cellA];
    const int tagA = cellAState.planeNeighborIds[lpA];
    const glm::dvec3 normA = computeOutwardNormal(tagA, cellA, seeds, outwardDomainPlanes);

    glm::dvec3 normB(0.0);
    if (cellB != InvalidId) {
        const BaseCellArrays cellBState = resolveBaseCell(baseCells, cellB);
        const uint32_t lpB = slotB - cellPlaneOffsets[cellB];
        const int tagB = cellBState.planeNeighborIds[lpB];
        normB = computeOutwardNormal(tagB, cellB, seeds, outwardDomainPlanes);
    }

    const uint32_t regionBegin = regionOffsets[faceId];
    const uint32_t regionEnd = regionOffsets[faceId + 1];
    const uint32_t regionCount = regionEnd - regionBegin;
    if (regionCount > 0u) {
        for (uint32_t r = regionBegin; r < regionEnd; ++r) {
            const FaceRegion& region = regions[r];
            const uint32_t fragA = fragmentForMaterial(cellA, region.materialA, modelCount, runtimeModelIds);
            const uint32_t fragB = cellB != InvalidId
                ? fragmentForMaterial(cellB, region.materialB, modelCount, runtimeModelIds)
                : InvalidId;
            if (fragA == InvalidId || (cellB != InvalidId && fragB == InvalidId)) {
                reportFragmentError(devPackedError, 9, faceId);
                return;
            }

            FragmentFace ff{};
            ff.fragmentA = fragA;
            ff.fragmentB = fragB;
            ff.areaA = region.measure.area;
            ff.areaVectorA = normA * ff.areaA;
            ff.momentA = glm::dvec3(region.measure.moment.x,
                                    region.measure.moment.y,
                                    region.measure.moment.z);
            if (cellB != InvalidId) {
                ff.areaB = region.measure.area;
                ff.areaVectorB = normB * ff.areaB;
                ff.momentB = ff.momentA;
                ff.type = FaceType::Shared;
                ff.domainFace = InvalidId;
            } else {
                ff.type = FaceType::DomainBoundary;
                ff.domainFace = face.domainFace;
            }
            outFragmentFaces[regionBegin + (r - regionBegin)] = ff;
        }
        return;
    }

    reportFragmentError(devPackedError, 13, faceId);
}

static __global__ void countAndEmitSurfaceFragmentFacesKernel(
    const RestrictedSurface* surfaces,
    const RestrictedSurfaceGeometry* surfaceGeometry,
    const double4* surfaceExposure,
    const Fragment* devCandidateFragments,
    uint32_t totalSurfaces,
    uint32_t modelIndex,
    uint32_t modelCount,
    uint32_t cellCount,
    double invBaseScale,
    glm::dvec3 domainMin,
    uint32_t* surfaceFaceCounts,
    const uint32_t* surfaceFaceOffsets,
    FragmentFace* outFragmentFaces,
    uint32_t totalBaseFacesCount,
    uint32_t* devContactFaceBySurface,
    uint32_t surfaceBase,
    bool emitPass,
    unsigned long long* devPackedError)
{
    const uint32_t surfIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (surfIdx >= totalSurfaces) return;

    const RestrictedSurface& surface = surfaces[surfIdx];
    if (surface.vertexCount < 3) {
        if (!emitPass) surfaceFaceCounts[surfIdx] = 0;
        return;
    }

    const uint32_t cell = surface.cellId;
    if (cell >= cellCount) {
        if (!emitPass) surfaceFaceCounts[surfIdx] = 0;
        return;
    }

    const RestrictedSurfaceGeometry& geom = surfaceGeometry[surfIdx];
    const double invScale2 = invBaseScale * invBaseScale;
    const double canonicalArea = double(geom.centroidArea.w);
    const double totalPolygonArea = canonicalArea * invScale2;
    if (!(totalPolygonArea > 0.0) || !isfinite(totalPolygonArea)) {
        if (!emitPass) surfaceFaceCounts[surfIdx] = 0;
        return;
    }

    glm::dvec3 normal(geom.plane.x, geom.plane.y, geom.plane.z);
    const double normalLen = glm::length(normal);
    if (normalLen <= 1e-14 || !isfinite(normalLen)) {
        if (!emitPass) surfaceFaceCounts[surfIdx] = 0;
        return;
    }
    normal /= normalLen;

    const glm::dvec3 canCent(geom.centroidArea.x, geom.centroidArea.y, geom.centroidArea.z);
    const glm::dvec3 worldCentroid = domainMin + canCent * invBaseScale;
    const glm::dvec3 totalPolygonMoment = worldCentroid * totalPolygonArea;

    const double4 rawExposure = surfaceExposure[surfIdx];
    double Aexposed = rawExposure.w;
    glm::dvec3 Mexposed(rawExposure.x, rawExposure.y, rawExposure.z);

    const double epsExposure = 64.0 * DBL_EPSILON * totalPolygonArea;
    if (Aexposed > totalPolygonArea && Aexposed <= totalPolygonArea + epsExposure) {
        Aexposed = totalPolygonArea;
        Mexposed = totalPolygonMoment;
    } else if (Aexposed < 0.0 && Aexposed >= -epsExposure) {
        Aexposed = 0.0;
        Mexposed = glm::dvec3(0.0);
    }

    if (Aexposed < 0.0 || Aexposed > totalPolygonArea) {
        if (!emitPass) surfaceFaceCounts[surfIdx] = 0;
        reportFragmentError(devPackedError, 10, cell);
        return;
    }

    if (Aexposed <= epsExposure) {
        Aexposed = 0.0;
        Mexposed = glm::dvec3(0.0);
    } else if (totalPolygonArea - Aexposed <= epsExposure) {
        Aexposed = totalPolygonArea;
        Mexposed = totalPolygonMoment;
    }

    const uint32_t solidFrag = cell * (modelCount + 1) + modelIndex;
    const uint32_t fluidFrag = cell * (modelCount + 1) + modelCount;

    const Fragment fluidCandFrag = devCandidateFragments[fluidFrag];
    const bool hasFluid = (fluidCandFrag.volume > 0.0) && isfinite(fluidCandFrag.volume);

    uint32_t localFaceCount = 0;
    const uint32_t dstBase = emitPass ? surfaceFaceOffsets[surfIdx] : 0;

    if (Aexposed > 0.0) {
        if (emitPass) {
            FragmentFace face{};
            face.fragmentA = solidFrag;
            face.fragmentB = hasFluid ? fluidFrag : InvalidId;
            face.areaA = Aexposed;
            face.areaVectorA = normal * Aexposed;
            face.momentA = Mexposed;
            if (hasFluid) {
                face.areaB = Aexposed;
                face.areaVectorB = -normal * Aexposed;
                face.momentB = Mexposed;
            }
            face.type = FaceType::Shared;
            face.domainFace = InvalidId;
            outFragmentFaces[dstBase + localFaceCount] = face;
        }
        localFaceCount++;
    }

    // Contact occluded portion
    const double Acontact = totalPolygonArea - Aexposed;
    const glm::dvec3 Mcontact = totalPolygonMoment - Mexposed;
    if (Acontact > 0.0) {
        if (emitPass) {
            FragmentFace solidContactFace{};
            solidContactFace.fragmentA = solidFrag;
            solidContactFace.fragmentB = InvalidId;
            solidContactFace.areaA = Acontact;
            solidContactFace.areaVectorA = normal * Acontact;
            solidContactFace.momentA = Mcontact;
            solidContactFace.type = FaceType::ContactSide;
            solidContactFace.domainFace = InvalidId;
            outFragmentFaces[dstBase + localFaceCount] = solidContactFace;

            if (devContactFaceBySurface != nullptr) {
                devContactFaceBySurface[surfaceBase + surfIdx] = totalBaseFacesCount + dstBase + localFaceCount;
            }
        }
        localFaceCount++;

        if (hasFluid) {
            if (emitPass) {
                FragmentFace fluidClosureFace{};
                fluidClosureFace.fragmentA = fluidFrag;
                fluidClosureFace.fragmentB = InvalidId;
                fluidClosureFace.areaA = Acontact;
                fluidClosureFace.areaVectorA = -normal * Acontact;
                fluidClosureFace.momentA = Mcontact;
                fluidClosureFace.type = FaceType::ContactSide;
                fluidClosureFace.domainFace = InvalidId;
                outFragmentFaces[dstBase + localFaceCount] = fluidClosureFace;
            }
            localFaceCount++;
        }
    }

    if (!emitPass) {
        surfaceFaceCounts[surfIdx] = localFaceCount;
    }
}

static __global__ void countCandidateFacesKernel(
    const FragmentFace* faces,
    uint32_t numFaces,
    uint32_t* candidateFaceCounts)
{
    const uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= numFaces) return;

    const FragmentFace& face = faces[f];
    if (face.fragmentA != InvalidId) atomicAdd(&candidateFaceCounts[face.fragmentA], 1u);
    if (face.fragmentB != InvalidId) atomicAdd(&candidateFaceCounts[face.fragmentB], 1u);
}

static __global__ void filterCandidatesKernel(
    const Fragment* candidates,
    const uint32_t* candidateFaceCounts,
    uint32_t totalCandidates,
    uint32_t* admitFlags)
{
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= totalCandidates) return;

    const Fragment& candidate = candidates[c];
    const bool keep = isfinite(candidate.volume) && candidate.volume > 0.0 &&
        candidateFaceCounts[c] > 0u;
    admitFlags[c] = keep ? 1u : 0u;
}

static __global__ void compactFragmentsKernel(
    const uint32_t* devAdmitFlags,
    const uint32_t* devAdmitOffsets,
    const Fragment* devCandidateFragments,
    uint32_t totalCandidates,
    Fragment* outFragments,
    uint32_t* outCandidateToGlobalFrag)
{
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= totalCandidates) return;

    if (devAdmitFlags[c] == 1) {
        const uint32_t globalFragId = devAdmitOffsets[c];
        outFragments[globalFragId] = devCandidateFragments[c];
        outCandidateToGlobalFrag[c] = globalFragId;
    } else {
        outCandidateToGlobalFrag[c] = InvalidId;
    }
}

static __global__ void remapFragmentFacesKernel(
    FragmentFace* faces,
    uint32_t totalFaces,
    const uint32_t* candidateToGlobalFrag)
{
    const uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= totalFaces) return;

    FragmentFace ff = faces[f];
    if (ff.fragmentA != InvalidId) {
        ff.fragmentA = candidateToGlobalFrag[ff.fragmentA];
    }
    if (ff.fragmentB != InvalidId) {
        ff.fragmentB = candidateToGlobalFrag[ff.fragmentB];
    }
    faces[f] = ff;
}

bool buildFragmentsAndFaces(
    const RVDInput& input,
    const BaseCellDomain& baseDomain,
    const BaseFaceTopology& faceTopology,
    const std::vector<ModelRestriction>& models,
    const cudaUtils::CudaBuffer<double4>& devSurfaceExposure,
    cudaUtils::CudaBuffer<uint32_t>& devContactFaceBySurface,
    uint32_t& totalBaseFragmentFaces,
    RVDResult& result,
    cudaStream_t stream)
{
    const uint32_t cellCount = baseDomain.count;
    const uint32_t modelCount = uint32_t(models.size());
    const uint32_t totalCandidates = cellCount * (modelCount + 1);
    totalBaseFragmentFaces = 0;
    if (cellCount == 0) {
        return true;
    }

    const float4* devSeeds = input.seedSpatialIndex ? input.seedSpatialIndex->deviceKnn().seeds : nullptr;
    if (!devSeeds) return false;

    // Upload model device moment pointers and runtime IDs
    std::vector<const double4*> hostModelMoments(modelCount);
    std::vector<uint32_t> hostRuntimeModelIds(modelCount);
    for (uint32_t g = 0; g < modelCount; ++g) {
        hostModelMoments[g] = models[g].devMoments.get();
        hostRuntimeModelIds[g] = models[g].runtimeModelId;
    }

    cudaUtils::CudaBuffer<const double4*> devModelMomentsArray;
    cudaUtils::CudaBuffer<uint32_t> devRuntimeModelIds;
    if (modelCount > 0) {
        if (!devModelMomentsArray.allocate(modelCount) ||
            !devModelMomentsArray.uploadAsync(hostModelMoments.data(), modelCount, stream) ||
            !devRuntimeModelIds.allocate(modelCount) ||
            !devRuntimeModelIds.uploadAsync(hostRuntimeModelIds.data(), modelCount, stream)) {
            return false;
        }
    }

    cudaUtils::CudaBuffer<Fragment> devCandidateFragments;
    cudaUtils::CudaBuffer<FragmentFace> devFragmentFaces;
    cudaUtils::CudaBuffer<Fragment> devFragments;
    cudaUtils::CudaBuffer<unsigned long long> devPackedError;

    if (!devCandidateFragments.allocate(totalCandidates) ||
        !devPackedError.allocate(1)) {
        return false;
    }

    cudaMemsetAsync(devPackedError.get(), 0, sizeof(unsigned long long), stream);

    constexpr uint32_t BlockSize = 256;
    const uint32_t blocks = (cellCount + BlockSize - 1) / BlockSize;

    buildFragmentsKernel<<<blocks, BlockSize, 0, stream>>>(
        baseDomain.devBaseMoments.get(),
        devModelMomentsArray.get(),
        devRuntimeModelIds.get(),
        devSeeds,
        cellCount,
        modelCount,
        input.nominalCellSize,
        baseDomain.invBaseVolumeScale,
        baseDomain.invBaseMomentScale,
        devCandidateFragments.get(),
        devPackedError.get());

    unsigned long long packedError = 0;
    cudaMemcpyAsync(&packedError, devPackedError.get(), sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (packedError != 0) {
        const uint32_t errorCell = static_cast<uint32_t>(packedError & 0xFFFFFFFFULL);
        std::cerr << "[RVD-Fragments-GPU] Restriction volume overcoverage detected on cell " << errorCell << std::endl;
        return false;
    }

    const uint32_t numBaseFaces = faceTopology.totalBaseFaces;

    cudaUtils::CudaBuffer<glm::dvec4> devOutwardPlanes;
    devOutwardPlanes.allocate(6);
    devOutwardPlanes.uploadAsync(baseDomain.outwardDomainPlanes.data(), 6, stream);

    const uint32_t baseBlocks = (numBaseFaces + BlockSize - 1) / BlockSize;

    uint32_t totalSurfacePatches = 0;
    for (uint32_t g = 0; g < modelCount; ++g) {
        totalSurfacePatches += models[g].totalPatches;
    }

    cudaUtils::CudaBuffer<uint32_t> devSurfaceCounts;
    cudaUtils::CudaBuffer<uint32_t> devSurfaceOffsets;
    if (totalSurfacePatches > 0) {
        devSurfaceCounts.allocate(totalSurfacePatches + 1);
        devSurfaceOffsets.allocate(totalSurfacePatches + 1);
        cudaMemsetAsync(devSurfaceCounts.get() + totalSurfacePatches, 0, sizeof(uint32_t), stream);

        for (uint32_t g = 0; g < modelCount; ++g) {
            const auto& restriction = models[g];
            if (restriction.totalPatches == 0) continue;
            const uint32_t sBlocks = (restriction.totalPatches + BlockSize - 1) / BlockSize;
            countAndEmitSurfaceFragmentFacesKernel<<<sBlocks, BlockSize, 0, stream>>>(
                restriction.devSurfaces.get(),
                restriction.devSurfaceGeometry.get(),
                devSurfaceExposure.get() + restriction.surfaceBase,
                devCandidateFragments.get(),
                restriction.totalPatches,
                g,
                modelCount,
                cellCount,
                baseDomain.invBaseScale,
                glm::dvec3(baseDomain.domainMin),
                devSurfaceCounts.get() + restriction.surfaceBase,
                devSurfaceOffsets.get() + restriction.surfaceBase,
                nullptr,
                0,
                nullptr,
                restriction.surfaceBase,
                false,
                devPackedError.get());
        }
    }

    uint32_t totalBaseFacesCount = 0;
    if (numBaseFaces > 0) {
        cudaMemcpyAsync(&totalBaseFacesCount, faceTopology.devRegionOffsets.get() + numBaseFaces,
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    }

    uint32_t totalSurfaceFacesCount = 0;
    if (totalSurfacePatches > 0) {
        size_t scanBytesSurf = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, scanBytesSurf, devSurfaceCounts.get(), devSurfaceOffsets.get(), int(totalSurfacePatches + 1), stream);
        cudaUtils::CudaBuffer<uint8_t> scanScratchSurf;
        scanScratchSurf.allocate(scanBytesSurf);
        cub::DeviceScan::ExclusiveSum(scanScratchSurf.get(), scanBytesSurf, devSurfaceCounts.get(), devSurfaceOffsets.get(), int(totalSurfacePatches + 1), stream);
        cudaMemcpyAsync(&totalSurfaceFacesCount, devSurfaceOffsets.get() + totalSurfacePatches, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    const uint32_t grandTotalFaces = totalBaseFacesCount + totalSurfaceFacesCount;
    totalBaseFragmentFaces = totalBaseFacesCount;
    if (!devFragmentFaces.allocate(grandTotalFaces)) return false;

    if (totalSurfacePatches > 0) {
        if (!devContactFaceBySurface.allocate(totalSurfacePatches)) return false;
        cudaMemsetAsync(devContactFaceBySurface.get(), 0xFF, totalSurfacePatches * sizeof(uint32_t), stream);
    }

    // Emit pass
    if (numBaseFaces > 0 && totalBaseFacesCount > 0) {
        emitBaseFragmentFacesKernel<<<baseBlocks, BlockSize, 0, stream>>>(
            faceTopology.devBaseFaces.get(),
            numBaseFaces,
            faceTopology.devMeasures.get(),
            faceTopology.devRegionOffsets.get(),
            faceTopology.devRegions.get(),
            devRuntimeModelIds.get(),
            baseDomain.deviceView(),
            baseDomain.devCellPlaneOffsets.get(),
            devSeeds,
            devOutwardPlanes.get(),
            modelCount,
            devFragmentFaces.get(),
            devPackedError.get());
    }

    if (totalSurfacePatches > 0 && totalSurfaceFacesCount > 0) {
        for (uint32_t g = 0; g < modelCount; ++g) {
            const auto& restriction = models[g];
            if (restriction.totalPatches == 0) continue;
            const uint32_t sBlocks = (restriction.totalPatches + BlockSize - 1) / BlockSize;
            countAndEmitSurfaceFragmentFacesKernel<<<sBlocks, BlockSize, 0, stream>>>(
                restriction.devSurfaces.get(),
                restriction.devSurfaceGeometry.get(),
                devSurfaceExposure.get() + restriction.surfaceBase,
                devCandidateFragments.get(),
                restriction.totalPatches,
                g,
                modelCount,
                cellCount,
                baseDomain.invBaseScale,
                glm::dvec3(baseDomain.domainMin),
                devSurfaceCounts.get() + restriction.surfaceBase,
                devSurfaceOffsets.get() + restriction.surfaceBase,
                devFragmentFaces.get() + totalBaseFacesCount,
                totalBaseFacesCount,
                devContactFaceBySurface.get(),
                restriction.surfaceBase,
                true,
                devPackedError.get());
        }
    }

    cudaMemcpyAsync(&packedError, devPackedError.get(), sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (packedError != 0) {
        const uint32_t errorFlag = static_cast<uint32_t>(packedError >> 32);
        const uint32_t errorCell = static_cast<uint32_t>(packedError & 0xFFFFFFFFULL);
        std::cerr << "[RVD-Faces-GPU] Topological error (code=" << errorFlag << ") on cell " << errorCell << std::endl;
        return false;
    }

    // Keep candidates with finite positive volume and incident faces
    cudaUtils::CudaBuffer<uint32_t> devCandidateFaceCounts;
    cudaUtils::CudaBuffer<uint32_t> devAdmitFlags;
    cudaUtils::CudaBuffer<uint32_t> devAdmitOffsets;
    if (!devCandidateFaceCounts.allocate(totalCandidates) ||
        !devAdmitFlags.allocate(totalCandidates + 1) ||
        !devAdmitOffsets.allocate(totalCandidates + 1)) return false;
    cudaMemsetAsync(devCandidateFaceCounts.get(), 0, totalCandidates * sizeof(uint32_t), stream);
    cudaMemsetAsync(devAdmitFlags.get() + totalCandidates, 0, sizeof(uint32_t), stream);

    const uint32_t candBlocks = (totalCandidates + BlockSize - 1) / BlockSize;
    if (grandTotalFaces > 0) {
        const uint32_t faceBlocks = (grandTotalFaces + BlockSize - 1) / BlockSize;
        countCandidateFacesKernel<<<faceBlocks, BlockSize, 0, stream>>>(
            devFragmentFaces.get(),
            grandTotalFaces,
            devCandidateFaceCounts.get());
    }
    filterCandidatesKernel<<<candBlocks, BlockSize, 0, stream>>>(
        devCandidateFragments.get(),
        devCandidateFaceCounts.get(),
        totalCandidates,
        devAdmitFlags.get());

    // Compaction and remapping
    size_t scanBytesAdmit = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scanBytesAdmit, devAdmitFlags.get(), devAdmitOffsets.get(), int(totalCandidates + 1), stream);
    cudaUtils::CudaBuffer<uint8_t> scanScratchAdmit;
    scanScratchAdmit.allocate(scanBytesAdmit);
    cub::DeviceScan::ExclusiveSum(scanScratchAdmit.get(), scanBytesAdmit, devAdmitFlags.get(), devAdmitOffsets.get(), int(totalCandidates + 1), stream);

    uint32_t totalFragments = 0;
    cudaMemcpyAsync(&totalFragments, devAdmitOffsets.get() + totalCandidates, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (totalFragments > 0) {
        if (!devFragments.allocate(totalFragments)) return false;
    }
    cudaUtils::CudaBuffer<uint32_t> devCandidateToGlobalFrag;
    if (!devCandidateToGlobalFrag.allocate(totalCandidates)) return false;

    compactFragmentsKernel<<<candBlocks, BlockSize, 0, stream>>>(
        devAdmitFlags.get(),
        devAdmitOffsets.get(),
        devCandidateFragments.get(),
        totalCandidates,
        devFragments.get(),
        devCandidateToGlobalFrag.get());

    if (grandTotalFaces > 0) {
        const uint32_t faceBlocks = (grandTotalFaces + BlockSize - 1) / BlockSize;
        remapFragmentFacesKernel<<<faceBlocks, BlockSize, 0, stream>>>(
            devFragmentFaces.get(),
            grandTotalFaces,
            devCandidateToGlobalFrag.get());
    }

    result.fragments.resize(totalFragments);
    if (totalFragments > 0) {
        devFragments.downloadAsync(result.fragments.data(), totalFragments, stream);
    }
    result.fragmentFaces.resize(grandTotalFaces);
    if (grandTotalFaces > 0) {
        devFragmentFaces.downloadAsync(result.fragmentFaces.data(), grandTotalFaces, stream);
    }
    cudaStreamSynchronize(stream);

    const uint32_t seedCount = input.seeds ? static_cast<uint32_t>(input.seeds->size()) : 0u;
    result.seedToFragment.assign(seedCount, InvalidId);
    for (uint32_t i = 0; i < totalFragments; ++i) {
        const Fragment& fragment = result.fragments[i];
        if (fragment.ownerModelId == InvalidModelId && fragment.cellId < seedCount) {
            result.seedToFragment[fragment.cellId] = i;
        }
    }

    return true;
}

// Geometric validation
static bool validateFragmentGeometryHost(
    const std::vector<Fragment>& fragments,
    const std::vector<FragmentFace>& faces,
    uint32_t baseFaceCount,
    double toleranceClosure,
    double toleranceGauss)
{
    const uint32_t fragmentCount = static_cast<uint32_t>(fragments.size());
    if (fragmentCount == 0) return true;

    bool valid = true;
    uint32_t sharedFaceCount = 0;
    double maxReciprocalAreaError = 0.0;
    double maxReciprocalVectorError = 0.0;
    double maxReciprocalMomentError = 0.0;
    for (const FragmentFace& face : faces) {
        if (face.type != FaceType::Shared ||
            !(face.areaA > 0.0) || !(face.areaB > 0.0)) {
            continue;
        }
        ++sharedFaceCount;
        const double areaScale = std::max(face.areaA, face.areaB);
        maxReciprocalAreaError = std::max(
            maxReciprocalAreaError,
            fabs(face.areaA - face.areaB) / areaScale);
        maxReciprocalVectorError = std::max(
            maxReciprocalVectorError,
            glm::length(face.areaVectorA + face.areaVectorB) / areaScale);
        const double momentScale = std::max(
            glm::length(face.momentA), glm::length(face.momentB));
        const double momentDifference =
            glm::length(face.momentA - face.momentB);
        if (momentScale > 0.0) {
            maxReciprocalMomentError = std::max(
                maxReciprocalMomentError, momentDifference / momentScale);
        }
    }
    if (sharedFaceCount > 0) {
        constexpr double ReciprocalFatalTolerance = 1e-4;
        constexpr double ReciprocalWarningTolerance = 1e-5;
        const double reciprocalError = std::max(
            maxReciprocalAreaError,
            std::max(maxReciprocalVectorError, maxReciprocalMomentError));
        if (reciprocalError > ReciprocalFatalTolerance) {
            std::cerr << "[RVD-FaceReciprocity] shared-face measures exceed tolerance "
                      << reciprocalError << " > " << ReciprocalFatalTolerance
                      << std::endl;
            valid = false;
        } else if (reciprocalError > ReciprocalWarningTolerance) {
            std::cerr << "[RVD-FaceReciprocity] shared-face measures near tolerance "
                      << reciprocalError << " > " << ReciprocalWarningTolerance
                      << std::endl;
        }
    }

    std::vector<double> areaSum(fragmentCount, 0.0);
    std::vector<glm::dvec3> areaVectorSum(fragmentCount, glm::dvec3(0.0));
    std::vector<double> gaussSum(size_t(fragmentCount) * 9, 0.0);
    std::vector<double> absTermSum(fragmentCount, 0.0);
    std::vector<uint32_t> incidentFaceCount(fragmentCount, 0);
    std::vector<double> baseAreaSum(fragmentCount, 0.0);
    std::vector<glm::dvec3> baseVectorSum(fragmentCount, glm::dvec3(0.0));
    std::vector<double> surfaceAreaSum(fragmentCount, 0.0);
    std::vector<glm::dvec3> surfaceVectorSum(fragmentCount, glm::dvec3(0.0));

    for (size_t faceIndex = 0; faceIndex < faces.size(); ++faceIndex) {
        const FragmentFace& face = faces[faceIndex];
        const bool isBaseFace = faceIndex < baseFaceCount;
        const uint32_t a = face.fragmentA;
        if (a < fragmentCount && face.areaA > 0.0 && std::isfinite(face.areaA)) {
            const glm::dvec3 x = face.momentA / face.areaA;
            const glm::dvec3 r = x - fragments[a].centroid;
            const glm::dvec3 n = face.areaVectorA;
            areaSum[a] += face.areaA;
            areaVectorSum[a] += n;
            if (isBaseFace) {
                baseAreaSum[a] += face.areaA;
                baseVectorSum[a] += n;
            } else {
                surfaceAreaSum[a] += face.areaA;
                surfaceVectorSum[a] += n;
            }
            absTermSum[a] += glm::length(r) * glm::length(n);
            ++incidentFaceCount[a];
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    gaussSum[size_t(a) * 9 + col * 3 + row] += r[row] * n[col];
                }
            }
        }
        const uint32_t b = face.fragmentB;
        if (b < fragmentCount && face.areaB > 0.0 && std::isfinite(face.areaB)) {
            const glm::dvec3 x = face.momentB / face.areaB;
            const glm::dvec3 r = x - fragments[b].centroid;
            const glm::dvec3 n = face.areaVectorB;
            areaSum[b] += face.areaB;
            areaVectorSum[b] += n;
            if (isBaseFace) {
                baseAreaSum[b] += face.areaB;
                baseVectorSum[b] += n;
            } else {
                surfaceAreaSum[b] += face.areaB;
                surfaceVectorSum[b] += n;
            }
            absTermSum[b] += glm::length(r) * glm::length(n);
            ++incidentFaceCount[b];
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    gaussSum[size_t(b) * 9 + col * 3 + row] += r[row] * n[col];
                }
            }
        }
    }

    uint32_t reportedCount = 0;
    for (uint32_t i = 0; i < fragmentCount; ++i) {
        const double volume = fragments[i].volume;
        if (!(volume > 0.0) || !std::isfinite(volume)) {
            if (reportedCount++ < 5) {
                std::cerr << "[RVDFragments] fragment " << i << " has non-positive volume " << volume << std::endl;
            }
            valid = false;
            continue;
        }
        if (!(areaSum[i] > 0.0)) {
            if (reportedCount++ < 5) {
                std::cerr << "[RVDFragments] fragment " << i << " has no positive-area faces" << std::endl;
            }
            valid = false;
            continue;
        }
        if (incidentFaceCount[i] < 4) {
            if (reportedCount++ < 5) {
                std::cerr << "[RVDFragments] fragment " << i << " has fewer than four incident faces ("
                          << incidentFaceCount[i] << ")" << std::endl;
            }
            valid = false;
            continue;
        }
        const double closureError = glm::length(areaVectorSum[i]) / areaSum[i];
        double frobSq = 0.0;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                const double target = (row == col) ? volume : 0.0;
                const double diff = gaussSum[size_t(i) * 9 + col * 3 + row] - target;
                frobSq += diff * diff;
            }
        }
        const double gaussError = std::sqrt(frobSq) / volume;
        const double errorBound =
            128.0 * std::numeric_limits<double>::epsilon() * absTermSum[i] / volume;
        if (closureError > toleranceClosure || (gaussError > toleranceGauss && gaussError > errorBound)) {
            const double baseClosure = baseAreaSum[i] > 0.0
                ? glm::length(baseVectorSum[i]) / baseAreaSum[i]
                : 0.0;
            const double surfaceClosure = surfaceAreaSum[i] > 0.0
                ? glm::length(surfaceVectorSum[i]) / surfaceAreaSum[i]
                : 0.0;
            if (reportedCount++ < 5) {
                std::cerr << "[RVDFragments] fragment " << i
                          << " owner=" << fragments[i].ownerModelId
                          << " cell=" << fragments[i].cellId
                          << " failed geometry validation:"
                          << " volume=" << volume
                          << " closureErr=" << closureError << " (tol " << toleranceClosure << ")"
                          << " baseArea=" << baseAreaSum[i]
                          << " baseClosure=" << baseClosure
                          << " surfaceArea=" << surfaceAreaSum[i]
                          << " surfaceClosure=" << surfaceClosure
                          << " gaussErr=" << gaussError << " (tol " << toleranceGauss
                          << ", bound " << errorBound << ")"
                          << std::endl;
            }
            valid = false;
        }
    }
    return valid;
}

bool validateFragmentTopology(
    const RVDResult& result,
    uint32_t totalBaseFragmentFaces,
    double toleranceClosure,
    double toleranceGauss)
{
    if (result.fragments.empty()) return true;

    if (!validateFragmentGeometryHost(
            result.fragments, result.fragmentFaces, totalBaseFragmentFaces,
            toleranceClosure, toleranceGauss)) {
        std::cerr << "[RVD] geometric topology validation failed" << std::endl;
        return false;
    }

    return true;
}

} // namespace voronoi
