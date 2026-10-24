#include "RVDTopology.hpp"
#include "ConvexCell.cuh"

#include <cub/cub.cuh>
#include <cfloat>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace voronoi {

static const char* topologyErrorString(uint32_t code) {
    switch (code) {
        case 1: return "NonFiniteArea";
        case 2: return "InvalidDomainFace";
        case 3: return "InvalidNeighborCell";
        case 4: return "MissingReciprocal";
        case 5: return "NonFiniteReciprocalArea";
        case 6: return "NonReciprocalActivity";
        case 7: return "AreaMismatch";
        case 8: return "TagMismatch";
        case 9: return "NormalNotAntiparallel";
        case 10: return "OffsetMismatch";
        default: return "None";
    }
}

__device__ inline void reportTopologyError(unsigned long long* devPackedError, uint32_t code, uint32_t cell) {
    const unsigned long long packed = (static_cast<unsigned long long>(code) << 32) | static_cast<unsigned long long>(cell);
    atomicCAS(devPackedError, 0ULL, packed);
}

static __global__ void classifyAndVerifyBasePlanesKernel(
    DeviceBaseCells baseCells,
    const uint32_t* cellPlaneOffsets,
    uint32_t cellCount,
    double invBaseScale,
    double invBaseAreaScale,
    double facetFloor,
    float nominalCellSize,
    uint32_t* faceFlags,
    uint32_t* reciprocalSlots,
    unsigned long long* devPackedError)
{
    const uint32_t cellA = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellA >= cellCount) return;

    const BaseCellArrays cellStateA = resolveBaseCell(baseCells, cellA);
    const uint32_t countA = cellStateA.planeCount;
    const uint32_t offsetA = cellPlaneOffsets[cellA];
    const double h = double(nominalCellSize);
    const double tol = fmax(facetFloor, 1e-10 * h * h);

    for (uint32_t localPlaneA = 0; localPlaneA < countA; ++localPlaneA) {
        const uint32_t slotA = offsetA + localPlaneA;
        faceFlags[slotA] = 0;
        reciprocalSlots[slotA] = InvalidId;

        const int tagA = cellStateA.planeNeighborIds[localPlaneA];
        const double canonicalAreaA = double(cellStateA.planeAreas[localPlaneA]);
        const double areaA = canonicalAreaA * invBaseAreaScale;

        if (!isfinite(areaA) || areaA < -tol) {
            reportTopologyError(devPackedError, 1, cellA);
            return;
        }

        if (tagA < 0) {
            const int domainFace = rvdDevice::domainFaceFromTag(tagA);
            if (domainFace < 0 || domainFace >= 6) {
                reportTopologyError(devPackedError, 2, cellA);
                return;
            }

            if (areaA > tol) {
                faceFlags[slotA] = 1;
                reciprocalSlots[slotA] = InvalidId;
            }
        } else {
            const uint32_t cellB = uint32_t(tagA);
            if (cellB >= cellCount || cellB == cellA) {
                reportTopologyError(devPackedError, 3, cellA);
                return;
            }

            const BaseCellArrays cellStateB = resolveBaseCell(baseCells, cellB);
            const uint32_t countB = cellStateB.planeCount;
            const uint32_t offsetB = cellPlaneOffsets[cellB];
            const double4 planeA = cellStateA.planes[localPlaneA];
            const uint64_t bisectorTagA = cellStateA.planeTags[localPlaneA];

            uint32_t localPlaneB = InvalidId;
            for (uint32_t p = 0; p < countB; ++p) {
                if (cellStateB.planeNeighborIds[p] == int(cellA)) {
                    localPlaneB = p;
                    break;
                }
            }

            if (localPlaneB == InvalidId) {
                if (areaA <= tol) {
                    continue;
                } else {
                    reportTopologyError(devPackedError, 4, cellA);
                    return;
                }
            }

            const double canonicalAreaB = double(cellStateB.planeAreas[localPlaneB]);
            const double areaB = canonicalAreaB * invBaseAreaScale;

            if (!isfinite(areaB) || areaB < -tol) {
                reportTopologyError(devPackedError, 5, cellA);
                return;
            }

            const bool activeA = areaA > tol;
            const bool activeB = areaB > tol;

            if (!activeA && !activeB) {
                continue;
            }

            if (activeA != activeB) {
                reportTopologyError(devPackedError, 6, cellA);
                return;
            }

            const double maxA = fmax(areaA, areaB);
            const double areaDiff = fabs(areaA - areaB);
            if (areaDiff > fmax(tol, 1e-8 * maxA)) {
                reportTopologyError(devPackedError, 7, cellA);
                return;
            }

            const uint64_t bisectorTagB = cellStateB.planeTags[localPlaneB];
            if (bisectorTagA != 0 && bisectorTagB != 0 && bisectorTagA != bisectorTagB) {
                reportTopologyError(devPackedError, 8, cellA);
                return;
            }

            const double4 planeB = cellStateB.planes[localPlaneB];
            const double dotN = planeA.x * planeB.x + planeA.y * planeB.y + planeA.z * planeB.z;
            if (dotN > -0.9999) {
                reportTopologyError(devPackedError, 9, cellA);
                return;
            }

            const double offsetSum = planeA.w + planeB.w;
            const double offsetTol = fmax(1e-5, 1e-4 * double(nominalCellSize));
            if (fabs(offsetSum) > offsetTol) {
                reportTopologyError(devPackedError, 10, cellA);
                return;
            }

            const uint32_t slotB = offsetB + localPlaneB;
            if (cellA < cellB) {
                faceFlags[slotA] = 1;
                reciprocalSlots[slotA] = slotB;
            }
        }
    }
}

static __global__ void emitBaseFacesKernel(
    DeviceBaseCells baseCells,
    const uint32_t* cellPlaneOffsets,
    const uint32_t* faceFlags,
    const uint32_t* faceOffsets,
    const uint32_t* reciprocalSlots,
    uint32_t cellCount,
    BaseFace* outBaseFaces)
{
    const uint32_t cellA = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellA >= cellCount) return;

    const BaseCellArrays cellStateA = resolveBaseCell(baseCells, cellA);
    const uint32_t countA = cellStateA.planeCount;
    const uint32_t offsetA = cellPlaneOffsets[cellA];

    for (uint32_t localPlaneA = 0; localPlaneA < countA; ++localPlaneA) {
        const uint32_t slotA = offsetA + localPlaneA;
        if (faceFlags[slotA] == 1) {
            const uint32_t dstIdx = faceOffsets[slotA];
            const uint32_t slotB = reciprocalSlots[slotA];
            const int tagA = cellStateA.planeNeighborIds[localPlaneA];
            const uint32_t cellB = (tagA >= 0) ? uint32_t(tagA) : InvalidId;
            const int domainIndex = rvdDevice::domainFaceFromTag(tagA);
            const uint32_t dFace = (domainIndex >= 0) ? uint32_t(domainIndex) : InvalidId;

            BaseFace face{};
            face.cellA = cellA;
            face.cellB = cellB;
            face.slotA = slotA;
            face.slotB = slotB;
            face.domainFace = dFace;

            outBaseFaces[dstIdx] = face;
        }
    }
}

static __global__ void buildBaseFaceMeasuresKernel(
    const BaseFace* baseFaces,
    uint32_t faceCount,
    DeviceBaseCells baseCells,
    const uint32_t* cellPlaneOffsets,
    double invBaseAreaScale,
    double invBaseScale,
    double3 domainMin,
    BaseFaceMeasure* outMeasures)
{
    const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
    if (faceId >= faceCount) return;

    const BaseFace& face = baseFaces[faceId];
    const BaseCellArrays cellA = resolveBaseCell(baseCells, face.cellA);
    const uint32_t lpA = face.slotA - cellPlaneOffsets[face.cellA];
    const double canAreaA = double(cellA.planeAreas[lpA]);
    const double areaA = canAreaA * invBaseAreaScale;
    const double3 canMomA = cellA.planeMoments[lpA];
    const double momentScale = invBaseAreaScale * invBaseScale;
    const double3 momentA = make_double3(
        domainMin.x * areaA + canMomA.x * momentScale,
        domainMin.y * areaA + canMomA.y * momentScale,
        domainMin.z * areaA + canMomA.z * momentScale);
    const double4 facePlane = cellA.planes[lpA];
    const double normalLength = sqrt(
        double(facePlane.x) * double(facePlane.x) +
        double(facePlane.y) * double(facePlane.y) +
        double(facePlane.z) * double(facePlane.z));
    const double3 normal = normalLength > 0.0
        ? make_double3(double(facePlane.x) / normalLength,
                       double(facePlane.y) / normalLength,
                       double(facePlane.z) / normalLength)
        : make_double3(0.0, 0.0, 0.0);
    const double absAreaA = cellA.planeAreaAbsSums != nullptr
        ? double(cellA.planeAreaAbsSums[lpA]) * invBaseAreaScale
        : fabs(areaA);
    double areaError = 256.0 * DBL_EPSILON * absAreaA;
    bool validA = areaA > 0.0 && isfinite(areaA) &&
        isfinite(momentA.x) && isfinite(momentA.y) && isfinite(momentA.z) &&
        normalLength > 0.0 && isfinite(normalLength);

    double area = areaA;
    double3 moment = momentA;
    bool valid = validA;

    if (face.cellB != InvalidId) {
        const BaseCellArrays cellB = resolveBaseCell(baseCells, face.cellB);
        const uint32_t lpB = face.slotB - cellPlaneOffsets[face.cellB];
        const double canAreaB = double(cellB.planeAreas[lpB]);
        const double areaB = canAreaB * invBaseAreaScale;
        const double3 canMomB = cellB.planeMoments[lpB];
        const double3 momentB = make_double3(
            domainMin.x * areaB + canMomB.x * momentScale,
            domainMin.y * areaB + canMomB.y * momentScale,
            domainMin.z * areaB + canMomB.z * momentScale);
        const double absAreaB = cellB.planeAreaAbsSums != nullptr
            ? double(cellB.planeAreaAbsSums[lpB]) * invBaseAreaScale
            : fabs(areaB);
        areaError += 256.0 * DBL_EPSILON * absAreaB;
        const bool validB = areaB > 0.0 && isfinite(areaB) &&
            isfinite(momentB.x) && isfinite(momentB.y) && isfinite(momentB.z);

        valid = validA || validB;

        if (!validA && validB) {
            area = areaB;
            moment = momentB;
        }
    }

    BaseFaceMeasure measure{};
    measure.area = area;
    measure.areaError = areaError;
    measure.moment = moment;
    measure.normal = make_double3(-normal.x, -normal.y, -normal.z);
    measure.valid = valid ? 1u : 0u;
    outMeasures[faceId] = measure;
}

static __global__ void countBaseFacePolygonVerticesKernel(
    const BaseFace* baseFaces,
    uint32_t faceCount,
    DeviceBaseCells baseCells,
    const uint32_t* cellPlaneOffsets,
    uint32_t* vertexCounts)
{
    const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
    if (faceId > faceCount) return;
    if (faceId == faceCount) {
        vertexCounts[faceId] = 0u;
        return;
    }

    const BaseFace& face = baseFaces[faceId];
    const BaseCellArrays cell = resolveBaseCell(baseCells, face.cellA);
    const uint32_t localPlane = face.slotA - cellPlaneOffsets[face.cellA];

    uint32_t count = 0;
    for (uint32_t v = 0; v < cell.vertexCount; ++v) {
        const uint3 triplet = cell.vertices[v];
        if (triplet.x == localPlane || triplet.y == localPlane || triplet.z == localPlane) {
            ++count;
        }
    }
    vertexCounts[faceId] = count;
}

static __global__ void emitBaseFacePolygonVerticesKernel(
    const BaseFace* baseFaces,
    uint32_t faceCount,
    DeviceBaseCells baseCells,
    const uint32_t* cellPlaneOffsets,
    const uint32_t* vertexOffsets,
    double invBaseScale,
    double3 domainMin,
    double3* outVertices)
{
    const uint32_t faceId = blockIdx.x * blockDim.x + threadIdx.x;
    if (faceId >= faceCount) return;

    const BaseFace& face = baseFaces[faceId];
    const BaseCellArrays cell = resolveBaseCell(baseCells, face.cellA);
    const uint32_t localPlane = face.slotA - cellPlaneOffsets[face.cellA];

    rvdDevice::ConvexCell view{};
    view.planes = const_cast<double4*>(cell.planes);
    view.planeCount = int(cell.planeCount);

    uint32_t dst = vertexOffsets[faceId];
    for (uint32_t v = 0; v < cell.vertexCount; ++v) {
        const uint3 triplet = cell.vertices[v];
        if (triplet.x != localPlane && triplet.y != localPlane && triplet.z != localPlane) {
            continue;
        }
        const double3 canonical = view.vertexPosition(triplet);
        outVertices[dst++] = make_double3(
            domainMin.x + canonical.x * invBaseScale,
            domainMin.y + canonical.y * invBaseScale,
            domainMin.z + canonical.z * invBaseScale);
    }
}

bool buildBaseFaces(
    const BaseCellDomain& baseDomain,
    BaseFaceTopology& faceTopology,
    cudaStream_t stream)
{
    const uint32_t cellCount = baseDomain.count;
    const uint32_t totalPlanes = baseDomain.totalPlanes;
    if (cellCount == 0 || totalPlanes == 0) {
        faceTopology.totalBaseFaces = 0;
        return true;
    }

    cudaUtils::CudaBuffer<uint32_t> devFaceFlags;
    cudaUtils::CudaBuffer<uint32_t> devFaceOffsets;
    cudaUtils::CudaBuffer<uint32_t> devReciprocalSlots;
    cudaUtils::CudaBuffer<unsigned long long> devPackedError;

    if (!devFaceFlags.allocate(totalPlanes + 1) ||
        !devFaceOffsets.allocate(totalPlanes + 1) ||
        !devReciprocalSlots.allocate(totalPlanes) ||
        !devPackedError.allocate(1)) {
        return false;
    }

    if (!cudaUtils::cudaCheck(cudaMemsetAsync(devFaceFlags.get() + totalPlanes, 0, sizeof(uint32_t), stream), "clear tail faceFlag") ||
        !cudaUtils::cudaCheck(cudaMemsetAsync(devPackedError.get(), 0, sizeof(unsigned long long), stream), "clear devPackedError")) {
        return false;
    }

    constexpr uint32_t BlockSize = 256;
    const uint32_t blocks = (cellCount + BlockSize - 1) / BlockSize;

    classifyAndVerifyBasePlanesKernel<<<blocks, BlockSize, 0, stream>>>(
        baseDomain.deviceView(),
        baseDomain.devCellPlaneOffsets.get(),
        cellCount,
        baseDomain.invBaseScale,
        baseDomain.invBaseAreaScale,
        baseDomain.facetFloor(),
        baseDomain.nominalCellSize,
        devFaceFlags.get(),
        devReciprocalSlots.get(),
        devPackedError.get());

    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch classifyAndVerifyBasePlanesKernel")) {
        return false;
    }

    unsigned long long packedError = 0;
    if (cudaMemcpyAsync(&packedError, devPackedError.get(), sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        !cudaUtils::cudaCheck(cudaStreamSynchronize(stream), "sync error check")) {
        return false;
    }

    if (packedError != 0) {
        const uint32_t errorFlag = static_cast<uint32_t>(packedError >> 32);
        const uint32_t errorCell = static_cast<uint32_t>(packedError & 0xFFFFFFFFULL);
        std::cerr << "[RVD-Topology-GPU] Topological defect detected: "
                  << topologyErrorString(errorFlag) << " (code=" << errorFlag
                  << ") on cell " << errorCell << std::endl;
        return false;
    }

    size_t scanBytes = 0;
    if (cub::DeviceScan::ExclusiveSum(
            nullptr, scanBytes,
            devFaceFlags.get(),
            devFaceOffsets.get(),
            int(totalPlanes + 1),
            stream) != cudaSuccess) {
        return false;
    }

    cudaUtils::CudaBuffer<uint8_t> scanScratch;
    if (!scanScratch.allocate(scanBytes)) return false;

    if (cub::DeviceScan::ExclusiveSum(
            scanScratch.get(), scanBytes,
            devFaceFlags.get(),
            devFaceOffsets.get(),
            int(totalPlanes + 1),
            stream) != cudaSuccess) {
        return false;
    }

    uint32_t totalFaces = 0;
    if (cudaMemcpyAsync(&totalFaces, devFaceOffsets.get() + totalPlanes, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        !cudaUtils::cudaCheck(cudaStreamSynchronize(stream), "readback totalFaces")) {
        return false;
    }

    faceTopology.totalBaseFaces = totalFaces;

    if (!faceTopology.devBaseFaces.allocate(totalFaces) ||
        !faceTopology.devMeasures.allocate(totalFaces)) {
        return false;
    }

    if (totalFaces > 0) {
        emitBaseFacesKernel<<<blocks, BlockSize, 0, stream>>>(
            baseDomain.deviceView(),
            baseDomain.devCellPlaneOffsets.get(),
            devFaceFlags.get(),
            devFaceOffsets.get(),
            devReciprocalSlots.get(),
            cellCount,
            faceTopology.devBaseFaces.get());

        if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch emitBaseFacesKernel")) {
            return false;
        }

        const uint32_t measureBlocks = (totalFaces + BlockSize - 1) / BlockSize;
        buildBaseFaceMeasuresKernel<<<measureBlocks, BlockSize, 0, stream>>>(
            faceTopology.devBaseFaces.get(),
            totalFaces,
            baseDomain.deviceView(),
            baseDomain.devCellPlaneOffsets.get(),
            baseDomain.invBaseAreaScale,
            baseDomain.invBaseScale,
            make_double3(baseDomain.domainMin.x, baseDomain.domainMin.y, baseDomain.domainMin.z),
            faceTopology.devMeasures.get());

        if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch buildBaseFaceMeasuresKernel")) {
            return false;
        }

        cudaUtils::CudaBuffer<uint32_t> devFaceVertexCounts;
        if (!devFaceVertexCounts.allocate(totalFaces + 1) ||
            !faceTopology.devFaceVertexOffsets.allocate(totalFaces + 1)) {
            return false;
        }

        const uint32_t polygonCountBlocks = (totalFaces + 1 + BlockSize - 1) / BlockSize;
        countBaseFacePolygonVerticesKernel<<<polygonCountBlocks, BlockSize, 0, stream>>>(
            faceTopology.devBaseFaces.get(),
            totalFaces,
            baseDomain.deviceView(),
            baseDomain.devCellPlaneOffsets.get(),
            devFaceVertexCounts.get());
        if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch countBaseFacePolygonVerticesKernel")) {
            return false;
        }

        size_t polygonScanBytes = 0;
        if (cub::DeviceScan::ExclusiveSum(
                nullptr, polygonScanBytes,
                devFaceVertexCounts.get(),
                faceTopology.devFaceVertexOffsets.get(),
                int(totalFaces + 1),
                stream) != cudaSuccess) {
            return false;
        }

        cudaUtils::CudaBuffer<uint8_t> polygonScanScratch;
        if (!polygonScanScratch.allocate(polygonScanBytes)) return false;

        if (cub::DeviceScan::ExclusiveSum(
                polygonScanScratch.get(), polygonScanBytes,
                devFaceVertexCounts.get(),
                faceTopology.devFaceVertexOffsets.get(),
                int(totalFaces + 1),
                stream) != cudaSuccess) {
            return false;
        }

        uint32_t totalFaceVertices = 0;
        if (cudaMemcpyAsync(&totalFaceVertices,
                            faceTopology.devFaceVertexOffsets.get() + totalFaces,
                            sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            !cudaUtils::cudaCheck(cudaStreamSynchronize(stream), "readback totalFaceVertices")) {
            return false;
        }

        if (totalFaceVertices > 0) {
            if (!faceTopology.devFaceVertices.allocate(totalFaceVertices)) return false;
            emitBaseFacePolygonVerticesKernel<<<measureBlocks, BlockSize, 0, stream>>>(
                faceTopology.devBaseFaces.get(),
                totalFaces,
                baseDomain.deviceView(),
                baseDomain.devCellPlaneOffsets.get(),
                faceTopology.devFaceVertexOffsets.get(),
                baseDomain.invBaseScale,
                make_double3(baseDomain.domainMin.x, baseDomain.domainMin.y, baseDomain.domainMin.z),
                faceTopology.devFaceVertices.get());
            if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch emitBaseFacePolygonVerticesKernel")) {
                return false;
            }
        }
    }

    return true;
}

} // namespace voronoi
