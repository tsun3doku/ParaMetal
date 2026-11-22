#include "DomainMoments.cuh"

#include "CudaBuffer.cuh"
#include "CudaCheck.cuh"
#include "GeometryMoments.cuh"
#include "../util/GeometryUtils.hpp"

#include <iostream>
#include <cmath>

static_assert(sizeof(glm::dvec3) == sizeof(double3), "glm::dvec3 must pack as double3");

namespace domainMoments {

static constexpr uint32_t MomentThreads = 96u;

static bool offsetsValid(const std::vector<uint32_t>& offsets, size_t entryCount) {
    if (offsets.size() != entryCount + 1u) return false;
    for (size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] < offsets[i - 1u]) return false;
    }
    return true;
}

static __global__ void cellMomentsKernel(
    const uint32_t* cellFaceOffsets,
    const uint32_t* cellFaces,
    const double* cellFaceSides,
    const double3* cellOrigins,
    const double3* faceNormals,
    const uint32_t* faceLoopOffsets,
    const uint32_t* loopVertexOffsets,
    const double3* loopVertices,
    double* outMoments) {

    const uint32_t cell = blockIdx.x;
    const uint32_t monomialSlot = threadIdx.x;
    if (monomialSlot >= cudaMoments::monomialLimit) return;

    const cudaMoments::Monomial monomial = cudaMoments::monomialAt(monomialSlot);
    const double3 origin = cellOrigins[cell];
    double total = 0.0;

    for (uint32_t k = cellFaceOffsets[cell]; k < cellFaceOffsets[cell + 1u]; ++k) {
        const uint32_t face = cellFaces[k];
        const double3 normal = faceNormals[face];
        const uint32_t loopBegin = faceLoopOffsets[face];
        const uint32_t loopEnd = faceLoopOffsets[face + 1u];
        const double3 anchor = loopVertices[loopVertexOffsets[loopBegin]];
        const double side = cellFaceSides[k];
        const double offset =
            side * cudaMath::dot3(cudaMath::sub3(anchor, origin), normal);
        for (uint32_t loop = loopBegin; loop < loopEnd; ++loop) {
            const uint32_t vertexBegin = loopVertexOffsets[loop];
            const uint32_t vertexEnd = loopVertexOffsets[loop + 1u];
            total += offset * cudaMoments::faceMonomial(
                loopVertices + vertexBegin, vertexEnd - vertexBegin, origin, normal, monomial);
        }
    }

    outMoments[size_t(cell) * cudaMoments::monomialLimit + monomialSlot] =
        total / double(cudaMoments::monomialDivisor(monomial));
}

bool buildCellMoments(
    uint32_t faceCount,
    const std::vector<double>& faceAreasA,
    const std::vector<double>& faceAreasB,
    const std::vector<glm::dvec3>& faceAreaVectorsA,
    const std::vector<glm::dvec3>& faceAreaVectorsB,
    const std::vector<uint32_t>& faceLoopOffsets,
    const std::vector<uint32_t>& loopVertexOffsets,
    const std::vector<glm::dvec3>& loopVertices,
    uint32_t cellCount,
    const std::vector<uint32_t>& cellFaceOffsets,
    const std::vector<uint32_t>& cellFaces,
    const std::vector<double>& cellFaceSides,
    const std::vector<glm::dvec3>& cellOrigins,
    std::vector<double>& outMoments) {

    outMoments.assign(size_t(cellCount) * cudaMoments::monomialLimit, 0.0);
    if (cellCount == 0u) return true;

    if (faceAreasA.size() != faceCount || faceAreasB.size() != faceCount ||
        faceAreaVectorsA.size() != faceCount || faceAreaVectorsB.size() != faceCount) {
        std::cerr << "[DomainMoments] face array sizes invalid faces=" << faceCount << "\n";
        return false;
    }
    if (faceLoopOffsets.empty() ||
        !offsetsValid(faceLoopOffsets, faceCount)) {
        std::cerr << "[DomainMoments] face loop offsets invalid faces=" << faceCount << "\n";
        return false;
    }
    if (loopVertexOffsets.empty() ||
        !offsetsValid(loopVertexOffsets, faceLoopOffsets.back()) ||
        loopVertices.size() != loopVertexOffsets.back()) {
        std::cerr << "[DomainMoments] loop vertex arrays incomplete loops="
                  << faceLoopOffsets.back() << "\n";
        return false;
    }
    if (cellFaceOffsets.empty() || !offsetsValid(cellFaceOffsets, cellCount) ||
        cellFaces.size() != cellFaceOffsets.back() ||
        cellFaceSides.size() != cellFaces.size() ||
        cellOrigins.size() != cellCount) {
        std::cerr << "[DomainMoments] cell incidence mismatch cells=" << cellCount << "\n";
        return false;
    }
    for (uint32_t cell = 0; cell < cellCount; ++cell) {
        if (!GeometryUtils::isFinite(cellOrigins[cell])) {
            std::cerr << "[DomainMoments] non-finite centroid on cell " << cell << "\n";
            return false;
        }
    }

    std::vector<uint8_t> incident(faceCount, 0u);
    for (size_t k = 0; k < cellFaces.size(); ++k) {
        if (cellFaces[k] >= faceCount) {
            std::cerr << "[DomainMoments] incidence references face " << cellFaces[k]
                      << " of " << faceCount << "\n";
            return false;
        }
        if (cellFaceSides[k] != 1.0 && cellFaceSides[k] != -1.0) {
            std::cerr << "[DomainMoments] incidence side " << cellFaceSides[k]
                      << " on face " << cellFaces[k] << "\n";
            return false;
        }
        incident[cellFaces[k]] = 1u;
    }

    std::vector<double3> faceNormals(faceCount, make_double3(0.0, 0.0, 0.0));
    for (size_t faceIndex = 0; faceIndex < faceCount; ++faceIndex) {
        if (!incident[faceIndex]) continue;
        const uint32_t loopBegin = faceLoopOffsets[faceIndex];
        const uint32_t loopEnd = faceLoopOffsets[faceIndex + 1u];
        if (loopBegin >= loopEnd) {
            std::cerr << "[DomainMoments] face " << faceIndex << " has no loops\n";
            return false;
        }
        for (uint32_t loop = loopBegin; loop < loopEnd; ++loop) {
            if (loopVertexOffsets[loop + 1u] < loopVertexOffsets[loop] + 3u) {
                std::cerr << "[DomainMoments] face " << faceIndex << " loop " << loop
                          << " has "
                          << (loopVertexOffsets[loop + 1u] - loopVertexOffsets[loop])
                          << " vertices\n";
                return false;
            }
        }
        glm::dvec3 normalA(0.0);
        if (std::isfinite(faceAreasA[faceIndex]) && faceAreasA[faceIndex] > 0.0) {
            normalA = faceAreaVectorsA[faceIndex] / faceAreasA[faceIndex];
        } else if (std::isfinite(faceAreasB[faceIndex]) && faceAreasB[faceIndex] > 0.0) {
            normalA = -faceAreaVectorsB[faceIndex] / faceAreasB[faceIndex];
        } else {
            std::cerr << "[DomainMoments] face " << faceIndex
                      << " has no positive area areaA=" << faceAreasA[faceIndex]
                      << " areaB=" << faceAreasB[faceIndex] << "\n";
            return false;
        }
        if (!GeometryUtils::isFinite(normalA)) {
            std::cerr << "[DomainMoments] face " << faceIndex
                      << " has non-finite normal areaA=" << faceAreasA[faceIndex]
                      << " areaB=" << faceAreasB[faceIndex] << "\n";
            return false;
        }
        faceNormals[faceIndex] = make_double3(normalA.x, normalA.y, normalA.z);
    }

    cudaUtils::CudaBuffer<uint32_t> deviceCellFaceOffsets;
    cudaUtils::CudaBuffer<uint32_t> deviceCellFaces;
    cudaUtils::CudaBuffer<double> deviceCellFaceSides;
    cudaUtils::CudaBuffer<double3> deviceCellOrigins;
    cudaUtils::CudaBuffer<double3> deviceFaceNormals;
    cudaUtils::CudaBuffer<uint32_t> deviceFaceLoopOffsets;
    cudaUtils::CudaBuffer<uint32_t> deviceLoopVertexOffsets;
    cudaUtils::CudaBuffer<double3> deviceLoopVertices;
    cudaUtils::CudaBuffer<double> deviceMoments;

    const auto stage = [](auto& buffer, const auto* source, size_t count) {
        return buffer.allocate(count) && buffer.upload(source, count);
    };
    if (!stage(deviceCellFaceOffsets, cellFaceOffsets.data(), cellFaceOffsets.size()) ||
        !stage(deviceCellFaces, cellFaces.data(), cellFaces.size()) ||
        !stage(deviceCellFaceSides, cellFaceSides.data(), cellFaceSides.size()) ||
        !stage(deviceCellOrigins, reinterpret_cast<const double3*>(cellOrigins.data()),
               cellOrigins.size()) ||
        !stage(deviceFaceNormals, faceNormals.data(), faceNormals.size()) ||
        !stage(deviceFaceLoopOffsets, faceLoopOffsets.data(),
               faceLoopOffsets.size()) ||
        !stage(deviceLoopVertexOffsets, loopVertexOffsets.data(),
               loopVertexOffsets.size()) ||
        !stage(deviceLoopVertices, reinterpret_cast<const double3*>(loopVertices.data()),
               loopVertices.size()) ||
        !deviceMoments.allocate(outMoments.size())) {
        std::cerr << "[DomainMoments] device upload failed cells=" << cellCount << "\n";
        return false;
    }

    cellMomentsKernel<<<cellCount, MomentThreads>>>(
        deviceCellFaceOffsets.get(), deviceCellFaces.get(), deviceCellFaceSides.get(),
        deviceCellOrigins.get(), deviceFaceNormals.get(), deviceFaceLoopOffsets.get(),
        deviceLoopVertexOffsets.get(), deviceLoopVertices.get(), deviceMoments.get());
    if (!cudaUtils::cudaCheck(cudaGetLastError(), "launch cell moments") ||
        !deviceMoments.download(outMoments.data(), outMoments.size())) {
        return false;
    }
    return true;
}

} // namespace domainMoments
