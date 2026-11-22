#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>
#include "VoronoiConfig.hpp"
#include "VoronoiGpuStructs.hpp"

struct VoronoiProduct;

namespace voronoi {
struct RVDResult;
}

class VoronoiRuntime {
public:
    bool isReady() const { return voronoiReady; }
    bool validateTopology() const;
    bool materializeTopology(
        voronoi::RVDResult&& result,
        const std::vector<uint32_t>& runtimeModelIds,
        const glm::vec3& domainMin,
        const glm::vec3& domainMax,
        const glm::vec3& gridMin,
        const glm::ivec3& gridDim,
        float gridSpacing);
    void publishTopology(VoronoiProduct& product) const;

    const std::vector<uint32_t>& getGlobalRemeshRuntimeModelIds() const { return inGlobalRemeshRuntimeModelIds; }
    const std::vector<std::vector<glm::vec3>>& getGlobalRemeshPositions() const { return inGlobalRemeshPositions; }
    const std::vector<std::vector<uint32_t>>& getGlobalRemeshTriangleIndices() const { return inGlobalRemeshTriangleIndices; }
    const std::vector<std::vector<glm::vec3>>& getGlobalRemeshSurfacePositions() const { return inGlobalRemeshSurfacePositions; }
    const std::vector<std::vector<uint32_t>>& getGlobalRemeshSurfaceTriangleIndices() const { return inGlobalRemeshSurfaceTriangleIndices; }
    float getGlobalVolumePadding() const { return inGlobalVolumePadding; }

    std::vector<glm::vec4>& getSeedPositions() { return inSeedPositions; }
    const std::vector<glm::vec4>& getSeedPositions() const { return inSeedPositions; }
    const std::array<glm::vec3, 8>& getPointDomainCorners() const { return inPointDomainCorners; }

    void clearGeometry();
    void setParams(int inVoxelResolution);
    int getVoxelResolution() const { return inVoxelResolution; }
    int getLloydIterations() const { return inLloydIterations; }
    float getLloydVolumeThreshold() const { return inLloydVolumeThreshold; }
    float getLloydDamping() const { return inLloydDamping; }
    float getContactRange() const { return inContactRange; }
    void setLloydParams(int iterations, float volumeThreshold, float damping);
    void setContactRange(float range);
    void markReady();

    void setSeedPositions(
        const std::vector<glm::vec4>& positions,
        const std::array<glm::vec3, 8>& domainCorners);
    void setGlobalGeometry(
        const std::vector<uint32_t>& runtimeModelIds,
        const std::vector<std::vector<glm::vec3>>& positions,
        const std::vector<std::vector<uint32_t>>& triangleIndices,
        const std::vector<std::vector<glm::vec3>>& surfacePositions,
        const std::vector<std::vector<uint32_t>>& surfaceTriangleIndices,
        float volumePadding);

    void cleanup();

private:
    void invalidateMaterialization();
    void clearMaterializedTopology();
    bool buildSpacePartition(size_t seedCount);

    std::vector<uint32_t> outFragmentInstanceIds;
    std::vector<uint32_t> outFragmentSeedIds;
    std::vector<glm::dvec3> outFragmentCentroids;
    std::vector<double> outFragmentVolumes;

    std::vector<uint32_t> outFaceFragmentA;
    std::vector<uint32_t> outFaceFragmentB;
    std::vector<double> outFaceAreasA;
    std::vector<double> outFaceAreasB;
    std::vector<glm::dvec3> outFaceAreaVectorsA;
    std::vector<glm::dvec3> outFaceAreaVectorsB;
    std::vector<glm::dvec3> outFaceMomentsA;
    std::vector<glm::dvec3> outFaceMomentsB;
    std::vector<uint8_t> outFaceTypes;
    std::vector<uint32_t> outFaceDomainFaces;

    std::vector<uint32_t> outBoundaryFaceIndices;
    std::vector<uint32_t> outFaceLoopOffsets;
    std::vector<uint32_t> outLoopVertexOffsets;
    std::vector<glm::dvec3> outLoopVertices;

    std::vector<uint32_t> outContactCouplingFaceA;
    std::vector<uint32_t> outContactCouplingFragmentA;
    std::vector<uint32_t> outContactCouplingFaceB;
    std::vector<uint32_t> outContactCouplingFragmentB;
    std::vector<double> outContactCouplingAreas;

    std::vector<glm::vec3> outSeedPositions;
    std::vector<uint32_t> outCellFragmentOffsets;
    std::vector<uint32_t> outCellFragments;
    std::vector<uint32_t> outInstanceFragmentCounts;

    std::vector<uint32_t> outFragmentToSpaceIndex;
    std::vector<uint32_t> outSpaceToFragmentIndex;
    std::vector<uint32_t> outSeedToSpaceIndex;
    uint32_t outSpaceFragmentCount = 0;
    bool outHasUniqueSeedToSpaceMapping = false;

    glm::vec3 outDomainMin{0.0f};
    glm::vec3 outDomainMax{0.0f};
    glm::vec3 outVolumeGridMin{0.0f};
    glm::ivec3 outVolumeGridDim{0, 0, 0};
    float outVolumeGridCellSize = 0.0f;
    std::array<glm::vec3, 8> outDomainCorners{};

    int inVoxelResolution = 128;
    int inLloydIterations = int(voronoi::DefaultLloydIterations);
    float inLloydVolumeThreshold = voronoi::DefaultLloydVolumeThresholdFraction;
    float inLloydDamping = voronoi::DefaultLloydDamping;
    float inContactRange = voronoi::DefaultContactRange;

    std::vector<glm::vec4> inSeedPositions;
    std::array<glm::vec3, 8> inPointDomainCorners{};
    std::vector<uint32_t> inGlobalRemeshRuntimeModelIds;
    std::vector<std::vector<glm::vec3>> inGlobalRemeshPositions;
    std::vector<std::vector<uint32_t>> inGlobalRemeshTriangleIndices;
    std::vector<std::vector<glm::vec3>> inGlobalRemeshSurfacePositions;
    std::vector<std::vector<uint32_t>> inGlobalRemeshSurfaceTriangleIndices;
    float inGlobalVolumePadding = 0.0f;

    bool voronoiReady = false;
};


