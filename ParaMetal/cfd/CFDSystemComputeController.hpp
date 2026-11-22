#pragma once

#include "cfd/CFDSystem.hpp"
#include <unordered_map>
#include <memory>
#include <utility>
#include <vector>
#include <cstdint>

struct CFDProduct;

namespace cfd {

class CFDSystemComputeController {
public:
    struct Config {
        std::vector<uint32_t> fragmentInstanceIds;
        std::vector<uint32_t> fragmentSeedIds;
        std::vector<glm::dvec3> fragmentCentroids;
        std::vector<double> fragmentVolumes;

        std::vector<uint32_t> faceFragmentA;
        std::vector<uint32_t> faceFragmentB;
        std::vector<double> faceAreasA;
        std::vector<double> faceAreasB;
        std::vector<glm::dvec3> faceAreaVectorsA;
        std::vector<glm::dvec3> faceAreaVectorsB;
        std::vector<glm::dvec3> faceMomentsA;
        std::vector<glm::dvec3> faceMomentsB;
        std::vector<uint8_t> faceTypes;
        std::vector<uint32_t> faceDomainFaces;

        std::vector<uint32_t> boundaryFaceIndices;
        std::vector<uint32_t> faceLoopOffsets;
        std::vector<uint32_t> loopVertexOffsets;
        std::vector<glm::dvec3> loopVertices;

        glm::vec3 domainMin{0.0f};
        glm::vec3 domainMax{0.0f};
        glm::vec3 volumeGridMin{0.0f};
        glm::ivec3 volumeGridDim{0, 0, 0};
        float volumeGridCellSize = 0.0f;

        std::vector<uint32_t> seedToSpaceIndex;
        std::vector<uint32_t> spaceToFragmentIndex;
        uint32_t spaceFragmentCount = 0;
        bool hasUniqueSeedToSpaceMapping = false;
        VkBuffer seedIdGridBuffer = VK_NULL_HANDLE;
        VkDeviceSize seedIdGridBufferOffset = 0;

        float kinematicViscosity = 1.516e-5f;
        glm::vec3 initialVelocity{0.0f};
        uint32_t domainBoundaryTypes[6] = {};
        glm::vec3 domainBoundaryVelocities[6] = {};
        float fixedTimeStep = 1.0f / 60.0f;
        float targetCourant = 0.5f;
        int pressureIterations = 0;
        float pressureTolerance = 0.0f;
        float simulationDuration = 10.0f;
        uint64_t computeHash = 0;
        uint64_t geometryHash = 0;
    };

    CFDSystemComputeController(
        VulkanDevice& device,
        MemoryAllocator& allocator,
        CommandPool& cmdPool,
        uint32_t maxFramesInFlight);
    ~CFDSystemComputeController();

    bool apply(uint64_t socketKey, const Config& config);
    void remove(uint64_t socketKey);

    void releaseWaits();

    bool buildProduct(uint64_t socketKey, CFDProduct& outProduct);
    const CFDSystem* getSystem(uint64_t socketKey) const;
    CFDSystem* getSystem(uint64_t socketKey);
    uint64_t getCurrentComputeHash(uint64_t socketKey) const {
        const auto it = configuredHashes.find(socketKey);
        return it != configuredHashes.end() ? it->second.second : 0;
    }
    uint64_t getCurrentGeometryHash(uint64_t socketKey) const {
        const auto it = configuredHashes.find(socketKey);
        return it != configuredHashes.end() ? it->second.first : 0;
    }

    bool hasAnyRetiring() const;
    void clearAllRetiring();
    void commitPending(uint64_t socketKey);
    void cancelCandidate(uint64_t socketKey);
    void quiescedClear() noexcept;

private:
    void retireCurrent(uint64_t socketKey);

    struct GenerationSlot {
        std::unique_ptr<CFDSystem> current;
        std::unique_ptr<CFDSystem> retiring;
        std::unique_ptr<CFDSystem> candidate;
    };

    bool getBuildHashes(
        uint64_t socketKey,
        uint64_t& geometryHash,
        uint64_t& simulationHash) const;

    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    CommandPool& commandPool;
    uint32_t maxFramesInFlight;

    std::unordered_map<uint64_t, GenerationSlot> generationSlots;
    std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> configuredHashes;
    std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> pendingHashes;
    std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> failedHashes;
};

} // namespace cfd
