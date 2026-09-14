#include "NodePoints.hpp"

#include "hash/HashBuilder.hpp"
#include "hash/HashNodeCache.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphUtils.hpp"
#include "NodePayloadRegistry.hpp"
#include "NodePointsParams.hpp"
#include "domain/PointData.hpp"

#include <cmath>
#include <random>
#include <vector>
#include <algorithm>

namespace {

inline float distance2(const glm::vec3& a, const glm::vec3& b) {
    const glm::vec3 d = a - b;
    return glm::dot(d, d);
}

void generateUniformGrid(
    uint32_t count,
    float dimX, float dimY, float dimZ,
    std::vector<glm::vec4>& positions) {

    if (count == 0) return;

    const float volume = dimX * dimY * dimZ;
    const float targetSpacing = std::cbrt(volume / static_cast<float>(count));

    uint32_t nx = std::max(1u, static_cast<uint32_t>(std::round(dimX / targetSpacing)));
    uint32_t ny = std::max(1u, static_cast<uint32_t>(std::round(dimY / targetSpacing)));
    uint32_t nz = std::max(1u, static_cast<uint32_t>(std::round(dimZ / targetSpacing)));

    while (uint64_t(nx) * uint64_t(ny) * uint64_t(nz) < uint64_t(count)) {
        const float stepX = dimX / static_cast<float>(nx);
        const float stepY = dimY / static_cast<float>(ny);
        const float stepZ = dimZ / static_cast<float>(nz);
        if (stepX >= stepY && stepX >= stepZ) {
            ++nx;
        } else if (stepY >= stepZ) {
            ++ny;
        } else {
            ++nz;
        }
    }

    const uint64_t totalCells =
        uint64_t(nx) * uint64_t(ny) * uint64_t(nz);

    const float stepX = dimX / static_cast<float>(nx);
    const float stepY = dimY / static_cast<float>(ny);
    const float stepZ = dimZ / static_cast<float>(nz);

    positions.clear();
    positions.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        const uint64_t latticeIndex =
            (uint64_t(i) * totalCells + totalCells / 2) /
            uint64_t(count);

        const uint32_t iz = static_cast<uint32_t>(latticeIndex % nz);
        const uint64_t rem = latticeIndex / nz;
        const uint32_t iy = static_cast<uint32_t>(rem % ny);
        const uint32_t ix = static_cast<uint32_t>(rem / ny);

        glm::vec4 pos{};
        pos.x = -dimX * 0.5f + stepX * (static_cast<float>(ix) + 0.5f);
        pos.y = -dimY * 0.5f + stepY * (static_cast<float>(iy) + 0.5f);
        pos.z = -dimZ * 0.5f + stepZ * (static_cast<float>(iz) + 0.5f);
        pos.w = 1.0f;
        positions.push_back(pos);
    }
}

void generateBlueNoise(
    uint32_t count,
    float dimX,
    float dimY,
    float dimZ,
    uint32_t seed,
    std::vector<glm::vec4>& positions)
{
    if (count == 0) return;

    const float minX = -dimX * 0.5f;
    const float minY = -dimY * 0.5f;
    const float minZ = -dimZ * 0.5f;
    const float maxX = +dimX * 0.5f;
    const float maxY = +dimY * 0.5f;
    const float maxZ = +dimZ * 0.5f;

    std::mt19937_64 rng(seed != 0 ? seed : 1u);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    positions.clear();
    positions.reserve(count);

    const glm::vec4 firstPoint(
        minX + dist01(rng) * dimX,
        minY + dist01(rng) * dimY,
        minZ + dist01(rng) * dimZ,
        1.0f);
    positions.push_back(firstPoint);

    if (count == 1) return;

    const float volume = dimX * dimY * dimZ;
    const float nominalSpacing = std::cbrt(volume / static_cast<float>(count));
    const float cellSize = nominalSpacing;

    const int gx = std::max(1, static_cast<int>(std::ceil(dimX / cellSize)));
    const int gy = std::max(1, static_cast<int>(std::ceil(dimY / cellSize)));
    const int gz = std::max(1, static_cast<int>(std::ceil(dimZ / cellSize)));
    const size_t totalGridCells = static_cast<size_t>(gx) * gy * gz;

    std::vector<int32_t> cellHead(totalGridCells, -1);
    std::vector<int32_t> nextPoint(count, -1);

    auto insertPointIntoGrid = [&](uint32_t index, const glm::vec4& pt) {
        const int cx = std::clamp(static_cast<int>((pt.x - minX) / cellSize), 0, gx - 1);
        const int cy = std::clamp(static_cast<int>((pt.y - minY) / cellSize), 0, gy - 1);
        const int cz = std::clamp(static_cast<int>((pt.z - minZ) / cellSize), 0, gz - 1);
        const size_t cIdx = static_cast<size_t>(cz) * gx * gy + static_cast<size_t>(cy) * gx + cx;
        nextPoint[index] = cellHead[cIdx];
        cellHead[cIdx] = static_cast<int32_t>(index);
    };

    insertPointIntoGrid(0, firstPoint);

    constexpr uint32_t CandidatesPerPoint = 24;
    constexpr uint32_t BruteForceThreshold = 512;

    while (positions.size() < count) {
        glm::vec3 bestCandidate{0.0f};
        float bestCandidateScoreSq = -1.0f;

        for (uint32_t c = 0; c < CandidatesPerPoint; ++c) {
            const glm::vec3 candidate(
                minX + dist01(rng) * dimX,
                minY + dist01(rng) * dimY,
                minZ + dist01(rng) * dimZ);

            const float wallDistance = std::min({
                candidate.x - minX, maxX - candidate.x,
                candidate.y - minY, maxY - candidate.y,
                candidate.z - minZ, maxZ - candidate.z
            });

            float nearestSq = 4.0f * wallDistance * wallDistance;

            if (positions.size() < BruteForceThreshold) {
                for (const auto& p : positions) {
                    nearestSq = std::min(nearestSq, distance2(candidate, glm::vec3(p)));
                }
            } else {
                const int cx = std::clamp(static_cast<int>((candidate.x - minX) / cellSize), 0, gx - 1);
                const int cy = std::clamp(static_cast<int>((candidate.y - minY) / cellSize), 0, gy - 1);
                const int cz = std::clamp(static_cast<int>((candidate.z - minZ) / cellSize), 0, gz - 1);

                // Search own cell first
                const size_t ownIdx = static_cast<size_t>(cz) * gx * gy + static_cast<size_t>(cy) * gx + cx;
                int32_t pIdx = cellHead[ownIdx];
                while (pIdx >= 0) {
                    nearestSq = std::min(nearestSq, distance2(candidate, glm::vec3(positions[pIdx])));
                    pIdx = nextPoint[pIdx];
                }

                // Search expanding shells (Chebyshev distance == shell)
                const int maxShell = std::max({cx, gx - 1 - cx, cy, gy - 1 - cy, cz, gz - 1 - cz});
                for (int shell = 1; shell <= maxShell; ++shell) {
                    const float lowerBound = static_cast<float>(shell - 1) * cellSize;
                    if (nearestSq <= lowerBound * lowerBound) {
                        break;
                    }

                    const int minKx = std::max(0, cx - shell);
                    const int maxKx = std::min(gx - 1, cx + shell);
                    const int minKy = std::max(0, cy - shell);
                    const int maxKy = std::min(gy - 1, cy + shell);
                    const int minKz = std::max(0, cz - shell);
                    const int maxKz = std::min(gz - 1, cz + shell);

                    for (int kz = minKz; kz <= maxKz; ++kz) {
                        const bool zFace = (kz == cz - shell || kz == cz + shell);
                        for (int ky = minKy; ky <= maxKy; ++ky) {
                            const bool yFace = (ky == cy - shell || ky == cy + shell);
                            for (int kx = minKx; kx <= maxKx; ++kx) {
                                const bool xFace = (kx == cx - shell || kx == cx + shell);
                                if (!xFace && !yFace && !zFace) {
                                    continue;
                                }

                                const size_t cIdx = static_cast<size_t>(kz) * gx * gy + static_cast<size_t>(ky) * gx + kx;
                                int32_t qIdx = cellHead[cIdx];
                                while (qIdx >= 0) {
                                    nearestSq = std::min(nearestSq, distance2(candidate, glm::vec3(positions[qIdx])));
                                    qIdx = nextPoint[qIdx];
                                }
                            }
                        }
                    }
                }
            }

            if (nearestSq > bestCandidateScoreSq) {
                bestCandidateScoreSq = nearestSq;
                bestCandidate = candidate;
            }
        }

        const uint32_t newIndex = static_cast<uint32_t>(positions.size());
        const glm::vec4 newPoint(bestCandidate.x, bestCandidate.y, bestCandidate.z, 1.0f);
        positions.push_back(newPoint);
        insertPointIntoGrid(newIndex, newPoint);
    }
}

} // namespace

const char* NodePoints::typeId() const {
    return nodegraphtypes::Points;
}

void NodePoints::execute(NodeKernelEval& eval) const {
    const PointsNodeParams params = readPointsNodeParams(eval.node);
    NodePayloadRegistry* const payloadRegistry = eval.runtime.payloadRegistry;

    for (std::size_t outputIndex = 0; outputIndex < eval.outputs.size() && outputIndex < eval.node.outputs.size(); ++outputIndex) {
        NodeDataBlock& outputValue = eval.outputs[outputIndex];
        const NodeGraphSocket& outputSocket = eval.node.outputs[outputIndex];
        outputValue = {};
        outputValue.dataType = outputSocket.contract.producedPayloadType;

        if (!payloadRegistry || outputValue.dataType != payloadtypes::Points || params.pointCount == 0) {
            populateMetadata(outputValue, nullptr, payloadRegistry);
            continue;
        }

        PointData payload{};
        payload.positions.reserve(params.pointCount);

        const uint32_t count = params.pointCount;
        const float dimX = std::max(params.dimX, 0.001f);
        const float dimY = std::max(params.dimY, 0.001f);
        const float dimZ = std::max(params.dimZ, 0.001f);

        payload.domainMinimum = glm::vec3(-dimX, -dimY, -dimZ) * 0.5f;
        payload.domainMaximum = glm::vec3( dimX,  dimY,  dimZ) * 0.5f;

        if (params.distribution == PointsDistribution::UniformGrid) {
            generateUniformGrid(count, dimX, dimY, dimZ, payload.positions);
        } else {
            generateBlueNoise(count, dimX, dimY, dimZ, params.seed, payload.positions);
        }

        payload.active = true;
        const uint64_t payloadKey = NodeSocketKey(eval.node.id, outputSocket.id);
        outputValue.payloadHandle = payloadRegistry->store(payloadKey, payload, eval.outputHashes);
        populateMetadata(outputValue, nullptr, payloadRegistry);
    }
}

HashValues NodePoints::computeOutputHashes(const NodeKernelHash& hash) const {
    const PointsNodeParams params = readPointsNodeParams(hash.node);
    uint64_t hashValue = HashBuilder::start();
    HashBuilder::combineString(hashValue, nodegraphtypes::Points);
    HashBuilder::combine(hashValue, static_cast<uint64_t>(params.pointCount));
    HashBuilder::combineFloat(hashValue, params.dimX);
    HashBuilder::combineFloat(hashValue, params.dimY);
    HashBuilder::combineFloat(hashValue, params.dimZ);
    HashBuilder::combine(hashValue, static_cast<uint64_t>(params.distribution));
    HashBuilder::combine(hashValue, static_cast<uint64_t>(params.seed));

    HashValues values{};
    values.full = hashValue;
    values.geometry = hashValue;
    values.simulation = hashValue;
    return values;
}
