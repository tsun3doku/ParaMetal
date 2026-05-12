#include "HeatContactRuntime.hpp"

#include "contact/ContactGpuStructs.hpp"
#include "contact/ContactSystemComputeStage.hpp"

#include "contact/ContactSampling.hpp"
#include "util/GMLS.hpp"
#include "vulkan/MemoryAllocator.hpp"
#include "vulkan/VulkanBuffer.hpp"
#include "vulkan/VulkanDevice.hpp"

#include "voronoi/VoronoiAdapters.hpp"
#include "HeatModelRuntime.hpp"
#include "util/GeometryUtils.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <memory>
#include <unordered_map>

#include <libs/nanoflann/include/nanoflann.hpp>

namespace {

struct BakedContact {
    float totalConductance = 0.0f;
    std::unordered_map<uint32_t, float> neighborWeights;
};

bool buildPointStencil(
    const glm::vec3& point,
    const StencilKDTree& kdTree,
    std::vector<contact::ContactSampleWeight>& valueWeightsOut,
    uint32_t& valueWeightOffsetOut,
    uint32_t& valueWeightCountOut,
    std::vector<contact::ContactSampleWeight>& scatterWeightsOut,
    uint32_t& scatterWeightOffsetOut,
    uint32_t& scatterWeightCountOut) {
    valueWeightCountOut = 0;
    scatterWeightCountOut = 0;

    if (!kdTree.isValid()) {
        return false;
    }

    const size_t supportCount = kdTree.supportCount;
    std::vector<size_t> retIndices(supportCount, 0);
    std::vector<float> outDistSq(supportCount, 0.0f);

    const float query[3] = { point.x, point.y, point.z };
    nanoflann::KNNResultSet<float> resultSet(supportCount);
    resultSet.init(retIndices.data(), outDistSq.data());
    kdTree.index.findNeighbors(resultSet, query);

    std::vector<glm::dvec3> supportPositions;
    supportPositions.reserve(supportCount);
    float maxDistance = 0.0f;
    for (size_t supportIndex = 0; supportIndex < supportCount; ++supportIndex) {
        supportPositions.push_back(glm::dvec3(kdTree.regularSeedPositions[retIndices[supportIndex]]));
        maxDistance = std::max(maxDistance, std::sqrt(outDistSq[supportIndex]));
    }

    std::vector<double> valueWeights;
    std::vector<glm::dvec3> gradientWeights;
    const double kernelRadius = std::max<double>(static_cast<double>(maxDistance) * 2.0, 1e-5);
    if (!GMLS::computeWeights(glm::dvec3(point), supportPositions, kernelRadius, valueWeights, gradientWeights)) {
        return false;
    }

    const std::vector<float> scatterWeights = ::GMLS::buildScatterWeights(valueWeights);
    std::vector<contact::ContactSampleWeight> localValueWeights;
    std::vector<contact::ContactSampleWeight> localScatterWeights;
    localValueWeights.reserve(supportCount);
    localScatterWeights.reserve(supportCount);
    for (size_t supportIndex = 0; supportIndex < supportCount; ++supportIndex) {
        const uint32_t localCellIndex = kdTree.regularLocalIndices[retIndices[supportIndex]];
        const float valueWeight = static_cast<float>(valueWeights[supportIndex]);
        if (std::abs(valueWeight) > 1e-7f) {
            localValueWeights.push_back({ localCellIndex, valueWeight, 0u, 0u });
            ++valueWeightCountOut;
        }

        const float scatterWeight = scatterWeights[supportIndex];
        if (scatterWeight > 1e-7f) {
            localScatterWeights.push_back({ localCellIndex, scatterWeight, 0u, 0u });
            ++scatterWeightCountOut;
        }
    }

    valueWeightOffsetOut = static_cast<uint32_t>(valueWeightsOut.size());
    valueWeightsOut.insert(valueWeightsOut.end(), localValueWeights.begin(), localValueWeights.end());
    scatterWeightOffsetOut = static_cast<uint32_t>(scatterWeightsOut.size());
    scatterWeightsOut.insert(scatterWeightsOut.end(), localScatterWeights.begin(), localScatterWeights.end());

    return (valueWeightCountOut != 0 && scatterWeightCountOut != 0);
}

void thresholdContactEdges(
    const std::vector<BakedContact>& baked,
    uint32_t nodeCount,
    std::vector<contact::ContactSampleWeight>& outEdges,
    std::vector<contact::ContactIndex>& outIndex) {

    outEdges.clear();
    outIndex.clear();
    outIndex.resize(nodeCount);

    for (uint32_t nodeId = 0; nodeId < nodeCount; ++nodeId) {
        const auto& bakedContact = baked[nodeId];

        contact::ContactIndex index{};
        index.offset = static_cast<uint32_t>(outEdges.size());
        index.count = 0;
        index.contactK = bakedContact.totalConductance;
        index._pad = 0;

        if (!bakedContact.neighborWeights.empty()) {
            for (const auto& [neighborId, w] : bakedContact.neighborWeights) {
                if (std::abs(w) < 1e-7f) {
                    continue;
                }
                outEdges.push_back({ neighborId, w, 0u, 0u });
                ++index.count;
            }
        }

        outIndex[nodeId] = index;
    }
}

} // namespace

HeatContactRuntime::~HeatContactRuntime() = default;

bool HeatContactRuntime::build(
    VulkanDevice& vulkanDevice,
    MemoryAllocator& memoryAllocator,
    const ContactCoupling& coupling,
    const HeatModelRuntime& modelA,
    const HeatModelRuntime& modelB,
    float contactThermalConductance) {

    cleanup(memoryAllocator);

    if (coupling.contactPairs.empty() || coupling.contactPairCount == 0) {
        std::cerr << "[HeatContactRuntime] build abort: missing contact pairs" << std::endl;
        return false;
    }

    modelARuntimeModelId = coupling.modelARuntimeModelId;
    modelBRuntimeModelId = coupling.modelBRuntimeModelId;

    const uint32_t modelANodeCount = modelA.getSimNodeCount();
    const uint32_t modelBNodeCount = modelB.getSimNodeCount();

    const SupportingHalfedge::IntrinsicMesh& modelAMesh = modelA.getIntrinsicMesh();
    const SupportingHalfedge::IntrinsicMesh& modelBMesh = modelB.getIntrinsicMesh();

    StencilKDTree* modelAKDTree = modelA.getStencilKDTree();
    StencilKDTree* modelBKDTree = modelB.getStencilKDTree();

    if (!modelAKDTree || !modelBKDTree || !modelAKDTree->isValid() || !modelBKDTree->isValid()) {
        std::cerr << "[HeatContactRuntime] build abort: missing or invalid shared KD-Trees" << std::endl;
        return false;
    }

    const auto& triangleIndices = coupling.modelBTriangleIndices;
    const std::size_t triangleCount = triangleIndices.size() / 3;
    const std::size_t contactPairCount = std::min<std::size_t>(coupling.contactPairCount, triangleCount);

    std::vector<BakedContact> bakedAToB(modelANodeCount);
    std::vector<BakedContact> bakedBToA(modelBNodeCount);

    for (std::size_t triangleIndex = 0; triangleIndex < contactPairCount; ++triangleIndex) {
        const ContactPair& contactPair = coupling.contactPairs[triangleIndex];
        if (contactPair.contactArea <= 0.0f) {
            continue;
        }

        const std::size_t triangleBase = triangleIndex * 3;
        const uint32_t vertexIndices[3] = {
            triangleIndices[triangleBase + 0],
            triangleIndices[triangleBase + 1],
            triangleIndices[triangleBase + 2],
        };
        if (vertexIndices[0] >= modelBMesh.vertices.size() ||
            vertexIndices[1] >= modelBMesh.vertices.size() ||
            vertexIndices[2] >= modelBMesh.vertices.size()) {
            continue;
        }

        const glm::vec3 modelBP0 = modelBMesh.vertices[vertexIndices[0]].position;
        const glm::vec3 modelBP1 = modelBMesh.vertices[vertexIndices[1]].position;
        const glm::vec3 modelBP2 = modelBMesh.vertices[vertexIndices[2]].position;

        for (uint32_t sampleIndex = 0; sampleIndex < Quadrature::count; ++sampleIndex) {
            const contact::Sample& samplePoint = contactPair.samples[sampleIndex];
            if (samplePoint.sourceTriangleIndex == std::numeric_limits<uint32_t>::max() || samplePoint.wArea <= 0.0f) {
                continue;
            }

            const glm::vec3 modelBPoint = barycentricToPosition(
                modelBP0, modelBP1, modelBP2,
                Quadrature::bary[sampleIndex]);

            std::vector<contact::ContactSampleWeight> bValueWeights, bScatterWeights;
            uint32_t bValueOffset, bValueCount, bScatterOffset, bScatterCount;
            if (!buildPointStencil(
                    modelBPoint, *modelBKDTree,
                    bValueWeights, bValueOffset, bValueCount,
                    bScatterWeights, bScatterOffset, bScatterCount)) {
                continue;
            }

            if (samplePoint.sourceTriangleIndex >= modelAMesh.triangles.size()) {
                continue;
            }

            const auto& modelATriangle = modelAMesh.triangles[samplePoint.sourceTriangleIndex];
            const uint32_t e0 = modelATriangle.vertexIndices[0];
            const uint32_t e1 = modelATriangle.vertexIndices[1];
            const uint32_t e2 = modelATriangle.vertexIndices[2];
            if (e0 >= modelAMesh.vertices.size() ||
                e1 >= modelAMesh.vertices.size() ||
                e2 >= modelAMesh.vertices.size()) {
                continue;
            }

            const glm::vec3 modelABary(
                1.0f - samplePoint.u - samplePoint.v,
                samplePoint.u,
                samplePoint.v);
            const glm::vec3 modelAPoint = barycentricToPosition(
                modelAMesh.vertices[e0].position,
                modelAMesh.vertices[e1].position,
                modelAMesh.vertices[e2].position,
                modelABary);

            std::vector<contact::ContactSampleWeight> aValueWeights, aScatterWeights;
            uint32_t aValueOffset, aValueCount, aScatterOffset, aScatterCount;
            if (!buildPointStencil(
                    modelAPoint, *modelAKDTree,
                    aValueWeights, aValueOffset, aValueCount,
                    aScatterWeights, aScatterOffset, aScatterCount)) {
                continue;
            }

            const float sampleConductance = contactThermalConductance * samplePoint.wArea;

            float aValueWeightSum = 0.0f;
            for (const auto& aw : aValueWeights) {
                if (aw.cellIndex < modelANodeCount) {
                    aValueWeightSum += std::abs(aw.weight);
                }
            }
            if (aValueWeightSum < 1e-12f) continue;

            float bValueWeightSum = 0.0f;
            for (const auto& bw : bValueWeights) {
                if (bw.cellIndex < modelBNodeCount) {
                    bValueWeightSum += std::abs(bw.weight);
                }
            }
            if (bValueWeightSum < 1e-12f) continue;

            for (const auto& aw : aValueWeights) {
                if (aw.cellIndex >= modelANodeCount) continue;
                float nodeConductance = sampleConductance * std::abs(aw.weight) / aValueWeightSum;
                bakedAToB[aw.cellIndex].totalConductance += nodeConductance;

                for (const auto& bw : bValueWeights) {
                    if (bw.cellIndex >= modelBNodeCount) continue;
                    bakedAToB[aw.cellIndex].neighborWeights[bw.cellIndex] += nodeConductance * (bw.weight / bValueWeightSum);
                }
            }

            for (const auto& bw : bValueWeights) {
                if (bw.cellIndex >= modelBNodeCount) continue;
                float nodeConductance = sampleConductance * std::abs(bw.weight) / bValueWeightSum;
                bakedBToA[bw.cellIndex].totalConductance += nodeConductance;

                for (const auto& aw : aValueWeights) {
                    if (aw.cellIndex >= modelANodeCount) continue;
                    bakedBToA[bw.cellIndex].neighborWeights[aw.cellIndex] += nodeConductance * (aw.weight / aValueWeightSum);
                }
            }
        }
    }

    std::vector<contact::ContactSampleWeight> flatEdgesAToB;
    std::vector<contact::ContactIndex> indexDataAToB;
    thresholdContactEdges(bakedAToB, modelANodeCount, flatEdgesAToB, indexDataAToB);

    std::vector<contact::ContactSampleWeight> flatEdgesBToA;
    std::vector<contact::ContactIndex> indexDataBToA;
    thresholdContactEdges(bakedBToA, modelBNodeCount, flatEdgesBToA, indexDataBToA);

    void* mapped = nullptr;
    if (createStorageBuffer(
            memoryAllocator,
            vulkanDevice,
            indexDataAToB.data(),
            sizeof(contact::ContactIndex) * indexDataAToB.size(),
            edgeIndexAToB,
            edgeIndexAToBOffset,
            &mapped,
            true) != VK_SUCCESS) {
        return false;
    }

    if (!flatEdgesAToB.empty()) {
        if (createStorageBuffer(
                memoryAllocator,
                vulkanDevice,
                flatEdgesAToB.data(),
                sizeof(contact::ContactSampleWeight) * flatEdgesAToB.size(),
                edgesAToB,
                edgesAToBOffset,
                &mapped,
                true) != VK_SUCCESS) {
            return false;
        }
        edgeCountAToB = static_cast<uint32_t>(flatEdgesAToB.size());
    }

    if (createStorageBuffer(
            memoryAllocator,
            vulkanDevice,
            indexDataBToA.data(),
            sizeof(contact::ContactIndex) * indexDataBToA.size(),
            edgeIndexBToA,
            edgeIndexBToAOffset,
            &mapped,
            true) != VK_SUCCESS) {
        return false;
    }

    if (!flatEdgesBToA.empty()) {
        if (createStorageBuffer(
                memoryAllocator,
                vulkanDevice,
                flatEdgesBToA.data(),
                sizeof(contact::ContactSampleWeight) * flatEdgesBToA.size(),
                edgesBToA,
                edgesBToAOffset,
                &mapped,
                true) != VK_SUCCESS) {
            return false;
        }
        edgeCountBToA = static_cast<uint32_t>(flatEdgesBToA.size());
    }

    return true;
}

bool HeatContactRuntime::createDescriptorSets(const ContactSystemComputeStage& contactStage, const HeatModelRuntime& modelA, const HeatModelRuntime& modelB) {
    setAA = VK_NULL_HANDLE;
    setAB = VK_NULL_HANDLE;
    setBA = VK_NULL_HANDLE;
    setBB = VK_NULL_HANDLE;

    const uint32_t countA = modelA.getSimNodeCount();
    const uint32_t countB = modelB.getSimNodeCount();

    if (countA == 0 || countB == 0) {
        return true;
    }

    bool success = true;

    if (edgesAToB != VK_NULL_HANDLE && edgeIndexAToB != VK_NULL_HANDLE && edgeCountAToB > 0) {
        success = success && contactStage.createGatherDescriptorSet(
            modelB.getTempBufferA(), modelB.getTempBufferAOffset(), countB,
            modelA.getContactAccumulatorBuffer(), modelA.getContactAccumulatorBufferOffset(), countA,
            edgesAToB, edgesAToBOffset, edgeCountAToB,
            edgeIndexAToB, edgeIndexAToBOffset, countA,
            setAA);

        success = success && contactStage.createGatherDescriptorSet(
            modelB.getTempBufferB(), modelB.getTempBufferBOffset(), countB,
            modelA.getContactAccumulatorBuffer(), modelA.getContactAccumulatorBufferOffset(), countA,
            edgesAToB, edgesAToBOffset, edgeCountAToB,
            edgeIndexAToB, edgeIndexAToBOffset, countA,
            setAB);
    }

    if (edgesBToA != VK_NULL_HANDLE && edgeIndexBToA != VK_NULL_HANDLE && edgeCountBToA > 0) {
        success = success && contactStage.createGatherDescriptorSet(
            modelA.getTempBufferA(), modelA.getTempBufferAOffset(), countA,
            modelB.getContactAccumulatorBuffer(), modelB.getContactAccumulatorBufferOffset(), countB,
            edgesBToA, edgesBToAOffset, edgeCountBToA,
            edgeIndexBToA, edgeIndexBToAOffset, countB,
            setBA);

        success = success && contactStage.createGatherDescriptorSet(
            modelA.getTempBufferB(), modelA.getTempBufferBOffset(), countA,
            modelB.getContactAccumulatorBuffer(), modelB.getContactAccumulatorBufferOffset(), countB,
            edgesBToA, edgesBToAOffset, edgeCountBToA,
            edgeIndexBToA, edgeIndexBToAOffset, countB,
            setBB);
    }

    return success;
}

void HeatContactRuntime::cleanup(MemoryAllocator& memoryAllocator) {
    if (edgesAToB != VK_NULL_HANDLE) memoryAllocator.free(edgesAToB, edgesAToBOffset);
    if (edgeIndexAToB != VK_NULL_HANDLE) memoryAllocator.free(edgeIndexAToB, edgeIndexAToBOffset);
    if (edgesBToA != VK_NULL_HANDLE) memoryAllocator.free(edgesBToA, edgesBToAOffset);
    if (edgeIndexBToA != VK_NULL_HANDLE) memoryAllocator.free(edgeIndexBToA, edgeIndexBToAOffset);

    edgesAToB = VK_NULL_HANDLE;
    edgesAToBOffset = 0;
    edgeCountAToB = 0;
    edgeIndexAToB = VK_NULL_HANDLE;
    edgeIndexAToBOffset = 0;

    edgesBToA = VK_NULL_HANDLE;
    edgesBToAOffset = 0;
    edgeCountBToA = 0;
    edgeIndexBToA = VK_NULL_HANDLE;
    edgeIndexBToAOffset = 0;

    setAA = VK_NULL_HANDLE;
    setAB = VK_NULL_HANDLE;
    setBA = VK_NULL_HANDLE;
    setBB = VK_NULL_HANDLE;

    modelARuntimeModelId = 0;
    modelBRuntimeModelId = 0;
}
