#include "HeatDomainRuntime.hpp"

#include "heat/HeatGlobalSolver.hpp"
#include "heat/HeatModelRuntime.hpp"
#include "heat/HeatFluidRuntime.hpp"
#include "cuda/CudaVulkanDevice.hpp"
#include "vulkan/VulkanDevice.hpp"

#include <cuda_runtime.h>
#include "../vem/VEMMeshBuilder.hpp"
#include "../vem/VEMDiffusion.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>


HeatDomainRuntime::~HeatDomainRuntime() {
    waitGlobalSolveIdle();
}

HeatModelRuntime* HeatDomainRuntime::getModelByRuntimeId(uint32_t runtimeModelId) const {
    auto it = activeModels.find(runtimeModelId);
    return (it != activeModels.end()) ? it->second.get() : nullptr;
}

void HeatDomainRuntime::removeModelsNotIn(const std::unordered_set<uint32_t>& runtimeModelIds) {
    for (auto it = activeModels.begin(); it != activeModels.end();) {
        if (runtimeModelIds.find(it->first) == runtimeModelIds.end()) {
            it = activeModels.erase(it);
        } else {
            ++it;
        }
    }
}

void HeatDomainRuntime::setModel(
    uint32_t runtimeModelId,
    std::unique_ptr<HeatModelRuntime> model) {
    activeModels[runtimeModelId] = std::move(model);
}

void HeatDomainRuntime::removeModel(uint32_t runtimeModelId) {
    activeModels.erase(runtimeModelId);
}

bool HeatDomainRuntime::configureGlobalDomain(
    const std::vector<uint32_t>& fragmentInstanceIds,
    const std::vector<uint32_t>& fragmentSeedIds,
    const std::vector<glm::dvec3>& fragmentCentroids,
    const std::vector<double>& fragmentVolumes,
    const std::vector<uint32_t>& faceFragmentA,
    const std::vector<uint32_t>& faceFragmentB,
    const std::vector<double>& faceAreasA,
    const std::vector<double>& faceAreasB,
    const std::vector<glm::dvec3>& faceAreaVectorsA,
    const std::vector<glm::dvec3>& faceAreaVectorsB,
    const std::vector<glm::dvec3>& faceMomentsA,
    const std::vector<glm::dvec3>& faceMomentsB,
    const std::vector<uint8_t>& faceTypes,
    const std::vector<uint32_t>& faceDomainFaces,
    const std::vector<uint32_t>& boundaryFaceIndices,
    const std::vector<uint32_t>& contactFaceA,
    const std::vector<uint32_t>& contactFragmentA,
    const std::vector<uint32_t>& contactFaceB,
    const std::vector<uint32_t>& contactFragmentB,
    const std::vector<double>& contactAreas,
    const std::vector<uint32_t>& newSeedToSpaceIndex,
    const std::vector<uint32_t>& newSpaceToFragmentIndex,
    bool newSeedMappingUnique,
    VkBuffer newSeedIdGridBuffer,
    VkDeviceSize newSeedIdGridBufferOffset,
    const glm::ivec3& newVolumeGridDim) {
    waitGlobalSolveIdle();
    globalSolver.reset();

    const size_t fragmentCount = fragmentInstanceIds.size();
    const size_t faceCount = faceFragmentA.size();
    if (fragmentCount == 0 || faceCount == 0 ||
        fragmentSeedIds.size() != fragmentCount ||
        fragmentCentroids.size() != fragmentCount ||
        fragmentVolumes.size() != fragmentCount ||
        faceFragmentB.size() != faceCount ||
        faceAreasA.size() != faceCount ||
        faceAreasB.size() != faceCount ||
        faceAreaVectorsA.size() != faceCount ||
        faceAreaVectorsB.size() != faceCount ||
        faceMomentsA.size() != faceCount ||
        faceMomentsB.size() != faceCount ||
        faceTypes.size() != faceCount ||
        faceDomainFaces.size() != faceCount ||
        contactFaceA.size() != contactFaceB.size() ||
        contactFaceA.size() != contactFragmentA.size() ||
        contactFaceA.size() != contactFragmentB.size() ||
        contactAreas.size() != contactFaceA.size()) {
        return false;
    }
    for (size_t coupling = 0; coupling < contactFaceA.size(); ++coupling) {
        if (!std::isfinite(contactAreas[coupling]) || !(contactAreas[coupling] > 0.0)) {
            return false;
        }
    }
    for (size_t face = 0; face < faceCount; ++face) {
        if ((faceFragmentA[face] != UINT32_MAX && faceFragmentA[face] >= fragmentCount) ||
            (faceFragmentB[face] != UINT32_MAX && faceFragmentB[face] >= fragmentCount)) {
            return false;
        }
    }
    size_t boundarySlot = 0;
    for (size_t face = 0; face < faceCount; ++face) {
        if (faceTypes[face] != 1u) continue;
        if (boundarySlot >= boundaryFaceIndices.size() ||
            boundaryFaceIndices[boundarySlot] != face) {
            return false;
        }
        ++boundarySlot;
    }
    if (boundarySlot != boundaryFaceIndices.size()) {
        return false;
    }

    vemMeshCache = vem::VEMMeshBuilder::buildFromArrays(
        fragmentInstanceIds,
        fragmentSeedIds,
        fragmentCentroids,
        fragmentVolumes,
        faceFragmentA,
        faceFragmentB,
        faceAreasA,
        faceAreasB,
        faceAreaVectorsA,
        faceAreaVectorsB,
        faceMomentsA,
        faceMomentsB,
        faceTypes,
        faceDomainFaces,
        contactFaceA,
        contactFragmentA,
        contactFaceB,
        contactFragmentB,
        contactAreas);
    if (vemMeshCache.cells.empty()) {
        return false;
    }

    const uint32_t fragmentCount32 = static_cast<uint32_t>(fragmentCount);
    advectionFaces.resize(faceFragmentA.size());
    for (size_t i = 0; i < faceFragmentA.size(); ++i) {
        advectionFaces[i].nodeA = faceFragmentA[i] < fragmentCount32
            ? faceFragmentA[i]
            : InvalidSolverNode;
        advectionFaces[i].nodeB = faceFragmentB[i] < fragmentCount32
            ? faceFragmentB[i]
            : InvalidSolverNode;
    }
    advectionBoundaryFaces.resize(boundaryFaceIndices.size());
    faceFragmentPairs.clear();
    faceFragmentPairs.reserve(faceFragmentA.size());
    for (size_t i = 0; i < faceFragmentA.size(); ++i) {
        if (faceFragmentA[i] < fragmentCount && faceFragmentB[i] < fragmentCount) {
            faceFragmentPairs.emplace_back(faceFragmentA[i], faceFragmentB[i]);
        }
    }
    for (size_t i = 0; i < boundaryFaceIndices.size(); ++i) {
        const uint32_t faceIndex = boundaryFaceIndices[i];
        advectionBoundaryFaces[i].node = faceIndex < faceFragmentA.size() &&
            faceFragmentA[faceIndex] < fragmentCount
            ? faceFragmentA[faceIndex]
            : InvalidSolverNode;
        advectionBoundaryFaces[i].domainFace = faceIndex < faceDomainFaces.size()
            ? faceDomainFaces[faceIndex]
            : UINT32_MAX;
    }

    seedIdGridBuffer = newSeedIdGridBuffer;
    seedIdGridBufferOffset = newSeedIdGridBufferOffset;
    seedToSpaceIndex = newSeedToSpaceIndex;
    spaceToFragmentIndex = newSpaceToFragmentIndex;
    seedMappingUnique = newSeedMappingUnique;
    volumeGridDim = newVolumeGridDim;
    hasGlobalDomainFlag = true;
    globalTopologyDirtyFlag = true;
    globalValuesDirtyFlag = true;
    globalAdvectionDirtyFlag = true;
    return true;
}

void HeatDomainRuntime::clearGlobalDomain() {
    waitGlobalSolveIdle();
    hasGlobalDomainFlag = false;
    seedIdGridBuffer = VK_NULL_HANDLE;
    seedIdGridBufferOffset = 0;
    seedToSpaceIndex.clear();
    spaceToFragmentIndex.clear();
    seedMappingUnique = false;
    volumeGridDim = glm::ivec3(0);
    advectionFaces.clear();
    advectionBoundaryFaces.clear();
    faceFragmentPairs.clear();
    vemMeshCache = vem::VEMMesh{};
    globalSolver.reset();
    globalTopologyDirtyFlag = false;
    globalValuesDirtyFlag = false;
    globalAdvectionDirtyFlag = false;
    fluidNodeToGlobalFragment.clear();
    globalFragmentToFluidNode.clear();
    clearGlobalSynchronization();
    globalSolveInFlight = false;
    pendingDestinationIsA = false;
    cudaReadyCounter = 0;
    vulkanReleaseCounter = 0;
    pendingCudaReadyValue = 0;
    completedCudaReadyValue = 0;
    pendingSolveReleaseValue = 0;
    lastReaderReleaseValue[0] = 0;
    lastReaderReleaseValue[1] = 0;
    stagingCopyScheduled = false;
    advectionConnected = false;
    advectionFaceFluxCuda.cleanup();
    advectionBoundaryFluxCuda.cleanup();
    advectionFaceFlux.cleanup();
    advectionBoundaryFlux.cleanup();
    vulkanDeviceHandle = VK_NULL_HANDLE;
}

void HeatDomainRuntime::cleanup() {
    clearGlobalDomain();
    activeModels.clear();
}

std::vector<vem::DirichletTrace> HeatDomainRuntime::buildDirichletTraces(
    const vem::VEMMesh& vemMesh,
    const std::vector<float>& fixedCellTemperatures) const {
    std::unordered_map<uint32_t, float> traceToFixedTemp;
    const uint32_t cellCount = static_cast<uint32_t>(vemMesh.cells.size());
    for (uint32_t c = 0; c < cellCount; ++c) {
        if (std::isnan(fixedCellTemperatures[c])) continue;
        const float T_fix = fixedCellTemperatures[c];
        const vem::CellGeometry& cell = vemMesh.cells[c];
        for (uint32_t i = 0; i < cell.faceCount; ++i) {
            const uint32_t traceIndex = vemMesh.faceIncidences[cell.faceOffset + i].traceIndex;
            auto it = traceToFixedTemp.find(traceIndex);
            if (it == traceToFixedTemp.end()) {
                traceToFixedTemp[traceIndex] = T_fix;
            } else if (std::abs(it->second - T_fix) > 1e-3f) {
                std::cerr << "[HeatDomainRuntime] Warning: Dirichlet trace " << traceIndex
                          << " has conflicting temperatures between adjoining bodies: "
                          << it->second << " C vs " << T_fix << " C (retaining " << it->second << " C).\n";
            }
        }
    }
    std::vector<vem::DirichletTrace> dirichletTraces;
    dirichletTraces.reserve(traceToFixedTemp.size());
    for (const auto& [traceIdx, temp] : traceToFixedTemp) {
        dirichletTraces.push_back({ traceIdx, temp });
    }
    return dirichletTraces;
}

bool HeatDomainRuntime::extractVEMInputs(
    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& models,
    HeatFluidRuntime* fluidRuntime,
    float fixedTimeStep,
    std::vector<float>& cellConductivity,
    std::vector<float>& cellCapacityOverDt,
    std::vector<float>& cellSourceAdditions,
    std::vector<float>& fixedCellTemperatures,
    std::array<vem::BoundaryCondition, 6>& vemBcs) const {

    if (!hasGlobalDomainFlag || vemMeshCache.cells.empty()) {
        return false;
    }

    const uint32_t cellCount = static_cast<uint32_t>(vemMeshCache.cells.size());
    cellConductivity.assign(cellCount, 0.0f);
    cellCapacityOverDt.assign(cellCount, 0.0f);
    cellSourceAdditions.assign(cellCount, 0.0f);
    fixedCellTemperatures.assign(cellCount, std::numeric_limits<float>::quiet_NaN());

    for (uint32_t c = 0; c < cellCount; ++c) {
        const vem::CellGeometry& cell = vemMeshCache.cells[c];
        float k = 0.0f;
        float rho = 0.0f;
        float cp = 0.0f;
        float qVol = 0.0f;
        bool isFixed = false;
        bool hasMaterial = false;

        if (cell.ownerModelId != UINT32_MAX) {
            const auto it = models.find(cell.ownerModelId);
            if (it == models.end() || !it->second) {
                std::cerr << "[HeatDomainRuntime] cell " << c
                          << " references missing model " << cell.ownerModelId << std::endl;
                return false;
            }
            k = it->second->getConductivity();
            rho = it->second->getDensity();
            cp = it->second->getSpecificHeat();
            qVol = it->second->getVolumetricPowerDensity();
            if (it->second->hasPrescribedTemperature()) {
                isFixed = true;
                fixedCellTemperatures[c] = it->second->getPrescribedTemperatureC();
            }
            hasMaterial = true;
        } else if (fluidRuntime) {
            k = fluidRuntime->getConductivity();
            rho = fluidRuntime->getDensity();
            cp = fluidRuntime->getSpecificHeat();
            hasMaterial = true;
        }

        if (!hasMaterial ||
            !std::isfinite(k) || !(k > 0.0f) ||
            !std::isfinite(rho) || !(rho > 0.0f) ||
            !std::isfinite(cp) || !(cp > 0.0f)) {
            std::cerr << "[HeatDomainRuntime] cell " << c
                      << " has invalid material properties (k=" << k
                      << ", rho=" << rho << ", cp=" << cp << ')' << std::endl;
            return false;
        }

        cellConductivity[c] = k;
        cellCapacityOverDt[c] = static_cast<float>((double(rho) * double(cp) * double(cell.volume)) / double(fixedTimeStep));
        if (!isFixed && qVol != 0.0f && std::isfinite(qVol)) {
            cellSourceAdditions[c] = static_cast<float>(double(qVol) * cell.volume);
        }
    }

    for (size_t i = 0; i < 6; ++i) {
        const auto& bc = domainBoundaries[i];
        switch (bc.type) {
        case heat::BoundaryType::Adiabatic:
            vemBcs[i].type = vem::BoundaryType::Neumann;
            vemBcs[i].value = 0.0f;
            break;
        case heat::BoundaryType::Dirichlet:
            vemBcs[i].type = vem::BoundaryType::Dirichlet;
            vemBcs[i].value = bc.temperatureC;
            break;
        case heat::BoundaryType::Neumann:
            vemBcs[i].type = vem::BoundaryType::Neumann;
            vemBcs[i].value = bc.heatFlux;
            break;
        case heat::BoundaryType::Robin:
            vemBcs[i].type = vem::BoundaryType::Robin;
            vemBcs[i].robinCoefficient = bc.htc;
            vemBcs[i].referenceValue = bc.temperatureC;
            break;
        }
    }

    return true;
}

bool HeatDomainRuntime::rebuildGlobalSolver(
    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& models,
    HeatFluidRuntime* fluidRuntime,
    VulkanDevice& vulkanDevice,
    float fixedTimeStep) {
    waitGlobalSolveIdle();
    globalSolver.reset();

    if (!hasGlobalDomainFlag || vemMeshCache.cells.empty()) {
        return false;
    }

    std::vector<float> cellConductivity;
    std::vector<float> cellCapacityOverDt;
    std::vector<float> cellSourceAdditions;
    std::vector<float> fixedCellTemperatures;
    std::array<vem::BoundaryCondition, 6> vemBcs;
    if (!extractVEMInputs(models, fluidRuntime, fixedTimeStep,
                          cellConductivity, cellCapacityOverDt, cellSourceAdditions, fixedCellTemperatures, vemBcs)) {
        return false;
    }

    std::string meshValidationReason;
    if (!vem::validateMesh(vemMeshCache, cellConductivity, &meshValidationReason)) {
        std::cerr << "[HeatDomainRuntime] VEM mesh validation failed: "
                  << meshValidationReason << std::endl;
        return false;
    }
    for (auto& pair : vemMeshCache.traceCouplings) {
        pair.coefficientPerArea = HeatSimDefaults::contactConductance;
    }
    const uint32_t cellCount = static_cast<uint32_t>(vemMeshCache.cells.size());
    if (cellCount == 0 || vemMeshCache.totalTraceCount == 0 ||
        cellCapacityOverDt.size() != cellCount) {
        return false;
    }

    std::vector<vem::DirichletTrace> dirichletTraces = buildDirichletTraces(vemMeshCache, fixedCellTemperatures);

    vem::VEMTraceSystem traceSystem;
    if (!vem::VEMDiffusion::assembleTraceSystem(
            vemMeshCache.faceIncidences,
            vemMeshCache.cells,
            cellConductivity,
            cellCapacityOverDt,
            vemMeshCache.totalTraceCount,
            vemMeshCache.perfectContinuityPairs,
            vemBcs.data(),
            static_cast<uint32_t>(vemBcs.size()),
            vemMeshCache.traceCouplings,
            dirichletTraces,
            traceSystem)) {
        std::cerr << "[HeatDomainRuntime] VEMDiffusion::assembleTraceSystem FAILED\n";
        return false;
    }

    std::vector<uint32_t> modelIds;
    modelIds.reserve(models.size());
    for (const auto& [runtimeModelId, model] : models) {
        if (model) modelIds.push_back(runtimeModelId);
    }
    std::sort(modelIds.begin(), modelIds.end());

    std::vector<HeatGlobalSolver::TemperatureNodes> solverTemperatureNodes;

    std::vector<uint32_t> localGlobalFragmentToFluidNode(cellCount, UINT32_MAX);
    std::vector<uint32_t> localFluidNodeToGlobalFragment;
    std::vector<uint32_t> fragmentToModelLocalNode(cellCount, UINT32_MAX);

    for (uint32_t runtimeModelId : modelIds) {
        uint32_t localNode = 0;
        for (uint32_t fragment = 0; fragment < cellCount; ++fragment) {
            if (vemMeshCache.cells[fragment].ownerModelId == runtimeModelId) {
                fragmentToModelLocalNode[fragment] = localNode++;
            }
        }
    }

    for (uint32_t runtimeModelId : modelIds) {
        const auto it = models.find(runtimeModelId);
        if (it == models.end() || !it->second) continue;

        HeatModelRuntime& model = *it->second;
        HeatGlobalSolver::TemperatureNodes block{};
        block.externalA = &model.getExternalTempBufferA();
        block.externalB = &model.getExternalTempBufferB();
        block.temperatureA = &model.getCudaTempBufferA();
        block.temperatureB = &model.getCudaTempBufferB();

        block.gatherOffsets.push_back(0);
        for (uint32_t c = 0; c < cellCount; ++c) {
            if (vemMeshCache.cells[c].ownerModelId != runtimeModelId) continue;
            if (fragmentToModelLocalNode[c] == UINT32_MAX) {
                std::cerr << "[HeatDomainRuntime] invalid solid fragment mapping for VEM cell " << c << '\n';
                return false;
            }
            const uint32_t localNode = fragmentToModelLocalNode[c];
            block.gatherSolverNodeIds.push_back(c);
            block.gatherLocalNodeIds.push_back(localNode);
            block.gatherWeights.push_back(1.0f);
            block.gatherOffsets.push_back(static_cast<uint32_t>(block.gatherLocalNodeIds.size()));
            block.scatterLocalNodeIds.push_back(localNode);
            block.scatterSolverNodeIds.push_back(c);
        }
        if (!block.gatherSolverNodeIds.empty()) {
            solverTemperatureNodes.push_back(std::move(block));
        }
    }

    if (fluidRuntime && fluidRuntime->getNodeCount() > 0) {
        HeatGlobalSolver::TemperatureNodes fluidBlock{};
        fluidBlock.externalA = &fluidRuntime->getExternalTempBufferA();
        fluidBlock.externalB = &fluidRuntime->getExternalTempBufferB();
        fluidBlock.temperatureA = &fluidRuntime->getCudaTempBufferA();
        fluidBlock.temperatureB = &fluidRuntime->getCudaTempBufferB();

        fluidBlock.gatherOffsets.push_back(0);
        uint32_t localFluidIdx = 0;
        for (uint32_t fragment = 0; fragment < cellCount; ++fragment) {
            if (vemMeshCache.cells[fragment].ownerModelId != UINT32_MAX) continue;
            localGlobalFragmentToFluidNode[fragment] = localFluidIdx;
            localFluidNodeToGlobalFragment.push_back(fragment);
            ++localFluidIdx;
        }
        for (uint32_t c = 0; c < cellCount; ++c) {
            if (vemMeshCache.cells[c].ownerModelId != UINT32_MAX) continue;
            if (localGlobalFragmentToFluidNode[c] == UINT32_MAX) {
                std::cerr << "[HeatDomainRuntime] fluid cells must remain one-to-one with RVD fragments\n";
                return false;
            }
            const uint32_t localNode = localGlobalFragmentToFluidNode[c];
            fluidBlock.gatherSolverNodeIds.push_back(c);
            fluidBlock.gatherLocalNodeIds.push_back(localNode);
            fluidBlock.gatherWeights.push_back(1.0f);
            fluidBlock.gatherOffsets.push_back(static_cast<uint32_t>(fluidBlock.gatherLocalNodeIds.size()));
            fluidBlock.scatterLocalNodeIds.push_back(localNode);
            fluidBlock.scatterSolverNodeIds.push_back(c);
        }
        if (!fluidBlock.gatherSolverNodeIds.empty()) {
            solverTemperatureNodes.push_back(std::move(fluidBlock));
        }
    }

    fluidNodeToGlobalFragment = std::move(localFluidNodeToGlobalFragment);
    globalFragmentToFluidNode = std::move(localGlobalFragmentToFluidNode);

    if (solverTemperatureNodes.empty()) {
    cudaReadyCounter = 0;
    vulkanReleaseCounter = 0;
    pendingCudaReadyValue = 0;
    completedCudaReadyValue = 0;
    pendingSolveReleaseValue = 0;
    lastReaderReleaseValue[0] = 0;
    lastReaderReleaseValue[1] = 0;
        globalSolveInFlight = false;
        globalTopologyDirtyFlag = false;
        globalValuesDirtyFlag = false;
        clearGlobalSynchronization();
        return true;
    }

    globalSolver = std::make_unique<HeatGlobalSolver>();
    globalSolver->setSolverParameters(solverMaxIterations, solverConvergenceTolerance);
    if (!globalSolver->initialize(
            vulkanDevice, traceSystem, cellCapacityOverDt, cellSourceAdditions,
            solverTemperatureNodes, fixedCellTemperatures)) {
        globalSolver.reset();
        clearGlobalSynchronization();
        return false;
    }

    vulkanDeviceHandle = vulkanDevice.getDevice();
    const int cudaDevice = cudaVulkan::findDevice(vulkanDevice.getPhysicalDevice());

    if (fluidRuntime && fluidRuntime->getNodeCount() > 0 && advectionInput.enabled) {
        if (advectionInput.faceFluxBufferSize > 0) {
            if (!advectionFaceFlux.isValid() || advectionFaceFlux.getSize() < advectionInput.faceFluxBufferSize) {
                advectionFaceFlux.cleanup();
                if (!advectionFaceFlux.initialize(vulkanDevice, advectionInput.faceFluxBufferSize,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
                    globalSolver.reset();
                    clearGlobalSynchronization();
                    return false;
                }
            }
            if (!advectionFaceFluxCuda.ensureMapped(advectionFaceFlux, cudaDevice)) {
                globalSolver.reset();
                clearGlobalSynchronization();
                return false;
            }
        }
        if (advectionInput.boundaryFluxBufferSize > 0) {
            if (!advectionBoundaryFlux.isValid() || advectionBoundaryFlux.getSize() < advectionInput.boundaryFluxBufferSize) {
                advectionBoundaryFlux.cleanup();
                if (!advectionBoundaryFlux.initialize(vulkanDevice, advectionInput.boundaryFluxBufferSize,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
                    globalSolver.reset();
                    clearGlobalSynchronization();
                    return false;
                }
            }
            if (!advectionBoundaryFluxCuda.ensureMapped(advectionBoundaryFlux, cudaDevice)) {
                globalSolver.reset();
                clearGlobalSynchronization();
                return false;
            }
        }

        {
            std::vector<float> cellVolumes(cellCount, 0.0f);
            for (uint32_t c = 0; c < cellCount; ++c) {
                cellVolumes[c] = static_cast<float>(vemMeshCache.cells[c].volume);
            }

            if (!globalSolver->configureAdvectionTopology(
                    static_cast<uint32_t>(advectionFaces.size()),
                    static_cast<uint32_t>(advectionBoundaryFaces.size()),
                    advectionFaces, advectionBoundaryFaces,
                    reinterpret_cast<const float*>(advectionFaceFluxCuda.getPointer()),
                    reinterpret_cast<const float*>(advectionBoundaryFluxCuda.getPointer()),
                    cellVolumes)) {
                globalSolver.reset();
                clearGlobalSynchronization();
                return false;
            }
        }

        float inflowTemps[6];
        for (size_t i = 0; i < 6; ++i) {
            inflowTemps[i] = domainBoundaries[i].inflowTemperatureC;
        }
        if (!globalSolver->updateAdvectionStep(
            inflowTemps,
            fixedTimeStep)) {
            return false;
        }
    } else {
        advectionFaceFluxCuda.cleanup();
        advectionBoundaryFluxCuda.cleanup();
        advectionFaceFlux.cleanup();
        advectionBoundaryFlux.cleanup();
    }

    cudaReadyCounter = 0;
    vulkanReleaseCounter = 0;
    pendingCudaReadyValue = 0;
    completedCudaReadyValue = 0;
    pendingSolveReleaseValue = 0;
    lastReaderReleaseValue[0] = 0;
    lastReaderReleaseValue[1] = 0;
    globalSolveInFlight = false;
    globalTopologyDirtyFlag = false;
    globalValuesDirtyFlag = false;
    globalAdvectionDirtyFlag = false;
    clearGlobalSynchronization();
    return true;
}

bool HeatDomainRuntime::updateGlobalValues(
    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& models,
    HeatFluidRuntime* fluidRuntime,
    float fixedTimeStep) {
    if (!globalSolver) {
        return false;
    }
    waitGlobalSolveIdle();

    std::vector<float> cellConductivity;
    std::vector<float> cellCapacityOverDt;
    std::vector<float> cellSourceAdditions;
    std::vector<float> fixedCellTemperatures;
    std::array<vem::BoundaryCondition, 6> vemBcs;
    if (!extractVEMInputs(models, fluidRuntime, fixedTimeStep,
                          cellConductivity, cellCapacityOverDt, cellSourceAdditions, fixedCellTemperatures, vemBcs)) {
        return false;
    }

    // Geometry is cached from the last topology build, only coefficients change here
    if (vemMeshCache.cells.empty()) {
        std::cerr << "[HeatDomainRuntime] updateGlobalValues: VEM mesh missing (topology rebuild required)\n";
        return false;
    }
    std::vector<vem::DirichletTrace> dirichletTraces = buildDirichletTraces(vemMeshCache, fixedCellTemperatures);

    vem::VEMTraceSystem traceSystem;
    if (!vem::VEMDiffusion::assembleTraceSystem(
            vemMeshCache.faceIncidences,
            vemMeshCache.cells,
            cellConductivity,
            cellCapacityOverDt,
            vemMeshCache.totalTraceCount,
            vemMeshCache.perfectContinuityPairs,
            vemBcs.data(),
            static_cast<uint32_t>(vemBcs.size()),
            vemMeshCache.traceCouplings,
            dirichletTraces,
            traceSystem)) {
        return false;
    }

    if (!globalSolver->updateValues(traceSystem, cellCapacityOverDt, cellSourceAdditions, fixedCellTemperatures)) {
        return false;
    }

    globalValuesDirtyFlag = false;
    return true;
}

bool HeatDomainRuntime::advectionActive() const {
    const bool faceSourceReady = advectionInput.faceFluxBuffer != VK_NULL_HANDLE &&
        advectionInput.faceFluxBufferSize >=
            static_cast<VkDeviceSize>(advectionInput.faceCount) * sizeof(float);
    const bool boundarySourceReady = advectionInput.boundaryFaceCount == 0 ||
        (advectionInput.boundaryFluxBuffer != VK_NULL_HANDLE &&
         advectionInput.boundaryFluxBufferSize >=
             static_cast<VkDeviceSize>(advectionInput.boundaryFaceCount) * sizeof(float) &&
         advectionBoundaryFlux.isValid());

    return advectionInput.enabled && advectionConnected &&
        advectionInput.faceCount > 0 &&
        faceSourceReady && advectionFaceFlux.isValid() &&
        boundarySourceReady;
}

bool HeatDomainRuntime::updateGlobalAdvection(float fixedTimeStep) {
    if (!globalSolver) {
        return false;
    }
    if (globalSolveInFlight) {
        return true;
    }

    if (advectionActive()) {
        float inflowTemps[6];
        for (size_t i = 0; i < 6; ++i) {
            inflowTemps[i] = domainBoundaries[i].inflowTemperatureC;
        }
        if (!globalSolver->updateAdvectionStep(
            inflowTemps,
            fixedTimeStep)) {
            return false;
        }
    }

    globalAdvectionDirtyFlag = false;
    return true;
}

bool HeatDomainRuntime::launchGlobalSolve(bool sourceIsA) {
    if (!globalSolver || globalSolveInFlight) {
        return false;
    }

    const uint64_t cudaReady = ++cudaReadyCounter;

    stagingCopyScheduled = advectionActive();

    // Destination overwrite only waits for its last real reader, staging copies wait on this frame submission
    const uint64_t destinationLastReader = lastReaderReleaseValue[sourceIsA ? 1u : 0u];
    pendingSolveReleaseValue = stagingCopyScheduled
        ? (vulkanReleaseCounter + 1)
        : (destinationLastReader != 0 ? destinationLastReader : (vulkanReleaseCounter + 1));

    if (!globalSolver->launch(sourceIsA, pendingSolveReleaseValue, cudaReady, stagingCopyScheduled)) {
        return false;
    }

    globalSolveInFlight = true;
    pendingDestinationIsA = !sourceIsA;
    pendingCudaReadyValue = cudaReady;
    return true;
}

bool HeatDomainRuntime::pollGlobalSolve(bool& completedDestinationIsA) {
    if (!globalSolveInFlight || !globalSolver) {
        return false;
    }
    const auto result = globalSolver->poll();
    if (result == HeatGlobalSolver::SolvePollResult::Complete) {
        globalSolveInFlight = false;
        completedDestinationIsA = pendingDestinationIsA;
        // Capture completion value for render wait
        completedCudaReadyValue = pendingCudaReadyValue;
        return true;
    }
    if (result == HeatGlobalSolver::SolvePollResult::Failed) {
        globalSolveInFlight = false;
        return false;
    }
    return false;
}

void HeatDomainRuntime::prepareGlobalRenderSynchronization(bool newSolveCompleted) {
    clearGlobalSynchronization();
    if (!globalSolver) {
        return;
    }

    const uint64_t vulkanRelease = ++vulkanReleaseCounter;

    if (newSolveCompleted) {
        globalSynchronization.addWait(globalSolver->getCudaReadySemaphore(), completedCudaReadyValue, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    }
    globalSynchronization.addSignal(globalSolver->getVulkanReleaseSemaphore(), vulkanRelease);
}

void HeatDomainRuntime::noteTemperatureReader(bool currentIsA) {
    if (!globalSolver) {
        return;
    }
    // Release value already assigned by prepareGlobalRenderSynchronization
    lastReaderReleaseValue[currentIsA ? 0u : 1u] = vulkanReleaseCounter;
}

void HeatDomainRuntime::waitGlobalSolveIdle() {
    if (globalSolver) {
        globalSolver->waitIdle();
    }
    globalSolveInFlight = false;
}

void HeatDomainRuntime::releasePendingSolveWaitOnHost() {
    if (!globalSolver || pendingSolveReleaseValue == 0 || vulkanDeviceHandle == VK_NULL_HANDLE) {
        return;
    }

    const VkSemaphore semaphore = globalSolver->getVulkanReleaseSemaphore();
    uint64_t current = 0;
    if (semaphore != VK_NULL_HANDLE &&
        vkGetSemaphoreCounterValue(vulkanDeviceHandle, semaphore, &current) == VK_SUCCESS &&
        current < pendingSolveReleaseValue) {
        VkSemaphoreSignalInfo signalInfo{};
        signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
        signalInfo.semaphore = semaphore;
        signalInfo.value = pendingSolveReleaseValue;
        vkSignalSemaphore(vulkanDeviceHandle, &signalInfo);
    }
    pendingSolveReleaseValue = 0;
}

