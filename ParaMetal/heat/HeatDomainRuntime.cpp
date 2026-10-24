#include "HeatDomainRuntime.hpp"

#include "heat/HeatGlobalSolver.hpp"
#include "heat/HeatModelRuntime.hpp"
#include "heat/HeatFluidRuntime.hpp"
#include "cuda/CudaVulkanDevice.hpp"
#include "vulkan/VulkanDevice.hpp"
#include "../vem/VEMMeshBuilder.hpp"
#include "../vem/VEMDiffusion.hpp"
#include "../vem/VEMAgglomerator.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
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

void HeatDomainRuntime::setGlobalDomain(topology::GlobalDomain domain) {
    globalDomain = std::move(domain);
    vemMeshCache = vem::VEMMesh{};
    vemCellRemapCache = vem::VEMCellRemap{};
    globalTopologyDirtyFlag = true;
    globalValuesDirtyFlag = true;
    globalAdvectionDirtyFlag = true;
}

void HeatDomainRuntime::clearGlobalDomain() {
    waitGlobalSolveIdle();
    globalDomain = topology::GlobalDomain{};
    vemMeshCache = vem::VEMMesh{};
    vemCellRemapCache = vem::VEMCellRemap{};
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

    if (!globalDomain.isValid() || globalDomain.fragments.empty()) {
        return false;
    }

    const uint32_t cellCount = static_cast<uint32_t>(globalDomain.fragments.size());
    cellConductivity.assign(cellCount, 0.0f);
    cellCapacityOverDt.assign(cellCount, 0.0f);
    cellSourceAdditions.assign(cellCount, 0.0f);
    fixedCellTemperatures.assign(cellCount, std::numeric_limits<float>::quiet_NaN());

    for (uint32_t c = 0; c < cellCount; ++c) {
        const topology::Fragment& frag = globalDomain.fragments[c];
        float k = 0.0f;
        float rho = 0.0f;
        float cp = 0.0f;
        float qVol = 0.0f;
        bool isFixed = false;

        if (frag.instanceId != topology::InvalidInstanceId) {
            const auto it = models.find(frag.instanceId);
            if (it != models.end() && it->second) {
                k = it->second->getConductivity();
                rho = it->second->getDensity();
                cp = it->second->getSpecificHeat();
                qVol = it->second->getVolumetricPowerDensity();
                if (it->second->hasPrescribedTemperature()) {
                    isFixed = true;
                    fixedCellTemperatures[c] = it->second->getPrescribedTemperatureC();
                }
            }
        } else if (fluidRuntime) {
            k = fluidRuntime->getConductivity();
            rho = fluidRuntime->getDensity();
            cp = fluidRuntime->getSpecificHeat();
        }

        if (k <= 0.0f || !std::isfinite(k)) k = 1.0f;
        if (rho <= 0.0f || !std::isfinite(rho)) rho = 1000.0f;
        if (cp <= 0.0f || !std::isfinite(cp)) cp = 1000.0f;

        cellConductivity[c] = k;
        cellCapacityOverDt[c] = static_cast<float>((double(rho) * double(cp) * double(frag.volume)) / double(fixedTimeStep));
        if (!isFixed && qVol != 0.0f && std::isfinite(qVol)) {
            cellSourceAdditions[c] = qVol * frag.volume;
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

    if (!globalDomain.isValid() || globalDomain.fragments.empty()) {
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

    const vem::VEMCellInputs fragmentInputs{
        cellConductivity, cellCapacityOverDt, cellSourceAdditions, fixedCellTemperatures };
    vem::VEMAgglomerationResult agglomeration;
    if (!vem::VEMAgglomerator::build(
            vem::VEMMeshBuilder::buildFromGlobalDomain(globalDomain),
            fragmentInputs, agglomeration)) {
        std::cerr << "[HeatDomainRuntime] VEM cell agglomeration failed\n";
        return false;
    }
    vemMeshCache = std::move(agglomeration.mesh);
    vemCellRemapCache = std::move(agglomeration.remap);
    cellConductivity = std::move(agglomeration.inputs.conductivity);
    cellCapacityOverDt = std::move(agglomeration.inputs.capacityOverDt);
    cellSourceAdditions = std::move(agglomeration.inputs.sourceAdditions);
    fixedCellTemperatures = std::move(agglomeration.inputs.fixedTemperatures);
    for (auto& pair : vemMeshCache.traceCouplings) {
        pair.coefficientPerArea = HeatSimDefaults::contactConductance;
    }
    const uint32_t cellCount = static_cast<uint32_t>(vemMeshCache.cells.size());
    if (cellCount == 0 || vemMeshCache.totalTraceCount == 0 ||
        vemCellRemapCache.fragmentToCell.size() != globalDomain.fragments.size() ||
        vemCellRemapCache.cellFragmentOffsets.size() != size_t(cellCount) + 1 ||
        vemCellRemapCache.cellFragmentOffsets.back() != vemCellRemapCache.cellFragments.size()) {
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

    std::vector<uint32_t> localGlobalFragmentToFluidNode(globalDomain.fragments.size(), UINT32_MAX);
    std::vector<uint32_t> localFluidNodeToGlobalFragment;
    std::vector<uint32_t> fragmentToModelLocalNode(globalDomain.fragments.size(), UINT32_MAX);

    for (uint32_t runtimeModelId : modelIds) {
        uint32_t localNode = 0;
        for (uint32_t fragment = 0; fragment < globalDomain.fragments.size(); ++fragment) {
            if (globalDomain.fragments[fragment].instanceId == runtimeModelId) {
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
            const uint32_t memberBegin = vemCellRemapCache.cellFragmentOffsets[c];
            const uint32_t memberEnd = vemCellRemapCache.cellFragmentOffsets[c + 1];
            if (memberBegin >= memberEnd || memberEnd > vemCellRemapCache.cellFragments.size()) return false;
            double totalWeight = 0.0;
            for (uint32_t member = memberBegin; member < memberEnd; ++member) {
                const uint32_t fragment = vemCellRemapCache.cellFragments[member];
                if (fragment >= fragmentInputs.capacityOverDt.size() ||
                    fragmentToModelLocalNode[fragment] == UINT32_MAX) {
                    std::cerr << "[HeatDomainRuntime] invalid solid fragment mapping for VEM cell " << c << '\n';
                    return false;
                }
                totalWeight += std::max(0.0f, fragmentInputs.capacityOverDt[fragment]);
            }
            const bool useCapacityWeights = totalWeight > 0.0;
            if (!(totalWeight > 0.0)) {
                for (uint32_t member = memberBegin; member < memberEnd; ++member) {
                    totalWeight += globalDomain.fragments[vemCellRemapCache.cellFragments[member]].volume;
                }
            }
            if (!(totalWeight > 0.0)) return false;

            block.gatherSolverNodeIds.push_back(c);
            for (uint32_t member = memberBegin; member < memberEnd; ++member) {
                const uint32_t fragment = vemCellRemapCache.cellFragments[member];
                block.gatherLocalNodeIds.push_back(fragmentToModelLocalNode[fragment]);
                const double rawWeight = useCapacityWeights
                    ? std::max(0.0f, fragmentInputs.capacityOverDt[fragment])
                    : globalDomain.fragments[fragment].volume;
                block.gatherWeights.push_back(static_cast<float>(rawWeight / totalWeight));
                block.scatterLocalNodeIds.push_back(fragmentToModelLocalNode[fragment]);
                block.scatterSolverNodeIds.push_back(c);
            }
            block.gatherOffsets.push_back(static_cast<uint32_t>(block.gatherLocalNodeIds.size()));
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
        for (uint32_t fragment = 0; fragment < globalDomain.fragments.size(); ++fragment) {
            if (globalDomain.fragments[fragment].instanceId != topology::InvalidInstanceId) continue;
            localGlobalFragmentToFluidNode[fragment] = localFluidIdx;
            localFluidNodeToGlobalFragment.push_back(fragment);
            ++localFluidIdx;
        }
        for (uint32_t c = 0; c < cellCount; ++c) {
            if (vemMeshCache.cells[c].ownerModelId != topology::InvalidInstanceId) continue;
            const uint32_t memberBegin = vemCellRemapCache.cellFragmentOffsets[c];
            const uint32_t memberEnd = vemCellRemapCache.cellFragmentOffsets[c + 1];
            if (memberBegin >= memberEnd || memberEnd > vemCellRemapCache.cellFragments.size() ||
                memberEnd != memberBegin + 1 ||
                localGlobalFragmentToFluidNode[vemCellRemapCache.cellFragments[memberBegin]] == UINT32_MAX) {
                std::cerr << "[HeatDomainRuntime] fluid cells must remain one-to-one with RVD fragments\n";
                return false;
            }
            const uint32_t fragment = vemCellRemapCache.cellFragments[memberBegin];
            const uint32_t localNode = localGlobalFragmentToFluidNode[fragment];
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

        // Shared-face advection topology from global domain
        {
            const uint32_t fragmentCount = static_cast<uint32_t>(globalDomain.fragments.size());
            std::vector<heat::AdvectionFace> faces(globalDomain.faces.size());
            for (size_t i = 0; i < globalDomain.faces.size(); ++i) {
                const uint32_t fragmentA = globalDomain.faces[i].fragmentA;
                const uint32_t fragmentB = globalDomain.faces[i].fragmentB;
                faces[i].nodeA = fragmentA < fragmentCount
                    ? vemCellRemapCache.fragmentToCell[fragmentA] : InvalidSolverNode;
                faces[i].nodeB = fragmentB < fragmentCount
                    ? vemCellRemapCache.fragmentToCell[fragmentB] : InvalidSolverNode;
            }
            std::vector<heat::AdvectionBoundaryFace> boundaryFaces(globalDomain.boundaryFaces.size());
            for (size_t i = 0; i < globalDomain.boundaryFaces.size(); ++i) {
                const uint32_t fragment = globalDomain.boundaryFaces[i].fragment;
                boundaryFaces[i].node = fragment < fragmentCount
                    ? vemCellRemapCache.fragmentToCell[fragment] : InvalidSolverNode;
                boundaryFaces[i].domainFace = globalDomain.boundaryFaces[i].domainFace;
            }
            std::vector<float> cellVolumes(cellCount, 0.0f);
            for (uint32_t c = 0; c < cellCount; ++c) {
                cellVolumes[c] = static_cast<float>(vemMeshCache.cells[c].volume);
            }

            if (!globalSolver->configureAdvectionTopology(
                    static_cast<uint32_t>(faces.size()),
                    static_cast<uint32_t>(boundaryFaces.size()),
                    faces, boundaryFaces,
                    reinterpret_cast<const float*>(advectionFaceFluxCuda.getPointer()),
                    reinterpret_cast<const float*>(advectionBoundaryFluxCuda.getPointer()),
                    cellVolumes)) {
                globalSolver.reset();
                clearGlobalSynchronization();
                return false;
            }
            std::cerr << "[HeatDomainRuntime] advection configured: faces=" << faces.size()
                      << " boundaryFaces=" << boundaryFaces.size() << std::endl;
        }

        float inflowTemps[6];
        for (size_t i = 0; i < 6; ++i) {
            inflowTemps[i] = domainBoundaries[i].temperatureC;
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

    const vem::VEMCellInputs fragmentInputs{
        cellConductivity, cellCapacityOverDt, cellSourceAdditions, fixedCellTemperatures };
    vem::VEMCellInputs remappedInputs;
    if (!vem::VEMAgglomerator::remapInputs(
            vemCellRemapCache, fragmentInputs, remappedInputs)) {
        std::cerr << "[HeatDomainRuntime] failed to remap updated VEM cell inputs\n";
        return false;
    }
    cellConductivity = std::move(remappedInputs.conductivity);
    cellCapacityOverDt = std::move(remappedInputs.capacityOverDt);
    cellSourceAdditions = std::move(remappedInputs.sourceAdditions);
    fixedCellTemperatures = std::move(remappedInputs.fixedTemperatures);

    // Geometry is cached from the last topology build, only coefficients change here
    if (vemMeshCache.cells.empty()) {
        std::cerr << "[HeatDomainRuntime] updateGlobalValues: VEM mesh missing (topology rebuild required)\n";
        return false;
    }
    const uint32_t cellCount = static_cast<uint32_t>(vemMeshCache.cells.size());

    std::vector<vem::DirichletTrace> dirichletTraces = buildDirichletTraces(vemMeshCache, fixedCellTemperatures);

    vem::VEMTraceSystem traceSystem;
    if (!vem::VEMDiffusion::assembleTraceSystem(
            vemMeshCache.faceIncidences,
            vemMeshCache.cells,
            cellConductivity,
            cellCapacityOverDt,
            vemMeshCache.totalTraceCount,
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

bool HeatDomainRuntime::updateGlobalAdvection(HeatFluidRuntime* fluidRuntime, float fixedTimeStep) {
    if (!globalSolver) {
        return false;
    }

    if (fluidRuntime && fluidRuntime->getNodeCount() > 0 && advectionInput.enabled && advectionConnected) {
        waitGlobalSolveIdle();
        float inflowTemps[6];
        for (size_t i = 0; i < 6; ++i) {
            inflowTemps[i] = domainBoundaries[i].temperatureC;
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

    stagingCopyScheduled = advectionInput.enabled &&
        advectionConnected &&
        advectionFaceFlux.isValid() && advectionBoundaryFlux.isValid();

    // Destination overwrite only waits for its last real reader, staging copies wait on this frame submission
    const uint64_t destinationLastReader = lastReaderReleaseValue[sourceIsA ? 1u : 0u];
    pendingSolveReleaseValue = stagingCopyScheduled
        ? (vulkanReleaseCounter + 1)
        : (destinationLastReader != 0 ? destinationLastReader : (vulkanReleaseCounter + 1));

    if (!globalSolver->launch(sourceIsA, pendingSolveReleaseValue, cudaReady)) {
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
