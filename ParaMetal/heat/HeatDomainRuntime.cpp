#include "HeatDomainRuntime.hpp"

#include "heat/HeatGlobalSolver.hpp"
#include "heat/HeatModelRuntime.hpp"
#include "heat/HeatFluidRuntime.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>


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
    globalTopologyDirtyFlag = true;
    globalValuesDirtyFlag = true;
    globalAdvectionDirtyFlag = true;
}

void HeatDomainRuntime::clearGlobalDomain() {
    waitGlobalSolveIdle();
    globalDomain = topology::GlobalDomain{};
    globalSolver.reset();
    globalTopologyDirtyFlag = false;
    globalValuesDirtyFlag = false;
    globalAdvectionDirtyFlag = false;
    fragmentSolverNodes.clear();
    fluidNodeToGlobalFragment.clear();
    globalFragmentToFluidNode.clear();
    clearGlobalSynchronization();
    globalSolveInFlight = false;
    pendingDestinationIsA = false;
    cudaReadyCounter = 0;
    vulkanReleaseCounter = 0;
    pendingCudaReadyValue = 0;
    lastVulkanReleaseValue[0] = 0;
    lastVulkanReleaseValue[1] = 0;
}

void HeatDomainRuntime::cleanup() {
    clearGlobalDomain();
    activeModels.clear();
}

bool HeatDomainRuntime::assembleGlobalMatrix(
    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& models,
    HeatFluidRuntime* fluidRuntime,
    float fixedTimeStep,
    std::vector<int>& rowOffsets,
    std::vector<int>& columnIndices,
    std::vector<float>& values,
    std::vector<float>& thermalMasses,
    std::vector<HeatGlobalSolver::TemperatureNodes>& temperatureNodes,
    std::vector<float>& rhsAdditions) {
    rowOffsets.clear();
    columnIndices.clear();
    values.clear();
    thermalMasses.clear();
    temperatureNodes.clear();
    rhsAdditions.clear();

    if (!globalDomain.isValid() || globalDomain.fragments.empty()) {
        return false;
    }

    std::vector<uint32_t> modelIds;
    modelIds.reserve(models.size());
    for (const auto& [runtimeModelId, model] : models) {
        if (model) modelIds.push_back(runtimeModelId);
    }
    std::sort(modelIds.begin(), modelIds.end());
    if (modelIds.empty() && (!fluidRuntime || fluidRuntime->getNodeCount() == 0)) {
        return false;
    }

    const uint32_t fragmentCount = static_cast<uint32_t>(globalDomain.fragments.size());

    // Identify fixed fragments strictly from models
    std::vector<bool> fragmentIsFixed(fragmentCount, false);
    std::vector<float> fragmentFixedTemp(fragmentCount, 0.0f);
    for (uint32_t f = 0; f < fragmentCount; ++f) {
        const topology::Fragment& frag = globalDomain.fragments[f];
        if (frag.instanceId == topology::InvalidInstanceId) {
            continue;
        }
        const auto it = models.find(frag.instanceId);
        if (it == models.end() || !it->second) {
            continue;
        }
        const HeatModelRuntime& model = *it->second;
        if (model.hasPrescribedTemperature()) {
            fragmentIsFixed[f] = true;
            fragmentFixedTemp[f] = model.getPrescribedTemperatureC();
        }
    }

    // Build solver node mapping
    uint32_t currentSolverNode = 0;
    std::vector<uint32_t> localFragmentSolverNodes(fragmentCount, InvalidSolverNode);
    std::vector<uint32_t> localGlobalFragmentToFluidNode(fragmentCount, UINT32_MAX);
    std::vector<uint32_t> localFluidNodeToGlobalFragment;
    for (uint32_t runtimeModelId : modelIds) {
        const auto it = models.find(runtimeModelId);
        if (it == models.end() || !it->second) continue;

        HeatModelRuntime& model = *it->second;

        HeatGlobalSolver::TemperatureNodes block{};
        block.solverNodeOffset = currentSolverNode;
        block.externalA = &model.getExternalTempBufferA();
        block.externalB = &model.getExternalTempBufferB();
        block.temperatureA = &model.getCudaTempBufferA();
        block.temperatureB = &model.getCudaTempBufferB();

        uint32_t localFragmentIndex = 0;
        for (uint32_t f = 0; f < fragmentCount; ++f) {
            const topology::Fragment& frag = globalDomain.fragments[f];
            if (frag.instanceId == runtimeModelId) {
                if (!fragmentIsFixed[f]) {
                    localFragmentSolverNodes[f] = currentSolverNode++;
                    block.localNodeIds.push_back(localFragmentIndex);
                }
                ++localFragmentIndex;
            }
        }
        if (!block.localNodeIds.empty()) {
            temperatureNodes.push_back(std::move(block));
        }
    }

    // Assign fluid DOFs from unowned fragments
    if (fluidRuntime && fluidRuntime->getNodeCount() > 0) {
        HeatGlobalSolver::TemperatureNodes fluidBlock{};
        fluidBlock.solverNodeOffset = currentSolverNode;
        fluidBlock.externalA = &fluidRuntime->getExternalTempBufferA();
        fluidBlock.externalB = &fluidRuntime->getExternalTempBufferB();
        fluidBlock.temperatureA = &fluidRuntime->getCudaTempBufferA();
        fluidBlock.temperatureB = &fluidRuntime->getCudaTempBufferB();

        uint32_t localFluidIndex = 0;
        for (uint32_t f = 0; f < fragmentCount; ++f) {
            const topology::Fragment& frag = globalDomain.fragments[f];
            if (frag.instanceId == topology::InvalidInstanceId) {
                localGlobalFragmentToFluidNode[f] = localFluidIndex;
                localFluidNodeToGlobalFragment.push_back(f);
                localFragmentSolverNodes[f] = currentSolverNode++;
                fluidBlock.localNodeIds.push_back(localFluidIndex);
                ++localFluidIndex;
            }
        }
        if (!fluidBlock.localNodeIds.empty()) {
            temperatureNodes.push_back(std::move(fluidBlock));
        }
    }

    const uint32_t solverNodeCount = currentSolverNode;
    if (solverNodeCount == 0) {
        fragmentSolverNodes = std::move(localFragmentSolverNodes);
        return true;
    }

    rhsAdditions.assign(solverNodeCount, 0.0f);
    std::vector<double> thermalMassesDouble(solverNodeCount, 0.0);
    struct MatrixEntry {
        int col = 0;
        double value = 0.0;
    };
    std::vector<std::vector<MatrixEntry>> nodeEntries(solverNodeCount);

    // Populate thermal masses and diagonal baseline 
    for (uint32_t f = 0; f < fragmentCount; ++f) {
        const uint32_t node = localFragmentSolverNodes[f];
        if (node == InvalidSolverNode) continue;
        const auto& frag = globalDomain.fragments[f];
        float rho = 0.0f;
        float cp = 0.0f;
        float qVol = 0.0f;
        if (frag.instanceId != topology::InvalidInstanceId) {
            const auto it = models.find(frag.instanceId);
            if (it != models.end() && it->second) {
                rho = it->second->getDensity();
                cp = it->second->getSpecificHeat();
                qVol = it->second->getVolumetricPowerDensity();
            }
        } else if (fluidRuntime) {
            rho = fluidRuntime->getDensity();
            cp = fluidRuntime->getSpecificHeat();
        }

        if (!(rho > 0.0f) || !(cp > 0.0f) || !std::isfinite(rho) || !std::isfinite(cp)) {
            std::cerr << "[HeatDomainRuntime] assembleGlobalMatrix FAILED: invalid density (" << rho
                      << ") or specific heat (" << cp << ") for fragment " << f
                      << " (instance " << frag.instanceId << ")" << std::endl;
            return false;
        }

        if (!(frag.volume > 0.0f) || !std::isfinite(frag.volume)) {
            std::cerr << "[HeatDomainRuntime] assembleGlobalMatrix FAILED: invalid volume (" << frag.volume
                      << ") for fragment " << f << std::endl;
            return false;
        }

        const double mass = double(rho) * double(cp) * double(frag.volume);
        thermalMassesDouble[node] = mass;
        nodeEntries[node].push_back({static_cast<int>(node), mass});

        if (qVol != 0.0f && std::isfinite(qVol)) {
            rhsAdditions[node] += fixedTimeStep * qVol * frag.volume;
        }
    }

    bool assemblyValid = true;

    auto addThermalConnection = [&](uint32_t fragA, uint32_t fragB, double conductance) {
        if (!std::isfinite(conductance) || conductance < 0.0) {
            std::cerr << "[HeatDomainRuntime] invalid thermal conductance fragA="
                      << fragA << " fragB=" << fragB << " G=" << conductance << std::endl;
            assemblyValid = false;
            return;
        }
        if (conductance == 0.0) {
            return;
        }

        const double dtG = double(fixedTimeStep) * conductance;
        const uint32_t nodeA = localFragmentSolverNodes[fragA];
        const uint32_t nodeB = localFragmentSolverNodes[fragB];

        if (nodeA != InvalidSolverNode && nodeB != InvalidSolverNode) {
            // free <-> free
            nodeEntries[nodeA][0].value += dtG;
            nodeEntries[nodeA].push_back({static_cast<int>(nodeB), -dtG});
            nodeEntries[nodeB][0].value += dtG;
            nodeEntries[nodeB].push_back({static_cast<int>(nodeA), -dtG});
        } else if (nodeA != InvalidSolverNode && fragmentIsFixed[fragB]) {
            // free <-> fixed
            nodeEntries[nodeA][0].value += dtG;
            rhsAdditions[nodeA] += static_cast<float>(dtG * double(fragmentFixedTemp[fragB]));
        } else if (nodeB != InvalidSolverNode && fragmentIsFixed[fragA]) {
            // fixed <-> free
            nodeEntries[nodeB][0].value += dtG;
            rhsAdditions[nodeB] += static_cast<float>(dtG * double(fragmentFixedTemp[fragA]));
        } else if (fragmentIsFixed[fragA] && fragmentIsFixed[fragB]) {
            // fixed <-> fixed
        } else {
            std::cerr << "[HeatDomainRuntime] unexpected solver-node mapping for thermal connection fragments "
                      << fragA << " and " << fragB << " (nodeA=" << nodeA << ", fixedA=" << fragmentIsFixed[fragA]
                      << ", nodeB=" << nodeB << ", fixedB=" << fragmentIsFixed[fragB] << ")" << std::endl;
            assemblyValid = false;
        }
    };

    // General face conduction (solid-solid, solid-fluid, fluid-fluid)
    for (size_t faceIdx = 0; faceIdx < globalDomain.faces.size(); ++faceIdx) {
        const topology::Face& face = globalDomain.faces[faceIdx];
        if (face.fragmentA >= fragmentCount || face.fragmentB >= fragmentCount) continue;
        const auto& fragA = globalDomain.fragments[face.fragmentA];
        const auto& fragB = globalDomain.fragments[face.fragmentB];

        float kA = 0.0f;
        if (fragA.instanceId != topology::InvalidInstanceId) {
            const auto itA = models.find(fragA.instanceId);
            if (itA != models.end() && itA->second) {
                kA = itA->second->getConductivity();
            }
        } else if (fluidRuntime) {
            kA = fluidRuntime->getConductivity();
        }

        float kB = 0.0f;
        if (fragB.instanceId != topology::InvalidInstanceId) {
            const auto itB = models.find(fragB.instanceId);
            if (itB != models.end() && itB->second) {
                kB = itB->second->getConductivity();
            }
        } else if (fluidRuntime) {
            kB = fluidRuntime->getConductivity();
        }

        if (!(kA > 0.0f) || !(kB > 0.0f) || !std::isfinite(kA) || !std::isfinite(kB)) continue;
        if (!(face.area > 0.0f) || !std::isfinite(face.area)) continue;

        const double dA = static_cast<double>(face.orthogonalDistanceA);
        const double dB = static_cast<double>(face.orthogonalDistanceB);

        if (!std::isfinite(dA) || !std::isfinite(dB) ||
            !(dA > 0.0) || !(dB > 0.0)) {
            std::cerr
                << "[HeatDomainRuntime] assembleGlobalMatrix FAILED: "
                << "non-positive face orthogonal distance "
                << "(dA=" << dA << ", dB=" << dB << ") "
                << "between fragments "
                << face.fragmentA << " and " << face.fragmentB
                << std::endl;
            return false;
        }

        const double resistanceA =
            dA / (double(kA) * double(face.area));
        const double resistanceB =
            dB / (double(kB) * double(face.area));

        if (!std::isfinite(resistanceA) ||
            !std::isfinite(resistanceB) ||
            !(resistanceA > 0.0) ||
            !(resistanceB > 0.0)) {
            std::cerr << "[HeatDomainRuntime] assembleGlobalMatrix FAILED: non-physical resistance on face ("
                      << "rA=" << resistanceA << ", rB=" << resistanceB << ")" << std::endl;
            return false;
        }

        const double conductance = 1.0 / (resistanceA + resistanceB);
        addThermalConnection(face.fragmentA, face.fragmentB, conductance);
    }

    // Cross-instance contact conductance (strictly solid-solid only)
    for (const topology::ContactFace& face : globalDomain.contactFaces) {
        if (face.fragmentA >= fragmentCount || face.fragmentB >= fragmentCount) continue;
        const auto& fragA = globalDomain.fragments[face.fragmentA];
        const auto& fragB = globalDomain.fragments[face.fragmentB];
        // Guard against unowned fragments: ContactFace is strictly for solid-solid contact
        if (fragA.instanceId == topology::InvalidInstanceId || fragB.instanceId == topology::InvalidInstanceId) continue;
        if (fragA.instanceId == fragB.instanceId) continue;

        const auto itA = models.find(fragA.instanceId);
        const auto itB = models.find(fragB.instanceId);
        if (itA == models.end() || !itA->second || itB == models.end() || !itB->second) continue;

        if (!(face.area > 0.0f) || !std::isfinite(face.area)) continue;

        const double kA = double(itA->second->getConductivity());
        const double kB = double(itB->second->getConductivity());
        if (!(kA > 0.0) || !(kB > 0.0) || !std::isfinite(kA) || !std::isfinite(kB)) continue;

        const double dA = double(face.orthogonalDistanceA);
        const double dB = double(face.orthogonalDistanceB);
        const double area = double(face.area);

        const double denom = (dA / kA) + (dB / kB);
        if (!(denom > 0.0) || !std::isfinite(denom)) continue;

        const double conductance = area / denom;
        addThermalConnection(face.fragmentA, face.fragmentB, conductance);
    }

    if (!assemblyValid) {
        std::cerr << "[HeatDomainRuntime] assembleGlobalMatrix FAILED due to invalid thermal connections." << std::endl;
        return false;
    }

    for (const topology::BoundaryFace& bf : globalDomain.boundaryFaces) {
        if (bf.fragment >= fragmentCount || bf.domainFace >= 6) continue;
        const uint32_t node = localFragmentSolverNodes[bf.fragment];
        if (node == InvalidSolverNode) continue;
        if (!(bf.area > 0.0f) || !std::isfinite(bf.area)) continue;

        const uint32_t bcType = domainBoundaryTypes[bf.domainFace];
        if (bcType == heat::bc::Adiabatic) {
            continue;
        }

        if (bcType == heat::bc::Neumann) {
            const float q = domainBoundaryHeatFluxes[bf.domainFace];
            // Heat flux into the domain increases energy
            const float dtQ = fixedTimeStep * q * bf.area;
            rhsAdditions[node] += dtQ;
            continue;
        }

        // Both Dirichlet and Robin require fragment conductivity k and orthogonalDistance d
        const topology::Fragment& frag = globalDomain.fragments[bf.fragment];
        float k = 0.0f;
        if (frag.instanceId != topology::InvalidInstanceId) {
            const auto it = models.find(frag.instanceId);
            if (it != models.end() && it->second) {
                k = it->second->getConductivity();
            }
        } else if (fluidRuntime) {
            k = fluidRuntime->getConductivity();
        }

        if (!(k > 0.0f) || !std::isfinite(k)) continue;

        const float d = bf.orthogonalDistance;
        if (!std::isfinite(d) || !(d > 0.0f)) {
            std::cerr << "[HeatDomainRuntime] assembleGlobalMatrix FAILED: non-positive boundary orthogonal distance ("
                      << "d=" << d << ") for boundary face on fragment "
                      << bf.fragment << std::endl;
            return false;
        }

        if (bcType == heat::bc::Dirichlet) {
            const double boundaryTemperature = double(domainBoundaryTemperaturesC[bf.domainFace]);
            const double resistance = double(d) / (double(k) * double(bf.area));
            if (!std::isfinite(resistance) || !(resistance > 0.0)) {
                std::cerr << "[HeatDomainRuntime] assembleGlobalMatrix FAILED: non-physical Dirichlet resistance on boundary face ("
                          << "r=" << resistance << ")" << std::endl;
                return false;
            }
            const double wallConductance = 1.0 / resistance;
            const double dtG = double(fixedTimeStep) * wallConductance;
            nodeEntries[node][0].value += dtG;
            rhsAdditions[node] += static_cast<float>(dtG * boundaryTemperature);
        } else if (bcType == heat::bc::Robin) {
            const double h = double(domainBoundaryHeatTransferCoefficients[bf.domainFace]);
            if (h > 0.0 && std::isfinite(h)) {
                const double Tinf = double(domainBoundaryTemperaturesC[bf.domainFace]);
                const double rCond = double(d) / (double(k) * double(bf.area));
                const double rFilm = 1.0 / (h * double(bf.area));
                const double totalResistance = rCond + rFilm;
                if (!std::isfinite(totalResistance) || !(totalResistance > 0.0)) {
                    std::cerr << "[HeatDomainRuntime] assembleGlobalMatrix FAILED: non-physical Robin resistance on boundary face ("
                              << "r=" << totalResistance << ")" << std::endl;
                    return false;
                }
                const double boundaryConductance = 1.0 / totalResistance;
                const double dtG = double(fixedTimeStep) * boundaryConductance;
                nodeEntries[node][0].value += dtG;
                rhsAdditions[node] += static_cast<float>(dtG * Tinf);
            }
        }
    }

    // Flatten into CSR format
    rowOffsets.resize(solverNodeCount + 1, 0);
    for (uint32_t i = 0; i < solverNodeCount; ++i) {
        auto& row = nodeEntries[i];
        std::sort(row.begin(), row.end(), [](const MatrixEntry& a, const MatrixEntry& b) {
            return a.col < b.col;
        });
        std::vector<MatrixEntry> merged;
        for (const auto& entry : row) {
            if (!merged.empty() && merged.back().col == entry.col) {
                merged.back().value += entry.value;
            } else {
                merged.push_back(entry);
            }
        }
        rowOffsets[i + 1] = rowOffsets[i] + static_cast<int>(merged.size());
        for (const auto& entry : merged) {
            columnIndices.push_back(entry.col);
            values.push_back(static_cast<float>(entry.value));
        }
    }

    thermalMasses.resize(solverNodeCount);
    for (uint32_t i = 0; i < solverNodeCount; ++i) {
        thermalMasses[i] = static_cast<float>(thermalMassesDouble[i]);
    }

    fragmentSolverNodes = std::move(localFragmentSolverNodes);
    fluidNodeToGlobalFragment = std::move(localFluidNodeToGlobalFragment);
    globalFragmentToFluidNode = std::move(localGlobalFragmentToFluidNode);
    return true;
}

bool HeatDomainRuntime::rebuildGlobalSolver(
    const std::unordered_map<uint32_t, std::unique_ptr<HeatModelRuntime>>& models,
    HeatFluidRuntime* fluidRuntime,
    VulkanDevice& vulkanDevice,
    float fixedTimeStep) {
    waitGlobalSolveIdle();
    globalSolver.reset();

    std::vector<int> rowOffsets;
    std::vector<int> columnIndices;
    std::vector<float> values;
    std::vector<float> thermalMasses;
    std::vector<HeatGlobalSolver::TemperatureNodes> solverTemperatureNodes;
    std::vector<float> rhsAdditions;
    if (!assembleGlobalMatrix(
            models, fluidRuntime, fixedTimeStep, rowOffsets, columnIndices, values, thermalMasses,
            solverTemperatureNodes, rhsAdditions)) {
        return false;
    }

    if (solverTemperatureNodes.empty()) {
        cudaReadyCounter = 0;
        vulkanReleaseCounter = 0;
        pendingCudaReadyValue = 0;
        lastVulkanReleaseValue[0] = 0;
        lastVulkanReleaseValue[1] = 0;
        globalSolveInFlight = false;
        globalTopologyDirtyFlag = false;
        globalValuesDirtyFlag = false;
        clearGlobalSynchronization();
        return true;
    }

    globalSolver = std::make_unique<HeatGlobalSolver>();
    if (!globalSolver->initialize(
            vulkanDevice, rowOffsets, columnIndices, values, thermalMasses,
            solverTemperatureNodes, rhsAdditions)) {
        globalSolver.reset();
        clearGlobalSynchronization();
        return false;
    }

    if (fluidRuntime && fluidRuntime->getNodeCount() > 0) {
        if (!heatFluidFlow.initialize(vulkanDevice)) {
            std::cerr << "[HeatDomainRuntime] Failed to initialize heatFluidFlow" << std::endl;
            globalSolver.reset();
            clearGlobalSynchronization();
            return false;
        }

        if (!heatFluidFlow.buildTopology(
                globalDomain.fragments,
                globalDomain.faces,
                globalDomain.boundaryFaces,
                globalFragmentToFluidNode,
                fluidNodeToGlobalFragment)) {
            std::cerr << "[HeatDomainRuntime] Failed to build heatFluidFlow topology" << std::endl;
            globalSolver.reset();
            clearGlobalSynchronization();
            return false;
        }

        std::vector<heat::AdvectionFace> gpuFaces;
        gpuFaces.reserve(heatFluidFlow.getFaces().size());
        for (const auto& ff : heatFluidFlow.getFaces()) {
            const auto& face = globalDomain.faces[ff.faceIndex];
            const uint32_t nodeA = fragmentSolverNodes[face.fragmentA];
            const uint32_t nodeB = fragmentSolverNodes[face.fragmentB];
            if (nodeA != InvalidSolverNode && nodeB != InvalidSolverNode) {
                const float invVolA = 1.0f / globalDomain.fragments[face.fragmentA].volume;
                const float invVolB = 1.0f / globalDomain.fragments[face.fragmentB].volume;
                gpuFaces.push_back({ nodeA, nodeB, invVolA, invVolB });
            }
        }

        std::vector<heat::AdvectionBoundaryFace> gpuBfaces;
        gpuBfaces.reserve(heatFluidFlow.getBoundaryFaces().size());
        for (const auto& fbf : heatFluidFlow.getBoundaryFaces()) {
            const auto& bf = globalDomain.boundaryFaces[fbf.boundaryFaceIndex];
            const uint32_t node = fragmentSolverNodes[bf.fragment];
            if (node != InvalidSolverNode) {
                const float invVol = 1.0f / globalDomain.fragments[bf.fragment].volume;
                gpuBfaces.push_back({ node, invVol });
            }
        }

        if (!globalSolver->configureAdvectionTopology(
                gpuFaces, gpuBfaces,
                heatFluidFlow.getDeviceFaceFlux(),
                heatFluidFlow.getDeviceBoundaryFlux())) {
            globalSolver.reset();
            clearGlobalSynchronization();
            return false;
        }

        if (!updateFluidFlow(fluidRuntime, fixedTimeStep)) {
            globalSolver.reset();
            clearGlobalSynchronization();
            return false;
        }
    } else {
        heatFluidFlow.cleanup();
    }

    cudaReadyCounter = 0;
    vulkanReleaseCounter = 0;
    pendingCudaReadyValue = 0;
    lastVulkanReleaseValue[0] = 0;
    lastVulkanReleaseValue[1] = 0;
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

    std::vector<int> rowOffsets;
    std::vector<int> columnIndices;
    std::vector<float> values;
    std::vector<float> thermalMasses;
    std::vector<HeatGlobalSolver::TemperatureNodes> dummyTemperatureNodes;
    std::vector<float> rhsAdditions;
    if (!assembleGlobalMatrix(
            models, fluidRuntime, fixedTimeStep, rowOffsets, columnIndices, values, thermalMasses,
            dummyTemperatureNodes, rhsAdditions)) {
        return false;
    }

    if (!globalSolver->updateValues(values, thermalMasses, rhsAdditions)) {
        return false;
    }

    globalValuesDirtyFlag = false;
    return true;
}

bool HeatDomainRuntime::updateGlobalAdvection(HeatFluidRuntime* fluidRuntime, float fixedTimeStep) {
    if (!globalSolver) {
        return false;
    }

    if (fluidRuntime && fluidRuntime->getNodeCount() > 0) {
        if (!updateFluidFlow(fluidRuntime, fixedTimeStep)) {
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

    const uint32_t destIdx = sourceIsA ? 1 : 0;
    const uint64_t destVulkanRelease = lastVulkanReleaseValue[destIdx];
    const uint64_t cudaReady = ++cudaReadyCounter;

    if (!globalSolver->launch(sourceIsA, destVulkanRelease, cudaReady)) {
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
        return true;
    }
    if (result == HeatGlobalSolver::SolvePollResult::Failed) {
        globalSolveInFlight = false;
        return false;
    }
    return false;
}

void HeatDomainRuntime::prepareGlobalRenderSynchronization(bool activeIsA, bool newSolveCompleted) {
    if (!globalSolver) {
        clearGlobalSynchronization();
        return;
    }
    const uint32_t activeIdx = activeIsA ? 0 : 1;
    const uint64_t vulkanRelease = ++vulkanReleaseCounter;
    lastVulkanReleaseValue[activeIdx] = vulkanRelease;

    if (newSolveCompleted) {
        globalSynchronization.waitSemaphore = globalSolver->getCudaReadySemaphore();
        globalSynchronization.waitValue = pendingCudaReadyValue;
    } else {
        globalSynchronization.waitSemaphore = VK_NULL_HANDLE;
        globalSynchronization.waitValue = 0;
    }
    globalSynchronization.signalSemaphore = globalSolver->getVulkanReleaseSemaphore();
    globalSynchronization.signalValue = vulkanRelease;
}

void HeatDomainRuntime::waitGlobalSolveIdle() {
    if (globalSolver) {
        globalSolver->waitIdle();
    }
    globalSolveInFlight = false;
}

bool HeatDomainRuntime::updateFluidFlow(HeatFluidRuntime* fluidRuntime, float fixedTimeStep) {
    if (!globalSolver || !fluidRuntime || fluidRuntime->getNodeCount() == 0) {
        return false;
    }

    waitGlobalSolveIdle();

    if (!heatFluidFlow.updatePrescribedFlow(
            fluidRuntime->getPrescribedVelocity(),
            fixedTimeStep)) {
        return false;
    }

    const auto& timeStep = heatFluidFlow.getTimeStep();

    globalSolver->updateAdvectionStep(
        fluidRuntime->getInletTemperatureC(),
        timeStep.subcycleCount,
        timeStep.subcycleDt);
    return true;
}