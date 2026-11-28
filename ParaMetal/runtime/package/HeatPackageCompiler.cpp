#include "HeatPackageCompiler.hpp"

#include "PackageCompileContext.hpp"
#include "domain/BoundaryData.hpp"
#include "domain/CFDData.hpp"
#include "domain/FluidData.hpp"
#include "domain/HeatData.hpp"
#include "domain/HeatModelData.hpp"
#include "domain/RemeshData.hpp"
#include "domain/VoronoiData.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashProductHandle.hpp"
#include "nodegraph/NodeGraphState.hpp"
#include "nodegraph/NodeGraphUtils.hpp"
#include "nodegraph/NodeHeatProbeParams.hpp"
#include "nodegraph/NodeHeatSolveParams.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/package/RuntimePackages.hpp"
#include "runtime/package/passes/CoordinatePass.hpp"

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace runtimepackage {

static HeatProbeConfig makeHeatProbeConfig(const HeatProbeNodeParams& params) {
    HeatProbeConfig config{};
    config.enabled = params.enabled;
    config.opcPort = params.opcPort;
    config.showTemperature = params.showTemperature;
    config.showName = params.showName;
    config.items.reserve(params.items.size());
    for (const HeatProbeItem& item : params.items) {
        HeatProbeConfigItem converted{};
        converted.id = item.id;
        converted.name = item.name;
        converted.worldPosition = item.worldPosition;
        converted.enabled = item.enabled;
        config.items.push_back(std::move(converted));
    }
    return config;
}

static uint64_t computeHeatProbeConfigHash(
    const std::unordered_map<uint64_t, HeatProbeConfig>& probes) {
    std::vector<uint64_t> sourceKeys;
    sourceKeys.reserve(probes.size());
    for (const auto& [sourceKey, config] : probes) {
        (void)config;
        sourceKeys.push_back(sourceKey);
    }
    std::sort(sourceKeys.begin(), sourceKeys.end());

    uint64_t hash = HashBuilder::start();
    HashBuilder::combine(hash, static_cast<uint64_t>(sourceKeys.size()));
    for (uint64_t sourceKey : sourceKeys) {
        const HeatProbeConfig& config = probes.at(sourceKey);
        HashBuilder::combine(hash, sourceKey);
        HashBuilder::combine(hash, config.enabled ? 1u : 0u);
        HashBuilder::combine(hash, static_cast<uint64_t>(config.opcPort));
        HashBuilder::combine(hash, config.showTemperature ? 1u : 0u);
        HashBuilder::combine(hash, config.showName ? 1u : 0u);
        HashBuilder::combine(hash, static_cast<uint64_t>(config.items.size()));
        for (const HeatProbeConfigItem& item : config.items) {
            HashBuilder::combine(hash, static_cast<uint64_t>(item.id));
            HashBuilder::combineString(hash, item.name);
            HashBuilder::combineFloat(hash, item.worldPosition.x);
            HashBuilder::combineFloat(hash, item.worldPosition.y);
            HashBuilder::combineFloat(hash, item.worldPosition.z);
            HashBuilder::combine(hash, item.enabled ? 1u : 0u);
        }
    }
    return hash;
}

static void sealHeatPackage(
    HeatPackage& package,
    const HashValues& authoredHashes,
    uint64_t probeConfigHash) {
    uint64_t simulationHash = HashBuilder::start();
    HashBuilder::combine(simulationHash, authoredHashes.simulation);
    HashBuilder::combine(simulationHash, static_cast<uint64_t>(package.worldUnit));

    HashBuilder::combine(simulationHash, static_cast<uint64_t>(package.models.size()));
    for (const HeatModelPackage& model : package.models) {
        combineProductHandleHash(simulationHash, model.remeshProduct, HashDomain::Geometry);
        combineProductHandleHash(simulationHash, model.modelProduct, HashDomain::Geometry);
        HashBuilder::combinePod(simulationHash, model.remeshLocalToWorld);
        HashBuilder::combineFloat(simulationHash, model.density);
        HashBuilder::combineFloat(simulationHash, model.specificHeat);
        HashBuilder::combineFloat(simulationHash, model.conductivity);
        HashBuilder::combineFloat(simulationHash, model.initialTemperatureC);
        HashBuilder::combineFloat(simulationHash, model.volumetricPowerDensity);
        HashBuilder::combine(simulationHash, model.prescribedTemperatureEnabled ? 1u : 0u);
        HashBuilder::combineFloat(simulationHash, model.prescribedTemperatureC);
    }
    HashBuilder::combine(simulationHash, package.hasFluid ? 1u : 0u);
    if (package.hasFluid) {
        HashBuilder::combineFloat(simulationHash, package.fluidDensity);
        HashBuilder::combineFloat(simulationHash, package.fluidSpecificHeat);
        HashBuilder::combineFloat(simulationHash, package.fluidConductivity);
        HashBuilder::combineFloat(simulationHash, package.fluidInitialTemperatureC);
    }
    combineProductHandleHash(simulationHash, package.domainVoronoiProduct, HashDomain::Geometry);
    combineProductHandleHash(simulationHash, package.cfdProduct, HashDomain::Simulation);

    for (uint32_t face = 0; face < 6; ++face) {
        HashBuilder::combine(simulationHash, static_cast<uint64_t>(package.domainBoundaryTypes[face]));
        HashBuilder::combineFloat(simulationHash, package.domainBoundaryTemperaturesC[face]);
        HashBuilder::combineFloat(simulationHash, package.domainBoundaryHeatFluxes[face]);
        HashBuilder::combineFloat(simulationHash, package.domainBoundaryHeatTransferCoefficients[face]);
        HashBuilder::combineFloat(simulationHash, package.domainBoundaryInflowTemperaturesC[face]);
    }

    uint64_t displayHash = HashBuilder::start();
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showHeatOverlay ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showFluxVectors ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showHeatPalette ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showIsocontours ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showDomainBox ? 1u : 0u));
    HashBuilder::combine(displayHash, static_cast<uint64_t>(package.display.showFluidVolume ? 1u : 0u));
    HashBuilder::combineFloat(displayHash, package.display.fluidVolumeOpacity);
    HashBuilder::combineFloat(displayHash, package.display.fluxVectorScale);
    HashBuilder::combineFloat(displayHash, package.display.isocontourSpacingC);
    HashBuilder::combineFloat(displayHash, package.canonicalToWorldScale);
    for (const HeatModelPackage& model : package.models) {
        HashBuilder::combinePod(displayHash, model.localToWorld);
    }

    uint64_t fullHash = HashBuilder::start();
    HashBuilder::combine(fullHash, simulationHash);
    HashBuilder::combine(fullHash, displayHash);
    HashBuilder::combine(fullHash, authoredHashes.full);
    HashBuilder::combine(fullHash, probeConfigHash);

    package.hashes.simulation = simulationHash;
    package.hashes.geometry = simulationHash;
    package.hashes.full = fullHash;
    package.hashes.display = displayHash;
}

bool compileHeatPackage(const PackageCompileContext& context) {
    const HeatData* heat = context.payloads.get<HeatData>(context.output.payloadHandle);
    if (!heat) {
        context.errors.push_back("Heat package is missing its authored payload.");
        return false;
    }

    HeatPackage package{};
    const HeatPackage* previous = context.packages.findStored<HeatPackage>(context.outputSocketKey);
    package.productHandle = previous ? previous->productHandle : ProductHandle{};
    package.active = heat->active;
    package.simulationDuration = heat->simulationDuration;
    package.solverMaxIterations = heat->solverMaxIterations;
    package.solverConvergenceTolerance = heat->solverConvergenceTolerance;

    std::vector<NodeGraphNodeId> scanQueue = { context.node.id };
    std::unordered_set<uint32_t> visited = { context.node.id.value };

    while (!scanQueue.empty()) {
        const NodeGraphNodeId currentId = scanQueue.back();
        scanQueue.pop_back();

        for (const auto& [edgeId, edge] : context.graph.edges) {
            if (edge.fromNode == currentId) {
                const NodeGraphNode* downstreamNode = context.graph.node(edge.toNode);
                if (downstreamNode && getNodeTypeId(downstreamNode->typeId) == nodegraphtypes::HeatProbe) {
                    if (visited.insert(downstreamNode->id.value).second) {
                        const auto probeParams = readHeatProbeNodeParams(*downstreamNode);
                        package.resolvedHeatProbes[downstreamNode->id.value] =
                            makeHeatProbeConfig(probeParams);
                        scanQueue.push_back(downstreamNode->id);
                    }
                }
            }
        }
    }

    const HeatSolveNodeParams params = readHeatSolveNodeParams(context.node);
    package.display.showHeatOverlay = params.preview.showHeatOverlay;
    package.display.showFluxVectors = params.preview.showFluxVectors;
    package.display.showHeatPalette = params.preview.showHeatPalette;
    package.display.showIsocontours = params.preview.showIsocontours;
    package.display.showDomainBox = params.preview.showDomainBox;
    package.display.showFluidVolume = params.preview.showFluidVolume;
    package.display.fluidVolumeOpacity = static_cast<float>(params.preview.fluidVolumeOpacity);
    package.display.fluxVectorScale = static_cast<float>(params.preview.fluxVectorScale);
    package.display.isocontourSpacingC = params.isocontourSpacingC;

    const BoundaryData* boundary = heat->boundaryHandle.key != 0
        ? context.payloads.get<BoundaryData>(heat->boundaryHandle)
        : nullptr;
    if (boundary) {
        for (uint32_t face = 0; face < 6; ++face) {
            const BoundaryData::Face& authored = boundary->faces[face];
            package.domainBoundaryTypes[face] = static_cast<uint32_t>(authored.thermalType);
            package.domainBoundaryTemperaturesC[face] = authored.temperatureC;
            package.domainBoundaryHeatFluxes[face] = authored.heatFlux;
            package.domainBoundaryHeatTransferCoefficients[face] = authored.htc;
            package.domainBoundaryInflowTemperaturesC[face] = authored.inflowTemperatureC;
        }
    }

    struct AuthoredHeatModel {
        const HeatModelData* data = nullptr;
        uint64_t handleKey = 0;
    };
    std::map<uint64_t, AuthoredHeatModel> modelsByRemesh;
    for (const NodeDataHandle& handle : heat->heatModelHandles) {
        const HeatModelData* model = context.payloads.get<HeatModelData>(handle);
        if (!model || model->meshHandle.key == 0) {
            continue;
        }
        NodeDataHandle resolvedRemeshHandle{};
        if (!context.payloads.resolveRemesh(model->meshHandle, &resolvedRemeshHandle) ||
            resolvedRemeshHandle.key == 0) {
            if (heat->active) {
                context.errors.push_back("Heat package cannot resolve a heat-model remesh payload.");
                return false;
            }
            continue;
        }
        if (!modelsByRemesh.emplace(
                resolvedRemeshHandle.key,
                AuthoredHeatModel{model, handle.key}).second) {
            if (heat->active) {
                context.errors.push_back("Heat package contains duplicate heat-model remesh inputs.");
                return false;
            }
        }
    }

    const bool canResolveDomain = heat->domainVoronoiHandle.key != 0 &&
        heat->activeVoronoiCount == 1;
    const VoronoiData* voronoi = canResolveDomain
        ? context.payloads.get<VoronoiData>(heat->domainVoronoiHandle)
        : nullptr;

    if (heat->active) {
        if (!boundary) {
            context.errors.push_back("HeatSolve requires a Boundary Conditions input.");
            return false;
        }
        if (!canResolveDomain) {
            context.errors.push_back("HeatSolve requires exactly one active Voronoi input.");
            return false;
        }
        if (!voronoi || !voronoi->active) {
            context.errors.push_back("HeatSolve's domain Voronoi input is invalid.");
            return false;
        }
    }

    const VoronoiPackage* domainPackage = canResolveDomain
        ? context.packages.findAny<VoronoiPackage>(heat->domainVoronoiHandle.key)
        : nullptr;
    if (canResolveDomain && (!domainPackage || !domainPackage->productHandle.isValid())) {
        if (heat->active) {
            context.errors.push_back("HeatSolve is missing its Voronoi domain product.");
            return false;
        }
        domainPackage = nullptr;
    }

    if (heat->active) {
        if (modelsByRemesh.empty()) {
            context.errors.push_back("HeatSolve requires at least one heat model.");
            return false;
        }
        std::unordered_set<uint64_t> domainRemeshKeys;
        for (const NodeDataHandle& handle : voronoi->modelMeshHandles) {
            NodeDataHandle resolvedRemeshHandle{};
            if (!context.payloads.resolveRemesh(handle, &resolvedRemeshHandle) ||
                resolvedRemeshHandle.key == 0 ||
                !domainRemeshKeys.insert(resolvedRemeshHandle.key).second) {
                context.errors.push_back("Heat package's Voronoi domain contains an empty or duplicate remesh input.");
                return false;
            }
        }
        if (domainRemeshKeys.size() != modelsByRemesh.size()) {
            context.errors.push_back("Heat package requires the heat-model remesh set to exactly equal the global Voronoi domain's Remeshes set.");
            return false;
        }
        for (const auto& [remeshKey, authoredModel] : modelsByRemesh) {
            (void)authoredModel;
            if (domainRemeshKeys.find(remeshKey) == domainRemeshKeys.end()) {
                context.errors.push_back("Heat package requires the heat-model remesh set to exactly equal the global Voronoi domain's Remeshes set.");
                return false;
            }
        }
    }

    std::unordered_set<uint64_t> domainProductKeys;
    if (heat->active) {
        for (const ProductHandle& handle : domainPackage->globalRemeshProducts) {
            if (!handle.isValid() || !domainProductKeys.insert(handle.outputSocketKey).second) {
                context.errors.push_back("Heat package's global Voronoi domain contains duplicate or invalid remesh publications.");
                return false;
            }
        }
        if (domainProductKeys.size() != modelsByRemesh.size()) {
            context.errors.push_back("Heat package requires one current global remesh publication per heat model.");
            return false;
        }
    }

    std::unordered_set<uint64_t> heatRemeshProductKeys;
    for (const auto& [remeshKey, authoredModel] : modelsByRemesh) {
        (void)remeshKey;
        const HeatModelData* model = authoredModel.data;
        const RemeshPackage* remeshPackage = context.packages.findAny<RemeshPackage>(remeshKey);
        const ModelPackage* modelPackage = remeshPackage
            ? context.packages.findAny<ModelPackage>(remeshPackage->sourceModelProduct.outputSocketKey)
            : nullptr;
        if (!modelPackage || !remeshPackage ||
            !modelPackage->productHandle.isValid() ||
            !remeshPackage->productHandle.isValid()) {
            if (heat->active) {
                context.errors.push_back("Heat package is missing an upstream runtime product.");
                return false;
            }
            continue;
        }
        if (heat->active) {
            if (!heatRemeshProductKeys.insert(remeshPackage->productHandle.outputSocketKey).second) {
                context.errors.push_back("Heat package contains duplicate heat-model remesh publications.");
                return false;
            }
            bool inDomain = false;
            for (const ProductHandle& handle : domainPackage->globalRemeshProducts) {
                inDomain = inDomain || isSamePublication(handle, remeshPackage->productHandle);
            }
            if (!inDomain) {
                context.errors.push_back("Heat package contains a heat model outside the global Voronoi domain.");
                return false;
            }
        }

        HeatModelPackage compiledModel{};
        compiledModel.heatModelKey = authoredModel.handleKey;
        compiledModel.modelProduct = modelPackage->productHandle;
        compiledModel.remeshProduct = remeshPackage->productHandle;
        compiledModel.localToWorld = modelPackage->localToWorld;
        if (!context.payloads.resolveLocalToWorld(model->meshHandle, compiledModel.remeshLocalToWorld)) {
            if (heat->active) {
                context.errors.push_back("Heat package is missing the remesh placement for a heat model.");
                return false;
            }
            continue;
        }
        CoordinatePass::scaleTranslation(
            compiledModel.remeshLocalToWorld,
            units::canonicalToWorldScale(context.worldUnit));
        compiledModel.density = model->density;
        compiledModel.specificHeat = model->specificHeat;
        compiledModel.conductivity = model->conductivity;
        compiledModel.initialTemperatureC = model->initialTemperatureC;
        compiledModel.volumetricPowerDensity = model->volumetricHeatSource.powerDensity;
        compiledModel.prescribedTemperatureEnabled = model->prescribedTemperatureEnabled;
        compiledModel.prescribedTemperatureC = model->prescribedTemperatureC;
        package.models.push_back(compiledModel);
    }

    package.hasFluid = false;
    if (heat->hasFluid && heat->heatFluidHandle.key != 0) {
        const FluidData* fluid = context.payloads.get<FluidData>(heat->heatFluidHandle);
        if (!fluid) {
            if (heat->active) {
                context.errors.push_back("HeatSolve cannot resolve its Fluid input.");
                return false;
            }
        } else {
            package.hasFluid = true;
            package.fluidDensity = fluid->density;
            package.fluidSpecificHeat = fluid->specificHeat;
            package.fluidConductivity = fluid->conductivity;
            package.fluidInitialTemperatureC = fluid->initialTemperatureC;
        }
    }
    if (heat->cfdHandle.key != 0) {
        const CFDPackage* cfdPackage = context.packages.findAny<CFDPackage>(heat->cfdHandle.key);
        const CFDData* cfdData = context.payloads.get<CFDData>(heat->cfdHandle);
        if (cfdPackage && cfdPackage->productHandle.isValid()) {
            if (domainPackage && isSamePublication(cfdPackage->domainVoronoiProduct, domainPackage->productHandle)) {
                package.cfdProduct = cfdPackage->productHandle;
            } else if (domainPackage) {
                context.errors.push_back("Heat package CFD input does not share the same Voronoi domain as Heat.");
                return false;
            }
            if (cfdData &&
                heat->heatFluidHandle.key != 0 && cfdData->fluidHandle.key != 0 &&
                heat->heatFluidHandle.key != cfdData->fluidHandle.key) {
                context.errors.push_back("Heat and CFD must share the same Fluid input.");
                return false;
            }
            if (cfdData &&
                heat->boundaryHandle.key != 0 && cfdData->boundaryHandle.key != 0 &&
                heat->boundaryHandle.key != cfdData->boundaryHandle.key) {
                context.errors.push_back("Heat and CFD must share the same Boundary Conditions input.");
                return false;
            }
        }
    }

    CoordinatePass{}.run(package, context.worldUnit);
    if (domainPackage) {
        package.domainVoronoiProduct = domainPackage->productHandle;
    }
    for (const HeatModelPackage& model : package.models) {
        if (model.remeshProduct.isValid()) {
            package.displayDependencyKeys.push_back(
                model.remeshProduct.outputSocketKey);
        } else if (model.modelProduct.isValid()) {
            package.displayDependencyKeys.push_back(
                model.modelProduct.outputSocketKey);
        }
    }
    if (package.domainVoronoiProduct.isValid()) {
        package.displayDependencyKeys.push_back(package.domainVoronoiProduct.outputSocketKey);
    }
    if (package.cfdProduct.isValid()) {
        package.displayDependencyKeys.push_back(package.cfdProduct.outputSocketKey);
    }

    package.probeConfigHash = computeHeatProbeConfigHash(package.resolvedHeatProbes);
    sealHeatPackage(package, context.output.hashes, package.probeConfigHash);
    context.packages.apply<HeatPackage>(context.outputSocketKey, std::move(package));
    return true;
}

}
