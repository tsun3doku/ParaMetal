#include "RuntimePackageCompiler.hpp"

#include "nodegraph/NodeGraphPayloadTypes.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeGraphUtils.hpp"
#include "nodegraph/NodeGraphHash.hpp"
#include "nodegraph/NodeContactParams.hpp"
#include "nodegraph/NodeHeatSolveParams.hpp"
#include "nodegraph/NodeRemeshParams.hpp"
#include "nodegraph/NodeVoronoiParams.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "domain/HeatModelData.hpp"
#include "heat/HeatSystemPresets.hpp"

#include <set>
#include <unordered_set>

static void combineDependencyKey(uint64_t& hash, const NodeDataHandle& handle) {
    NodeGraphHash::combine(hash, handle.key);
}

static uint64_t buildModelPackageHash(const ModelPackage& package) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, 1u);
    NodeGraphHash::combine(hash, package.geometry.payloadHash);
    NodeGraphHash::combinePod(hash, package.localToWorld);
    return hash;
}

static uint64_t buildRemeshPackageHash(const RemeshPackage& package, const ECSRegistry& ecsRegistry) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, 2u);
    NodeGraphHash::combine(hash, package.sourceGeometry.payloadHash);
    NodeGraphHash::combine(hash, package.sourceMeshHandle.key);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(package.iterations));
    NodeGraphHash::combineFloat(hash, package.minAngleDegrees);
    NodeGraphHash::combineFloat(hash, package.maxEdgeLength);
    NodeGraphHash::combineFloat(hash, package.stepSize);
    return hash;
}

static uint64_t buildVoronoiPackageHash(const VoronoiPackage& package, const ECSRegistry& ecsRegistry) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, 3u);
    NodeGraphHash::combine(hash, package.authored.payloadHash);
    NodeGraphHash::combine(hash, package.voronoiHandle.key);
    NodeGraphHash::combine(hash, package.modelMeshHandle.key);
    NodeGraphHash::combine(hash, package.modelRemeshHandle.key);
    NodeGraphHash::combinePod(hash, package.modelLocalToWorld);
    return hash;
}

static uint64_t buildContactPackageHash(const ContactPackage& package, const ECSRegistry& ecsRegistry) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, 4u);
    NodeGraphHash::combine(hash, package.authored.payloadHash);
    NodeGraphHash::combine(hash, package.contactHandle.key);
    NodeGraphHash::combine(hash, package.modelAMeshHandle.key);
    NodeGraphHash::combine(hash, package.modelBMeshHandle.key);
    return hash;
}

static uint64_t buildHeatPackageHash(const HeatPackage& package, const ECSRegistry& ecsRegistry) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, 5u);
    NodeGraphHash::combine(hash, package.authored.payloadHash);
    NodeGraphHash::combine(hash, package.heatHandle.key);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(package.authored.voronoiHandles.size()));
    for (const NodeDataHandle& handle : package.authored.voronoiHandles) {
        combineDependencyKey(hash, handle);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(package.authored.contactHandles.size()));
    for (const NodeDataHandle& handle : package.authored.contactHandles) {
        combineDependencyKey(hash, handle);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(package.resolvedRemeshHandles.size()));
    for (size_t i = 0; i < package.resolvedRemeshHandles.size(); ++i) {
        NodeGraphHash::combine(hash, package.resolvedRemeshHandles[i].key);
        NodeGraphHash::combine(hash, package.resolvedModelHandles[i].key);
        NodeGraphHash::combineFloat(hash, package.resolvedDensity[i]);
        NodeGraphHash::combineFloat(hash, package.resolvedSpecificHeat[i]);
        NodeGraphHash::combineFloat(hash, package.resolvedConductivity[i]);
        NodeGraphHash::combineFloat(hash, package.resolvedInitialTemperature[i]);
        NodeGraphHash::combine(hash, static_cast<uint64_t>(package.resolvedBoundaryConditions[i]));
        NodeGraphHash::combineFloat(hash, package.resolvedFixedTemperatureValues[i]);
    }
    return hash;
}

ModelPackage RuntimePackageCompiler::buildModelPackage(const GeometryData& geometry) const {
    ModelPackage package{};
    package.geometry = geometry;
    package.localToWorld = geometry.localToWorld;
    package.packageHash = buildModelPackageHash(package);
    return package;
}

RemeshPackage RuntimePackageCompiler::buildRemeshPackage(
    const NodeGraphNode& node,
    const RemeshData& remesh,
    const NodePayloadRegistry* payloadRegistry,
    const ECSRegistry& registry,
    const NodeDataHandle& remeshHandle) const {
    RemeshPackage package{};
    if (!payloadRegistry || !remesh.active || remesh.sourceMeshHandle.key == 0) {
        return package;
    }

    const GeometryData* sourceGeometry = payloadRegistry->resolveGeometry(remesh.sourceMeshHandle);
    if (!sourceGeometry) {
        return package;
    }

    package.sourceGeometry = *sourceGeometry;
    package.iterations = remesh.iterations;
    package.minAngleDegrees = remesh.minAngleDegrees;
    package.maxEdgeLength = remesh.maxEdgeLength;
    package.stepSize = remesh.stepSize;
    const RemeshNodeParams nodeParams = readRemeshNodeParams(node);
    package.display.showRemeshOverlay = nodeParams.preview.showRemeshOverlay;
    package.display.showFaceNormals = nodeParams.preview.showFaceNormals;
    package.display.showVertexNormals = nodeParams.preview.showVertexNormals;
    package.display.normalLength = static_cast<float>(nodeParams.normalLength);
    package.remeshHandle = remeshHandle;
    package.sourceMeshHandle = remesh.sourceMeshHandle;
    package.packageHash = buildRemeshPackageHash(package, registry);
    return package;
}

VoronoiPackage RuntimePackageCompiler::buildVoronoiPackage(
    const NodeGraphNode& node,
    const NodePayloadRegistry* payloadRegistry,
    const ECSRegistry& registry,
    const VoronoiData& voronoi,
    const NodeDataHandle& voronoiHandle) const {
    VoronoiPackage package{};
    package.authored = voronoi;
    package.voronoiHandle = voronoiHandle;

    if (!payloadRegistry || !voronoi.active || voronoi.modelMeshHandle.key == 0) {
        return package;
    }

    NodeDataHandle modelHandle;
    const GeometryData* resolvedGeometry = payloadRegistry->resolveGeometry(voronoi.modelMeshHandle, &modelHandle);
    if (!resolvedGeometry || modelHandle.key == 0) {
        return package;
    }
    package.modelLocalToWorld = resolvedGeometry->localToWorld;
    package.modelMeshHandle = modelHandle;
    package.modelRemeshHandle = voronoi.modelMeshHandle;

    const VoronoiNodeParams nodeParams = readVoronoiNodeParams(node);
    package.display.showVoronoi = nodeParams.preview.showVoronoi;
    package.display.showPoints = nodeParams.preview.showPoints;
    package.packageHash = buildVoronoiPackageHash(package, registry);
    return package;
}

HeatPackage RuntimePackageCompiler::buildHeatPackage(
    const NodeGraphNode& node,
    const NodePayloadRegistry* payloadRegistry,
    const ECSRegistry& registry,
    const HeatData& heat,
    const NodeDataHandle& heatHandle) const {
    HeatPackage package{};
    package.authored = heat;
    package.heatHandle = heatHandle;

    const HeatSolveNodeParams nodeParams = readHeatSolveNodeParams(node);
    package.display.showHeatOverlay = nodeParams.preview.showHeatOverlay;
    package.display.showFluxVectors = nodeParams.preview.showFluxVectors;
    package.display.showHeatPalette = nodeParams.preview.showHeatPalette;
    package.display.fluxVectorScale = static_cast<float>(nodeParams.preview.fluxVectorScale);

    if (payloadRegistry) {
        std::unordered_set<uint64_t> seenRemeshKeys;
        for (const NodeDataHandle& heatModelHandle : heat.heatModelHandles) {
            const HeatModelData* heatModel = payloadRegistry->get<HeatModelData>(heatModelHandle);
            if (!heatModel || heatModel->meshHandle.key == 0) {
                continue;
            }
            if (!seenRemeshKeys.insert(heatModel->meshHandle.key).second) {
                continue;
            }

            NodeDataHandle modelHandle;
            const GeometryData* geometry = payloadRegistry->resolveGeometry(heatModel->meshHandle, &modelHandle);
            if (!geometry || modelHandle.key == 0) {
                continue;
            }

            float density = heatModel->density;
            float specificHeat = heatModel->specificHeat;
            float conductivity = heatModel->conductivity;
            const uint32_t modelNodeId = static_cast<uint32_t>(heatModelHandle.key);
            for (const HeatMaterialBinding& binding : heat.materialBindings) {
                if (binding.modelNodeId == modelNodeId) {
                    const HeatMaterialPreset& preset = heatMaterialPresetById(binding.presetId);
                    density = preset.density;
                    specificHeat = preset.specificHeat;
                    conductivity = preset.conductivity;
                    break;
                }
            }

            package.resolvedRemeshHandles.push_back(heatModel->meshHandle);
            package.resolvedModelHandles.push_back(modelHandle);
            package.resolvedDensity.push_back(density);
            package.resolvedSpecificHeat.push_back(specificHeat);
            package.resolvedConductivity.push_back(conductivity);
            package.resolvedInitialTemperature.push_back(heatModel->initialTemperature);
            package.resolvedBoundaryConditions.push_back(static_cast<uint32_t>(heatModel->boundaryCondition));
            package.resolvedFixedTemperatureValues.push_back(heatModel->fixedTemperatureValue);
        }
    }

    package.packageHash = buildHeatPackageHash(package, registry);
    return package;
}

ContactPackage RuntimePackageCompiler::buildContactPackage(
    const NodeGraphNode& node,
    const NodePayloadRegistry* payloadRegistry,
    const ECSRegistry& registry,
    const ContactData& contact,
    const NodeDataHandle& contactHandle) const {
    ContactPackage package{};
    package.authored = contact;
    package.contactHandle = contactHandle;
    const ContactNodeParams nodeParams = readContactNodeParams(node);
    package.display.showContactLines = nodeParams.preview.showContactLines;
    if (!payloadRegistry || !contact.active || !contact.pair.hasValidContact) {
        return package;
    }

    const ContactPairData& pair = contact.pair;
    if (pair.endpointA.meshHandle.key == 0 ||
        pair.endpointB.meshHandle.key == 0 ||
        pair.endpointA.meshHandle == pair.endpointB.meshHandle) {
        return package;
    }

    const ContactPairEndpoint& endpointA = pair.endpointA;
    const ContactPairEndpoint& endpointB = pair.endpointB;

    const GeometryData* geometryA = payloadRegistry->resolveGeometry(endpointA.meshHandle);
    const GeometryData* geometryB = payloadRegistry->resolveGeometry(endpointB.meshHandle);
    if (!geometryA || !geometryB) {
        return package;
    }

    package.modelALocalToWorld = geometryA->localToWorld;
    package.modelBLocalToWorld = geometryB->localToWorld;
    package.modelAMeshHandle = endpointA.meshHandle;
    package.modelBMeshHandle = endpointB.meshHandle;

    package.packageHash = buildContactPackageHash(package, registry);
    return package;
}

template <typename PackageT>
void RuntimePackageCompiler::applyPackage(ECSRegistry& registry, uint64_t socketKey, const PackageT& pkg, std::unordered_set<ECSEntity>& staleEntities) {
    auto entity = static_cast<ECSEntity>(socketKey);
    if (!registry.valid(entity)) {
        static_cast<void>(registry.create(entity));
    }
    if (registry.all_of<PackageT>(entity)) {
        if (!registry.get<PackageT>(entity).matches(pkg)) {
            registry.replace<PackageT>(entity, pkg);
        }
    } else {
        registry.emplace<PackageT>(entity, pkg);
    }
    if (registry.all_of<Stale>(entity)) {
        registry.remove<Stale>(entity);
    }
    staleEntities.erase(entity);
}

void RuntimePackageCompiler::compileAndApply(
    const NodeGraphState& graphState,
    const NodeGraphEvaluationState& evaluationState,
    const NodePayloadRegistry* payloadRegistry,
    ECSRegistry& registry,
    std::unordered_set<ECSEntity>& staleEntities) const {

    for (const auto& [id, node] : graphState.nodes) {
        for (const NodeGraphSocket& output : node.outputs) {
            if (output.contract.producedPayloadType == payloadtypes::None) {
                continue;
            }

            const uint64_t outSocketKey = (static_cast<uint64_t>(node.id.value) << 32) | static_cast<uint64_t>(output.id.value);
            const auto valueIt = evaluationState.outputBySocket.find(outSocketKey);
            if (valueIt == evaluationState.outputBySocket.end() ||
                valueIt->second.status != EvaluatedSocketStatus::Value ||
                valueIt->second.data.payloadHandle.key == 0) {
                continue;
            }

            const NodeDataHandle payloadHandle = valueIt->second.data.payloadHandle;
            const auto payloadType = output.contract.producedPayloadType;

            if (payloadType == payloadtypes::Geometry) {
                if (node.frozen) {
                    auto entity = static_cast<ECSEntity>(outSocketKey);
                    if (registry.valid(entity) && registry.all_of<ModelPackage>(entity)) {
                        staleEntities.erase(entity);
                    }
                    break;
                }

                const GeometryData* geometry = payloadRegistry->get<GeometryData>(payloadHandle);
                if (!geometry) {
                    break;
                }

                ModelPackage package = buildModelPackage(*geometry);
                applyPackage<ModelPackage>(registry, outSocketKey, package, staleEntities);
            } else if (payloadType == payloadtypes::Remesh) {
                if (node.frozen) {
                    auto entity = static_cast<ECSEntity>(outSocketKey);
                    if (registry.valid(entity) && registry.all_of<RemeshPackage>(entity)) {
                        staleEntities.erase(entity);
                    }
                    break;
                }

                const RemeshData* remesh = payloadRegistry->get<RemeshData>(payloadHandle);
                if (!remesh || !remesh->active || remesh->sourceMeshHandle.key == 0) {
                    break;
                }

                RemeshPackage package = buildRemeshPackage(node, *remesh, payloadRegistry, registry, payloadHandle);
                applyPackage<RemeshPackage>(registry, outSocketKey, package, staleEntities);
            } else if (payloadType == payloadtypes::Voronoi) {
                if (node.frozen) {
                    auto entity = static_cast<ECSEntity>(outSocketKey);
                    if (registry.valid(entity) && registry.all_of<VoronoiPackage>(entity)) {
                        staleEntities.erase(entity);
                    }
                    break;
                }

                const VoronoiData* voronoi = payloadRegistry->get<VoronoiData>(payloadHandle);
                if (!voronoi) {
                    break;
                }

                VoronoiPackage package = buildVoronoiPackage(node, payloadRegistry, registry, *voronoi, payloadHandle);
                applyPackage<VoronoiPackage>(registry, outSocketKey, package, staleEntities);
            } else if (payloadType == payloadtypes::Contact) {
                if (node.frozen) {
                    auto entity = static_cast<ECSEntity>(outSocketKey);
                    if (registry.valid(entity) && registry.all_of<ContactPackage>(entity)) {
                        staleEntities.erase(entity);
                    }
                    break;
                }

                const ContactData* contact = payloadRegistry->get<ContactData>(payloadHandle);
                if (!contact) {
                    break;
                }

                ContactPackage package = buildContactPackage(node, payloadRegistry, registry, *contact, payloadHandle);
                applyPackage<ContactPackage>(registry, outSocketKey, package, staleEntities);
            } else if (payloadType == payloadtypes::Heat) {
                if (node.frozen) {
                    auto entity = static_cast<ECSEntity>(outSocketKey);
                    if (registry.valid(entity) && registry.all_of<HeatPackage>(entity)) {
                        staleEntities.erase(entity);
                    }
                    break;
                }

                const HeatData* heat = payloadRegistry->get<HeatData>(payloadHandle);
                if (!heat) {
                    break;
                }

                HeatPackage package = buildHeatPackage(node, payloadRegistry, registry, *heat, payloadHandle);
                applyPackage<HeatPackage>(registry, outSocketKey, package, staleEntities);
            }
        }
    }
}
