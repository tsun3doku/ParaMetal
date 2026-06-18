#include "RuntimePackageCompiler.hpp"

#include "nodegraph/NodeGraphPayloadTypes.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeGraphUtils.hpp"
#include "hash/HashBuilder.hpp"
#include "hash/HashPackage.hpp"
#include "nodegraph/NodeContactParams.hpp"
#include "nodegraph/NodeHeatSolveParams.hpp"
#include "nodegraph/NodeRemeshParams.hpp"
#include "nodegraph/NodeVoronoiParams.hpp"
#include "nodegraph/NodeTransform.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "domain/HeatData.hpp"
#include "domain/HeatModelData.hpp"
#include "domain/PointData.hpp"
#include "domain/VoronoiData.hpp"
#include "../util/GeometryUtils.hpp"

#include <unordered_map>
#include <unordered_set>

ModelPackage RuntimePackageCompiler::buildModelPackage(const GeometryData& geometry) const {
    ModelPackage package{};
    package.geometry = geometry;
    package.localToWorld = geometry.localToWorld;
    HashPackage::seal(package);
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
    HashPackage::seal(package);
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
    package.domainType = voronoi.domainType;

    if (!payloadRegistry || !voronoi.active) {
        return package;
    }

    if (voronoi.domainType == DomainType::Mesh) {
        if (voronoi.modelMeshHandle.key == 0 || voronoi.pointsPayloadHandle.key == 0) {
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
        package.pointsPayloadHandle = voronoi.pointsPayloadHandle;
    } else if (voronoi.domainType == DomainType::Points) {
        if (voronoi.pointsPayloadHandle.key == 0) {
            return package;
        }
        package.pointsPayloadHandle = voronoi.pointsPayloadHandle;
    }

    const VoronoiNodeParams nodeParams = readVoronoiNodeParams(node);
    package.display.showVoronoi = nodeParams.preview.showVoronoi;
    package.display.showPoints = nodeParams.preview.showPoints;
    HashPackage::seal(package);
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

    const auto invalidatePackage = [&package]() {
        package.authored.active = false;
        HashPackage::seal(package);
        return package;
    };

    if (!payloadRegistry) {
        return invalidatePackage();
    }

    std::unordered_map<uint64_t, const HeatModelData*> heatModelByRemeshKey;
    for (const NodeDataHandle& heatModelHandle : heat.heatModelHandles) {
        const HeatModelData* heatModel = payloadRegistry->get<HeatModelData>(heatModelHandle);
        if (!heatModel || heatModel->meshHandle.key == 0) {
            return invalidatePackage();
        }
        if (!heatModelByRemeshKey.emplace(heatModel->meshHandle.key, heatModel).second) {
            return invalidatePackage();
        }
    }

    // Resolve materials by pairing each Voronoi domain with its matching HeatModel
    std::unordered_set<uint64_t> usedHeatModelRemeshKeys;
    for (const NodeDataHandle& voronoiHandle : heat.voronoiHandles) {
        const VoronoiData* voronoi = payloadRegistry->get<VoronoiData>(voronoiHandle);
        if (!voronoi || !voronoi->active || voronoi->modelMeshHandle.key == 0) {
            return invalidatePackage();
        }

        auto it = heatModelByRemeshKey.find(voronoi->modelMeshHandle.key);
        if (it == heatModelByRemeshKey.end()) {
            return invalidatePackage();
        }
        usedHeatModelRemeshKeys.insert(voronoi->modelMeshHandle.key);

        const HeatModelData* heatModel = it->second;
        NodeDataHandle modelHandle;
        const GeometryData* geometry = payloadRegistry->resolveGeometry(heatModel->meshHandle, &modelHandle);
        if (!geometry || modelHandle.key == 0) {
            return invalidatePackage();
        }

        package.resolvedRemeshHandles.push_back(heatModel->meshHandle);
        package.resolvedModelHandles.push_back(modelHandle);
        package.resolvedDensity.push_back(heatModel->density);
        package.resolvedSpecificHeat.push_back(heatModel->specificHeat);
        package.resolvedConductivity.push_back(heatModel->conductivity);
        package.resolvedInitialTemperature.push_back(heatModel->initialTemperature);
        package.resolvedBoundaryConditions.push_back(static_cast<uint32_t>(heatModel->boundaryCondition));
        package.resolvedFixedTemperatureValues.push_back(heatModel->fixedTemperatureValue);
    }

    if (usedHeatModelRemeshKeys.size() != heatModelByRemeshKey.size()) {
        return invalidatePackage();
    }

    HashPackage::seal(package);
    return package;
}

PointPackage RuntimePackageCompiler::buildPointPackage(
    const NodePayloadRegistry* payloadRegistry,
    const PointData& pointData,
    const NodeDataHandle& pointHandle,
    const ECSRegistry& registry) const {
    (void)registry;
    PointPackage package{};
    package.pointsPayloadHandle = pointHandle;
    if (!payloadRegistry || !pointData.active) {
        return package;
    }
    package.positions = pointData.positions;
    package.pointCount = static_cast<uint32_t>(pointData.positions.size());
    package.localToWorld = pointData.localToWorld;
    HashPackage::seal(package);
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

    HashPackage::seal(package);
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

void RuntimePackageCompiler::compileMeshPoints(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    ECSRegistry& registry,
    std::unordered_set<ECSEntity>& staleEntities) const {

    if (!payloadRegistry || node.outputs.empty() || node.inputs.empty()) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        if (node.frozen) {
            auto entity = static_cast<ECSEntity>(outSocketKey);
            if (registry.valid(entity) && registry.all_of<PointPackage>(entity)) {
                staleEntities.erase(entity);
            }
            continue;
        }

        const auto valueIt = execState.outputBySocket.find(outSocketKey);
        if (valueIt == execState.outputBySocket.end() ||
            valueIt->second.status != EvaluatedSocketStatus::Value ||
            valueIt->second.data.payloadHandle.key == 0) {
            continue;
        }

        const PointData* pointData = payloadRegistry->get<PointData>(valueIt->second.data.payloadHandle);
        if (!pointData || !pointData->active) {
            continue;
        }

        PointPackage package{};
        package.pointsPayloadHandle = valueIt->second.data.payloadHandle;
        package.positions = pointData->positions;
        package.pointCount = static_cast<uint32_t>(pointData->positions.size());
        package.localToWorld = pointData->localToWorld;

        // Look up upstream remesh from Geometry input socket
        const uint64_t inSocketKey = NodeSocketKey(node.id, node.inputs[0].id).value;
        const auto upstreamIt = execState.upstreamSocket.find(inSocketKey);
        if (upstreamIt != execState.upstreamSocket.end() && upstreamIt->second != 0) {
            auto upstreamEntity = static_cast<ECSEntity>(upstreamIt->second);
            if (registry.valid(upstreamEntity) && registry.all_of<RemeshProduct>(upstreamEntity)) {
                const auto& product = registry.get<RemeshProduct>(upstreamEntity);
                if (product.isValid()) {
                    package.positions.clear();
                    package.positions.reserve(product.intrinsicMesh.vertices.size());
                    for (const auto& vertex : product.intrinsicMesh.vertices) {
                        package.positions.push_back(glm::vec4(vertex.position, 1.0f));
                    }
                    package.pointCount = static_cast<uint32_t>(package.positions.size());
                }
            }
        }

        HashPackage::seal(package);
        applyPackage<PointPackage>(registry, outSocketKey, package, staleEntities);
    }
}

void RuntimePackageCompiler::compilePoints(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    ECSRegistry& registry,
    std::unordered_set<ECSEntity>& staleEntities) const {

    if (!payloadRegistry || node.outputs.empty()) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        if (node.frozen) {
            auto entity = static_cast<ECSEntity>(outSocketKey);
            if (registry.valid(entity) && registry.all_of<PointPackage>(entity)) {
                staleEntities.erase(entity);
            }
            continue;
        }

        const auto valueIt = execState.outputBySocket.find(outSocketKey);
        if (valueIt == execState.outputBySocket.end() ||
            valueIt->second.status != EvaluatedSocketStatus::Value ||
            valueIt->second.data.payloadHandle.key == 0) {
            continue;
        }

        const PointData* pointData = payloadRegistry->get<PointData>(valueIt->second.data.payloadHandle);
        if (!pointData) {
            continue;
        }

        PointPackage package = buildPointPackage(payloadRegistry, *pointData, valueIt->second.data.payloadHandle, registry);
        applyPackage<PointPackage>(registry, outSocketKey, package, staleEntities);
    }
}

void RuntimePackageCompiler::compileMerge(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    ECSRegistry& registry,
    std::unordered_set<ECSEntity>& staleEntities) const {

    if (node.outputs.empty() || node.inputs.empty()) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        if (node.frozen) {
            auto entity = static_cast<ECSEntity>(outSocketKey);
            if (registry.valid(entity)) {
                if (registry.all_of<ModelPackage>(entity) || registry.all_of<PointPackage>(entity)) {
                    staleEntities.erase(entity);
                }
            }
            continue;
        }

        const auto valueIt = execState.outputBySocket.find(outSocketKey);
        if (valueIt == execState.outputBySocket.end() ||
            valueIt->second.status != EvaluatedSocketStatus::Value ||
            valueIt->second.data.payloadHandle.key == 0) {
            continue;
        }

        const uint8_t payloadType = valueIt->second.data.dataType;

        if (payloadType == payloadtypes::Geometry) {
            const GeometryData* geometry = payloadRegistry->get<GeometryData>(valueIt->second.data.payloadHandle);
            if (!geometry) {
                continue;
            }
            ModelPackage package = buildModelPackage(*geometry);
            applyPackage<ModelPackage>(registry, outSocketKey, package, staleEntities);
        } else if (payloadType == payloadtypes::Points) {
            const NodeGraphSocket* inSocket = nullptr;
            for (const auto& s : node.inputs) {
                if (s.variadic) {
                    inSocket = &s;
                    break;
                }
            }
            if (!inSocket) {
                continue;
            }

            const uint64_t inSocketKey = NodeSocketKey(node.id, inSocket->id).value;
            const auto socketsIt = execState.upstreamSockets.find(inSocketKey);
            if (socketsIt == execState.upstreamSockets.end() || socketsIt->second.empty()) {
                continue;
            }

            std::vector<const PointPackage*> upstreamPackages;
            std::vector<glm::mat4> upstreamL2W;
            for (uint64_t upKey : socketsIt->second) {
                auto upEntity = static_cast<ECSEntity>(upKey);
                if (!registry.valid(upEntity) || !registry.all_of<PointPackage>(upEntity)) {
                    continue;
                }
                const auto& upPkg = registry.get<PointPackage>(upEntity);
                if (upPkg.positions.empty()) {
                    continue;
                }
                upstreamPackages.push_back(&upPkg);
                upstreamL2W.push_back(toMat4(upPkg.localToWorld));
            }

            if (upstreamPackages.empty()) {
                continue;
            }

            const glm::mat4 refL2W = upstreamL2W[0];
            const glm::mat4 invRefL2W = glm::inverse(refL2W);

            PointPackage package{};
            package.pointsPayloadHandle = valueIt->second.data.payloadHandle;
            package.localToWorld = upstreamPackages[0]->localToWorld;
            for (size_t i = 0; i < upstreamPackages.size(); ++i) {
                const glm::mat4 upToRef = invRefL2W * upstreamL2W[i];
                for (const auto& pos : upstreamPackages[i]->positions) {
                    package.positions.push_back(upToRef * pos);
                }
            }
            package.pointCount = static_cast<uint32_t>(package.positions.size());
            HashPackage::seal(package);
            applyPackage<PointPackage>(registry, outSocketKey, package, staleEntities);
        }
    }
}

void RuntimePackageCompiler::compileTransform(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    ECSRegistry& registry,
    std::unordered_set<ECSEntity>& staleEntities) const {

    if (node.outputs.empty() || node.inputs.empty()) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        if (node.frozen) {
            auto entity = static_cast<ECSEntity>(outSocketKey);
            if (registry.valid(entity)) {
                if (registry.all_of<ModelPackage>(entity) || registry.all_of<PointPackage>(entity)) {
                    staleEntities.erase(entity);
                }
            }
            continue;
        }

        const auto valueIt = execState.outputBySocket.find(outSocketKey);
        if (valueIt == execState.outputBySocket.end() ||
            valueIt->second.status != EvaluatedSocketStatus::Value ||
            valueIt->second.data.payloadHandle.key == 0) {
            continue;
        }

        const uint8_t payloadType = valueIt->second.data.dataType;

        if (payloadType == payloadtypes::Geometry) {
            const GeometryData* geometry = payloadRegistry->get<GeometryData>(valueIt->second.data.payloadHandle);
            if (!geometry) {
                continue;
            }
            ModelPackage package = buildModelPackage(*geometry);
            applyPackage<ModelPackage>(registry, outSocketKey, package, staleEntities);
        } else if (payloadType == payloadtypes::Points) {
            const uint64_t inSocketKey = NodeSocketKey(node.id, node.inputs[0].id).value;
            const auto upstreamIt = execState.upstreamSocket.find(inSocketKey);
            if (upstreamIt == execState.upstreamSocket.end() || upstreamIt->second == 0) {
                continue;
            }

            auto upEntity = static_cast<ECSEntity>(upstreamIt->second);
            if (!registry.valid(upEntity) || !registry.all_of<PointPackage>(upEntity)) {
                continue;
            }

            const auto& upPkg = registry.get<PointPackage>(upEntity);
            if (upPkg.positions.empty()) {
                continue;
            }

            PointPackage package{};
            package.pointsPayloadHandle = valueIt->second.data.payloadHandle;
            package.positions = upPkg.positions;
            package.pointCount = upPkg.pointCount;
            const std::array<float, 16> localTransform = NodeTransform::buildLocalTransformArray(node);
            package.localToWorld = toMatrixArray(toMat4(upPkg.localToWorld) * toMat4(localTransform));
            HashPackage::seal(package);
            applyPackage<PointPackage>(registry, outSocketKey, package, staleEntities);
        }
    }
}

void RuntimePackageCompiler::compileModel(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    ECSRegistry& registry,
    std::unordered_set<ECSEntity>& staleEntities) const {

    if (!payloadRegistry) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        if (node.frozen) {
            auto entity = static_cast<ECSEntity>(outSocketKey);
            if (registry.valid(entity) && registry.all_of<ModelPackage>(entity)) {
                staleEntities.erase(entity);
            }
            continue;
        }

        const auto valueIt = execState.outputBySocket.find(outSocketKey);
        if (valueIt == execState.outputBySocket.end() ||
            valueIt->second.status != EvaluatedSocketStatus::Value ||
            valueIt->second.data.payloadHandle.key == 0) {
            continue;
        }

        const GeometryData* geometry = payloadRegistry->get<GeometryData>(valueIt->second.data.payloadHandle);
        if (!geometry) {
            continue;
        }

        ModelPackage package = buildModelPackage(*geometry);
        applyPackage<ModelPackage>(registry, outSocketKey, package, staleEntities);
    }
}

void RuntimePackageCompiler::compileGroup(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    ECSRegistry& registry,
    std::unordered_set<ECSEntity>& staleEntities) const {
    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        if (node.frozen) {
            auto entity = static_cast<ECSEntity>(outSocketKey);
            if (registry.valid(entity) && registry.all_of<ModelPackage>(entity)) {
                staleEntities.erase(entity);
            }
            continue;
        }

        const auto valueIt = execState.outputBySocket.find(outSocketKey);
        if (valueIt == execState.outputBySocket.end() ||
            valueIt->second.status != EvaluatedSocketStatus::Value ||
            valueIt->second.data.payloadHandle.key == 0) {
            continue;
        }

        const GeometryData* geometry = payloadRegistry->get<GeometryData>(valueIt->second.data.payloadHandle);
        if (!geometry) {
            continue;
        }

        ModelPackage package = buildModelPackage(*geometry);
        applyPackage<ModelPackage>(registry, outSocketKey, package, staleEntities);
    }
}

void RuntimePackageCompiler::compileRemesh(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    ECSRegistry& registry,
    std::unordered_set<ECSEntity>& staleEntities) const {

    if (!payloadRegistry) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        if (node.frozen) {
            auto entity = static_cast<ECSEntity>(outSocketKey);
            if (registry.valid(entity) && registry.all_of<RemeshPackage>(entity)) {
                staleEntities.erase(entity);
            }
            continue;
        }

        const auto valueIt = execState.outputBySocket.find(outSocketKey);
        if (valueIt == execState.outputBySocket.end() ||
            valueIt->second.status != EvaluatedSocketStatus::Value ||
            valueIt->second.data.payloadHandle.key == 0) {
            continue;
        }

        const RemeshData* remesh = payloadRegistry->get<RemeshData>(valueIt->second.data.payloadHandle);
        if (!remesh || !remesh->active || remesh->sourceMeshHandle.key == 0) {
            continue;
        }

        RemeshPackage package = buildRemeshPackage(node, *remesh, payloadRegistry, registry, valueIt->second.data.payloadHandle);
        applyPackage<RemeshPackage>(registry, outSocketKey, package, staleEntities);
    }
}

void RuntimePackageCompiler::compileVoronoi(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    ECSRegistry& registry,
    std::unordered_set<ECSEntity>& staleEntities) const {

    if (!payloadRegistry) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        if (node.frozen) {
            auto entity = static_cast<ECSEntity>(outSocketKey);
            if (registry.valid(entity) && registry.all_of<VoronoiPackage>(entity)) {
                staleEntities.erase(entity);
            }
            continue;
        }

        const auto valueIt = execState.outputBySocket.find(outSocketKey);
        if (valueIt == execState.outputBySocket.end() ||
            valueIt->second.status != EvaluatedSocketStatus::Value ||
            valueIt->second.data.payloadHandle.key == 0) {
            continue;
        }

        const VoronoiData* voronoi = payloadRegistry->get<VoronoiData>(valueIt->second.data.payloadHandle);
        if (!voronoi) {
            continue;
        }

        VoronoiPackage package = buildVoronoiPackage(node, payloadRegistry, registry, *voronoi, valueIt->second.data.payloadHandle);
        applyPackage<VoronoiPackage>(registry, outSocketKey, package, staleEntities);
    }
}

void RuntimePackageCompiler::compileContact(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    ECSRegistry& registry,
    std::unordered_set<ECSEntity>& staleEntities) const {

    if (!payloadRegistry) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        if (node.frozen) {
            auto entity = static_cast<ECSEntity>(outSocketKey);
            if (registry.valid(entity) && registry.all_of<ContactPackage>(entity)) {
                staleEntities.erase(entity);
            }
            continue;
        }

        const auto valueIt = execState.outputBySocket.find(outSocketKey);
        if (valueIt == execState.outputBySocket.end() ||
            valueIt->second.status != EvaluatedSocketStatus::Value ||
            valueIt->second.data.payloadHandle.key == 0) {
            continue;
        }

        const ContactData* contact = payloadRegistry->get<ContactData>(valueIt->second.data.payloadHandle);
        if (!contact) {
            continue;
        }

        ContactPackage package = buildContactPackage(node, payloadRegistry, registry, *contact, valueIt->second.data.payloadHandle);
        applyPackage<ContactPackage>(registry, outSocketKey, package, staleEntities);
    }
}

void RuntimePackageCompiler::compileHeatSolve(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    ECSRegistry& registry,
    std::unordered_set<ECSEntity>& staleEntities) const {

    if (!payloadRegistry) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        if (node.frozen) {
            auto entity = static_cast<ECSEntity>(outSocketKey);
            if (registry.valid(entity) && registry.all_of<HeatPackage>(entity)) {
                staleEntities.erase(entity);
            }
            continue;
        }

        const auto valueIt = execState.outputBySocket.find(outSocketKey);
        if (valueIt == execState.outputBySocket.end() ||
            valueIt->second.status != EvaluatedSocketStatus::Value ||
            valueIt->second.data.payloadHandle.key == 0) {
            continue;
        }

        const HeatData* heat = payloadRegistry->get<HeatData>(valueIt->second.data.payloadHandle);
        if (!heat) {
            continue;
        }

        HeatPackage package = buildHeatPackage(node, payloadRegistry, registry, *heat, valueIt->second.data.payloadHandle);
        applyPackage<HeatPackage>(registry, outSocketKey, package, staleEntities);
    }
}

void RuntimePackageCompiler::compileAndApply(
    const NodeGraphState& graphState,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    ECSRegistry& registry,
    std::unordered_set<ECSEntity>& staleEntities) const {

    for (NodeGraphNodeId nodeId : execState.executionOrder) {
        const auto it = graphState.nodes.find(nodeId.value);
        if (it == graphState.nodes.end()) {
            continue;
        }

        const NodeGraphNode& node = it->second;
        const NodeTypeId typeId = getNodeTypeId(node.typeId);

        if (typeId == nodegraphtypes::Model) {
            compileModel(node, execState, payloadRegistry, registry, staleEntities);
        } else if (typeId == nodegraphtypes::Group) {
            compileGroup(node, execState, payloadRegistry, registry, staleEntities);
        } else if (typeId == nodegraphtypes::Transform) {
            compileTransform(node, execState, payloadRegistry, registry, staleEntities);
        } else if (typeId == nodegraphtypes::Remesh) {
            compileRemesh(node, execState, payloadRegistry, registry, staleEntities);
        } else if (typeId == nodegraphtypes::MeshPoints) {
            compileMeshPoints(node, execState, payloadRegistry, registry, staleEntities);
        } else if (typeId == nodegraphtypes::Points) {
            compilePoints(node, execState, payloadRegistry, registry, staleEntities);
        } else if (typeId == nodegraphtypes::Merge) {
            compileMerge(node, execState, payloadRegistry, registry, staleEntities);
        } else if (typeId == nodegraphtypes::Voronoi) {
            compileVoronoi(node, execState, payloadRegistry, registry, staleEntities);
        } else if (typeId == nodegraphtypes::Contact) {
            compileContact(node, execState, payloadRegistry, registry, staleEntities);
        } else if (typeId == nodegraphtypes::HeatSolve) {
            compileHeatSolve(node, execState, payloadRegistry, registry, staleEntities);
        }
    }
}
