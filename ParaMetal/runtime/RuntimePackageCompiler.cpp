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

#include <iostream>
#include <unordered_map>
#include <unordered_set>

HashValues RuntimePackageCompiler::resolveHandleHashes(const NodePayloadRegistry* payloadRegistry, const NodeDataHandle& handle) const {
    if (!payloadRegistry || handle.key == 0) return {};
    HashValues values{};
    values.full = payloadRegistry->resolveHash(handle, HashDomain::Full);
    values.geometry = payloadRegistry->resolveHash(handle, HashDomain::Geometry);
    values.thermal = payloadRegistry->resolveHash(handle, HashDomain::Thermal);
    values.simulation = payloadRegistry->resolveHash(handle, HashDomain::Simulation);
    values.display = payloadRegistry->resolveHash(handle, HashDomain::Display);
    return values;
}

ModelPackage RuntimePackageCompiler::buildModelPackage(const GeometryData& geometry, const HashValues& geometryHashes) const {
    ModelPackage package{};
    package.geometry = geometry;
    HashPackage::seal(package, geometryHashes);
    return package;
}

RemeshPackage RuntimePackageCompiler::buildRemeshPackage(
    const NodeGraphNode& node,
    const RemeshData& remesh,
    const HashValues& sourceGeometryHashes,
    const NodePayloadRegistry* payloadRegistry,
    const ProductHandle& sourceModelProduct,
    const NodeDataHandle& remeshHandle) const {
    RemeshPackage package{};
    if (!payloadRegistry || !remesh.active || remesh.sourceMeshHandle.key == 0) {
        return package;
    }

    const GeometryData* sourceGeometry = payloadRegistry->resolveGeometry(remesh.sourceMeshHandle);
    if (!sourceGeometry) {
        return package;
    }
    if (sourceGeometry->pointPositions.empty() || sourceGeometry->triangleIndices.empty()) {
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
    package.sourceModelProduct = sourceModelProduct;
    if (!package.sourceModelProduct.isValid()) {
        return {};
    }
    HashPackage::seal(package, sourceGeometryHashes);
    return package;
}

VoronoiPackage RuntimePackageCompiler::buildVoronoiPackage(
    const NodeGraphNode& node,
    const NodePayloadRegistry* payloadRegistry,
    const VoronoiData& voronoi,
    const ProductHandle& modelProduct,
    const ProductHandle& remeshProduct,
    const HashValues& authoredHashes,
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
        package.modelProduct = modelProduct;
        package.remeshProduct = remeshProduct;
        if (!package.modelProduct.isValid() || !package.remeshProduct.isValid()) {
            return package;
        }
    } else if (voronoi.domainType == DomainType::Points) {
        if (voronoi.pointsPayloadHandle.key == 0) {
            return package;
        }
        package.pointsPayloadHandle = voronoi.pointsPayloadHandle;
    }

    const VoronoiNodeParams nodeParams = readVoronoiNodeParams(node);
    package.display.showVoronoi = nodeParams.preview.showVoronoi;
    package.display.showPoints = nodeParams.preview.showPoints;
    HashPackage::seal(package, authoredHashes);
    return package;
}

HeatPackage RuntimePackageCompiler::buildHeatPackage(
    const NodeGraphNode& node,
    const NodePayloadRegistry* payloadRegistry,
    const RuntimePackageManager& packages,
    const NodeGraphEvaluationState& execState,
    const HeatData& heat,
    const HashValues& authoredHashes,
    const NodeDataHandle& heatHandle) const {
    HeatPackage package{};
    package.authored = heat;
    package.heatHandle = heatHandle;

    const HeatSolveNodeParams nodeParams = readHeatSolveNodeParams(node);
    package.display.showHeatOverlay = nodeParams.preview.showHeatOverlay;
    package.display.showFluxVectors = nodeParams.preview.showFluxVectors;
    package.display.showHeatPalette = nodeParams.preview.showHeatPalette;
    package.display.fluxVectorScale = static_cast<float>(nodeParams.preview.fluxVectorScale);

    const auto invalidatePackage = [&package, &authoredHashes]() {
        package.authored.active = false;
        HashPackage::seal(package, authoredHashes);
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
            return {};
        }

        auto it = heatModelByRemeshKey.find(voronoi->modelMeshHandle.key);
        if (it == heatModelByRemeshKey.end()) {
            return {};
        }
        usedHeatModelRemeshKeys.insert(voronoi->modelMeshHandle.key);

        const HeatModelData* heatModel = it->second;
        NodeDataHandle modelHandle;
        const GeometryData* geometry = payloadRegistry->resolveGeometry(heatModel->meshHandle, &modelHandle);
        if (!geometry || modelHandle.key == 0) {
            return {};
        }

        ProductHandle remeshProduct = execState.productFor(heatModel->meshHandle.key, NodeProductType::Remesh);
        ProductHandle modelProduct = execState.productFor(modelHandle.key, NodeProductType::Model);
        if (!remeshProduct.isValid() || !modelProduct.isValid()) {
            return {};
        }

        const VoronoiPackage* voronoiPackage = packages.findAny<VoronoiPackage>(voronoiHandle.key);
        if (voronoiPackage && voronoiPackage->remeshProduct.isValid() &&
            !(voronoiPackage->remeshProduct == remeshProduct)) {
            return {};
        }

        package.remeshProducts.push_back(remeshProduct);
        package.modelProducts.push_back(modelProduct);
        package.resolvedDensity.push_back(heatModel->density);
        package.resolvedSpecificHeat.push_back(heatModel->specificHeat);
        package.resolvedConductivity.push_back(heatModel->conductivity);
        package.resolvedInitialTemperaturesC.push_back(heatModel->initialTemperatureC);
        package.resolvedBoundaryConditionTypes.push_back(static_cast<uint32_t>(heatModel->boundaryCondition.type));
        package.resolvedBoundaryTemperaturesC.push_back(heatModel->boundaryCondition.temperatureC);
        package.resolvedBoundaryHeatFluxes.push_back(heatModel->boundaryCondition.heatFlux);
        package.resolvedBoundaryHeatTransferCoefficients.push_back(heatModel->boundaryCondition.heatTransferCoefficient);
        package.resolvedVolumetricPowerDensities.push_back(heatModel->volumetricHeatSource.powerDensity);
    }

    if (usedHeatModelRemeshKeys.size() != heatModelByRemeshKey.size()) {
        return invalidatePackage();
    }

    for (const NodeDataHandle& voronoiHandle : heat.voronoiHandles) {
        ProductHandle voronoiProduct = execState.productFor(voronoiHandle.key, NodeProductType::Voronoi);
        if (!voronoiProduct.isValid()) {
            return {};
        }
        package.voronoiProducts.push_back(voronoiProduct);
    }
    for (const NodeDataHandle& contactHandle : heat.contactHandles) {
        ProductHandle contactProduct = execState.productFor(contactHandle.key, NodeProductType::Contact);
        if (!contactProduct.isValid()) {
            return {};
        }

        const ContactPackage* contactPackage = packages.findAny<ContactPackage>(contactHandle.key);
        if (contactPackage) {
            bool hasA = false;
            bool hasB = false;
            for (const ProductHandle& remeshProductHandle : package.remeshProducts) {
                if (remeshProductHandle == contactPackage->modelARemeshProduct) {
                    hasA = true;
                }
                if (remeshProductHandle == contactPackage->modelBRemeshProduct) {
                    hasB = true;
                }
            }
            if (!hasA || !hasB) {
                return {};
            }
        }
        package.contactProducts.push_back(contactProduct);
    }

    HashPackage::seal(package, authoredHashes);
    return package;
}

PointPackage RuntimePackageCompiler::buildPointPackage(
    const NodePayloadRegistry* payloadRegistry,
    const PointData& pointData,
    const NodeDataHandle& pointHandle) const {
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
    const ContactData& contact,
    const ProductHandle& modelARemeshProduct,
    const ProductHandle& modelBRemeshProduct,
    const HashValues& authoredHashes,
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
    package.modelARemeshProduct = modelARemeshProduct;
    package.modelBRemeshProduct = modelBRemeshProduct;
    if (!package.modelARemeshProduct.isValid() || !package.modelBRemeshProduct.isValid()) {
        return package;
    }

    HashPackage::seal(package, authoredHashes);
    return package;
}

void RuntimePackageCompiler::compileMeshPoints(
    const NodeGraphState& graphState,
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    RuntimeProductManager& products,
    RuntimePackageManager& packages) const {

    if (!payloadRegistry || node.outputs.empty() || node.inputs.empty()) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        const EvaluatedSocketValue* value = execState.valueFor(outSocketKey);
        if (!value ||
            value->status != EvaluatedSocketStatus::Value ||
            value->data.payloadHandle.key == 0) {
            continue;
        }

        if (value->data.isFrozen) {
            packages.retain(outSocketKey);
            continue;
        }

        const NodeDataBlock& data = value->data;
        const PointData* pointData = payloadRegistry->get<PointData>(data.payloadHandle);
        if (!pointData || !pointData->active) {
            continue
                ;
        }

        PointPackage package{};
        package.productHandle = execState.productFor(outSocketKey, NodeProductType::Point);
        package.pointsPayloadHandle = data.payloadHandle;
        package.localToWorld = pointData->localToWorld;

        bool usedRemesh = false;
        const NodeSocketKey upstreamKey = graphState.edges.upstream(node.id, node.inputs[0].id);
        if (upstreamKey.value != 0) {
            const ProductHandle remeshProductHandle = execState.productFor(upstreamKey.value, NodeProductType::Remesh);
            const RemeshProduct* product = products.resolve<RemeshProduct>(remeshProductHandle);
            if (product && product->isValid()) {
                package.positions.reserve(product->surfacePositions.size());
                for (const glm::vec3& position : product->surfacePositions) {
                    package.positions.push_back(glm::vec4(position, 1.0f));
                }
                package.pointCount = static_cast<uint32_t>(package.positions.size());
                usedRemesh = true;
            }
        }

        if (!usedRemesh) {
            package.positions = pointData->positions;
            package.pointCount = static_cast<uint32_t>(pointData->positions.size());
        }

        HashPackage::seal(package);
        packages.apply<PointPackage>(outSocketKey, package);
    }
}

void RuntimePackageCompiler::compilePoints(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    RuntimeProductManager& products,
    RuntimePackageManager& packages) const {
    (void)products;

    if (!payloadRegistry || node.outputs.empty()) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        const EvaluatedSocketValue* value = execState.valueFor(outSocketKey);
        if (!value ||
            value->status != EvaluatedSocketStatus::Value ||
            value->data.payloadHandle.key == 0) {
            continue;
        }

        if (value->data.isFrozen) {
            packages.retain(outSocketKey);
            continue;
        }

        const NodeDataBlock& data = value->data;
        const PointData* pointData = payloadRegistry->get<PointData>(data.payloadHandle);
        if (!pointData) {
            continue;
        }

        PointPackage package = buildPointPackage(payloadRegistry, *pointData, data.payloadHandle);
        package.productHandle = execState.productFor(outSocketKey, NodeProductType::Point);
        packages.apply<PointPackage>(outSocketKey, package);
    }
}

void RuntimePackageCompiler::compileMerge(
    const NodeGraphState& graphState,
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    RuntimeProductManager& products,
    RuntimePackageManager& packages) const {
    (void)products;

    if (node.outputs.empty() || node.inputs.empty()) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        const EvaluatedSocketValue* value = execState.valueFor(outSocketKey);
        if (!value ||
            value->status != EvaluatedSocketStatus::Value ||
            value->data.payloadHandle.key == 0) {
            continue;
        }

        if (value->data.isFrozen) {
            packages.retain(outSocketKey);
            continue;
        }

        const NodeDataBlock& data = value->data;
        const uint8_t payloadType = data.dataType;

        if (payloadType == payloadtypes::Geometry) {
            const GeometryData* geometry = payloadRegistry->get<GeometryData>(data.payloadHandle);
            if (!geometry) {
                continue;
            }
            ModelPackage package = buildModelPackage(*geometry, data.hashes);
            package.productHandle = execState.productFor(outSocketKey, NodeProductType::Model);
            packages.apply<ModelPackage>(outSocketKey, package);
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

            const std::vector<NodeSocketKey> upstreamKeys = graphState.edges.upstreams(node.id, inSocket->id);
            if (upstreamKeys.empty()) {
                continue;
            }

            std::vector<const PointPackage*> upstreamPackages;
            std::vector<glm::mat4> upstreamL2W;
            for (NodeSocketKey upKey : upstreamKeys) {
                const PointPackage* upPkg = packages.findAny<PointPackage>(upKey.value);
                if (!upPkg) {
                    continue;
                }
                if (upPkg->positions.empty()) {
                    continue;
                }
                upstreamPackages.push_back(upPkg);
                upstreamL2W.push_back(toMat4(upPkg->localToWorld));
            }

            if (upstreamPackages.empty()) {
                continue;
            }

            const glm::mat4 refL2W = upstreamL2W[0];
            const glm::mat4 invRefL2W = glm::inverse(refL2W);

            PointPackage package{};
            package.productHandle = execState.productFor(outSocketKey, NodeProductType::Point);
            package.pointsPayloadHandle = data.payloadHandle;
            package.localToWorld = upstreamPackages[0]->localToWorld;
            for (size_t i = 0; i < upstreamPackages.size(); ++i) {
                const glm::mat4 upToRef = invRefL2W * upstreamL2W[i];
                for (const auto& pos : upstreamPackages[i]->positions) {
                    package.positions.push_back(upToRef * pos);
                }
            }
            package.pointCount = static_cast<uint32_t>(package.positions.size());
            HashPackage::seal(package);
            packages.apply<PointPackage>(outSocketKey, package);
        }
    }
}

void RuntimePackageCompiler::compileTransform(
    const NodeGraphState& graphState,
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    RuntimeProductManager& products,
    RuntimePackageManager& packages) const {
    (void)products;

    if (node.outputs.empty() || node.inputs.empty()) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        const EvaluatedSocketValue* value = execState.valueFor(outSocketKey);
        if (!value ||
            value->status != EvaluatedSocketStatus::Value ||
            value->data.payloadHandle.key == 0) {
            continue;
        }

        if (value->data.isFrozen) {
            packages.retain(outSocketKey);
            continue;
        }

        const NodeDataBlock& data = value->data;
        const uint8_t payloadType = data.dataType;

        if (payloadType == payloadtypes::Geometry) {
            const GeometryData* geometry = payloadRegistry->get<GeometryData>(data.payloadHandle);
            if (!geometry) {
                continue;
            }

            ModelPackage package = buildModelPackage(*geometry, data.hashes);
            package.productHandle = execState.productFor(outSocketKey, NodeProductType::Model);
            packages.apply<ModelPackage>(outSocketKey, package);
        } else if (payloadType == payloadtypes::Points) {
            const NodeSocketKey upstreamKey = graphState.edges.upstream(node.id, node.inputs[0].id);
            if (upstreamKey.value == 0) {
                continue;
            }

            const PointPackage* upPkg = packages.findAny<PointPackage>(upstreamKey.value);
            if (!upPkg) {
                continue;
            }
            if (upPkg->positions.empty()) {
                continue;
            }

            PointPackage package{};
            package.productHandle = execState.productFor(outSocketKey, NodeProductType::Point);
            package.pointsPayloadHandle = data.payloadHandle;
            package.positions = upPkg->positions;
            package.pointCount = upPkg->pointCount;
            const std::array<float, 16> localTransform = NodeTransform::buildLocalTransformArray(node);
            package.localToWorld = toMatrixArray(toMat4(upPkg->localToWorld) * toMat4(localTransform));
            HashPackage::seal(package);
            packages.apply<PointPackage>(outSocketKey, package);
        }
    }
}

void RuntimePackageCompiler::compileModel(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    RuntimeProductManager& products,
    RuntimePackageManager& packages) const {
    (void)products;

    if (!payloadRegistry) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        const EvaluatedSocketValue* value = execState.valueFor(outSocketKey);
        if (!value ||
            value->status != EvaluatedSocketStatus::Value ||
            value->data.payloadHandle.key == 0) {
            continue;
        }

        if (value->data.isFrozen) {
            packages.retain(outSocketKey);
            continue;
        }

        const NodeDataBlock& data = value->data;
        const GeometryData* geometry = payloadRegistry->get<GeometryData>(data.payloadHandle);
        if (!geometry) {
            continue;
        }

        ModelPackage package = buildModelPackage(*geometry, data.hashes);
        package.productHandle = execState.productFor(outSocketKey, NodeProductType::Model);
        packages.apply<ModelPackage>(outSocketKey, package);
    }
}

void RuntimePackageCompiler::compileGroup(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    RuntimeProductManager& products,
    RuntimePackageManager& packages) const {
    (void)products;
    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        const EvaluatedSocketValue* value = execState.valueFor(outSocketKey);
        if (!value ||
            value->status != EvaluatedSocketStatus::Value ||
            value->data.payloadHandle.key == 0) {
            continue;
        }

        if (value->data.isFrozen) {
            packages.retain(outSocketKey);
            continue;
        }

        const NodeDataBlock& data = value->data;
        const GeometryData* geometry = payloadRegistry->get<GeometryData>(data.payloadHandle);
        if (!geometry) {
            continue;
        }

        ModelPackage package = buildModelPackage(*geometry, data.hashes);
        package.productHandle = execState.productFor(outSocketKey, NodeProductType::Model);
        packages.apply<ModelPackage>(outSocketKey, package);
    }
}

void RuntimePackageCompiler::compileRemesh(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    RuntimeProductManager& products,
    RuntimePackageManager& packages) const {

    if (!payloadRegistry) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        const EvaluatedSocketValue* value = execState.valueFor(outSocketKey);
        if (!value ||
            value->status != EvaluatedSocketStatus::Value ||
            value->data.payloadHandle.key == 0) {
            continue;
        }

        if (value->data.isFrozen) {
            packages.retain(outSocketKey);
            continue;
        }

        const NodeDataBlock& data = value->data;
        const RemeshData* remesh = payloadRegistry->get<RemeshData>(data.payloadHandle);
        if (!remesh || !remesh->active || remesh->sourceMeshHandle.key == 0) {
            continue;
        }

        ProductHandle sourceModelProduct = execState.productFor(remesh->sourceMeshHandle.key, NodeProductType::Model);
        if (!sourceModelProduct.isValid()) {
            packages.retain(outSocketKey);
            continue;
        }

        const HashValues sourceGeometryHashes = resolveHandleHashes(payloadRegistry, remesh->sourceMeshHandle);

        RemeshPackage package = buildRemeshPackage(node, *remesh, sourceGeometryHashes, payloadRegistry, sourceModelProduct, data.payloadHandle);
        package.productHandle = execState.productFor(outSocketKey, NodeProductType::Remesh);
        packages.apply<RemeshPackage>(outSocketKey, package);
    }
}

void RuntimePackageCompiler::compileVoronoi(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    RuntimeProductManager& products,
    RuntimePackageManager& packages) const {

    if (!payloadRegistry) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        const EvaluatedSocketValue* value = execState.valueFor(outSocketKey);
        if (!value ||
            value->status != EvaluatedSocketStatus::Value ||
            value->data.payloadHandle.key == 0) {
            continue;
        }

        if (value->data.isFrozen) {
            packages.retain(outSocketKey);
            continue;
        }

        const NodeDataBlock& data = value->data;
        const VoronoiData* voronoi = payloadRegistry->get<VoronoiData>(data.payloadHandle);
        if (!voronoi) {
            continue;
        }

        ProductHandle modelProduct{};
        ProductHandle remeshProduct{};
        if (voronoi->domainType == DomainType::Mesh) {
            NodeDataHandle modelHandle;
            (void)payloadRegistry->resolveGeometry(voronoi->modelMeshHandle, &modelHandle);
            modelProduct = execState.productFor(modelHandle.key, NodeProductType::Model);
            remeshProduct = execState.productFor(voronoi->modelMeshHandle.key, NodeProductType::Remesh);
            if (!modelProduct.isValid() || !remeshProduct.isValid()) {
                packages.retain(outSocketKey);
                continue;
            }
        }

        VoronoiPackage package = buildVoronoiPackage(node, payloadRegistry, *voronoi, modelProduct, remeshProduct, data.hashes, data.payloadHandle);
        package.productHandle = execState.productFor(outSocketKey, NodeProductType::Voronoi);

        // Resolve point positions from PointPackage
        if (package.pointsPayloadHandle.key != 0) {
            const PointPackage* pointPackage = packages.findAny<PointPackage>(package.pointsPayloadHandle.key);
            if (pointPackage && !pointPackage->positions.empty()) {
                package.pointPositions = pointPackage->positions;
                if (package.domainType == DomainType::Mesh) {
                    const glm::mat4 pointsToMeshLocal = glm::inverse(toMat4(package.modelLocalToWorld)) * toMat4(pointPackage->localToWorld);
                    for (glm::vec4& pos : package.pointPositions) {
                        pos = pointsToMeshLocal * pos;
                    }
                }
            }
        }

        HashPackage::seal(package, data.hashes);
        packages.apply<VoronoiPackage>(outSocketKey, package);
    }
}

void RuntimePackageCompiler::compileContact(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    RuntimeProductManager& products,
    RuntimePackageManager& packages) const {

    if (!payloadRegistry) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        const EvaluatedSocketValue* value = execState.valueFor(outSocketKey);
        if (!value ||
            value->status != EvaluatedSocketStatus::Value ||
            value->data.payloadHandle.key == 0) {
            continue;
        }

        if (value->data.isFrozen) {
            packages.retain(outSocketKey);
            continue;
        }

        const NodeDataBlock& data = value->data;
        const ContactData* contact = payloadRegistry->get<ContactData>(data.payloadHandle);
        if (!contact) {
            continue;
        }

        ProductHandle modelARemeshProduct = execState.productFor(contact->pair.endpointA.meshHandle.key, NodeProductType::Remesh);
        ProductHandle modelBRemeshProduct = execState.productFor(contact->pair.endpointB.meshHandle.key, NodeProductType::Remesh);
        if (!modelARemeshProduct.isValid() || !modelBRemeshProduct.isValid()) {
            packages.retain(outSocketKey);
            continue;
        }

        ContactPackage package = buildContactPackage(
            node,
            payloadRegistry,
            *contact,
            modelARemeshProduct,
            modelBRemeshProduct,
            data.hashes,
            data.payloadHandle);
        package.productHandle = execState.productFor(outSocketKey, NodeProductType::Contact);
        packages.apply<ContactPackage>(outSocketKey, package);
    }
}

void RuntimePackageCompiler::compileHeatSolve(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    RuntimeProductManager& products,
    RuntimePackageManager& packages) const {

    if (!payloadRegistry) {
        return;
    }

    for (const NodeGraphSocket& output : node.outputs) {
        const uint64_t outSocketKey = NodeSocketKey(node.id, output.id).value;

        const EvaluatedSocketValue* value = execState.valueFor(outSocketKey);
        if (!value ||
            value->status != EvaluatedSocketStatus::Value ||
            value->data.payloadHandle.key == 0) {
            continue;
        }

        if (value->data.isFrozen) {
            packages.retain(outSocketKey);
            continue;
        }

        const NodeDataBlock& data = value->data;
        const HeatData* heat = payloadRegistry->get<HeatData>(data.payloadHandle);
        if (!heat) {
            continue;
        }

        HeatPackage package = buildHeatPackage(
            node,
            payloadRegistry,
            packages,
            execState,
            *heat,
            data.hashes,
            data.payloadHandle);
        if (package.hashes.full == 0) {
            packages.retain(outSocketKey);
            continue;
        }
        package.productHandle = execState.productFor(outSocketKey, NodeProductType::Heat);
        packages.apply<HeatPackage>(outSocketKey, package);
    }
}

void RuntimePackageCompiler::compileNode(
    const NodeGraphState& graphState,
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& execState,
    const NodePayloadRegistry* payloadRegistry,
    RuntimeProductManager& products,
    RuntimePackageManager& packages) const {

    const NodeTypeId typeId = getNodeTypeId(node.typeId);

    if (typeId == nodegraphtypes::Model) {
        compileModel(node, execState, payloadRegistry, products, packages);
    } else if (typeId == nodegraphtypes::Group) {
        compileGroup(node, execState, payloadRegistry, products, packages);
    } else if (typeId == nodegraphtypes::Transform) {
        compileTransform(graphState, node, execState, payloadRegistry, products, packages);
    } else if (typeId == nodegraphtypes::Remesh) {
        compileRemesh(node, execState, payloadRegistry, products, packages);
    } else if (typeId == nodegraphtypes::MeshPoints) {
        compileMeshPoints(graphState, node, execState, payloadRegistry, products, packages);
    } else if (typeId == nodegraphtypes::Points) {
        compilePoints(node, execState, payloadRegistry, products, packages);
    } else if (typeId == nodegraphtypes::Merge) {
        compileMerge(graphState, node, execState, payloadRegistry, products, packages);
    } else if (typeId == nodegraphtypes::Voronoi) {
        compileVoronoi(node, execState, payloadRegistry, products, packages);
    } else if (typeId == nodegraphtypes::Contact) {
        compileContact(node, execState, payloadRegistry, products, packages);
    } else if (typeId == nodegraphtypes::HeatSolve) {
        compileHeatSolve(node, execState, payloadRegistry, products, packages);
    }
}
