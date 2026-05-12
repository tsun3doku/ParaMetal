#include "RuntimePackageCompiler.hpp"

#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeGraphRuntimeBridge.hpp"
#include "nodegraph/NodeGraphUtils.hpp"
#include "nodegraph/NodeGraphHash.hpp"
#include "nodegraph/NodeContactParams.hpp"
#include "nodegraph/NodeHeatSolveParams.hpp"
#include "nodegraph/NodeRemeshParams.hpp"
#include "nodegraph/NodeVoronoiParams.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/RuntimeHandleResolver.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "domain/HeatModelData.hpp"

#include <iostream>
#include <optional>
#include <set>
#include <unordered_set>

static uint64_t buildModelPackageHash(const ModelPackage& package) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, 1u);
    NodeGraphHash::combine(hash, package.geometry.payloadHash);
    hash = RuntimeProductHash::mixPod(hash, package.localToWorld);
    return hash;
}

static void combineProductHandleHash(uint64_t& hash, const ProductHandle& handle) {
    NodeGraphHash::combine(hash, static_cast<uint64_t>(handle.type));
    NodeGraphHash::combine(hash, handle.outputSocketKey);
}

static uint64_t productContentHashFromHandle(const ECSRegistry& ecsRegistry, const ProductHandle& handle) {
    if (!handle.isValid()) {
        return 0;
    }

    switch (handle.type) {
    case NodeProductType::Model: {
        const ModelProduct* product = tryGetProduct<ModelProduct>(ecsRegistry, handle.outputSocketKey);
        return product ? product->productHash : 0;
    }
    case NodeProductType::Remesh: {
        const RemeshProduct* product = tryGetProduct<RemeshProduct>(ecsRegistry, handle.outputSocketKey);
        return product ? product->productHash : 0;
    }
    case NodeProductType::Voronoi: {
        const VoronoiProduct* product = tryGetProduct<VoronoiProduct>(ecsRegistry, handle.outputSocketKey);
        return product ? product->productHash : 0;
    }
    case NodeProductType::Contact: {
        const ContactProduct* product = tryGetProduct<ContactProduct>(ecsRegistry, handle.outputSocketKey);
        return product ? product->productHash : 0;
    }
    case NodeProductType::Heat: {
        const HeatProduct* product = tryGetProduct<HeatProduct>(ecsRegistry, handle.outputSocketKey);
        return product ? product->productHash : 0;
    }
    default:
        return 0;
    }
}

static void combineProductDependencyHash(uint64_t& hash, const ECSRegistry& ecsRegistry, const ProductHandle& handle) {
    combineProductHandleHash(hash, handle);
    NodeGraphHash::combine(hash, productContentHashFromHandle(ecsRegistry, handle));
}

// Package hashes intentionally combine authored payload state with resolved
// runtime dependency identity. Payload hash answers "did authored graph data
// change?", while ProductHandle identity answers "did an upstream realized
// dependency change?"

static uint64_t buildRemeshPackageHash(const RemeshPackage& package, const ECSRegistry& ecsRegistry) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, 2u);
    NodeGraphHash::combine(hash, package.sourceGeometry.payloadHash);
    combineProductDependencyHash(hash, ecsRegistry, package.modelProductHandle);
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
    NodeGraphHash::combine(hash, static_cast<uint64_t>(package.modelProducts.size()));
    for (const ProductHandle& handle : package.modelProducts) {
        combineProductDependencyHash(hash, ecsRegistry, handle);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(package.modelRemeshProducts.size()));
    for (const ProductHandle& handle : package.modelRemeshProducts) {
        combineProductDependencyHash(hash, ecsRegistry, handle);
    }
    return hash;
}

static uint64_t buildContactPackageHash(const ContactPackage& package, const ECSRegistry& ecsRegistry) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, 4u);
    NodeGraphHash::combine(hash, package.authored.payloadHash);
    combineProductDependencyHash(hash, ecsRegistry, package.modelAModelProduct);
    combineProductDependencyHash(hash, ecsRegistry, package.modelBModelProduct);
    combineProductDependencyHash(hash, ecsRegistry, package.modelARemeshProduct);
    combineProductDependencyHash(hash, ecsRegistry, package.modelBRemeshProduct);
    return hash;
}

static uint64_t buildHeatPackageHash(const HeatPackage& package, const ECSRegistry& ecsRegistry) {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, 5u);
    NodeGraphHash::combine(hash, package.authored.payloadHash);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(package.voronoiProducts.size()));
    for (const ProductHandle& handle : package.voronoiProducts) {
        combineProductDependencyHash(hash, ecsRegistry, handle);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(package.contactProducts.size()));
    for (const ProductHandle& handle : package.contactProducts) {
        combineProductDependencyHash(hash, ecsRegistry, handle);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(package.modelProducts.size()));
    for (const ProductHandle& handle : package.modelProducts) {
        combineProductDependencyHash(hash, ecsRegistry, handle);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(package.remeshProducts.size()));
    for (const ProductHandle& handle : package.remeshProducts) {
        combineProductDependencyHash(hash, ecsRegistry, handle);
    }
    return hash;
}

namespace {

struct OutputValue {
    uint64_t socketKey = 0;
    const NodeDataBlock* block = nullptr;
};

// geometryFromHandle, modelProductFromHandle, remeshProductFromHandle
// live in RuntimeHandleResolver.hpp/.cpp

uint64_t socketKey(NodeGraphNodeId nodeId, NodeGraphSocketId socketId) {
    return (static_cast<uint64_t>(nodeId.value) << 32) | static_cast<uint64_t>(socketId.value);
}

std::vector<ProductHandle> resolveHeatVoronoiInputs(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& evaluationState,
    const ECSRegistry& ecsRegistry) {
    std::vector<ProductHandle> results;
    const NodeGraphSocket* volumeSocket = findInputSocket(node, NodeGraphValueType::Volume);
    if (!volumeSocket) {
        return results;
    }

    const uint64_t inputKey = socketKey(node.id, volumeSocket->id);
    const auto socketsIt = evaluationState.sourceSocketsByInputSocket.find(inputKey);
    if (socketsIt == evaluationState.sourceSocketsByInputSocket.end()) {
        return results;
    }

    for (uint64_t sourceSocketKey : socketsIt->second) {
        const auto outputIt = evaluationState.outputBySocket.find(sourceSocketKey);
        if (outputIt == evaluationState.outputBySocket.end() ||
            outputIt->second.status != EvaluatedSocketStatus::Value) {
            continue;
        }

        const NodeDataBlock& inputBlock = outputIt->second.data;
        if (inputBlock.dataType != NodePayloadType::Voronoi || inputBlock.payloadHandle.key == 0) {
            continue;
        }

        ProductHandle handle = getPublishedHandle<VoronoiProduct>(ecsRegistry, sourceSocketKey);
        if (handle.isValid()) {
            results.push_back(handle);
        }
    }

    return results;
}

std::vector<ProductHandle> resolveHeatContactInputs(
    const NodeGraphNode& node,
    const NodeGraphEvaluationState& evaluationState,
    const ECSRegistry& ecsRegistry) {
    std::vector<ProductHandle> results;
    const NodeGraphSocket* fieldSocket = findInputSocket(node, NodeGraphValueType::Field);
    if (!fieldSocket) {
        return results;
    }

    const uint64_t inputKey = socketKey(node.id, fieldSocket->id);
    const auto socketsIt = evaluationState.sourceSocketsByInputSocket.find(inputKey);
    if (socketsIt == evaluationState.sourceSocketsByInputSocket.end()) {
        return results;
    }

    for (uint64_t sourceSocketKey : socketsIt->second) {
        const auto outputIt = evaluationState.outputBySocket.find(sourceSocketKey);
        if (outputIt == evaluationState.outputBySocket.end() ||
            outputIt->second.status != EvaluatedSocketStatus::Value) {
            continue;
        }

        const NodeDataBlock& inputBlock = outputIt->second.data;
        if (inputBlock.dataType != NodePayloadType::Contact || inputBlock.payloadHandle.key == 0) {
            continue;
        }

        ProductHandle handle = getPublishedHandle<ContactProduct>(ecsRegistry, sourceSocketKey);
        if (handle.isValid()) {
            results.push_back(handle);
        }
    }

    return results;
}

std::optional<OutputValue> resolveOutputValue(
    const NodeGraphNode& node,
    const NodeGraphSocket& output,
    const NodeGraphEvaluationState& evaluationState) {
    const uint64_t outputSocketKey = socketKey(node.id, output.id);
    const auto valueIt = evaluationState.outputBySocket.find(outputSocketKey);
    if (valueIt == evaluationState.outputBySocket.end() ||
        valueIt->second.status != EvaluatedSocketStatus::Value ||
        valueIt->second.data.payloadHandle.key == 0) {
        return std::nullopt;
    }

    return OutputValue{outputSocketKey, &valueIt->second.data};
}

template <typename T>
const T* resolvePayload(
    const NodePayloadRegistry* payloadRegistry,
    const std::optional<OutputValue>& outputValue) {
    if (!payloadRegistry || !outputValue.has_value() || !outputValue->block) {
        return nullptr;
    }

    return payloadRegistry->get<T>(outputValue->block->payloadHandle);
}


}

void RuntimePackageCompiler::setRuntimeBridge(const NodeGraphRuntimeBridge* updatedRuntimeBridge) {
    runtimeBridge = updatedRuntimeBridge;
}

ModelPackage RuntimePackageCompiler::buildModelPackage(
    const GeometryData& geometry) const {
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

    const GeometryData* sourceGeometry = payloadRegistry->resolveGeometryHandle(remesh.sourceMeshHandle);
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
    package.modelProductHandle = modelProductFromHandle(
        payloadRegistry,
        registry,
        remesh.sourceMeshHandle);
    package.packageHash = buildRemeshPackageHash(package, registry);
    return package;
}

VoronoiPackage RuntimePackageCompiler::buildVoronoiPackage(
    const NodeGraphNode& node,
    const NodePayloadRegistry* payloadRegistry,
    const ECSRegistry& registry,
    const VoronoiData& voronoi) const {
    VoronoiPackage package{};
    package.authored = voronoi;

    if (!payloadRegistry || !voronoi.active || voronoi.modelMeshHandles.empty()) {
        return package;
    }

    package.modelProducts.reserve(voronoi.modelMeshHandles.size());
    package.modelRemeshProducts.reserve(voronoi.modelMeshHandles.size());
    package.modelLocalToWorlds.reserve(voronoi.modelMeshHandles.size());

    std::set<NodeDataHandle> seenMeshHandles;
    for (size_t i = 0; i < voronoi.modelMeshHandles.size(); ++i) {
        const NodeDataHandle& meshHandle = voronoi.modelMeshHandles[i];
        
        if (!seenMeshHandles.insert(meshHandle).second) {
            continue;
        }

        const std::optional<GeometryData> modelGeometry = geometryFromHandle(payloadRegistry, meshHandle);
        if (!modelGeometry.has_value()) {
            continue;
        }

        const ProductHandle modelRemeshProduct =
            remeshProductFromHandle(runtimeBridge, payloadRegistry, registry, meshHandle);
        if (!modelRemeshProduct.isValid()) {
            continue;
        }

        const ProductHandle modelProduct = modelProductFromHandle(payloadRegistry, registry, meshHandle);

        package.modelLocalToWorlds.push_back(modelGeometry->localToWorld);
        package.modelProducts.push_back(modelProduct);
        package.modelRemeshProducts.push_back(modelRemeshProduct);
    }

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
    const std::vector<ProductHandle>& voronoiProducts,
    const std::vector<ProductHandle>& contactProducts) const {
    HeatPackage package{};
    package.authored = heat;
    package.voronoiProducts = voronoiProducts;
    package.contactProducts = contactProducts;

    const HeatSolveNodeParams nodeParams = readHeatSolveNodeParams(node);
    package.display.showHeatOverlay = nodeParams.preview.showHeatOverlay;
    package.display.showFluxVectors = nodeParams.preview.showFluxVectors;
    package.display.fluxVectorScale = static_cast<float>(nodeParams.preview.fluxVectorScale);

    if (!payloadRegistry) {
        return package;
    }

    std::unordered_set<uint64_t> seenRemeshSocketKeys;
    std::set<NodeDataHandle> seenHandles;
    package.modelProducts.reserve(heat.heatModelHandles.size());
    package.remeshProducts.reserve(heat.heatModelHandles.size());
    package.models.reserve(heat.heatModelHandles.size());

    for (size_t i = 0; i < heat.heatModelHandles.size(); ++i) {
        const NodeDataHandle& modelHandle = heat.heatModelHandles[i];
        
        if (modelHandle.key == 0) {
            continue;
        }

        const HeatModelData* heatModel = payloadRegistry->get<HeatModelData>(modelHandle);
        if (!heatModel || heatModel->meshHandle.key == 0) {
            continue;
        }
        
        if (!seenHandles.insert(heatModel->meshHandle).second) {
            continue;
        }

        const std::optional<GeometryData> geometry = geometryFromHandle(payloadRegistry, heatModel->meshHandle);
        if (!geometry.has_value()) {
            continue;
        }

        const ProductHandle modelProduct =
            modelProductFromHandle(payloadRegistry, registry, heatModel->meshHandle);
        if (!modelProduct.isValid()) {
            continue;
        }
        
        const ProductHandle remeshProduct =
            remeshProductFromHandle(runtimeBridge, payloadRegistry, registry, heatModel->meshHandle);
        if (!remeshProduct.isValid()) {
            continue;
        }
        if (!seenRemeshSocketKeys.insert(remeshProduct.outputSocketKey).second) {
            continue;
        }

        package.modelProducts.push_back(modelProduct);
        package.remeshProducts.push_back(remeshProduct);

        // Copy HeatModelData and merge material preset
        HeatModelData modelData = *heatModel;

        // Apply material preset if bound
        const uint32_t modelNodeId = static_cast<uint32_t>(modelHandle.key);
        for (const HeatMaterialBinding& binding : heat.materialBindings) {
            if (binding.modelNodeId == modelNodeId) {
                const HeatMaterialPreset& preset = heatMaterialPresetById(binding.presetId);
                modelData.density = preset.density;
                modelData.specificHeat = preset.specificHeat;
                modelData.conductivity = preset.conductivity;
                break;
            }
        }

        package.models.push_back(std::move(modelData));
    }

    package.packageHash = buildHeatPackageHash(package, registry);
    return package;
}

ContactPackage RuntimePackageCompiler::buildContactPackage(
    const NodeGraphNode& node,
    const NodePayloadRegistry* payloadRegistry,
    const ECSRegistry& registry,
    const ContactData& contact) const {
    ContactPackage package{};
    package.authored = contact;
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

    const std::optional<GeometryData> geometryA = geometryFromHandle(payloadRegistry, endpointA.meshHandle);
    const std::optional<GeometryData> geometryB = geometryFromHandle(payloadRegistry, endpointB.meshHandle);
    if (!geometryA.has_value() || !geometryB.has_value()) {
        return package;
    }

    package.modelALocalToWorld = geometryA->localToWorld;
    package.modelBLocalToWorld = geometryB->localToWorld;
    package.modelAModelProduct = modelProductFromHandle(payloadRegistry, registry, endpointA.meshHandle);
    package.modelBModelProduct = modelProductFromHandle(payloadRegistry, registry, endpointB.meshHandle);
    package.modelARemeshProduct = remeshProductFromHandle(runtimeBridge, payloadRegistry, registry, endpointA.meshHandle);
    package.modelBRemeshProduct = remeshProductFromHandle(runtimeBridge, payloadRegistry, registry, endpointB.meshHandle);

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
    staleEntities.erase(entity);
}

void RuntimePackageCompiler::compileAndApply(
    const NodeGraphState& graphState,
    const NodeGraphEvaluationState& evaluationState,
    const NodePayloadRegistry* payloadRegistry,
    ECSRegistry& registry,
    std::unordered_set<ECSEntity>& staleEntities) const {

    for (const NodeGraphNode& node : graphState.nodes) {
        for (const NodeGraphSocket& output : node.outputs) {
            if (output.contract.producedPayloadType == NodePayloadType::None) {
                continue;
            }

            const std::optional<OutputValue> outputValue = resolveOutputValue(node, output, evaluationState);
            if (!outputValue.has_value()) {
                continue;
            }

            switch (output.contract.producedPayloadType) {
            case NodePayloadType::Geometry: {
                if (node.frozen) {
                    auto entity = static_cast<ECSEntity>(outputValue->socketKey);
                    if (registry.valid(entity) && registry.all_of<ModelPackage>(entity)) {
                        staleEntities.erase(entity);
                    }
                    break;
                }

                const GeometryData* geometry = resolvePayload<GeometryData>(payloadRegistry, outputValue);
                if (!geometry) {
                    break;
                }

                ModelPackage package = buildModelPackage(*geometry);
                applyPackage<ModelPackage>(registry, outputValue->socketKey, package, staleEntities);
                break;
            }
            case NodePayloadType::Remesh: {
                if (node.frozen) {
                    auto entity = static_cast<ECSEntity>(outputValue->socketKey);
                    if (registry.valid(entity) && registry.all_of<RemeshPackage>(entity)) {
                        staleEntities.erase(entity);
                    }
                    break;
                }

                const RemeshData* remesh = resolvePayload<RemeshData>(payloadRegistry, outputValue);
                if (!remesh || !remesh->active || remesh->sourceMeshHandle.key == 0) {
                    break;
                }

                RemeshPackage package =
                    buildRemeshPackage(node, *remesh, payloadRegistry, registry, outputValue->block->payloadHandle);
                applyPackage<RemeshPackage>(registry, outputValue->socketKey, package, staleEntities);
                break;
            }
            case NodePayloadType::Voronoi: {
                if (node.frozen) {
                    auto entity = static_cast<ECSEntity>(outputValue->socketKey);
                    if (registry.valid(entity) && registry.all_of<VoronoiPackage>(entity)) {
                        staleEntities.erase(entity);
                    }
                    break;
                }

                const VoronoiData* voronoi = resolvePayload<VoronoiData>(payloadRegistry, outputValue);
                if (!voronoi) {
                    break;
                }

                VoronoiPackage package = buildVoronoiPackage(node, payloadRegistry, registry, *voronoi);

                applyPackage<VoronoiPackage>(registry, outputValue->socketKey, package, staleEntities);
                break;
            }
            case NodePayloadType::Contact: {
                if (node.frozen) {
                    auto entity = static_cast<ECSEntity>(outputValue->socketKey);
                    if (registry.valid(entity) && registry.all_of<ContactPackage>(entity)) {
                        staleEntities.erase(entity);
                    }
                    break;
                }

                const ContactData* contact = resolvePayload<ContactData>(payloadRegistry, outputValue);
                if (!contact) {
                    break;
                }

                ContactPackage package = buildContactPackage(node, payloadRegistry, registry, *contact);
                applyPackage<ContactPackage>(registry, outputValue->socketKey, package, staleEntities);
                break;
            }
            case NodePayloadType::Heat: {
                if (node.frozen) {
                    auto entity = static_cast<ECSEntity>(outputValue->socketKey);
                    if (registry.valid(entity) && registry.all_of<HeatPackage>(entity)) {
                        staleEntities.erase(entity);
                    }
                    break;
                }

                const HeatData* heat = resolvePayload<HeatData>(payloadRegistry, outputValue);
                if (!heat) {
                    break;
                }

                const std::vector<ProductHandle> voronoiProducts =
                    resolveHeatVoronoiInputs(node, evaluationState, registry);
                const std::vector<ProductHandle> contactProducts =
                    resolveHeatContactInputs(node, evaluationState, registry);
                HeatPackage package = buildHeatPackage(
                    node,
                    payloadRegistry,
                    registry,
                    *heat,
                    voronoiProducts,
                    contactProducts);
                applyPackage<HeatPackage>(registry, outputValue->socketKey, package, staleEntities);
                break;
            }
            default:
                break;
            }
        }
    }
}
