#include "NodePayloadRegistry.hpp"

#include "NodeGraphHash.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "domain/ContactData.hpp"
#include "domain/GeometryData.hpp"
#include "domain/HeatData.hpp"
#include "domain/HeatModelData.hpp"
#include "domain/RemeshData.hpp"
#include "domain/VoronoiData.hpp"

void GeometryData::sealPayload() {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combineString(hash, baseModelPath);
    for (float value : localToWorld) {
        NodeGraphHash::combineFloat(hash, value);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(pointPositions.size()));
    for (float value : pointPositions) {
        NodeGraphHash::combineFloat(hash, value);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(triangleIndices.size()));
    for (uint32_t value : triangleIndices) {
        NodeGraphHash::combine(hash, static_cast<uint64_t>(value));
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(triangleGroupIds.size()));
    for (uint32_t value : triangleGroupIds) {
        NodeGraphHash::combine(hash, static_cast<uint64_t>(value));
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(groups.size()));
    for (const GeometryGroup& group : groups) {
        NodeGraphHash::combine(hash, static_cast<uint64_t>(group.id));
        NodeGraphHash::combineString(hash, group.name);
        NodeGraphHash::combineString(hash, group.source);
    }
    payloadHash = hash;
}

void RemeshData::sealPayload() {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, static_cast<uint64_t>(active ? 1u : 0u));
    NodeGraphHash::combine(hash, sourcePayloadHash);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(iterations));
    NodeGraphHash::combineFloat(hash, minAngleDegrees);
    NodeGraphHash::combineFloat(hash, maxEdgeLength);
    NodeGraphHash::combineFloat(hash, stepSize);
    payloadHash = hash;
}


void HeatModelData::sealPayload() {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, meshPayloadHash);
    NodeGraphHash::combineFloat(hash, density);
    NodeGraphHash::combineFloat(hash, specificHeat);
    NodeGraphHash::combineFloat(hash, conductivity);
    NodeGraphHash::combineFloat(hash, initialTemperature);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(boundaryCondition));
    NodeGraphHash::combineFloat(hash, fixedTemperatureValue);
    payloadHash = hash;
}

void HeatData::sealPayload() {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, voronoiPayloadHash);
    NodeGraphHash::combine(hash, contactPayloadHash);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(voronoiHandles.size()));
    for (const NodeDataHandle& handle : voronoiHandles) {
        NodeGraphHash::combine(hash, handle.key);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(contactHandles.size()));
    for (const NodeDataHandle& handle : contactHandles) {
        NodeGraphHash::combine(hash, handle.key);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(active ? 1u : 0u));
    NodeGraphHash::combine(hash, static_cast<uint64_t>(paused ? 1u : 0u));
    NodeGraphHash::combine(hash, static_cast<uint64_t>(resetRequested ? 1u : 0u));
    NodeGraphHash::combineFloat(hash, contactThermalConductance);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(heatModelHandles.size()));
    for (const NodeDataHandle& handle : heatModelHandles) {
        NodeGraphHash::combine(hash, handle.key);
    }
    NodeGraphHash::combine(hash, static_cast<uint64_t>(materialBindings.size()));
    for (const HeatMaterialBinding& binding : materialBindings) {
        NodeGraphHash::combine(hash, static_cast<uint64_t>(binding.modelNodeId));
        NodeGraphHash::combine(hash, static_cast<uint64_t>(binding.presetId));
    }
    payloadHash = hash;
}

void ContactData::sealPayload() {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, static_cast<uint64_t>(active ? 1u : 0u));
    NodeGraphHash::combine(hash, emitterPayloadHash);
    NodeGraphHash::combine(hash, receiverPayloadHash);
    NodeGraphHash::combineFloat(hash, pair.minNormalDot);
    NodeGraphHash::combineFloat(hash, pair.contactRadius);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(pair.hasValidContact ? 1u : 0u));
    payloadHash = hash;
}

void VoronoiData::sealPayload() {
    uint64_t hash = NodeGraphHash::start();
    NodeGraphHash::combine(hash, static_cast<uint64_t>(active ? 1u : 0u));
    NodeGraphHash::combineFloat(hash, cellSize);
    NodeGraphHash::combine(hash, static_cast<uint64_t>(voxelResolution));
    NodeGraphHash::combine(hash, static_cast<uint64_t>(modelPayloadHashes.size()));
    for (uint64_t modelPayloadHash : modelPayloadHashes) {
        NodeGraphHash::combine(hash, modelPayloadHash);
    }
    payloadHash = hash;
}

void NodePayloadRegistry::erase(uint64_t key) {
    entries.erase(key);
}

void NodePayloadRegistry::clear() {
    entries.clear();
}

NodeDataHandle NodePayloadRegistry::resolveMeshHandle(uint8_t type, const NodeDataHandle& handle) const {
    if (handle.key == 0) {
        return {};
    }

    if (type == payloadtypes::Geometry || type == payloadtypes::Remesh) {
        return handle;
    }
    if (type == payloadtypes::HeatModel) {
        const HeatModelData* heatModel = get<HeatModelData>(handle);
        return heatModel ? heatModel->meshHandle : NodeDataHandle{};
    }
    return {};
}

const GeometryData* NodePayloadRegistry::resolveGeometry(const NodeDataHandle& handle, NodeDataHandle* outSourceHandle) const {
    if (handle.key == 0) {
        return nullptr;
    }

    if (const GeometryData* g = get<GeometryData>(handle)) {
        if (outSourceHandle) *outSourceHandle = handle;
        return g;
    }
    if (const RemeshData* r = get<RemeshData>(handle)) {
        if (outSourceHandle) *outSourceHandle = r->sourceMeshHandle;
        return get<GeometryData>(r->sourceMeshHandle);
    }
    if (const HeatModelData* h = get<HeatModelData>(handle)) {
        return resolveGeometry(h->meshHandle, outSourceHandle);
    }
    return nullptr;
}

uint64_t NodePayloadRegistry::resolvePayloadHash(const NodeDataHandle& handle) const {
    if (handle.key == 0) {
        return 0;
    }

    if (const auto* p = get<GeometryData>(handle)) return p->payloadHash;
    if (const auto* p = get<RemeshData>(handle)) return p->payloadHash;
    if (const auto* p = get<HeatModelData>(handle)) return p->payloadHash;
    if (const auto* p = get<ContactData>(handle)) return p->payloadHash;
    if (const auto* p = get<HeatData>(handle)) return p->payloadHash;
    if (const auto* p = get<VoronoiData>(handle)) return p->payloadHash;
    return 0;
}
