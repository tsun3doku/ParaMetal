#include "NodePayloadRegistry.hpp"

#include "NodeGraphPayloadTypes.hpp"
#include "domain/GeometryData.hpp"
#include "domain/HeatModelData.hpp"
#include "domain/PointData.hpp"
#include "domain/RemeshData.hpp"

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

const PointData* NodePayloadRegistry::resolvePoints(const NodeDataHandle& handle) const {
    if (handle.key == 0) {
        return nullptr;
    }
    return get<PointData>(handle);
}

uint64_t NodePayloadRegistry::resolveHash(const NodeDataHandle& handle, HashDomain domain) const {
    if (handle.key == 0) {
        return 0;
    }

    const auto it = entries.find(handle.key);
    return it != entries.end() ? it->second.hashes.get(domain) : 0;
}
