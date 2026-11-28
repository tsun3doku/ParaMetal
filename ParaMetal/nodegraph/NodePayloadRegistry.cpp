#include "NodePayloadRegistry.hpp"

#include "domain/CFDData.hpp"
#include "domain/GeometryData.hpp"
#include "domain/GroupData.hpp"
#include "domain/HeatData.hpp"
#include "domain/HeatModelData.hpp"
#include "domain/PointData.hpp"
#include "domain/RemeshData.hpp"
#include "domain/TransformData.hpp"
#include "domain/VoronoiData.hpp"

#include <utility>

const NodePayloadRegistry::Entry* NodePayloadRegistry::findEntry(NodeDataHandle handle) const {
    const auto it = entries.find(handle.key);
    if (it == entries.end()) {
        return nullptr;
    }
    if (it->second.current.has_value() &&
        it->second.current->generation == handle.generation) {
        return &it->second.current.value();
    }
    if (it->second.retiring.has_value() &&
        it->second.retiring->generation == handle.generation) {
        return &it->second.retiring.value();
    }
    return nullptr;
}

void NodePayloadRegistry::setLiveRoots(std::vector<NodeDataHandle> roots) {
    liveRoots = std::move(roots);
}

void NodePayloadRegistry::collectDependencies(
    const Entry& entry,
    std::vector<NodeDataHandle>& outDependencies) const {
    const NodeDataHandle handle{entry.key, entry.generation};
    if (const GroupData* group = get<GroupData>(handle)) {
        outDependencies.push_back(group->sourceHandle);
    }
    if (const TransformData* transform = get<TransformData>(handle)) {
        outDependencies.push_back(transform->sourceHandle);
    }
    if (const PointData* points = get<PointData>(handle)) {
        outDependencies.insert(
            outDependencies.end(),
            points->sources.begin(),
            points->sources.end());
    }
    if (const RemeshData* remesh = get<RemeshData>(handle)) {
        outDependencies.push_back(remesh->sourceMeshHandle);
    }
    if (const HeatModelData* heatModel = get<HeatModelData>(handle)) {
        outDependencies.push_back(heatModel->meshHandle);
    }
    if (const HeatData* heat = get<HeatData>(handle)) {
        outDependencies.push_back(heat->domainVoronoiHandle);
        outDependencies.insert(
            outDependencies.end(),
            heat->heatModelHandles.begin(),
            heat->heatModelHandles.end());
        outDependencies.push_back(heat->heatFluidHandle);
        outDependencies.push_back(heat->cfdHandle);
        outDependencies.push_back(heat->boundaryHandle);
    }
    if (const CFDData* cfd = get<CFDData>(handle)) {
        outDependencies.push_back(cfd->domainVoronoiHandle);
        outDependencies.push_back(cfd->fluidHandle);
        outDependencies.push_back(cfd->boundaryHandle);
    }
    if (const VoronoiData* voronoi = get<VoronoiData>(handle)) {
        outDependencies.push_back(voronoi->pointsPayloadHandle);
        outDependencies.insert(
            outDependencies.end(),
            voronoi->modelMeshHandles.begin(),
            voronoi->modelMeshHandles.end());
    }
}

void NodePayloadRegistry::collectReachable(
    const NodeDataHandle& handle,
    std::unordered_set<const Entry*>& reachable) const {
    std::vector<NodeDataHandle> pending{handle};
    while (!pending.empty()) {
        const NodeDataHandle current = pending.back();
        pending.pop_back();
        if (current.key == 0) {
            continue;
        }
        const Entry* entry = findEntry(current);
        if (!entry || !reachable.insert(entry).second) {
            continue;
        }
        std::vector<NodeDataHandle> dependencies;
        collectDependencies(*entry, dependencies);
        pending.insert(pending.end(), dependencies.begin(), dependencies.end());
    }
}

void NodePayloadRegistry::reclaimUnreachable() {
    std::unordered_set<const Entry*> reachable;
    reachable.reserve(entries.size() * 2);
    for (const NodeDataHandle& root : liveRoots) {
        collectReachable(root, reachable);
    }

    for (auto it = entries.begin(); it != entries.end();) {
        PayloadSlot& slot = it->second;
        if (slot.current.has_value() &&
            reachable.find(&slot.current.value()) == reachable.end()) {
            slot.current.reset();
        }
        if (slot.retiring.has_value() &&
            reachable.find(&slot.retiring.value()) == reachable.end()) {
            slot.retiring.reset();
        }
        if (!slot.current.has_value() && !slot.retiring.has_value()) {
            it = entries.erase(it);
        } else {
            ++it;
        }
    }
}

bool NodePayloadRegistry::canPublish(uint64_t key) {
    if (key == 0) {
        return false;
    }
    reclaimUnreachable();
    const auto it = entries.find(key);
    if (it == entries.end()) {
        return true;
    }
    return !(it->second.current.has_value() && it->second.retiring.has_value());
}

const GeometryData* NodePayloadRegistry::resolveGeometry(const NodeDataHandle& handle, NodeDataHandle* outGeometryHandle) const {
    if (handle.key == 0) {
        return nullptr;
    }

    if (const GeometryData* geometry = get<GeometryData>(handle)) {
        if (outGeometryHandle) *outGeometryHandle = handle;
        return geometry;
    }
    if (const TransformData* transform = get<TransformData>(handle)) {
        return resolveGeometry(transform->sourceHandle, outGeometryHandle);
    }
    if (const GroupData* group = get<GroupData>(handle)) {
        return resolveGeometry(group->sourceHandle, outGeometryHandle);
    }
    return nullptr;
}

const RemeshData* NodePayloadRegistry::resolveRemesh(
    const NodeDataHandle& handle,
    NodeDataHandle* outRemeshHandle) const {
    if (handle.key == 0) return nullptr;

    if (const RemeshData* remesh = get<RemeshData>(handle)) {
        if (outRemeshHandle) *outRemeshHandle = handle;
        return remesh;
    }
    if (const HeatModelData* heatModel = get<HeatModelData>(handle)) {
        return resolveRemesh(heatModel->meshHandle, outRemeshHandle);
    }
    if (const GroupData* group = get<GroupData>(handle)) {
        return resolveRemesh(group->sourceHandle, outRemeshHandle);
    }
    return nullptr;
}

const PointData* NodePayloadRegistry::resolvePoints(const NodeDataHandle& handle, NodeDataHandle* outPointHandle) const {
    if (handle.key == 0) {
        return nullptr;
    }

    if (const PointData* points = get<PointData>(handle)) {
        if (outPointHandle) *outPointHandle = handle;
        return points;
    }
    if (const TransformData* transform = get<TransformData>(handle)) {
        return resolvePoints(transform->sourceHandle, outPointHandle);
    }
    return nullptr;
}

bool NodePayloadRegistry::resolveLocalToWorld(
    const NodeDataHandle& handle,
    std::array<float, 16>& outLocalToWorld) const {
    if (handle.key == 0) return false;

    if (const TransformData* transform = get<TransformData>(handle)) {
        outLocalToWorld = transform->localToWorld;
        return true;
    }
    if (const GeometryData* geometry = get<GeometryData>(handle)) {
        outLocalToWorld = geometry->localToWorld;
        return true;
    }
    if (const GroupData* group = get<GroupData>(handle)) {
        return resolveLocalToWorld(group->sourceHandle, outLocalToWorld);
    }
    if (const PointData* points = get<PointData>(handle)) {
        outLocalToWorld = points->localToWorld;
        return true;
    }
    if (const RemeshData* remesh = get<RemeshData>(handle)) {
        return resolveLocalToWorld(remesh->sourceMeshHandle, outLocalToWorld);
    }
    if (const HeatModelData* heatModel = get<HeatModelData>(handle)) {
        return resolveLocalToWorld(heatModel->meshHandle, outLocalToWorld);
    }
    return false;
}

static bool appendGroups(
    const NodePayloadRegistry& registry,
    const NodeDataHandle& handle,
    std::vector<MeshGroup>& outGroups) {
    if (handle.key == 0) {
        return false;
    }
    if (const GeometryData* geometry = registry.get<GeometryData>(handle)) {
        outGroups.insert(outGroups.end(), geometry->groups.begin(), geometry->groups.end());
        return true;
    }
    if (const TransformData* transform = registry.get<TransformData>(handle)) {
        return appendGroups(registry, transform->sourceHandle, outGroups);
    }
    if (registry.get<RemeshData>(handle) != nullptr) {
        return true;
    }
    if (const GroupData* group = registry.get<GroupData>(handle)) {
        if (!appendGroups(registry, group->sourceHandle, outGroups)) {
            return false;
        }
        outGroups.push_back(group->group);
        return true;
    }
    return false;
}

bool NodePayloadRegistry::resolveGroups(
    const NodeDataHandle& handle,
    std::vector<MeshGroup>& outGroups) const {
    outGroups.clear();
    return appendGroups(*this, handle, outGroups);
}

const HashValues* NodePayloadRegistry::findHashes(const NodeDataHandle& handle) const {
    const Entry* entry = findEntry(handle);
    return entry ? &entry->hashes : nullptr;
}

uint64_t NodePayloadRegistry::resolveHash(const NodeDataHandle& handle, HashDomain domain) const {
    const Entry* entry = findEntry(handle);
    return entry ? entry->hashes.get(domain) : 0;
}
