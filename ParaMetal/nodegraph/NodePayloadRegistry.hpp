#pragma once

#include "domain/GeometryData.hpp"
#include "hash/HashValues.hpp"
#include "NodeGraphCoreTypes.hpp"
#include "NodeGraphTypes.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <typeindex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct PointData;
struct RemeshData;

class NodePayloadRegistry {
public:
    NodePayloadRegistry() = default;

    template <typename T>
    NodeDataHandle store(uint64_t key, T&& payload, const HashValues& hashes) {
        if (key == 0) {
            return {};
        }

        PayloadSlot& slot = entries[key];
        if (slot.current.has_value() && slot.retiring.has_value()) {
            return {};
        }

        using PayloadType = std::decay_t<T>;
        Entry entry{};
        entry.payload = std::make_shared<PayloadType>(std::forward<T>(payload));
        entry.type = std::type_index(typeid(PayloadType));
        entry.hashes = hashes;
        entry.key = key;
        entry.generation = nextGeneration++;

        if (slot.current.has_value()) {
            slot.retiring = std::move(slot.current);
        }
        slot.current = std::move(entry);
        return NodeDataHandle{key, slot.current->generation};
    }

    void setLiveRoots(std::vector<NodeDataHandle> roots);
    void reclaimUnreachable();
    bool canPublish(uint64_t key);

    template <typename T>
    const T* get(const NodeDataHandle& handle) const {
        const Entry* entry = findEntry(handle);
        if (!entry || entry->type != std::type_index(typeid(T))) {
            return nullptr;
        }
        return static_cast<const T*>(entry->payload.get());
    }

    const GeometryData* resolveGeometry(const NodeDataHandle& handle, NodeDataHandle* outGeometryHandle = nullptr) const;
    const RemeshData* resolveRemesh(const NodeDataHandle& handle, NodeDataHandle* outRemeshHandle = nullptr) const;
    const PointData* resolvePoints(const NodeDataHandle& handle, NodeDataHandle* outPointHandle = nullptr) const;
    bool resolveLocalToWorld(const NodeDataHandle& handle, std::array<float, 16>& outLocalToWorld) const;
    bool resolveGroups(const NodeDataHandle& handle, std::vector<MeshGroup>& outGroups) const;
    const HashValues* findHashes(const NodeDataHandle& handle) const;
    uint64_t resolveHash(const NodeDataHandle& handle, HashDomain domain) const;

private:
    struct Entry {
        std::shared_ptr<void> payload;
        std::type_index type{typeid(void)};
        HashValues hashes{};
        uint64_t key = 0;
        uint64_t generation = 0;
    };

    struct PayloadSlot {
        std::optional<Entry> current;
        std::optional<Entry> retiring;
    };

    const Entry* findEntry(NodeDataHandle handle) const;
    void collectReachable(
        const NodeDataHandle& handle,
        std::unordered_set<const Entry*>& reachable) const;
    void collectDependencies(
        const Entry& entry,
        std::vector<NodeDataHandle>& outDependencies) const;

    std::unordered_map<uint64_t, PayloadSlot> entries;
    std::vector<NodeDataHandle> liveRoots;
    uint64_t nextGeneration = 1;
};
