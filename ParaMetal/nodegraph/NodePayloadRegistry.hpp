#pragma once

#include "hash/HashValues.hpp"
#include "NodeGraphCoreTypes.hpp"
#include "NodeGraphTypes.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <typeindex>
#include <unordered_map>

struct GeometryData;
struct PointData;

class NodePayloadRegistry {
public:
    NodePayloadRegistry() = default;

    template <typename T>
    NodeDataHandle store(uint64_t key, const T& payload, const HashValues& hashes) {
        if (key == 0) {
            return {};
        }
        T stored = payload;
        stored.hashes = hashes;
        Entry entry{};
        entry.payload = std::make_shared<T>(stored);
        entry.type = std::type_index(typeid(T));
        entry.hashes = hashes;
        entries[key] = entry;
        NodeDataHandle handle{};
        handle.key = key;
        return handle;
    }

    template <typename T>
    const T* get(const NodeDataHandle& handle) const {
        auto it = entries.find(handle.key);
        if (it == entries.end()) {
            return nullptr;
        }
        if (it->second.type != std::type_index(typeid(T))) {
            return nullptr;
        }
        return static_cast<const T*>(it->second.payload.get());
    }

    void erase(uint64_t key);
    void clear();
    NodeDataHandle resolveMeshHandle(uint8_t type, const NodeDataHandle& handle) const;
    const GeometryData* resolveGeometry(const NodeDataHandle& handle, NodeDataHandle* outSourceHandle = nullptr) const;
    const PointData* resolvePoints(const NodeDataHandle& handle) const;
    uint64_t resolveHash(const NodeDataHandle& handle, HashDomain domain) const;

private:
    struct Entry {
        std::shared_ptr<void> payload;
        std::type_index type{typeid(void)};
        HashValues hashes{};
    };

    std::unordered_map<uint64_t, Entry> entries;
};
