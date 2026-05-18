#pragma once

#include "NodeGraphTypes.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class NodeGraphTypeRegistry {
public:
    uint8_t registerPayloadType(const std::string& name, NodeGraphValueType displayType);
    uint8_t getTypeId(const std::string& name) const;
    const std::string* getTypeName(uint8_t typeId) const;
    NodeGraphValueType getDisplayType(uint8_t typeId) const;
    size_t typeCount() const;

private:
    struct Entry {
        std::string name;
        NodeGraphValueType displayType = NodeGraphValueType::None;
    };
    std::unordered_map<std::string, uint8_t> nameToId;
    std::vector<Entry> idToEntry;
    uint8_t nextTypeId = 1;
};
