#include "NodeGraphTypeRegistry.hpp"

uint8_t NodeGraphTypeRegistry::registerPayloadType(const std::string& name, NodeGraphValueType displayType) {
    if (nameToId.count(name)) {
        return 0;
    }
    if (nextTypeId == 0) {
        return 0;
    }

    uint8_t id = nextTypeId++;
    nameToId[name] = id;
    idToEntry.resize(nextTypeId);
    idToEntry[id].name = name;
    idToEntry[id].displayType = displayType;
    return id;
}

uint8_t NodeGraphTypeRegistry::getTypeId(const std::string& name) const {
    auto it = nameToId.find(name);
    return it != nameToId.end() ? it->second : 0;
}

const std::string* NodeGraphTypeRegistry::getTypeName(uint8_t typeId) const {
    if (typeId == 0 || typeId >= idToEntry.size()) {
        return nullptr;
    }
    return &idToEntry[typeId].name;
}

NodeGraphValueType NodeGraphTypeRegistry::getDisplayType(uint8_t typeId) const {
    if (typeId == 0 || typeId >= idToEntry.size()) {
        return NodeGraphValueType::None;
    }
    return idToEntry[typeId].displayType;
}

size_t NodeGraphTypeRegistry::typeCount() const {
    return nameToId.size();
}
