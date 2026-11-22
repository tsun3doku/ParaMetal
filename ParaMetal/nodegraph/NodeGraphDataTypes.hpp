#pragma once

#include "../hash/HashValues.hpp"
#include "NodeGraphCoreTypes.hpp"
#include "NodeGraphTypes.hpp"

#include <cstdint>

struct NodeDataBlock {
    uint8_t dataType = 0;
    NodeDataHandle payloadHandle{};
    HashValues hashes{};
    bool isFrozen = false;
};