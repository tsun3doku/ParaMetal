#pragma once

#include <cstdint>

#include "hash/HashValues.hpp"
#include "nodegraph/NodeGraphCoreTypes.hpp"

enum class NodeProductType : uint8_t {
    None,
    Model,
    Remesh,
    Voronoi,
    Point,
    Heat,
    CFD,
};

struct ProductHandle {
    NodeProductType type = NodeProductType::None;
    uint64_t outputSocketKey = 0;
    HashValues hashes{};
    // Exact publication identity. Content comparisons stay hash-based; this field
    // is only used for exact PM lookup so older generations remain resolvable.
    uint64_t publicationId = 0;

    bool isValid() const {
        return type != NodeProductType::None &&
            outputSocketKey != 0 &&
            hashes.full != 0;
    }

    bool operator==(const ProductHandle& other) const {
        return type == other.type &&
            outputSocketKey == other.outputSocketKey &&
            hashes.full == other.hashes.full &&
            hashes.geometry == other.hashes.geometry &&
            hashes.thermal == other.hashes.thermal &&
            hashes.simulation == other.hashes.simulation &&
            hashes.display == other.hashes.display;
    }
};
