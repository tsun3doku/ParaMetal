#pragma once

#include <cstdint>

#include "hash/HashBuilder.hpp"
#include "hash/HashValues.hpp"
#include "runtime/ProductHandle.hpp"

inline void combineProductHandleHash(uint64_t& hash, const ProductHandle& handle, HashDomain domain) {
    HashBuilder::combine(hash, static_cast<uint64_t>(handle.type));
    HashBuilder::combine(hash, handle.outputSocketKey);
    HashBuilder::combine(hash, handle.publicationId);
    HashBuilder::combine(hash, handle.hashes.get(domain));
}

inline uint64_t surfaceGeometryIdentity(const ProductHandle& handle) {
    uint64_t hash = HashBuilder::start();
    combineProductHandleHash(hash, handle, HashDomain::Geometry);
    return hash;
}
