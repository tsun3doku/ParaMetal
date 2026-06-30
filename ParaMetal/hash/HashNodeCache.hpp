#pragma once

#include "hash/HashValues.hpp"

#include <cstdint>
#include <vector>

//                                                          [ HashNodeCache
//                                                            -Domain aware node cache hash construction
//                                                            -Reads hash identity from NodeDataBlock
//                                                            -Socket helpers contain lookup + read + hash ]

struct NodeDataBlock;
struct NodeKernelHash;
enum class NodeGraphValueType : uint8_t;

class HashNodeCache {
public:
    static void combineInput(
        uint64_t& hash,
        const NodeDataBlock* input,
        HashDomain domain);

    // Hash a single input socket by value type and domain
    static void combineSocket(
        uint64_t& hash,
        const NodeKernelHash& context,
        const char* socketName,
        HashDomain domain);

    static void combineSocket(
        uint64_t& hash,
        const NodeKernelHash& context,
        NodeGraphValueType socketType,
        HashDomain domain);

    static void combineOptionalSocket(
        uint64_t& hash,
        const NodeKernelHash& context,
        const char* socketName,
        HashDomain domain);

    static void combineOptionalSocket(
        uint64_t& hash,
        const NodeKernelHash& context,
        NodeGraphValueType socketType,
        HashDomain domain);

    // Hash a variadic input socket by value type and domain
    static void combineSocketList(
        uint64_t& hash,
        const NodeKernelHash& context,
        const char* socketName,
        HashDomain domain);

    static void combineSocketList(
        uint64_t& hash,
        const NodeKernelHash& context,
        NodeGraphValueType socketType,
        HashDomain domain);

    static void combineOptionalSocketList(
        uint64_t& hash,
        const NodeKernelHash& context,
        const char* socketName,
        HashDomain domain);

    static void combineOptionalSocketList(
        uint64_t& hash,
        const NodeKernelHash& context,
        NodeGraphValueType socketType,
        HashDomain domain);
};
