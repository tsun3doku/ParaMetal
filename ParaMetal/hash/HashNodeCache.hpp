#pragma once

#include "hash/HashValues.hpp"

#include <cstdint>
#include <vector>

//                                                          [ HashNodeCache
//                                                            -Domain aware node cache hash construction
//                                                            -Resolves payload content hashes by domain
//                                                            -Socket helpers contain lookup + read + hash ]

struct NodeDataBlock;
struct EvaluatedSocketValue;
struct NodeGraphKernelHashContext;
enum class NodeGraphValueType : uint8_t;
class NodePayloadRegistry;

class HashNodeCache {
public:
    static void combineInput(
        uint64_t& hash,
        const NodeDataBlock* input,
        const NodePayloadRegistry* payloadRegistry,
        HashDomain domain);

    static void combineInputList(
        uint64_t& hash,
        const std::vector<const EvaluatedSocketValue*>& inputs,
        const NodePayloadRegistry* payloadRegistry,
        HashDomain domain);

    // Hash a single input socket by value type and domain
    static void combineSocket(
        uint64_t& hash,
        const NodeGraphKernelHashContext& context,
        const char* socketName,
        HashDomain domain);

    static void combineSocket(
        uint64_t& hash,
        const NodeGraphKernelHashContext& context,
        NodeGraphValueType socketType,
        HashDomain domain);

    static void combineOptionalSocket(
        uint64_t& hash,
        const NodeGraphKernelHashContext& context,
        const char* socketName,
        HashDomain domain);

    static void combineOptionalSocket(
        uint64_t& hash,
        const NodeGraphKernelHashContext& context,
        NodeGraphValueType socketType,
        HashDomain domain);

    // Hash a variadic input socket by value type and domain
    static void combineSocketList(
        uint64_t& hash,
        const NodeGraphKernelHashContext& context,
        const char* socketName,
        HashDomain domain);

    static void combineSocketList(
        uint64_t& hash,
        const NodeGraphKernelHashContext& context,
        NodeGraphValueType socketType,
        HashDomain domain);

    static void combineOptionalSocketList(
        uint64_t& hash,
        const NodeGraphKernelHashContext& context,
        const char* socketName,
        HashDomain domain);

    static void combineOptionalSocketList(
        uint64_t& hash,
        const NodeGraphKernelHashContext& context,
        NodeGraphValueType socketType,
        HashDomain domain);
};
