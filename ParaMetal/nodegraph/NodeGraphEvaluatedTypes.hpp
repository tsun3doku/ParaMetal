#pragma once

#include "NodeGraphCoreTypes.hpp"
#include "NodeGraphDataTypes.hpp"
#include "NodeGraphProductTypes.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

enum class EvaluatedSocketStatus : uint8_t {
    Missing,
    Value,
    Error,
};

struct EvaluatedSocketValue {
    EvaluatedSocketStatus status = EvaluatedSocketStatus::Missing;
    NodeDataBlock data{};
    std::string error;
};

// Per-tick socket-output state produced by the runtime.
//
// outputBySocket is the evaluated result: for each output socket, whether it
// resolved and the NodeDataBlock it produced.
//
// productBySocket is NOT evaluation output - it holds GPU/runtime product
// handles seeded externally (via setOutputProductHandle) and read alongside
// outputs during package compilation. It is bundled here as a per-tick
// convenience, not because it is an evaluated value. Splitting it out is a
// deferred question.
//
struct NodeGraphEvaluationState {
    std::unordered_map<uint64_t, EvaluatedSocketValue> outputBySocket;
    std::unordered_map<uint64_t, ProductHandle> productBySocket;

    const EvaluatedSocketValue* valueFor(uint64_t socketKey) const {
        if (socketKey == 0) return nullptr;
        auto it = outputBySocket.find(socketKey);
        return it != outputBySocket.end() ? &it->second : nullptr;
    }

    ProductHandle productFor(uint64_t socketKey, NodeProductType expectedType) const {
        if (socketKey == 0) return {};
        auto it = productBySocket.find(socketKey);
        if (it != productBySocket.end() &&
            it->second.type == expectedType &&
            it->second.isValid()) {
            return it->second;
        }
        return {};
    }
};
