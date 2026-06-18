#include "HashNodeCache.hpp"
#include "HashBuilder.hpp"

#include "nodegraph/NodeGraphDataTypes.hpp"
#include "nodegraph/NodeGraphCoreTypes.hpp"
#include "nodegraph/NodeGraphKernels.hpp"
#include "nodegraph/NodeGraphUtils.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"

#include <cassert>
#include <stdexcept>

void HashNodeCache::combineInput(
    uint64_t& hash,
    const NodeDataBlock* input,
    const NodePayloadRegistry* payloadRegistry,
    HashDomain domain) {
    if (!input) {
        HashBuilder::combine(hash, 0u);
        return;
    }
    HashBuilder::combine(hash, static_cast<uint64_t>(input->dataType));
    assert(payloadRegistry && input->payloadHandle.key != 0);
    HashBuilder::combine(hash, payloadRegistry->resolveHash(input->payloadHandle, domain));
}

void HashNodeCache::combineInputList(
    uint64_t& hash,
    const std::vector<const EvaluatedSocketValue*>& inputs,
    const NodePayloadRegistry* payloadRegistry,
    HashDomain domain) {
    HashBuilder::combine(hash, static_cast<uint64_t>(inputs.size()));
    for (const EvaluatedSocketValue* input : inputs) {
        const NodeDataBlock* block = nullptr;
        if (input && input->status == EvaluatedSocketStatus::Value &&
            input->data.payloadHandle.key != 0) {
            block = &input->data;
        }
        combineInput(hash, block, payloadRegistry, domain);
    }
}

void HashNodeCache::combineSocket(
    uint64_t& hash,
    const NodeGraphKernelHashContext& context,
    const char* socketName,
    HashDomain domain) {
    const NodeGraphSocket* socket = context.node.input(socketName);
    if (!socket) {
        throw std::logic_error("Required hash socket is missing from node schema.");
    }
    const EvaluatedSocketValue* eval = readEvaluatedInput(context.node, socket->id, context.executionState);
    if (!eval || eval->status != EvaluatedSocketStatus::Value || eval->data.payloadHandle.key == 0) {
        throw std::logic_error("Required hash socket has no evaluated payload.");
    }
    combineInput(hash, &eval->data, context.executionState.services.payloadRegistry, domain);
}

void HashNodeCache::combineSocket(
    uint64_t& hash,
    const NodeGraphKernelHashContext& context,
    NodeGraphValueType socketType,
    HashDomain domain) {
    const NodeGraphSocket* socket = context.node.input(socketType);
    if (!socket) {
        throw std::logic_error("Required hash socket is missing from node schema.");
    }
    const EvaluatedSocketValue* eval = readEvaluatedInput(context.node, socket->id, context.executionState);
    if (!eval || eval->status != EvaluatedSocketStatus::Value || eval->data.payloadHandle.key == 0) {
        throw std::logic_error("Required hash socket has no evaluated payload.");
    }
    combineInput(hash, &eval->data, context.executionState.services.payloadRegistry, domain);
}

void HashNodeCache::combineOptionalSocket(
    uint64_t& hash,
    const NodeGraphKernelHashContext& context,
    const char* socketName,
    HashDomain domain) {
    const NodeGraphSocket* socket = context.node.input(socketName);
    if (!socket) {
        throw std::logic_error("Optional hash socket is missing from node schema.");
    }
    const EvaluatedSocketValue* eval = readEvaluatedInput(context.node, socket->id, context.executionState);
    const NodeDataBlock* block = nullptr;
    if (eval && eval->status == EvaluatedSocketStatus::Value && eval->data.payloadHandle.key != 0) {
        block = &eval->data;
    }
    combineInput(hash, block, context.executionState.services.payloadRegistry, domain);
}

void HashNodeCache::combineOptionalSocket(
    uint64_t& hash,
    const NodeGraphKernelHashContext& context,
    NodeGraphValueType socketType,
    HashDomain domain) {
    const NodeGraphSocket* socket = context.node.input(socketType);
    if (!socket) {
        throw std::logic_error("Optional hash socket is missing from node schema.");
    }
    const EvaluatedSocketValue* eval = readEvaluatedInput(context.node, socket->id, context.executionState);
    const NodeDataBlock* block = nullptr;
    if (eval && eval->status == EvaluatedSocketStatus::Value && eval->data.payloadHandle.key != 0) {
        block = &eval->data;
    }
    combineInput(hash, block, context.executionState.services.payloadRegistry, domain);
}

void HashNodeCache::combineSocketList(
    uint64_t& hash,
    const NodeGraphKernelHashContext& context,
    const char* socketName,
    HashDomain domain) {
    const NodeGraphSocket* socket = context.node.input(socketName);
    if (!socket) {
        throw std::logic_error("Required hash socket list is missing from node schema.");
    }
    const auto evals = readEvaluatedInputs(context.node, socket->id, context.executionState);
    if (evals.empty()) {
        throw std::logic_error("Required hash socket list has no evaluated payloads.");
    }
    HashBuilder::combine(hash, static_cast<uint64_t>(evals.size()));
    for (const EvaluatedSocketValue* eval : evals) {
        if (!eval || eval->status != EvaluatedSocketStatus::Value || eval->data.payloadHandle.key == 0) {
            throw std::logic_error("Required hash socket list contains an invalid evaluated payload.");
        }
        combineInput(hash, &eval->data, context.executionState.services.payloadRegistry, domain);
    }
}

void HashNodeCache::combineSocketList(
    uint64_t& hash,
    const NodeGraphKernelHashContext& context,
    NodeGraphValueType socketType,
    HashDomain domain) {
    const NodeGraphSocket* socket = context.node.input(socketType);
    if (!socket) {
        throw std::logic_error("Required hash socket list is missing from node schema.");
    }
    const auto evals = readEvaluatedInputs(context.node, socket->id, context.executionState);
    if (evals.empty()) {
        throw std::logic_error("Required hash socket list has no evaluated payloads.");
    }
    HashBuilder::combine(hash, static_cast<uint64_t>(evals.size()));
    for (const EvaluatedSocketValue* eval : evals) {
        if (!eval || eval->status != EvaluatedSocketStatus::Value || eval->data.payloadHandle.key == 0) {
            throw std::logic_error("Required hash socket list contains an invalid evaluated payload.");
        }
        combineInput(hash, &eval->data, context.executionState.services.payloadRegistry, domain);
    }
}

void HashNodeCache::combineOptionalSocketList(
    uint64_t& hash,
    const NodeGraphKernelHashContext& context,
    const char* socketName,
    HashDomain domain) {
    const NodeGraphSocket* socket = context.node.input(socketName);
    if (!socket) {
        throw std::logic_error("Optional hash socket list is missing from node schema.");
    }
    const auto evals = readEvaluatedInputs(context.node, socket->id, context.executionState);
    combineInputList(hash, evals, context.executionState.services.payloadRegistry, domain);
}

void HashNodeCache::combineOptionalSocketList(
    uint64_t& hash,
    const NodeGraphKernelHashContext& context,
    NodeGraphValueType socketType,
    HashDomain domain) {
    const NodeGraphSocket* socket = context.node.input(socketType);
    if (!socket) {
        throw std::logic_error("Optional hash socket list is missing from node schema.");
    }
    const auto evals = readEvaluatedInputs(context.node, socket->id, context.executionState);
    combineInputList(hash, evals, context.executionState.services.payloadRegistry, domain);
}
