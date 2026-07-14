#include "HashNodeCache.hpp"
#include "HashBuilder.hpp"

#include "nodegraph/NodeGraphDataTypes.hpp"
#include "nodegraph/NodeGraphCoreTypes.hpp"
#include "nodegraph/NodeGraphKernels.hpp"
#include "nodegraph/NodeGraphUtils.hpp"

#include <cassert>
#include <stdexcept>

void HashNodeCache::combineInput(
    uint64_t& hash,
    const NodeDataBlock* input,
    HashDomain domain) {
    if (!input) {
        HashBuilder::combine(hash, 0u);
        return;
    }
    HashBuilder::combine(hash, static_cast<uint64_t>(input->dataType));
    HashBuilder::combine(hash, input->hashes.get(domain));
}

static void combineInputList(
    uint64_t& hash,
    const std::vector<const NodeDataBlock*>& inputs,
    HashDomain domain) {
    HashBuilder::combine(hash, static_cast<uint64_t>(inputs.size()));
    for (const NodeDataBlock* input : inputs) {
        const NodeDataBlock* block = (input && input->payloadHandle.key != 0) ? input : nullptr;
        HashNodeCache::combineInput(hash, block, domain);
    }
}

static const NodeDataBlock* firstPayload(const std::vector<const NodeDataBlock*>& inputs) {
    for (const NodeDataBlock* input : inputs) {
        if (input && input->payloadHandle.key != 0) {
            return input;
        }
    }
    return nullptr;
}

void HashNodeCache::combineSocket(
    uint64_t& hash,
    const NodeKernelHash& context,
    const char* socketName,
    HashDomain domain) {
    const std::size_t index = inputIndexOf(context.node, socketName);
    if (index >= context.node.inputs.size()) {
        throw std::logic_error("Required hash socket is missing from node schema.");
    }
    const NodeDataBlock* block = firstPayload(context.inputs[index]);
    if (!block) {
        throw std::logic_error("Required hash socket has no evaluated payload.");
    }
    combineInput(hash, block, domain);
}

void HashNodeCache::combineSocket(
    uint64_t& hash,
    const NodeKernelHash& context,
    NodeGraphValueType socketType,
    HashDomain domain) {
    const std::size_t index = inputIndexOf(context.node, socketType);
    if (index >= context.node.inputs.size()) {
        throw std::logic_error("Required hash socket is missing from node schema.");
    }
    const NodeDataBlock* block = firstPayload(context.inputs[index]);
    if (!block) {
        throw std::logic_error("Required hash socket has no evaluated payload.");
    }
    combineInput(hash, block, domain);
}

void HashNodeCache::combineOptionalSocket(
    uint64_t& hash,
    const NodeKernelHash& context,
    const char* socketName,
    HashDomain domain) {
    const std::size_t index = inputIndexOf(context.node, socketName);
    if (index >= context.node.inputs.size()) {
        throw std::logic_error("Optional hash socket is missing from node schema.");
    }
    combineInput(hash, firstPayload(context.inputs[index]), domain);
}

void HashNodeCache::combineOptionalSocket(
    uint64_t& hash,
    const NodeKernelHash& context,
    NodeGraphValueType socketType,
    HashDomain domain) {
    const std::size_t index = inputIndexOf(context.node, socketType);
    if (index >= context.node.inputs.size()) {
        throw std::logic_error("Optional hash socket is missing from node schema.");
    }
    combineInput(hash, firstPayload(context.inputs[index]), domain);
}

void HashNodeCache::combineSocketList(
    uint64_t& hash,
    const NodeKernelHash& context,
    const char* socketName,
    HashDomain domain) {
    const std::size_t index = inputIndexOf(context.node, socketName);
    if (index >= context.node.inputs.size()) {
        throw std::logic_error("Required hash socket list is missing from node schema.");
    }
    const std::vector<const NodeDataBlock*>& inputs = context.inputs[index];
    if (inputs.empty()) {
        throw std::logic_error("Required hash socket list has no evaluated payloads.");
    }
    for (const NodeDataBlock* input : inputs) {
        if (!input || input->payloadHandle.key == 0) {
            throw std::logic_error("Required hash socket list contains an invalid evaluated payload.");
        }
    }
    combineInputList(hash, inputs, domain);
}

void HashNodeCache::combineSocketList(
    uint64_t& hash,
    const NodeKernelHash& context,
    NodeGraphValueType socketType,
    HashDomain domain) {
    const std::size_t index = inputIndexOf(context.node, socketType);
    if (index >= context.node.inputs.size()) {
        throw std::logic_error("Required hash socket list is missing from node schema.");
    }
    const std::vector<const NodeDataBlock*>& inputs = context.inputs[index];
    if (inputs.empty()) {
        throw std::logic_error("Required hash socket list has no evaluated payloads.");
    }
    for (const NodeDataBlock* input : inputs) {
        if (!input || input->payloadHandle.key == 0) {
            throw std::logic_error("Required hash socket list contains an invalid evaluated payload.");
        }
    }
    combineInputList(hash, inputs, domain);
}

void HashNodeCache::combineOptionalSocketList(
    uint64_t& hash,
    const NodeKernelHash& context,
    const char* socketName,
    HashDomain domain) {
    const std::size_t index = inputIndexOf(context.node, socketName);
    if (index >= context.node.inputs.size()) {
        throw std::logic_error("Optional hash socket list is missing from node schema.");
    }
    combineInputList(hash, context.inputs[index], domain);
}

void HashNodeCache::combineOptionalSocketList(
    uint64_t& hash,
    const NodeKernelHash& context,
    NodeGraphValueType socketType,
    HashDomain domain) {
    const std::size_t index = inputIndexOf(context.node, socketType);
    if (index >= context.node.inputs.size()) {
        throw std::logic_error("Optional hash socket list is missing from node schema.");
    }
    combineInputList(hash, context.inputs[index], domain);
}
