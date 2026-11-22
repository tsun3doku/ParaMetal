#pragma once

#include "NodeGraphDataTypes.hpp"
#include "NodeGraphState.hpp"
#include "NodeGraphPayloadTypes.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class NodePayloadRegistry;

struct NodeKernelRuntime {
    const NodeGraphState& graph;
    NodePayloadRegistry* const payloadRegistry;
};

struct NodeKernelHash {
    const NodeGraphNode& node;
    const std::vector<std::vector<const NodeDataBlock*>>& inputs;
    const NodeKernelRuntime& runtime;
};

struct NodeKernelEval {
    const NodeGraphNode& node;
    const std::vector<std::vector<const NodeDataBlock*>>& inputs;
    std::vector<NodeDataBlock>& outputs;
    const NodeKernelRuntime& runtime;
    HashValues outputHashes{};
    std::string error;
};

class NodeKernel {
public:
    virtual ~NodeKernel() = default;
    virtual const char* typeId() const = 0;
    virtual void execute(NodeKernelEval& eval) const = 0;
    virtual HashValues computeOutputHashes(const NodeKernelHash& hash) const = 0;
};

class NodeGraphKernels {
public:
    NodeGraphKernels();

    bool hasKernel(const NodeTypeId& typeId) const;
    bool registerKernel(std::unique_ptr<NodeKernel> kernel);
    bool registerKernellessNode(const NodeTypeId& typeId);
    bool requiresKernel(const NodeTypeId& typeId) const;
    bool isValid() const { return registrationValid; }
    HashValues computeOutputHashes(
        const NodeGraphNode& node,
        const NodeKernelRuntime& runtime,
        const std::vector<std::vector<const NodeDataBlock*>>& inputs) const;
    void executeNode(
        const NodeGraphNode& node,
        const NodeKernelRuntime& runtime,
        const std::vector<std::vector<const NodeDataBlock*>>& inputs,
        std::vector<NodeDataBlock>& outputs,
        HashValues outputHashes,
        std::string& error) const;

private:
    std::unordered_map<NodeTypeId, std::unique_ptr<NodeKernel>> kernels;
    std::unordered_set<NodeTypeId> kernellessTypeIds;
    bool registrationValid = true;
};

