#include "NodeGraphKernels.hpp"

#include "NodeGraphModules.hpp"
#include "NodeGraphUtils.hpp"

#include <utility>

NodeGraphKernels::NodeGraphKernels() {
    registrationValid = registerNodeKernels(*this);
}

bool NodeGraphKernels::hasKernel(const NodeTypeId& typeId) const {
    const auto it = kernels.find(getNodeTypeId(typeId));
    return it != kernels.end() && it->second != nullptr;
}

const NodeKernel* NodeGraphKernels::findKernel(const NodeTypeId& typeId) const {
    const auto it = kernels.find(getNodeTypeId(typeId));
    return it == kernels.end() ? nullptr : it->second.get();
}

bool NodeGraphKernels::registerKernel(std::unique_ptr<NodeKernel> kernel) {
    if (!kernel || !kernel->typeId() || kernel->typeId()[0] == '\0') {
        registrationValid = false;
        return false;
    }

    const NodeTypeId canonicalTypeId = getNodeTypeId(kernel->typeId());
    if (kernels.find(canonicalTypeId) != kernels.end() ||
        kernellessTypeIds.find(canonicalTypeId) != kernellessTypeIds.end()) {
        registrationValid = false;
        return false;
    }
    kernels.emplace(canonicalTypeId, std::move(kernel));
    return true;
}

bool NodeGraphKernels::registerKernellessNode(const NodeTypeId& typeId) {
    const NodeTypeId canonicalTypeId = getNodeTypeId(typeId);
    if (kernels.find(canonicalTypeId) != kernels.end() ||
        !kernellessTypeIds.insert(canonicalTypeId).second) {
        registrationValid = false;
        return false;
    }
    return true;
}

bool NodeGraphKernels::requiresKernel(const NodeTypeId& typeId) const {
    return kernellessTypeIds.find(getNodeTypeId(typeId)) == kernellessTypeIds.end();
}

HashValues NodeGraphKernels::computeOutputHashes(
    const NodeGraphNode& node,
    const NodeKernelRuntime& runtime,
    const std::vector<std::vector<const NodeDataBlock*>>& inputs) const {
    const NodeTypeId canonicalTypeId = getNodeTypeId(node.typeId);
    const auto kernelIt = kernels.find(canonicalTypeId);
    if (kernelIt == kernels.end() || !kernelIt->second) {
        return {};
    }

    NodeKernelHash hash{node, inputs, runtime};
    return kernelIt->second->computeOutputHashes(hash);
}

void NodeGraphKernels::executeNode(
    const NodeGraphNode& node,
    const NodeKernelRuntime& runtime,
    const std::vector<std::vector<const NodeDataBlock*>>& inputs,
    std::vector<NodeDataBlock>& outputs,
    HashValues outputHashes,
    std::string& error) const {
    const NodeTypeId canonicalTypeId = getNodeTypeId(node.typeId);
    const auto kernelIt = kernels.find(canonicalTypeId);
    if (kernelIt == kernels.end() || !kernelIt->second) {
        return;
    }

    NodeKernelEval eval{node, inputs, outputs, runtime, outputHashes};

    kernelIt->second->execute(eval);
    error = std::move(eval.error);
}
