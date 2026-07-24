#pragma once

#include "NodeGraphDataTypes.hpp"
#include "NodeGraphState.hpp"
#include "NodeGraphPayloadTypes.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class HeatSystemComputeController;
class NodePayloadRegistry;
class ModelRegistry;
class RuntimeModelComputeTransport;
class RuntimeRemeshComputeTransport;
class RuntimeVoronoiComputeTransport;
class RuntimeContactComputeTransport;
class RuntimeHeatComputeTransport;
class RuntimePointComputeTransport;
class RuntimePointDisplayTransport;
class RuntimeModelDisplayTransport;
class RuntimeRemeshDisplayTransport;
class RuntimeVoronoiDisplayTransport;
class RuntimeContactDisplayTransport;
class RuntimeHeatDisplayTransport;
class SceneController;
class VulkanDevice;
class MemoryAllocator;

struct NodeRuntimeServices {
    SceneController* sceneController = nullptr;
    RuntimeModelComputeTransport* modelComputeTransport = nullptr;
    RuntimeRemeshComputeTransport* remeshComputeTransport = nullptr;
    RuntimeVoronoiComputeTransport* voronoiComputeTransport = nullptr;
    RuntimeContactComputeTransport* contactComputeTransport = nullptr;
    RuntimeHeatComputeTransport* heatComputeTransport = nullptr;
    RuntimeModelDisplayTransport* modelDisplayTransport = nullptr;
    RuntimePointDisplayTransport* pointDisplayTransport = nullptr;
    RuntimeRemeshDisplayTransport* remeshDisplayTransport = nullptr;
    RuntimeVoronoiDisplayTransport* voronoiDisplayTransport = nullptr;
    RuntimeContactDisplayTransport* contactDisplayTransport = nullptr;
    RuntimeHeatDisplayTransport* heatDisplayTransport = nullptr;
    RuntimePointComputeTransport* pointComputeTransport = nullptr;
    HeatSystemComputeController* heatSystemController = nullptr;
    NodePayloadRegistry* payloadRegistry = nullptr;
    ModelRegistry* resourceManager = nullptr;
    VulkanDevice* vulkanDevice = nullptr;
    MemoryAllocator* memoryAllocator = nullptr;
};

struct NodeKernelRuntime {
    const NodeGraphState& graph;
    NodePayloadRegistry* const payloadRegistry;
    const NodeRuntimeServices& services;
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
    HashValues computeOutputHashes(
        const NodeGraphNode& node,
        const NodeKernelRuntime& runtime,
        const std::vector<std::vector<const NodeDataBlock*>>& inputs) const;
    void executeNode(
        const NodeGraphNode& node,
        const NodeKernelRuntime& runtime,
        const std::vector<std::vector<const NodeDataBlock*>>& inputs,
        std::vector<NodeDataBlock>& outputs,
        HashValues outputHashes) const;

private:
    void registerDefaultKernels();
    void registerKernel(std::unique_ptr<NodeKernel> kernel);

    std::vector<std::unique_ptr<NodeKernel>> kernels;
    std::unordered_map<NodeTypeId, const NodeKernel*> kernelByTypeId;
};

