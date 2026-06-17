#pragma once

#include "NodeGraphKernels.hpp"

#include <vector>

class NodeHeatSolve final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeGraphKernelContext& context) const override;
    bool computeInputHash(const NodeGraphKernelHashContext& context, uint64_t& outHash) const override;

    static NodeGraphNodeId selectHeatSolveNode(const NodeGraphState& state);

private:
    static void populateOutputPayloads(
        NodeGraphKernelContext& context,
        const std::vector<NodeDataHandle>& heatModelHandles,
        const std::vector<NodeDataHandle>& voronoiHandles,
        const std::vector<NodeDataHandle>& contactHandles,
        uint64_t voronoiPayloadHash,
        uint64_t contactPayloadHash,
        float contactThermalConductance,
        float simulationDuration,
        bool active,
        bool paused,
        uint32_t resetCounter,
        uint32_t rewindFrame);
};
