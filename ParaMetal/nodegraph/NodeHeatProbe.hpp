#pragma once

#include "NodeGraphKernels.hpp"
#include "NodeHeatProbeParams.hpp"

#include <vector>

struct NodeGraphState;

NodeTypeDefinition buildHeatProbeDefinition();
std::unique_ptr<NodeKernel> createHeatProbeKernel();
std::vector<NodeHeatProbeMarker> collectHeatProbeMarkers(
    const NodeGraphState& state);
bool buildHeatProbePositionParameters(
    const NodeGraphState& state,
    uint32_t nodeId,
    int probeIndex,
    const glm::vec3& position,
    uint64_t& outOriginInstanceId,
    std::vector<NodeGraphParamValue>& outParameters);

class NodeHeatProbe final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;
};
