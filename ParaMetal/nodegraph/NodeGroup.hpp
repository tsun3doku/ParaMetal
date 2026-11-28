#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildGroupDefinition();
std::unique_ptr<NodeKernel> createGroupKernel();

class NodeGroup final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;
    void inspect(
        const NodeAuthoringContext& context,
        NodeInspection& out) const override;
    bool beginInteraction(
        const NodeAuthoringContext& context,
        uint32_t actionId,
        NodeInteractionRequest& out,
        std::string& error) const override;
    bool completeInteraction(
        const NodeAuthoringContext& context,
        const NodeInteractionSession& session,
        const NodeInteractionResult& result,
        std::vector<NodeGraphParamValue>& patch,
        std::string& error) const override;
    bool previewInteraction(
        const NodeAuthoringContext& context,
        const NodeInteractionSession& session,
        const NodeInteractionValue& authored,
        NodeInteractionValue& display,
        std::string& error) const override;
};

namespace nodegraphtypes {
inline constexpr const char* Group = "group";
}
