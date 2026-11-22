#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildGroupDefinition();
std::unique_ptr<NodeKernel> createGroupKernel();

#include "NodeGraphKernels.hpp"

class NodeGroup final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;

private:
    static bool equalsIgnoreCase(const std::string& lhs, const std::string& rhs);
    static uint32_t resolveTargetGroupId(GeometryData& geometry, const std::string& targetGroupName);
    static bool applyAssignment(
        GeometryData& geometry,
        const std::string& sourceGroupName,
        const std::string& targetGroupName);
};

namespace nodegraphtypes {
inline constexpr const char* Group = "group";
}

namespace nodegraphparams::group {
constexpr uint32_t Enabled = 1;
constexpr uint32_t SourceName = 2;
constexpr uint32_t TargetName = 3;
constexpr uint32_t SourceType = 4;

namespace sourcetype {
constexpr int64_t Vertex = 0;
}
}
