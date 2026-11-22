#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildModelDefinition();
std::unique_ptr<NodeKernel> createModelKernel();

#include "NodeGraphKernels.hpp"

#include <cstdint>
#include <utility>

class NodeModel final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;

private:
    static bool parseMeshGeometry(const std::string& modelPath, GeometryData& geometry);
    static bool loadGeometryFromModelPath(const std::string& modelPath, GeometryData& geometry);
    static std::vector<std::string> resolveCandidateModelPaths(const std::string& modelPath);
};

namespace nodegraphtypes {
inline constexpr const char* Model = "model";
}
