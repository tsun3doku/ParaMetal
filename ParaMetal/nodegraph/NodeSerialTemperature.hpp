#pragma once

#include "NodeGraphKernels.hpp"

NodeTypeDefinition buildSerialTemperatureDefinition();
std::unique_ptr<NodeKernel> createSerialTemperatureKernel();

#include "NodeGraphKernels.hpp"

class NodeSerialTemperature final : public NodeKernel {
public:
    const char* typeId() const override;
    void execute(NodeKernelEval& eval) const override;
    HashValues computeOutputHashes(const NodeKernelHash& hash) const override;
};

namespace nodegraphtypes {
inline constexpr const char* SerialTemperature = "serial_temperature";
}
