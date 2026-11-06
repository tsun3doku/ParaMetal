#pragma once

#include "ExportTypes.hpp"

#include <cstdint>
#include <string>
#include <vector>

class NodeGraphController;
struct NodeGraphNode;
struct NodeGraphSocket;
struct EvaluatedSocketValue;

struct ExportOutput {
    NodeGraphNodeId nodeId{};
    NodeGraphSocketId socketId{};
    const NodeGraphNode* node = nullptr;
    const NodeGraphSocket* socket = nullptr;
    const EvaluatedSocketValue* evaluated = nullptr;
};

class PayloadExporter {
public:
    virtual ~PayloadExporter() = default;
    virtual uint8_t payloadType() const noexcept = 0;
    virtual void appendProducts(
        const ExportOutput& output,
        const NodeGraphController& graph,
        std::vector<ExportProduct>& out) const = 0;
    virtual ExportResult writeProduct(
        const ExportOutput& output,
        const ExportProductKey& key,
        ExportFormat format,
        const std::string& path,
        const NodeGraphController& graph) const = 0;
};
