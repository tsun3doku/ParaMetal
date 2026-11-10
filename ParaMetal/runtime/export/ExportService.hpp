#pragma once

#include "ExportTypes.hpp"
#include "PayloadExporter.hpp"
#include "mesh/IsocontourBuilder.hpp"
#include "mesh/Polyline.hpp"
#include "nodegraph/NodeWriteParams.hpp"
#include "nodegraph/NodeGraphCoreTypes.hpp"

class NodeGraphController;
class NodePayloadRegistry;
class HeatSystemComputeController;
class ExporterRegistry;

class ExportService {
public:
    static bool discoverProducts(
        const ExporterRegistry& exporters,
        const NodeGraphController& graph,
        NodeGraphNodeId writeNodeId,
        std::vector<ExportProduct>& out,
        std::string& outError);
    static ExportResult exportProduct(
        const ExporterRegistry& exporters,
        const NodeGraphController& graph,
        NodeGraphNodeId writeNodeId);
    static bool buildHeatContours(
        uint64_t heatSocketKey,
        uint64_t heatModelKey,
        float spacingC,
        bool applyTransform,
        HeatSystemComputeController& heatController,
        const std::vector<std::vector<float>>* temperaturesByModel,
        std::vector<mesh::Polyline>& out,
        std::string& outError);
};
