#include "ExporterRegistry.hpp"

#include "ExportService.hpp"
#include "LineWriter.hpp"
#include "MeshWriter.hpp"
#include "PointWriter.hpp"
#include "domain/GeometryData.hpp"
#include "domain/HeatData.hpp"
#include "domain/PointData.hpp"
#include "heat/HeatSystemComputeController.hpp"
#include "nodegraph/NodeGraphController.hpp"
#include "nodegraph/NodeGraphDataTypes.hpp"
#include "nodegraph/NodeGraphPayloadTypes.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "runtime/RuntimeProductManager.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/package/RuntimePackages.hpp"
#include "util/GeometryUtils.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <memory>
#include <utility>

namespace {

const std::vector<ExportFormat> meshFormats{ExportFormat::OBJ, ExportFormat::PLY, ExportFormat::STL};
const std::vector<ExportFormat> pointFormats{ExportFormat::XYZ, ExportFormat::CSV, ExportFormat::PLY};
const std::vector<ExportFormat> lineFormats{ExportFormat::OBJ};

size_t activeHeatSolveCount(const NodeGraphController& graph) {
    size_t count = 0;
    for (const auto& entry : graph.graphState().nodes) {
        const NodeGraphNode& node = entry.second;
        for (const NodeGraphSocket& socket : node.outputs) {
            const EvaluatedSocketValue* value = graph.evaluation().outputFor(NodeSocketKey(node.id, socket.id).value);
            if (!value || value->status != EvaluatedSocketStatus::Value || value->data.dataType != payloadtypes::Heat) continue;
            const HeatData* heat = graph.payloads().get<HeatData>(value->data.payloadHandle);
            if (heat && heat->active) ++count;
        }
    }
    return count;
}

bool hasValue(const ExportOutput& output) {
    return output.evaluated && output.evaluated->status == EvaluatedSocketStatus::Value;
}

std::string evaluationError(const ExportOutput& output) {
    return output.evaluated && output.evaluated->status == EvaluatedSocketStatus::Error
        ? output.evaluated->error : std::string{};
}

std::vector<std::pair<uint64_t, NodeGraphNodeId>> heatModelProducts(
    const NodeGraphController& graph, NodeGraphNodeId heatSolveNode) {
    std::vector<std::pair<uint64_t, NodeGraphNodeId>> products;
    const NodeGraphNode* node = graph.graphState().node(heatSolveNode);
    if (!node) return products;
    for (const NodeGraphSocket& input : node->inputs) {
        if (input.valueType != NodeGraphValueType::HeatModel) continue;
        for (const NodeGraphEdge* edge : graph.graphState().edges.incoming(heatSolveNode, input.id)) {
            if (!edge) continue;
            products.emplace_back(NodeSocketKey(edge->fromNode, edge->fromSocket).value, edge->fromNode);
        }
    }
    std::sort(products.begin(), products.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    products.erase(std::unique(products.begin(), products.end(), [](const auto& a, const auto& b) {
        return a.first == b.first;
    }), products.end());
    return products;
}

std::array<float, 16> transformFor(const NodeGraphController& graph, const NodeDataHandle& handle) {
    std::array<float, 16> transform{
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    graph.payloads().resolveLocalToWorld(handle, transform);
    return transform;
}

ExportResult writeGeometry(
    const std::string& path, const std::vector<float>& positions,
    const std::vector<uint32_t>& triangles, const std::vector<float>& normals,
    ExportFormat format,
    const std::array<float, 16>* transform) {
    if (positions.empty() || triangles.empty())
        return {false, "Mesh product has no triangle geometry", 0};
    if (format == ExportFormat::PLY)
        return MeshWriter::writePLY(path, positions, triangles, transform);
    if (format == ExportFormat::STL)
        return MeshWriter::writeSTL(path, positions, triangles, transform);
    return MeshWriter::writeOBJ(path, positions, triangles, normals, transform);
}

class MeshExporter final : public PayloadExporter {
public:
    uint8_t payloadType() const noexcept override { return payloadtypes::Geometry; }
    void appendProducts(const ExportOutput& output, const NodeGraphController& graph,
                       std::vector<ExportProduct>& out) const override {
        const GeometryData* geometry = hasValue(output)
            ? graph.payloads().resolveGeometry(output.evaluated->data.payloadHandle) : nullptr;
        const std::string error = evaluationError(output);
        ExportProduct p{};
        p.key = {output.nodeId, output.socketId, {}};
        p.labelNodeId = output.nodeId;
        p.typeName = "geometry";
        p.supportedFormats = meshFormats;
        p.defaultFormat = ExportFormat::OBJ;
        p.available = error.empty() && geometry &&
            !geometry->pointPositions.empty() && !geometry->triangleIndices.empty();
        if (!p.available) p.unavailableReason = error.empty() ? "Mesh has no triangle geometry" : error;
        out.push_back(std::move(p));
    }
    ExportResult writeProduct(const ExportOutput& output, const ExportProductKey&,
                              ExportFormat format, const std::string& path,
                              const NodeGraphController& graph) const override {
        const GeometryData* geometry = hasValue(output)
            ? graph.payloads().resolveGeometry(output.evaluated->data.payloadHandle) : nullptr;
        if (!geometry) return {false, "Mesh payload is unavailable", 0};
        const auto transform = transformFor(graph, output.evaluated->data.payloadHandle);
        return writeGeometry(path, geometry->pointPositions, geometry->triangleIndices,
                             geometry->renderNormals, format, &transform);
    }
};

class RemeshExporter final : public PayloadExporter {
public:
    uint8_t payloadType() const noexcept override { return payloadtypes::Remesh; }
    void appendProducts(const ExportOutput& output, const NodeGraphController& graph,
                       std::vector<ExportProduct>& out) const override {
        ExportProduct p{};
        p.key = {output.nodeId, output.socketId, {}};
        p.labelNodeId = output.nodeId;
        p.typeName = "geometry";
        p.supportedFormats = meshFormats;
        p.defaultFormat = ExportFormat::OBJ;
        const uint64_t socketKey = NodeSocketKey(output.nodeId, output.socketId).value;
        const RuntimePackageManager* packages = graph.getPackageManager();
        const RuntimeProductManager* products = graph.getProductManager();
        const RemeshPackage* package = packages ? packages->findAny<RemeshPackage>(socketKey) : nullptr;
        const RemeshProduct* product = package && products
            ? products->resolve<RemeshProduct>(package->productHandle) : nullptr;
        const std::string error = evaluationError(output);
        p.available = error.empty() && product && !product->geometryPositions.empty() &&
            !product->geometryTriangleIndices.empty();
        if (!p.available) p.unavailableReason = error.empty() ? "Remesh runtime product is unavailable" : error;
        out.push_back(std::move(p));
    }
    ExportResult writeProduct(const ExportOutput& output, const ExportProductKey&,
                              ExportFormat format, const std::string& path,
                              const NodeGraphController& graph) const override {
        const uint64_t socketKey = NodeSocketKey(output.nodeId, output.socketId).value;
        const RuntimePackageManager* packages = graph.getPackageManager();
        const RuntimeProductManager* products = graph.getProductManager();
        const RemeshPackage* package = packages ? packages->findAny<RemeshPackage>(socketKey) : nullptr;
        const RemeshProduct* product = package && products
            ? products->resolve<RemeshProduct>(package->productHandle) : nullptr;
        if (!package || !product) return {false, "Remesh runtime product is unavailable", 0};
        std::vector<float> positions;
        positions.reserve(product->geometryPositions.size() * 3);
        for (const glm::vec3& v : product->geometryPositions) {
            positions.push_back(v.x); positions.push_back(v.y); positions.push_back(v.z);
        }
        const auto transform = package->localToWorld;
        return writeGeometry(path, positions, product->geometryTriangleIndices, {}, format, &transform);
    }
};

class PointExporter final : public PayloadExporter {
public:
    uint8_t payloadType() const noexcept override { return payloadtypes::Points; }
    void appendProducts(const ExportOutput& output, const NodeGraphController& graph,
                       std::vector<ExportProduct>& out) const override {
        const PointData* points = hasValue(output)
            ? graph.payloads().resolvePoints(output.evaluated->data.payloadHandle) : nullptr;
        const std::string error = evaluationError(output);
        ExportProduct p{};
        p.key = {output.nodeId, output.socketId, {}};
        p.labelNodeId = output.nodeId;
        p.typeName = "point";
        p.supportedFormats = pointFormats;
        p.defaultFormat = ExportFormat::XYZ;
        p.available = error.empty() && points && !points->positions.empty();
        if (!p.available) p.unavailableReason = error.empty() ? "Point product is empty" : error;
        out.push_back(std::move(p));
    }
    ExportResult writeProduct(const ExportOutput& output, const ExportProductKey&,
                              ExportFormat format, const std::string& path,
                              const NodeGraphController& graph) const override {
        const PointData* points = hasValue(output)
            ? graph.payloads().resolvePoints(output.evaluated->data.payloadHandle) : nullptr;
        if (!points || points->positions.empty()) return {false, "Point product is unavailable or empty", 0};
        const auto transform = transformFor(graph, output.evaluated->data.payloadHandle);
        if (format == ExportFormat::CSV) return PointWriter::writeCSV(path, points->positions, &transform);
        if (format == ExportFormat::PLY) return PointWriter::writePLY(path, points->positions, &transform);
        if (format == ExportFormat::XYZ) return PointWriter::writeXYZ(path, points->positions, &transform);
        return {false, "Unsupported point format", 0};
    }
};

class HeatContourExporter final : public PayloadExporter {
public:
    explicit HeatContourExporter(HeatSystemComputeController* controller) : heatController(controller) {}
    uint8_t payloadType() const noexcept override { return payloadtypes::Heat; }
    void appendProducts(const ExportOutput& output, const NodeGraphController& graph,
                       std::vector<ExportProduct>& out) const override {
        const HeatData* heat = hasValue(output)
            ? graph.payloads().get<HeatData>(output.evaluated->data.payloadHandle) : nullptr;
        const std::string error = evaluationError(output);
        const uint64_t socketKey = NodeSocketKey(output.nodeId, output.socketId).value;
        const auto* config = heatController ? heatController->getConfig(socketKey) : nullptr;
        const bool solveActive = heat && heat->active;
        const size_t activeCount = solveActive ? 1 : activeHeatSolveCount(graph);
        for (const auto& [modelKey, modelNode] : heatModelProducts(graph, output.nodeId)) {
            ExportProduct p{};
            p.key = {output.nodeId, output.socketId, std::to_string(modelKey)};
            p.labelNodeId = modelNode;
            p.typeName = "isocontour";
            p.supportedFormats = lineFormats;
            p.defaultFormat = ExportFormat::OBJ;
            const bool keyPresent = config &&
                std::find(config->heatModelKeys.begin(), config->heatModelKeys.end(), modelKey) != config->heatModelKeys.end();
            p.available = error.empty() && solveActive && keyPresent;
            if (!error.empty()) p.unavailableReason = error;
            else if (!heat) p.unavailableReason = "Ember heat payload is unavailable";
            else if (!solveActive) p.unavailableReason = activeCount > 0
                ? "Exactly one Ember heat solve must be enabled"
                : "Ember heat solve is not enabled";
            else if (!keyPresent) p.unavailableReason = "Heat model is not in the committed Ember runtime";
            out.push_back(std::move(p));
        }
    }
    ExportResult writeProduct(const ExportOutput& output, const ExportProductKey& key,
                              ExportFormat format, const std::string& path,
                              const NodeGraphController& graph) const override {
        const HeatData* heat = hasValue(output)
            ? graph.payloads().get<HeatData>(output.evaluated->data.payloadHandle) : nullptr;
        if (!heat) return {false, "Ember heat payload is unavailable", 0};
        if (!heat->active) return {false, "Ember heat solve is not enabled", 0};
        if (!heatController) return {false, "Heat runtime is unavailable", 0};
        if (format != ExportFormat::OBJ) return {false, "Heat contours support OBJ only", 0};
        uint64_t heatModelKey = 0;
        for (const auto& [modelKey, modelNode] : heatModelProducts(graph, output.nodeId)) {
            if (std::to_string(modelKey) == key.productId) {
                heatModelKey = modelKey;
                break;
            }
        }
        if (heatModelKey == 0) return {false, "Selected Heat Model no longer belongs to Ember", 0};
        const uint64_t heatSocketKey = NodeSocketKey(output.nodeId, output.socketId).value;
        std::vector<IsocontourPolyline> lines;
        std::string error;
        if (!ExportService::buildHeatContours(
                heatSocketKey, heatModelKey, heat->isocontourSpacingC, true,
                *heatController, nullptr, lines, error)) {
            return {false, error, 0};
        }
        return LineWriter::writeOBJ(path, lines);
    }
private:
    HeatSystemComputeController* heatController = nullptr;
};

} // namespace

ExporterRegistry::ExporterRegistry(HeatSystemComputeController* heatController) {
    auto add = [this](std::unique_ptr<PayloadExporter> exporter) {
        const uint8_t type = exporter->payloadType();
        exporters.emplace(type, std::move(exporter));
    };
    add(std::make_unique<MeshExporter>());
    add(std::make_unique<RemeshExporter>());
    add(std::make_unique<PointExporter>());
    add(std::make_unique<HeatContourExporter>(heatController));
}

const PayloadExporter* ExporterRegistry::find(uint8_t payloadType) const {
    const auto it = exporters.find(payloadType);
    return it == exporters.end() ? nullptr : it->second.get();
}
