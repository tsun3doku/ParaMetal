#include "ExportService.hpp"

#include "ExporterRegistry.hpp"
#include "domain/HeatData.hpp"
#include "heat/HeatSystem.hpp"
#include "heat/HeatIsocontourLevels.hpp"
#include "heat/HeatSystemComputeController.hpp"
#include "mesh/IsocontourBuilder.hpp"
#include "nodegraph/NodeGraphController.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeGraphRuntime.hpp"
#include "nodegraph/NodePayloadRegistry.hpp"
#include "nodegraph/NodeWriteParams.hpp"
#include "runtime/RuntimeProductManager.hpp"
#include "runtime/RuntimeProducts.hpp"
#include "runtime/package/RuntimePackageManager.hpp"
#include "runtime/package/RuntimePackages.hpp"
#include "util/GeometryUtils.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iterator>
#include <limits>
#include <set>
#include <unordered_set>

namespace {

struct ProductKeyLess {
    bool operator()(const ExportProductKey& a, const ExportProductKey& b) const {
        if (a.producerNodeId.value != b.producerNodeId.value) return a.producerNodeId.value < b.producerNodeId.value;
        if (a.outputSocketId.value != b.outputSocketId.value) return a.outputSocketId.value < b.outputSocketId.value;
        return a.productId < b.productId;
    }
};

const EvaluatedSocketValue* evaluatedOutput(const NodeGraphController& graph, NodeGraphNodeId nodeId, NodeGraphSocketId socketId) {
    return graph.evaluation().outputFor(NodeSocketKey(nodeId, socketId).value);
}

ExportOutput makeOutput(const NodeGraphController& graph, NodeGraphNodeId nodeId, NodeGraphSocketId socketId) {
    ExportOutput result{};
    result.nodeId = nodeId;
    result.socketId = socketId;
    result.node = graph.graphState().node(nodeId);
    result.socket = result.node ? result.node->output(socketId) : nullptr;
    result.evaluated = evaluatedOutput(graph, nodeId, socketId);
    return result;
}

std::string extensionFor(ExportFormat format) {
    switch (format) {
    case ExportFormat::OBJ: return ".obj";
    case ExportFormat::PLY: return ".ply";
    case ExportFormat::STL: return ".stl";
    case ExportFormat::XYZ: return ".xyz";
    case ExportFormat::CSV: return ".csv";
    default: return {};
    }
}

} // namespace

bool ExportService::discoverProducts(
    const ExporterRegistry& exporters,
    const NodeGraphController& graph,
    NodeGraphNodeId writeNodeId,
    std::vector<ExportProduct>& out,
    std::string& outError) {
    out.clear();
    outError.clear();
    const NodeGraphNode* writeNode = graph.graphState().node(writeNodeId);
    if (!writeNode || writeNode->typeId != nodegraphtypes::Write) {
        outError = "Node is not a Write node";
        return false;
    }
    const NodeGraphSocket* dataInput = writeNode->input("Data");
    if (!dataInput) {
        outError = "Write node has no Data input";
        return false;
    }

    std::vector<NodeGraphNodeId> frontier;
    for (const NodeGraphEdge* edge : graph.graphState().edges.incoming(writeNodeId, dataInput->id))
        if (edge) frontier.push_back(edge->fromNode);
    std::sort(frontier.begin(), frontier.end(), [](auto a, auto b) { return a.value < b.value; });
    std::unordered_set<uint32_t> visited;
    std::set<ExportProductKey, ProductKeyLess> emitted;

    while (!frontier.empty()) {
        std::vector<NodeGraphNodeId> next;
        for (NodeGraphNodeId nodeId : frontier) {
            if (!visited.insert(nodeId.value).second) continue;
            const NodeGraphNode* node = graph.graphState().node(nodeId);
            if (!node) continue;

            std::vector<const NodeGraphSocket*> outputs;
            outputs.reserve(node->outputs.size());
            for (const NodeGraphSocket& socket : node->outputs) outputs.push_back(&socket);
            std::sort(outputs.begin(), outputs.end(), [](const auto* a, const auto* b) { return a->id.value < b->id.value; });
            for (const NodeGraphSocket* socket : outputs) {
                const EvaluatedSocketValue* value = evaluatedOutput(graph, nodeId, socket->id);
                const uint8_t payloadType = value && value->status == EvaluatedSocketStatus::Value
                    ? value->data.dataType
                    : socket->contract.producedPayloadType;
                const PayloadExporter* exporter = exporters.find(payloadType);
                if (!exporter) continue;
                ExportOutput output = makeOutput(graph, nodeId, socket->id);
                std::vector<ExportProduct> products;
                exporter->appendProducts(output, graph, products);
                std::sort(products.begin(), products.end(), [](const ExportProduct& a, const ExportProduct& b) {
                    return a.key.productId < b.key.productId;
                });
                for (ExportProduct& product : products) {
                    if (emitted.insert(product.key).second) out.push_back(std::move(product));
                }
            }

            std::vector<const NodeGraphEdge*> edges;
            for (const NodeGraphSocket& input : node->inputs) {
                auto incoming = graph.graphState().edges.incoming(nodeId, input.id);
                edges.insert(edges.end(), incoming.begin(), incoming.end());
            }
            std::sort(edges.begin(), edges.end(), [](const NodeGraphEdge* a, const NodeGraphEdge* b) {
                if (a->fromNode.value != b->fromNode.value) return a->fromNode.value < b->fromNode.value;
                if (a->fromSocket.value != b->fromSocket.value) return a->fromSocket.value < b->fromSocket.value;
                return a->id.value < b->id.value;
            });
            for (const NodeGraphEdge* edge : edges) if (edge) next.push_back(edge->fromNode);
        }
        std::sort(next.begin(), next.end(), [](auto a, auto b) { return a.value < b.value; });
        next.erase(std::unique(next.begin(), next.end()), next.end());
        frontier = std::move(next);
    }
    return true;
}

ExportResult ExportService::exportProduct(
    const ExporterRegistry& exporters, const NodeGraphController& graph, NodeGraphNodeId writeNodeId) {
    const NodeGraphNode* writeNode = graph.graphState().node(writeNodeId);
    if (!writeNode || writeNode->typeId != nodegraphtypes::Write) return {false, "Node is not a Write node", 0};
    const WriteNodeParams params = readWriteNodeParams(*writeNode);
    const WriteProductSelection& selection = params.product;
    if (selection.empty()) return {false, "Select a product to export", 0};

    std::vector<ExportProduct> catalog;
    std::string error;
    if (!discoverProducts(exporters, graph, writeNodeId, catalog, error)) return {false, error, 0};
    const auto product = std::find_if(catalog.begin(), catalog.end(), [&](const ExportProduct& candidate) {
        return candidate.key.producerNodeId.value == selection.producerNodeId &&
            candidate.key.outputSocketId.value == selection.outputSocketId &&
            candidate.key.productId == selection.productId;
    });
    if (product == catalog.end()) return {false, "Selected product is no longer upstream of Write", 0};
    if (!product->available) return {false, product->unavailableReason, 0};

    const ExportFormat format = exportutils::stringToExportFormat(selection.format);
    if (format == ExportFormat::Unknown || format == ExportFormat::Auto ||
        std::find(product->supportedFormats.begin(), product->supportedFormats.end(), format) == product->supportedFormats.end()) {
        return {false, "Selected format is not supported by this product", 0};
    }

    const ExportOutput output = makeOutput(graph, product->key.producerNodeId, product->key.outputSocketId);
    const PayloadExporter* exporter = output.evaluated ? exporters.find(output.evaluated->data.dataType) : nullptr;
    if (!exporter) return {false, "Payload exporter is unavailable", 0};

    std::filesystem::path destination(params.filePath);
    std::error_code ec;
    if (destination.empty() || destination.filename().empty() || destination.filename() == "." ||
        std::filesystem::is_directory(destination, ec))
        return {false, "Set an output file name", 0};
    destination.replace_extension(extensionFor(format));
    if (destination.has_parent_path()) {
        std::filesystem::create_directories(destination.parent_path(), ec);
        if (ec) return {false, "Failed to create destination directory: " + ec.message(), 0};
    }
    const ExportResult result = exporter->writeProduct(output, product->key, format, destination.string(), graph);
    if (!result.success) return result;
    return {true, "Exported " + destination.string(), result.writtenCount};
}

bool ExportService::buildHeatContours(
    uint64_t heatSocketKey, uint64_t heatModelKey, float spacingC, bool applyTransform,
    HeatSystemComputeController& heatController,
    const std::vector<std::vector<float>>* temperaturesByModel,
    std::vector<IsocontourPolyline>& out,
    std::string& outError) {
    out.clear();
    outError.clear();
    if (!std::isfinite(spacingC) || spacingC <= 0.0f) {
        outError = "Isocontour spacing must be finite and greater than zero";
        return false;
    }
    HeatSystem* system = heatController.getSystem(heatSocketKey);
    const auto* config = heatController.getConfig(heatSocketKey);
    if (!system || !config) {
        outError = "Committed Ember heat system is unavailable";
        return false;
    }
    const size_t count = config->modelRuntimeModelIds.size();
    if (count == 0 || config->heatModelKeys.size() != count ||
        config->modelSurfacePositions.size() != count ||
        config->modelSurfaceTriangleIndices.size() != count || config->modelSurfaceNormals.size() != count) {
        outError = "Heat model configuration arrays are inconsistent";
        return false;
    }
    std::vector<std::vector<float>> readbackTemperatures;
    if (!temperaturesByModel) {
        if (!system->readCurrentSurfaceTemperatures(readbackTemperatures, outError)) return false;
        temperaturesByModel = &readbackTemperatures;
    }
    if (temperaturesByModel->size() != count) {
        outError = "Surface temperature model count does not match heat configuration";
        return false;
    }

    std::unordered_set<uint32_t> seenRuntimeIds;
    std::unordered_set<uint64_t> seenHeatModelKeys;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t id = config->modelRuntimeModelIds[i];
        const uint64_t key = config->heatModelKeys[i];
        const auto transform = config->modelLocalToWorldByModelId.find(id);
        const auto& modelPositions = config->modelSurfacePositions[i];
        const auto& modelTriangles = config->modelSurfaceTriangleIndices[i];
        const auto& modelTemperatures = (*temperaturesByModel)[i];
        if (id == 0 || key == 0 || !seenRuntimeIds.insert(id).second ||
            !seenHeatModelKeys.insert(key).second || transform == config->modelLocalToWorldByModelId.end() ||
            modelPositions.empty() || modelTriangles.empty() || modelTriangles.size() % 3 != 0 ||
            modelPositions.size() != modelTemperatures.size()) {
            outError = "Heat model arrays, IDs, transforms, or surface buffers are inconsistent";
            return false;
        }
        for (float temperature : modelTemperatures) {
            if (!std::isfinite(temperature) || temperature < -1.0e20f) {
                outError = "Heat surface contains invalid temperatures";
                return false;
            }
        }
        for (uint32_t index : modelTriangles) {
            if (index >= modelPositions.size()) {
                outError = "Heat surface contains an invalid triangle index";
                return false;
            }
        }
    }

    size_t selected = count;
    for (size_t i = 0; i < count; ++i) if (config->heatModelKeys[i] == heatModelKey) { selected = i; break; }
    if (selected == count) { outError = "Selected Heat Model is not in the committed Ember runtime"; return false; }
    const uint32_t modelId = config->modelRuntimeModelIds[selected];
    const auto transformIt = config->modelLocalToWorldByModelId.find(modelId);
    if (modelId == 0 || transformIt == config->modelLocalToWorldByModelId.end()) {
        outError = "Heat model ID or transform is invalid";
        return false;
    }
    const auto& positions = config->modelSurfacePositions[selected];
    const auto& triangles = config->modelSurfaceTriangleIndices[selected];
    const auto& temperatures = (*temperaturesByModel)[selected];
    if (positions.empty() || triangles.empty() || positions.size() != temperatures.size() || triangles.size() % 3 != 0) {
        outError = "Heat surface geometry and temperatures are inconsistent";
        return false;
    }
    std::vector<float> levels;
    if (!heat::makeIsocontourLevels(temperatures, spacingC, levels, outError)) return false;
    if (levels.empty()) return true;
    out = IsocontourBuilder::build(positions, triangles, temperatures, levels);
    if (applyTransform) {
        const glm::mat4 transform = toMat4(transformIt->second);
        for (IsocontourPolyline& line : out)
            for (glm::vec3& point : line.points) point = glm::vec3(transform * glm::vec4(point, 1.0f));
    }
    return true;
}
