#include "RuntimeExport.hpp"

#include "ExportService.hpp"
#include "ExporterRegistry.hpp"
#include "heat/HeatSystemComputeController.hpp"
#include "nodegraph/NodeGraphController.hpp"

#include <utility>

void RuntimeExport::request(int nodeId) {
    if (nodeId > 0) pendingNodeIds.push_back(nodeId);
}

void RuntimeExport::requestExportProducts(int nodeId, uint64_t requestId, uint64_t graphRevision) {
    if (nodeId > 0 && requestId != 0)
        pendingProductRequests.push_back({nodeId, requestId, graphRevision});
}

void RuntimeExport::processPending(
    NodeGraphController* graph,
    HeatSystemComputeController* heatController) {
    if (pendingNodeIds.empty() && pendingProductRequests.empty()) return;
    std::vector<int> requests = std::move(pendingNodeIds);
    pendingNodeIds.clear();
    std::vector<ProductRequest> catalogRequests = std::move(pendingProductRequests);
    pendingProductRequests.clear();

    ExporterRegistry exporters(heatController);
    for (const ProductRequest& request : catalogRequests) {
        if (!graph || graph->runtimeCompilationPending()) {
            pendingProductRequests.push_back(request);
            continue;
        }
        ExportProductsReply reply{};
        reply.nodeId = request.nodeId;
        reply.requestId = request.requestId;
        reply.graphRevision = request.graphRevision;
        ExportService::discoverProducts(exporters, *graph,
            NodeGraphNodeId{static_cast<uint32_t>(request.nodeId)}, reply.products, reply.error);
        productsReady.push_back(std::move(reply));
    }
    for (int nodeId : requests) {
        if (!graph) {
            results.push_back({nodeId, false, "Graph controller is not available"});
            continue;
        }
        if (!graph->compiledState().isValid) {
            std::string error = "Graph compilation failed";
            if (!graph->compiledState().compilationErrors.empty()) {
                error += ": " + graph->compiledState().compilationErrors.front();
            } else if (!graph->lastPackageErrors().empty()) {
                error += ": " + graph->lastPackageErrors();
            }
            results.push_back({nodeId, false, std::move(error)});
            continue;
        }
        if (graph->runtimeCompilationPending()) {
            const std::string error = graph->lastPackageErrors().empty()
                ? "Graph cook/evaluation is still pending"
                : "Package compilation failed: " + graph->lastPackageErrors();
            results.push_back({nodeId, false, error});
            continue;
        }

        ExportResult exportResult = ExportService::exportProduct(
            exporters, *graph, NodeGraphNodeId{static_cast<uint32_t>(nodeId)});
        results.push_back({nodeId, exportResult.success, exportResult.message});
    }
}

bool RuntimeExport::takeProductsReady(ExportProductsReply& outReply) {
    if (productsReady.empty()) return false;
    outReply = std::move(productsReady.front());
    productsReady.erase(productsReady.begin());
    return true;
}

bool RuntimeExport::takeResult(
    int& outNodeId,
    bool& outSuccess,
    std::string& outMessage) {
    if (results.empty()) return false;
    Result result = std::move(results.front());
    results.erase(results.begin());
    outNodeId = result.nodeId;
    outSuccess = result.success;
    outMessage = std::move(result.message);
    return true;
}

void RuntimeExport::clear() {
    pendingNodeIds.clear();
    results.clear();
    pendingProductRequests.clear();
    productsReady.clear();
}
