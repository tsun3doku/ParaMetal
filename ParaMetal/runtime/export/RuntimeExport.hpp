#pragma once

#include <string>
#include <vector>

#include "ExportTypes.hpp"

class NodeGraphController;
class HeatSystemComputeController;
class ExporterRegistry;
namespace cfd { class CFDSystemComputeController; }

struct ExportProductsReply {
    int nodeId = 0;
    uint64_t requestId = 0;
    uint64_t graphRevision = 0;
    std::vector<ExportProduct> products;
    std::string error;
};

class RuntimeExport {
public:
    void request(int nodeId);
    void requestExportProducts(int nodeId, uint64_t requestId, uint64_t graphRevision);
    void processPending(
        NodeGraphController* graph,
        HeatSystemComputeController* heatController,
        cfd::CFDSystemComputeController* cfdController);
    bool takeResult(
        int& outNodeId,
        bool& outSuccess,
        std::string& outMessage);
    bool takeProductsReady(ExportProductsReply& outReply);
    void clear();

private:
    struct Result {
        int nodeId = 0;
        bool success = false;
        std::string message;
    };

    std::vector<int> pendingNodeIds;
    std::vector<Result> results;
    struct ProductRequest { int nodeId = 0; uint64_t requestId = 0; uint64_t graphRevision = 0; };
    std::vector<ProductRequest> pendingProductRequests;
    std::vector<ExportProductsReply> productsReady;
};
