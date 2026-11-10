#pragma once

#include "PayloadExporter.hpp"

#include <memory>
#include <unordered_map>

class HeatSystemComputeController;
namespace cfd { class CFDSystemComputeController; }

class ExporterRegistry {
public:
    ExporterRegistry(
        HeatSystemComputeController* heatController,
        cfd::CFDSystemComputeController* cfdController);
    const PayloadExporter* find(uint8_t payloadType) const;

private:
    std::unordered_map<uint8_t, std::unique_ptr<PayloadExporter>> exporters;
};
