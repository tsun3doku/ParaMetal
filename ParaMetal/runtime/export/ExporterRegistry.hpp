#pragma once

#include "PayloadExporter.hpp"

#include <memory>
#include <unordered_map>

class HeatSystemComputeController;

class ExporterRegistry {
public:
    explicit ExporterRegistry(HeatSystemComputeController* heatController);
    const PayloadExporter* find(uint8_t payloadType) const;

private:
    std::unordered_map<uint8_t, std::unique_ptr<PayloadExporter>> exporters;
};
