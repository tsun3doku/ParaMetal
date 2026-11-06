#include "NodeGraphDataTypes.hpp"
#include "NodeGraphPayloadTypes.hpp"
#include "NodeGraphRegistry.hpp"
#include "NodeGraphTypeRegistry.hpp"
#include "NodeGraphUtils.hpp"

#include "NodePayloadRegistry.hpp"
#include "domain/FluidData.hpp"
#include "domain/HeatModelData.hpp"
#include "domain/SerialTemperatureData.hpp"
#include "domain/HeatProbeData.hpp"

#include <cstddef>

void populateMetadata(NodeDataBlock& dataBlock, const NodeGraphTypeRegistry* typeRegistry, const NodePayloadRegistry* registry) {
    const std::string* typeName = typeRegistry ? typeRegistry->getTypeName(dataBlock.dataType) : nullptr;
    dataBlock.metadata["data.type"] = typeName ? *typeName : "unknown";

    if (dataBlock.dataType == payloadtypes::Geometry ||
        dataBlock.dataType == payloadtypes::Remesh ||
        dataBlock.dataType == payloadtypes::HeatModel) {
        const GeometryData* geometry = registry ? registry->resolveGeometry(dataBlock.payloadHandle) : nullptr;
        if (!geometry || geometry->baseModelPath.empty()) {
            dataBlock.metadata.erase("geometry.model_path");
        } else {
            dataBlock.metadata["geometry.model_path"] = geometry->baseModelPath;
        }
        const size_t pointCount = geometry ? geometry->pointPositions.size() / 3 : 0u;
        const size_t triangleCount = geometry ? geometry->triangleIndices.size() / 3 : 0u;
        dataBlock.metadata.erase("geometry.model_id");
        dataBlock.metadata["geometry.point_count"] = std::to_string(pointCount);
        dataBlock.metadata["geometry.triangle_count"] = std::to_string(triangleCount);
    } else {
        dataBlock.metadata.erase("geometry.model_path");
        dataBlock.metadata.erase("geometry.model_id");
        dataBlock.metadata.erase("geometry.point_count");
        dataBlock.metadata.erase("geometry.triangle_count");
    }

    if (dataBlock.dataType == payloadtypes::HeatModel) {
        const HeatModelData* heatModel = registry ? registry->get<HeatModelData>(dataBlock.payloadHandle) : nullptr;
        dataBlock.metadata["heat_model.initial_temperature_c"] =
            heatModel ? std::to_string(heatModel->initialTemperatureC) : std::string();
    } else {
        dataBlock.metadata.erase("heat_model.initial_temperature_c");
    }

    if (dataBlock.dataType == payloadtypes::Fluid) {
        const FluidData* fluid = registry ? registry->get<FluidData>(dataBlock.payloadHandle) : nullptr;
        dataBlock.metadata["fluid.density"] = fluid ? std::to_string(fluid->density) : std::string();
        dataBlock.metadata["fluid.specific_heat"] = fluid ? std::to_string(fluid->specificHeat) : std::string();
        dataBlock.metadata["fluid.conductivity"] = fluid ? std::to_string(fluid->conductivity) : std::string();
        dataBlock.metadata["fluid.initial_temperature_c"] =
            fluid ? std::to_string(fluid->initialTemperatureC) : std::string();
    } else {
        dataBlock.metadata.erase("fluid.density");
        dataBlock.metadata.erase("fluid.specific_heat");
        dataBlock.metadata.erase("fluid.conductivity");
        dataBlock.metadata.erase("fluid.initial_temperature_c");
    }

    if (dataBlock.dataType == payloadtypes::SerialTemperature) {
        const auto* serial = registry ? registry->get<SerialTemperatureData>(dataBlock.payloadHandle) : nullptr;
        dataBlock.metadata["serial.enabled"] = (serial && serial->enabled) ? "true" : "false";
        dataBlock.metadata["serial.port"] = serial ? serial->portName : std::string();
        dataBlock.metadata["serial.baud_rate"] = std::to_string(serial ? serial->baudRate : 0u);
    } else {
        dataBlock.metadata.erase("serial.enabled");
        dataBlock.metadata.erase("serial.port");
        dataBlock.metadata.erase("serial.baud_rate");
    }

    if (dataBlock.dataType == payloadtypes::HeatProbe) {
        const auto* probe = registry ? registry->get<HeatProbeData>(dataBlock.payloadHandle) : nullptr;
        dataBlock.metadata["probe.enabled"] = (probe && probe->enabled) ? "true" : "false";
        dataBlock.metadata["probe.count"] = std::to_string(probe ? probe->items.size() : 0u);
        dataBlock.metadata["probe.opc_port"] = std::to_string(probe ? probe->opcPort : 0u);
    } else {
        dataBlock.metadata.erase("probe.enabled");
        dataBlock.metadata.erase("probe.count");
        dataBlock.metadata.erase("probe.opc_port");
    }

    if (dataBlock.dataType == payloadtypes::Heat) {
        const HeatData* heatData = registry ? registry->get<HeatData>(dataBlock.payloadHandle) : nullptr;
        dataBlock.metadata["heat.active"] = (heatData && heatData->active) ? "true" : "false";
        dataBlock.metadata["heat.model_count"] =
            std::to_string(heatData ? heatData->heatModelHandles.size() : 0u);
    } else {
        dataBlock.metadata.erase("heat.active");
        dataBlock.metadata.erase("heat.model_count");
    }

    if (dataBlock.dataType == payloadtypes::Voronoi) {
        const VoronoiData* voronoiData = registry ? registry->get<VoronoiData>(dataBlock.payloadHandle) : nullptr;
        dataBlock.metadata["voronoi.active"] = (voronoiData && voronoiData->active) ? "true" : "false";
        dataBlock.metadata["voronoi.model_count"] =
            std::to_string(voronoiData ? voronoiData->modelMeshHandles.size() : 0u);
        dataBlock.metadata["voronoi.cell_size"] =
            voronoiData ? std::to_string(voronoiData->cellSize) : std::string();
        dataBlock.metadata["voronoi.voxel_resolution"] =
            voronoiData ? std::to_string(voronoiData->voxelResolution) : std::string();
        dataBlock.metadata["voronoi.contact_range"] =
            voronoiData ? std::to_string(voronoiData->contactRange) : std::string();
        dataBlock.metadata["voronoi.lloyd_iterations"] =
            voronoiData ? std::to_string(voronoiData->lloydIterations) : std::string();
        dataBlock.metadata["voronoi.lloyd_volume_threshold"] =
            voronoiData ? std::to_string(voronoiData->lloydVolumeThreshold) : std::string();
        dataBlock.metadata["voronoi.lloyd_damping"] =
            voronoiData ? std::to_string(voronoiData->lloydDamping) : std::string();
    } else {
        dataBlock.metadata.erase("voronoi.active");
        dataBlock.metadata.erase("voronoi.cell_size");
        dataBlock.metadata.erase("voronoi.voxel_resolution");
        dataBlock.metadata.erase("voronoi.contact_range");
        dataBlock.metadata.erase("voronoi.lloyd_iterations");
        dataBlock.metadata.erase("voronoi.lloyd_volume_threshold");
        dataBlock.metadata.erase("voronoi.lloyd_damping");
    }
}
