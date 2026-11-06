#pragma once

#include <functional>

class ModelComputeController;
class PointComputeRuntime;
class RemeshController;
class VoronoiSystemComputeController;
class HeatSystemComputeController;
class SimGraphRuntime;
namespace cfd {
class CFDSystemComputeController;
}

class RuntimeModelComputeTransport;
class RuntimePointComputeTransport;
class RuntimeRemeshComputeTransport;
class RuntimeVoronoiComputeTransport;
class RuntimeHeatComputeTransport;
class RuntimeCFDComputeTransport;
class RuntimeModelDisplayTransport;
class RuntimePointDisplayTransport;
class RuntimeRemeshDisplayTransport;
class RuntimeVoronoiDisplayTransport;
class RuntimeHeatDisplayTransport;

struct RuntimeConnections {
    RuntimeModelComputeTransport* modelComputeTransport = nullptr;
    RuntimePointComputeTransport* pointComputeTransport = nullptr;
    RuntimeRemeshComputeTransport* remeshComputeTransport = nullptr;
    RuntimeVoronoiComputeTransport* voronoiComputeTransport = nullptr;
    RuntimeHeatComputeTransport* heatComputeTransport = nullptr;
    RuntimeCFDComputeTransport* cfdComputeTransport = nullptr;

    RuntimeModelDisplayTransport* modelDisplayTransport = nullptr;
    RuntimePointDisplayTransport* pointDisplayTransport = nullptr;
    RuntimeRemeshDisplayTransport* remeshDisplayTransport = nullptr;
    RuntimeVoronoiDisplayTransport* voronoiDisplayTransport = nullptr;
    RuntimeHeatDisplayTransport* heatDisplayTransport = nullptr;

    ModelComputeController* modelComputeController = nullptr;
    PointComputeRuntime* pointComputeRuntime = nullptr;
    RemeshController* remeshController = nullptr;
    VoronoiSystemComputeController* voronoiController = nullptr;
    HeatSystemComputeController* heatController = nullptr;
    cfd::CFDSystemComputeController* cfdController = nullptr;

    SimGraphRuntime* simGraph = nullptr;

    std::function<bool()> quiesceGpu;
};
