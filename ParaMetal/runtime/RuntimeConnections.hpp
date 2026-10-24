#pragma once

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

// Runtime transport wiring only without graph state or payload registries
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
};
