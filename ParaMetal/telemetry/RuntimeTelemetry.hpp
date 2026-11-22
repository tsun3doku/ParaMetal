#pragma once

#include "telemetry/TemperatureTelemetry.hpp"
#include "serial/SerialTemperatureRuntime.hpp"

struct TelemetryConnection final : RuntimeTelemetry {
    const SerialTemperatureRuntime::Status* status = nullptr;
};

class ConnectionTelemetryReceiver {
public:
    virtual ~ConnectionTelemetryReceiver() = default;
    virtual void onConnection(const TelemetryConnection&) = 0;
};
