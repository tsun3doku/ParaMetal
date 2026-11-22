#pragma once

#include <cstddef>
#include <cstdint>

struct RuntimeTelemetry {
    uint32_t nodeId = 0;
};

struct TelemetryTemperatures final : RuntimeTelemetry {
    const float* values = nullptr;
    std::size_t count = 0;
};

class TemperatureTelemetryReceiver {
public:
    virtual ~TemperatureTelemetryReceiver() = default;
    virtual void onTemperatures(const TelemetryTemperatures&) = 0;
};
