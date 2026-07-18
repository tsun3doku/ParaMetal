#pragma once

#include "serial/SerialPort.hpp"
#include "serial/SerialTemperatureParser.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

class SerialTemperatureRuntime {
public:
    enum class ConnectionState {
        Disabled,
        NoPortSelected,
        Connecting,
        WaitingForReading,
        Live,
        StaleHolding,
        DisconnectedHolding,
        Error
    };

    struct Reading {
        float temperatureC = 0.0f;
        uint64_t revision = 0;
        std::chrono::steady_clock::time_point receivedAt{};
        float pollingRateHz = 0.0f;
    };

    struct Status {
        ConnectionState connection = ConnectionState::Disabled;
        std::optional<Reading> latestReading;
        std::string detail;
    };

    void configure(bool enabled, const std::string& portName, uint32_t baudRate);
    void update();

    const Status& status() const { return currentStatus; }
    const Reading* latestReading() const {
        return currentStatus.latestReading ? &*currentStatus.latestReading : nullptr;
    }

private:
    void open();
    void close();
    void fail(const std::string& error);
    void publish(float temperatureC);

    bool enabled = false;
    std::string portName;
    uint32_t baudRate = 115200;
    std::unique_ptr<SerialPort> port;
    SerialTemperatureParser parser;
    Status currentStatus;
    bool isConfigured = false;
    uint64_t nextRevision = 1;
    std::chrono::steady_clock::time_point nextReconnect{};
};
