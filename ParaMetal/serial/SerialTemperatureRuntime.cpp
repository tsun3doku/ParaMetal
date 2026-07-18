#include "SerialTemperatureRuntime.hpp"

#include <array>

namespace {
constexpr auto ReconnectDelay = std::chrono::seconds(1);
constexpr auto StaleDelay = std::chrono::seconds(2);
}

void SerialTemperatureRuntime::configure(
    bool nextEnabled, const std::string& nextPortName, uint32_t nextBaudRate) {
    if (isConfigured && enabled == nextEnabled && portName == nextPortName && baudRate == nextBaudRate) return;
    close();
    enabled = nextEnabled;
    portName = nextPortName;
    baudRate = nextBaudRate;
    isConfigured = true;
    parser.reset();
    currentStatus.latestReading.reset();
    nextRevision = 1;
    if (!enabled) {
        currentStatus.connection = ConnectionState::Disabled;
        currentStatus.detail = "Disabled";
        return;
    }
    if (portName.empty()) {
        currentStatus.connection = ConnectionState::NoPortSelected;
        currentStatus.detail = "No port selected";
        return;
    }
    open();
}

void SerialTemperatureRuntime::open() {
    if (!enabled || portName.empty()) return;
    if (!port) port = SerialPort::create();
    currentStatus.connection = ConnectionState::Connecting;
    currentStatus.detail = "Connecting";
    std::string error;
    if (!port || !port->open({portName, baudRate}, error)) {
        fail(error.empty() ? "Unable to open serial port" : error);
        return;
    }
    currentStatus.connection = ConnectionState::WaitingForReading;
    currentStatus.detail = "Waiting for reading";
}

void SerialTemperatureRuntime::fail(const std::string& error) {
    if (port) port->close();
    currentStatus.connection = currentStatus.latestReading
        ? ConnectionState::DisconnectedHolding : ConnectionState::Error;
    currentStatus.detail = error;
    nextReconnect = std::chrono::steady_clock::now() + ReconnectDelay;
}

void SerialTemperatureRuntime::publish(float temperatureC) {
    const auto receivedAt = std::chrono::steady_clock::now();
    float pollingRateHz = 0.0f;
    if (currentStatus.latestReading) {
        const float intervalSeconds = std::chrono::duration<float>(
            receivedAt - currentStatus.latestReading->receivedAt).count();
        if (intervalSeconds > 0.0f) {
            const float measuredRateHz = 1.0f / intervalSeconds;
            pollingRateHz = currentStatus.latestReading->pollingRateHz > 0.0f
                ? currentStatus.latestReading->pollingRateHz * 0.8f + measuredRateHz * 0.2f
                : measuredRateHz;
        }
    }
    currentStatus.latestReading = Reading{
        temperatureC, nextRevision++, receivedAt, pollingRateHz};
    currentStatus.connection = ConnectionState::Live;
    currentStatus.detail = "Live";
}

void SerialTemperatureRuntime::update() {
    if (!enabled || portName.empty()) return;
    const auto now = std::chrono::steady_clock::now();
    if (!port || !port->isOpen()) {
        if (now >= nextReconnect) open();
        return;
    }

    std::array<uint8_t, 1024> bytes{};
    std::string error;
    const size_t count = port->readAvailable(bytes.data(), bytes.size(), error);
    if (!error.empty()) {
        fail(error);
        return;
    }
    if (count > 0) {
        const auto result = parser.append(bytes.data(), count);
        if (result.temperatureC) publish(*result.temperatureC);
    }
    if (currentStatus.latestReading && now - currentStatus.latestReading->receivedAt >= StaleDelay) {
        currentStatus.connection = ConnectionState::StaleHolding;
        currentStatus.detail = "Stale - holding last reading";
    }
}

void SerialTemperatureRuntime::close() {
    port.reset();
}
