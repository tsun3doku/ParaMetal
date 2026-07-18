#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct SerialPortConfig {
    std::string portName;
    uint32_t baudRate = 115200;
};

struct SerialPortInfo {
    std::string portName;
    std::string displayName;
};

class SerialPort {
public:
    virtual ~SerialPort() = default;
    virtual bool open(const SerialPortConfig& config, std::string& outError) = 0;
    virtual void close() = 0;
    virtual bool isOpen() const = 0;
    virtual size_t readAvailable(uint8_t* destination, size_t capacity, std::string& outError) = 0;

    static std::vector<SerialPortInfo> enumeratePorts();
    static std::unique_ptr<SerialPort> create();
};
