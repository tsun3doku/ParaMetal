#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

class SerialTemperatureParser {
public:
    struct Result {
        std::optional<float> temperatureC;
    };

    Result append(const uint8_t* bytes, size_t byteCount);
    void reset() { pendingInput.clear(); }

private:
    static constexpr size_t MaxPendingBytes = 4096;
    std::string pendingInput;
};
