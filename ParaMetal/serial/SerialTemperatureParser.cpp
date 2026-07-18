#include "SerialTemperatureParser.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>

SerialTemperatureParser::Result SerialTemperatureParser::append(const uint8_t* bytes, size_t byteCount) {
    Result result{};
    if (bytes && byteCount > 0) pendingInput.append(reinterpret_cast<const char*>(bytes), byteCount);
    if (pendingInput.size() > MaxPendingBytes && pendingInput.find('\n') == std::string::npos) {
        pendingInput.clear();
        return result;
    }

    size_t lineEnd = 0;
    while ((lineEnd = pendingInput.find('\n')) != std::string::npos) {
        std::string line = pendingInput.substr(0, lineEnd);
        pendingInput.erase(0, lineEnd + 1);
        const size_t first = line.find_first_not_of(" \t\r");
        const size_t last = line.find_last_not_of(" \t\r");
        if (first == std::string::npos) {
            continue;
        }
        line = line.substr(first, last - first + 1);

        const size_t separator = line.find(':');
        if (separator != std::string::npos) {
            std::string channel = line.substr(0, separator);
            const size_t channelFirst = channel.find_first_not_of(" \t");
            const size_t channelLast = channel.find_last_not_of(" \t");
            if (channelFirst == std::string::npos) {
                continue;
            }
            channel = channel.substr(channelFirst, channelLast - channelFirst + 1);
            if (channel.size() != 1 || (channel[0] != 'T' && channel[0] != 't')) {
                continue;
            }

            line.erase(0, separator + 1);
            const size_t valueStart = line.find_first_not_of(" \t");
            if (valueStart == std::string::npos) {
                continue;
            }
            line.erase(0, valueStart);
        }

        char* end = nullptr;
        errno = 0;
        const float value = std::strtof(line.c_str(), &end);
        if (errno == ERANGE || end != line.c_str() + line.size() || !std::isfinite(value) ||
            value < -273.15f || value > 10000.0f) {
            continue;
        }
        result.temperatureC = value;
    }
    return result;
}
