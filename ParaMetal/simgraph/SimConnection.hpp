#pragma once

#include <cstdint>

enum class SimConnectionMode : uint8_t {
    Staged,
    ZeroCopy
};

struct SimConnection {
    uint64_t connectionId = 0;
    uint64_t producerSocket = 0;
    uint64_t consumerSocket = 0;
    SimConnectionMode mode = SimConnectionMode::Staged;
};
