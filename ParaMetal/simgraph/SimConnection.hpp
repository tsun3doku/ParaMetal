#pragma once

#include <cstdint>

enum class SimConnectionMode : uint8_t {
    Staged,
    ZeroCopy
};

// One cross-simulation edge. Identity is the connection id; the socket pair must
// also be unique, while several distinct connections may feed one consumer.
struct SimConnection {
    uint64_t connectionId = 0;
    uint64_t producerSocket = 0;
    uint64_t consumerSocket = 0;
    SimConnectionMode mode = SimConnectionMode::Staged;
};
